"""TRD：环面秩离散扩散（Toroidal Rank Diffusion）—— 新架构 v1。

设计的每一处都对着旧模型（`model/model.py`）的一个具体失败（见 `docs/arch_plan.md` §0）：

1. **环面相对位置，无绝对位置** → 平移等变。旧模型逐格学绝对位置，砖行要在每个相位
   单独学、每材质只有 ~8 张样本，于是有几何结构的材质出噪点。纹理在环面上是平稳的，
   这里注意力偏置只依赖 **(dy, dx) 在环面上的相对偏移**，并且用**归一化连续坐标**
   （偏移 / N，折回 [-0.5, 0.5)）经小 MLP 给出——所以同一个模型对 16/24/32 都成立，
   且结构随瓦片大小等比缩放，正好对应"真人每图画的结构单元数与分辨率无关（~3.2）"。
   可平铺性由构造保证。
2. **开放词表文本条件**（冻结 CLIP 文本编码器的池化嵌入）→ 任意材质名都能处理，
   不再是 521 个封闭 ID。
3. **调色板与结构联合生成**：序列 = [16 个调色板槽（按亮度排序的颜色码）] + [N×N 秩网格]，
   两者同在一个吸收态离散扩散里（MaskGIT 式掩码 + 迭代解码）。不做逐像素回归——
   "没有标准答案"（真人逐格一致率 9.8%）决定了必须建模分布。
4. 可选**平均色条件**（训练时 50% 丢弃）：交付时把用户区域的颜色喂进来，配色即跟随区域；
   评测时丢掉，与各基线在同等条件下比。

条件经 adaLN-Zero 注入每个块（DiT 做法）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

K_MAX = 16          # 调色板槽数（最多 16 色）


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class ToroidalBias(nn.Module):
    """每个头一张注意力偏置：网格-网格由环面归一化偏移经 MLP 给出；其余块为可学标量。"""

    def __init__(self, heads: int, hidden: int = 64, freqs: int = 1):
        """freqs：环面偏移用几个谐波的 sin/cos 编码。

        v1 只用基频（freqs=1）。**这是 v1 结构学不出来的设计缺陷**：模型里没有绝对位置，
        "往上看 4 行"只能靠这个偏置表达；只有基频时偏置天生偏向"越近越相关"的平滑衰减，
        很难表达"每隔 4 格一道砖缝"的周期 → 采样出来是斑点（2026-09-11 诊断，砖/木板无结构）。
        N 格环面上的任意函数可由 ≤N/2 次谐波精确表示，v2 取 freqs=8（N=16 满表达）。
        """
        super().__init__()
        self.heads = heads
        self.freqs = freqs
        self.mlp = nn.Sequential(nn.Linear(4 * freqs, hidden), nn.GELU(), nn.Linear(hidden, heads))
        self.pal_pal = nn.Parameter(torch.zeros(heads, K_MAX, K_MAX))
        self.pal_grid = nn.Parameter(torch.zeros(heads))
        self.grid_pal = nn.Parameter(torch.zeros(heads))
        self._cache = {}

    def grid_offsets(self, n, device):
        key = (n, str(device))
        if key not in self._cache:
            ys, xs = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
            ys, xs = ys.flatten().float(), xs.flatten().float()
            dy = (ys[:, None] - ys[None, :]) / n
            dx = (xs[:, None] - xs[None, :]) / n
            dy = (dy + 0.5) % 1.0 - 0.5          # 折回环面 [-0.5, 0.5)
            dx = (dx + 0.5) % 1.0 - 0.5
            # 用各次谐波的 sin/cos 编码周期偏移：天然满足环面连续性，且能表达任意周期
            f = torch.arange(1, self.freqs + 1).float()
            ay, ax = 2 * math.pi * dy[..., None] * f, 2 * math.pi * dx[..., None] * f
            feat = torch.cat([ay.sin(), ay.cos(), ax.sin(), ax.cos()], -1)
            self._cache[key] = feat.to(device)
        return self._cache[key]

    def forward(self, n, device):
        g = self.mlp(self.grid_offsets(n, device)).permute(2, 0, 1)      # [H, n², n²]
        P, G = K_MAX, n * n
        bias = torch.zeros(self.heads, P + G, P + G, device=device)
        bias[:, :P, :P] = self.pal_pal
        bias[:, :P, P:] = self.pal_grid[:, None, None]
        bias[:, P:, :P] = self.grid_pal[:, None, None]
        bias[:, P:, P:] = g
        return bias


class Block(nn.Module):
    def __init__(self, d, heads, drop):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)
        self.drop = drop

    def forward(self, x, c, bias):
        B, L, D = x.shape
        s1, g1, a1, s2, g2, a2 = self.ada(c).chunk(6, -1)
        h = modulate(self.n1(x), s1, g1)
        q, k, v = self.qkv(h).view(B, L, 3, self.heads, D // self.heads).permute(2, 0, 3, 1, 4)
        # 偏置 [1,H,L,L] 按批次广播，不复制（按批次复制 256x6x272x272 每层每步要几个 GB，
        # 还会逼 SDPA 退回慢内核——v1 首次开训因此每千步 >10 分钟）
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype)[None],
                                           dropout_p=self.drop if self.training else 0.0)
        x = x + a1[:, None] * self.proj(o.transpose(1, 2).reshape(B, L, D))
        x = x + a2[:, None] * self.mlp(modulate(self.n2(x), s2, g2))
        return x


class TRD(nn.Module):
    """输入：pal [B,16] 颜色码（PAD/MASK 另编码），grid [B,N,N] 秩（MASK 另编码），
    k [B]，text [B,Dt] 池化文本嵌入（空文本用 null），color [B,4] = [r,g,b,1] 平均色（丢弃时全零）。"""

    def __init__(self, n_codes: int, text_dim: int = 512, d: int = 384, depth: int = 12,
                 heads: int = 6, drop: float = 0.1, bias_freqs: int = 1, level_emb: bool = False,
                 bias_hidden: int = 64):
        super().__init__()
        self.n_codes = n_codes
        self.PAL_MASK, self.PAL_PAD = n_codes, n_codes + 1
        self.GRID_MASK = K_MAX
        self.pal_emb = nn.Embedding(n_codes + 2, d)
        self.slot_emb = nn.Parameter(torch.randn(K_MAX, d) * 0.02)
        self.grid_emb = nn.Embedding(K_MAX + 1, d)
        self.k_emb = nn.Embedding(K_MAX + 1, d)
        self.text_proj = nn.Sequential(nn.Linear(text_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.null_text = nn.Parameter(torch.zeros(text_dim))
        self.color_proj = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, d))
        # 颜色 = [r,g,b,1]（给出）或全零（丢弃）。第 4 维是标志位：纯黑的平均色也是 (0,0,0)，
        # 不加标志位模型分不清「没给颜色」和「给的是黑色」。
        self.null_color = nn.Parameter(torch.zeros(4), requires_grad=False)
        self.bias = ToroidalBias(heads, hidden=bias_hidden, freqs=bias_freqs)
        # 归一化色阶嵌入：秩 r 在 k 色瓦片里的相对亮度位置 round(15 r/(k-1))。
        # 单纯的秩记号含义随 k 变（k=3 的"秩 2"是最亮，k=16 的"秩 2"很暗），
        # 同样的结构换个 k 就是完全不同的记号；加上它，"最亮/最暗"跨 k 共享。
        self.level_emb = nn.Embedding(K_MAX + 1, d) if level_emb else None
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(depth)])
        self.nf = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_f = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        nn.init.zeros_(self.ada_f[1].weight)
        nn.init.zeros_(self.ada_f[1].bias)
        self.head_pal = nn.Linear(d, n_codes)
        self.head_grid = nn.Linear(d, K_MAX)

    def forward(self, pal, grid, k, text, color):
        B, N, _ = grid.shape
        xp = self.pal_emb(pal) + self.slot_emb[None]
        xg = self.grid_emb(grid.view(B, -1))
        if self.level_emb is not None:
            g = grid.view(B, -1)
            lvl = torch.round(15.0 * g.clamp(max=K_MAX - 1).float()
                              / (k[:, None].float() - 1).clamp(min=1)).long().clamp(0, 15)
            lvl = torch.where(g == self.GRID_MASK, torch.full_like(lvl, K_MAX), lvl)
            xg = xg + self.level_emb(lvl)
        x = torch.cat([xp, xg], 1)
        c = self.text_proj(text) + self.k_emb(k) + self.color_proj(color)
        # PAD 调色板槽不屏蔽：它是可学的 PAD 记号（"此槽不用"本身就是信息），只是不计损失
        bias = self.bias(N, x.device)
        for blk in self.blocks:
            x = blk(x, c, bias)
        sh, sc = self.ada_f(c).chunk(2, -1)
        x = modulate(self.nf(x), sh, sc)
        return self.head_pal(x[:, :K_MAX]), self.head_grid(x[:, K_MAX:]).view(B, N, N, K_MAX)


# ------------------------------------------------------------------ 训练用：掩码与损失
def mask_ratio(u):
    """MaskGIT 余弦日程：u~U(0,1) → 掩码比例。"""
    return torch.cos(0.5 * math.pi * u)


def training_loss(model, pal, grid, k, text, color, pal_smooth: float = 0.0):
    B, N, _ = grid.shape
    r = mask_ratio(torch.rand(B, device=grid.device))
    valid_pal = pal != model.PAL_PAD
    mp = (torch.rand(pal.shape, device=pal.device) < r[:, None]) & valid_pal
    mg = torch.rand(grid.shape, device=grid.device) < r[:, None, None]
    # 至少掩一个，避免空损失
    mg[:, 0, 0] |= ~(mp.any(1) | mg.flatten(1).any(1))
    pal_in = torch.where(mp, torch.full_like(pal, model.PAL_MASK), pal)
    grid_in = torch.where(mg, torch.full_like(grid, model.GRID_MASK), grid)
    lp, lg = model(pal_in, grid_in, k, text, color)
    # 秩必须 < k：屏蔽越界类别
    rank_ok = torch.arange(K_MAX, device=grid.device)[None] < k[:, None]
    lg = lg.masked_fill(~rank_ok[:, None, None, :], float("-inf"))
    loss_g = F.cross_entropy(lg[mg], grid[mg]) if mg.any() else lg.sum() * 0
    loss_p = F.cross_entropy(lp[mp], pal[mp], label_smoothing=pal_smooth) if mp.any() else lp.sum() * 0
    return loss_g + loss_p, {"loss_grid": loss_g.item(), "loss_pal": loss_p.item()}


# ------------------------------------------------------------------ 采样
@torch.no_grad()
def top_p_filter(logits, p):
    """nucleus 截断：只保留累计概率达到 p 的最高那些类别。"""
    if p >= 1.0:
        return logits
    sl, si = logits.sort(-1, descending=True)
    cum = F.softmax(sl, -1).cumsum(-1)
    drop = cum - F.softmax(sl, -1) > p          # 第一个越过 p 的类别本身保留
    sl = sl.masked_fill(drop, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, si, sl)


def sample(model, text, k, n=16, color=None, steps=24, temp=1.0, cfg=2.0,
           choice_temp=4.5, null_text=None, pal_top_p=0.9):
    """MaskGIT 式迭代解码 + CFG。返回 (pal_codes [B,16], ranks [B,n,n])。

    pal_top_p：调色板槽的 nucleus 截断。训练时对调色板码用了标签平滑（v2 的 0.1），
    它把 10% 概率均匀摊到 512 个颜色上——直接按模型分布采样，每个槽约有 10% 机会抽到
    一个完全随机的颜色（v2 第 3000 步诊断图里红底上的青黄线即此）。截掉尾部即可。
    """
    B = text.shape[0]
    dev = text.device
    pal = torch.full((B, K_MAX), model.PAL_MASK, dtype=torch.long, device=dev)
    pal[torch.arange(K_MAX, device=dev)[None] >= k[:, None]] = model.PAL_PAD
    grid = torch.full((B, n, n), model.GRID_MASK, dtype=torch.long, device=dev)
    color = color if color is not None else model.null_color[None].expand(B, -1)
    nt = model.null_text[None].expand(B, -1) if null_text is None else null_text
    total = (pal == model.PAL_MASK).sum(1) + n * n
    rank_ok = torch.arange(K_MAX, device=dev)[None] < k[:, None]
    for s in range(steps):
        lp, lg = model(pal, grid, k, text, color)
        if cfg != 1.0:
            up, ug = model(pal, grid, k, nt, color)
            lp, lg = up + cfg * (lp - up), ug + cfg * (lg - ug)
        lg = lg.masked_fill(~rank_ok[:, None, None, :], float("-inf"))
        # 分别采样调色板槽与网格格
        pp = F.softmax(top_p_filter(lp / temp, pal_top_p), -1)
        sp = torch.multinomial(pp.view(-1, pp.shape[-1]), 1).view(B, K_MAX)
        cp = torch.gather(pp, -1, sp[..., None]).squeeze(-1).log()
        pg = F.softmax(lg / temp, -1)
        sg = torch.multinomial(pg.view(-1, K_MAX), 1).view(B, n, n)
        cg = torch.gather(pg, -1, sg[..., None]).squeeze(-1).log()
        pal_m, grid_m = pal == model.PAL_MASK, grid == model.GRID_MASK
        conf = torch.cat([cp, cg.view(B, -1)], 1)
        still = torch.cat([pal_m, grid_m.view(B, -1)], 1)
        # 置信度加 Gumbel 噪声，噪声随步数衰减（MaskGIT）
        g = -torch.log(-torch.log(torch.rand_like(conf).clamp(1e-9, 1)))
        conf = conf + choice_temp * (1 - (s + 1) / steps) * g
        conf = conf.masked_fill(~still, float("inf"))      # 已确定的不动
        keep_masked = (total.float() * mask_ratio(torch.tensor((s + 1) / steps))).long()
        if s == steps - 1:
            keep_masked = torch.zeros_like(keep_masked)
        order = conf.argsort(1)                            # 置信度低的排前面 → 继续掩着
        rank = order.argsort(1)
        remask = rank < keep_masked[:, None]
        newp = torch.where(pal_m, sp, pal)
        newg = torch.where(grid_m, sg, grid)
        pal = torch.where(remask[:, :K_MAX] & pal_m, pal, newp)
        grid = torch.where(remask[:, K_MAX:].view(B, n, n) & grid_m, grid, newg)
    return pal, grid
