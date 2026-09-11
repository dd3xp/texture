"""结构范例库（v7，检索增强）：给每个目标挑几张**同材质、其他画师**的 16px 真人瓦片当结构条件。

判官在验证集上：真人 vs B2 真人胜 66%，TRD vs B2 TRD 只胜 35%；并排看，TRD 的结构与材质关系很弱
（圆石、砖、黏土都画成差不多的横条纹）。一个材质名在训练集里往往只出现几次、各包画法又不同，
单凭名字学不到"这种材质长什么样"。调色板那一步已经证明"检索真人数据当条件"在这个数据规模下极有效，
结构这一步照做：把同名材质在其他包里的画法给模型看，让它学"跨画师概括"。

防抄（写死）：
- 训练时范例只取**其他包**的瓦片，且与目标的归一化色阶网格逐格差异 < `min_diff` 的剔除
  （训练集里也有变体包，结构孪生不剔掉，模型就会学成"照抄范例"）；
- 范例每次随机循环平移 + 水平翻转（没有绝对位置可抄）；
- 以 p_drop 的概率整组丢弃（保留纯文本能力）。
候选：同文件名（跨包对齐的材质标签）的瓦片；不足 2 张时退到 CLIP 文本嵌入最近的 5 个材质。
测试时（val/test 包都不在库里）同一规则，天然全是其他画师。
"""
import numpy as np
import torch


def levels(s):
    """亮度序秩 → 归一化色阶 0..15（与 TRD 的 level_emb 同一换算）。"""
    k = max(int(s["k_used"]) - 1, 1)
    return np.round(15.0 * np.asarray(s["idx"], np.float64) / k).astype(np.int64)


class ExemplarBank:
    def __init__(self, pool, mat_emb, dev, C=16, min_diff=0.3, seed=0):
        """pool：16px 训练瓦片（tiles_data.load 的输出）；mat_emb：{材质名: 归一化文本嵌入}。"""
        self.dev, self.C, self.min_diff = dev, C, min_diff
        self.lv = torch.tensor(np.stack([levels(s) for s in pool]), device=dev)       # [P,16,16]
        self.mat = [s["material"] for s in pool]
        self.pack = [s.get("pack") for s in pool]
        self.by_mat = {}
        for j, m in enumerate(self.mat):
            self.by_mat.setdefault(m, []).append(j)
        self.mats = sorted(self.by_mat)
        self.E = torch.stack([mat_emb[m] for m in self.mats]).float().to(dev)
        self.mat_emb = mat_emb
        self.rng = np.random.default_rng(seed)

    def _near(self, material, emb=None, n=5):
        e = (emb if emb is not None else self.mat_emb[material]).float().to(self.dev)
        sims = (self.E @ e).cpu().numpy()
        order = [self.mats[i] for i in np.argsort(-sims)]
        return [m for m in order if m != material][:n]

    def candidates(self, samples, emb=None):
        """每个目标的候选范例下标 → LongTensor [N, C]（-1 填充）。emb：可选 [N,D] 文本嵌入（库外材质名用）。"""
        out = np.full((len(samples), self.C), -1, np.int64)
        for i, s in enumerate(samples):
            own_pack, m = s.get("pack"), s["material"]
            cand = [j for j in self.by_mat.get(m, []) if self.pack[j] != own_pack]
            if len(cand) < 2:
                near = self._near(m, None if emb is None else emb[i]) if (m in self.mat_emb or emb is not None) else []
                cand += [j for mm in near for j in self.by_mat[mm] if self.pack[j] != own_pack]
            if "idx" in s and np.asarray(s["idx"]).shape == (16, 16) and cand:    # 剔除结构孪生
                me = torch.tensor(levels(s), device=self.dev)
                diff = (self.lv[cand] != me).float().mean((1, 2)).cpu().numpy()
                cand = [j for j, d in zip(cand, diff) if d >= self.min_diff]
            if len(cand) > self.C:
                cand = list(self.rng.choice(cand, self.C, replace=False))
            out[i, :len(cand)] = cand
        return torch.tensor(out, device=self.dev)

    def draw(self, cand, E, train_mode, p_drop=0.0, gen=None):
        """cand [B,C] → 范例色阶 [B,E,16,16]（无候选或丢弃处为 -1）。"""
        B = cand.shape[0]
        n = (cand >= 0).sum(1)
        pick = (torch.rand(B, E, device=self.dev, generator=gen) * n.clamp(min=1)[:, None]).long()
        idx = torch.gather(cand, 1, pick)                                   # [B,E]
        ex = self.lv[idx.clamp(min=0)]                                      # [B,E,16,16]
        if train_mode:                                                      # 随机循环平移 + 翻转：没有绝对位置可抄
            dy = torch.randint(0, 16, (B, E), device=self.dev, generator=gen)
            dx = torch.randint(0, 16, (B, E), device=self.dev, generator=gen)
            ar = torch.arange(16, device=self.dev)
            rows = (ar[None, None] - dy[..., None]) % 16
            cols = (ar[None, None] - dx[..., None]) % 16
            bi = torch.arange(B, device=self.dev)[:, None, None, None]
            ei = torch.arange(E, device=self.dev)[None, :, None, None]
            ex = ex[bi, ei, rows[..., :, None], cols[..., None, :]]
            flip = torch.rand(B, E, device=self.dev, generator=gen) < 0.5
            ex = torch.where(flip[..., None, None], ex.flip(-1), ex)
        bad = (idx < 0) | (n == 0)[:, None]
        if train_mode and p_drop > 0:
            bad = bad | (torch.rand(B, 1, device=self.dev, generator=gen) < p_drop)
        return torch.where(bad[..., None, None], torch.full_like(ex, -1), ex)
