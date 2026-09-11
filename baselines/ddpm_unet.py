"""B7：同数据从零训练的像素空间扩散 UNet（标准做法的基线）。

回答"不用 TRD 的表示与检索、换成标准的文本条件扩散模型，同样的数据能做到哪"——
区分"架构的贡献"与"数据的贡献"。为了让它尽量强、比较公平：
- **数据与 TRD v11d 完全相同**：训练包（去污染 + 补回）16px + 32px（含 64→32 众数降采样），不含模组；
- **条件相同**：材质名的 CLIP-B/32 池化文本嵌入（同一模板）+ 平均颜色 [r,g,b,1]；各自随机丢弃，推理用 CFG；
- **同样的增广**：循环平移 + 水平翻转 + 色相/亮度抖动；
- **环形填充**（所有卷积 `padding_mode="circular"`）→ 天然可平铺，与 TRD 的环面位置编码对等；
- 全卷积 + 注意力，16/24/32 都能出；输出量化到 ≤16 色（与数据集同一中位切分），符合像素画格式。
模型：UNet（128/256/256 通道，每级 2 个残差块，低两级带自注意力，adaGN 注入时间 + 条件），
余弦日程 1000 步、v-prediction，DDIM 50 步采样。

    python baselines/ddpm_unet.py train --out runs/b7_ddpm
    python baselines/ddpm_unet.py gen --run runs/b7_ddpm --set V_mat --size 16 --n 2 --tag B7
    python baselines/ddpm_unet.py gen --run runs/b7_ddpm --set V_mat --size 16 --colour_task --tag B7
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "analysis" / "dataset"))

EXTRA = "train_extra_packs_only.json+train_64to32.json"


# ------------------------------------------------------------------ 模型
def conv(ci, co, k=3, s=1):
    return nn.Conv2d(ci, co, k, s, padding=k // 2, padding_mode="circular")


class Res(nn.Module):
    def __init__(self, ci, co, de):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(32, ci), conv(ci, co)
        self.n2, self.c2 = nn.GroupNorm(32, co), conv(co, co)
        self.emb = nn.Linear(de, 2 * co)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.emb(e)[:, :, None, None].chunk(2, 1)
        h = self.c2(F.silu(self.n2(h) * (1 + sc) + sh))
        return self.skip(x) + h


class Attn(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.n, self.qkv, self.o, self.h = nn.GroupNorm(32, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), heads
        nn.init.zeros_(self.o.weight)
        nn.init.zeros_(self.o.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).view(B, 3, self.h, C // self.h, H * W).permute(1, 0, 2, 4, 3)
        y = F.scaled_dot_product_attention(q, k, v)                 # [B,h,HW,d]
        return x + self.o(y.permute(0, 1, 3, 2).reshape(B, C, H, W))


class UNet(nn.Module):
    def __init__(self, ch=(128, 256, 256), text_dim=512, de=512):
        super().__init__()
        self.de = de
        self.t_mlp = nn.Sequential(nn.Linear(128, de), nn.SiLU(), nn.Linear(de, de))
        self.txt = nn.Sequential(nn.Linear(text_dim, de), nn.SiLU(), nn.Linear(de, de))
        self.col = nn.Sequential(nn.Linear(4, de), nn.SiLU(), nn.Linear(de, de))
        self.null_text = nn.Parameter(torch.zeros(text_dim))
        self.inp = conv(3, ch[0])
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        c_prev, skips = ch[0], [ch[0]]
        for i, c in enumerate(ch):
            for _ in range(2):
                self.down.append(nn.ModuleList([Res(c_prev, c, de), Attn(c) if i > 0 else nn.Identity()]))
                c_prev = c
                skips.append(c)
            if i < len(ch) - 1:
                self.down.append(conv(c, c, 3, 2))
                skips.append(c)
        self.mid = nn.ModuleList([Res(c_prev, c_prev, de), Attn(c_prev), Res(c_prev, c_prev, de)])
        for i, c in reversed(list(enumerate(ch))):
            for _ in range(3):
                self.up.append(nn.ModuleList([Res(c_prev + skips.pop(), c, de), Attn(c) if i > 0 else nn.Identity()]))
                c_prev = c
            if i > 0:
                self.up.append(conv(c, c))                           # 最近邻放大后接卷积
        self.out = nn.Sequential(nn.GroupNorm(32, c_prev), nn.SiLU(), conv(c_prev, 3))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, text, color):
        f = torch.exp(-math.log(10000) * torch.arange(64, device=x.device) / 64)
        a = t[:, None].float() * 1000 * f[None]
        e = self.t_mlp(torch.cat([a.sin(), a.cos()], 1)) + self.txt(text) + self.col(color)
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](h, e))
            else:
                h = m(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, e)), e)
        for m in self.up:
            if isinstance(m, nn.ModuleList):
                h = m[1](m[0](torch.cat([h, hs.pop()], 1), e))
            else:
                h = m(F.interpolate(h, size=hs[-1].shape[-2:], mode="nearest"))
        return self.out(h)


# ------------------------------------------------------------------ 扩散（余弦日程，连续 t∈[0,1]，v-prediction）
def alpha_sigma(t):
    return torch.cos(0.5 * math.pi * t), torch.sin(0.5 * math.pi * t)


# ------------------------------------------------------------------ 数据
def load_data(size, dev):
    from tiles_data import load
    from train_trd import clip_text, text_prompt
    rows = load(size, "train", extra=EXTRA)
    x = torch.tensor(np.stack([s["palette"][s["idx"]] for s in rows])).permute(0, 3, 1, 2).float() / 127.5 - 1
    mats = sorted({s["material"] for s in rows})
    te = clip_text([text_prompt(m) for m in mats], dev).float().to(dev)
    pos = {m: i for i, m in enumerate(mats)}
    ix = torch.tensor([pos[s["material"]] for s in rows], device=dev)
    return x.to(dev), te[ix]


def augment(x, pal_aug=0.3):
    B, _, N, _ = x.shape
    dy, dx = torch.randint(0, N, (2,)).tolist()
    x = torch.roll(x, (dy, dx), (2, 3))                           # 同一批统一平移：可平铺的纹理任意平移都合法
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    th = (torch.rand(B, device=x.device) * 2 - 1) * pal_aug * math.pi / 3     # 色相绕灰轴旋转 + 亮度（与 TRD 同式）
    br = 1 + (torch.rand(B, device=x.device) * 2 - 1) * pal_aug * 0.5
    ax = torch.ones(3, device=x.device) / math.sqrt(3)
    K = torch.tensor([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], device=x.device)
    rgb = (x + 1) * 127.5
    r = rgb.permute(0, 2, 3, 1)
    s_, c_ = th.sin()[:, None, None, None], (1 - th.cos())[:, None, None, None]
    r = (r + s_ * (r @ K.T) + c_ * (r @ (K @ K).T)) * br[:, None, None, None]
    return (r.clamp(0, 255).permute(0, 3, 1, 2) / 127.5 - 1)


def train(a):
    dev = "cuda"
    torch.manual_seed(0)
    torch.set_num_threads(4)
    a.out.mkdir(parents=True, exist_ok=True)
    X16, T16 = load_data(16, dev)
    X32, T32 = load_data(32, dev)
    print(f"B7 数据：16px {len(X16)}、32px {len(X32)}", flush=True)
    net = UNet().to(dev)
    ema = UNet().to(dev).requires_grad_(False)
    ema.load_state_dict(net.state_dict())
    print(f"参数 {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000))
    json.dump({k: str(v) for k, v in vars(a).items()}, open(a.out / "config.json", "w"), indent=1)
    t0, log = time.time(), []
    for step in range(a.steps + 1):
        use32 = torch.rand(()).item() < a.p32
        X, T, bs = (X32, T32, a.batch // 2) if use32 else (X16, T16, a.batch)
        idx = torch.randint(0, len(X), (bs,), device=dev)
        x0 = augment(X[idx])
        text = T[idx].clone()
        col = torch.cat([((x0 + 1) / 2).mean((2, 3)), torch.ones(bs, 1, device=dev)], 1)
        text[torch.rand(bs, device=dev) < a.p_text_drop] = net.null_text
        col[torch.rand(bs, device=dev) < a.p_color_drop] = 0
        t = torch.rand(bs, device=dev)
        al, si = alpha_sigma(t)
        eps = torch.randn_like(x0)
        xt = al[:, None, None, None] * x0 + si[:, None, None, None] * eps
        v = al[:, None, None, None] * eps - si[:, None, None, None] * x0
        with torch.autocast(dev, dtype=torch.bfloat16):
            pred = net(xt, t, text, col)
        loss = F.mse_loss(pred.float(), v)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), net.parameters()):
                pe.mul_(0.999).add_(pm.detach(), alpha=0.001)
        if step % 1000 == 0:
            rec = {"step": step, "loss": loss.item(), "min": (time.time() - t0) / 60}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(log, open(a.out / "log.json", "w"))
            torch.save({"model": ema.state_dict(), "step": step}, a.out / "last.pt")


@torch.no_grad()
def ddim(net, text, color, size, steps=50, cfg=3.0, seed=0):
    B = text.shape[0]
    dev = text.device
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(B, 3, size, size, device=dev, generator=g)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    nt, nc = net.null_text[None].expand(B, -1), torch.zeros_like(color)
    for i in range(steps):
        t, tn = ts[i].expand(B), ts[i + 1].expand(B)
        v = net(x, t, text, color)
        if cfg != 1:
            vu = net(x, t, nt, nc)
            v = vu + cfg * (v - vu)
        al, si = alpha_sigma(t)
        aln, sin = alpha_sigma(tn)
        al, si, aln, sin = (z[:, None, None, None] for z in (al, si, aln, sin))
        x0 = (al * x - si * v).clamp(-1, 1)
        eps = (si * x + al * v)
        x = aln * x0 + sin * eps
    return x0


def to_pixel_art(x, k=16):
    from prepare import quantize
    out = []
    for t in ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy():
        idx, pal = quantize(t, k)
        out.append(pal[idx].astype(np.uint8))
    return out


def gen(a):
    from PIL import Image
    from prompts import load_set
    from train_trd import clip_text, TEXT_TMPL
    dev = "cuda"
    net = UNet().to(dev).eval()
    net.load_state_dict(torch.load(a.run / "last.pt", map_location=dev)["model"])
    prompts, _ = load_set(a.set)
    temb = clip_text([TEXT_TMPL.format(p=e["prompt"]) for e in prompts], dev).float().to(dev)
    if a.colour_task:
        from colour_task import targets
        T = targets(a.set, a.size if a.size in (16, 32) else 16)
        pidx = {e["prompt"]: i for i, e in enumerate(prompts)}
        out = ROOT / "experiments/colour" / a.tag / str(a.size)
        out.mkdir(parents=True, exist_ok=True)
        for i in range(0, len(T), a.bs):
            ch = T[i:i + a.bs]
            text = temb[[pidx[t["prompt"]] for t in ch]]
            col = torch.tensor([[*(np.array(t["rgb"]) / 255.0), 1.0] for t in ch], device=dev).float()
            for j, im in enumerate(to_pixel_art(ddim(net, text, col, a.size, a.steps, a.cfg, seed=i))):
                Image.fromarray(im).save(out / f"{ch[j]['slug']}_{ch[j]['j']}.png")
        print("->", out, len(T))
        return
    out = ROOT / "experiments/baselines" / a.tag / str(a.size)
    out.mkdir(parents=True, exist_ok=True)
    for kk in range(a.n):
        for i in range(0, len(prompts), a.bs):
            ch = prompts[i:i + a.bs]
            text = temb[i:i + len(ch)]
            col = torch.zeros(len(ch), 4, device=dev)                   # 名字 → 纹理：不给颜色（与 TRD、B1/B2 同等）
            for j, im in enumerate(to_pixel_art(ddim(net, text, col, a.size, a.steps, a.cfg, seed=1000 * kk + i))):
                Image.fromarray(im).save(out / f"{ch[j]['material'].rsplit('.', 1)[0]}_{kk}.png")
        print(f"sample {kk + 1}/{a.n} done", flush=True)
    print("->", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=40000)
    t.add_argument("--batch", type=int, default=128)
    t.add_argument("--p32", type=float, default=0.5)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--p_text_drop", type=float, default=0.1)
    t.add_argument("--p_color_drop", type=float, default=0.5)
    g = sub.add_parser("gen")
    g.add_argument("--run", type=Path, required=True)
    g.add_argument("--set", default="V_mat")
    g.add_argument("--size", type=int, default=16)
    g.add_argument("--n", type=int, default=2)
    g.add_argument("--bs", type=int, default=64)
    g.add_argument("--steps", type=int, default=50)
    g.add_argument("--cfg", type=float, default=3.0)
    g.add_argument("--colour_task", action="store_true")
    g.add_argument("--tag", default="B7")
    a = ap.parse_args()
    train(a) if a.cmd == "train" else gen(a)


if __name__ == "__main__":
    main()
