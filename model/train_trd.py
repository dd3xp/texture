"""训练 TRD（新架构 v1）。数据：`split=train` 的 16×16 真人瓦片；`split=val` 选检查点。

用法（服务器，离线）：
    HF_HUB_OFFLINE=1 python model/train_trd.py --out runs/trd_v1 --steps 40000
冒烟：
    python model/train_trd.py --smoke        （极小配置跑几十步 + 采样一次）
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from tiles_data import load, W_LUM                      # noqa: E402
from trd import TRD, K_MAX, training_loss, sample       # noqa: E402
from prompts import prompt_words                        # noqa: E402

TEXT_TMPL = "pixel art texture of {p}"


# ------------------------------------------------------------------ 颜色码本
def kmeans(x, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)].astype(np.float64)
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1) if len(x) * k < 4e7 else None
        if d is None:                                   # 大时分块
            lab = np.concatenate([((x[i:i + 20000, None] - c[None]) ** 2).sum(-1).argmin(1)
                                  for i in range(0, len(x), 20000)])
        else:
            lab = d.argmin(1)
        for j in range(k):
            sel = x[lab == j]
            c[j] = sel.mean(0) if len(sel) else x[rng.integers(len(x))]
    return c


def build_codebook(samples, k):
    cols = np.concatenate([s["palette"].astype(np.float64) for s in samples])
    cb = kmeans(cols, k)
    # 按亮度排序，便于查看
    cb = cb[np.argsort(cb @ W_LUM)]
    err = np.sqrt(((cols[:, None] - cb[None]) ** 2).sum(-1).min(1)).mean()
    return cb.astype(np.float32), float(err)


def encode_palette(pal, cb):
    d = ((pal[:, None].astype(np.float32) - cb[None]) ** 2).sum(-1)
    return d.argmin(1)


# ------------------------------------------------------------------ 文本
def text_prompt(material):
    w = prompt_words(material)
    return TEXT_TMPL.format(p=" ".join(w) if w else material)


@torch.no_grad()
def clip_text(prompts, dev):
    from transformers import CLIPModel, CLIPTokenizer
    name = "openai/clip-vit-base-patch32"
    m = CLIPModel.from_pretrained(name).to(dev).eval()
    tok = CLIPTokenizer.from_pretrained(name)
    out = []
    for i in range(0, len(prompts), 256):
        t = tok(prompts[i:i + 256], padding=True, return_tensors="pt").to(dev)
        out.append(F.normalize(m.get_text_features(**t).float(), dim=-1).cpu())
    del m
    return torch.cat(out)


# ------------------------------------------------------------------ 数据张量
def to_tensors(samples, cb, n_codes, text_index, text_emb):
    B = len(samples)
    pal = np.full((B, K_MAX), n_codes + 1, np.int64)          # PAD
    N = samples[0]["idx"].shape[0]
    grid = np.zeros((B, N, N), np.int64)
    k = np.zeros(B, np.int64)
    col = np.zeros((B, 4), np.float32)
    tix = np.zeros(B, np.int64)
    rgb = np.zeros((B, K_MAX, 3), np.float32)
    hist = np.zeros((B, K_MAX), np.float32)
    for i, s in enumerate(samples):
        kk = s["k_used"]
        pal[i, :kk] = encode_palette(s["palette"], cb)
        grid[i] = s["idx"]
        k[i] = kk
        col[i, :3] = (s["palette"][s["idx"]].reshape(-1, 3).mean(0) / 255.0).astype(np.float32)
        col[i, 3] = 1.0                                   # 标志位：颜色已给出
        rgb[i, :kk] = s["palette"]
        hist[i, :kk] = np.bincount(s["idx"].reshape(-1), minlength=kk)[:kk] / s["idx"].size
        tix[i] = text_index[s["material"]]
    return {"pal": torch.from_numpy(pal), "grid": torch.from_numpy(grid), "k": torch.from_numpy(k),
            "color": torch.from_numpy(col), "text": text_emb[torch.from_numpy(tix)],
            "rgb": torch.from_numpy(rgb), "hist": torch.from_numpy(hist)}


def augment(grid):
    """循环平移（每样本独立）+ 水平翻转。纹理可平铺，任意循环平移都是合法样本。"""
    B, N, _ = grid.shape
    dy = torch.randint(0, N, (B,))
    dx = torch.randint(0, N, (B,))
    ar = torch.arange(N)
    rows = (ar[None] - dy[:, None]) % N
    cols = (ar[None] - dx[:, None]) % N
    g = grid[torch.arange(B)[:, None, None], rows[:, :, None], cols[:, None, :]]
    flip = torch.rand(B) < 0.5
    g[flip] = g[flip].flip(-1)
    return g


def decode(pal_codes, ranks, cb):
    """pal_codes [B,16], ranks [B,N,N] -> RGB uint8 [B,N,N,3]"""
    cbt = torch.as_tensor(cb)
    codes = pal_codes.clamp(max=len(cb) - 1).cpu()
    cols = cbt[codes]                                      # [B,16,3]
    r = ranks.cpu()
    img = torch.gather(cols, 1, r.view(r.shape[0], -1, 1).expand(-1, -1, 3))
    return img.view(*r.shape, 3).round().clamp(0, 255).byte().numpy()


def model_from_args(a, drop=None):
    """从检查点里存的参数建模型（v1 检查点没有 v2 字段，按 v1 默认值补）。"""
    g = (lambda k, d: a.get(k, d)) if isinstance(a, dict) else (lambda k, d: getattr(a, k, d))
    return TRD(int(g("codes", 512)), d=int(g("d", 384)), depth=int(g("depth", 12)),
               heads=int(g("heads", 6)), drop=float(g("drop", 0.1) if drop is None else drop),
               bias_freqs=int(g("bias_freqs", 1)), level_emb=bool(g("level_emb", False)),
               bias_hidden=int(g("bias_hidden", 64)))


def hue_rotation(theta):
    """RGB 空间绕灰轴旋转 theta 弧度（Rodrigues），近似色相旋转。"""
    k = torch.ones(3) / math.sqrt(3.0)
    K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return torch.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/trd_v1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--codes", type=int, default=512)
    ap.add_argument("--p_text_drop", type=float, default=0.1)
    ap.add_argument("--p_color_drop", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16],
                    help="训练用的分辨率。32px 真人数据只有 517 张，与 16 混训；选检查点只看 16px val")
    ap.add_argument("--p32", type=float, default=0.3, help="每步取 32px 批次的概率")
    ap.add_argument("--batch32", type=int, default=64, help="32px 序列长 4 倍，批次相应缩小")
    ap.add_argument("--bias_freqs", type=int, default=1, help="v1=1；v2=8（见 trd.ToroidalBias）")
    ap.add_argument("--bias_hidden", type=int, default=64)
    ap.add_argument("--level_emb", action="store_true", help="v2：秩的归一化色阶嵌入")
    ap.add_argument("--pal_aug", type=float, default=0.0,
                    help="调色板颜色抖动幅度：色相 ±pal_aug*60°、亮度 ±pal_aug*50%%（v1=0）")
    ap.add_argument("--pal_smooth", type=float, default=0.0, help="调色板码标签平滑（v1=0）")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps, a.batch, a.d, a.depth, a.heads, a.codes, a.eval_every = 40, 16, 64, 2, 2, 64, 20
        a.out = ROOT / "runs/trd_smoke"
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    train, val = load(16, "train"), load(16, "val")
    test = load(16, "test")
    train32 = load(32, "train") if 32 in a.sizes else []
    val32 = load(32, "val") if 32 in a.sizes else []
    cb, err = build_codebook(train + train32, a.codes)
    np.save(a.out / "codebook.npy", cb)
    print(f"train {len(train)}  val {len(val)}  码本 {a.codes} 色，平均量化误差 {err:.2f}/255", flush=True)

    mats = sorted({s["material"] for s in train + val + test + train32 + val32})
    prompts = [text_prompt(m) for m in mats]
    temb = clip_text(prompts, dev) if not a.smoke else F.normalize(torch.randn(len(mats), 512), dim=-1)
    tindex = {m: i for i, m in enumerate(mats)}
    torch.save({"materials": mats, "prompts": prompts, "emb": temb}, a.out / "text_emb.pt")

    T = to_tensors(train, cb, a.codes, tindex, temb)
    V = to_tensors(val, cb, a.codes, tindex, temb)
    T32 = to_tensors(train32, cb, a.codes, tindex, temb) if train32 else None
    V32 = to_tensors(val32, cb, a.codes, tindex, temb) if val32 else None
    model = model_from_args(a).to(dev)
    cbt = torch.as_tensor(cb, device=dev)
    print(f"参数 {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    cfg = {kk: (str(v) if isinstance(v, Path) else v) for kk, v in vars(a).items()} | {"codebook_err": err}
    json.dump(cfg, open(a.out / "config.json", "w"), indent=1)

    def batch_of(D, idx, train_mode):
        g = D["grid"][idx]
        if train_mode:
            g = augment(g)
        t = D["text"][idx].clone()
        c = D["color"][idx].clone()
        pal = D["pal"][idx]
        if train_mode and a.pal_aug > 0:
            # 颜色抖动：同一材质各包配色本就不同；逼模型学相对配色而不是背训练包的色值
            B = len(idx)
            rgb = D["rgb"][idx].clone()
            th = (torch.rand(B) * 2 - 1) * a.pal_aug * math.pi / 3
            br = 1 + (torch.rand(B) * 2 - 1) * a.pal_aug * 0.5
            for j in range(B):
                rgb[j] = (rgb[j] @ hue_rotation(float(th[j])).T) * br[j]
            rgb = rgb.clamp(0, 255)
            valid = pal != a.codes + 1
            codes = ((rgb.to(dev)[:, :, None] - cbt[None, None]) ** 2).sum(-1).argmin(-1).cpu()
            pal = torch.where(valid, codes, pal)
            mean = (D["hist"][idx][:, :, None] * rgb).sum(1) / 255.0
            c = torch.cat([mean, torch.ones(B, 1)], 1)
        if train_mode:
            dt = torch.rand(len(idx)) < a.p_text_drop
            dc = torch.rand(len(idx)) < a.p_color_drop
            t[dt] = model.null_text.detach().cpu()
            c[dc] = model.null_color.detach().cpu()
        else:                                             # 评估：丢颜色条件（与基线同等）
            c[:] = model.null_color.detach().cpu()
        return [x.to(dev) for x in (pal, g, D["k"][idx], t, c)]

    val_parts = [0.0, 0.0]

    def val_loss(Vd, nval):
        """固定掩码随机性让各次验证损失可比；fork_rng 隔离，不污染训练的随机序列。"""
        tot, tg, tp, cnt = 0.0, 0.0, 0.0, 0
        devs = [torch.cuda.current_device()] if dev == "cuda" else []
        with torch.random.fork_rng(devices=devs), torch.no_grad(), \
                torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            torch.manual_seed(123)
            for i in range(0, nval, 256):
                idx = torch.arange(i, min(i + 256, nval))
                l, parts = training_loss(model, *batch_of(Vd, idx, False))
                tot += l.item() * len(idx)
                tg += parts["loss_grid"] * len(idx)
                tp += parts["loss_pal"] * len(idx)
                cnt += len(idx)
        # 分开报结构与调色板：两者过拟合的原因不同，混在一起看不出是哪边在涨
        val_parts[:] = [tg / cnt, tp / cnt]
        return tot / cnt

    best, log = float("inf"), []
    t0 = time.time()
    n = len(train)
    for step in range(a.steps + 1):
        model.train()
        use32 = T32 is not None and torch.rand(()).item() < a.p32
        D, nD, bs = (T32, len(train32), a.batch32) if use32 else (T, n, a.batch)
        idx = torch.randint(0, nD, (bs,))
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            loss, parts = training_loss(model, *batch_of(D, idx, True), pal_smooth=a.pal_smooth)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % a.eval_every == 0:
            model.eval()
            vl = val_loss(V, len(val))                       # 选检查点只看 16px val
            vg, vp = val_parts
            v32 = val_loss(V32, len(val32)) if V32 is not None else None
            rec = {"step": step, "train": loss.item(), **parts, "val": vl, "val_grid": vg, "val_pal": vp, "val32": v32,
                   "lr": sched.get_last_lr()[0], "min": (time.time() - t0) / 60}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            json.dump(log, open(a.out / "log.json", "w"), indent=0)
            torch.save({"model": model.state_dict(), "args": cfg,
                        "step": step, "val": vl}, a.out / "last.pt")
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(), "args": cfg,
                            "step": step, "val": vl}, a.out / "best.pt")

    # 训练末：用最佳检查点给 val 材质各采一张，存图便于目视
    ck = torch.load(a.out / "best.pt", map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    from PIL import Image
    for size, pool in ((16, val), (32, val32)):
        if not pool:
            continue
        sv = pool[:64]
        V2 = to_tensors(sv, cb, a.codes, tindex, temb)
        pal, grid = sample(model, V2["text"].to(dev), V2["k"].to(dev), n=size,
                           steps=24 if not a.smoke else 4)
        imgs = decode(pal, grid, cb)
        rows = len(imgs) // 8
        sheet = np.concatenate([np.concatenate(list(imgs[r * 8:(r + 1) * 8]), 1) for r in range(rows)], 0)
        up = 64 // size
        name = "val_samples.png" if size == 16 else f"val_samples_{size}.png"
        Image.fromarray(sheet).resize((sheet.shape[1] * up, sheet.shape[0] * up),
                                      Image.NEAREST).save(a.out / name)
    print(f"最佳 val {best:.4f} @ step {ck['step']}；样例 -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
