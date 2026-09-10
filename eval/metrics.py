"""像素画纹理生成的通用评测指标（M1，服务于 `GOAL.md`）。

全部是领域通用指标（出处见 `docs/arch_plan.md` 第 3 节），不自造：

  KID / FID        Inception pool3（pytorch-fid 的 FID 权重），与参照集的分布距离
  FD-DINOv2        DINOv2-small CLS 特征上的 Fréchet 距离（Stein et al. 2023 推荐）
  CLIP score       100 * cos(图, "pixel art texture of {材质}")，CLIP ViT-B/32
  LPIPS 多样性     同一提示词多个样本两两 LPIPS(alex) 的均值
  平铺性           2x2 平铺后接缝处邻格差 / 内部邻格差（1 = 接缝与内部无异，越低不一定越好，
                   只看是否明显 >1）

预处理：所有瓦片（真人与生成）走**同一套**最近邻放大（像素画不能用双线性糊掉硬边）。
服务器离线：权重在 `<repo>/weights/`（本机下载后 scp）或 HF 缓存。
没有 scipy：矩阵平方根用对称特征分解实现。

自检：`python eval/metrics.py selftest`（天花板 = 真人 val 对 test；地板 = 打乱/噪声/纯色）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "model"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- 预处理
def upscale(tiles, side):
    """list[HxWx3 uint8] -> float tensor [N,3,side,side] in [0,1]，最近邻放大。"""
    out = [np.asarray(Image.fromarray(np.asarray(t, np.uint8)).resize((side, side), Image.NEAREST))
           for t in tiles]
    x = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    return x


def batched(fn, x, bs=128):
    outs = []
    for i in range(0, len(x), bs):
        with torch.no_grad():
            outs.append(fn(x[i:i + bs].to(DEV)).float().cpu())
    return torch.cat(outs)


# ---------------------------------------------------------------- 特征网络
_NETS = {}


def inception():
    if "inc" not in _NETS:
        from fid_inception import InceptionV3
        _NETS["inc"] = InceptionV3([3], resize_input=False, normalize_input=True).to(DEV).eval()
    return _NETS["inc"]


def inception_feats(tiles):
    net = inception()
    return batched(lambda b: net(b)[0].flatten(1), upscale(tiles, 299)).double().numpy()


def dino():
    if "dino" not in _NETS:
        from transformers import AutoModel
        _NETS["dino"] = AutoModel.from_pretrained("facebook/dinov2-small").to(DEV).eval()
    return _NETS["dino"]


_IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def dino_feats(tiles):
    net = dino()
    x = (upscale(tiles, 224) - _IMNET_MEAN) / _IMNET_STD
    return batched(lambda b: net(pixel_values=b).last_hidden_state[:, 0], x).double().numpy()


def clip():
    if "clip" not in _NETS:
        from transformers import CLIPModel, CLIPTokenizer
        name = "openai/clip-vit-base-patch32"
        _NETS["clip"] = (CLIPModel.from_pretrained(name).to(DEV).eval(),
                         CLIPTokenizer.from_pretrained(name))
    return _NETS["clip"]


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_image_emb(tiles):
    m, _ = clip()
    x = (upscale(tiles, 224) - _CLIP_MEAN) / _CLIP_STD
    e = batched(lambda b: m.get_image_features(pixel_values=b), x)
    return F.normalize(e, dim=-1)


def clip_text_emb(texts):
    m, tok = clip()
    outs = []
    for i in range(0, len(texts), 256):
        t = tok(texts[i:i + 256], padding=True, return_tensors="pt").to(DEV)
        with torch.no_grad():
            outs.append(m.get_text_features(**t).float().cpu())
    return F.normalize(torch.cat(outs), dim=-1)


def prompt_of(material: str) -> str:
    return f"pixel art texture of {material}"


# ---------------------------------------------------------------- 分布距离
def _sqrtm_trace(a, b):
    """tr( sqrt(a b) )，a、b 为对称半正定；用 sqrt(a) b sqrt(a) 的特征值。"""
    wa, va = np.linalg.eigh(a)
    sa = (va * np.sqrt(np.clip(wa, 0, None))) @ va.T
    w = np.linalg.eigvalsh(sa @ b @ sa)
    return float(np.sqrt(np.clip(w, 0, None)).sum())


def frechet(fa, fb):
    mu_a, mu_b = fa.mean(0), fb.mean(0)
    ca, cb = np.cov(fa, rowvar=False), np.cov(fb, rowvar=False)
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(ca) + np.trace(cb) - 2 * _sqrtm_trace(ca, cb))


def kid(fa, fb, n_subsets=100, subset=1000, seed=0):
    """无偏 MMD²，三次多项式核 k(x,y)=(x·y/d+1)^3，按子集平均（torch-fidelity 同口径）。"""
    rng = np.random.default_rng(seed)
    d = fa.shape[1]
    m = min(subset, len(fa), len(fb))
    vals = []
    for _ in range(n_subsets):
        x = fa[rng.choice(len(fa), m, replace=False)]
        y = fb[rng.choice(len(fb), m, replace=False)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1))
                    + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


# ---------------------------------------------------------------- LPIPS (alex, v0.1)
class LPIPSAlex(torch.nn.Module):
    """LPIPS(alex) 的最小实现：AlexNet relu1..5 特征，通道归一，1x1 线性层加权，空间平均。
    与官方 lpips 包 v0.1 的前向一致（同一份线性层权重）。"""
    CH = [64, 192, 384, 256, 256]

    def __init__(self):
        super().__init__()
        import torchvision
        a = torchvision.models.alexnet(weights=None)
        a.load_state_dict(torch.load(WEIGHTS / "alexnet-owt-7be5be79.pth", map_location="cpu"))
        f = a.features
        self.slices = torch.nn.ModuleList([f[0:2], f[2:5], f[5:8], f[8:10], f[10:12]])
        lin = torch.load(WEIGHTS / "lpips_alex_v0.1.pth", map_location="cpu")
        self.lins = torch.nn.ModuleList()
        for i, c in enumerate(self.CH):
            conv = torch.nn.Conv2d(c, 1, 1, bias=False)
            conv.weight.data = lin[f"lin{i}.model.1.weight"]
            self.lins.append(conv)
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188]).view(1, 3, 1, 1))
        self.register_buffer("scale", torch.tensor([.458, .448, .450]).view(1, 3, 1, 1))
        self.eval()

    def forward(self, x, y):              # x,y in [0,1]
        x = ((x * 2 - 1) - self.shift) / self.scale
        y = ((y * 2 - 1) - self.shift) / self.scale
        d = 0
        for sl, lin in zip(self.slices, self.lins):
            x, y = sl(x), sl(y)
            nx = x / (x.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            ny = y / (y.pow(2).sum(1, keepdim=True).sqrt() + 1e-10)
            d = d + lin((nx - ny) ** 2).mean([2, 3])
        return d.flatten()


def lpips_net():
    if "lpips" not in _NETS:
        _NETS["lpips"] = LPIPSAlex().to(DEV)
    return _NETS["lpips"]


def lpips_diversity(groups, side=64):
    """groups: list[list[tile]]，每组是同一提示词的多个样本。返回组内两两 LPIPS 的均值。"""
    net = lpips_net()
    vals = []
    for g in groups:
        if len(g) < 2:
            continue
        x = upscale(g, side).to(DEV)
        ii, jj = np.triu_indices(len(g), 1)
        with torch.no_grad():
            vals.append(net(x[ii], x[jj]).mean().item())
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------- 平铺性
def tile_seam_ratio(t):
    """接缝处邻格差 / 内部邻格差。1 ≈ 平铺接缝与内部一样连续。内部近乎平涂返回 nan。"""
    a = np.asarray(t, np.float64)
    seam = (np.abs(a[:, -1] - a[:, 0]).mean() + np.abs(a[-1, :] - a[0, :]).mean()) / 2
    inner = (np.abs(np.diff(a, axis=1)).mean() + np.abs(np.diff(a, axis=0)).mean()) / 2
    return float(seam / inner) if inner > 1e-6 else float("nan")


def tileability(tiles):
    r = np.array([tile_seam_ratio(t) for t in tiles])
    r = r[np.isfinite(r)]
    return float(np.median(r)) if len(r) else float("nan")


# ---------------------------------------------------------------- 一站式
def evaluate(gen_tiles, gen_materials, ref_tiles, groups=None, ref_cache=None):
    """gen_tiles 与 gen_materials 一一对应。返回 dict。ref_cache 可复用参照集特征。"""
    rc = ref_cache if ref_cache is not None else {}
    if "inc" not in rc:
        rc["inc"] = inception_feats(ref_tiles)
        rc["dino"] = dino_feats(ref_tiles)
    gi, gd = inception_feats(gen_tiles), dino_feats(gen_tiles)
    k_mean, k_std = kid(gi, rc["inc"])
    ie = clip_image_emb(gen_tiles)
    te = clip_text_emb([prompt_of(m) for m in gen_materials])
    cs = (100 * (ie * te).sum(-1).clamp(min=0)).mean().item()
    out = {"n": len(gen_tiles), "KID_x1e3": 1e3 * k_mean, "KID_std_x1e3": 1e3 * k_std,
           "FID": frechet(gi, rc["inc"]), "FD_DINOv2": frechet(gd, rc["dino"]),
           "CLIP": cs, "tile_seam_ratio": tileability(gen_tiles)}
    if groups:
        out["LPIPS_div"] = lpips_diversity(groups)
    return out, rc


# ---------------------------------------------------------------- 数据
def artist_tiles(split, size=16):
    from tiles_data import load
    s = load(size=size, split=split)
    return [x["palette"][x["idx"]] for x in s], [" ".join(x["words"]) for x in s]


def selftest():
    """指标方向自检：天花板（真人 val）必须优于各种地板。任何一项方向反了就退 1。"""
    ref, _ = artist_tiles("test")
    val, val_m = artist_tiles("val")
    rng = np.random.default_rng(0)
    shuf = [t.reshape(-1, 3)[rng.permutation(t.shape[0] * t.shape[1])].reshape(t.shape) for t in val]
    noise = [rng.integers(0, 256, t.shape).astype(np.uint8) for t in val]
    flat = [np.broadcast_to(t.reshape(-1, 3).mean(0).astype(np.uint8), t.shape).copy() for t in val]
    print(f"参照 test {len(ref)} 张；val {len(val)} 张")
    res, cache = {}, None
    for name, tiles in (("ceiling_val", val), ("shuffled", shuf), ("flat", flat), ("noise", noise)):
        r, cache = evaluate(tiles, val_m, ref, ref_cache=cache)
        res[name] = r
        print(f"  {name:<12} " + "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "n"))
    bad = []
    for metric, better in (("KID_x1e3", "low"), ("FID", "low"), ("FD_DINOv2", "low")):
        for floor in ("shuffled", "flat", "noise"):
            a, b = res["ceiling_val"][metric], res[floor][metric]
            if not (a < b):
                bad.append(f"{metric}: ceiling {a:.3f} !< {floor} {b:.3f}")
    # CLIP：真人图配正确材质名应高于配错位材质名
    ie = clip_image_emb(val)
    te = clip_text_emb([prompt_of(m) for m in val_m])
    right = (ie * te).sum(-1).mean().item()
    wrong = (ie * te.roll(37, 0)).sum(-1).mean().item()
    print(f"  CLIP cos: 正确材质 {right:.4f}  vs 错位材质 {wrong:.4f}")
    if not right > wrong:
        bad.append("CLIP: 正确材质不高于错位材质")
    # LPIPS：同图为 0，不同图 > 0
    net = lpips_net()
    x = upscale(val[:32], 64).to(DEV)
    with torch.no_grad():
        same, diff = net(x, x).abs().max().item(), net(x, x.roll(1, 0)).mean().item()
    print(f"  LPIPS 同图 max {same:.2e}  不同图均值 {diff:.3f}")
    if not (same < 1e-4 and diff > 0.05):
        bad.append("LPIPS: 同图不为 0 或不同图过小")
    print(f"  平铺性（真人 val 中位）{tileability(val):.3f}   噪声 {tileability(noise):.3f}")
    json.dump(res, open(ROOT / "experiments/metrics_selftest.json", "w"), indent=1)
    print("\n" + ("自检通过：所有指标方向正确" if not bad else "自检失败：\n  " + "\n  ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest"])
    a = ap.parse_args()
    raise SystemExit(selftest())
