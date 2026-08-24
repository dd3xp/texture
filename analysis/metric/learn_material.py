"""M2 候选指标（学习式表征）：16x16 图能不能被认出是什么材质？

前面两条路都失败了：CLIP 域外失效（真人原生像素画也只有 12.3%），
七个手工特征不编码材质身份（跨材质距离 < 跨作者距离）。
这里换成直接学一个：给定 16x16 像素画，分类它是哪种材质。

**按作者（材质包）留出测试集**，不是随机划分——
随机划分会让同一位作者的画风泄漏到测试集，测出来的是"记住了这个包"，
不是"材质身份可泛化"。

两种结果都有价值：
- 能学会 → 拿到了低分辨率材质表征，它本身就是尺子（M2 的产出）
- 学不会 → "材质身份在 16px 大量丢失"成为有实证的结论
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# 留出的测试包：覆盖度高，且画风各异
HELD_OUT = ["sofar__pixelperfection", "Liil__12345",
            "ROllerozxa__vilja_pix_2", "zayuim__isabellaii"]

# 对照开关：置 True 则把图转灰度并逐图标准化，抹掉调色板身份。
# 用来检验分类器学到的到底是材质结构还是配色签名。
GRAY = False


class SmallCNN(nn.Module):
    """16x16 输入的小卷积网。刻意做小——数据只有 ~3400 张，大模型必过拟合。"""

    def __init__(self, n_cls: int):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                   # 4x4
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, n_cls)

    def forward(self, x, feats=False):
        z = self.f(x).flatten(1)
        return z if feats else self.head(z)


def load_split(pairs: dict, size: int, min_artists: int):
    mats = [m for m, v in pairs.items() if len(v["low"]) >= min_artists]
    cls = {m: i for i, m in enumerate(sorted(mats))}
    tr, te = [], []
    for m in mats:
        for pack, path in pairs[m]["low"].items():
            try:
                im = Image.open(path).convert("RGB")
            except Exception:
                continue
            if im.size != (size, size):
                continue
            a = np.asarray(im, np.float32) / 255.0
            if GRAY:
                # 灰度 + 每图标准化：抹掉调色板身份，只留结构
                g = a @ np.array([0.299, 0.587, 0.114], np.float32)
                g = (g - g.mean()) / (g.std() + 1e-6)
                a = np.repeat(g[..., None], 3, -1)
            (te if pack in HELD_OUT else tr).append((a, cls[m]))
    return tr, te, cls


def to_tensor(ds):
    x = torch.tensor(np.stack([a for a, _ in ds])).permute(0, 3, 1, 2)
    y = torch.tensor([c for _, c in ds])
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/contentdb/pairs.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--min-artists", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("experiments/metric"))
    ap.add_argument("--gray", action="store_true", help="灰度对照：抹掉颜色只留结构")
    args = ap.parse_args()

    global GRAY
    GRAY = args.gray
    print("灰度对照（无颜色）" if GRAY else "彩色（原设置）")

    pairs = json.loads(args.pairs.read_text())
    tr, te, cls = load_split(pairs, args.size, args.min_artists)
    print(f"类别 {len(cls)}  训练 {len(tr)}  测试 {len(te)}（留出包 {HELD_OUT}）")
    print(f"随机基线 {1/len(cls):.2%}\n")

    xtr, ytr = to_tensor(tr)
    xte, yte = to_tensor(te)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xtr, ytr, xte, yte = xtr.to(dev), ytr.to(dev), xte.to(dev), yte.to(dev)

    torch.manual_seed(0)
    net = SmallCNN(len(cls)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 0.0
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            xb, yb = xtr[idx], ytr[idx]
            # 轻量增广：翻转 + 90 度旋转。材质是各向同性的，这不改变类别
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, [3])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                xb = torch.rot90(xb, k, [2, 3])
            opt.zero_grad()
            F.cross_entropy(net(xb), yb).backward()
            opt.step()
        sch.step()

        if (ep + 1) % 20 == 0 or ep == args.epochs - 1:
            net.eval()
            with torch.no_grad():
                lo = net(xte)
                top1 = (lo.argmax(1) == yte).float().mean().item()
                top5 = (lo.topk(5, 1).indices == yte[:, None]).any(1).float().mean().item()
                tr_acc = (net(xtr).argmax(1) == ytr).float().mean().item()
            best = max(best, top1)
            print(f"ep {ep+1:>4}  训练 {tr_acc:6.1%}  测试top1 {top1:6.1%}  测试top5 {top5:6.1%}")

    print(f"\n留出作者上的最佳 top-1: {best:.1%}   （随机 {1/len(cls):.2%}，"
          f"提升 {best*len(cls):.0f}x）")
    torch.save({"state": net.state_dict(), "classes": cls}, args.out / "material_cnn.pt")
    (args.out / "learn_material.json").write_text(json.dumps({
        "n_classes": len(cls), "n_train": len(tr), "n_test": len(te),
        "held_out_packs": HELD_OUT, "best_top1": best,
        "random_baseline": 1 / len(cls),
    }, indent=1))
    print(f"模型与结果写入 {args.out}")


if __name__ == "__main__":
    main()
