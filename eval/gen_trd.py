"""用 TRD 检查点给评测提示词集出图（与基线同一批提示词、同样每材质 n 张）。

色阶数 k：从训练集 k_used 的经验分布里抽（v1 的做法，写在这里以免事后改）；
颜色条件丢弃（与基线同等条件）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from trd import TRD, sample                      # noqa: E402
from train_trd import decode, clip_text, TEXT_TMPL, model_from_args   # noqa: E402
from tiles_data import load                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=32, help="采样批次（CFG 要两次前向，共享卡上别开太大）")
    ap.add_argument("--pal_top_p", type=float, default=0.9)
    ap.add_argument("--choice_temp", type=float, default=4.5)
    ap.add_argument("--tag", default=None, help="输出目录名（默认由参数拼出）")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    ck = torch.load(a.run / a.ckpt, map_location=dev)
    args = ck["args"]
    cb = np.load(a.run / "codebook.npy")
    model = model_from_args(args, drop=0.0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    from prompts import load_set
    prompts, _ = load_set(a.set)
    temb = clip_text([TEXT_TMPL.format(p=e["prompt"]) for e in prompts], dev).to(dev)
    kdist = np.bincount([s["k_used"] for s in load(16, "train")], minlength=17).astype(float)
    kdist /= kdist.sum()
    rng = np.random.default_rng(a.seed)

    tag = a.tag or f"TRD_{a.run.name}_cfg{a.cfg}_t{a.temp}_p{a.pal_top_p}_{a.set}"
    out = (a.out or ROOT / "experiments/baselines") / tag / str(a.size)
    out.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for kk in range(a.n):
        ks = torch.tensor(rng.choice(17, len(prompts), p=kdist), device=dev)
        for i in range(0, len(prompts), a.bs):
            sl = slice(i, i + a.bs)
            pal, grid = sample(model, temb[sl], ks[sl], n=a.size, steps=a.steps,
                               cfg=a.cfg, temp=a.temp, pal_top_p=a.pal_top_p,
                               choice_temp=a.choice_temp)
            imgs = decode(pal, grid, cb)
            for j, e in enumerate(prompts[sl]):
                slug = e["material"].rsplit(".", 1)[0]
                Image.fromarray(imgs[j]).save(out / f"{slug}_{kk}.png")
        print(f"sample {kk + 1}/{a.n} done", flush=True)
    print("->", out.parent)


if __name__ == "__main__":
    main()
