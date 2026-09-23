"""命题（循环平移等变）的数值核验：对训练好的检查点，把输入网格循环平移 (dy,dx)，
检查网格 logits 恰好随之平移、调色板 logits 不变。零训练、零判官。

用法：python analysis/arch/equiv_check.py RUN[:CKPT] [RUN[:CKPT] ...] --out /tmp/equiv.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from trd import K_MAX                             # noqa: E402
from train_trd import model_from_args             # noqa: E402


def load(spec, dev):
    run, _, ck = spec.partition(":")
    run = Path(run)
    if not ck:
        ck = "last.pt" if (run / "last.pt").exists() else sorted(run.glob("step_*.pt"))[-1].name
    d = torch.load(run / ck, map_location=dev)
    m = model_from_args(d["args"], drop=0.0).to(dev)
    m.load_state_dict(d["model"])
    return m.eval(), d["args"], ck


@torch.no_grad()
def check(m, n, dev, B=16, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    k = torch.randint(3, K_MAX + 1, (B,), generator=g)
    grid = (torch.rand(B, n, n, generator=g) * k[:, None, None]).long()
    grid = torch.where(torch.rand(B, n, n, generator=g) < 0.5, torch.full_like(grid, m.GRID_MASK), grid)
    pal = torch.randint(0, m.n_codes, (B, K_MAX), generator=g)
    pal = torch.where(torch.rand(B, K_MAX, generator=g) < 0.5, torch.full_like(pal, m.PAL_MASK), pal)
    text = torch.nn.functional.normalize(torch.randn(B, m.text_proj[0].in_features, generator=g), dim=-1)
    color = torch.cat([torch.rand(B, 3, generator=g), torch.ones(B, 1)], 1)
    ex = torch.randint(0, 16, (B, m.n_ex, 16, 16), generator=g) if m.n_ex else None
    t = lambda x: None if x is None else x.to(dev)
    pal, grid, k, text, color, ex = map(t, (pal, grid, k, text, color, ex))
    lp0, lg0 = m(pal, grid, k, text, color, ex=ex)
    scale = lg0.float().std().item()
    res = []
    for dy, dx in [(1, 0), (0, 1), (3, 5), (n // 2, n // 2), (n - 1, 2)]:
        lp1, lg1 = m(pal, torch.roll(grid, (dy, dx), (1, 2)), k, text, color, ex=ex)
        eg = (lg1 - torch.roll(lg0, (dy, dx), (1, 2))).abs().max().item()
        ep = (lp1 - lp0).abs().max().item()
        res.append({"shift": [dy, dx], "grid_maxabs": eg, "pal_maxabs": ep})
    return {"logit_std": scale,
            "grid_maxabs": max(r["grid_maxabs"] for r in res),
            "pal_maxabs": max(r["pal_maxabs"] for r in res),
            "per_shift": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--out", default="/tmp/equiv.json")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = {}
    for spec in a.runs:
        m, args, ck = load(spec, dev)
        wrap = getattr(args, "bias_wrap", None) if not isinstance(args, dict) else args.get("bias_wrap")
        off = getattr(args, "bias_off", None) if not isinstance(args, dict) else args.get("bias_off")
        key = f"{Path(spec.split(':')[0]).name}:{ck}"
        out[key] = {"bias_wrap": wrap or "torus", "bias_off": bool(off),
                    **{str(n): check(m, n, dev) for n in a.sizes}}
        for n in a.sizes:
            r = out[key][str(n)]
            print(f"{key:40s} wrap={out[key]['bias_wrap']:7s} n={n:2d}  grid max|Δ|={r['grid_maxabs']:.2e}  "
                  f"pal max|Δ|={r['pal_maxabs']:.2e}  logit std={r['logit_std']:.2f}")
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
