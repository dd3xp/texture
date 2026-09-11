"""结构诊断：结构型材质（砖、木板、圆石……）的生成样本到底有没有结构。

v1 的失败模式是"颜色对、结构无"（砖墙没有砖缝，只有斑点）。目视之外给一个客观量：
**结构触发率** = 样本中「检出主周期 且 各向异性 ≥ 0.20」的比例（本仓库的双条件门，
瓦片口径 lo=2、hi_frac=0.625，与 `analysis/premise/dither.py` 相同）。
参照 = 同名材质的真人 16px 瓦片（train+val）的触发率。斑点/噪声的触发率会远低于真人。

另报：验证集结构损失 / 调色板损失。

    python eval/diag_structure.py --run runs/trd_v2 --ckpt last.pt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
for sub in ("model", "eval", "tools"):
    sys.path.insert(0, str(ROOT / sub))
from trd import sample, training_loss                          # noqa: E402
from train_trd import to_tensors, decode, clip_text, TEXT_TMPL, model_from_args   # noqa: E402
from tiles_data import load                                    # noqa: E402
from downsample import dominant_period, anisotropy             # noqa: E402

STRUCT = ["stone brick", "brick", "wood planks", "oak planks", "cobble",
          "sandstone brick", "desert stone brick", "junglewood planks"]
KEYS = ["brick", "planks", "cobble"]           # 真人参照：名字含这些词的瓦片


def gated(t):
    rgb = np.asarray(t, float)
    return dominant_period(rgb, lo=2, hi_frac=0.625) > 0 and anisotropy(rgb) >= 0.20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--n", type=int, default=8, help="每个提示词的样本数")
    ap.add_argument("--k", type=int, default=0, help="色阶数；0 = 按训练集分布抽")
    ap.add_argument("--cfg", type=float, default=2.0)
    ap.add_argument("--choice_temp", type=float, default=4.5)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--pal_top_p", type=float, default=0.9)
    a = ap.parse_args()
    dev = "cuda"
    ck = torch.load(a.run / a.ckpt, map_location=dev)
    model = model_from_args(ck["args"], drop=0.0).to(dev).eval()
    model.load_state_dict(ck["model"])
    cb = np.load(a.run / "codebook.npy")
    te = torch.load(a.run / "text_emb.pt")
    tindex = {m: i for i, m in enumerate(te["materials"])}

    # 验证损失（拆开）
    val = load(16, "val")
    V = to_tensors(val, cb, int(ck["args"]["codes"]), tindex, te["emb"])
    tg = tp = 0.0
    torch.manual_seed(123)
    with torch.no_grad():
        for i in range(0, len(val), 256):
            idx = torch.arange(i, min(i + 256, len(val)))
            _, p = training_loss(model, V["pal"][idx].to(dev), V["grid"][idx].to(dev),
                                 V["k"][idx].to(dev), V["text"][idx].to(dev),
                                 torch.zeros(len(idx), 4, device=dev))
            tg += p["loss_grid"] * len(idx)
            tp += p["loss_pal"] * len(idx)

    # 真人参照
    ref = [s for s in load(16, "train") + val if any(k in s["material"] for k in KEYS)]
    ref_rate = float(np.mean([gated(s["palette"][s["idx"]]) for s in ref]))
    kdist = np.bincount([s["k_used"] for s in ref], minlength=17).astype(float)
    kdist /= kdist.sum()

    temb = clip_text([TEXT_TMPL.format(p=p) for p in STRUCT], dev).to(dev)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    rows, fires = [], []
    for rep in range(a.n):
        ks = torch.tensor([a.k] * len(STRUCT) if a.k else rng.choice(17, len(STRUCT), p=kdist), device=dev)
        pal, grid = sample(model, temb, ks, n=16, cfg=a.cfg, choice_temp=a.choice_temp, steps=a.steps,
                           pal_top_p=a.pal_top_p)
        imgs = decode(pal, grid, cb)
        fires += [gated(t) for t in imgs]
        rows.append(np.concatenate(list(imgs), 1))
    sheet = np.concatenate(rows, 0)
    tag = f"{a.run.name}_{a.ckpt[:-3]}_ct{a.choice_temp}_s{a.steps}_p{a.pal_top_p}"
    out = ROOT / f"experiments/diag_struct_{tag}.png"
    Image.fromarray(sheet).resize((sheet.shape[1] * 6, sheet.shape[0] * 6), Image.NEAREST).save(out)
    res = {"run": str(a.run), "ckpt": a.ckpt, "step": ck["step"], "val_grid": tg / len(val),
           "val_pal": tp / len(val), "struct_rate": float(np.mean(fires)),
           "artist_struct_rate": ref_rate, "n_samples": len(fires), "n_artist": len(ref),
           "choice_temp": a.choice_temp, "steps": a.steps, "cfg": a.cfg, "pal_top_p": a.pal_top_p}
    print(json.dumps(res, ensure_ascii=False))
    print("列:", STRUCT, "->", out)
    with open(ROOT / "experiments/diag_struct.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
