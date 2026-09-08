"""把一批纹理拼成紧凑网格总览，并**如实数出有多少能用**。

`pack_sheet.py` 是每材质一行、三档并排，材质一多就成了长条。
这里一档一张网格，适合几十个材质一眼扫完。

「能用」的判据不是我看着顺眼，而是沿用已确立的量：调色板亮度跨度。
交付批次上量过——门触发组中位 0.262、拒绝组 0.071、真人 0.292
（精确置换 p=0.0018）。低于 0.134（触发组的最小值）的基本是平涂，
不该往模组里放。这个阈值是**从旧批次定的，不是照这批调的**。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from make_texture import W                                  # noqa: E402

FLAT_MAX = 0.134     # 旧批次里门触发组的最小跨度；低于此视为接近平涂


def realized_spread(img: np.ndarray) -> float:
    u = np.unique(img.reshape(-1, 3), axis=0).astype(float)
    lum = np.sort(u @ W)
    return float(lum[-1] - lum[0]) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=ROOT / "experiments/pack60")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--upscale", type=int, default=6)
    ap.add_argument("--out", type=Path, default=ROOT / "figures/pack60_grid.png")
    a = ap.parse_args()

    man = json.loads((a.pack / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in man if r["variant"] == a.variant and r["size"] == a.size]
    rows.sort(key=lambda r: r["material"])
    d = a.pack / a.variant / str(a.size)

    cell = a.size * a.upscale
    pad, lab = 3, 13
    cols = a.cols
    nrow = (len(rows) + cols - 1) // cols
    Wpx = cols * (cell + 2 * pad)
    Hpx = nrow * (cell + 2 * pad + lab)
    canvas = Image.new("RGB", (Wpx, Hpx), "white")
    draw = ImageDraw.Draw(canvas)

    flat = fired = 0
    for i, r in enumerate(rows):
        f = d / (r["material"].replace(" ", "_") + ".png")
        t = np.asarray(Image.open(f).convert("RGB"))
        sp = realized_spread(t)
        r["spread"] = sp
        flat += sp < FLAT_MAX
        fired += bool(r["gate_fired"])
        cx = (i % cols) * (cell + 2 * pad) + pad
        cy = (i // cols) * (cell + 2 * pad + lab) + pad
        canvas.paste(Image.fromarray(t).resize((cell, cell), Image.NEAREST), (cx, cy))
        # **两件事分开标**：门未触发 != 不能用。
        # 红框 = 接近平涂，别往模组里放；左上角灰角标 = 门未触发（只是没裁）。
        if sp < FLAT_MAX:
            draw.rectangle([cx - 1, cy - 1, cx + cell, cy + cell], outline="#d02020")
        if not r["gate_fired"]:
            draw.line([cx, cy, cx + 7, cy], fill="#808080", width=2)
            draw.line([cx, cy, cx, cy + 7], fill="#808080", width=2)
        name = r["material"]
        draw.text((cx, cy + cell + 2), name[:cell // 5], fill="black")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(a.out)
    n = len(rows)
    print(f"{n} 个材质 @ {a.size}px")
    print(f"  门触发            {fired}/{n} = {fired/n:.0%}")
    print(f"  接近平涂（跨度<{FLAT_MAX}） {flat}/{n} = {flat/n:.0%}  <- 不建议使用")
    print(f"  **可用**          {n-flat}/{n} = {(n-flat)/n:.0%}")
    med_f = np.median([r["spread"] for r in rows if r["gate_fired"]])
    med_d = np.median([r["spread"] for r in rows if not r["gate_fired"]])
    print(f"  跨度中位：触发 {med_f:.3f}  未触发 {med_d:.3f}  （真人 0.292）")
    # 门拒绝 != 不能用：把这两类交叉数出来，否则会低估交付物
    dec = [r for r in rows if not r["gate_fired"]]
    dec_ok = sum(r["spread"] >= FLAT_MAX for r in dec)
    print(f"  门未触发但**仍可用** {dec_ok}/{len(dec)}"
          f" —— 门拒绝只是不裁，不等于不能用")
    print(f"写入 {a.out}（红框=接近平涂勿用；左上灰角=门未触发）")


if __name__ == "__main__":
    main()
