"""把若干 SD-piXL run 的中间/最终结果拼成一张对照图。

每行一个 run，列为指定的若干 step。所有 cell 用最近邻放大到同一尺寸，
这样像素格子是可见的——判读像素画结果时这一点是必须的，
双线性放大会把抖动图案糊掉，正好掩盖我们要看的东西。

用法:
    python montage.py <run_root> <out.png> --steps 0 1950 --label-runs p1 p2 p3 p4
"""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

CELL = 256          # 每个 cell 的边长
PAD = 8
LABEL_W = 200       # 左侧行标签宽度
HEADER_H = 28


def latest_step(png_logs: Path) -> int:
    steps = [
        int(m.group(1))
        for f in png_logs.iterdir()
        if (m := re.fullmatch(r"(\d+)_hard\.png", f.name))
    ]
    return max(steps) if steps else -1


def load_cell(png_logs: Path, step: int) -> Image.Image | None:
    f = png_logs / f"{step}_hard.png"
    if not f.exists():
        return None
    # 最近邻，保住像素边界
    return Image.open(f).convert("RGB").resize((CELL, CELL), Image.NEAREST)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, nargs="+", required=True,
                    help="要对照的 step；-1 表示各 run 自己的最新一步")
    args = ap.parse_args()

    runs = sorted(d for d in args.run_root.iterdir()
                  if d.is_dir() and (d / "png_logs").is_dir()
                  and latest_step(d / "png_logs") >= 0)
    if not runs:
        raise SystemExit(f"no runs with png_logs under {args.run_root}")

    ncol = len(args.steps)
    W = LABEL_W + ncol * (CELL + PAD) + PAD
    H = HEADER_H + len(runs) * (CELL + PAD) + PAD
    canvas = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    for ci, step in enumerate(args.steps):
        x = LABEL_W + ci * (CELL + PAD)
        draw.text((x + 4, 8), f"step {step}" if step >= 0 else "latest", fill="#000000")

    for ri, run in enumerate(runs):
        png_logs = run / "png_logs"
        y = HEADER_H + ri * (CELL + PAD)

        # 行标签：从目录名里抠出分辨率
        m = re.search(r"im(\d+)x(\d+)", run.name)
        label = f"{m.group(1)}x{m.group(2)}" if m else run.name[:20]
        draw.text((6, y + CELL // 2), label, fill="#000000")

        for ci, step in enumerate(args.steps):
            s = latest_step(png_logs) if step < 0 else step
            cell = load_cell(png_logs, s)
            x = LABEL_W + ci * (CELL + PAD)
            if cell is None:
                draw.rectangle([x, y, x + CELL, y + CELL], fill="#eeeeee")
                draw.text((x + 8, y + 8), f"missing {s}", fill="#999999")
            else:
                canvas.paste(cell, (x, y))
                draw.rectangle([x, y, x + CELL, y + CELL], outline="#cccccc")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}  ({W}x{H}, {len(runs)} runs x {ncol} steps)")
    for r in runs:
        print(f"  {r.name}  latest_step={latest_step(r / 'png_logs')}")


if __name__ == "__main__":
    main()
