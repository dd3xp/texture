"""基线：高分辨率纹理 → 面积降采样 → 量化到调色板 → 硬裁进给定区域。

这个基线是 S1 的直接产物。S1 证明了真人的低分辨率纹理与
「降采样+量化」在统计和视觉上都难以区分，所以它不再只是对照组，
而是本项目**必须打败的对手**。

它相对 SD-πXL 的结构性优势：区域是**硬裁**的，所以轮廓天然完全保持。
SD-πXL 做不到这一点不是调参问题，是架构里没有区域语义通道
（见 related-work.md）。这里把那个差距量化出来。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_palette(path: Path) -> np.ndarray:
    cols = []
    for line in path.read_text().split():
        line = line.strip().lstrip("#")
        if len(line) == 6:
            cols.append([int(line[i:i + 2], 16) for i in (0, 2, 4)])
    return np.array(cols, dtype=np.float64)


def quantize_to(a: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """每个像素取调色板里最近的颜色（欧氏距离）。"""
    flat = a.reshape(-1, 1, 3)
    d = ((flat - palette.reshape(1, -1, 3)) ** 2).sum(-1)
    return palette[d.argmin(1)].reshape(a.shape)


def region_mask(img: Image.Image, bg: tuple[int, int, int], size: int) -> np.ndarray:
    """把区域图降到目标尺寸，非背景为 True。

    用最近邻——区域边界是硬的，任何插值都会在边上造出不属于调色板的中间色。
    """
    small = np.asarray(img.convert("RGB").resize((size, size), Image.NEAREST))
    return (small != np.array(bg)).any(-1)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True, help="区域图（纯色形状 + 背景）")
    ap.add_argument("--texture", type=Path, required=True, help="高分辨率纹理源")
    ap.add_argument("--palette", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--bg", type=int, nargs=3, default=[255, 255, 255])
    ap.add_argument("--out", type=Path, default=Path("experiments/dq_baseline"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    palette = read_palette(args.palette)
    bg = tuple(args.bg)
    # 背景色必须在调色板里，否则输出会含调色板外的颜色
    bg_in_palette = bool((palette == np.array(bg)).all(-1).any())
    fg_palette = palette[~(palette == np.array(bg)).all(-1)] if bg_in_palette else palette

    src = Image.open(args.input)
    tex = Image.open(args.texture).convert("RGB")

    results = []
    for n in args.sizes:
        mask = region_mask(src, bg, n)
        # 面积平均降采样纹理，再量化到前景调色板
        t = np.asarray(tex.resize((n, n), Image.BOX), dtype=np.float64)
        t = quantize_to(t, fg_palette)

        out = np.where(mask[..., None], t, np.array(bg, dtype=np.float64))
        out_mask = (out != np.array(bg)).any(-1)

        # 输出用色是否全在调色板内
        used = np.unique(out.reshape(-1, 3), axis=0)
        in_pal = all(bool((palette == c).all(-1).any()) for c in used)

        results.append({
            "size": n,
            "iou": round(iou(mask, out_mask), 4),
            "n_colors": len(used),
            "palette_compliant": in_pal,
        })
        Image.fromarray(out.astype(np.uint8)).save(args.out / f"dq_{n}.png")
        Image.fromarray(out.astype(np.uint8)).resize((512, 512), Image.NEAREST).save(
            args.out / f"dq_{n}_x512.png")

    print(f"{'尺寸':<8}{'轮廓 IoU':>10}{'用色数':>8}{'调色板合规':>12}")
    print("-" * 40)
    for r in results:
        print(f"{str(r['size'])+'x'+str(r['size']):<8}{r['iou']:>10.4f}"
              f"{r['n_colors']:>8}{'是' if r['palette_compliant'] else '否':>12}")

    (args.out / "metrics.json").write_text(json.dumps(results, indent=1))
    print(f"\n结果写入 {args.out}")


if __name__ == "__main__":
    main()
