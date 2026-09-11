"""区域颜色任务的判官输入：每材质第 0 个目标（该材质第一张真人参照瓦片的平均色），各方法都对齐到同一目标色。

    python eval/build_colour_dirs.py --set E_mat --methods TRD16 B7 B2
输出 `experiments/baselines/C_<方法>_<set>/16/<slug>_0.png`：
- 基线 B1/B2/B4 取其瓦片（B1 第 0 张、B2 唯一一张）后 labshift；
- 其余（TRD、B7）取 `experiments/colour/<方法>/16/<slug>_0.png`（原生颜色条件）后 labshift 残差校正。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from colour_task import targets, labshift     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--methods", nargs="+", required=True)
    a = ap.parse_args()
    base = ROOT / ("experiments/baselines_val" if a.set.startswith("V_") else "experiments/baselines")
    for m in a.methods:
        d = ROOT / f"experiments/baselines/C_{m}_{a.set}/16"
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        for t in targets(a.set, 16):
            if t["j"] != 0:
                continue
            if m in ("B1", "B2", "B4"):
                src = next((p for p in (base / m / "16" / f"{t['slug']}.png", base / m / "16" / f"{t['slug']}_0.png")
                            if p.exists()), None)
            else:
                src = ROOT / f"experiments/colour/{m}/16/{t['slug']}_0.png"
            if src is None or not Path(src).exists():
                continue
            tile = np.asarray(Image.open(src).convert("RGB"))
            Image.fromarray(np.asarray(labshift(tile, t["rgb"]), np.uint8)).save(d / f"{t['slug']}_0.png")
            n += 1
        print(f"C_{m}_{a.set}: {n} 个材质 -> {d}")


if __name__ == "__main__":
    main()
