"""任务本身的评测：区域颜色 + 材质名 → 纹理（`GOAL.md` 的任务定义）。

主表（`run_eval.py`）只看"材质名 → 纹理"；这里把**区域颜色**也放进输入：
每张真人参照瓦片给出一个目标 = (它的材质名, 它的平均颜色)，每个方法对每个目标出一张瓦片。

- 颜色误差：生成瓦片平均色与目标色在 CIELAB 里的距离 ΔE76（越小越好）。
- 质量：FID / KID / FD-DINOv2 / CLIP，参照 = 同一批真人瓦片（此时颜色分布是对齐的，
  不跟颜色走的方法会在分布指标上被罚）。

方法的"跟随颜色"机制（写死，出结果前定）：
- 新架构 TRD：原生颜色条件（`gen_trd.py --colour_task`，出图目录 `experiments/colour/<tag>/<size>/`）。
- 基线没有颜色输入，只能事后改色，两种都报，取对它更有利的那种：
    `+recolor`  本仓库管线原有的 `recolor_to`（换色相/饱和、保留亮度结构）；
    `+labshift` 在 CIELAB 里把整张瓦片平移到目标平均色（亮度也对齐，ΔE 几乎为 0）。
  基线瓦片：B1 第 j%4 张、B2 唯一一张、B5 检索到的第 j 张真人瓦片。

用法（验证集调参；测试集 E_mat 只在最终配置定下后跑一次）：
    python eval/colour_task.py --set V_mat --methods B2+labshift B2+recolor B5+labshift TRD_xxx
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))
from prompts import is_material, load_set    # noqa: E402
from tiles_data import load                  # noqa: E402

# ---- sRGB <-> CIELAB（D65），不依赖 skimage ----
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def rgb_to_lab(rgb):
    c = np.asarray(rgb, float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = lin @ _M.T / _WHITE
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_rgb(lab):
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * _WHITE
    lin = np.clip(xyz @ np.linalg.inv(_M).T, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def mean_rgb(tile):
    return np.asarray(tile, float).reshape(-1, 3).mean(0)


def delta_e(tile, target_rgb):
    """瓦片平均色（先在 RGB 里取平均，与"区域颜色"的定义一致）与目标色的 ΔE76。"""
    return float(np.linalg.norm(rgb_to_lab(mean_rgb(tile)) - rgb_to_lab(target_rgb)))


def labshift(tile, target_rgb):
    """整张瓦片在 Lab 里平移；平均色是在 RGB 里取的（非线性），所以迭代几次把残差消掉。"""
    lab, goal = rgb_to_lab(tile), rgb_to_lab(target_rgb)
    x = np.asarray(tile)
    for _ in range(4):
        lab = lab + (goal - rgb_to_lab(mean_rgb(x)))
        x = lab_to_rgb(lab)
    return x


def recolor(tile, target_rgb):
    from make_texture import recolor_to
    t = np.asarray(tile)
    cols, inv = np.unique(t.reshape(-1, 3), axis=0, return_inverse=True)
    hexc = "%02x%02x%02x" % tuple(int(round(v)) for v in target_rgb)
    # recolor_to 会按亮度重排输出，所以逐色调用，保持原色与新色一一对应
    new = np.stack([np.asarray(recolor_to(c[None].astype(np.uint8), hexc))[0] for c in cols])
    return new[inv.reshape(-1)].reshape(t.shape).astype(np.uint8)


def targets(set_name, size=16):
    """每张真人参照瓦片一个目标，按材质分组编号 j。"""
    prompts, split = load_set(set_name)
    by_mat = {e["material"]: e for e in prompts}
    out, count = [], {}
    for s in load(size, split):
        if s["material"] not in by_mat or not is_material(s["material"]):
            continue
        e = by_mat[s["material"]]
        slug = e["material"].rsplit(".", 1)[0]
        j = count.get(slug, 0)
        count[slug] = j + 1
        tile = s["palette"][s["idx"]]
        out.append({"prompt": e["prompt"], "slug": slug, "j": j,
                    "rgb": mean_rgb(tile).tolist(), "ref": tile})
    return out


def baseline_tile(root, name, t, size, retr):
    if name == "B5":
        g = retr[t["prompt"]]
        return g[t["j"] % len(g)]
    d = root / name / str(size)
    p = d / f"{t['slug']}.png"
    if not p.exists():
        p = d / f"{t['slug']}_{t['j'] % 4}.png"
    return np.asarray(Image.open(p).convert("RGB")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="V_mat")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--methods", nargs="+", required=True,
                    help="基线写成 B1+labshift / B2+recolor / B5+labshift；TRD 写 experiments/colour/ 下的目录名，"
                         "可加 +labshift 做事后残差校正")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    from metrics import evaluate
    from run_eval import retrieval

    T = targets(a.set, a.size)
    ref = [t["ref"] for t in T]
    base_root = ROOT / ("experiments/baselines_val" if a.set.startswith("V_") else "experiments/baselines")
    prompts, _ = load_set(a.set)
    r_first, r_groups = retrieval(prompts, 4)
    retr = {e["prompt"]: g for e, g in zip(prompts, r_groups)}
    print(f"{a.set}: {len(T)} 个目标（{len({t['slug'] for t in T})} 个材质）", flush=True)

    rows, cache = {}, None
    for m in a.methods:
        name, _, post = m.partition("+")
        tiles, mats, des = [], [], []
        for t in T:
            if name in ("B1", "B2", "B4", "B5"):
                x = baseline_tile(base_root, name, t, a.size, retr)
            else:
                p = ROOT / "experiments/colour" / name / str(a.size) / f"{t['slug']}_{t['j']}.png"
                x = np.asarray(Image.open(p).convert("RGB")) if p.exists() else None
            if x is None:
                continue
            if post == "labshift":
                x = labshift(x, t["rgb"])
            elif post == "recolor":
                x = recolor(x, t["rgb"])
            tiles.append(x)
            mats.append(t["prompt"])
            des.append(delta_e(x, t["rgb"]))
        if len(tiles) < 0.9 * len(T):
            print(f"  {m}: 只有 {len(tiles)}/{len(T)} 张，跳过")
            continue
        r, cache = evaluate(tiles, mats, ref, ref_cache=cache)
        r["dE_mean"] = float(np.mean(des))
        r["dE_median"] = float(np.median(des))
        r["n"] = len(tiles)
        rows[m] = r
        print(f"  {m:<32} " + "  ".join(f"{k}={v:.3f}" for k, v in r.items() if k != "n"), flush=True)

    out = a.out or ROOT / f"experiments/colour_{a.set}_{a.size}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    cols = ["dE_mean", "KID_x1e3", "FID", "FD_DINOv2", "CLIP", "tile_seam_ratio", "n"]
    print("\n| 方法 | " + " | ".join(cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for m, r in rows.items():
        print(f"| {m} | " + " | ".join(f"{r.get(c, float('nan')):.3f}" if c != "n" else str(r.get(c))
                                        for c in cols) + " |")
    print(f"\n-> {out}")


def selftest():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, (500, 3))
    back = lab_to_rgb(rgb_to_lab(x))
    assert np.abs(back.astype(int) - x).max() <= 1, "Lab 往返误差"
    t = rng.integers(40, 200, (16, 16, 3)).astype(np.uint8)
    tgt = [120.0, 80.0, 40.0]
    assert delta_e(labshift(t, tgt), tgt) < 2.0, delta_e(labshift(t, tgt), tgt)
    r = recolor(t, tgt)
    assert r.shape == t.shape and len(np.unique(r.reshape(-1, 3), axis=0)) <= len(np.unique(t.reshape(-1, 3), axis=0))
    print("selftest ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
