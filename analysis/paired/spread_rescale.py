"""门拒绝的材质：把调色板亮度跨度拉到真人惯例，能不能救回来？

`isotropic_scale.py` 关掉了「按特征尺度裁剪」那条路：这些材质没有可对齐的
空间单元。它同时指出剩下的方向是**匹配颜色分布**而不是空间尺度。
这里检验其中最直接的一条：亮度跨度。

已测得（`experiments/pack`，18 材质）：门拒绝组的调色板亮度跨度中位 0.071，
触发组 0.262，真人 0.292（`SPREAD_ARTIST_MEDIAN`）。差得很远。

--- 判据（跑之前写下并 commit）---
操作检验：重标后亮度跨度必须落进 0.292±20%，否则操作无效，后两条不算数。
主判据：经验证的判官（claude-opus-5，正反两问去偏、不一致弃用）
        偏好重标版 > 50%，且二项 p < 0.05 -> 亮度跨度是缺的那一环。
证伪：胜率 ≤ 50% 或不显著 -> 不是。
方向性限定：该判官**压缩**效应，故
  - 若为正：可作为下界报告；
  - 若为负：**只能说没测到大效应**，不能声称「无效」（判官不能确认零假设）。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis/annotate"))
from make_texture import extract_palette, quantize, W, SPREAD_ARTIST_MEDIAN  # noqa: E402
from exact import binom_test, jeffreys                                       # noqa: E402

Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")


def spread(pal: np.ndarray) -> float:
    lum = np.sort(pal.astype(float) @ W)
    return float(lum[-1] - lum[0]) / 255.0


def rescale(tile: np.ndarray, colors: int, target: float) -> tuple[np.ndarray, float, float]:
    """把调色板的亮度跨度线性拉到 target，色相饱和不动。"""
    pal = extract_palette(tile, colors, seed=0).astype(float)
    before = spread(pal)
    if before <= 1e-6:
        return tile, before, before
    lum = pal @ W
    mid = (lum.max() + lum.min()) / 2
    k = (target * 255.0) / (lum.max() - lum.min())
    new_lum = np.clip(mid + (lum - mid) * k, 0, 255)
    scale = np.divide(new_lum, np.maximum(lum, 1e-6))[:, None]
    newpal = np.clip(pal * scale, 0, 255)
    return quantize(tile, newpal).astype(np.uint8), before, spread(newpal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, default=ROOT / "experiments/pack")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--target", type=float, default=SPREAD_ARTIST_MEDIAN)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true", help="只做操作检验，不调判官")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/spread_rescale.json")
    a = ap.parse_args()

    man = json.loads((a.pack / "manifest.json").read_text(encoding="utf-8"))
    rows = [r for r in man if r["variant"] == a.variant and not r["gate_fired"]]
    print(f"门拒绝的记录 {len(rows)} 条（{len({r['material'] for r in rows})} 个材质）")

    recs = []
    for r in rows:
        f = a.pack / a.variant / str(r["size"]) / (r["material"].replace(" ", "_") + ".png")
        tile = np.asarray(Image.open(f).convert("RGB")).astype(np.uint8)
        new, s0, s1 = rescale(tile, a.colors, a.target)
        recs.append({"material": r["material"], "size": r["size"],
                     "spread_before": s0, "spread_after": s1,
                     "before": tile, "after": new})
        print(f"  {r['material']:<22} {r['size']:>3}px  跨度 {s0:.3f} -> {s1:.3f}")

    # —— 操作检验 ——
    got = [x["spread_after"] for x in recs]
    lo, hi = a.target * 0.8, a.target * 1.2
    ok = sum(lo <= g <= hi for g in got)
    print(f"\n操作检验：{ok}/{len(got)} 落进 {lo:.3f}–{hi:.3f}"
          f"（目标 {a.target}±20%）")
    if ok < 0.8 * len(got):
        print("  -> **操作无效**，主判据不予评估（判据已在跑前固定）")
        return
    print("  -> 操作有效，继续主判据")

    if a.no_judge:
        return

    from vlm_judge import upscale_b64, ask
    import base64, io

    def b64(arr):
        buf = io.BytesIO()
        Image.fromarray(arr.astype(np.uint8)).resize((256, 256), Image.NEAREST).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    win = tot = incons = 0
    out = []
    for x in recs:
        q = Q.format(n=x["size"], label=x["material"])
        B, A = b64(x["before"]), b64(x["after"])
        o1, o2 = ask(a.model, q, [B, A], base_url, key), ask(a.model, q, [A, B], base_url, key)
        if not o1 or not o2:
            continue
        p1 = "before" if o1.upper().startswith("A") else "after"
        p2 = "after" if o2.upper().startswith("A") else "before"
        if p1 != p2:
            incons += 1
            pick = "inconsistent"
        else:
            tot += 1
            win += p1 == "after"
            pick = p1
        out.append({k: v for k, v in x.items() if k not in ("before", "after")} | {"vlm": pick})
        print(f"  [{len(out)}/{len(recs)}] {x['material']:<22} {x['size']:>3}px -> {pick}",
              flush=True)

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if not tot:
        print("没有有效判断"); return
    p = binom_test(win, tot)
    jl, jh = jeffreys(win, tot)
    print(f"\n重标版被选 {win}/{tot} = {win/tot:.0%}  p={p:.3g}  [{jl:.0%},{jh:.0%}]"
          f"   （正反不一致弃 {incons}）")
    if win / tot > 0.5 and p < 0.05:
        print("判读：主判据成立 -> 亮度跨度是缺的那一环（判官压缩，故为**下界**）")
    else:
        print("判读：主判据不成立 -> **只能说没测到大效应**；")
        print("      该判官压缩效应，不能据此声称重标无效（判官不能确认零假设）")


if __name__ == "__main__":
    main()
