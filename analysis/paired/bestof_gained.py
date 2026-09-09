"""判据③：best-of-N **新救回来的**那些材质，成品是否真的更好？

`bestof_units.py` 的两条判据留了一格没测：
  ① 有效裁剪率 52% -> 90%（新增 16、失去 0，p=3.05e-5）——证明了「能处理」；
  ② 在**两者本来都能处理**的材质上挑样本，3/6=50%（n 太小，只能说没测到大效应）。
但**新增的那 16 个**没被评过：它们原来单样本根本裁不了（直接降采样），
现在 best-of-4 找到了能裁的那一版。**这才是端到端的收益所在。**

§5.3 已经证明「裁比不裁好」（三口径 72–89%），所以这一条是可预期的；
但那是在**门本来就触发**的材质上测的，这 16 个恰恰是门原本拒绝的那批，
不能直接外推——所以要单独测。

--- 判据（跑之前写下并 commit）---
只取 `single_eff=False 且 best_eff=True` 的材质（新救回来的那批）。
主判据：经验证判官（claude-opus-5，正反去偏、不一致弃用）偏好 best 版
        >50% 且二项 p<0.05 -> 多采样带来的适用率提升是**真收益**，不只是「能跑」。
证伪：≤50% 或不显著 -> 多采样只是让门触发，成品并没更好；
        那么方法一的价值仅限于「覆盖更多材质」，须如实这样写。
方向性限定：判官压缩效应，为正作下界；为负只能说「没测到大效应」。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sub in ("analysis", "analysis/annotate"):
    sys.path.insert(0, str(ROOT / sub))
from exact import binom_test, jeffreys                          # noqa: E402

Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "experiments/bestof_units.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/bestof")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/bestof_gained.json")
    a = ap.parse_args()

    recs = json.loads(a.data.read_text(encoding="utf-8"))
    gained = [r for r in recs if r["best_eff"] and not r["single_eff"]]
    print(f"新救回来的材质 {len(gained)} 个（原来无有效裁剪，现在有）")
    if not gained:
        raise SystemExit("没有可评的材质")

    from vlm_judge import ask
    import base64, io

    def b64(p):
        arr = np.asarray(Image.open(p).convert("RGB"))
        buf = io.BytesIO()
        Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    win = tot = inc = 0
    out = []
    for r in gained:
        slug = r["prompt"].replace(" ", "_")
        S = b64(a.tiles / f"{slug}_single.png")
        B = b64(a.tiles / f"{slug}_best.png")
        q = Q.format(n=a.size, label=r["prompt"])
        o1, o2 = ask(a.model, q, [S, B], base_url, key), ask(a.model, q, [B, S], base_url, key)
        if not o1 or not o2:
            continue
        p1 = "single" if o1.upper().startswith("A") else "best"
        p2 = "best" if o2.upper().startswith("A") else "single"
        pick = p1 if p1 == p2 else "inconsistent"
        if pick == "inconsistent":
            inc += 1
        else:
            tot += 1; win += pick == "best"
        out.append({"prompt": r["prompt"], "single_frac": r["single_frac"],
                    "best_frac": r["best_frac"], "vlm": pick})
        print(f"  {r['prompt']:<32} -> {pick}", flush=True)

    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    if not tot:
        print("无有效判断"); return
    pv = binom_test(win, tot); lo, hi = jeffreys(win, tot)
    print(f"\n判据③：best 被选 {win}/{tot} = {win/tot:.0%}  p={pv:.3g}"
          f"  [{lo:.0%},{hi:.0%}]   （正反不一致弃 {inc}）")
    if win / tot > 0.5 and pv < 0.05:
        print("  -> **成立**：多采样带来的适用率提升是真收益（判官压缩，故为下界）")
    else:
        print("  -> 不成立：只能说没测到大效应；方法一的价值须限定为「覆盖更多材质」")


if __name__ == "__main__":
    main()
