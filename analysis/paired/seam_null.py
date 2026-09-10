"""方法三的操作检验 (a) 用错了零假设 —— 本脚本去量正确的那个。

**问题**。`seam_crop.py` 的操作检验 (a) 是：出厂瓦片的接缝比
`ratio_seam < ratio_center`，逐材质符号检验。两批各 22/22、35/35 全降，
报出 p=4.8e-7、p=5.8e-11（`tools/downsample.py:210,218`，README 与交付页同步）。

但 `ratio_seam` 是**在约 3930 个可行候选位置上取的 argmin**（中位数，见 JSON 的
`n_admissible`），`ratio_center` 是**一个固定点**的值。「几千个里挑最小的」
几乎必然小于「其中某一个固定点」——符号检验那个 p=0.5 的零假设**根本不成立**，
于是 22/22 这个结果里有多少是效应、有多少是 min 选择的构造，分不开。
这与本项目已经栽过的「纯平窗口 ratio=0 拿满分」是同一类坑的兄弟：
**指标本身没错，错在拿它跟一个不对的基准比。**

⚠ 唯一让 (a) 不完全同义反复的是**代理间隙**：搜索在**量化前的 16px 图**上打分，
而 (a) 量的是**最终出厂的量化瓦片**。所以 (a) 真正想问的是
「这个代理传不传得过量化」。那么正确的零假设不是"抛硬币"，而是
**一个随机抽的可行位置**——若代理完全不传递，被选中的位置在出厂瓦片上
就该和随机可行位置没有区别。

--- 判据（跑之前写下并 commit，事后不改）---

**自检（不过则本轮作废，不作任何判读）**：源图是重渲染的（原 `_src.png` 已因
服务器磁盘清理删除），必须先确认重渲染与当初同图：逐材质要求重算的
`off_seam` 与 JSON 一致、且 `ratio_center`/`ratio_seam` 与 JSON 相对差 <1e-6。
达标材质须 >= 90%；否则只报"渲染未复现"，Q1/Q2 一律不判读。

**Q1 —— 22/22 里有多少是"居中本来就差"**：每个材质抽 K=20 个随机可行位置
（与被选中的那个互斥），算它们在**出厂瓦片**上的接缝比。
逐材质取这 20 个的**中位**，与 `ratio_center` 做符号检验。
  - 若随机位置也显著地打败居中 -> 原来的 22/22 主要在说
    **"居中是个差位置"**，不是"接缝代理有效"；p=4.8e-7 只能重新写成
    **选择性构造下的界**，不能当作效应的证据。
  - 若随机位置打不过居中 -> 居中本身不差，22/22 里确有内容。

**Q2 —— 代理到底传不传得过量化（这才是 (a) 该有的零假设）**：
逐材质比 `ratio_seam`（选中）与那 20 个随机可行位置的**中位**，符号检验。
  - 方向为降且 p<0.05 -> 代理确实传递，方法三的操作层结论**在正确零假设下仍成立**，
    只需把报告口径从"对居中"改成"对随机可行位置"。
  - 否则 -> 操作检验 (a) 除了 min 选择的构造之外**什么也没建立**，
    `downsample.py` 那两个 p 必须撤下。

**描述性（不参与判定）**：`ratio_center` 在那 20 个随机抽样里的分位。
居中若落在很高的分位，说明"居中"确实是个糟糕的默认位置。

**本脚本不判定人工/判官层的主判据**（10/12、12/18 不受影响，那是另一层证据）。
只判操作检验 (a) 的零假设是否站得住。
"""
import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/paired"):
    sys.path.insert(0, str(ROOT / sub))
from downsample import auto_crop                       # noqa: E402
from seam_crop import small_of, seam_stats, to_tile, TMPL, NEG  # noqa: E402
from exact import binom_test                           # noqa: E402

K = 20          # 每个材质抽多少个随机可行位置（判据里写死）
RATIO_TOL = 1e-6


def admissible(src, side, size, step, keep=0.85):
    """枚举可行位置（与 best_offset 同一约束），返回 (argmin, 全部可行位置)。

    与 `seam_crop.best_offset` 逐字同构，只是把可行集也留下来——
    随机对照必须抽自**同一个**可行集，否则比的不是同一件事。
    """
    H, W = src.shape[:2]
    ys = list(range(0, H - side + 1, step)) or [0]
    xs = list(range(0, W - side + 1, step)) or [0]
    cy, cx = (H - side) // 2, (W - side) // 2
    floor = keep * seam_stats(small_of(src[cy:cy + side, cx:cx + side], size))[1]
    best, arg, adm = None, (cy, cx), []
    for y in ys:
        for x in xs:
            s_, i_ = seam_stats(small_of(src[y:y + side, x:x + side], size))
            if i_ < floor:
                continue
            adm.append((y, x))
            r = s_ / max(i_, 1e-6)
            if best is None or r < best:
                best, arg = r, (y, x)
    return arg, adm


def ratio_of_tile(t):
    s, i = seam_stats(t)
    return s / max(i, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--ref", type=Path, default=ROOT / "experiments/seam_crop.json",
                    help="当初那轮的 JSON，用来做渲染复现自检")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/seam_null.json")
    a = ap.parse_args()

    import re
    if a.prompts:
        prompts = json.loads(a.prompts.read_text(encoding="utf-8"))
    else:
        src_py = (ROOT / "analysis/paired/crop_scale_study.py").read_text(encoding="utf-8")
        prompts = re.findall(r'"([^"]+)"',
                             re.search(r"PROMPTS\s*=\s*\[(.*?)\]", src_py, re.S).group(1))
    if a.limit:
        prompts = prompts[:a.limit]

    ref = {r["prompt"]: r for r in json.loads(a.ref.read_text(encoding="utf-8"))}

    import torch
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    recs = []
    for pi, p in enumerate(prompts):
        g = torch.Generator("cuda").manual_seed(a.seed + pi)
        im = pipe(TMPL.format(p=p), negative_prompt=NEG, num_inference_steps=a.steps,
                  generator=g, height=a.render, width=a.render).images[0]
        src = np.asarray(im).astype(float)

        _, frac = auto_crop(src, a.size)
        if frac >= 0.999:
            print(f"[{pi+1}/{len(prompts)}] {p:<32} 门未触发，按构造排除", flush=True)
            continue
        H, W = src.shape[:2]
        side = int(round(frac * min(H, W)))
        cy, cx = (H - side) // 2, (W - side) // 2
        (by, bx), adm = admissible(src, side, a.size, a.grid)

        t_ctr = to_tile(src[cy:cy + side, cx:cx + side], a.size, a.colors, pi)
        t_sel = to_tile(src[by:by + side, bx:bx + side], a.size, a.colors, pi)
        r_ctr, r_sel = ratio_of_tile(t_ctr), ratio_of_tile(t_sel)

        pool = [o for o in adm if o != (by, bx)]
        rng = random.Random(1000 + pi)          # 逐材质固定，可复跑
        picks = rng.sample(pool, min(K, len(pool)))
        r_rand = []
        for (y, x) in picks:
            r_rand.append(ratio_of_tile(to_tile(src[y:y + side, x:x + side],
                                                a.size, a.colors, pi)))

        R = ref.get(p, {})
        same = (R.get("off_seam") == [by, bx]
                and abs(r_ctr - R.get("ratio_center", -1)) <= RATIO_TOL * max(1.0, r_ctr)
                and abs(r_sel - R.get("ratio_seam", -1)) <= RATIO_TOL * max(1.0, r_sel))
        recs.append({"prompt": p, "side": side, "n_admissible": len(adm),
                     "off_center": [cy, cx], "off_seam": [by, bx],
                     "ratio_center": r_ctr, "ratio_seam": r_sel,
                     "ratio_random": r_rand, "render_matches_ref": bool(same),
                     "ref_n_admissible": R.get("n_admissible")})
        mr = statistics.median(r_rand)
        print(f"[{pi+1}/{len(prompts)}] {p:<32} 居中 {r_ctr:6.2f}  选中 {r_sel:6.2f}  "
              f"随机中位 {mr:6.2f}  可行 {len(adm)}  复现{'OK' if same else '**不符**'}",
              flush=True)
        a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    report(recs)


def report(recs):
    if not recs:
        print("无适用材质"); return
    n = len(recs)
    good = sum(r["render_matches_ref"] for r in recs)
    print(f"\n== 自检：渲染复现 {good}/{n} = {good/n:.0%}（判据 >=90%）==")
    if good / n < 0.90:
        print("**不达标**：重渲染与当初不是同一批图，Q1/Q2 按判据不予判读。")
        return

    med_rand = [statistics.median(r["ratio_random"]) for r in recs]

    print("\n== Q1：随机可行位置 vs 居中 ==")
    w = sum(1 for r, m in zip(recs, med_rand) if m < r["ratio_center"])
    p1 = binom_test(w, n)
    print(f"  随机中位比居中低：{w}/{n}，符号检验 p={p1:.3g}")
    print(f"  接缝比中位：居中 {statistics.median(r['ratio_center'] for r in recs):.2f}"
          f" -> 随机 {statistics.median(med_rand):.2f}")
    q1_beats = w / n > 0.5 and p1 < 0.05
    print("  -> " + ("**随机位置也打败居中**：原 22/22 主要在说「居中是个差位置」，"
                     "那两个 p 只能作选择性构造下的界" if q1_beats else
                     "随机位置打不过居中：居中本身不差，22/22 里确有内容"))

    print("\n== Q2：选中位置 vs 随机可行位置（(a) 该有的零假设）==")
    w2 = sum(1 for r, m in zip(recs, med_rand) if r["ratio_seam"] < m)
    p2 = binom_test(w2, n)
    print(f"  选中比随机中位低：{w2}/{n}，符号检验 p={p2:.3g}")
    print(f"  接缝比中位：随机 {statistics.median(med_rand):.2f}"
          f" -> 选中 {statistics.median(r['ratio_seam'] for r in recs):.2f}")
    ok2 = w2 / n > 0.5 and p2 < 0.05
    print("  -> " + ("**代理传递**：换成正确零假设后操作层结论仍成立，"
                     "报告口径应从「对居中」改为「对随机可行位置」" if ok2 else
                     "**代理未传递**：操作检验 (a) 除 min 选择的构造外什么也没建立，"
                     "downsample.py 那两个 p 必须撤下"))

    pct = [100.0 * sum(1 for v in r["ratio_random"] if v < r["ratio_center"])
           / len(r["ratio_random"]) for r in recs]
    print(f"\n（描述性）居中在 {K} 个随机可行位置中的分位：中位 "
          f"{statistics.median(pct):.0f}% 的随机位置比它更好")


if __name__ == "__main__":
    main()
