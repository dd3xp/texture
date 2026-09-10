"""方法三：裁**哪里**——把裁剪窗口的位置也选出来。

`auto_crop` 把窗口**边长**选得很讲究（`side = 4.5 x period`，B9/B13/B14 全靠它），
但窗口**位置**从头到尾是写死居中的（`downsample.py:243` 的 `(H-side)//2`）。
位置从来没被当成一个可选的量。

而对用户的真实用途（Minecraft 贴图）来说位置**直接可见**：贴图是**平铺**在墙面上的，
玩家看到的从来不是一张瓦片，而是 3x3、5x5 地铺开的一片。窗口若没落在周期格点上，
瓦片首尾接不上，平铺后会出现一条贯穿整面墙的断线——**这个项目至今没量过接缝**。

  基线：居中裁（现行生产管线）
  挑战：**同样的 side**（门与尺度完全不变，只换位置），在候选位置里取接缝分最低的

只换位置这一件事，是为了让结论能归因到位置上，而不是又一次改了尺度。

**接缝分**（在瓦片上、平铺时真正会露出来的那条缝上量）：
    seam     = 首尾列之差 + 首尾行之差（逐通道平均绝对差）
    internal = 内部相邻列/行之差的**中位数**
    ratio    = seam / internal
用**比值**而不是 seam 本身，是因为 seam 最小的窗口是**平的窗口**——
那正是本项目栽过三次的坑（调色板跨度、伪触发、B11：操作检验"过了"而图没变好）。
⚠ **但比值本身也不够**：写脚本时自测发现纯平窗口的 ratio 是 **0**
（seam=0、internal=0 -> 0/eps=0），即平坦窗口反而拿满分。
所以真正防住这个坑的是 `best_offset` 里的**硬约束**：候选窗口的 internal
须 >= 居中窗口的 85%。在保住同等结构的窗口里，才比谁平铺得好。

⚠ 搜索时在**量化前**的 16px 图上算分（快，2000 个候选位置才可行），
   而操作检验量的是**最终出厂的量化瓦片**。代理若不传递，操作检验会抓住。

--- 判据（跑之前写下并 commit，事后不改）---

**适用范围**：只在门触发且真的裁了（frac<0.999）的材质上比。门没触发时整图返回，
位置无从选起——这部分按构造排除，不算在分母里。

**操作检验（两条都要过，缺一即作废）**
  (a) 出厂瓦片的接缝比 ratio 下降：逐材质配对符号检验 p<0.05 且方向为降。
  (b) **平坦守卫**：选中窗口的 internal 中位不得低于居中窗口的 **85%**。
      低于则判定"接缝是靠把画面拍平换来的"，**主判据不予评估**，本轮作废。
      （该约束同时**内建在搜索里**：候选窗口的 internal 须 >= 居中窗口的 85%，
       否则不进候选集。自测发现纯平窗口的接缝比是 0 即满分——
       "接缝最小"单独当目标必然拍平画面。居中窗口恒满足，故最差退回居中。）

**主判据**：把两张瓦片各自 3x3 平铺（**玩家真正看到的样子**）后交经验证判官
  （claude-opus-5，正反两问去偏、不一致弃用）：接缝裁被选 >50% 且二项 p<0.05
  -> **位置这条杠杆有效**，收进交付管线。

**证伪**：<=50% 或不显著 -> 位置杠杆**没测到大效应**，不进管线，路线关闭。
  （判官压缩效应：只能作下界、只能证伪零假设，**不能确认零假设**——
   所以负结果写成"没测到"，不写成"位置无关"。）

**次判据（描述性，不参与判定）**：单张瓦片（不平铺）上同样问一次。
  若平铺下赢、单张下不赢，那正说明收益确实来自接缝而非别的画质差异。
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/annotate"):
    sys.path.insert(0, str(ROOT / sub))
from downsample import auto_crop, dominant_period, anisotropy, UNITS_PER_TILE  # noqa: E402
from make_texture import extract_palette, quantize                            # noqa: E402
from exact import binom_test, jeffreys                                        # noqa: E402

TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")
NEG = "perspective, 3d render, vignette, watermark, text, border, blurry"
Q = ("下面是两张「{label}」的像素画材质贴图，各自平铺成 3x3 铺满了一面墙。\n"
     "哪一面墙看起来更像一张能用的无缝游戏贴图——接缝更不明显、图案接得更顺？\n"
     "只回答 A 或 B，不要解释。（第一张是 A，第二张是 B）")
Q1 = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
      "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
      "（第一张是 A，第二张是 B）")


def small_of(win, size):
    """裁下来的窗口 -> size x size（量化前）。"""
    return np.asarray(Image.fromarray(win.astype(np.uint8))
                      .resize((size,) * 2, Image.BOX)).astype(float)


def seam_stats(t):
    """(seam, internal)：平铺时露出来的那条缝，与内部相邻差的中位。"""
    t = t.astype(float)
    seam = float(np.abs(t[:, -1] - t[:, 0]).mean() + np.abs(t[-1, :] - t[0, :]).mean())
    dv = [float(np.abs(t[:, i + 1] - t[:, i]).mean()) for i in range(t.shape[1] - 1)]
    dh = [float(np.abs(t[i + 1, :] - t[i, :]).mean()) for i in range(t.shape[0] - 1)]
    return seam, float(statistics.median(dv + dh))


def ratio_of(t):
    s, i = seam_stats(t)
    return s / max(i, 1e-6)


def tile3(t, up=8):
    """3x3 平铺 + NEAREST 放大 —— 玩家真正看到的那个视图。"""
    g = np.tile(t, (3, 3, 1))
    return np.asarray(Image.fromarray(g.astype(np.uint8))
                      .resize((g.shape[1] * up, g.shape[0] * up), Image.NEAREST))


def best_offset(src, side, size, step, keep=0.85):
    """在候选位置里取接缝比最低的（量化前评分），**但只在保住结构的窗口里取**。

    ⚠ 这条约束不是装饰：写这个脚本时自测发现，**纯平窗口的 ratio 是 0**
    （seam=0、internal=0，0/eps=0），即平坦窗口拿满分——
    "接缝最小"单独拿来当目标必然把画面拍平，正是 B11 那类坑。
    所以把"内部相邻差不低于居中窗口的 `keep`"做成**搜索时的硬约束**：
    在与现行生产裁法保留同等结构的窗口里，挑平铺得最好的那个。
    居中窗口自己恒满足（比值 1.0），故候选集非空，最差退回居中。
    """
    H, W = src.shape[:2]
    ys = list(range(0, H - side + 1, step)) or [0]
    xs = list(range(0, W - side + 1, step)) or [0]
    cy, cx = (H - side) // 2, (W - side) // 2
    _, i_ctr = seam_stats(small_of(src[cy:cy + side, cx:cx + side], size))
    floor = keep * i_ctr
    best, arg, n_adm = None, (cy, cx), 0
    for y in ys:
        for x in xs:
            t = small_of(src[y:y + side, x:x + side], size)
            s_, i_ = seam_stats(t)
            if i_ < floor:                       # 结构比现行裁法还少，不要
                continue
            n_adm += 1
            r = s_ / max(i_, 1e-6)
            if best is None or r < best:
                best, arg = r, (y, x)
    return arg[0], arg[1], (len(ys) * len(xs), n_adm)


def to_tile(win, size, colors, seed):
    small = small_of(win, size)
    return quantize(np.asarray(small, np.uint8),
                    extract_palette(np.asarray(small, np.uint8), colors,
                                    seed=seed)).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--render", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=21, help="与 lora_eval / crop_scale_study 同口径")
    ap.add_argument("--grid", type=int, default=12, help="位置搜索步长（像素）")
    ap.add_argument("--prompts", type=Path,
                    help="外挂材质表（JSON 数组）。缺省用 crop_scale_study 的 42 条。"
                         "**纯增量开关，跑过的默认路径分毫未动**——加它是为了在"
                         "B21 那 60 个零重叠材质上做泛化复现。")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--srcdir", type=Path, default=None,
                    help="给定则复用已渲染的 <slug>.png，不重新渲染")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/seam_crop.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/seamcrop")
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
    a.tiles.mkdir(parents=True, exist_ok=True)

    pipe = None
    if a.srcdir is None:
        import torch
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
            variant="fp16", use_safetensors=True).to("cuda")
        pipe.set_progress_bar_config(disable=True)

    recs = []
    for pi, p in enumerate(prompts):
        slug = p.replace(" ", "_")
        if a.srcdir is not None:
            f = a.srcdir / f"{slug}.png"
            if not f.exists():
                print(f"[{pi+1}/{len(prompts)}] {p:<32} 缺源图，跳过", flush=True)
                continue
            src = np.asarray(Image.open(f).convert("RGB")).astype(float)
        else:
            import torch
            g = torch.Generator("cuda").manual_seed(a.seed + pi)
            im = pipe(TMPL.format(p=p), negative_prompt=NEG,
                      num_inference_steps=a.steps, generator=g,
                      height=a.render, width=a.render).images[0]
            src = np.asarray(im).astype(float)
            Image.fromarray(src.astype(np.uint8)).save(a.tiles / f"{slug}_src.png")

        # 与生产管线完全相同的门与尺度：只借它的 side，不改任何判断
        _, frac = auto_crop(src, a.size)
        if frac >= 0.999:
            recs.append({"prompt": p, "eligible": False, "frac": float(frac)})
            print(f"[{pi+1}/{len(prompts)}] {p:<32} 门未触发/未真裁，按构造排除", flush=True)
            continue
        H, W = src.shape[:2]
        side = int(round(frac * min(H, W)))
        cy, cx = (H - side) // 2, (W - side) // 2
        by, bx, (n_all, n_adm) = best_offset(src, side, a.size, a.grid)

        t_ctr = to_tile(src[cy:cy + side, cx:cx + side], a.size, a.colors, pi)
        t_seam = to_tile(src[by:by + side, bx:bx + side], a.size, a.colors, pi)
        Image.fromarray(t_ctr).save(a.tiles / f"{slug}_center.png")
        Image.fromarray(t_seam).save(a.tiles / f"{slug}_seam.png")
        Image.fromarray(tile3(t_ctr)).save(a.tiles / f"{slug}_center_x9.png")
        Image.fromarray(tile3(t_seam)).save(a.tiles / f"{slug}_seam_x9.png")

        s_c, i_c = seam_stats(t_ctr)
        s_s, i_s = seam_stats(t_seam)
        recs.append({"prompt": p, "eligible": True, "frac": float(frac), "side": side,
                     "n_cand": n_all, "n_admissible": n_adm, "off_center": [cy, cx], "off_seam": [by, bx],
                     "moved": bool((by, bx) != (cy, cx)),
                     "seam_center": s_c, "int_center": i_c,
                     "seam_seam": s_s, "int_seam": i_s,
                     "ratio_center": s_c / max(i_c, 1e-6),
                     "ratio_seam": s_s / max(i_s, 1e-6)})
        print(f"[{pi+1}/{len(prompts)}] {p:<32} side={side} 候选{n_adm}/{n_all} "
              f"接缝比 {s_c/max(i_c,1e-6):.2f} -> {s_s/max(i_s,1e-6):.2f}"
              f"{'' if (by,bx)!=(cy,cx) else '  （仍是居中）'}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")

    el = [r for r in recs if r.get("eligible")]
    print(f"\n适用材质（门触发且真裁）：{len(el)}/{len(recs)}；"
          f"其中位置真的移动了：{sum(r['moved'] for r in el)}")
    if not el:
        print("无适用材质，无从评估"); return

    # —— 操作检验 (a)：出厂瓦片的接缝比确实降了吗 ——
    down = sum(1 for r in el if r["ratio_seam"] < r["ratio_center"])
    pa = binom_test(down, len(el))
    mc = statistics.median(r["ratio_center"] for r in el)
    ms = statistics.median(r["ratio_seam"] for r in el)
    print(f"\n操作检验(a) 接缝比中位 {mc:.2f} -> {ms:.2f}；"
          f"降低 {down}/{len(el)}，符号检验 p={pa:.3g}")
    ok_a = ms < mc and pa < 0.05
    print(f"  -> {'过' if ok_a else '**不过**'}")

    # —— 操作检验 (b)：平坦守卫 ——
    gc = statistics.median(r["int_center"] for r in el)
    gs = statistics.median(r["int_seam"] for r in el)
    keep = gs / max(gc, 1e-6)
    print(f"操作检验(b) 平坦守卫：内部相邻差中位 {gc:.2f} -> {gs:.2f}"
          f"（保留 {keep:.0%}）；阈值 85%")
    ok_b = keep >= 0.85
    print(f"  -> {'过' if ok_b else '**不过：接缝是靠把画面拍平换来的**'}")

    if not (ok_a and ok_b):
        print("\n操作检验未全过，**主判据不予评估**（判据已在跑前固定）。")
        return
    if a.no_judge:
        print("\n（--no-judge：跳过判官）"); return

    from vlm_judge import ask
    import base64, io

    def b64(p_):
        buf = io.BytesIO()
        Image.open(p_).convert("RGB").save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    cmp_ = [r for r in el if r["moved"]]
    print(f"\n位置未移动的 {len(el)-len(cmp_)} 个两张图相同，无从比较，不入判官。")
    tal = {"x9": [0, 0, 0], "one": [0, 0, 0]}      # win, tot, inconsistent
    for r in cmp_:
        slug = r["prompt"].replace(" ", "_")
        for k, (sfx, q) in enumerate((("_x9", Q.format(label=r["prompt"])),
                                      ("", Q1.format(n=a.size, label=r["prompt"])))):
            key_ = "x9" if sfx else "one"
            S = b64(a.tiles / f"{slug}_seam{sfx}.png")
            C = b64(a.tiles / f"{slug}_center{sfx}.png")
            o1 = ask(a.model, q, [S, C], base_url, key)
            o2 = ask(a.model, q, [C, S], base_url, key)
            if not o1 or not o2:
                continue
            p1 = "seam" if o1.upper().startswith("A") else "center"
            p2 = "center" if o2.upper().startswith("A") else "seam"
            v = p1 if p1 == p2 else "inconsistent"
            r[f"vlm_{key_}"] = v
            if v == "inconsistent":
                tal[key_][2] += 1
            else:
                tal[key_][1] += 1
                tal[key_][0] += v == "seam"
        print(f"  {r['prompt']:<32} 平铺 {r.get('vlm_x9')}   单张 {r.get('vlm_one')}",
              flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    for key_, name in (("x9", "主判据（3x3 平铺）"), ("one", "次判据（单张，描述性）")):
        w, t, inc = tal[key_]
        if not t:
            print(f"{name}：无有效判断"); continue
        pv = binom_test(w, t); lo, hi = jeffreys(w, t)
        print(f"{name}：接缝裁被选 {w}/{t} = {w/t:.0%}  p={pv:.3g}  [{lo:.0%},{hi:.0%}]"
              f"   （弃 {inc}）")
        if key_ != "x9":
            continue
        if w / t > 0.5 and pv < 0.05:
            print("  -> **成立**：裁剪位置这条杠杆有效（判官压缩，故为下界），收进交付管线")
        else:
            print("  -> 不成立：**没测到大效应**，位置杠杆不进管线，本路线关闭。")
            print("     （判官压缩只能证伪零假设、不能确认零假设，故不写成「位置无关」。）")


if __name__ == "__main__":
    main()
