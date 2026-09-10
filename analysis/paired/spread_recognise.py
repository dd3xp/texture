"""补一个违规：`spread_rescale` 那条路是被**目视**关掉的，可辨认性从未测过。

**为什么做这个**。各向同性那半边记作"四条路全关"，其中第二条
（`spread_rescale`：把图内**实际**亮度跨度拉到真人惯例 0.292，**保住空间排布**
再重量化）当初的结果是**经验证判官 9/10 = 90%，p=0.0215**——按本项目协议，
判官压缩下的正结果是**下界**。正文 §5 如实报了这个 9/10。

它仍被算作关闭，理由写在 2026-09-07 的记录里，是一句**目视**：
「明暗层次回来了（grass turf 有了绿色层次、矿点可见、树皮出木纹），
**但材质并没有更可辨认**，dry sand 拉伸后像木板」。

⚠ 而本项目自己的规矩是「**目视判断不是结论**，标 ⚠ 并注明待盲比」。
这条**从未做过盲比，也从未测过可辨认性**。当初判官 21 对里弃了 11 对
（52% 不一致），也提示**问的问题与排除的理由不是同一个**：
判官被问的是"更像这个材质、更像能用的贴图"，而排除它靠的是**可辨认性**。

**所以本轮换的是问题，不是判据的宽严**——换的正是当初据以关闭的那条轴。
两个问题**都问、都报**，不论结果。

--- 已经做完的部分（跑前如实交代）---

操作检验是**确定性图像运算、不涉及判官**，已先跑并通过：
13 个门拒绝材质 × 3 档 = **39 条**，重标后图内实际跨度全部落进
**0.292 ± 20%**（39/39；例：concrete sidewalk 16px 0.018 → 0.291）。
沿用原轮预注册的那条操作检验（`spread_rescale.py`：必须量**图内实际出现的颜色**，
不是调色板对象——那是 B11 同类错误，原轮为此作废过一整轮）。
**尚未知的是任何判官结果。**

--- 样本框（不是现在挑的）---

`experiments/pack78_out/manifest.json` 里 `gate_fired=False` 的 **13 个材质**
× 16/24/32 三档。这个集合由**门**在既有交付批次上定义，不由我事后选择。
n=39，比原轮的 18 条大一倍多，且用的是**真实出厂的瓦片**。

--- 判据（跑之前写下并 commit，事后不改）---

**主判据 —— 可辨认性**（这正是关闭理由所在的那条轴）：
经验证判官（claude-opus-5，正反两问去偏、不一致弃用）被问
「哪一张更容易**认出**是「<材质名>」？」
  **成立**：重标版被选 >50% 且二项 p<0.05
        -> **关闭理由不成立**。那条路以下界形式**重新打开**，
           并应升级到人工盲比（用 v2 那套两序去偏的仪器）。
  **证伪**：≤50% 或不显著
        -> 目视判断在可辨认性这条轴上**被证据支持**，这条路**带证据地关闭**，
           不再是"凭目视关的"。
        ⚠ 判官压缩：负结果只能说**"没测到大效应"**，不能宣称
           "重标一定不提升可辨认性"（判官不能确认零假设）。

**次判据（预先指定，描述性，不参与判定）—— 原轮那个问题**：
「哪一张更像这个材质、更像一张能用的游戏贴图？」
用于与当初的 9/10 对齐比较。两问的方向若不一致，本身就是有信息的：
说明"好看/像"与"认得出"在这半边材质上分家。

**弃样处理**：正反两问不一致即弃用，并报出弃了几对（原轮弃了 11/21，
不一致率本身要看）。

**不做的事**：不改 `spread_rescale.py`、不改任何已发表数字、不动 `paper/`。
本脚本只回答"关闭理由站不站得住"。

纯本地、不用 GPU、不写服务器磁盘。
"""
import argparse
import base64
import io
import json
import os
import statistics
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools", "analysis", "analysis/annotate", "analysis/paired"):
    sys.path.insert(0, str(ROOT / sub))
from make_texture import SPREAD_ARTIST_MEDIAN                     # noqa: E402
from spread_rescale import realized_spread, rescale               # noqa: E402
from exact import binom_test, jeffreys                            # noqa: E402

Q_RECOG = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
           "哪一张更容易**认出**是这个材质？只回答 A 或 B，不要解释。\n"
           "（第一张是 A，第二张是 B）")
Q_PREF = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
          "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
          "（第一张是 A，第二张是 B）")
TOL = 0.20            # 操作检验：0.292 ± 20%（原轮预注册）


def b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).resize((256, 256), Image.NEAREST).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def judge(ask, model, q, imgA, imgB, base_url, key):
    """A/B 各问一次，返回 'A'/'B'/'inconsistent'/None。"""
    x, y = b64(imgA), b64(imgB)
    o1 = ask(model, q, [x, y], base_url, key)
    o2 = ask(model, q, [y, x], base_url, key)
    if not o1 or not o2:
        return None
    p1 = "A" if o1.upper().startswith("A") else "B"
    p2 = "B" if o2.upper().startswith("A") else "A"
    return p1 if p1 == p2 else "inconsistent"


def declined_materials(manifest: Path = None):
    """门拒绝的材质。⚠ 这是**渲染**的属性不是材质的属性（2026-09-11 查明）：
    种子含材质在列表里的索引，同一材质在不同批次拿到不同渲染，判定会翻
    （pack78 的 13 个里 8 个在 pack53 批次触发了门）。所以清单必须跟着批次走。"""
    manifest = manifest or (ROOT / "experiments/pack78_out/manifest.json")
    recs = json.loads(Path(manifest).read_text(encoding="utf-8"))
    return sorted({r["material"] for r in recs
                   if r["variant"] == "base" and not r["gate_fired"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--colors", type=int, default=12)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--pack", type=Path, default=ROOT / "experiments/pack78_out/base")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="门拒绝清单跟着批次走（缺省 pack78_out）。纯增量，试点路径不变。")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/spread_recognise.json")
    ap.add_argument("--tiles", type=Path, default=ROOT / "experiments/spreadrecog")
    a = ap.parse_args()

    mats = declined_materials(a.manifest)
    a.tiles.mkdir(parents=True, exist_ok=True)
    print(f"门拒绝材质 {len(mats)} 个 x {len(a.sizes)} 档；目标跨度 {SPREAD_ARTIST_MEDIAN}")

    recs = []
    for m in mats:
        slug = m.replace(" ", "_")
        for size in a.sizes:
            f = a.pack / str(size) / f"{slug}.png"
            if not f.exists():
                print(f"  缺 {f.name}（{size}px），跳过", flush=True)
                continue
            before_img = np.asarray(Image.open(f).convert("RGB"))
            after_img, _, _ = rescale(before_img, a.colors, SPREAD_ARTIST_MEDIAN)
            sb, sa = realized_spread(before_img), realized_spread(after_img)
            Image.fromarray(after_img.astype(np.uint8)).save(
                a.tiles / f"{slug}_{size}_after.png")
            recs.append({"material": m, "size": size,
                         "spread_before": sb, "spread_after": sa,
                         "manip_ok": bool(abs(sa - SPREAD_ARTIST_MEDIAN)
                                          <= TOL * SPREAD_ARTIST_MEDIAN)})
            print(f"  {m:<26} {size}px  跨度 {sb:.3f} -> {sa:.3f}"
                  f"{'' if recs[-1]['manip_ok'] else '   **未命中目标**'}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(r["manip_ok"] for r in recs)
    print(f"\n操作检验：图内实际跨度落进 {SPREAD_ARTIST_MEDIAN}±{TOL:.0%} 的 {ok}/{len(recs)}")
    if not recs or ok / len(recs) < 0.9:
        print("  -> **操作检验不过**，主判据不予评估（判据已在跑前固定）")
        return
    print("  -> 过")
    if a.no_judge:
        print("（--no-judge：跳过判官）")
        return

    from vlm_judge import ask
    base_url, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base_url or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    tal = {"recog": [0, 0, 0], "pref": [0, 0, 0]}     # wins(after), tot, inconsistent
    for r in recs:
        slug = r["material"].replace(" ", "_")
        before_img = np.asarray(Image.open(a.pack / str(r["size"]) / f"{slug}.png")
                                .convert("RGB"))
        after_img = np.asarray(Image.open(a.tiles / f"{slug}_{r['size']}_after.png")
                               .convert("RGB"))
        for tag, tpl in (("recog", Q_RECOG), ("pref", Q_PREF)):
            q = tpl.format(n=r["size"], label=r["material"])
            v = judge(ask, a.model, q, after_img, before_img, base_url, key)
            if v is None:
                continue
            r[f"vlm_{tag}"] = v
            if v == "inconsistent":
                tal[tag][2] += 1
            else:
                tal[tag][1] += 1
                tal[tag][0] += v == "A"           # A = 重标版
        print(f"  {r['material']:<26} {r['size']}px  "
              f"认出 {r.get('vlm_recog')}   偏好 {r.get('vlm_pref')}", flush=True)

    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print()
    for tag, name, primary in (("recog", "主判据（更容易认出）", True),
                               ("pref", "次判据（更像/更能用，原轮那个问题）", False)):
        w, t, inc = tal[tag]
        if not t:
            print(f"{name}：无有效判断（弃 {inc}）")
            continue
        p = binom_test(w, t)
        lo, hi = jeffreys(w, t)
        print(f"{name}：重标版被选 {w}/{t} = {w/t:.0%}  p={p:.3g}  [{lo:.0%},{hi:.0%}]"
              f"   （不一致弃 {inc}，不一致率 {inc/(inc+t):.0%}）")
        if not primary:
            continue
        if w / t > 0.5 and p < 0.05:
            print("  -> **关闭理由不成立**：重标确实更容易认出（判官压缩，故为下界）。")
            print("     该路线以下界形式重新打开，应升级到人工盲比（v2 两序去偏仪器）。")
        else:
            print("  -> 目视判断**被证据支持**：这条路带证据地关闭，不再是凭目视关的。")
            print("     ⚠ 判官压缩：只能说「没测到大效应」，不能宣称重标一定不提升可辨认性。")


if __name__ == "__main__":
    main()
