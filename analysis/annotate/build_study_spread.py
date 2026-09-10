"""人工盲比：拉亮度跨度（保住空间排布）到底有没有让材质更容易认出。

**为什么要人来判**。这条路当初被算作关闭，靠的是一句**目视**
（「明暗层次回来了但材质并没有更可辨认」）；`spread_recognise.py`
（预注册 `e9e3045`）真去问了这条轴，判官选重标版 **10/11 = 91%，p=0.0117**，
原轮的 9/10 也在更大集上复现（11/13）。**关闭理由不成立。**
但那是**判官**，且 **72% 的对判不上来**。按预注册，下一步是人。

**问的是"更容易认出"，不是"更像/更好看"**——因为关闭理由落在可辨认性上。
本页因此把模板写死的那句标题换掉了（模板本身不动，v2 重建后会验证问题仍在）。

**不重新生成任何像素**：`before` 直接取 `experiments/pack78_out/base/<size>/`
的出厂瓦片，`after` 取 `spread_recognise.py` 已写出的 `experiments/spreadrecog/`。
所以人与判官看的是**同一批图**。

用法：
    python analysis/annotate/build_study_spread.py            # 出 v1（每对一次）
    python analysis/annotate/build_paired_orders.py \
        experiments/annotate/study_spread.html                # 转成两序去偏版
⚠ **交给用户的必须是 v2**（两序 + 同图检查）——v1 对"只看一侧就点"没有防线，
接缝那次实测 71% 选左。

判据沿用 `spread_recognise.py` 已预注册的那套，只把判官换成人：
主判据 = 重标版被选 >50% 且二项 p<0.05（分析脚本另出，不在此文件）。
"""
import argparse
import base64
import io
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "哪一张更容易认出是这个材质？"
TEMPLATE_Q = "哪一张更像这个材质？"        # 模板写死的那句，只在本页替换


def b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32])
    ap.add_argument("--n-check", type=int, default=4)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--pack", type=Path, default=ROOT / "experiments/pack78_out/base")
    ap.add_argument("--after", type=Path, default=ROOT / "experiments/spreadrecog")
    ap.add_argument("--recs", type=Path, default=ROOT / "experiments/spread_recognise.json")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/annotate/study_spread.html")
    a = ap.parse_args()

    recs = json.loads(a.recs.read_text(encoding="utf-8"))
    rng = random.Random(a.seed)
    items, sides, missing = [], [0, 0], 0
    for r in recs:
        slug = r["material"].replace(" ", "_")
        bp = a.pack / str(r["size"]) / f"{slug}.png"
        ap_ = a.after / f"{slug}_{r['size']}_after.png"
        if not (bp.exists() and ap_.exists()):
            missing += 1
            continue
        imgs = {"before": b64(np.asarray(Image.open(bp).convert("RGB"))),
                "after": b64(np.asarray(Image.open(ap_).convert("RGB")))}
        first = sides[0] <= sides[1]          # 重标版在左的次数逐格配平
        sides[0 if first else 1] += 1
        l, r_ = ("after", "before") if first else ("before", "after")
        items.append({"material": f'{r["material"]} · {r["size"]}px',
                      "label": r["material"], "kind": "real",
                      "struct": round(r["spread_after"] - r["spread_before"], 3),
                      "stratum": f'{r["size"]}px', "left": l, "right": r_,
                      "limg": imgs[l], "rimg": imgs[r_]})
    n_real = len(items)

    # 模糊检查（v2 builder 还会另加"两边同图"那种）
    for it in rng.sample(items, min(a.n_check, len(items))):
        good = Image.open(io.BytesIO(base64.b64decode(it["limg"]))).convert("RGB")
        blur = (good.resize((max(good.size[0] // 3, 4),) * 2, Image.NEAREST)
                .filter(ImageFilter.GaussianBlur(2)).resize(good.size, Image.BILINEAR))
        gg = {"good": b64(np.asarray(good)), "blur": b64(np.asarray(blur))}
        l, r_ = ("good", "blur") if rng.random() < 0.5 else ("blur", "good")
        items.append({"material": it["material"], "label": it["label"], "kind": "check",
                      "struct": 0.0, "stratum": it["stratum"],
                      "left": l, "right": r_, "limg": gg[l], "rimg": gg[r_]})

    rng.shuffle(items)
    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    assert tpl.count(TEMPLATE_Q) == 1, "模板的问题句变了，先确认再改"
    html = (tpl.replace(TEMPLATE_Q, QUESTION)
            .replace("__ITEMS__", json.dumps(items, ensure_ascii=False)))
    assert QUESTION in html and TEMPLATE_Q not in html
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(html, encoding="utf-8")

    print(f"可比 {n_real} 对（13 材质 x 3 档），模糊检查 {len(items)-n_real}，缺图 {missing}")
    print(f"左右平衡（重标版在左/在右）：{sides[0]} / {sides[1]}")
    print(f"问题句：{QUESTION}")
    print(f"写入 {a.out}  ({a.out.stat().st_size // 1024} KB)")
    print("⚠ 下一步必须转 v2：python analysis/annotate/build_paired_orders.py "
          f"{a.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
