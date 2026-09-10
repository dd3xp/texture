"""接缝对齐的人工盲比：居中裁 vs 接缝对齐，两轮材质合起来出一份。

为什么必须做这个：VLM 判官在这个比较上**分辨力到头了**——
两轮 3x3 平铺下不一致率 33% / 38%，单张瓦片上 87%。首轮判官 10/12 成立、
泛化轮 12/18 不成立，两个区间宽到互相包含（[56,96] 与 [44,85]）。
再换一个材质集只会得到第三个同样宽的区间。**换仪器，不是换样本。**

**不重新渲染**：直接用两轮实验已经存下的瓦片
（`experiments/seamcrop_demo/`、`seamcrop60_demo/`，各自的 `*_center_x9.png`
与 `*_seam_x9.png`）。所以这份盲比与 VLM 看到的**是同一批图**，
两个仪器的差异不会被"图不一样"污染。

**看的是 3x3 平铺视图**，因为效应就在那儿：同样两张瓦片单看时判官 15 判 13
自相矛盾，铺开就稳定选接缝裁。人也应该看玩家真正看到的那个视图。

--- 判据（跑之前写下并 commit，事后不改）---

**主判据**：全部可比材质合并（两轮共 57），标注者偏好接缝对齐 >50% 且二项 p<0.05
  -> **人能看出来**，"接得上"这件事对人可见，方法三的说法可以从"客观测量"
     升级为"人工盲比验证"。
**证伪**：<=50% 或不显著 -> 人也看不出来。那么接缝对齐只是一个**客观上更对、
  但观感上无差别**的改动；交付仍可保留（它不会更差且平坦守卫排除了作弊），
  但**任何"更好看"的说法都要撤掉**，正文只能写成"改善了可测的平铺连续性"。

**预先指定的次判据（描述性，不参与判定）**：按材质集分层（原 42 条 / 留出 60 条）。
  两层方向一致 -> VLM 泛化轮的不复现是**判官分辨力**问题；
  两层方向相反 -> 是**材质集**问题，那比判官问题严重，要单独查。

**合并两轮在这里是合法的**，与此前"不许合并 VLM 两轮凑显著"不矛盾：
  那条禁的是**事后**把两个已知结果拼起来救显著性；
  这里是**换了仪器、跑之前**就把两个材质集定义成同一个盲比的样本框。

**注意力检查**：好图 vs 同图重糊，混进去若干。检查错 >1 个则该次标注作废。
**左右平衡**：接缝版在左/在右的次数逐格配平，避免位置偏好。
**盲**：页面只显示材质名，不显示哪边是哪个方法。

跑法（纯本地，不需要 GPU）：`python analysis/annotate/build_study_seam.py`
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
SETS = [("orig42", ROOT / "experiments/seamcrop_demo", ROOT / "experiments/seam_crop.json"),
        ("holdout60", ROOT / "experiments/seamcrop60_demo", ROOT / "experiments/seam_crop60.json")]


def b64_of(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-check", type=int, default=5)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/annotate/study_seam.html")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    items, sides, missing = [], [0, 0], 0
    for tag, d, js in SETS:
        recs = [r for r in json.loads(js.read_text(encoding="utf-8")) if r.get("eligible")]
        for r in recs:
            slug = r["prompt"].replace(" ", "_")
            c, s = d / f"{slug}_center_x9.png", d / f"{slug}_seam_x9.png"
            if not (c.exists() and s.exists()):
                missing += 1
                continue
            imgs = {"center": b64_of(c), "seam": b64_of(s)}
            first = sides[0] <= sides[1]            # 接缝版在左的次数逐格配平
            sides[0 if first else 1] += 1
            l, r_ = ("seam", "center") if first else ("center", "seam")
            items.append({"material": r["prompt"], "label": r["prompt"], "kind": "real",
                          "struct": round(r["ratio_center"] - r["ratio_seam"], 3),
                          "stratum": tag, "left": l, "right": r_,
                          "limg": imgs[l], "rimg": imgs[r_]})
    n_real = len(items)

    # 注意力检查：好图 vs 同一张图重糊。答错 >1 个即该次标注作废。
    for it in rng.sample(items, min(a.n_check, len(items))):
        good_im = Image.open(io.BytesIO(base64.b64decode(it["limg"]))).convert("RGB")
        blur_im = (good_im.resize((72, 72), Image.NEAREST)
                   .filter(ImageFilter.GaussianBlur(6))
                   .resize(good_im.size, Image.BILINEAR))

        def enc(im):
            buf = io.BytesIO()
            im.save(buf, "PNG")
            return base64.b64encode(buf.getvalue()).decode()

        gg = {"good": enc(good_im), "blur": enc(blur_im)}
        l, r_ = ("good", "blur") if rng.random() < 0.5 else ("blur", "good")
        items.append({"material": it["material"], "label": it["label"], "kind": "check",
                      "struct": 0.0, "stratum": it["stratum"], "left": l, "right": r_,
                      "limg": gg[l], "rimg": gg[r_]})

    rng.shuffle(items)
    print(f"共 {len(items)} 对：可比材质 {n_real}（"
          + "、".join(f"{t} {sum(1 for i in items if i['kind']=='real' and i['stratum']==t)}"
                      for t, _, _ in SETS)
          + f"），注意力检查 {len(items)-n_real}，缺图跳过 {missing}")
    print(f"左右平衡（接缝版在左 / 在右）：{sides[0]} / {sides[1]}")

    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)),
                     encoding="utf-8")
    print(f"写入 {a.out}  ({a.out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
