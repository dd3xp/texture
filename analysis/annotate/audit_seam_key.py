"""`study_seam.html` 的验钥：从**标注者真正看到的像素**反推答案键。

为什么必须在标注之前做：这份盲比是方法三唯一的人工仪器（62 对），
而 `experiments/seamcrop*_demo/` 的 x9 图是**源图被磁盘清理后重渲染**出来的
（60 批一度用错种子，得 0/32）。若 `_seam_x9.png` 与 `_center_x9.png` 在某一步
被写反，或 HTML 的 `left/right` 与图不对应，那么用户几小时的标注会得出
**方向相反**的结论，而且事后无从分辨。

⚠ 不许用启发式验钥。此前拿"中心裁剪像不像"这类代理去验 study_crop 的键，
39 对里假报 26 对反钥（见记忆）。本脚本只用**定义本身**：
x9 是 16x16 瓦片的 3x3 平铺、8 倍 NEAREST 放大，可逐像素还原出那张瓦片；
`downsample.seam_stats` 是产出 JSON 时用的同一个函数，重算即可与 JSON 对照。
即"这张图是不是接缝对齐版"这一问，有**唯一确定的算术答案**，无需代理。

--- 判据（跑之前写下并 commit，事后不改）---

A **键正确**：57 个 real 项，被标 `seam` 的那侧算出的 `(seam, internal)` 与该材质
  JSON 的 `(seam_seam, int_seam)` 相对差 <= 1e-9，标 `center` 的一侧同理对
  `(seam_center, int_center)`。**任一项不符 -> REFUSE**，打印材质名与两侧实测值，
  仪器必须重建后才能标注。
B **可逆性**：每张 x9 必须是 3x3 块逐像素相同、且 8 倍 NEAREST 精确可逆
  （放大回去与原图逐位相同）。不满足则该项无法验钥，计入失败。
C **注意力检查有效**：每个 check 项的两侧必须一侧清晰一侧模糊——
  清晰侧的像素必须与某个 real 项的 x9 逐像素相同，模糊侧必须与之不同
  且相邻像素差的均值更低（模糊 = 高频更少）。5 个全过才算有效。
D **无退化对**：任一 real 对的两侧不得逐像素相同（否则该对不可答）。
E **左右平衡**：`seam` 在左与在右的次数差 <= 1。

全部通过 -> 打印 `INSTRUMENT-OK`，CSV 一到即可直接信任，不必再回头验钥。
任一不过 -> 打印 `INSTRUMENT-FAILED` 与具体项，**不要**开始标注。

跑法（纯本地，不需要 GPU、不导 torch）：`python analysis/annotate/audit_seam_key.py`
"""
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from downsample import seam_stats  # noqa: E402

HTML = ROOT / "experiments/annotate/study_seam.html"
JSONS = [ROOT / "experiments/seam_crop.json", ROOT / "experiments/seam_crop60.json"]
TOL = 1e-9
SCALE = 8          # x9 里每个瓦片像素占 8x8
TILE = 16          # 瓦片边长


def items_of(html: Path) -> list:
    txt = html.read_text(encoding="utf-8")
    m = re.search(r"const ITEMS\s*=\s*(\[.*?\]);", txt, re.S)
    if not m:
        m = re.search(r"(\[\{\"material\".*?\}\]);", txt, re.S)
    if not m:
        raise SystemExit("在 HTML 里找不到 ITEMS 数组")
    return json.loads(m.group(1))


def arr(b64: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))


def untile(a: np.ndarray) -> np.ndarray | None:
    """x9 -> 16x16 瓦片；不满足 B（3x3 相同 + 8x NEAREST 精确）则返回 None。"""
    h, w = a.shape[:2]
    if h != w or h != 3 * TILE * SCALE:
        return None
    b = TILE * SCALE
    blocks = [a[i * b:(i + 1) * b, j * b:(j + 1) * b] for i in range(3) for j in range(3)]
    if not all(np.array_equal(blocks[0], t) for t in blocks):
        return None
    t = blocks[0][::SCALE, ::SCALE]
    if not np.array_equal(np.repeat(np.repeat(t, SCALE, 0), SCALE, 1), blocks[0]):
        return None
    return t.astype(float)


def hf(a: np.ndarray) -> float:
    """高频能量代理：相邻像素绝对差的均值。"""
    a = a.astype(float)
    return float(np.abs(np.diff(a, axis=0)).mean() + np.abs(np.diff(a, axis=1)).mean())


def close(x: float, y: float) -> bool:
    return abs(x - y) <= TOL * max(1.0, abs(y))


def main():
    recs = {}
    for js in JSONS:
        for r in json.loads(js.read_text(encoding="utf-8")):
            if r.get("eligible"):
                recs[r["prompt"]] = r

    items = items_of(HTML)
    real = [it for it in items if it["kind"] == "real"]
    check = [it for it in items if it["kind"] == "check"]
    print(f"HTML 共 {len(items)} 项：real {len(real)}，check {len(check)}；"
          f"JSON 可比材质 {len(recs)}")

    fails, sides, real_pix = [], {"left": 0, "right": 0}, []
    for it in real:
        r = recs.get(it["material"])
        if r is None:
            fails.append(f"A/材质不在 JSON: {it['material']}")
            continue
        sides["left" if it["left"] == "seam" else "right"] += 1
        pair = {}
        for side, key in (("left", it["left"]), ("right", it["right"])):
            a = arr(it["limg"] if side == "left" else it["rimg"])
            pair[side] = a
            t = untile(a)
            if t is None:
                fails.append(f"B/x9 不可逆: {it['material']} ({side}, 标为 {key})")
                continue
            s, i_ = seam_stats(t)
            es, ei = r[f"seam_{key}"], r[f"int_{key}"]
            if not (close(s, es) and close(i_, ei)):
                fails.append(f"A/键不符: {it['material']} {side} 标为 {key}，"
                             f"实测 seam={s:.4f} int={i_:.4f}，"
                             f"JSON({key}) seam={es:.4f} int={ei:.4f}")
        if "left" in pair and "right" in pair and np.array_equal(pair["left"], pair["right"]):
            fails.append(f"D/两侧同图: {it['material']}")
        real_pix.append(pair.get("left"))
        real_pix.append(pair.get("right"))

    for it in check:
        g_side = "left" if it["left"] == "good" else "right"
        g = arr(it["limg"] if g_side == "left" else it["rimg"])
        b = arr(it["rimg"] if g_side == "left" else it["limg"])
        if not any(p is not None and p.shape == g.shape and np.array_equal(p, g)
                   for p in real_pix):
            fails.append(f"C/清晰侧不是任何一张 real 图: {it['material']}")
        if np.array_equal(g, b) or hf(b) >= hf(g):
            fails.append(f"C/模糊侧未更模糊: {it['material']} "
                         f"hf(good)={hf(g):.3f} hf(blur)={hf(b):.3f}")

    if abs(sides["left"] - sides["right"]) > 1:
        fails.append(f"E/左右不平衡: seam 在左 {sides['left']}，在右 {sides['right']}")

    print(f"左右平衡：seam 在左 {sides['left']} / 在右 {sides['right']}")
    print(f"分层：" + "、".join(
        f"{t} {sum(1 for it in real if it['stratum'] == t)}"
        for t in dict.fromkeys(it["stratum"] for it in real)))

    if fails:
        print(f"\nINSTRUMENT-FAILED（{len(fails)} 条）：")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"\nINSTRUMENT-OK：{len(real)}/{len(real)} 对逐像素验钥通过，"
          f"{len(check)} 个注意力检查有效。CSV 到了可直接信任。")


if __name__ == "__main__":
    main()
