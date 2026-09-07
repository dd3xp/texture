"""第 30 轮一次性审计：两份盲比 HTML 的答案钥匙与仪器完整性。

审什么：CSV 标签的意义完全取决于 HTML 里 items 的 left/right 字段是否如实
指向对应图像。若钥匙错乱，用户标注到了也是废数据（甚至是反的）。

1. study_ab60.html：artist 侧可由 dataset_k16.json 精确重建 -> 逐对像素级验钥
2. 两份 HTML 的 check 对：blur 侧应实际更糊（梯度能量）
3. study_crop.html：统计验钥（正钥 vs 反钥的中心裁剪相关）——
   ⚠ 第 30 轮实跑证明此启发式**有偏不可用**（zoom 压方差抬相关，26/39 假报反钥）；
   crop 的权威验钥见 spotcheck_crop_key.py（GPU 种子确定重生成，39/39 逐像素 KEY-OK）。
   此段保留仅作反面教材。
4. 左右平衡、字段完备性、计数

第 30 轮实跑：ab60 60/60 精确验钥、平衡 30/30、check 6/6；crop 平衡 19/20、check 3/3。
"""
import base64, io, json, re
from pathlib import Path

import numpy as np
from PIL import Image


def items_of(p):
    s = Path(p).read_text(encoding="utf-8")
    return json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", s, re.S).group(1))


def img(b):
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")).astype(float)


def grad_energy(a):
    g = a.mean(-1)
    return float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())


def summary(name, items):
    from collections import Counter
    kinds = Counter(i["kind"] for i in items)
    bal = Counter()
    for i in items:
        bal[tuple(sorted([i["left"], i["right"]])) + (i["left"],)] += 1
    print(f"\n=== {name}: {len(items)} 对  kinds={dict(kinds)} ===")
    for k, v in sorted(bal.items()):
        print(f"  左右平衡 {k[0]}/{k[1]}: 左={k[2]} 计 {v}")
    miss = [j for j, i in enumerate(items)
            if not all(f in i for f in ("material", "label", "kind", "left", "right", "limg", "rimg"))]
    print(f"  字段缺失项: {miss if miss else '无'}")
    return kinds


def check_blur(name, items):
    bad = 0
    for j, it in enumerate(items):
        if it["kind"] != "check":
            continue
        L, R = img(it["limg"]), img(it["rimg"])
        e = {it["left"]: grad_energy(L), it["right"]: grad_energy(R)}
        ok = e["good"] > e["blur"]
        if not ok:
            bad += 1
            print(f"  ! {name} check idx={j} 钥匙可疑: good={e['good']:.2f} blur={e['blur']:.2f}")
    n = sum(1 for i in items if i["kind"] == "check")
    print(f"  {name} check 对 blur 侧确实更糊: {n - bad}/{n}")


# ---------- study_ab60: 精确验钥 ----------
ab = items_of("experiments/annotate/study_ab60.html")
summary("study_ab60", ab)

ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
ref = {}
for s in ds["samples"]:
    if s["size"] == 16 and s["split"] == "test" and s["material"] not in ref:
        ref[s["material"]] = s

exact, wrong, nomat = 0, 0, 0
base_in_pal = 0
for j, it in enumerate(ab):
    if it["kind"] != "real":
        continue
    m = it["material"]
    if m not in ref:
        nomat += 1
        print(f"  ! real idx={j} 材质 {m} 不在 test ref 中")
        continue
    s = ref[m]
    pal = np.array(s["palette"], np.uint8)
    art = pal[np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)].astype(float)
    L, R = img(it["limg"]), img(it["rimg"])
    side = {it["left"]: L, it["right"]: R}
    a_ok = np.array_equal(side["artist"], art)
    b_differs = not np.array_equal(side["baseline"], art)
    # baseline 应只用 artist 调色板颜色（构造即 pal[argmin]）
    bp = side["baseline"].reshape(-1, 3)
    inpal = all(any(np.array_equal(c, p) for p in pal.astype(float)) for c in bp)
    if a_ok and b_differs:
        exact += 1
    else:
        wrong += 1
        print(f"  ! real idx={j} {m}: artist匹配={a_ok} baseline不同={b_differs}")
    if inpal:
        base_in_pal += 1
nreal = sum(1 for i in ab if i["kind"] == "real")
print(f"  artist 侧像素级验钥: {exact}/{nreal} 精确匹配（错 {wrong}，缺材质 {nomat}）")
print(f"  baseline 侧颜色全在 artist 调色板内: {base_in_pal}/{nreal}")
check_blur("study_ab60", ab)

# ---------- study_crop: 统计验钥 ----------
cr = items_of("experiments/annotate/study_crop.html")
summary("study_crop", cr)


def center_zoom(tile, frac):
    im = Image.fromarray(tile.astype(np.uint8))
    c = 16 * (1 - frac) / 2
    return np.asarray(im.resize((16, 16), Image.BILINEAR, box=(c, c, 16 - c, 16 - c))).astype(float)


def corr(a, b):
    x, y = a.ravel(), b.ravel()
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d else 0.0


right_key, swap_better, ties = 0, 0, 0
diffs = []
for j, it in enumerate(cr):
    if it["kind"] != "real":
        continue
    frac = 1.0 / it["struct"]
    assert 0 < frac < 1, f"idx={j} struct={it['struct']}"
    L, R = img(it["limg"]), img(it["rimg"])
    side = {it["left"]: L, it["right"]: R}
    r_correct = corr(center_zoom(side["before"], frac), side["after"])
    r_swapped = corr(center_zoom(side["after"], frac), side["before"])
    diffs.append(r_correct - r_swapped)
    if r_correct > r_swapped:
        right_key += 1
    elif r_correct < r_swapped:
        swap_better += 1
        print(f"  ? real idx={j} {it['material']}: r_correct={r_correct:.3f} < r_swapped={r_swapped:.3f}")
    else:
        ties += 1
nreal_c = sum(1 for i in cr if i["kind"] == "real")
print(f"  裁剪几何验钥（正钥相关 > 反钥）: {right_key}/{nreal_c}"
      f"（反向 {swap_better}，平 {ties}），中位差 {np.median(diffs):+.3f}")
check_blur("study_crop", cr)

# check 对材质须来自 real 材质（builder 设计如此）
real_mats = {i["material"] for i in cr if i["kind"] == "real"}
chk_bad = [i["material"] for i in cr if i["kind"] == "check" and i["material"] not in real_mats]
print(f"  check 对材质均出自 real 材质: {'是' if not chk_bad else chk_bad}")
