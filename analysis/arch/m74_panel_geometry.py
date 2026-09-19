"""(M74) **刺激显示几何**审计：判官与人工盲比到底把一个纹素画成多大、画成什么形状。

零 GPU、零 API、零新读数：只对**已在盘的活件与页面**做确定性的几何测量。

为什么是这件事（与 GOAL.md 的关系）
----------------------------------
32px 上唯一还活着的认证工具是 (h) 人工盲比（(M64)(M66)(M67) 三候选表第一行，
页面 `10f78c8` 已备好、等用户点头）。在花掉用户 190 次判断**之前**，
有一个从没被问过、且零成本就能答的问题：**标注者/判官看到的那张图，几何上对不对？**
这不是"审计旧数字"，是 `GOAL.md` 防跑偏规则第 2 条允许的**挡住评测的缺陷**排查，
排查对象是**正要被使用的仪器**，不是任何已发表的读数。

量什么（三项，全部是确定性算术，无统计、无判据可调）
--------------------------------------------------
**(G1) `panel()` 的放大倍率随画布反比缩小。** `eval/judge_pairs.py:panel` 把任意 n×n 瓦片
一律渲染成 192×392 的面板（上 192×192 单张、下 192×192 的 3×3 平铺）。
倍率 = 192/n ⇒ 16px 得 12×、24px 得 8×、32px 得 6×。
⇒ **一个 32px 纹素占的显示面积只有 16px 纹素的 1/4**，而面板总尺寸一个像素都没跟着涨。

**(G2) 24px 的 3×3 子面板是非整数重采样。** 3n→192：16px 得 48→192＝4×（整）、
32px 得 96→192＝2×（整）、**24px 得 72→192＝2.667×（非整）** ⇒ NEAREST 会把一部分源列
复制 3 次、另一部分复制 2 次 ⇒ 纹素网格**不均匀**。像素画的"格子是不是方的、周期是不是齐的"
正是被判的性质之一。

**(G3) 页面 CSS 把面板压成正方形。** `analysis/annotate/task_template.html` 写死
`img{width:320px;height:320px}`，而面板是 192×392（长宽比 0.49）
⇒ 横向放大 320/192、纵向缩小 320/392 ⇒ **纵横比畸变 = (320/192)/(320/392) = 392/192 ≈ 2.04**，
方形纹素被显示成约 2:1 的扁矩形。

⛔ 禁止读法（写在跑之前）
------------------------
1. ⛔⛔ **本轮不重判、不撤回任何已发表判决。** 47 条判官臂的读数原样成立；
   本轮**一个胜负数字都不重算**。几何缺陷对两臂是**对称**的（同一个 `panel`），
   ⇒ 它可能压低**可解率**，⛔ 但不构成任何**方向性**偏倚的证据。
2. ⛔ **不许改 `eval/judge_pairs.py`**（活件；47 条已发表臂全靠它逐字不变才可比）。
   本脚本只 `import`，一个字不改。
3. ⚠⚠ **披露：第四项（可解率随画布）不是盲判。** 账本早已写着"32px 校准不过关、
   16px 可解率 60% 出头"，我看过。⇒ 它在本轮**只登记、不下判决**，且只在
   **同一对方法跨尺寸**的家族内看（控制住"比的是谁"这个混淆），仍然只能当方向。
4. ⛔ (G1)(G2)(G3) 都**不授权**任何新臂、不授权重跑任何判官臂、不授权解禁 32px 判官。

产物：`experiments/m74_panel_geometry.json`。
用法：
    python analysis/arch/m74_panel_geometry.py --selftest
    python analysis/arch/m74_panel_geometry.py --out experiments/m74_panel_geometry.json
"""
import argparse
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from judge_pairs import panel                     # noqa: E402  活件，只 import 不改

PANEL_W, PANEL_H = 192, 392                        # panel() 写死的画布
SIZES = (16, 24, 32)


# ---------------------------------------------------------------- 基本几何
def run_lengths(mag_from: int, mag_to: int):
    """NEAREST 把 mag_from 列放大到 mag_to 列时，每个源列实得的输出列数（多重集）。

    PIL 的 NEAREST 对输出列 j 取源列 floor((j+0.5)*from/to)，这里按同一条规则数。
    """
    src = [int((j + 0.5) * mag_from / mag_to) for j in range(mag_to)]
    cnt = [0] * mag_from
    for s in src:
        cnt[min(s, mag_from - 1)] += 1
    return cnt


def geometry(n: int):
    """对画布 n 报 panel() 的显示几何（不碰任何真料，纯算术 + 一次合成渲染自检）。"""
    one_rl = run_lengths(n, PANEL_W)               # 单张子面板：n -> 192
    tri_rl = run_lengths(3 * n, PANEL_W)           # 3x3 子面板：3n -> 192
    return {
        "size": n,
        "mag_single": PANEL_W / n,
        "mag_tiled": PANEL_W / (3 * n),
        "single_integer": PANEL_W % n == 0,
        "tiled_integer": PANEL_W % (3 * n) == 0,
        "single_runs": sorted(set(one_rl)),
        "tiled_runs": sorted(set(tri_rl)),
        "texel_area_single_px": (PANEL_W / n) ** 2,
        "texel_area_ratio_vs16": ((PANEL_W / n) ** 2) / ((PANEL_W / 16) ** 2),
    }


def render_check(n: int):
    """把一张已知瓦片真的过一遍活件 `panel()`，核对上面的算术与实际字节一致。

    合成刺激：逐列递增的灰阶（每一列都与邻列不同）⇒ 输出里每个源列的宽度可直接数出来。
    """
    t = np.zeros((n, n, 3), np.uint8)
    for x in range(n):
        t[:, x, :] = int(255 * (x + 0.5) / n)
    im = Image.open(io.BytesIO(base64.b64decode(panel(t))))
    a = np.asarray(im.convert("RGB"))
    assert im.size == (PANEL_W, PANEL_H), f"面板尺寸变了：{im.size}"

    def runs(row):
        w, cur = [], 1
        for j in range(1, PANEL_W):
            if (row[j] == row[j - 1]).all():
                cur += 1
            else:
                w.append(cur)
                cur = 1
        return w + [cur]

    one = runs(a[0])                                # 单张子面板的第 0 行
    tri = runs(a[200])                              # 3x3 子面板的第 0 行（panel 把它贴在 y=200）
    return {"size": n, "panel_wh": list(im.size),
            "measured_runs": sorted(set(one)), "n_runs": len(one),
            "measured_runs_tiled": sorted(set(tri)), "n_runs_tiled": len(tri)}


# ---------------------------------------------------------------- 页面侧
CSS_IMG = re.compile(r"\bimg\{[^}]*?width:(\d+)px;height:(\d+)px", re.S)


def page_geometry(page: Path):
    """从一张已生成的标注页里量出：面板真实尺寸、CSS 显示框、纵横比畸变。"""
    txt = page.read_text(encoding="utf-8")
    m = CSS_IMG.search(txt)
    css = [int(m.group(1)), int(m.group(2))] if m else None
    mi = re.search(r'"limg"\s*:\s*"([A-Za-z0-9+/=]+)"', txt)
    if mi is None:
        return {"page": page.name, "css_box": css, "img_wh": None}
    im = Image.open(io.BytesIO(base64.b64decode(mi.group(1))))
    out = {"page": page.name, "css_box": css, "img_wh": list(im.size)}
    if css:
        sx, sy = css[0] / im.size[0], css[1] / im.size[1]
        out["scale_x"], out["scale_y"] = sx, sy
        out["aspect_distortion"] = sx / sy          # 1.0 = 纹素仍是方的
    return out


# ---------------------------------------------------------------- 只登记：可解率随画布
def resolve_by_size(d: Path):
    """从已落盘的 judge_full_*.json 读可解率（decided / 已作答），按尺寸与方法对分组。

    ⛔ 只读 `decided`/`api_fail`/记录条数，**一个胜负字段都不读**（可解率与谁赢无关）。
    """
    rows = []
    for p in sorted(d.glob("judge_full_*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        tag = o.get("tag", p.stem)
        m = re.search(r"_(\d+)(?:_|$)", tag)
        if not m:
            continue
        n_rec = len(o.get("records", []))
        asked = n_rec - int(o.get("api_fail", 0))
        if asked <= 0:
            continue
        rows.append({"tag": tag, "size": int(m.group(1)),
                     "decided": int(o["decided"]), "asked": asked,
                     "resolve_rate": o["decided"] / asked})
    return rows


def family(tag: str):
    """把臂名归成"同一对方法"的家族：去掉尺寸与集合后缀、去掉 TRD 的尺寸数字。"""
    t = re.sub(r"_(16|24|32)(_.*)?$", "", tag)
    t = re.sub(r"TRD\d+c?", "TRD", t)
    t = re.sub(r"_rr4|_rr2", "", t)
    return t


def cross_size(rows):
    """只保留**跨尺寸出现过**的家族，family -> {size: 该尺寸下的可解率均值}。"""
    byf = {}
    for r in rows:
        byf.setdefault(family(r["tag"]), {}).setdefault(r["size"], []).append(r["resolve_rate"])
    out = {}
    for f, d in byf.items():
        if len(d) >= 2:
            out[f] = {str(s): float(np.mean(v)) for s, v in sorted(d.items())}
    return out


# ---------------------------------------------------------------- selftest
def selftest():
    ok = 0

    def chk(c, msg):
        nonlocal ok
        assert c, msg
        ok += 1

    # run_lengths：整数倍时每个源单元宽度一致
    chk(run_lengths(16, 192) == [12] * 16, "16->192 应当每列 12")
    chk(run_lengths(48, 192) == [4] * 48, "48->192 应当每列 4")
    chk(run_lengths(96, 192) == [2] * 96, "96->192 应当每列 2")
    chk(sum(run_lengths(72, 192)) == 192, "72->192 总宽必须是 192")
    chk(set(run_lengths(72, 192)) == {2, 3}, "72->192 必须出现 2 与 3 两种宽度")
    chk(run_lengths(24, 192) == [8] * 24, "24->192 单张是整数倍")
    chk(len(run_lengths(5, 7)) == 5, "run_lengths 长度 = 源单元数")
    chk(min(run_lengths(5, 7)) >= 1, "每个源单元至少占 1 列")

    # geometry：三档的倍率与整除性
    g = {n: geometry(n) for n in SIZES}
    chk(g[16]["mag_single"] == 12 and g[24]["mag_single"] == 8 and g[32]["mag_single"] == 6,
        "单张倍率应为 12/8/6")
    chk(g[16]["tiled_integer"] and g[32]["tiled_integer"] and not g[24]["tiled_integer"],
        "只有 24px 的 3x3 子面板是非整数倍")
    chk(abs(g[32]["texel_area_ratio_vs16"] - 0.25) < 1e-12, "32px 纹素面积应为 16px 的 1/4")
    chk(abs(g[24]["texel_area_ratio_vs16"] - 4 / 9) < 1e-12, "24px 纹素面积应为 16px 的 4/9")
    chk(g[16]["texel_area_ratio_vs16"] == 1.0, "16px 自比应为 1")

    # render_check：真的过一遍活件，算术与字节必须对上
    for n in SIZES:
        r = render_check(n)
        chk(r["panel_wh"] == [PANEL_W, PANEL_H], f"{n}px 面板尺寸")
        chk(r["n_runs"] == n, f"{n}px 单张子面板应恰好数出 {n} 个源列")
        chk(r["measured_runs"] == [PANEL_W // n], f"{n}px 单张子面板列宽应恒为 {PANEL_W // n}")
        # (G2) 的实测版：3x3 子面板的列宽要么只有一种（整除），要么真的出现两种（24px）
        chk(r["measured_runs_tiled"] == sorted(set(run_lengths(3 * n, PANEL_W))),
            f"{n}px 的 3x3 子面板实测列宽与算术不符：{r['measured_runs_tiled']}")
    chk(len(render_check(24)["measured_runs_tiled"]) == 2,
        "24px 的 3x3 子面板必须实测到两种列宽（非整数重采样）")
    chk(len(render_check(16)["measured_runs_tiled"]) == 1, "16px 的 3x3 子面板列宽应一致")

    # CSS 解析
    chk(CSS_IMG.search("x img{width:320px;height:320px;image-rendering:pixelated}").groups()
        == ("320", "320"), "CSS 正则应取到 320/320")
    chk(CSS_IMG.search("img{color:red}") is None, "没有宽高时不该误匹配")

    # family 归并
    chk(family("TRD32_rr4_vs_B2_32") == family("TRD16c_rr4_vs_B2_16"), "TRD vs B2 应归一族")
    chk(family("TRD24_rr4_vs_B7_24") == family("TRD16_vs_B7_16"), "TRD vs B7 应归一族")
    chk(family("B2_vs_B3_16_sdpixl_subset") == family("B2_vs_B3_32_sdpixl_subset"),
        "B2 vs B3 应归一族")
    chk(family("TRD16_vs_B2_16") != family("TRD16_vs_B5_16"), "不同对手不许归一族")

    # cross_size 只保留跨尺寸家族
    rows = [{"tag": "A_vs_B_16", "size": 16, "resolve_rate": 0.6},
            {"tag": "A_vs_B_32", "size": 32, "resolve_rate": 0.4},
            {"tag": "C_vs_D_16", "size": 16, "resolve_rate": 0.9}]
    cs = cross_size(rows)
    chk(len(cs) == 1 and "16" in list(cs.values())[0], "只有跨尺寸的家族该留下")
    chk(list(cs.values())[0]["32"] == 0.4, "跨尺寸家族的读数要对")

    # 纵横比畸变的算术
    chk(abs((320 / 192) / (320 / 392) - 392 / 192) < 1e-12, "畸变 = 392/192")

    print(f"selftest {ok}/{ok} 全过")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pages", type=Path, default=ROOT / "experiments/annotate")
    ap.add_argument("--judge", type=Path, default=ROOT / "experiments")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    geo = [geometry(n) for n in SIZES]
    ren = [render_check(n) for n in SIZES]
    pages = [page_geometry(p) for p in sorted(a.pages.glob("study_*.html"))]
    rows = resolve_by_size(a.judge)
    cs = cross_size(rows)
    by_size = {}
    for r in rows:
        by_size.setdefault(str(r["size"]), []).append(r["resolve_rate"])
    by_size = {k: {"n_arms": len(v), "mean": float(np.mean(v))} for k, v in sorted(by_size.items())}

    out = {"panel_wh": [PANEL_W, PANEL_H], "geometry": geo, "render_check": ren,
           "pages": pages, "resolve_by_size_all_arms": by_size,
           "resolve_cross_size_families": cs, "n_judge_arms": len(rows)}
    if a.out:                                       # 落盘一律在打印之前
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("=== (G1) panel() 的显示几何（面板恒为 192x392）")
    for g in geo:
        print(f"  {g['size']}px：单张 {g['mag_single']:g}x（整除 {g['single_integer']}）  "
              f"3x3 {g['mag_tiled']:.4g}x（整除 {g['tiled_integer']}，列宽 {g['tiled_runs']}）  "
              f"每纹素 {g['texel_area_single_px']:.0f} 显示像素 = 16px 的 {g['texel_area_ratio_vs16']:.3f}")
    print("=== 活件复核（合成刺激真过一遍 panel）")
    for r in ren:
        print(f"  {r['size']}px：面板 {r['panel_wh']}，单张子面板数出 {r['n_runs']} 列，列宽 {r['measured_runs']}")
    print("=== (G3) 标注页的 CSS 显示框")
    for p in pages:
        if p.get("img_wh"):
            print(f"  {p['page']}：图 {p['img_wh']}  CSS 框 {p['css_box']}  "
                  f"纵横比畸变 {p.get('aspect_distortion', float('nan')):.4g}")
        else:
            print(f"  {p['page']}：CSS 框 {p['css_box']}  （页面里没取到 limg）")
    print(f"=== 只登记：可解率随画布（{len(rows)} 条已发表 full 臂，⛔ 不下判决）")
    for k, v in by_size.items():
        print(f"  全部臂 {k}px：n={v['n_arms']}  可解率均值 {v['mean']:.1%}")
    for f, d in sorted(cs.items()):
        print(f"  同对方法跨尺寸 [{f}]：" + "  ".join(f"{s}px {r:.1%}" for s, r in d.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
