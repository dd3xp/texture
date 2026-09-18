#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""(M56) 判读器：参考图这一路条件到底动不动像素 —— 零 API、零训练、零判官。

=====================================================================================
本文件**盲写于任何读数存在之前**并与预注册同批提交。⛔ 跑完不许改判据、不许换配对。
=====================================================================================

## 问的是什么（判据全文在 docs/arch_progress.md 的 (M56) 节）

(M55) 判了 `REFS_NULL_32`（SDXL 参考图条件在 32px 上没抬 CLIP），并在第四节第 4 条留下一个
**未量过的假说**：「这一路条件对输出的影响本来就很小」。本轮直接量它：

  臂 A = 正确的参考嵌入（复用 (M55) 已出的图）
  臂 B = **错配**的参考嵌入（同检查点、同 seed、同命令，只把 prompt->嵌入 的对应关系错排）
  臂 A' = 把 A 的命令原样再跑一次（确定性操作检验）

同检查点 + 同 seed + 生成逐像素确定 ⇒ **零效应下的 pixfrac 是精确的 0**，不需要借门槛。

## 判据（主判据 = 32px 一档；16px 只登记）

  pixfrac(A,B)@32 <  0.01  -> REFCOND_INERT   这一路几乎不碰输出
  pixfrac(A,B)@32 >= 0.10  -> REFCOND_MOVES   这一路确实在改图
  其余                      -> REFCOND_WEAK    只登记，不下判

## 作废条件（任一触发则本轮无读数）

  VOID_NONDETERMINISTIC  pixfrac(A,A')@32 != 0（生成不确定 ⇒ 差值不能归给参考嵌入）
  VOID_REF_COLLAPSE      V_mat 提示词的参考嵌入两两余弦**中位数** >= 0.99（错配等于没换）
  VOID_PERM_BROKEN       错排有不动点，或两个 refs 文件的键集不同
  VOID_NO_DATA           任何一档的配对数为 0

## ⛔ 读法（写死，越界的一条都不许）

1. ⛔ 无论判成什么，都**不许**读成「参考图有用 / 没用」——(M55) 的 `REFS_NULL_32` 原样封着。
2. ⛔ pixfrac **不许**换算成胜率、⛔ 不许当门（(M44) 撤门、(M45) 连"跑前先验"也撤了）。
3. ⚑ `REFCOND_INERT` 的唯一许可用法：今后再想「把大模型语义接进小模型」，必须先加大**注入剂量**
   或换注入位置（(M41)：决定图改多少的是剂量）；⛔ 不许读成"这条路死了"。
4. ⚑ `REFCOND_MOVES` 的唯一许可用法：(M55) 第四节第 4 条那个假说**被证伪**，那条 null 只能读成
   「图动了，但没往 CLIP 喜欢的方向动」。⛔ 仍不许由此推任何胜率。

用法（本机，只需 numpy/PIL）：
  python analysis/arch/m56_read_refcond.py --selftest
  python analysis/arch/m56_read_refcond.py
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "remote_tmp" / "m56"
OUT = REPO / "experiments" / "m56_refcond.json"

# ⛔ 判据常数写死在这里，跑完不许动
THR_INERT = 0.01
THR_MOVES = 0.10
COS_COLLAPSE = 0.99

A_TAG, B_TAG, A2_TAG = "m56refs", "m56shuf", "m56rerun"
SIZES = ["32", "16"]
MAIN = "32"


def _stats(pairs):
    """与 (M39)(M41)(M42)(M43)(M45) 的尺子**逐字相同**。⛔ 一个字都不许改。"""
    import numpy as np
    if not pairs:
        return {"n": 0, "identical": 0, "pixfrac": None, "mae": None}
    fr, ae, ident = [], [], 0
    for _, a, b in pairs:
        a = a.astype(np.int32)
        b = b.astype(np.int32)
        d = np.abs(a - b).reshape(-1, 3)
        nd = int((d.sum(axis=-1) > 0).sum())
        fr.append(nd / d.shape[0])
        ae.append(float(d.mean()) / 255.0)
        if nd == 0:
            ident += 1
    return {"n": len(pairs), "identical": ident,
            "pixfrac": float(sum(fr) / len(fr)), "mae": float(sum(ae) / len(ae))}


def _load_pairs(da, db, problems):
    import numpy as np
    from PIL import Image
    na = {p.name for p in da.glob("*.png")} if da.is_dir() else set()
    nb = {p.name for p in db.glob("*.png")} if db.is_dir() else set()
    if na != nb:
        problems.append("file_sets_differ:%d vs %d" % (len(na), len(nb)))
    out = []
    for n in sorted(na & nb):
        a = np.array(Image.open(da / n).convert("RGB"))
        b = np.array(Image.open(db / n).convert("RGB"))
        if a.shape != b.shape:
            problems.append("shape_differ:" + n)
            continue
        out.append((n, a, b))
    return out


def verdict(pf):
    if pf is None:
        return "VOID_NO_DATA"
    if pf < THR_INERT:
        return "REFCOND_INERT"
    if pf >= THR_MOVES:
        return "REFCOND_MOVES"
    return "REFCOND_WEAK"


def voids(det_pf, det_n, refstats, per_size):
    """按固定次序查作废条件；返回全部触发项（⛔ 不许因为"只差一点"就放过）。"""
    v = []
    if det_n == 0:
        v.append("VOID_NO_DATA:determinism_pairs=0")
    elif det_pf != 0.0:
        v.append("VOID_NONDETERMINISTIC:pixfrac=%r" % det_pf)
    if refstats is None:
        v.append("VOID_NO_DATA:refstats_missing")
    else:
        if refstats.get("cos_median") is None:
            v.append("VOID_NO_DATA:cos_median_missing")
        elif refstats["cos_median"] >= COS_COLLAPSE:
            v.append("VOID_REF_COLLAPSE:cos_median=%.4f" % refstats["cos_median"])
        if refstats.get("fixed_points") is None:
            v.append("VOID_NO_DATA:fixed_points_missing")
        elif refstats["fixed_points"] > 0:
            v.append("VOID_PERM_BROKEN:fixed_points=%d" % refstats["fixed_points"])
        if refstats.get("keys_equal") is not True:
            v.append("VOID_PERM_BROKEN:keys_equal=%r" % refstats.get("keys_equal"))
    for s, st in per_size.items():
        if st["n"] == 0:
            v.append("VOID_NO_DATA:size%s_pairs=0" % s)
    return v


def selftest():
    import numpy as np
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print("  %-46s %s" % (name, "ok" if cond else "【禁】FAIL"))

    z = np.zeros((32, 32, 3), dtype=np.uint8)
    o = z.copy()
    o[0, 0] = [255, 0, 0]
    chk("T 同图 pixfrac=0", _stats([("x", z, z.copy())])["pixfrac"] == 0.0)
    chk("T 差一格 pixfrac=1/1024", abs(_stats([("x", z, o)])["pixfrac"] - 1 / 1024) < 1e-12)
    chk("T 空输入 pixfrac=None", _stats([])["pixfrac"] is None)
    chk("T 判 0.0 -> INERT", verdict(0.0) == "REFCOND_INERT")
    chk("T 判 0.0099 -> INERT", verdict(0.0099) == "REFCOND_INERT")
    chk("T 判 0.01 -> WEAK（>= 就不是 INERT）", verdict(0.01) == "REFCOND_WEAK")
    chk("T 判 0.0999 -> WEAK", verdict(0.0999) == "REFCOND_WEAK")
    chk("T 判 0.10 -> MOVES（贴线算过）", verdict(0.10) == "REFCOND_MOVES")
    chk("T 判 None -> VOID_NO_DATA", verdict(None) == "VOID_NO_DATA")
    good = {"cos_median": 0.5, "fixed_points": 0, "keys_equal": True}
    chk("T 全好 -> 无作废", voids(0.0, 250, good, {"32": {"n": 250}}) == [])
    chk("T 确定性破 -> VOID_NONDETERMINISTIC",
        any(x.startswith("VOID_NONDETERMINISTIC") for x in
            voids(1e-9, 250, good, {"32": {"n": 250}})))
    chk("T 嵌入塌缩 -> VOID_REF_COLLAPSE",
        any(x.startswith("VOID_REF_COLLAPSE") for x in
            voids(0.0, 250, {"cos_median": 0.99, "fixed_points": 0, "keys_equal": True},
                  {"32": {"n": 250}})))
    chk("T 不动点 -> VOID_PERM_BROKEN",
        any(x.startswith("VOID_PERM_BROKEN") for x in
            voids(0.0, 250, {"cos_median": 0.5, "fixed_points": 1, "keys_equal": True},
                  {"32": {"n": 250}})))
    chk("T 键集不同 -> VOID_PERM_BROKEN",
        any(x.startswith("VOID_PERM_BROKEN") for x in
            voids(0.0, 250, {"cos_median": 0.5, "fixed_points": 0, "keys_equal": False},
                  {"32": {"n": 250}})))
    chk("T 缺 refstats -> VOID_NO_DATA",
        any(x.startswith("VOID_NO_DATA") for x in voids(0.0, 250, None, {"32": {"n": 250}})))
    chk("T 某档零配对 -> VOID_NO_DATA",
        any("size16" in x for x in voids(0.0, 250, good, {"16": {"n": 0}})))
    chk("T 门槛没被改", (THR_INERT, THR_MOVES, COS_COLLAPSE) == (0.01, 0.10, 0.99))
    chk("T 主判据是 32px", MAIN == "32")
    print("[selftest] %d/%d" % (ok[0], ok[1]))
    return 0 if ok[0] == ok[1] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    root = Path(a.root)
    problems = []
    res = {"root": str(root), "thresholds": {"inert": THR_INERT, "moves": THR_MOVES,
                                             "cos_collapse": COS_COLLAPSE},
           "disclosure": "图在 V_mat(125 材质 x 2 次，验证集)上生成；本轮只读我们自己两臂的输出像素做差，"
                         "⛔ 不碰真人瓦片、不跑判官、不改任何已发表数字。引用必须同引本行。",
           "sizes": {}, "problems": []}

    rs_path = root / "m56_refstats.json"
    refstats = None
    if rs_path.is_file():
        refstats = json.loads(rs_path.read_text(encoding="utf-8"))
    else:
        problems.append("refstats_missing:%s" % rs_path)
    res["refstats"] = refstats

    for s in SIZES:
        probs = []
        pairs = _load_pairs(root / A_TAG / s, root / B_TAG / s, probs)
        res["sizes"][s] = _stats(pairs)
        problems += ["size%s:%s" % (s, p) for p in probs]

    dprobs = []
    dpairs = _load_pairs(root / A_TAG / MAIN, root / A2_TAG / MAIN, dprobs)
    det = _stats(dpairs)
    problems += ["determinism:%s" % p for p in dprobs]
    res["determinism_A_vs_Arerun"] = det

    res["voids"] = voids(det["pixfrac"], det["n"], refstats, res["sizes"])
    pf32 = res["sizes"][MAIN]["pixfrac"]
    pf16 = res["sizes"]["16"]["pixfrac"]
    res["verdict"] = res["voids"][0].split(":")[0] if res["voids"] else verdict(pf32)
    res["problems"] = problems

    cosmed = None if refstats is None else refstats.get("cos_median")
    res["predictions"] = {
        "P1_pf32_ge_0.10": {"pred": ">=0.10", "got": pf32,
                            "correct": pf32 is not None and pf32 >= THR_MOVES},
        "P2_pf16_gt_pf32": {"pred": "pixfrac@16 > pixfrac@32",
                            "got": None if None in (pf16, pf32) else pf16 - pf32,
                            "correct": None not in (pf16, pf32) and pf16 > pf32},
        "P3_cos_median_lt_0.9": {"pred": "<0.9", "got": cosmed,
                                 "correct": cosmed is not None and cosmed < 0.9},
    }

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> %s" % a.out)
    for s in SIZES:
        d = res["sizes"][s]
        print("  %spx  pixfrac=%-9s mae=%-9s n=%-5s 同图=%s"
              % (s, "None" if d["pixfrac"] is None else "%.4f" % d["pixfrac"],
                 "None" if d["mae"] is None else "%.4f" % d["mae"], d["n"], d["identical"]))
    print("  确定性检验 A vs A'@%s: pixfrac=%s n=%d 同图=%d"
          % (MAIN, det["pixfrac"], det["n"], det["identical"]))
    if refstats is not None:
        print("  参考嵌入：两两余弦 中位数=%s 均值=%s；错排不动点=%s；键集相同=%s"
              % (refstats.get("cos_median"), refstats.get("cos_mean"),
                 refstats.get("fixed_points"), refstats.get("keys_equal")))
    for k, v in res["predictions"].items():
        print("  %-22s 预测 %-26s -> %s" % (k, v["pred"], "对" if v["correct"] else "【禁】错"))
    print("  problems=%d %s" % (len(problems), problems))
    print("  作废条件触发：%s" % (res["voids"] or "无"))
    print("判决 = %s" % res["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
