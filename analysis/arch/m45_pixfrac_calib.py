#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""(M45) 给 pixfrac 这把尺子补**已知真效应**的标定点 —— 零 API、零 GPU、零训练。

=====================================================================================
本文件写于**任何读数存在之前**并单独提交 ＝ 预注册。⛔ 跑完不许改预测、不许换配对。
=====================================================================================

## 为什么做这件事（与它会量出什么无关）

(M44) 把下一步收敛成「**去找预期 ≥65% 的架构级杠杆**」，但**选杠杆时我们唯一的零成本
先验就是 pixfrac**。而这把尺子现在的五个点里，**判官显著的只有一个**（D37=0.8173），
而那一个是「欠训 ck3k vs 训练完 ck20k」——**根本不是一次架构改动**。
⇒ 「多大的 pixfrac 对应一个真的 ≥65% 效应」这个问题，**目前没有任何数据点**。

`remote_tmp/abl16/` 里躺着 (M16) 16px 消融表的五条臂的图，其判决**早已发表并冻结**：
  - `AB_pal`（去掉调色板记忆库）126/190 = **66%**，p=8e-06，(M18) 族级 [.574,.747]、
    LOFO 118/118 零翻侧 ＝ `W1_robust` —— **全项目唯一一个"架构部件"级别的 ≥65% 已知效应**。
  - `AB_ct` 56%、`AB_ex` 50% ＝ **真平局**（可解率 57%/64%，地板才 16%）。
  - `AB_xm`、`AB_rr` **没有 full**（试点不过门）⇒ ⛔ 判决不明，**只登记读数、不当标定点**。
⇒ 把同一把尺子架到这五条臂上，**一次拿到 1 个 ≥65% 的点 + 2 个真平局的点**。

## ⛔⛔ 四条纪律（跑之前写死）

1. ⛔ **不恢复 pixfrac 为门。** (M44)(`522297e`) 的撤门理由 (a)(b) 与任何判决无关，
   **无论本轮量出什么都不许翻案**。本轮产出的一切只是**参考读数**。
2. ⛔ **不许挪 D37=0.8173 / D38=0.6608 / D40=0.6431**，⛔ 不许换尺子（`mae` 给的次序相反）。
3. ⚠⚠ **E_mat 披露**：这五条臂的图是在 **E_mat(272，测试集)** 上生成的。本轮
   **只读我们自己两个臂的输出像素做差**，⛔ **不碰任何真人瓦片、不跑判官、不产生新判决**，
   也**不改动任何已发表数字**。但今后引用本轮读数**必须同引这一行**。
4. ⛔ **`AB_xm` / `AB_rr` 的读数不许当标定点**（它们没有 full 判决）。

## 跑前预测（⛔ 写死，跑完照录）

⚠ 交代前科：**跑前猜 pixfrac 已连续三次全错，且三次都是往高了猜**
（(M41) 猜 ≥0.8173 得 0.3493；(M42) 猜 ≥0.8173 得 0.6643；(M43) 猜 ≥0.8173 得 0.7907）。
下面这条预测是在**知道自己有系统性高估**的情况下写的：

- **(P1) `AB_pal` 的 pixfrac ≥ 0.8173**（＝落在 D37 那条 GO 线之上）。
  理由：它整个去掉一个承重部件，剂量应当大于「换 CFG / 换代次 / 换训练数据」。
- **(P2) 次序 `AB_pal` > `AB_ct`，且 `AB_pal` > `AB_ex`**（66% 的臂比两条真平局臂改图更多）。
- **(P3) `AB_ex` 的 pixfrac ≤ 0.7907**（50% ＝ 判官完全分不出，应当低于 (M43) 那个 NULL 点）。

⚑ 读法也写死，两个方向都写：
  - (P1)+(P2) 都成立 ⇒ 尺子在「真效应 ≥65%」这一档上**首次有了正对照**，
    今后选杠杆可以拿 0.8173 当**参考线**（⛔ 仍然只是参考，不是门）。
  - (P1) 或 (P2) 不成立 ⇒ **尺子与真效应在架构改动上不同向** ⇒ ⛔ **今后选杠杆不许再用 pixfrac**，
    (M45) 必须另找跑前先验。⛔ 不许因为不喜欢这个结果就换配对或换尺子。

用法（本机，零依赖除 numpy/PIL）：
  python analysis/arch/m45_pixfrac_calib.py --selftest
  python analysis/arch/m45_pixfrac_calib.py
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "remote_tmp" / "abl16"
OUT = REPO / "experiments" / "m45_pixfrac_calib.json"
SIZE = "16"
BASE = "TRD16c_rr4"

# ⛔ 三个锚点写死，本文件只读不写
D37, D38, D40 = 0.8173, 0.6608, 0.6431
D_SEED, D_DATA = 0.6643, 0.79075

# 臂 -> (已发表真效应, 是否可当标定点, 备注)
ARMS = {
    "AB_pal_rr4": (0.663, True, "126/190=66% p=8e-06 族级[.574,.747] W1_robust"),
    "AB_ct_rr4": (0.56, True, "真平局（可解率 57%，地板 16%）"),
    "AB_ex_rr4": (0.50, True, "真平局（可解率 64%，地板 16%）"),
    "AB_xm_rr4": (None, False, "无 full（试点不过门）-> 只登记，⛔ 不当标定点"),
}


def _stats(pairs):
    """与 (M39)(M41)(M42)(M43) 的尺子**逐字相同**。⛔ 一个字都不许改。"""
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
    na = {p.name for p in da.glob("*.png")}
    nb = {p.name for p in db.glob("*.png")}
    if na != nb:
        problems.append("file_sets_differ:%s vs %s" % (len(na), len(nb)))
    out = []
    for n in sorted(na & nb):
        a = np.array(Image.open(da / n).convert("RGB"))
        b = np.array(Image.open(db / n).convert("RGB"))
        if a.shape != b.shape:
            problems.append("shape_differ:" + n)
            continue
        out.append((n, a, b))
    return out


def band(d):
    """只报它落在哪条带里，⛔ 不做任何判决。"""
    if d is None:
        return "NO_DATA"
    if d >= D37:
        return "ABOVE_D37"
    if d <= D38:
        return "BELOW_D38"
    return "BETWEEN"


def selftest():
    import numpy as np
    ok = [0, 0]

    def chk(name, cond):
        ok[1] += 1
        ok[0] += bool(cond)
        print("  %-42s %s" % (name, "ok" if cond else "【禁】FAIL"))

    z = np.zeros((16, 16, 3), dtype=np.uint8)
    o = z.copy()
    o[0, 0] = [255, 0, 0]
    chk("T 同图 pixfrac=0", _stats([("x", z, z.copy())])["pixfrac"] == 0.0)
    chk("T 同图 identical=1", _stats([("x", z, z.copy())])["identical"] == 1)
    chk("T 差一格 pixfrac=1/256", abs(_stats([("x", z, o)])["pixfrac"] - 1 / 256) < 1e-12)
    chk("T 差一格 identical=0", _stats([("x", z, o)])["identical"] == 0)
    chk("T 空输入 n=0", _stats([])["n"] == 0)
    chk("T 空输入 pixfrac=None", _stats([])["pixfrac"] is None)
    chk("T mae 归一化", abs(_stats([("x", z, o)])["mae"] - (255 / 3 / 256) / 255) < 1e-9)
    chk("T band(0.8173)=ABOVE_D37", band(D37) == "ABOVE_D37")
    chk("T band(0.8174)=ABOVE_D37", band(0.8174) == "ABOVE_D37")
    chk("T band(0.8172)=BETWEEN", band(0.8172) == "BETWEEN")
    chk("T band(0.6608)=BELOW_D38", band(D38) == "BELOW_D38")
    chk("T band(0.6609)=BETWEEN", band(0.6609) == "BETWEEN")
    chk("T band(None)=NO_DATA", band(None) == "NO_DATA")
    chk("T 锚点没被改", (D37, D38, D40) == (0.8173, 0.6608, 0.6431))
    chk("T AB_xm 不当标定点", ARMS["AB_xm_rr4"][1] is False)
    chk("T AB_pal 是标定点", ARMS["AB_pal_rr4"][1] is True)
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
    res = {"root": str(root), "base": BASE, "size": SIZE,
           "anchors": {"D37_sig": D37, "D38_null": D38, "D40_null": D40,
                       "D_seed_constructive_zero": D_SEED, "D_data_null": D_DATA},
           "disclosure": "图在 E_mat(272，测试集)上生成；本轮只读两臂自己的输出像素做差，"
                         "不碰真人瓦片、不跑判官、不改任何已发表数字。引用必须同引本行。",
           "arms": {}, "problems": []}

    for arm, (eff, usable, note) in ARMS.items():
        probs = []
        pairs = _load_pairs(root / BASE / SIZE, root / arm / SIZE, probs)
        st = _stats(pairs)
        res["arms"][arm] = {"published_effect": eff, "calibration_point": usable,
                            "note": note, "band": band(st["pixfrac"]), **st}
        res["problems"] += ["%s:%s" % (arm, p) for p in probs]

    pal = res["arms"]["AB_pal_rr4"]["pixfrac"]
    ct = res["arms"]["AB_ct_rr4"]["pixfrac"]
    ex = res["arms"]["AB_ex_rr4"]["pixfrac"]
    res["predictions"] = {
        "P1_AB_pal_ge_D37": {"pred": ">=0.8173", "got": pal,
                             "correct": pal is not None and pal >= D37},
        "P2_pal_gt_ct_and_ex": {"pred": "pal>ct and pal>ex",
                                "correct": None not in (pal, ct, ex) and pal > ct and pal > ex},
        "P3_AB_ex_le_D_data": {"pred": "<=0.79075", "got": ex,
                               "correct": ex is not None and ex <= D_DATA},
    }
    res["verdict_on_ruler"] = (
        "RULER_HAS_POSITIVE_CONTROL"
        if res["predictions"]["P1_AB_pal_ge_D37"]["correct"]
        and res["predictions"]["P2_pal_gt_ct_and_ex"]["correct"]
        else "RULER_NOT_ALIGNED_ON_ARCH_CHANGES")

    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-> %s" % a.out)
    for arm, d in res["arms"].items():
        print("  %-12s pixfrac=%-8s mae=%-8s n=%-4s 同图%-4s %-10s 真效应=%s %s"
              % (arm.replace("_rr4", ""),
                 "None" if d["pixfrac"] is None else "%.4f" % d["pixfrac"],
                 "None" if d["mae"] is None else "%.4f" % d["mae"],
                 d["n"], d["identical"], d["band"],
                 d["published_effect"], "" if d["calibration_point"] else "(不当标定点)"))
    for k, v in res["predictions"].items():
        print("  %-22s 预测 %-16s -> %s" % (k, v["pred"], "对" if v["correct"] else "【禁】错"))
    print("  problems=%d %s" % (len(res["problems"]), res["problems"]))
    print("对尺子的结论 = %s（【禁】无论如何都不恢复为门，见文件头纪律 1）" % res["verdict_on_ruler"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
