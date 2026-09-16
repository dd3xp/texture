#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""(M39) 跑前筛子的判读器 —— **盲写于 ret* 产物存在之前**（预注册 eval/retnn16_screen.sh）。

这把尺子只回答一个问题：**候选杠杆 `--ret_nname` 改出来的产物差别，够不够judge 判得动？**
⛔ 它不回答"哪个 N 更好"，⛔ 它的任何数字都不是质量证据，⛔ 不许写进任何主张。

尺子：d(A,B) = 125 个材质上「A、B 的 `_rr4` 瓦片逐像素不同的比例」的均值（0..1）。
锚点（全部来自已判过的臂）：
  D37 = d(ck3k_rr4, ck20k_rr4)   <- (M37) 判官判**显著**（族级 [0.217,0.438]，W1_robust）
  D38 = d(cfg25_rr4, cfg15_rr4)  <- (M38) 判官判 **CFG_NULL**（族级 [0.4827,0.6970] 含 0.5）
  D0  = d(ret100_rr4, cfg15_rr4) <- 零点（两条命令逐字相同）

预注册判据（与 eval/retnn16_screen.sh 第三节逐字对应）：
  (C1) D37 > D38            否则 METER_UNCALIBRATED
  (C2) D0 <= D38/10         否则 METER_UNCALIBRATED
  (V)  d_c >= D37 -> SCREEN_GO；d_c <= D38 -> SCREEN_NOGO；两者之间 -> SCREEN_AMBIGUOUS
  (P)  两条都 GO 只许提拔 d 大的那条（设计选择，不是证据）

缺数据时**必须**报 VOID_NO_DATA 并写清已查几项（⛔ 不许拿空集冒充"量过没事"）。
用法：
  python analysis/arch/m39_diff_meter.py --selftest
  python analysis/arch/m39_diff_meter.py --ck /tmp/ckab16 --cfg /tmp/cfgab16 --ret /tmp/retnn16 \
         --out /tmp/m39_retnn_screen.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

SIZE = "16"


def _stats(pairs):
    """pairs: [(name, arrA, arrB)] -> 逐像素差别统计。"""
    if not pairs:
        return {"n": 0, "identical": 0, "pixfrac": None, "mae": None}
    fr, ae, ident = [], [], 0
    for _, a, b in pairs:
        a = a.astype(np.int32)
        b = b.astype(np.int32)
        diff = np.abs(a - b)
        # 一个像素只要任一通道不同就算"不同"
        npix = diff.reshape(-1, diff.shape[-1]).shape[0]
        nd = int((diff.reshape(-1, diff.shape[-1]).sum(axis=-1) > 0).sum())
        fr.append(nd / npix)
        ae.append(float(diff.mean()) / 255.0)
        if nd == 0:
            ident += 1
    return {"n": len(pairs), "identical": ident,
            "pixfrac": float(np.mean(fr)), "mae": float(np.mean(ae))}


def _load_pairs(da, db, problems, label):
    from PIL import Image
    pa, pb = pathlib.Path(da) / SIZE, pathlib.Path(db) / SIZE
    if not pa.is_dir() or not pb.is_dir():
        problems.append("MISSING_DIR:%s:%s" % (label, pa if not pa.is_dir() else pb))
        return []
    out = []
    for f in sorted(pa.glob("*.png")):
        g = pb / f.name
        if not g.exists():
            problems.append("MISSING_FILE:%s:%s" % (label, f.name))
            continue
        x = np.asarray(Image.open(f).convert("RGB"))
        y = np.asarray(Image.open(g).convert("RGB"))
        if x.shape != y.shape:
            problems.append("SHAPE:%s:%s" % (label, f.name))
            continue
        out.append((f.name, x, y))
    return out


def verdict(d_c, D37, D38):
    """(V) 三分判决。⚠ 门是闭区间边界：>=D37 才 GO，<=D38 才 NOGO。"""
    if d_c >= D37:
        return "SCREEN_GO"
    if d_c <= D38:
        return "SCREEN_NOGO"
    return "SCREEN_AMBIGUOUS"


def calibrate(D37, D38, D0):
    """(C1)(C2) 标定门。返回 (ok, 原因列表)。"""
    why = []
    if not (D37 > D38):
        why.append("C1:D37<=D38")
    if not (D0 <= D38 / 10.0):
        why.append("C2:D0>D38/10")
    return (not why), why


def selftest():
    ok, fail = 0, []

    def chk(name, cond):
        nonlocal ok
        if cond:
            ok += 1
        else:
            fail.append(name)

    z = np.zeros((16, 16, 3), np.uint8)
    o = np.full((16, 16, 3), 255, np.uint8)
    half = z.copy()
    half[:8] = 255
    s = _stats([("a", z, z)])
    chk("T1 同图 pixfrac=0", s["pixfrac"] == 0.0 and s["identical"] == 1)
    s = _stats([("a", z, o)])
    chk("T2 全异 pixfrac=1", s["pixfrac"] == 1.0 and s["identical"] == 0 and abs(s["mae"] - 1.0) < 1e-9)
    s = _stats([("a", z, half)])
    chk("T3 半异 pixfrac=0.5", abs(s["pixfrac"] - 0.5) < 1e-9)
    s = _stats([])
    chk("T4 空集报 n=0 而不是 0.0", s["n"] == 0 and s["pixfrac"] is None)
    chk("T5 GO", verdict(0.40, 0.30, 0.10) == "SCREEN_GO")
    chk("T6 NOGO", verdict(0.05, 0.30, 0.10) == "SCREEN_NOGO")
    chk("T7 中间", verdict(0.20, 0.30, 0.10) == "SCREEN_AMBIGUOUS")
    chk("T8 边界 d==D37 算 GO", verdict(0.30, 0.30, 0.10) == "SCREEN_GO")
    chk("T9 边界 d==D38 算 NOGO", verdict(0.10, 0.30, 0.10) == "SCREEN_NOGO")
    chk("T10 标定通过", calibrate(0.30, 0.10, 0.005)[0])
    chk("T11 C1 不过", calibrate(0.10, 0.30, 0.0)[1] == ["C1:D37<=D38"])
    chk("T12 C2 不过", calibrate(0.30, 0.10, 0.05)[1] == ["C2:D0>D38/10"])
    print("[selftest] %d/%d 通过 %s" % (ok, ok + len(fail), fail))
    return 0 if not fail else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default="/tmp/ckab16")
    ap.add_argument("--cfg", default="/tmp/cfgab16")
    ap.add_argument("--ret", default="/tmp/retnn16")
    ap.add_argument("--out", default="/tmp/m39_retnn_screen.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())

    ck, cfg, ret = pathlib.Path(a.ck), pathlib.Path(a.cfg), pathlib.Path(a.ret)
    problems = []
    jobs = {
        "D37_ck3k_vs_ck20k": (ck / "ck3k_rr4", ck / "ck20k_rr4"),
        "D38_cfg25_vs_cfg15": (cfg / "cfg25_rr4", cfg / "cfg15_rr4"),
        "D0_ret100_vs_cfg15": (ret / "ret100_rr4", cfg / "cfg15_rr4"),
        "d_ret30_vs_ret100": (ret / "ret30_rr4", ret / "ret100_rr4"),
        "d_ret300_vs_ret100": (ret / "ret300_rr4", ret / "ret100_rr4"),
    }
    res = {}
    for k, (da, db) in jobs.items():
        res[k] = _stats(_load_pairs(da, db, problems, k))
        res[k]["dirs"] = [str(da), str(db)]

    out = {"meter": "pixfrac = 逐像素不同比例的材质均值", "pairs": res, "problems": problems,
           "n_checked": len(jobs)}

    missing = [k for k, v in res.items() if v["n"] == 0]
    if missing:
        out["verdict"] = "VOID_NO_DATA"
        out["missing"] = missing
        out["note"] = "已查 %d 项，其中 %d 项没有数据 -> 本轮什么都没量到" % (len(jobs), len(missing))
    else:
        D37 = res["D37_ck3k_vs_ck20k"]["pixfrac"]
        D38 = res["D38_cfg25_vs_cfg15"]["pixfrac"]
        D0 = res["D0_ret100_vs_cfg15"]["pixfrac"]
        cal_ok, why = calibrate(D37, D38, D0)
        out["anchors"] = {"D37": D37, "D38": D38, "D0": D0}
        out["calibration"] = {"ok": cal_ok, "why": why}
        if not cal_ok:
            out["verdict"] = "METER_UNCALIBRATED"
            out["note"] = "标定门未过 -> 本轮不出 GO/NO-GO（判据 C1/C2）"
        else:
            v = {c: verdict(res["d_%s_vs_ret100" % c]["pixfrac"], D37, D38) for c in ("ret30", "ret300")}
            out["verdict"] = v
            gos = [c for c in ("ret30", "ret300") if v[c] == "SCREEN_GO"]
            if len(gos) > 1:
                gos = [max(gos, key=lambda c: res["d_%s_vs_ret100" % c]["pixfrac"])]
            out["promote"] = gos[0] if gos else None   # (P) 只许提拔一条；None = 本轮不花 API

    pathlib.Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ⚠ 落盘已在打印之前完成（非 GBK 字形会让 Windows 控制台崩在写文件之前）
    print("== (M39) 跑前筛子（零 API）==")
    for k, v in res.items():
        if v["n"] == 0:
            print("  %-22s 【禁】无数据（n=0）" % k)
        else:
            print("  %-22s n=%3d  pixfrac=%.4f  mae=%.4f  逐像素全同 %d/%d"
                  % (k, v["n"], v["pixfrac"], v["mae"], v["identical"], v["n"]))
    print("  已查 %d 项，problems=%d %s" % (out["n_checked"], len(problems), problems[:5]))
    print("  判决: %s" % json.dumps(out["verdict"], ensure_ascii=False))
    if "promote" in out:
        print("  提拔（仅授权去预注册，不代表更好）: %s" % out["promote"])
    print("  已写 %s" % a.out)


if __name__ == "__main__":
    main()
