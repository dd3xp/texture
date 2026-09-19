#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""(M73) 取消资格式尺子的「分辨力预算」审计。

判据在跑任何算术之前已冻结于 docs/arch_progress.md 的 (M73) 预注册（commit ef52676）。
本文件只实现它，不重新定义任何门槛。

定义（预注册第一节，逐字）：
    H  = 人的锚点点估计
    [L,U] = 按包自助 95%CI，   W = U - L
    C  = 负对照点估计
    D  = |C - H|                        动态范围
    f  = W / D                          主判据：分辨力预算
    margin_ctrl_in_W  = 对照离最近 CI 边界的距离 / W（正 = 在 CI 外 = 取消得动）
    disq_frac = 在 [min(H,C), max(H,C)] 轴段上落在 CI 之外的长度占比

判据（预注册第二节，逐字）：
    (J1) f < 0.5 -> POWER_OK ; 0.5 <= f < 1 -> POWER_THIN ; f >= 1 -> POWER_NONE
    (J2) 只用同时有 16/32 两档的三把 (P9)(M71)(M72)：
         med f@32 >= 1.5 * med f@16  -> SIZE32_POWER_DEFICIT
         med f@32 <= med f@16 / 1.5  -> SIZE16_POWER_DEFICIT
         否则                         -> POWER_SIZE_TIE
    (J3) 只登记每轮已发表的交付余量，折成 W 的倍数。不重判。

操作检验：
    (OP1) 自洽：每格 H 必须落在它自己的 [L,U] 内。
    (OP2) 出处核对：从 JSON 取出的 H / CI / C 四舍五入到 4 位，必须与账本里写下的数逐位相同。
    (OP3) 自助稳健：对有 seed1 CI 的三把尺子，f(seed1) 与 f(seed0) 差 < 0.10。
    (OP4) 覆盖：五个产物文件全部读到，且格数 = 8（m65/m66 各 1 档，p9/m71/m72 各 2 档）。

零 GPU、零 API、只读已落盘 JSON。
"""
import argparse
import io
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------- 冻结件


def power_budget(H, lo, hi, C):
    """给定锚点、按包自助 CI、负对照，返回分辨力预算。任何将来的预注册都可 import 这个函数。"""
    W = float(hi) - float(lo)
    D = abs(float(C) - float(H))
    f = float("inf") if D == 0.0 else W / D
    if C < lo:
        margin = (lo - C) / W
    elif C > hi:
        margin = (C - hi) / W
    else:
        margin = -min(C - lo, hi - C) / W
    a, b = min(H, C), max(H, C)
    inter = max(0.0, min(hi, b) - max(lo, a))
    disq = 0.0 if D == 0.0 else 1.0 - inter / D
    return {"W": W, "D": D, "f": f, "tier": tier(f),
            "margin_ctrl_in_W": margin, "disq_frac": disq}


def tier(f):
    """(J1) 三档。门槛冻结：0.5 / 1.0。"""
    if f < 0.5:
        return "POWER_OK"
    if f < 1.0:
        return "POWER_THIN"
    return "POWER_NONE"


def margin_in_W(v, lo, hi):
    """(J3) 某个方法读数离 CI 的余量，折成 W 的倍数。正 = 在 CI 外（被取消资格）。"""
    W = hi - lo
    if v < lo:
        return (lo - v) / W
    if v > hi:
        return (v - hi) / W
    return -min(v - lo, hi - v) / W


def size_verdict(med16, med32):
    """(J2)。"""
    if med32 >= 1.5 * med16:
        return "SIZE32_POWER_DEFICIT"
    if med32 <= med16 / 1.5:
        return "SIZE16_POWER_DEFICIT"
    return "POWER_SIZE_TIE"


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


# ---------------------------------------------------------------- 取数

def _load(name):
    p = os.path.join(ROOT, "experiments", name)
    with io.open(p, encoding="utf-8") as fh:
        return json.load(fh)


# 账本里写下的数（docs/arch_progress.md），(OP2) 逐位核对用。None = 账本没写死、不核。
LEDGER = {
    ("M65", 32): (1.0623, 1.0100, 1.5472, 1.5068),
    ("M66", 32): (0.3624, 0.3181, 0.6087, 0.7078),
    ("P9", 16): (0.5480, 0.4004, 0.6787, 0.7481),
    ("P9", 32): (None, 0.3073, 0.7297, 0.5350),
    ("M71", 16): (0.4758, 0.4357, 0.5000, 0.1864),
    ("M71", 32): (0.6460, 0.4693, 0.7617, 0.1893),
    ("M72", 16): (0.0902, 0.0772, 0.1031, 0.0117),
    ("M72", 32): (0.0943, 0.0705, 0.1288, 0.0117),
}


def collect():
    """返回 8 个格子：每格 (tag, size, 尺子名, H, lo, hi, C, 对照名, 交付读数, 交付方法名)。"""
    cells = []

    d = _load("m65_seam_anchor.json")
    cells.append(dict(tag="M65", size=32, ruler="tile_seam_ratio(signed)",
                      H=d["human_anchor_H"], lo=d["human_ci95_pack_bootstrap"][0],
                      hi=d["human_ci95_pack_bootstrap"][1],
                      C=d["R_crop_median"], ctrl="R_crop", ci1=None,
                      deliv=d["methods"]["TRD32_rr4"]["median"], deliv_name="TRD32_rr4"))

    d = _load("m66_div_anchor.json")
    # 交付形状 TRD32_rr4 在这把尺子上构造性不可评分（4 选 1 只交 1 张）=> 退到 TRD32 并标注
    cells.append(dict(tag="M66", size=32, ruler="LPIPS_div",
                      H=d["human_anchor_H_div"], lo=d["human_ci95_pack_bootstrap"][0],
                      hi=d["human_ci95_pack_bootstrap"][1],
                      C=d["R_div_control"], ctrl="R_div", ci1=None,
                      deliv=d["methods"]["TRD32"]["LPIPS_div"], deliv_name="TRD32(rr4 不可评分)"))

    spec = [("P9", "p9_seam_abs.json", "median_abs_dev", "human_abs_ci95",
             "R_crop", "R_crop", "op6_ci_seed1", "|seam-1|"),
            ("M71", "m71_struct_scale.json", "median_LF", "human_ci95",
             "R_shuf", "R_shuf", "op5_ci_seed1", "LF"),
            ("M72", "m72_phase_excess.json", "median_PX", "human_ci95",
             "R_phase", "R_phase", "op5_ci_seed1", "PX")]
    for tag, fn, vkey, cikey, ckey, cname, ci1key, ruler in spec:
        d = _load(fn)
        for size in (16, 32):
            s = d["sizes"][str(size)]
            dn = "TRD16c_rr4" if size == 16 else "TRD32_rr4"
            cells.append(dict(tag=tag, size=size, ruler=ruler,
                              H=s["human"][vkey], lo=s[cikey][0], hi=s[cikey][1],
                              C=s[ckey][vkey], ctrl=cname, ci1=s.get(ci1key),
                              deliv=s["methods"][dn][vkey], deliv_name=dn))
    return cells


def analyse():
    cells = collect()
    rows, op1, op2, op3 = [], [], [], []
    for c in cells:
        pb = power_budget(c["H"], c["lo"], c["hi"], c["C"])
        row = dict(c)
        row.pop("ci1")
        row.update(pb)
        row["deliv_margin_in_W"] = margin_in_W(c["deliv"], c["lo"], c["hi"])
        row["deliv_outside_CI"] = bool(c["deliv"] < c["lo"] or c["deliv"] > c["hi"])
        rows.append(row)

        # (OP1)
        op1.append(dict(cell="%s@%d" % (c["tag"], c["size"]),
                        ok=bool(c["lo"] <= c["H"] <= c["hi"])))
        # (OP2)
        led = LEDGER[(c["tag"], c["size"])]
        chk = []
        for got, want in zip((c["H"], c["lo"], c["hi"], c["C"]), led):
            chk.append(True if want is None else bool(round(got, 4) == want))
        op2.append(dict(cell="%s@%d" % (c["tag"], c["size"]), ok=all(chk), detail=chk))
        # (OP3)
        if c["ci1"] is not None:
            f1 = power_budget(c["H"], c["ci1"][0], c["ci1"][1], c["C"])["f"]
            op3.append(dict(cell="%s@%d" % (c["tag"], c["size"]),
                            f_seed0=pb["f"], f_seed1=f1,
                            ok=bool(abs(f1 - pb["f"]) < 0.10)))

    both = {t: {} for t in ("P9", "M71", "M72")}
    for r in rows:
        if r["tag"] in both:
            both[r["tag"]][r["size"]] = r["f"]
    med16 = median([v[16] for v in both.values()])
    med32 = median([v[32] for v in both.values()])
    j2 = size_verdict(med16, med32)

    op4 = dict(n_cells=len(rows), ok=bool(len(rows) == 8))
    ops = {"OP1_anchor_inside_own_CI": {"ok": all(x["ok"] for x in op1), "detail": op1},
           "OP2_matches_ledger": {"ok": all(x["ok"] for x in op2), "detail": op2},
           "OP3_bootstrap_seed_robust": {"ok": all(x["ok"] for x in op3), "detail": op3},
           "OP4_coverage": op4}
    return {"note": "(M73) disqualification-ruler power budget; criteria pre-registered in ef52676",
            "rows": rows,
            "J2": {"verdict": j2, "med_f_16": med16, "med_f_32": med32,
                   "ratio_32_over_16": med32 / med16, "per_ruler": both},
            "op": ops,
            "all_op_ok": all(v["ok"] for v in ops.values())}


# ---------------------------------------------------------------- selftest

def selftest():
    n = 0
    def ck(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # 解析算得准的 8 个构造
    p = power_budget(0.0, -1.0, 1.0, 10.0)
    ck(p["W"] == 2.0, "W")
    ck(p["D"] == 10.0, "D")
    ck(abs(p["f"] - 0.2) < 1e-12, "f")
    ck(p["tier"] == "POWER_OK", "tier ok")
    ck(abs(p["margin_ctrl_in_W"] - 4.5) < 1e-12, "margin")
    ck(abs(p["disq_frac"] - 0.9) < 1e-12, "disq")

    p = power_budget(0.0, -1.0, 1.0, 2.0)          # f = 1.0 恰好
    ck(abs(p["f"] - 1.0) < 1e-12, "f=1")
    ck(p["tier"] == "POWER_NONE", "f=1 -> NONE")
    ck(abs(p["margin_ctrl_in_W"] - 0.5) < 1e-12, "margin2")
    ck(abs(p["disq_frac"] - 0.5) < 1e-12, "disq2")

    p = power_budget(0.0, -1.0, 1.0, 3.0)          # f = 2/3
    ck(abs(p["f"] - 2.0 / 3.0) < 1e-12, "f=2/3")
    ck(p["tier"] == "POWER_THIN", "THIN")

    p = power_budget(0.0, -1.0, 1.0, 4.0)          # f = 0.5 恰好 -> THIN（"< 0.5 才 OK"）
    ck(abs(p["f"] - 0.5) < 1e-12, "f=0.5")
    ck(p["tier"] == "POWER_THIN", "0.5 边界归 THIN")
    ck(tier(0.49999) == "POWER_OK", "0.5 下方归 OK")

    p = power_budget(0.0, -1.0, 1.0, -10.0)        # 对照在锚点下方，应完全对称
    ck(abs(p["f"] - 0.2) < 1e-12, "f 对称")
    ck(abs(p["margin_ctrl_in_W"] - 4.5) < 1e-12, "margin 对称")
    ck(abs(p["disq_frac"] - 0.9) < 1e-12, "disq 对称")

    p = power_budget(0.0, -1.0, 1.0, 0.5)          # 对照落在 CI 内 = (V0) 那种情形
    ck(abs(p["f"] - 4.0) < 1e-12, "f=4")
    ck(p["tier"] == "POWER_NONE", "对照在内 -> NONE")
    ck(abs(p["margin_ctrl_in_W"] + 0.25) < 1e-12, "对照在内 margin 为负")
    ck(p["disq_frac"] == 0.0, "对照在内 disq=0")

    p = power_budget(1.0, 0.0, 2.0, 1.0)           # D = 0
    ck(math.isinf(p["f"]), "D=0 -> inf")
    ck(p["tier"] == "POWER_NONE", "D=0 -> NONE")

    # margin_in_W（J3）
    ck(abs(margin_in_W(3.0, -1.0, 1.0) - 1.0) < 1e-12, "J3 在上方")
    ck(abs(margin_in_W(-3.0, -1.0, 1.0) - 1.0) < 1e-12, "J3 在下方")
    ck(abs(margin_in_W(0.5, -1.0, 1.0) + 0.25) < 1e-12, "J3 在内为负")
    ck(margin_in_W(1.0, -1.0, 1.0) == 0.0, "J3 贴边界为 0")

    # (J2) 三档与边界
    ck(size_verdict(1.0, 1.5) == "SIZE32_POWER_DEFICIT", "J2 1.5 恰好触发")
    ck(size_verdict(1.0, 1.4999) == "POWER_SIZE_TIE", "J2 1.5 下方 TIE")
    ck(size_verdict(1.5, 1.0) == "SIZE16_POWER_DEFICIT", "J2 反向恰好")
    ck(size_verdict(1.0, 0.7) == "POWER_SIZE_TIE", "J2 反向 TIE")
    ck(size_verdict(1.0, 1.0) == "POWER_SIZE_TIE", "J2 相等")

    # median
    ck(median([3.0, 1.0, 2.0]) == 2.0, "median 奇数")
    ck(median([1.0, 2.0, 3.0, 4.0]) == 2.5, "median 偶数")

    # 阳性哨兵：一把「好尺子」与一把「瞎尺子」必须分到不同档
    good = power_budget(1.0, 0.95, 1.05, 2.0)      # 窄 CI、远对照
    blind = power_budget(1.0, 0.9, 1.6, 1.2)       # 宽 CI、近对照
    ck(good["tier"] == "POWER_OK" and blind["tier"] == "POWER_NONE", "哨兵分档")
    ck(good["f"] < blind["f"], "哨兵单调")

    # LEDGER 覆盖 8 格且键与 collect() 一致
    ck(len(LEDGER) == 8, "LEDGER 8 格")

    print("selftest OK: %d/%d" % (n, n))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    res = analyse()

    if a.out:
        with io.open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)

    print("=== (M73) 分辨力预算 f = CI宽 / 动态范围 ===")
    hdr = "%-6s %-4s %-22s %9s %9s %8s %8s %-11s %9s %9s"
    print(hdr % ("tag", "size", "ruler", "W", "D", "f", "disq", "tier",
                 "ctrlMrg", "delivMrg"))
    for r in res["rows"]:
        print(hdr % (r["tag"], r["size"], r["ruler"],
                     "%.4f" % r["W"], "%.4f" % r["D"], "%.3f" % r["f"],
                     "%.3f" % r["disq_frac"], r["tier"],
                     "%.2f" % r["margin_ctrl_in_W"],
                     "%.2f" % r["deliv_margin_in_W"]))
    j = res["J2"]
    print("--- (J2) med f@16=%.3f  med f@32=%.3f  ratio=%.3f  => %s"
          % (j["med_f_16"], j["med_f_32"], j["ratio_32_over_16"], j["verdict"]))
    print("--- (J3) 交付余量（正=在CI外=被取消资格，单位 W）:")
    for r in res["rows"]:
        print("      %-6s@%-3d %-14s %8.4f  margin=%+.2f W  outside=%s"
              % (r["tag"], r["size"], r["deliv_name"], r["deliv"],
                 r["deliv_margin_in_W"], r["deliv_outside_CI"]))
    print("--- 操作检验:")
    for k, v in res["op"].items():
        print("      %-28s ok=%s" % (k, v["ok"]))
    print("ALL_OP_OK=%s" % res["all_op_ok"])
    print("M73_DONE")


if __name__ == "__main__":
    main()
