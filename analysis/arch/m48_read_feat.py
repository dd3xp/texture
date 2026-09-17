"""(M48) 判读器：把 (M46) 的 `P_sel` 6.54（"挑哪张调色板"）拆成四桶。

盲写 —— 写于任何读数之前，判据一字来自 `docs/arch_progress.md` 的 (M48) 预注册节。
用法：
    python analysis/arch/m48_read_feat.py --selftest
    python analysis/arch/m48_read_feat.py --json experiments/m48_diag_feat_v8_Vmat.json \
        --out experiments/m48_verdict.json

四桶（相加 = xmodal - real = P_sel）：
    S1 = xmodal      - incbest5      挑错（重排器）
    S2 = incbest5    - incbest30     池太窄（放宽 topk）
    S3 = incbest30   - incbest512r   名字预筛丢货（换检索）
    S4 = incbest512r - real          库里没有（合成调色板）
"""
import argparse
import json
import sys
from pathlib import Path

THR = 4.656          # (M33) m=1 KID 噪声下限
REPRO_TOL = 0.5      # (OP2) 复现 (M46) 的容差
M46_XMODAL = 13.431
M46_REAL = 6.894

ROWS = ["TRD", "real_half", "palette=real", "palette=xmodal",
        "palette=incbest5", "palette=incbest30", "palette=incbest30r", "palette=incbest512r"]
SEL_KEYS = ["palette=xmodal", "palette=incbest5", "palette=incbest30"]

BUCKETS = [
    ("S1", "palette=xmodal", "palette=incbest5", "GAP_PICK_IN_TOP5",
     "可捞量在「现有 5 个候选里没挑对」=> 下个部件造重排器"),
    ("S2", "palette=incbest5", "palette=incbest30", "GAP_POOL_TOO_NARROW",
     "可捞量在「候选池只有 5 张」=> 放宽 topk 到 30 并重排"),
    ("S3", "palette=incbest30", "palette=incbest512r", "GAP_NAME_PREFILTER",
     "可捞量在「按名字预筛 30 条」=> 换检索方式"),
    ("S4", "palette=incbest512r", "palette=real", "GAP_BANK_COVERAGE",
     "可捞量在「候选池里根本没有这张」=> 得合成调色板"),
]


def analyse(j):
    """j: 落盘 JSON（行名 -> 指标 dict，外加可选的 '_sel'）。返回判决 dict。"""
    out = {"thr": THR, "checks": {}, "buckets": {}, "notes": []}

    missing = [r for r in ROWS if r not in j or "KID_x1e3" not in j.get(r, {})]
    sel = j.get("_sel") or {}
    missing_sel = [k for k in SEL_KEYS if k not in sel or "mean_d" not in sel.get(k, {})]
    out["missing_rows"] = missing
    out["missing_sel"] = missing_sel
    if missing or missing_sel:
        out["verdict"] = "VOID_NO_DATA"
        out["n_checks_done"] = 0
        out["why"] = ("缺行 " + ",".join(missing) if missing else "") + \
                     ("; 缺选择距离 " + ",".join(missing_sel) if missing_sel else "")
        return out

    K = {r: float(j[r]["KID_x1e3"]) for r in ROWS}
    out["kid"] = K

    # (OP1) 动态范围
    dyn = K["TRD"] - K["real_half"]
    out["checks"]["OP1_dyn_range"] = {"value": dyn, "need": f">= {THR}", "pass": dyn >= THR}
    # (OP2) 复现 (M46)
    d_x = K["palette=xmodal"] - M46_XMODAL
    d_r = K["palette=real"] - M46_REAL
    op2 = abs(d_x) <= REPRO_TOL and abs(d_r) <= REPRO_TOL
    out["checks"]["OP2_repro_M46"] = {"d_xmodal": d_x, "d_real": d_r,
                                      "need": f"|d| <= {REPRO_TOL}", "pass": op2}

    for tag, a, b, _, _ in BUCKETS:
        out["buckets"][tag] = K[a] - K[b]

    # (OP3) 量具同向：只查真正嵌套的 S1/S2
    s1, s2 = out["buckets"]["S1"], out["buckets"]["S2"]
    op3 = s1 >= -THR and s2 >= -THR
    out["checks"]["OP3_metric_agrees"] = {"S1": s1, "S2": s2,
                                          "need": f"S1,S2 >= -{THR}", "pass": op3}
    # (OP4) 选择按构造单调
    d = [float(sel[k]["mean_d"]) for k in SEL_KEYS]
    op4 = d[0] >= d[1] - 1e-9 and d[1] >= d[2] - 1e-9
    out["checks"]["OP4_sel_monotone"] = {"mean_d": dict(zip(SEL_KEYS, d)),
                                         "need": "d_xmodal >= d5 >= d30", "pass": op4}
    out["n_checks_done"] = 4

    if not out["checks"]["OP1_dyn_range"]["pass"]:
        out["verdict"] = "VOID_NO_DYNRANGE"
        return out
    if not op2:
        out["verdict"] = "VOID_NO_REPRO"
        return out
    if not op4:
        out["verdict"] = "VOID_SELECTION_BUG"
        return out
    if not op3:
        out["verdict"] = "VOID_METRIC_DISAGREE"
        return out

    # S4 显著为负 = 合法发现（参照瓦片自己的调色板配模型网格并非特征空间最优），不是 VOID
    if out["buckets"]["S4"] < -THR:
        out["notes"].append("ORACLE_REAL_NOT_BEST: S4 显著为负，palette=real 不是特征空间上界")

    passed = [(tag, out["buckets"][tag], vd, txt) for tag, _, _, vd, txt in BUCKETS
              if out["buckets"][tag] >= THR]
    if not passed:
        out["verdict"] = "SPLIT_UNRESOLVED"
        out["why"] = "四桶全部小于 thr，只许读成「测不出」"
        return out
    tag, val, vd, txt = max(passed, key=lambda x: x[1])
    out["verdict"] = vd
    out["win_bucket"] = tag
    out["win_value"] = val
    out["why"] = txt
    out["P_sel"] = K["palette=xmodal"] - K["palette=real"]
    return out


def report(v):
    print(f"判决: {v['verdict']}")
    if v.get("why"):
        print("  " + v["why"])
    if v["verdict"] == "VOID_NO_DATA":
        print(f"  已查 {v['n_checks_done']} 项（数据不全，【禁】不许当作「量过没事」）")
        return
    print(f"  已查 {v['n_checks_done']} 项操作检验")
    for k, c in v["checks"].items():
        vals = "  ".join(f"{a}={b}" for a, b in c.items() if a not in ("need", "pass"))
        print(f"  [{'过' if c['pass'] else '不过'}] {k:<20} {vals}   need {c['need']}")
    print(f"  P_sel = xmodal - real = {v.get('P_sel', float('nan')):.3f}")
    for tag, a, b, _, _ in BUCKETS:
        s = v["buckets"][tag]
        print(f"  {tag} = {a} - {b} = {s:+.3f}   " + ("过门" if s >= THR else "噪声内"))
    for n in v["notes"]:
        print("  注: " + n)
    if v["verdict"].startswith("VOID"):
        print("  【禁】VOID 之后上面的差值一条都不许当结论引用")


def _mk(kid_map, sel_d=(3.0, 2.0, 1.0)):
    j = {r: {"KID_x1e3": kid_map[r]} for r in ROWS}
    j["_sel"] = {k: {"mean_d": d} for k, d in zip(SEL_KEYS, sel_d)}
    return j


def selftest():
    base = {"TRD": 38.9, "real_half": 3.8, "palette=real": 6.9, "palette=xmodal": 13.4,
            "palette=incbest5": 12.0, "palette=incbest30": 11.0,
            "palette=incbest30r": 11.5, "palette=incbest512r": 10.0}
    ok, bad = 0, []

    def chk(name, got, want):
        nonlocal ok
        if got == want:
            ok += 1
        else:
            bad.append(f"{name}: got {got} want {want}")

    # 1 四桶全小于 thr -> SPLIT_UNRESOLVED
    chk("all-small", analyse(_mk(base))["verdict"], "SPLIT_UNRESOLVED")
    # 2 S1 最大且过门
    d = dict(base, **{"palette=incbest5": 7.5, "palette=incbest30": 7.2, "palette=incbest512r": 7.0})
    chk("S1-wins", analyse(_mk(d))["verdict"], "GAP_PICK_IN_TOP5")
    # 3 S2 最大且过门
    d = dict(base, **{"palette=incbest5": 13.0, "palette=incbest30": 7.5, "palette=incbest512r": 7.2})
    chk("S2-wins", analyse(_mk(d))["verdict"], "GAP_POOL_TOO_NARROW")
    # 4 S3 最大且过门
    d = dict(base, **{"palette=incbest5": 13.2, "palette=incbest30": 13.0, "palette=incbest512r": 7.5})
    chk("S3-wins", analyse(_mk(d))["verdict"], "GAP_NAME_PREFILTER")
    # 5 S4 最大且过门
    d = dict(base, **{"palette=incbest5": 13.2, "palette=incbest30": 13.0, "palette=incbest512r": 12.9})
    chk("S4-wins", analyse(_mk(d))["verdict"], "GAP_BANK_COVERAGE")
    # 6 并列时取最大：S2 = 6.0 > S1 = 5.0
    d = dict(base, **{"palette=incbest5": 8.4, "palette=incbest30": 2.4, "palette=incbest512r": 2.2})
    r = analyse(_mk(d))
    chk("max-bucket", r["verdict"], "GAP_POOL_TOO_NARROW")
    chk("max-tag", r["win_bucket"], "S2")
    # 7 (OP1) 不过
    chk("OP1", analyse(_mk(dict(base, TRD=5.0)))["verdict"], "VOID_NO_DYNRANGE")
    # 8 (OP2) xmodal 复现不上
    chk("OP2-x", analyse(_mk(dict(base, **{"palette=xmodal": 20.0})))["verdict"], "VOID_NO_REPRO")
    # 9 (OP2) real 复现不上
    chk("OP2-r", analyse(_mk(dict(base, **{"palette=real": 2.0})))["verdict"], "VOID_NO_REPRO")
    # 10 (OP2) 容差内（0.4）不触发
    chk("OP2-tol", analyse(_mk(dict(base, **{"palette=xmodal": M46_XMODAL + 0.4})))["verdict"],
        "SPLIT_UNRESOLVED")
    # 11 (OP3) S2 显著为负 -> VOID_METRIC_DISAGREE（复刻 (M47) 的形状）
    d = dict(base, **{"palette=incbest5": 10.3, "palette=incbest30": 18.4, "palette=incbest512r": 20.4})
    chk("OP3", analyse(_mk(d))["verdict"], "VOID_METRIC_DISAGREE")
    # 12 (OP3) S1 显著为负
    chk("OP3-S1", analyse(_mk(dict(base, **{"palette=incbest5": 19.0})))["verdict"],
        "VOID_METRIC_DISAGREE")
    # 13 (OP4) 选择距离非单调
    chk("OP4", analyse(_mk(base, sel_d=(1.0, 2.0, 3.0)))["verdict"], "VOID_SELECTION_BUG")
    # 14 缺行
    j = _mk(base)
    del j["palette=incbest512r"]
    r = analyse(j)
    chk("miss-row", r["verdict"], "VOID_NO_DATA")
    chk("miss-checks0", r["n_checks_done"], 0)
    # 15 缺选择距离
    j = _mk(base)
    del j["_sel"]["palette=incbest30"]
    chk("miss-sel", analyse(j)["verdict"], "VOID_NO_DATA")
    # 16 S4 显著为负 -> 不 VOID，但要留 note
    d = dict(base, **{"palette=incbest5": 13.2, "palette=incbest30": 13.0,
                      "palette=incbest512r": 0.5, "palette=real": 6.9})
    r = analyse(_mk(d))
    chk("S4-neg-note", any(n.startswith("ORACLE_REAL_NOT_BEST") for n in r["notes"]), True)
    chk("S4-neg-verdict", r["verdict"], "GAP_NAME_PREFILTER")
    # 17 VOID 优先于判桶：OP1 不过时不看桶
    d = dict(base, TRD=5.0, **{"palette=incbest5": 7.5})
    chk("void-first", "win_bucket" in analyse(_mk(d)), False)

    print(f"selftest {ok}/{ok + len(bad)}")
    for b in bad:
        print("  失败 " + b)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    if not a.json:
        ap.error("要么 --selftest，要么 --json")
    j = json.loads(a.json.read_text(encoding="utf-8"))
    v = analyse(j)
    if a.out:                       # 落盘一律挪到打印之前
        a.out.write_text(json.dumps(v, indent=1, ensure_ascii=False), encoding="utf-8")
    report(v)
    if a.out:
        print("->", a.out)


if __name__ == "__main__":
    main()
