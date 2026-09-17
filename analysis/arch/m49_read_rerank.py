"""(M49) 判读器：调色板重排器的**可达读数**能追回 (M48) 那 5.105 的多少。

盲写 —— 写于任何读数之前，判据一字来自 `docs/arch_progress.md` 的 (M49) 预注册节。
用法：
    python analysis/arch/m49_read_rerank.py --selftest
    python analysis/arch/m49_read_rerank.py --json experiments/m49_rerank_v8_Vmat.json \
        --out experiments/m49_verdict.json

G(row) = KID(palette=xmodal) - KID(row)；G_best = 三个可达重排器里最大的 G。
    G_best >= thr        -> RERANK_RECOVERS（点名哪一行 + 追回几成）
    0 < G_best < thr     -> RERANK_TOO_SMALL
    G_best <= 0          -> RERANK_NULL
"""
import argparse
import json
import sys
from pathlib import Path

THR = 4.656          # (M33) m=1 KID 噪声下限
REPRO_TOL = 0.5      # (OP2) 复现容差
M46_XMODAL = 13.431  # (M46)/(M48) 已发表
M48_BEST5 = 8.326    # (M48) 已发表（不可达上界那一行）

BASE = "palette=xmodal"
ORACLE = "palette=incbest5"
RR = ["palette=rr_argmax", "palette=rr_trdpal", "palette=rr_incTRD"]
ROWS = ["TRD", "real_half", BASE, ORACLE] + RR

CHANGE_MIN = 0.5     # (OP3) 合格线
CHANGE_DROP = 0.1    # (OP3) 低于此值的行剔出 G_best


def analyse(j):
    """j: 落盘 JSON（行名 -> 指标 dict，外加 '_rr' 诊断）。返回判决 dict。"""
    out = {"thr": THR, "checks": {}, "G": {}, "notes": []}

    missing = [r for r in ROWS if r not in j or "KID_x1e3" not in j.get(r, {})]
    rr = j.get("_rr") or {}
    missing_rr = [k for k in RR if k not in rr or "change_rate" not in rr.get(k, {})]
    out["missing_rows"] = missing
    out["missing_rr"] = missing_rr
    if missing or missing_rr:
        out["verdict"] = "VOID_NO_DATA"
        out["n_checks_done"] = 0
        out["why"] = ("缺行 " + ",".join(missing) if missing else "") + \
                     ("; 缺诊断 " + ",".join(missing_rr) if missing_rr else "")
        return out

    K = {r: float(j[r]["KID_x1e3"]) for r in ROWS}
    out["kid"] = K
    for r in RR:
        out["G"][r] = K[BASE] - K[r]
    out["S1_now"] = K[BASE] - K[ORACLE]

    # (OP1) 动态范围
    dyn = K["TRD"] - K["real_half"]
    out["checks"]["OP1_dyn_range"] = {"value": dyn, "need": f">= {THR}", "pass": dyn >= THR}
    # (OP2) 复现两个锚点
    d_x = K[BASE] - M46_XMODAL
    d_b = K[ORACLE] - M48_BEST5
    op2 = abs(d_x) <= REPRO_TOL and abs(d_b) <= REPRO_TOL
    out["checks"]["OP2_repro"] = {"d_xmodal": d_x, "d_incbest5": d_b,
                                  "need": f"|d| <= {REPRO_TOL}", "pass": op2}
    # (OP3) 真的改了选择
    ch = {r: float(rr[r]["change_rate"]) for r in RR}
    dropped = [r for r in RR if ch[r] < CHANGE_DROP]
    op3 = all(ch[r] >= CHANGE_MIN for r in RR)
    out["checks"]["OP3_changed"] = {"change_rate": ch, "need": f">= {CHANGE_MIN} 每行",
                                    "pass": op3}
    out["dropped_rows"] = dropped
    # (OP4) 同一候选集
    sub = bool(j.get("_rr_subset_ok", rr.get("subset_ok", False)))
    out["checks"]["OP4_same_pool"] = {"subset_ok": sub, "need": "True", "pass": sub}
    out["n_checks_done"] = 4

    if not out["checks"]["OP1_dyn_range"]["pass"]:
        out["verdict"] = "VOID_NO_DYNRANGE"
        return out
    if not op2:
        out["verdict"] = "VOID_NO_REPRO"
        return out
    if not sub:
        out["verdict"] = "VOID_SELECTION_BUG"
        return out
    live = [r for r in RR if r not in dropped]
    if not live:
        out["verdict"] = "VOID_NO_CHANGE"
        out["why"] = "三行都几乎没改变选择，读数就是 xmodal 那一行"
        return out
    if dropped:
        out["notes"].append("DEGENERATE_ROWS: " + ",".join(dropped) +
                            " 几乎没改变选择，已剔出 G_best")
    if not op3:
        out["notes"].append("LOW_CHANGE_RATE: 有行 change_rate 低于 " + str(CHANGE_MIN) +
                            "，其 G 只许读成「测不出」")

    win = max(live, key=lambda r: out["G"][r])
    g = out["G"][win]
    out["win_row"] = win
    out["G_best"] = g
    out["recovered_frac"] = (g / out["S1_now"]) if out["S1_now"] > 0 else None
    if g >= THR:
        out["verdict"] = "RERANK_RECOVERS"
        out["why"] = f"{win} 追回 {g:.3f}（上界 {out['S1_now']:.3f} 的 " + \
                     (f"{100 * g / out['S1_now']:.0f}%）" if out["S1_now"] > 0 else "—）")
        out["notes"].append("MULTI_ARM_3: 三行取最大＝三重比较；【禁】落选两行的 G 只许读成「测不出」")
    elif g > 0:
        out["verdict"] = "RERANK_TOO_SMALL"
        out["why"] = f"最好的一行 {win} 只追回 {g:.3f} < thr {THR} ＝测不出，【禁】不许开判官臂"
    else:
        out["verdict"] = "RERANK_NULL"
        out["why"] = f"三行没有一行比随机好（最好 {win} 的 G = {g:.3f}）"
    return out


def report(v):
    print(f"判决: {v['verdict']}")
    if v.get("why"):
        print("  " + v["why"])
    if v["verdict"] == "VOID_NO_DATA":
        print(f"  已查 {v['n_checks_done']} 项（数据不全，【禁】不许当作「量过没事」）")
        print("  " + (v.get("why") or ""))
        return
    print(f"  已查 {v['n_checks_done']} 项操作检验")
    for k, c in v["checks"].items():
        vals = "  ".join(f"{a}={b}" for a, b in c.items() if a not in ("need", "pass"))
        print(f"  [{'过' if c['pass'] else '不过'}] {k:<16} {vals}   need {c['need']}")
    print(f"  上界 S1_now = xmodal - incbest5 = {v.get('S1_now', float('nan')):+.3f}"
          f"   （【禁】不可达）")
    for r in RR:
        g = v["G"].get(r)
        if g is None:
            continue
        tag = "过门" if g >= THR else ("噪声内" if g > 0 else "反向")
        mark = "（已剔）" if r in v.get("dropped_rows", []) else ""
        print(f"  G[{r:<20}] = {g:+.3f}   {tag}{mark}")
    for n in v["notes"]:
        print("  注: " + n)
    if v["verdict"].startswith("VOID"):
        print("  【禁】VOID 之后上面的差值一条都不许当结论引用")


def _mk(kid_map, change=(0.8, 0.8, 0.8), subset=True):
    j = {r: {"KID_x1e3": kid_map[r]} for r in ROWS}
    j["_rr"] = {r: {"change_rate": c, "oracle_hit": 0.3} for r, c in zip(RR, change)}
    j["_rr_subset_ok"] = subset
    return j


def selftest():
    base = {"TRD": 38.9, "real_half": 3.8, BASE: 13.431, ORACLE: 8.326,
            "palette=rr_argmax": 13.0, "palette=rr_trdpal": 13.2, "palette=rr_incTRD": 13.1}
    ok, bad = 0, []

    def chk(name, got, want):
        nonlocal ok
        if got == want:
            ok += 1
        else:
            bad.append(f"{name}: got {got} want {want}")

    # 1 三行都只小赢 -> RERANK_TOO_SMALL
    chk("small", analyse(_mk(base))["verdict"], "RERANK_TOO_SMALL")
    # 2 一行过门 -> RECOVERS 并点名
    d = dict(base, **{"palette=rr_incTRD": 8.5})
    r = analyse(_mk(d))
    chk("recovers", r["verdict"], "RERANK_RECOVERS")
    chk("recovers-who", r["win_row"], "palette=rr_incTRD")
    chk("recovers-multiarm", any(n.startswith("MULTI_ARM_3") for n in r["notes"]), True)
    # 3 追回比例
    chk("frac", round(r["recovered_frac"], 3), round((13.431 - 8.5) / (13.431 - 8.326), 3))
    # 4 全部反向 -> RERANK_NULL
    d = dict(base, **{"palette=rr_argmax": 15.0, "palette=rr_trdpal": 16.0,
                      "palette=rr_incTRD": 14.0})
    chk("null", analyse(_mk(d))["verdict"], "RERANK_NULL")
    # 5 恰好 0 也算 NULL
    d = dict(base, **{"palette=rr_argmax": 13.431, "palette=rr_trdpal": 16.0,
                      "palette=rr_incTRD": 14.0})
    chk("zero", analyse(_mk(d))["verdict"], "RERANK_NULL")
    # 6 取三行最大
    d = dict(base, **{"palette=rr_argmax": 8.0, "palette=rr_trdpal": 7.0,
                      "palette=rr_incTRD": 9.0})
    chk("max-row", analyse(_mk(d))["win_row"], "palette=rr_trdpal")
    # 7 (OP1) 不过
    chk("OP1", analyse(_mk(dict(base, TRD=5.0)))["verdict"], "VOID_NO_DYNRANGE")
    # 8 (OP2) xmodal 复现不上
    chk("OP2-x", analyse(_mk(dict(base, **{BASE: 20.0})))["verdict"], "VOID_NO_REPRO")
    # 9 (OP2) incbest5 复现不上
    chk("OP2-b", analyse(_mk(dict(base, **{ORACLE: 2.0})))["verdict"], "VOID_NO_REPRO")
    # 10 (OP2) 容差内不触发
    chk("OP2-tol", analyse(_mk(dict(base, **{ORACLE: M48_BEST5 + 0.4})))["verdict"],
        "RERANK_TOO_SMALL")
    # 11 (OP4) 不过
    chk("OP4", analyse(_mk(base, subset=False))["verdict"], "VOID_SELECTION_BUG")
    # 12 (OP3) 全不改 -> VOID_NO_CHANGE
    chk("OP3-all", analyse(_mk(base, change=(0.0, 0.05, 0.0)))["verdict"], "VOID_NO_CHANGE")
    # 13 (OP3) 只有一行退化：剔出并留注记
    d = dict(base, **{"palette=rr_argmax": 8.0})
    r = analyse(_mk(d, change=(0.02, 0.8, 0.8)))
    chk("OP3-drop-verdict", r["verdict"], "RERANK_TOO_SMALL")
    chk("OP3-drop-note", any(n.startswith("DEGENERATE_ROWS") for n in r["notes"]), True)
    chk("OP3-drop-win", r["win_row"], "palette=rr_incTRD")
    # 14 (OP3) 中间地带（0.1~0.5）：不剔，但留低变动注记
    r = analyse(_mk(base, change=(0.3, 0.8, 0.8)))
    chk("OP3-mid", any(n.startswith("LOW_CHANGE_RATE") for n in r["notes"]), True)
    chk("OP3-mid-verdict", r["verdict"], "RERANK_TOO_SMALL")
    # 15 缺行
    j = _mk(base)
    del j["palette=rr_trdpal"]
    r = analyse(j)
    chk("miss-row", r["verdict"], "VOID_NO_DATA")
    chk("miss-checks0", r["n_checks_done"], 0)
    # 16 缺诊断
    j = _mk(base)
    del j["_rr"]["palette=rr_incTRD"]
    chk("miss-rr", analyse(j)["verdict"], "VOID_NO_DATA")
    # 17 VOID 优先于点名
    chk("void-first", "win_row" in analyse(_mk(dict(base, TRD=5.0))), False)

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
