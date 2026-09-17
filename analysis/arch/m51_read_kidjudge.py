"""(M51) 判读器：拆解尺度的 KID 差**预测**判官胜率吗？

判据逐字抄自 `docs/arch_progress.md` 的 (M51) 预注册（写于任何 KID/胜率读数之前）。
零 API、零 GPU：判官侧从逐对记录重算，指标侧读已有的 JSON。

    python analysis/arch/m51_read_kidjudge.py --selftest
    python analysis/arch/m51_read_kidjudge.py --out experiments/m51_verdict.json

判决：
  VOID_NO_DATA        料不齐 / 拆解尺度子集 < 3 对
  VOID_JUDGE_MISMATCH 判官数字与 recheck_judge.EXPECT 对不上
  KID_PREDICTS        子集上「KID 低的那方赢」符号检验 p<0.05 且一致率 >0.5
  KID_NOT_PREDICTIVE  其余
"""
import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "analysis" / "arch"))
from exact import binom_test  # noqa: E402

# (M33) noise_floor.py 的 m=1 不成对噪声下限。⛔ 本轮不许改。
THR_KID = 4.656

# (判官 JSON 文件名, 指标 JSON 相对路径, A 的指标键, B 的指标键)
PAIRS = [
    ("judge_full_TRD16c_rr4_vs_AB_pal_rr4_16.json",
     "experiments/abl16_metrics.json", "TRD16c_rr4", "AB_pal_rr4"),
    ("judge_full_TRD16c_rr4_vs_AB_ex_rr4_16.json",
     "experiments/abl16_metrics.json", "TRD16c_rr4", "AB_ex_rr4"),
    ("judge_full_TRD16c_rr4_vs_AB_ct_rr4_16.json",
     "experiments/abl16_metrics.json", "TRD16c_rr4", "AB_ct_rr4"),
    ("judge_full_ck3k_rr4_vs_ck20k_rr4_16_V_mat.json",
     "remote_tmp/m37_ckpt16.json", "ck3k_rr4", "ck20k_rr4"),
    ("judge_full_cfg25_rr4_vs_cfg15_rr4_16_V_mat.json",
     "experiments/m38_cfg16.json", "cfg25_rr4", "cfg15_rr4"),
    ("judge_full_gen10_rr4_vs_gen8_rr4_16_V_mat.json",
     "experiments/m40_gen16.json", "gen10_rr4", "gen8_rr4"),
    ("judge_full_data7_rr4_vs_data8_rr4_16_V_mat.json",
     "experiments/m43_gen16.json", "data7_rr4", "data8_rr4"),
]


def _ranks(xs):
    """平均秩（处理并列）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    """Spearman rho + 穷举置换双尾 p（n<=8 时精确；更大时返回 None）。"""
    n = len(xs)
    if n < 3:
        return float("nan"), None
    rx, ry = _ranks(xs), _ranks(ys)

    def pear(a, b):
        ma, mb = sum(a) / n, sum(b) / n
        da = [v - ma for v in a]
        db = [v - mb for v in b]
        sa = math.sqrt(sum(v * v for v in da))
        sb = math.sqrt(sum(v * v for v in db))
        if sa == 0 or sb == 0:
            return float("nan")
        return sum(p * q for p, q in zip(da, db)) / (sa * sb)

    rho = pear(rx, ry)
    if n > 8 or not math.isfinite(rho):
        return rho, None
    cnt = tot = 0
    for perm in itertools.permutations(ry):
        r = pear(rx, list(perm))
        tot += 1
        if math.isfinite(r) and abs(r) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, cnt / tot


def recount_judge(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    r = d["records"]
    return (sum(x["verdict"] == "A" for x in r),
            sum(x["verdict"] in ("A", "B") for x in r),
            sum(x["verdict"] == "inconsistent" for x in r),
            sum(x["verdict"] is None for x in r))


def decide(rows, n_listed, ops):
    """rows: 已通过 (OP3)(OP5) 的有效配对。返回 (判决, 详情)。"""
    if not ops["op1_files_ok"]:
        return "VOID_NO_DATA", {"why": "(OP1) 料不齐"}
    if not ops["op2_judge_ok"]:
        return "VOID_JUDGE_MISMATCH", {"why": "(OP2) 判官数字与 EXPECT 对不上"}
    if not ops["op4_kid_finite"]:
        return "VOID_NO_DATA", {"why": "(OP4) 有 KID 非有限值"}
    if len(rows) < 6:
        return "VOID_NO_DATA", {"why": f"(OP3)/(OP5) 作废后只剩 {len(rows)} 对（<6）"}
    sub = [r for r in rows if abs(r["dKID"]) >= THR_KID]
    if len(sub) < 3:
        return "VOID_NO_DATA", {"why": f"拆解尺度子集只有 {len(sub)} 对（<3）",
                                "n_subset": len(sub)}
    k = sum(1 for r in sub if r["sign_agree"])
    p = binom_test(k, len(sub))
    if p < 0.05 and k / len(sub) > 0.5:
        return "KID_PREDICTS", {"k": k, "n_subset": len(sub), "p": p}
    return "KID_NOT_PREDICTIVE", {"k": k, "n_subset": len(sub), "p": p}


def selftest():
    ok = fail = 0

    def chk(name, got, want):
        nonlocal ok, fail
        good = got == want if not isinstance(want, float) else abs(got - want) < 1e-9
        ok, fail = ok + good, fail + (not good)
        print(f"  {'OK ' if good else '**FAIL**'} {name}: {got!r} (期望 {want!r})")

    base = dict(op1_files_ok=True, op2_judge_ok=True, op4_kid_finite=True)

    def row(d, agree):
        return {"dKID": d, "sign_agree": agree}

    # 1-4 各 VOID 分支
    chk("(OP1) 不过 -> VOID_NO_DATA",
        decide([], 7, dict(base, op1_files_ok=False))[0], "VOID_NO_DATA")
    chk("(OP2) 不过 -> VOID_JUDGE_MISMATCH",
        decide([], 7, dict(base, op2_judge_ok=False))[0], "VOID_JUDGE_MISMATCH")
    chk("(OP4) 不过 -> VOID_NO_DATA",
        decide([], 7, dict(base, op4_kid_finite=False))[0], "VOID_NO_DATA")
    chk("有效对 5 (<6) -> VOID_NO_DATA",
        decide([row(9.0, True)] * 5, 7, base)[0], "VOID_NO_DATA")
    # 5-7 子集大小
    chk("子集 0 对 -> VOID_NO_DATA",
        decide([row(1.0, True)] * 7, 7, base)[0], "VOID_NO_DATA")
    chk("子集 2 对 -> VOID_NO_DATA",
        decide([row(9.0, True)] * 2 + [row(1.0, True)] * 5, 7, base)[0], "VOID_NO_DATA")
    chk("子集 3 对全一致 -> 仍不显著(p=0.25)",
        decide([row(9.0, True)] * 3 + [row(1.0, True)] * 4, 7, base)[0],
        "KID_NOT_PREDICTIVE")
    # 8-10 显著性
    chk("子集 6 对全一致 -> KID_PREDICTS",
        decide([row(9.0, True)] * 6 + [row(1.0, True)], 7, base)[0], "KID_PREDICTS")
    chk("子集 6 对全不一致 -> NOT_PREDICTIVE（方向反了也不许判 PREDICTS）",
        decide([row(9.0, False)] * 6 + [row(1.0, True)], 7, base)[0],
        "KID_NOT_PREDICTIVE")
    chk("子集 6 对 4 一致 -> NOT_PREDICTIVE",
        decide([row(9.0, True)] * 4 + [row(9.0, False)] * 2 + [row(1.0, True)], 7, base)[0],
        "KID_NOT_PREDICTIVE")
    # 11-13 门槛边界（0.0002 纪律：差一点也是没过）
    chk("|dKID| 恰好 4.656 算进子集",
        decide([row(4.656, True)] * 3 + [row(0.1, True)] * 4, 7, base)[1]["n_subset"], 3)
    chk("|dKID| 4.6559 不算进子集",
        decide([row(4.6559, True)] * 3 + [row(0.1, True)] * 4, 7, base)[1]["n_subset"], 0)
    chk("负的 |dKID| 也按绝对值进子集",
        decide([row(-9.0, True)] * 3 + [row(0.1, True)] * 4, 7, base)[1]["n_subset"], 3)
    # 14-19 统计函数
    chk("binom_test(6,6)", round(binom_test(6, 6), 6), 0.03125)
    chk("binom_test(3,3)", round(binom_test(3, 3), 6), 0.25)
    chk("binom_test(5,7)", round(binom_test(5, 7), 6), 0.453125)
    rho, p = spearman([1, 2, 3, 4], [1, 2, 3, 4])
    chk("spearman 完全同向 rho", round(rho, 6), 1.0)
    chk("spearman 完全同向 p", round(p, 6), round(2 / 24, 6))
    rho, _ = spearman([1, 2, 3, 4], [4, 3, 2, 1])
    chk("spearman 完全反向 rho", round(rho, 6), -1.0)
    rho, _ = spearman([1, 2, 3], [1, 1, 1])
    chk("spearman 常数列 -> nan", math.isnan(rho), True)
    chk("_ranks 并列取平均", _ranks([5, 5, 1]), [2.5, 2.5, 1.0])
    # 20-22 符号定义自检：dKID = KID(B)-KID(A) > 0 表示 A 的 KID 更低（更好）
    chk("A 的 KID 低且 A 赢 -> 一致", (5.0 > 0) == (0.6 > 0.5), True)
    chk("A 的 KID 低但 A 输 -> 不一致", (5.0 > 0) == (0.4 > 0.5), False)
    chk("A 的 KID 高且 A 输 -> 一致", (-5.0 > 0) == (0.4 > 0.5), True)
    # 23-25 配对表本身
    chk("配对表 7 条", len(PAIRS), 7)
    chk("配对表无重复判官臂", len({p[0] for p in PAIRS}), 7)
    chk("门槛未被改动", THR_KID, 4.656)
    # 26-28 缺数据不许冒充测过
    chk("空 rows + 全 OP 过 -> VOID_NO_DATA",
        decide([], 7, base)[0], "VOID_NO_DATA")
    chk("VOID 时不得给出 k", "k" in decide([row(1.0, True)] * 7, 7, base)[1], False)
    chk("子集恰 3 对全一致时 k=3", decide([row(9.0, True)] * 3 + [row(1.0, True)] * 4,
                                          7, base)[1]["k"], 3)

    print(f"\n自检 {ok}/{ok + fail} 通过")
    return 0 if fail == 0 else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "m51_verdict.json"))
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    try:
        from recheck_judge import EXPECT
    except Exception as e:                                    # pragma: no cover
        print(f"无法 import recheck_judge.EXPECT: {e}")
        return 2

    ops = {"op1_files_ok": True, "op2_judge_ok": True, "op4_kid_finite": True}
    rows, dropped, notes = [], [], []
    for jname, mpath, ka, kb in PAIRS:
        jp = ROOT / "experiments" / jname
        mp = ROOT / mpath
        if not jp.exists() or not mp.exists():
            ops["op1_files_ok"] = False
            notes.append(f"缺文件: {jname if not jp.exists() else mpath}")
            continue
        got = recount_judge(jp)
        exp = EXPECT.get(jname)
        if exp is None or got[:2] != exp[:2]:
            ops["op2_judge_ok"] = False
            notes.append(f"{jname}: 重算 {got[:2]} != EXPECT {None if exp is None else exp[:2]}")
        m = json.loads(mp.read_text(encoding="utf-8"))
        if ka not in m or kb not in m:
            ops["op1_files_ok"] = False
            notes.append(f"{mpath}: 缺键 {ka if ka not in m else kb}")
            continue
        A, B = m[ka], m[kb]
        kid_a, kid_b = A["KID_x1e3"], B["KID_x1e3"]
        if not (math.isfinite(kid_a) and math.isfinite(kid_b)):
            ops["op4_kid_finite"] = False
            notes.append(f"{jname}: KID 非有限")
            continue
        same_n = (A.get("n") == B.get("n") and A.get("materials") == B.get("materials"))
        dkid = kid_b - kid_a
        wr = got[0] / got[1] if got[1] else float("nan")
        rec = {"arm": jname, "A": ka, "B": kb, "metrics": mpath,
               "KID_A": kid_a, "KID_B": kid_b, "dKID": dkid,
               "abs_dKID_over_thr": abs(dkid) / THR_KID,
               "a_wins": got[0], "decided": got[1], "winrate_A": wr,
               "n_A": A.get("n"), "n_B": B.get("n"),
               "mat_A": A.get("materials"), "mat_B": B.get("materials"),
               "op3_same_denom": same_n,
               "sign_agree": (dkid > 0) == (wr > 0.5)}
        if not same_n:
            rec["drop"] = "(OP3) 分母不等"
            dropped.append(rec)
        elif dkid == 0:
            rec["drop"] = "(OP5) dKID == 0"
            dropped.append(rec)
        else:
            rows.append(rec)

    verdict, detail = decide(rows, len(PAIRS), ops)

    diag = {}
    if len(rows) >= 3:
        rho, p = spearman([r["dKID"] for r in rows],
                          [r["winrate_A"] - 0.5 for r in rows])
        k_all = sum(1 for r in rows if r["sign_agree"])
        diag = {"spearman_rho_all": rho, "spearman_perm_p_all": p,
                "sign_agree_all": f"{k_all}/{len(rows)}",
                "sign_agree_all_p": binom_test(k_all, len(rows)),
                "max_abs_dKID": max(abs(r["dKID"]) for r in rows),
                "n_over_thr": sum(1 for r in rows if abs(r["dKID"]) >= THR_KID)}

    out = {"round": "M51", "question": "拆解尺度的 KID 差是否预测判官胜率",
           "thr_kid": THR_KID, "thr_source": "(M33) noise_floor.py m=1 不成对",
           "n_pairs_listed": len(PAIRS), "n_pairs_valid": len(rows),
           "ops": ops, "notes": notes,
           "verdict": verdict, "detail": detail,
           "rows": rows, "dropped": dropped, "diagnostics": diag}
    # 落盘一律在打印之前
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"料：列出 {len(PAIRS)} 对，有效 {len(rows)} 对，作废 {len(dropped)} 对")
    for n in notes:
        print("  【禁】", n)
    print(f"{'臂':52s} {'KID_A':>8s} {'KID_B':>8s} {'dKID':>8s} {'/thr':>6s} "
          f"{'胜率A':>7s} {'同向':>5s}")
    for r in rows + dropped:
        tag = r["arm"].replace("judge_full_", "").replace(".json", "")
        mark = "是" if r["sign_agree"] else "否"
        if "drop" in r:
            mark = "作废"
        print(f"{tag:52s} {r['KID_A']:8.3f} {r['KID_B']:8.3f} {r['dKID']:8.3f} "
              f"{r['abs_dKID_over_thr']:6.2f} {r['winrate_A']:7.1%} {mark:>5s}")
    if diag:
        print(f"\n诊断（不作判决）：全 {len(rows)} 对 Spearman rho="
              f"{diag['spearman_rho_all']:.3f} 置换 p={diag['spearman_perm_p_all']}; "
              f"符号一致 {diag['sign_agree_all']} p={diag['sign_agree_all_p']:.3g}; "
              f"最大 |dKID|={diag['max_abs_dKID']:.3f} "
              f"(门槛 {THR_KID} 的 {diag['max_abs_dKID'] / THR_KID:.2f} 倍); "
              f"过门槛 {diag['n_over_thr']} 对")
    print(f"\n判决：{verdict}  {detail}")
    print(f"已写 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
