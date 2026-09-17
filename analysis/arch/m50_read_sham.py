"""(M50) 判读器：拆解尺子的经验零分布（sham 重排器）。盲写于任何读数之前。

判据见 docs/arch_progress.md 的 (M50) 预注册节，本文件一个字不许在看到读数之后改。

  G(row, s) = KID(palette=xmodal, s) - KID(row, s)
  零分布    = {G(sham_j, s)}，每个 sham 在同一 5 张候选里独立均匀重抽一张
              （正式配置那张也是均匀随机 => 可交换 => G 真值恒为 0）
  主臂      = rr_trdpal 一行，秩检验（精确置换检验）

  pass1 = G(rr_trdpal, seed1) 严格大于全部 24 个 sham   -> p = 1/25 = 0.04   （主判据，全新数据）
  pass0 = G(rr_trdpal, seed0) 严格大于全部 12 个 sham   -> p = 1/13 = 0.077  （次要，复现那一跑）

  pass1 and pass0 -> SHAM_BEATEN        这把尺子分得开，重排器这条路重开
  pass1 xor pass0 -> SHAM_UNSTABLE      不复制，【禁】当证据
  两个都不过      -> SHAM_WITHIN_NULL   连零分布都分不开，这条路关闭

任一操作检验不过 -> VOID_*，一个差值都不许引。

用法：
  python analysis/arch/m50_read_sham.py --selftest
  python analysis/arch/m50_read_sham.py --s0 <seed0.json> --s1 <seed1.json> --out <verdict.json>
"""
import argparse
import json
from pathlib import Path
from statistics import median

# (M49) 已发表读数（experiments/m49_rerank_v8_Vmat.json），(OP1) 必须复现，容差 +-0.5
EXPECT_M49 = {
    "TRD": 38.909, "struct=real": 33.261, "palette=real": 6.894,
    "palette=retrieved": 12.843, "palette=xmodal": 13.431, "palette=incbest5": 8.326,
    "palette=rr_argmax": 16.902, "palette=rr_trdpal": 9.876, "palette=rr_incTRD": 16.376,
    "real_half": 3.791,
}
TOL = 0.5
N_EXPECT = 392                 # 196 张参照瓦片 x reps 2
N_SHAM = {"s0": 12, "s1": 24}  # 预注册写死：seed0 十二个、seed1 二十四个
ARM = "palette=rr_trdpal"
XM = "palette=xmodal"
CHANGE_LO, CHANGE_HI = 0.6, 0.95
MEDIAN_MAX = 1.0
BORROWED_THR = 4.656           # (M49) 借来的那把门，只作并排对照，【禁】当本轮判据


def sham_names(d):
    ns = [k for k in d if k.startswith("palette=sham")]
    return sorted(ns, key=lambda s: int(s.split("sham")[1]))


def kid(d, row):
    return float(d[row]["KID_x1e3"])


def analyse(d0, d1):
    """d0 = seed 0 的 JSON，d1 = seed 1 的 JSON；返回判决字典。缺数据一律 VOID_NO_DATA。"""
    out = {"verdict": None, "ops": {}, "checked": 0, "note": []}
    packs = {"s0": d0, "s1": d1}

    # (OP5) 料齐：两个 JSON 都在、行都在、n 对、sham 个数对
    bad = []
    for tag, d in packs.items():
        if not isinstance(d, dict) or not d:
            bad.append(f"{tag}: 空")
            continue
        for row in list(EXPECT_M49) + [ARM, XM]:
            if row not in d:
                bad.append(f"{tag}: 缺行 {row}")
            elif row != "real_half" and int(d[row].get("n", -1)) != N_EXPECT:
                bad.append(f"{tag}: {row} 的 n={d[row].get('n')} 不是 {N_EXPECT}")
        got = len(sham_names(d))
        if got != N_SHAM[tag]:
            bad.append(f"{tag}: sham 行 {got} 个，预注册要 {N_SHAM[tag]} 个")
    out["ops"]["OP5_data"] = {"ok": not bad, "problems": bad[:8]}
    out["checked"] += 1
    if bad:
        out["verdict"] = "VOID_NO_DATA"
        return out

    sh = {t: sham_names(packs[t]) for t in packs}

    # (OP1) seed 0 必须复现 (M49) 十行
    rep = {}
    for row, exp in EXPECT_M49.items():
        rep[row] = round(kid(d0, row) - exp, 4)
    op1 = all(abs(v) <= TOL for v in rep.values())
    out["ops"]["OP1_repro"] = {"ok": op1, "tol": TOL, "delta_vs_M49": rep}
    out["checked"] += 1

    # (OP2) sham 真的在随机挑
    cr, degen = {}, []
    for tag, d in packs.items():
        rr = d.get("_rr", {})
        for s in sh[tag]:
            c = rr.get(s, {}).get("change_rate")
            cr[f"{tag}:{s}"] = c
            if c is None or not (CHANGE_LO <= float(c) <= CHANGE_HI):
                degen.append(f"{tag}:{s} change_rate={c}")
        vals = {round(kid(d, s), 6) for s in sh[tag]}
        if len(vals) * 6 < len(sh[tag]) * 5:
            degen.append(f"{tag}: sham KID 去重 {len(vals)}/{len(sh[tag])} 过少")
    out["ops"]["OP2_sham_random"] = {"ok": not degen, "problems": degen[:8],
                                     "change_rate": cr}
    out["checked"] += 1

    # (OP3) 同一候选集
    sub = {t: bool(packs[t].get("_rr_subset_ok", False)) for t in packs}
    out["ops"]["OP3_subset"] = {"ok": all(sub.values()), "subset_ok": sub}
    out["checked"] += 1

    # G 与零分布
    G, shamG = {}, {}
    for tag, d in packs.items():
        base = kid(d, XM)
        G[tag] = round(base - kid(d, ARM), 4)
        shamG[tag] = [round(base - kid(d, s), 4) for s in sh[tag]]
    allsham = shamG["s0"] + shamG["s1"]

    # (OP4) 零分布居中
    med = round(median(allsham), 4)
    op4 = abs(med) <= MEDIAN_MAX
    out["ops"]["OP4_sham_centred"] = {"ok": op4, "median_G": med, "max_abs": MEDIAN_MAX}
    out["checked"] += 1

    out["G_rr_trdpal"] = G
    out["sham_G"] = shamG
    out["sham_stats"] = {t: {"n": len(shamG[t]), "min": min(shamG[t]), "max": max(shamG[t]),
                             "max_abs": round(max(abs(x) for x in shamG[t]), 4)} for t in shamG}
    out["sham_stats"]["pooled_max_abs"] = round(max(abs(x) for x in allsham), 4)
    out["borrowed_thr_M49"] = BORROWED_THR

    # 诊断：另两行的秩（只记账，不作判决）
    diag = {}
    for row in ("palette=rr_argmax", "palette=rr_incTRD"):
        diag[row] = {t: {"G": round(kid(packs[t], XM) - kid(packs[t], row), 4),
                         "rank_p": round((1 + sum(
                             1 for x in shamG[t]
                             if x >= kid(packs[t], XM) - kid(packs[t], row))) / (len(shamG[t]) + 1), 4)}
                     for t in packs}
    out["diag_other_arms"] = diag
    # 诊断：量尺检验本体 = 绝对 KID 的运行间抖动（【禁】当 G 的门槛）
    out["run_to_run_absKID"] = {row: round(abs(kid(d0, row) - kid(d1, row)), 4)
                                for row in EXPECT_M49}

    for bad_op, tag in (("OP1_repro", "VOID_NO_REPRO"),
                        ("OP2_sham_random", "VOID_SHAM_DEGENERATE"),
                        ("OP3_subset", "VOID_SELECTION_BUG"),
                        ("OP4_sham_centred", "VOID_SHAM_BIASED")):
        if not out["ops"][bad_op]["ok"]:
            out["verdict"] = tag
            return out

    p = {t: round((1 + sum(1 for x in shamG[t] if x >= G[t])) / (len(shamG[t]) + 1), 4)
         for t in packs}
    out["rank_p"] = p
    ps = {t: G[t] > max(shamG[t]) for t in packs}
    out["pass"] = ps
    out["verdict"] = ("SHAM_BEATEN" if ps["s0"] and ps["s1"] else
                      "SHAM_UNSTABLE" if ps["s0"] or ps["s1"] else "SHAM_WITHIN_NULL")
    if out["verdict"] == "SHAM_BEATEN":
        out["note"].append("SHAM_BEATEN 只说明这把尺子分得开；【禁】直接开判官臂，"
                           "须另行预注册并先过 power_curve 的『预期 >=65%』门")
    if out["verdict"] == "SHAM_UNSTABLE":
        out["note"].append("只有一个种子过 = 非复制；【禁】当证据")
    out["note"].append("(M49) 的 RERANK_TOO_SMALL 判决不因本轮改写；本轮问的是另一个问题")
    return out


def _row(k, n=N_EXPECT):
    return {"n": n, "KID_x1e3": k}


def _pack(base, arm, shams, tag="s1", change=0.8, subset=True, extra=None):
    d = {r: _row(v) for r, v in EXPECT_M49.items()}
    d["real_half"] = _row(EXPECT_M49["real_half"], 98)
    d[XM] = _row(base)
    d[ARM] = _row(base - arm)
    d["_rr"] = {}
    for j, g in enumerate(shams):
        d[f"palette=sham{j + 1}"] = _row(base - g)
        d["_rr"][f"palette=sham{j + 1}"] = {"change_rate": change}
    d["_rr_subset_ok"] = subset
    if extra:
        d.update(extra)
    return d


def selftest():
    n = 0

    def ck(cond, msg):
        nonlocal n
        assert cond, msg
        n += 1

    # ⚠ (OP1) 逼着 seed 0 的 rr_trdpal 必须贴近 9.876 => 合成用例只能改零分布的宽度，不能改 arm
    B, A = 13.431, 3.555                               # 13.431 - 3.555 = 9.876
    sm0 = [0.1 * (j - 6) for j in range(12)]           # -0.6 .. +0.5，窄
    sm1 = [0.1 * (j - 12) for j in range(24)]          # -1.2 .. +1.1，窄
    w0 = [0.8 * (j - 6) for j in range(12)]            # -4.8 .. +4.0，宽（max > A）
    w1 = [0.4 * (j - 12) for j in range(24)]           # -4.8 .. +4.4，宽（max > A）
    # 两个种子都远超零分布 -> SHAM_BEATEN
    r = analyse(_pack(B, A, sm0), _pack(B, A, sm1))
    ck(r["verdict"] == "SHAM_BEATEN", "两个都过应为 SHAM_BEATEN")
    ck(r["pass"] == {"s0": True, "s1": True}, "pass 两个都 True")
    ck(r["rank_p"]["s1"] == round(1 / 25, 4), "seed1 秩 p 应为 1/25")
    ck(r["rank_p"]["s0"] == round(1 / 13, 4), "seed0 秩 p 应为 1/13")
    ck(r["checked"] == 5, "应报已查 5 项")
    ck(abs(r["G_rr_trdpal"]["s1"] - A) < 1e-3, "G 应为 xmodal - arm")
    ck(r["sham_stats"]["s1"]["n"] == 24, "seed1 零分布 24 个")
    ck(r["borrowed_thr_M49"] == 4.656, "要并排报出借来的门")
    ck("power_curve" in " ".join(r["note"]), "SHAM_BEATEN 必须带『不许直接开判官臂』注记")

    # 只有 seed1 过 -> SHAM_UNSTABLE
    r = analyse(_pack(B, A, w0), _pack(B, A, sm1))
    ck(r["verdict"] == "SHAM_UNSTABLE", "一个过一个不过应为 SHAM_UNSTABLE")
    ck(r["pass"] == {"s0": False, "s1": True}, "该过的那个种子要标对")
    r = analyse(_pack(B, A, sm0), _pack(B, A, w1))
    ck(r["verdict"] == "SHAM_UNSTABLE", "反过来也应为 SHAM_UNSTABLE")

    # 两个都不过 -> SHAM_WITHIN_NULL
    r = analyse(_pack(B, A, w0), _pack(B, A, w1))
    ck(r["verdict"] == "SHAM_WITHIN_NULL", "都不过应为 SHAM_WITHIN_NULL")
    ck(r["rank_p"]["s1"] > 0.04, "不过时秩 p 必然大于 1/25")

    # 严格大于：恰好打平也算不过
    r = analyse(_pack(B, A, sm0[:11] + [A]), _pack(B, A, sm1[:23] + [A]))
    ck(r["verdict"] == "SHAM_WITHIN_NULL", "与最大值打平不算过")

    # VOID：复现不过
    bad = _pack(B, A, sm0)
    bad["TRD"] = _row(50.0)
    ck(analyse(bad, _pack(B, A, sm1))["verdict"] == "VOID_NO_REPRO", "复现不过应 VOID_NO_REPRO")
    # 容差内不算不过
    ok = _pack(B, A, sm0)
    ok["TRD"] = _row(EXPECT_M49["TRD"] + 0.4)
    ck(analyse(ok, _pack(B, A, sm1))["verdict"] == "SHAM_BEATEN", "容差内应放行")

    # VOID：sham change_rate 越界
    ck(analyse(_pack(B, A, sm0, change=0.2), _pack(B, A, sm1))["verdict"]
       == "VOID_SHAM_DEGENERATE", "change_rate 过低应 VOID_SHAM_DEGENERATE")
    # VOID：sham KID 全一样
    ck(analyse(_pack(B, A, [0.0] * 12), _pack(B, A, sm1))["verdict"]
       == "VOID_SHAM_DEGENERATE", "sham 退化应 VOID_SHAM_DEGENERATE")
    # VOID：候选集越界
    ck(analyse(_pack(B, A, sm0, subset=False), _pack(B, A, sm1))["verdict"]
       == "VOID_SELECTION_BUG", "subset_ok=False 应 VOID_SELECTION_BUG")
    # VOID：零分布偏心（arm 仍合规，逼它只能从 OP4 出局）
    ck(analyse(_pack(B, A, [x + 5 for x in sm0]),
               _pack(B, A, [x + 5 for x in sm1]))["verdict"]
       == "VOID_SHAM_BIASED", "零分布偏心应 VOID_SHAM_BIASED")

    # VOID：缺数据的三种形态，且必须另报已查几项
    ck(analyse({}, _pack(B, A, sm1))["verdict"] == "VOID_NO_DATA", "空包应 VOID_NO_DATA")
    r = analyse(_pack(B, A, sm0[:11]), _pack(B, A, sm1))
    ck(r["verdict"] == "VOID_NO_DATA", "sham 个数不对应 VOID_NO_DATA")
    ck(r["checked"] == 1, "缺数据时必须另报已查项数")
    short = _pack(B, A, sm0)
    short["TRD"] = _row(EXPECT_M49["TRD"], 250)
    ck(analyse(short, _pack(B, A, sm1))["verdict"] == "VOID_NO_DATA", "n 不对应 VOID_NO_DATA")

    # 诊断格存在且不影响判决
    r = analyse(_pack(B, A, sm0), _pack(B, A, sm1))
    ck("palette=rr_argmax" in r["diag_other_arms"], "要报另两行的秩")
    ck(set(r["run_to_run_absKID"]) == set(EXPECT_M49), "要报逐行运行间抖动")
    ck(r["run_to_run_absKID"]["TRD"] == 0.0, "同值时运行间抖动为 0")
    print(f"selftest {n}/{n} 全过")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=Path, help="seed 0 的 JSON（--sham 12）")
    ap.add_argument("--s1", type=Path, help="seed 1 的 JSON（--sham 24）")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not (a.s0 and a.s1):
        ap.error("要 --s0 与 --s1，或 --selftest")

    def rd(p):
        if not p.exists():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))

    r = analyse(rd(a.s0), rd(a.s1))
    r["_src"] = {"s0": str(a.s0), "s1": str(a.s1)}
    if a.out:                                   # 落盘一律挪到打印之前
        a.out.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"判决 {r['verdict']}   已查 {r['checked']} 项操作检验")
    for k, v in r["ops"].items():
        print(f"  {k:<18} {'过' if v['ok'] else '【禁】不过'}")
    if "G_rr_trdpal" in r:
        print(f"  G(rr_trdpal) s0={r['G_rr_trdpal']['s0']:+.4f}  s1={r['G_rr_trdpal']['s1']:+.4f}")
        for t in ("s0", "s1"):
            s = r["sham_stats"][t]
            print(f"  sham({t}) n={s['n']}  G 范围 [{s['min']:+.4f}, {s['max']:+.4f}]  "
                  f"max|G|={s['max_abs']:.4f}")
        print(f"  零分布合并 max|G|={r['sham_stats']['pooled_max_abs']:.4f}   "
              f"对照：(M49) 借来的门 {r['borrowed_thr_M49']}")
    if "rank_p" in r:
        print(f"  秩 p  s0={r['rank_p']['s0']}  s1={r['rank_p']['s1']}   pass={r['pass']}")
    for s in r["note"]:
        print("  注记：" + s)
    if a.out:
        print("->", a.out)


if __name__ == "__main__":
    main()
