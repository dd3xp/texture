"""(M59) 判读器：把 (M58) 的 `GAP32C_TIE` 拿到**配对**尺度上重问一遍。

盲写 —— 在任何 `--per_image` 读数存在之前写完并提交（见 `docs/arch_progress.md` 的 (M59) 预注册）。
判据一个字不许改；运行前先 `python analysis/arch/m59_read_pairedclip.py --selftest`。

背景：(M58) 的零分布 0.2513 是**跨种子行均值的极差**，而 dS/dP 是**跑内成对**差
（五行共用同一张网格）⇒ 那个 NULL 对成对差明显偏保守。本轮让 `evaluate()` 落逐图 CLIP，
于是 TRD 项在 dS-dP 里**逐图抵消**，直接对 `struct=real` 与 `palette=real` 做配对符号检验。

  D_m = mean_{seed,rep}[ CLIP(struct=real) - CLIP(palette=real) ]   （按材质聚类，n = 材质数）

聚类单位＝材质（`_mats` 里的提示词），⛔ 不是 148 张图：74 个目标里有 7 个提示词各出现两次，
且 reps=2 ⇒ 图与图之间不独立（(M14)/(M15)/(M16) 那条纪律）。
"""
import argparse
import json
import math
from pathlib import Path

ROWS = ("TRD", "struct=real", "palette=real")
SHAM_ROWS = ("TRD", "struct=real", "palette=real", "palette=xmodal", "palette=retrieved")
SEEDS = (0, 1, 2, 3, 4)
N_IMG = 148                 # 74 个 V_mat@32 目标 x reps=2
N_MAT = 67                  # 其中不同提示词的个数
ALPHA = 0.05                # 主判据显著性
SHAM_ALPHA = 0.01           # (OP4) 假阳门：5 条 sham 任一低于它即 VOID
TOL_SELF = 1e-4             # (OP2) mean(_CLIP_per) 对 CLIP
TOL_REPRO = 1e-3            # (OP3) 与 (M58) 已发表行均值


# ---------------------------------------------------------------- 统计
def sign_test(d):
    """d: list[float]。返回 (pos, neg, n_dec, p_two_sided)。零假设 P(>0)=0.5，精确二项。"""
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 0, 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return pos, neg, n, min(1.0, 2.0 * tail)


def by_material(per_img, mats):
    """逐图读数 -> {材质: 该材质内的均值}。"""
    acc = {}
    for v, m in zip(per_img, mats):
        acc.setdefault(m, []).append(float(v))
    return {m: sum(v) / len(v) for m, v in acc.items()}


def contrast(runs, row_a, row_b, seeds_a, seeds_b):
    """逐材质配对差：row_a 在 seeds_a 上的均值 - row_b 在 seeds_b 上的均值。
    runs: {seed: {"mats": [...], "rows": {row: [逐图]}}}。返回 {材质: 差值}。"""
    def agg(row, seeds):
        per_seed = [by_material(runs[s]["rows"][row], runs[s]["mats"]) for s in seeds]
        keys = set(per_seed[0])
        for d in per_seed[1:]:
            keys &= set(d)
        return {m: sum(d[m] for d in per_seed) / len(per_seed) for m in keys}
    A, B = agg(row_a, seeds_a), agg(row_b, seeds_b)
    return {m: A[m] - B[m] for m in sorted(set(A) & set(B))}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ---------------------------------------------------------------- 判读
def read(runs, m58=None):
    """runs: {seed: {"mats": [...], "rows": {...}, "clip": {row: 行均值}}}
    m58:  {seed: {row: 行均值}}（(M58) 已发表读数）或 None（则 (OP3) 记 SKIP）。
    返回结果 dict（含 verdict）。判据全在这一个函数里，⛔ 调用方不得再改。"""
    out = {"ops": {}, "checked": {}}

    # (OP1) 料齐
    miss = [s for s in SEEDS if s not in runs]
    bad = []
    for s in SEEDS:
        if s not in runs:
            continue
        r = runs[s]
        if len(r["mats"]) != N_IMG:
            bad.append(f"seed{s}:mats={len(r['mats'])}")
        if len(set(r["mats"])) != N_MAT:
            bad.append(f"seed{s}:materials={len(set(r['mats']))}")
        for row in ROWS:
            if row not in r["rows"]:
                bad.append(f"seed{s}:{row}:absent")
            elif len(r["rows"][row]) != N_IMG:
                bad.append(f"seed{s}:{row}:n={len(r['rows'][row])}")
    out["checked"]["seeds_present"] = sorted(runs)
    out["ops"]["OP1"] = {"ok": not miss and not bad, "missing_seeds": miss, "bad": bad}
    if not out["ops"]["OP1"]["ok"]:
        out["verdict"] = "VOID_NO_DATA"
        return out

    # (OP2) 逐图列自洽：mean(_CLIP_per) == 报告的 CLIP
    off = []
    for s in SEEDS:
        for row in ROWS:
            d = abs(mean(runs[s]["rows"][row]) - runs[s]["clip"][row])
            if d > TOL_SELF:
                off.append({"seed": s, "row": row, "abs_diff": d})
    out["ops"]["OP2"] = {"ok": not off, "tol": TOL_SELF, "off": off,
                         "n_checked": len(SEEDS) * len(ROWS)}
    if not out["ops"]["OP2"]["ok"]:
        out["verdict"] = "VOID_PER_IMAGE_MISMATCH"
        return out

    # (OP3) 与 (M58) 逐位复现（生成逐像素确定 => 行均值应当对上）
    if m58 is None:
        out["ops"]["OP3"] = {"ok": True, "skipped": True, "note": "no M58 reference given"}
    else:
        off = []
        n_ck = 0
        for s in SEEDS:
            for row in ROWS:
                if s not in m58 or row not in m58.get(s, {}):
                    off.append({"seed": s, "row": row, "abs_diff": None, "why": "absent_in_M58"})
                    continue
                n_ck += 1
                d = abs(runs[s]["clip"][row] - m58[s][row])
                if d > TOL_REPRO:
                    off.append({"seed": s, "row": row, "abs_diff": d})
        out["ops"]["OP3"] = {"ok": not off, "tol": TOL_REPRO, "off": off, "n_checked": n_ck}
        if not out["ops"]["OP3"]["ok"]:
            out["verdict"] = "VOID_NOT_REPRODUCED"
            return out

    # (OP4) 假阳检验：同一行、不相交的种子组 => 真值恒为 0
    shams = {}
    for row in SHAM_ROWS:
        if not all(row in runs[s]["rows"] for s in (0, 1, 2, 3)):
            continue
        d = contrast(runs, row, row, (0, 1), (2, 3))
        pos, neg, n, p = sign_test(list(d.values()))
        shams[row] = {"pos": pos, "neg": neg, "n": n, "p": p, "mean": mean(d.values())}
    fired = [r for r, v in shams.items() if v["p"] < SHAM_ALPHA]
    out["sham"] = shams
    out["NULL_paired_mean"] = max((abs(v["mean"]) for v in shams.values()), default=float("nan"))
    out["ops"]["OP4"] = {"ok": len(shams) >= 3 and not fired, "alpha": SHAM_ALPHA,
                         "n_sham": len(shams), "fired": fired}
    if not out["ops"]["OP4"]["ok"]:
        out["verdict"] = "VOID_PAIRED_FALSEPOS" if fired else "VOID_NO_DATA"
        return out

    # 主判据：struct=real vs palette=real，逐材质配对（TRD 项逐图抵消）
    D = contrast(runs, "struct=real", "palette=real", SEEDS, SEEDS)
    pos, neg, n, p = sign_test(list(D.values()))
    out["main"] = {"n_materials": n, "pos_struct": pos, "pos_palette": neg,
                   "frac_struct": pos / n if n else float("nan"), "p": p,
                   "mean_D": mean(D.values())}
    if p < ALPHA and pos > neg:
        out["verdict"] = "GAP32C_STRUCT"
    elif p < ALPHA and neg > pos:
        out["verdict"] = "GAP32C_PALETTE"
    else:
        out["verdict"] = "GAP32C_TIE_PAIRED"

    # 只登记、不下判
    dS = contrast(runs, "struct=real", "TRD", SEEDS, SEEDS)
    dP = contrast(runs, "palette=real", "TRD", SEEDS, SEEDS)
    pS, nS, _, ppS = sign_test(list(dS.values()))
    pP, nP, _, ppP = sign_test(list(dP.values()))
    out["registered"] = {
        "dS_bar": mean(dS.values()), "dS_pos": pS, "dS_neg": nS, "dS_p": ppS,
        "dP_bar": mean(dP.values()), "dP_pos": pP, "dP_neg": nP, "dP_p": ppP,
        "per_seed_D": {s: mean(contrast(runs, "struct=real", "palette=real", (s,), (s,)).values())
                       for s in SEEDS}}
    return out


# ---------------------------------------------------------------- 载入
def load(dirp, tag):
    runs = {}
    for s in SEEDS:
        p = Path(dirp) / f"{tag}_s{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows = {k: v["_CLIP_per"] for k, v in d.items()
                if isinstance(v, dict) and "_CLIP_per" in v}
        clip = {k: v["CLIP"] for k, v in d.items() if isinstance(v, dict) and "CLIP" in v}
        if "_mats" not in d or not rows:
            continue
        runs[s] = {"mats": d["_mats"], "rows": rows, "clip": clip}
    return runs


def load_m58(dirp, tag):
    ref = {}
    for s in SEEDS:
        p = Path(dirp) / f"{tag}_s{s}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        ref[s] = {k: v["CLIP"] for k, v in d.items() if isinstance(v, dict) and "CLIP" in v}
    return ref or None


# ---------------------------------------------------------------- selftest
def _mk(seed_offsets, mats=None, n_img=N_IMG, rows=SHAM_ROWS):
    """造一份假 runs：row 的逐图值 = base + 偏置，便于验判据分支。"""
    mats = mats or [f"m{i % N_MAT}" for i in range(n_img)]
    runs = {}
    for s in SEEDS:
        r = {}
        for row in rows:
            base = [10.0 + 0.001 * ((i * 37) % 23) for i in range(n_img)]   # 与 seed 无关 => sham 真值 0
            off = seed_offsets(row, s)
            r[row] = [b + off(i) for i, b in enumerate(base)]
        runs[s] = {"mats": mats, "rows": r, "clip": {k: mean(v) for k, v in r.items()}}
    return runs


def selftest():
    ok = []

    def ck(name, cond):
        ok.append((name, bool(cond)))

    # 1-3 精确二项
    ck("binom_all_pos", abs(sign_test([1] * 10)[3] - 2 / 1024) < 1e-12)
    ck("binom_balanced", abs(sign_test([1] * 5 + [-1] * 5)[3] - 1.0) < 1e-12)
    ck("binom_ties_dropped", sign_test([1, -1, 0, 0])[2] == 2)
    # 4 按材质聚类
    bm = by_material([1.0, 3.0, 5.0], ["a", "a", "b"])
    ck("by_material", bm["a"] == 2.0 and bm["b"] == 5.0)

    flat = (lambda row, s: (lambda i: 0.0))
    # 5 平局
    r = read(_mk(flat))
    ck("tie_when_identical", r["verdict"] == "GAP32C_TIE_PAIRED")
    # 6 结构侧全赢
    r = read(_mk(lambda row, s: (lambda i: 0.5 if row == "struct=real" else 0.0)))
    ck("struct_wins", r["verdict"] == "GAP32C_STRUCT" and r["main"]["pos_struct"] == N_MAT)
    # 7 配色侧全赢
    r = read(_mk(lambda row, s: (lambda i: 0.5 if row == "palette=real" else 0.0)))
    ck("palette_wins", r["verdict"] == "GAP32C_PALETTE" and r["main"]["pos_palette"] == N_MAT)
    # 8 小效应但一致 => 仍显著（配对的意义）
    r = read(_mk(lambda row, s: (lambda i: 1e-4 if row == "struct=real" else 0.0)))
    ck("tiny_consistent_is_significant", r["verdict"] == "GAP32C_STRUCT")
    # 9 (OP1) 缺种子
    runs = _mk(flat)
    runs.pop(4)
    ck("op1_missing_seed", read(runs)["verdict"] == "VOID_NO_DATA")
    # 10 (OP1) 图数不对
    runs = _mk(flat, mats=[f"m{i % N_MAT}" for i in range(N_IMG - 1)], n_img=N_IMG - 1)
    ck("op1_wrong_n", read(runs)["verdict"] == "VOID_NO_DATA")
    # 11 (OP2) 逐图列与 CLIP 对不上
    runs = _mk(flat)
    runs[2]["clip"]["TRD"] += 0.01
    ck("op2_mismatch", read(runs)["verdict"] == "VOID_PER_IMAGE_MISMATCH")
    # 12 (OP3) 与 M58 对不上
    runs = _mk(flat)
    ref = {s: {row: runs[s]["clip"][row] for row in ROWS} for s in SEEDS}
    ck("op3_pass", read(runs, ref)["verdict"] == "GAP32C_TIE_PAIRED")
    ref[1]["struct=real"] += 0.5
    ck("op3_fail", read(runs, ref)["verdict"] == "VOID_NOT_REPRODUCED")
    # 13 (OP4) 假阳：种子 0/1 与 2/3 系统性不同
    runs = _mk(lambda row, s: (lambda i: 0.5 if s in (0, 1) else 0.0))
    ck("op4_falsepos", read(runs)["verdict"] == "VOID_PAIRED_FALSEPOS")
    # 14 TRD 在主判据里逐图抵消（给 TRD 加任意偏置不改主判决）
    a = read(_mk(lambda row, s: (lambda i: 0.3 if row == "palette=real" else 0.0)))
    b = read(_mk(lambda row, s: (lambda i: 0.3 if row == "palette=real" else (7.0 if row == "TRD" else 0.0))))
    ck("trd_cancels", a["main"]["mean_D"] == b["main"]["mean_D"] and a["verdict"] == b["verdict"])
    # 15 dS/dP 登记项方向
    r = read(_mk(lambda row, s: (lambda i: 0.4 if row == "struct=real" else 0.0)))
    ck("registered_dS", r["registered"]["dS_bar"] > 0.39 and abs(r["registered"]["dP_bar"]) < 1e-9)

    for name, good in ok:
        print(f"  [{'ok' if good else 'FAIL'}] {name}")
    n_ok = sum(1 for _, g in ok if g)
    print(f"selftest {n_ok}/{len(ok)}")
    return n_ok == len(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp")
    ap.add_argument("--tag", default="m59_pairedclip")
    ap.add_argument("--m58dir", default=None, help="(M58) 五份 JSON 所在目录（做 (OP3) 复现检验）")
    ap.add_argument("--m58tag", default="m58_clipdecomp")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    runs = load(a.dir, a.tag)
    m58 = load_m58(a.m58dir, a.m58tag) if a.m58dir else None
    res = read(runs, m58)
    if a.out:                      # 落盘一律挪到打印之前
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
