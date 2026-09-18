"""(M58) 盲写判读器：把 32px 的 CLIP 缺口拆成「结构侧 vs 配色侧」。

跑前提交 = 预注册。判据一个字不许改（判据表见 docs/arch_progress.md 的 (M58) 预注册节）。
输入：五个种子各一份 `diag_decompose.py --size 32 --xmodal` 的 JSON，只读 `CLIP` 这一列。

为什么换 CLIP 这把尺子（(M53) 判的是 KID 那把，⛔ 本轮不复活它的任何读数）：
  CLIP 分数是图文相似度、**不需要真人参照集** ⇒ (M53) 那个失效机理（参照池只有 74 张 ⇒ KID 地板
  自己抖 13.6）在 CLIP 上结构性地不存在；而 (M55)/(M56) 已经在 32px 上拿 CLIP 当实际下判的尺子。
  本轮不借任何门槛：零分布**跨种子自造**（(M50) 那条纪律）。

用法：
  python m58_read_clipdecomp.py --selftest
  python m58_read_clipdecomp.py --dir /tmp --out /tmp/m58_clipdecomp.json
"""
import argparse
import json
import sys
from pathlib import Path

ROWS = ["TRD", "struct=real", "palette=real", "palette=retrieved", "palette=xmodal"]
SEEDS = [0, 1, 2, 3, 4]
N_ROW = 148                 # 74 个 V_mat 32px 目标 x reps 2
GAP_REF = 1.352             # (M55) 量到的 32px CLIP 缺口（v11dx_direct 34.324 vs B2val 35.676）
# 注意：GAP_REF 只用在作废条件 (OP3) 上当「量级参照」（尺子比要拆的东西还抖就整轮作废），
# 口径与本轮五行不同（不同 run、不同管线、不同 n） ⇒ 不许当判据的分母、不许拿它算比例。


def rng_(xs):
    return max(xs) - min(xs)


def mean_(xs):
    return sum(xs) / len(xs)


def analyse(per_seed):
    """per_seed: [{行名: {"CLIP": float, "n": int}}] x 5，顺序＝SEEDS。返回判决 dict。"""
    out = {"n_seeds": len(per_seed), "seeds": SEEDS, "checks": {}}

    # ---- (OP1) 料齐 ----
    if len(per_seed) != len(SEEDS):
        out["checks"]["OP1"] = False
        return dict(out, verdict="VOID_NO_DATA", why=f"种子数 {len(per_seed)} != {len(SEEDS)}")
    for s, d in zip(SEEDS, per_seed):
        for r in ROWS + ["real_half"]:
            if r not in d or "CLIP" not in d[r]:
                out["checks"]["OP1"] = False
                return dict(out, verdict="VOID_NO_DATA", why=f"seed{s} 缺行 {r} 或缺 CLIP 键")
        ns = [d[r].get("n") for r in ROWS]
        if len(set(ns)) != 1:
            out["checks"]["OP1"] = False
            return dict(out, verdict="VOID_NO_DATA", why=f"seed{s} 五行 n 不齐：{ns}")
        if ns[0] != N_ROW:
            out["checks"]["OP1"] = False
            return dict(out, verdict="VOID_NO_DATA", why=f"seed{s} n={ns[0]} != {N_ROW}")
    out["checks"]["OP1"] = True

    trd = [d["TRD"]["CLIP"] for d in per_seed]
    dS = [d["struct=real"]["CLIP"] - t for d, t in zip(per_seed, trd)]
    dP = [d["palette=real"]["CLIP"] - t for d, t in zip(per_seed, trd)]
    ceil_ = [d["real_half"]["CLIP"] for d in per_seed]
    spread = [max(d[r]["CLIP"] for r in ROWS) - min(d[r]["CLIP"] for r in ROWS) for d in per_seed]

    NULL = max(rng_(trd), rng_(dS), rng_(dP))
    dS_bar, dP_bar = mean_(dS), mean_(dP)
    dyn = mean_(spread)
    out.update({
        "TRD_per_seed": trd, "dS_per_seed": dS, "dP_per_seed": dP,
        "TRD_range": rng_(trd), "dS_range": rng_(dS), "dP_range": rng_(dP),
        "NULL": NULL, "dS_bar": dS_bar, "dP_bar": dP_bar, "diff_bar": dS_bar - dP_bar,
        "dyn_range": dyn, "ceiling_real_half": mean_(ceil_), "TRD_bar": mean_(trd),
        # 只登记、不下判：
        "_reg": {
            "P_sel_clip": mean_([d["palette=real"]["CLIP"] - d["palette=xmodal"]["CLIP"]
                                 for d in per_seed]),
            "retrieved_bar": mean_([d["palette=retrieved"]["CLIP"] - t
                                    for d, t in zip(per_seed, trd)]),
            "xmodal_bar": mean_([d["palette=xmodal"]["CLIP"] - t
                                 for d, t in zip(per_seed, trd)]),
            "ceiling_minus_TRD": mean_(ceil_) - mean_(trd),
            "real_half_n": [d["real_half"].get("n") for d in per_seed],
        },
    })

    # ---- (OP2) 动态范围 / (OP3) 零分布够小 / (OP4) 天花板方向 ----
    out["checks"]["OP2"] = dyn > NULL
    out["checks"]["OP3"] = NULL < GAP_REF
    out["checks"]["OP4"] = mean_(ceil_) > mean_(trd)
    for k, v in (("OP2", "VOID_CLIP_FLAT"), ("OP3", "VOID_CLIP_NOISY"), ("OP4", "VOID_CEILING_INVERTED")):
        if not out["checks"][k]:
            return dict(out, verdict=v, why=f"{k} 不过")

    # ---- 主判决 ----
    if dS_bar > NULL and (dS_bar - dP_bar) > NULL:
        v = "GAP32C_STRUCT"
    elif dP_bar > NULL and (dP_bar - dS_bar) > NULL:
        v = "GAP32C_PALETTE"
    elif dS_bar > NULL and dP_bar > NULL:
        v = "GAP32C_TIE"
    else:
        v = "GAP32C_BOTH_SMALL"
    out["verdict"] = v
    out["why"] = f"dS_bar={dS_bar:.4f} dP_bar={dP_bar:.4f} NULL={NULL:.4f}"
    return out


def load_dir(d):
    per = []
    for s in SEEDS:
        p = Path(d) / f"m58_clipdecomp_s{s}.json"
        if not p.exists():
            return None, f"缺 {p}"
        per.append(json.loads(p.read_text(encoding="utf-8")))
    return per, None


def selftest():
    def mk(trd, st, pa, re_, xm, ceil_, n=N_ROW, cn=37):
        return {"TRD": {"CLIP": trd, "n": n}, "struct=real": {"CLIP": st, "n": n},
                "palette=real": {"CLIP": pa, "n": n}, "palette=retrieved": {"CLIP": re_, "n": n},
                "palette=xmodal": {"CLIP": xm, "n": n}, "real_half": {"CLIP": ceil_, "n": cn}}
    bad = []

    def chk(name, got, want):
        if got != want:
            bad.append(f"{name}: 得 {got}，应 {want}")

    # 结构侧赢：dS=+1.0，dP=+0.1，跨种子抖动 0.02
    five = [mk(34.0 + 0.01 * i, 35.0 + 0.01 * i, 34.1 + 0.01 * i, 34.05, 34.02, 35.9)
            for i in range(5)]
    chk("STRUCT", analyse(five)["verdict"], "GAP32C_STRUCT")
    # 配色侧赢
    five = [mk(34.0 + 0.01 * i, 34.1 + 0.01 * i, 35.0 + 0.01 * i, 34.9, 34.2, 35.9)
            for i in range(5)]
    chk("PALETTE", analyse(five)["verdict"], "GAP32C_PALETTE")
    # 打平（两边都大、差值小）
    five = [mk(34.0 + 0.01 * i, 35.0 + 0.01 * i, 35.0 + 0.01 * i, 34.9, 34.2, 35.9)
            for i in range(5)]
    chk("TIE", analyse(five)["verdict"], "GAP32C_TIE")
    # 两边都小，但动态范围来自别的行（retrieved 很低）=> BOTH_SMALL
    five = [mk(34.0 + 0.01 * i, 34.01 + 0.01 * i, 34.02 + 0.01 * i, 33.0, 34.0, 35.9)
            for i in range(5)]
    chk("BOTH_SMALL", analyse(five)["verdict"], "GAP32C_BOTH_SMALL")
    # 五行几乎重合 => 动态范围不过
    five = [mk(34.0 + 0.3 * i, 34.0 + 0.3 * i, 34.0 + 0.3 * i, 34.0 + 0.3 * i, 34.0 + 0.3 * i, 35.9)
            for i in range(5)]
    chk("FLAT", analyse(five)["verdict"], "VOID_CLIP_FLAT")
    # 跨种子抖动 >= GAP_REF => 噪声作废
    five = [mk(34.0 + 0.8 * i, 36.0 + 0.8 * i, 34.2 + 0.8 * i, 34.1, 34.0, 45.9)
            for i in range(5)]
    chk("NOISY", analyse(five)["verdict"], "VOID_CLIP_NOISY")
    # 天花板反了
    five = [mk(36.0 + 0.01 * i, 37.0 + 0.01 * i, 36.1 + 0.01 * i, 36.05, 36.02, 35.0)
            for i in range(5)]
    chk("CEIL", analyse(five)["verdict"], "VOID_CEILING_INVERTED")
    # 料不齐三种
    chk("NO_DATA_seeds", analyse(five[:4])["verdict"], "VOID_NO_DATA")
    five = [mk(34.0, 35.0, 34.1, 34.05, 34.02, 35.9) for _ in range(5)]
    del five[2]["palette=xmodal"]
    chk("NO_DATA_row", analyse(five)["verdict"], "VOID_NO_DATA")
    five = [mk(34.0, 35.0, 34.1, 34.05, 34.02, 35.9) for _ in range(5)]
    five[1]["palette=real"]["n"] = 74
    chk("NO_DATA_n", analyse(five)["verdict"], "VOID_NO_DATA")
    five = [mk(34.0, 35.0, 34.1, 34.05, 34.02, 35.9, n=140) for _ in range(5)]
    chk("NO_DATA_nrow", analyse(five)["verdict"], "VOID_NO_DATA")
    # 数值：NULL 与 dS_bar 算对没有
    five = [mk(34.0 + 0.01 * i, 35.0 + 0.01 * i, 34.1 + 0.01 * i, 34.05, 34.02, 35.9)
            for i in range(5)]
    r = analyse(five)
    chk("dS_bar", round(r["dS_bar"], 6), 1.0)
    chk("dP_bar", round(r["dP_bar"], 6), 0.1)
    chk("NULL", round(r["NULL"], 6), round(0.04, 6))
    print(f"selftest: {13 - len(bad)}/13 通过")
    for b in bad:
        print("  失败 -", b)
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp", help="五份 m58_clipdecomp_s<seed>.json 所在目录")
    ap.add_argument("--out", type=Path, default=Path("/tmp/m58_clipdecomp.json"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    per, err = load_dir(a.dir)
    res = ({"n_seeds": 0, "checks": {"OP1": False}, "verdict": "VOID_NO_DATA", "why": err}
           if per is None else analyse(per))
    res["src_dir"] = str(a.dir)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")   # 落盘在打印之前
    print(f"判决: {res['verdict']}   ({res.get('why', '')})")
    print(f"操作检验: {res['checks']}")
    if "NULL" in res:
        print(f"  TRD 各种子 CLIP: " + " ".join(f"{x:.4f}" for x in res["TRD_per_seed"]))
        print(f"  dS 各种子: " + " ".join(f"{x:+.4f}" for x in res["dS_per_seed"]))
        print(f"  dP 各种子: " + " ".join(f"{x:+.4f}" for x in res["dP_per_seed"]))
        print(f"  dS_bar={res['dS_bar']:+.4f}  dP_bar={res['dP_bar']:+.4f}  "
              f"差={res['diff_bar']:+.4f}  NULL={res['NULL']:.4f}  dyn_range={res['dyn_range']:.4f}")
        print(f"  天花板(real_half)={res['ceiling_real_half']:.4f}  TRD_bar={res['TRD_bar']:.4f}  "
              f"差={res['_reg']['ceiling_minus_TRD']:+.4f}")
        print(f"  只登记不下判: {res['_reg']}")
    print("->", a.out)


if __name__ == "__main__":
    main()
