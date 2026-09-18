r"""(M62) 事后诊断：那张「32px MDE 表」到底准不准。⛔ 不下任何判决、不改任何判据。

背景：(M60) 起，功效/MDE 一律用这条公式（`m60_read_scale.py:225`）：

    sd(D_m) 的模型值 = sigma1 * sqrt(2 / K)     ← 假设 D_m 的全部离散都是**单次生成噪声**
    MDE            = 2.801586 * 模型值 / sqrt(n_mat)

本脚本只做两件事（全部零 GPU、只读已入库的 JSON）：

1. **对账**：把模型值与**实测** sd(D_m) 摆在一起；置换检验真正用的零分布 sd = rms(D)/sqrt(n)
   ⇒ 由它反推「这台仪器实际达到的 MDE」与读数的 95% CI。
2. **定性**：模型值偏小的那部分是**可复现的逐材质结构**，还是只是噪声不均匀？
   做法 = 把两臂的种子各按奇偶劈成两半，得到两份**独立**的 D 估计，算它们跨材质的相关 r；
   阴性对照 = 在**控制臂内部**做同样的劈法（真值 D≡0，不该有可复现结构）。

⚠ 本脚本写于 (M62) 判决**落地之后**，⛔ 因此不可能影响该判决；它只修正「MDE」这个
**只登记项**的算法，⛔ 判决名 `DOSE2_NULL_32` 与 p 值一个字不动（置换 p 本身不依赖这条公式）。

用法：
    python analysis/arch/m62_mde_audit.py --out experiments/m62_mde_audit.json
    python analysis/arch/m62_mde_audit.py --selftest
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m60_read_scale import (  # noqa: E402  冻结判读器本体，⛔ 不重写
    Z80_TWOSIDED, arm_mean, load_arm, mean_sd, perm_p, sigma1_of,
)

ROUNDS = [
    # (轮次, 目录, 控制 tag, 处理 tag, K)
    ("M60", "remote_tmp/m60_read", "m60_ctrl", "m60_more", 17),
    ("M62", "remote_tmp/m62", "m62_ctrl", "m62_dbl", 28),
]


def pearson(xs, ys):
    n = len(xs)
    mx, sx = mean_sd(xs)
    my, sy = mean_sd(ys)
    if sx == 0 or sy == 0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (n - 1)
    return cov / (sx * sy)


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def perm_r(xs, ys, n_perm=20000, seed=777):
    """相关系数的置换检验（双侧，打乱其中一列）。无 scipy ⇒ 不用 t 分布。"""
    obs = pearson(xs, ys)
    rng = random.Random(seed)
    sh = list(ys)
    hit = 0
    for _ in range(n_perm):
        rng.shuffle(sh)
        if abs(pearson(xs, sh)) >= abs(obs) - 1e-15:
            hit += 1
    return (hit + 1.0) / (n_perm + 1.0)


def diff_over(ctrl, trt, mats, cseeds, tseeds):
    a = arm_mean(trt, mats, tseeds)
    b = arm_mean(ctrl, mats, cseeds)
    return [a[m] - b[m] for m in mats]


def audit_round(name, d, ctag, ttag, k):
    ctrl, _, cprob = load_arm(d, ctag, k)
    trt, _, tprob = load_arm(d, ttag, k)
    if cprob or tprob or len(ctrl) != k or len(trt) != k:
        return {"round": name, "ok": False, "problems": [cprob, tprob]}
    mats = sorted(set(ctrl[sorted(ctrl)[0]]))
    n = len(mats)
    dm = diff_over(ctrl, trt, mats, sorted(ctrl), sorted(trt))
    obs, sd_emp = mean_sd(dm)

    s1 = sigma1_of(ctrl)
    sd_model = s1 * math.sqrt(2.0 / k)
    mde_model = Z80_TWOSIDED * sd_model / math.sqrt(n)
    se_true = rms(dm) / math.sqrt(n)           # 符号翻转置换检验的零分布 sd（精确）
    mde_true = Z80_TWOSIDED * se_true

    # 方差分解：sd_emp^2 = sd_model^2 + c^2 ⇒ c = 不随 K 变小的那部分
    c2 = sd_emp ** 2 - sd_model ** 2
    c = math.sqrt(c2) if c2 > 0 else 0.0

    # 劈半复现性：两份独立 D 估计跨材质的相关
    cs, ts = sorted(ctrl), sorted(trt)
    ev_c, od_c = cs[0::2], cs[1::2]
    ev_t, od_t = ts[0::2], ts[1::2]
    dA = diff_over(ctrl, trt, mats, ev_c, ev_t)
    dB = diff_over(ctrl, trt, mats, od_c, od_t)
    r_arms = pearson(dA, dB)

    # 阴性对照：控制臂内部四等分，真值 D≡0
    q = [cs[i::4] for i in range(4)]
    nA = diff_over(ctrl, ctrl, mats, q[0], q[1])
    nB = diff_over(ctrl, ctrl, mats, q[2], q[3])
    r_null = pearson(nA, nB)

    p, _ = perm_p(dm)
    return {
        "round": name, "ok": True, "k": k, "n_materials": n,
        "mean_D": obs, "p_perm": p,
        "sd_D_empirical": sd_emp,
        "sd_D_model_from_sigma1": sd_model,
        "sd_ratio_empirical_over_model": sd_emp / sd_model,
        "sigma1_ctrl": s1,
        "mde_registered_model": mde_model,
        "mde_actual": mde_true,
        "mde_inflation": mde_true / mde_model,
        "ci95": [obs - 1.96 * se_true, obs + 1.96 * se_true],
        "irreducible_per_material_sd_c": c,
        "mde_floor_at_infinite_K": Z80_TWOSIDED * c / math.sqrt(n),
        "splithalf_r_arms": r_arms,
        "splithalf_r_arms_p": perm_r(dA, dB),
        "splithalf_r_ctrl_only_negative_control": r_null,
        "splithalf_r_ctrl_only_p": perm_r(nA, nB),
    }


def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    chk(abs(rms([3.0, 4.0]) - math.sqrt(12.5)) < 1e-12, "rms")
    chk(abs(rms([2.0, 2.0]) - 2.0) < 1e-12, "rms const")
    chk(abs(pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) - 1.0) < 1e-12, "pearson +1")
    chk(abs(pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) + 1.0) < 1e-12, "pearson -1")
    chk(abs(pearson([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])) < 1e-12, "pearson const")
    # 相关为 0 的两列
    chk(abs(pearson([1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0])) < 1e-12, "pearson 0")
    # 方差分解：若 sd_emp 恰等于模型值，c 必须是 0
    chk(math.sqrt(max(0.0, 0.5 ** 2 - 0.5 ** 2)) == 0.0, "c=0")
    # MDE 与置换零分布口径一致性：D 全同号常数时 rms=|x|
    chk(abs(rms([0.2] * 10) - 0.2) < 1e-12, "rms of constant")
    chk(perm_r([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) < 0.2, "perm_r perfect corr is small")
    chk(perm_r([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0]) > 0.9, "perm_r const is 1")
    print("selftest OK %d/%d" % (ok, 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default=None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    res = {"note": "diagnostic only; no verdict, no criterion changed", "rounds": []}
    for name, d, ctag, ttag, k in ROUNDS:
        res["rounds"].append(audit_round(name, os.path.join(a.root, d), ctag, ttag, k))
    for r in res["rounds"]:
        if not r.get("ok"):
            print("%s: NOT OK %s" % (r["round"], r["problems"]))
            continue
        print("== %s  K=%d  n=%d" % (r["round"], r["k"], r["n_materials"]))
        print("   mean_D=%+.4f  p=%.4f  95%%CI=[%+.4f, %+.4f]"
              % (r["mean_D"], r["p_perm"], r["ci95"][0], r["ci95"][1]))
        print("   sd(D): empirical %.4f vs model %.4f  ratio %.2f"
              % (r["sd_D_empirical"], r["sd_D_model_from_sigma1"],
                 r["sd_ratio_empirical_over_model"]))
        print("   MDE: registered %.4f vs actual %.4f  (x%.2f)  floor@K=inf %.4f"
              % (r["mde_registered_model"], r["mde_actual"], r["mde_inflation"],
                 r["mde_floor_at_infinite_K"]))
        print("   split-half r: arms %+.3f (p=%.4f)   ctrl-only(negative control) %+.3f (p=%.4f)"
              % (r["splithalf_r_arms"], r["splithalf_r_arms_p"],
                 r["splithalf_r_ctrl_only_negative_control"], r["splithalf_r_ctrl_only_p"]))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
