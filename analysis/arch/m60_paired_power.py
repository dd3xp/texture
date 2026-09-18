"""(M60) 选题用的功效计算：**配对 CLIP@32px 能测得动多小的架构改动？**（零 GPU、纯本机）

## 为什么要算这个（写在看任何读数之前）

(M58) 把 32px 的 CLIP 拆解仪器造出来了（四条操作检验全过），(M59) 又给它加上了逐图列
（`--per_image`）。但两轮都是在**拆解**尺度上用它（把真人部件换进去）。本脚本问的是另一件事：

> 如果下一轮做一个**架构改动**（例如候选 G ＝ 训练规模），用这把尺子做
> 「同一批目标、同一批种子、只换检查点」的**逐材质配对**比较，**最小可检测效应（MDE）是多少**？

这件事必须先算，理由是 (M44) 写死的纪律：**跑前必须先问「它若有用、效应可能够大吗」**，
否则就是再造一次"测不动"的轮次。而且本轮有一个**新的、要紧的算式**要核对：

- (M33) 给 CLIP 的 m=1 **不成对**门槛是 **0.41**；
- (M58) 量到这把尺子在 32px 上的**全部天花板**（`real_half` − `TRD`）只有 **0.5127**。
- ⇒ 用不成对门槛，一个架构改动必须一口气吃掉**天花板的 80%** 才算过门 ⇒ 实际上判不动任何东西。
  （旁证：(M55) 的 `--refs` 整条新条件通路只动了 **0.0109**。）

所以"配对能不能把门降下来"这个问题，决定了 32px 这条线是**还有仪器**还是**仪器侧已尽**。

## 怎么算（口径先定死）

- **统计单位 ＝ 材质（提示词），n = 67** —— 沿用 (M59) 预注册第三节定死的聚类单位
  （74 个 V_mat@32 目标里 7 个提示词各出现两次、且 reps=2 ⇒ 148 张图不独立）。
  ⛔ 不许改用 148 张当分母。
- **经验零分布**：只用 `TRD` 这一行，取五个种子里**全部 10 个种子对** (s_i, s_j)，
  逐材质差 `d_m = x[m,s_i] − x[m,s_j]`。真值恒 0（同一个检查点、同一批目标）⇒
  它量到的正是"两次独立生成"的噪声，也正是「只换检查点」那种臂要面对的噪声。
  ⚑ 这条沿用 (M50) 的纪律：**门槛口径不对时就自己造经验零分布**。
- **σ1** ＝ 逐材质、逐种子的单次噪声 sd（由 10 个对的差 sd 反解：sd(1v1) = σ1·√2）。
- **K 个种子对 K 个种子**的配对差 sd ＝ σ1·√(2/K)。
- **MDE（配对幅度检验，双侧 α=0.05、功效 80%）** ＝ 2.802 × sd_diff / √n。
- **MDE（逐材质符号检验）**：先求 n=67 双侧 0.05 的临界计数，再求 80% 功效所需的真实
  P(d>0)=q，再按常效应 + 正态噪声反解 δ = Φ⁻¹(q) · sd_diff。
  ⚠ 符号检验只用差的正负、丢掉幅度 ⇒ 预期比幅度检验钝，这里两个都报、由读者对比。

参照点（只作比例的分母，⛔ 不当判据）：天花板 0.5127（(M58) 第三节，n=37 口径不同、只作量级）、
(M33) 不成对门槛 0.41、(M59) 的 `dS_bar` 0.5410 / `dP_bar` 0.4891。

用法：
    python analysis/arch/m60_paired_power.py --dir experiments --tag m59_pairedclip \
           --out experiments/m60_paired_power.json
    python analysis/arch/m60_paired_power.py --selftest
"""
import argparse
import json
import math
import os

N_MAT_EXPECT = 67
N_IMG_EXPECT = 148
SEEDS = [0, 1, 2, 3, 4]
ROW = "TRD"

# 参照点（只作分母/量级，⛔ 不是判据）
CEILING_M58 = 0.5127      # (M58) real_half - TRD_bar
THR_M33_UNPAIRED = 0.41   # (M33) CLIP m=1 不成对门槛
DS_BAR_M59 = 0.5410
DP_BAR_M59 = 0.4891

Z80_TWOSIDED = 2.801586   # z(0.975) + z(0.80) = 1.959964 + 0.841621


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p):
    """有理逼近（Acklam），精度足够本用途；无 scipy（jzs_train 没有）。"""
    if not (0.0 < p < 1.0):
        raise ValueError("p out of range")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def sign_test_p(pos, neg):
    """精确二项、双侧、平局已丢弃（math.comb，无 scipy）。"""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def sign_test_crit(n, alpha=0.05):
    """双侧 alpha 下，|pos| 至少要到多少才显著。返回最小的 pos(>n/2)。"""
    for pos in range((n // 2) + 1, n + 1):
        if sign_test_p(pos, n - pos) < alpha:
            return pos
    return None


def mean_sd(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def per_material(per_img, mats):
    """逐材质均值（该材质在这一份 JSON 里的全部图）。"""
    acc = {}
    for v, m in zip(per_img, mats):
        acc.setdefault(m, []).append(v)
    return {m: sum(v) / len(v) for m, v in acc.items()}


def load_rows(d, tag):
    """返回 {seed: {material: clip}}；缺任何一份直接报缺。"""
    out, missing, bad = {}, [], []
    for s in SEEDS:
        p = os.path.join(d, "%s_s%d.json" % (tag, s))
        if not os.path.exists(p):
            missing.append(s)
            continue
        with open(p, encoding="utf-8") as f:     # Windows 默认 GBK 会崩
            j = json.load(f)
        row = j.get(ROW, {})
        per = row.get("_CLIP_per")
        mats = j.get("_mats")
        if not isinstance(per, list) or not isinstance(mats, list):
            bad.append((s, "no _CLIP_per/_mats"))
            continue
        if len(per) != N_IMG_EXPECT or len(mats) != N_IMG_EXPECT:
            bad.append((s, "len %d/%d" % (len(per), len(mats))))
            continue
        out[s] = per_material(per, mats)
    return out, missing, bad


def analyse(by_seed):
    """核心计算。返回 dict；调用方负责落盘。"""
    seeds = sorted(by_seed)
    mats = sorted(set.intersection(*[set(by_seed[s]) for s in seeds])) if seeds else []
    res = {
        "n_seeds": len(seeds),
        "seeds": seeds,
        "n_materials": len(mats),
        "n_materials_expect": N_MAT_EXPECT,
    }
    if len(seeds) < 2 or len(mats) < 2:
        res["ok"] = False
        res["why"] = "need >=2 seeds and >=2 materials"
        return res
    res["ok"] = True

    # --- 经验零分布：全部种子对 ---
    pairs = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            si, sj = seeds[i], seeds[j]
            d = [by_seed[si][m] - by_seed[sj][m] for m in mats]
            mu, sd = mean_sd(d)
            pos = sum(1 for x in d if x > 0)
            neg = sum(1 for x in d if x < 0)
            pairs.append({
                "seeds": [si, sj], "mean": mu, "sd": sd,
                "pos": pos, "neg": neg, "p": sign_test_p(pos, neg),
            })
    res["null_pairs"] = pairs
    sd_1v1 = sum(p["sd"] for p in pairs) / len(pairs)
    res["sd_diff_1v1"] = sd_1v1
    res["sigma1"] = sd_1v1 / math.sqrt(2.0)          # 单次生成的逐材质噪声
    res["null_mean_absmax"] = max(abs(p["mean"]) for p in pairs)
    res["null_p_min"] = min(p["p"] for p in pairs)
    res["null_falsepos_at_05"] = sum(1 for p in pairs if p["p"] < 0.05)

    # --- MDE ---
    n = len(mats)
    crit = sign_test_crit(n)
    res["sign_crit_pos"] = crit
    # 符号检验 80% 功效所需的真实 q = P(d>0)：用正态近似解 pos>=crit 的功效
    q_need = None
    if crit is not None:
        lo, hi = 0.5, 0.999
        for _ in range(200):
            mid = (lo + hi) / 2
            # P(Binom(n,mid) >= crit) 的正态近似（含连续性校正）
            mu, sg = n * mid, math.sqrt(n * mid * (1 - mid))
            power = 1 - _norm_cdf((crit - 0.5 - mu) / sg)
            if power < 0.80:
                lo = mid
            else:
                hi = mid
        q_need = (lo + hi) / 2
    res["sign_q_need_80pct"] = q_need

    res["mde"] = {}
    for K in (1, 2, 3, 5):
        sd_diff = res["sigma1"] * math.sqrt(2.0 / K)
        mag = Z80_TWOSIDED * sd_diff / math.sqrt(n)
        entry = {
            "sd_diff": sd_diff,
            "mde_magnitude": mag,
            "mde_magnitude_frac_of_ceiling": mag / CEILING_M58,
        }
        if q_need is not None:
            delta = _norm_ppf(q_need) * sd_diff
            entry["mde_signtest"] = delta
            entry["mde_signtest_frac_of_ceiling"] = delta / CEILING_M58
        res["mde"][str(K)] = entry

    # --- 参照点（只作分母/量级） ---
    best = res["mde"]["5"]
    res["reference"] = {
        "ceiling_m58": CEILING_M58,
        "thr_m33_unpaired": THR_M33_UNPAIRED,
        "thr_m33_frac_of_ceiling": THR_M33_UNPAIRED / CEILING_M58,
        "dS_bar_m59": DS_BAR_M59,
        "dP_bar_m59": DP_BAR_M59,
        "gain_vs_m33_magnitude_K5": THR_M33_UNPAIRED / best["mde_magnitude"],
    }
    return res


def selftest():
    """构造数据核算式，⛔ 不碰真实读数。"""
    ok = 0

    # 1) 符号检验 p：对称、已知值
    assert abs(sign_test_p(67, 0) - 2.0 / 2 ** 67) < 1e-30; ok += 1
    assert abs(sign_test_p(34, 33) - 1.0) < 1e-9; ok += 1
    assert sign_test_p(39, 28) > 0.2 and sign_test_p(39, 28) < 0.25; ok += 1   # (M59) 实得 0.2215
    assert sign_test_p(48, 19) < 1e-3; ok += 1                                  # (M59) 实得 5.2e-4
    # 2) 临界计数
    c = sign_test_crit(67)
    assert c is not None and sign_test_p(c, 67 - c) < 0.05; ok += 1
    assert sign_test_p(c - 1, 67 - (c - 1)) >= 0.05; ok += 1
    # 3) 正态分位数/CDF 互逆
    for p in (0.01, 0.2, 0.5, 0.8, 0.975):
        assert abs(_norm_cdf(_norm_ppf(p)) - p) < 1e-6; ok += 1
    assert abs(_norm_ppf(0.975) - 1.959964) < 1e-4; ok += 1
    # 4) mean_sd
    mu, sd = mean_sd([1.0, 2.0, 3.0])
    assert abs(mu - 2.0) < 1e-12 and abs(sd - 1.0) < 1e-12; ok += 1
    # 5) per_material：两张图同材质取均值
    pm = per_material([1.0, 3.0, 5.0], ["a", "a", "b"])
    assert abs(pm["a"] - 2.0) < 1e-12 and abs(pm["b"] - 5.0) < 1e-12; ok += 1
    # 6) analyse：构造纯噪声 + 已知 sigma，检查 sigma1 反解与 MDE 单调性
    import random
    rng = random.Random(0)
    sig = 0.5
    mats = ["m%03d" % i for i in range(N_MAT_EXPECT)]
    base = {m: 30.0 + rng.random() for m in mats}
    by = {s: {m: base[m] + rng.gauss(0, sig) for m in mats} for s in SEEDS}
    r = analyse(by)
    assert r["ok"] and r["n_materials"] == N_MAT_EXPECT; ok += 1
    assert abs(r["sigma1"] - sig) < 0.12, r["sigma1"]; ok += 1
    assert r["mde"]["5"]["mde_magnitude"] < r["mde"]["1"]["mde_magnitude"]; ok += 1
    # 符号检验必须比幅度检验钝
    assert r["mde"]["5"]["mde_signtest"] > r["mde"]["5"]["mde_magnitude"]; ok += 1
    # 7) 缺数据分支
    r2 = analyse({0: {m: 1.0 for m in mats}})
    assert r2["ok"] is False; ok += 1
    print("selftest %d/%d OK" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments")
    ap.add_argument("--tag", default="m59_pairedclip")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    by_seed, missing, bad = load_rows(a.dir, a.tag)
    res = analyse(by_seed)
    res["missing_seeds"] = missing
    res["bad"] = bad
    if missing or bad:
        res["ok"] = False
        res["why"] = "VOID_NO_DATA: missing=%s bad=%s" % (missing, bad)

    # 落盘一律在打印之前（非 GBK 字形会崩在写 JSON 之前＝白跑）
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M60) paired CLIP@32px power ==")
    print("seeds=%s  materials=%d (expect %d)  ok=%s"
          % (res.get("seeds"), res.get("n_materials", -1), N_MAT_EXPECT, res.get("ok")))
    if not res.get("ok"):
        print("why:", res.get("why"))
        return
    print("empirical null (TRD row, %d seed pairs): sd(1v1)=%.4f  sigma1=%.4f"
          % (len(res["null_pairs"]), res["sd_diff_1v1"], res["sigma1"]))
    print("  |mean| max=%.4f   min p=%.4f   pairs with p<0.05: %d/%d"
          % (res["null_mean_absmax"], res["null_p_min"],
             res["null_falsepos_at_05"], len(res["null_pairs"])))
    print("sign test: crit pos>=%s of %d ; q needed for 80%% power = %.4f"
          % (res["sign_crit_pos"], res["n_materials"], res["sign_q_need_80pct"]))
    print("%-4s %-10s %-12s %-10s %-12s %-10s"
          % ("K", "sd_diff", "MDE(magn)", "/ceiling", "MDE(sign)", "/ceiling"))
    for K in ("1", "2", "3", "5"):
        e = res["mde"][K]
        print("%-4s %-10.4f %-12.4f %-10.2f %-12.4f %-10.2f"
              % (K, e["sd_diff"], e["mde_magnitude"],
                 e["mde_magnitude_frac_of_ceiling"],
                 e["mde_signtest"], e["mde_signtest_frac_of_ceiling"]))
    r = res["reference"]
    print("reference: ceiling=%.4f  (M33) unpaired thr=%.2f = %.0f%% of ceiling"
          % (r["ceiling_m58"], r["thr_m33_unpaired"], 100 * r["thr_m33_frac_of_ceiling"]))
    print("           paired magnitude (K=5) is %.1fx more sensitive than (M33) thr"
          % r["gain_vs_m33_magnitude_K5"])


if __name__ == "__main__":
    main()
