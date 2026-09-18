r"""(M61) 试点功效器：用**控制臂试点**（`runs/trd_v10`，种子 100..104）把 16px 上的
σ1、`ceiling16`、等效边界 δ 和每臂种子数 K 定出来。

⛔ 本文件随预注册提交，**跑完一个字不许改**。
⚠⚠ **试点那几份只用来定 K 与 δ，⛔ 一个数都不许进判据** —— 判据臂的种子是 0..K-1，
与试点的 100..104 **不相交**（(M60) 结账那节写死的纪律）。

## 为什么非得先跑试点

(M60) 的功效算式能直接复用 (M59) 的五份 32px 逐图 JSON；16px 上没有这种料：
(M46) 那批 16px JSON 是 **v8**、且在 `--per_image` 出生之前（没有 `_CLIP_per`/`_mats`）
⇒ **σ1@16px 从没量过**。⇒ 必须先出几份控制臂读数。

## 算式（全部写死在试点料落地之前）

    n_mat  = 125                                   # 零 GPU 事先量定，见判读器
    sigma1 = 由全部种子对的逐材质差反解（与 (M60) 逐字同一算法）
    ceiling16 = mean_seed[ real_half.CLIP - TRD.CLIP ]   # real_half 与模型/种子无关
    delta  = (0.1250 / 0.5127) * ceiling16 = 0.243807 * ceiling16
    K_req  = 2 * (z_.95 + z_.80)^2 * sigma1^2 / (n_mat * delta^2)      # 单侧非劣、80% 功效
    K      = min(K_MAX, max(K_MIN, 向上取到偶数(K_req)))                # 偶数 ⇒ (OP4) 奇偶对半平衡

`K_MIN = 6`、`K_MAX = 40`。⚠⚠ **若 K_req > K_MAX**：本轮**照 K_MAX 跑**，并在账本里
预先登记 `POWER_SHORT` —— 即"判决很可能是 `INCONCLUSIVE_16`，而那是预料之中的"。
⛔ 无论如何**不许事后放宽 δ**（那等于看见读数再改判据）。

## 顺带查一件事：这把尺子在 16px 上的零分布干不干净

试点的每个种子对真值恒 0（同一检查点、同一批目标，只换生成种子）⇒ 对每一对做与判据
逐字相同的双侧置换检验，α=0.05 上的假阳个数应当很小。⚠ 只作**记账**，⛔ 不是门
（(OP4) 才是判据侧的假阳检验，而且它在判读器里）。

用法：
    python analysis/arch/m61_power16.py --dir /tmp --tag m61_pilot --seeds 100 101 102 103 104 \
           --out /tmp/m61_power16.json
    python analysis/arch/m61_power16.py --selftest
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m61_read_noninf16 import (            # noqa: E402  (⚠ scp 上远程时两个文件要一起传)
    DELTA_FRAC, N_MAT_EXPECT, Z80_ONESIDED, Z80_TWOSIDED,
    k_required, load_arm, mean_sd, perm_p, sigma1_of,
)

K_MIN = 6
K_MAX = 40


def choose_k(k_req, k_min=K_MIN, k_max=K_MAX):
    """向上取到偶数，再夹在 [k_min, k_max] 里。返回 (K, power_short)。"""
    if k_req is None:
        return None, None
    k = int(math.ceil(k_req))
    if k % 2:
        k += 1
    short = k > k_max
    return max(k_min, min(k_max, k)), short


def mde(sigma1, k, n_mat=N_MAT_EXPECT, z=Z80_ONESIDED):
    if not sigma1 or not k:
        return None
    return z * sigma1 * math.sqrt(2.0 / k) / math.sqrt(n_mat)


def analyse(by, rowmean, floors, n_perm=4000):
    """by: {seed: {mat: clip}}；rowmean: {seed: TRD 行均值}；floors: {seed: (real_half CLIP, n)}。"""
    res = {"n_seeds": len(by), "seeds": sorted(by)}
    if len(by) < 2:
        res["verdict"] = "PILOT_TOO_THIN"
        return res
    mats = sorted(set.intersection(*[set(v) for v in by.values()]))
    res["n_materials"] = len(mats)
    res["n_materials_expected"] = N_MAT_EXPECT
    res["n_materials_ok"] = len(mats) == N_MAT_EXPECT

    s1 = sigma1_of(by)
    res["sigma1"] = s1

    fl = sorted({round(v[0], 12) for v in floors.values() if v and v[0] is not None})
    res["real_half_clip"] = fl[0] if len(fl) == 1 else fl
    res["real_half_identical"] = len(fl) == 1
    res["trd_row_means"] = rowmean
    if len(fl) == 1 and rowmean:
        ceil16 = sum(fl[0] - v for v in rowmean.values()) / len(rowmean)
    else:
        ceil16 = None
    res["ceiling16"] = ceil16
    res["ceiling16_per_seed"] = ({s: fl[0] - v for s, v in rowmean.items()}
                                 if len(fl) == 1 else None)

    delta = DELTA_FRAC * ceil16 if ceil16 else None
    res["delta_frac"] = DELTA_FRAC
    res["delta"] = delta
    kreq = k_required(s1, delta) if delta else None
    k, short = choose_k(kreq)
    res["k_required"] = kreq
    res["k_chosen"] = k
    res["power_short"] = short
    res["mde_noninf_at_k"] = mde(s1, k)
    res["mde_twosided_at_k"] = mde(s1, k, z=Z80_TWOSIDED)
    res["mde_frac_of_delta"] = (mde(s1, k) / delta) if (delta and k) else None

    # 零分布干净度（只登记）
    seeds = sorted(by)
    pairs, fp, ps, ms = 0, 0, [], []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            d = [by[seeds[i]][m] - by[seeds[j]][m] for m in mats]
            p, o = perm_p(d, n_perm=n_perm)
            pairs += 1
            fp += (p < 0.05)
            ps.append(p)
            ms.append(abs(o))
    res["null"] = {"pairs": pairs, "false_pos_at_05": fp,
                   "min_p": min(ps) if ps else None,
                   "max_abs_mean": max(ms) if ms else None}
    res["verdict"] = "PILOT_OK" if (res["n_materials_ok"] and s1 and delta) else "PILOT_BAD"
    return res


def selftest():
    import random
    ok = 0
    mats = ["m%03d" % i for i in range(N_MAT_EXPECT)]
    rng = random.Random(3)
    base = {m: 33.0 + rng.random() for m in mats}

    def mk(k, sd=0.9, seed=0):
        r = random.Random(seed)
        return {100 + s: {m: base[m] + r.gauss(0, sd) for m in mats} for s in range(k)}

    by = mk(5)
    rows = {s: sum(by[s].values()) / len(mats) for s in by}
    floors = {s: (34.0, 98) for s in by}
    r = analyse(by, rows, floors, n_perm=600)
    assert r["verdict"] == "PILOT_OK", r; ok += 1
    assert r["n_materials_ok"] and r["real_half_identical"]; ok += 1
    assert 0.7 < r["sigma1"] < 1.1, r["sigma1"]; ok += 1
    assert r["ceiling16"] is not None and r["delta"] is not None; ok += 1
    assert abs(r["delta"] - DELTA_FRAC * r["ceiling16"]) < 1e-12; ok += 1
    assert r["null"]["pairs"] == 10; ok += 1

    # K 的单调性与夹取
    assert choose_k(3.2) == (K_MIN, False); ok += 1          # 向上取偶=4 -> 夹到 6
    assert choose_k(19.1) == (20, False); ok += 1
    assert choose_k(20.0) == (20, False); ok += 1
    assert choose_k(21.0) == (22, False); ok += 1
    assert choose_k(100.0) == (K_MAX, True); ok += 1          # 超 cap -> POWER_SHORT
    assert choose_k(None) == (None, None); ok += 1
    assert all(choose_k(x)[0] % 2 == 0 for x in (7.0, 8.1, 15.5, 30.2)); ok += 1

    # MDE 随 K 下降、单侧比双侧小
    assert mde(0.9, 40) < mde(0.9, 20) < mde(0.9, 6); ok += 1
    assert mde(0.9, 20) < mde(0.9, 20, z=Z80_TWOSIDED); ok += 1
    assert mde(None, 20) is None and mde(0.9, None) is None; ok += 1
    # K_req 与 MDE 自洽：按 K_req 取的 K 应当让 MDE <= delta（未触 cap 时）
    for s1, dl in ((0.9, 0.05), (0.8, 0.08), (1.1, 0.06)):
        kq = k_required(s1, dl)
        k, short = choose_k(kq)
        if not short:
            assert mde(s1, k) <= dl + 1e-12, (s1, dl, kq, k, mde(s1, k)); ok += 1

    # 料太薄 / 参照集不一致
    assert analyse({100: {"a": 1.0}}, {}, {})["verdict"] == "PILOT_TOO_THIN"; ok += 1
    r = analyse(by, rows, {100: (34.0, 98), 101: (34.5, 98)}, n_perm=200)
    assert not r["real_half_identical"] and r["ceiling16"] is None; ok += 1
    assert r["verdict"] == "PILOT_BAD"; ok += 1
    # 材质数不对 -> PILOT_BAD
    thin = {s: {m: v for m, v in d.items() if m != mats[0]} for s, d in by.items()}
    assert analyse(thin, rows, floors, n_perm=200)["verdict"] == "PILOT_BAD"; ok += 1
    # mean_sd 引用可用（保证 import 没断）
    assert abs(mean_sd([1.0, 2.0, 3.0])[1] - 1.0) < 1e-12; ok += 1
    print("selftest %d/%d OK" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp")
    ap.add_argument("--tag", default="m61_pilot")
    ap.add_argument("--seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    by, rows, floors, prob = load_arm(a.dir, a.tag, a.seeds)
    res = analyse(by, rows, floors)
    res["load_problems"] = prob

    if a.out:                                   # 落盘一律在打印之前
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M61) 16px pilot power ==")
    print("  seeds=%s  n_mat=%s (expect %d, ok=%s)  problems=%s"
          % (res.get("seeds"), res.get("n_materials"), N_MAT_EXPECT,
             res.get("n_materials_ok"), prob))
    if res["verdict"] == "PILOT_TOO_THIN":
        print("VERDICT:", res["verdict"])
        return
    print("  real_half.CLIP=%s (identical=%s)  ceiling16=%s"
          % (res["real_half_clip"], res["real_half_identical"], res["ceiling16"]))
    print("  sigma1=%s  delta=%s  K_req=%s  K=%s  POWER_SHORT=%s"
          % (res["sigma1"], res["delta"], res["k_required"],
             res["k_chosen"], res["power_short"]))
    print("  MDE_1side@K=%s (%.0f%% of delta)" %
          (res["mde_noninf_at_k"], 100 * (res["mde_frac_of_delta"] or 0)))
    print("  null: %s" % res["null"])
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
