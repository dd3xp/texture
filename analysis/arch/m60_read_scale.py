r"""(M60) 盲写判读器：候选 G（训练规模 —— 纯加步数）在**配对 CLIP@32px** 上的判决。

⛔ 本文件随预注册提交，**跑完一个字不许改**。⛔ 写它的时候处理臂（新检查点）还不存在。

## 判据（唯一下判的一条）

统计单位 = **材质（提示词），n = 67**（沿用 (M59) 预注册第三节定死的聚类单位：74 个
V_mat@32 目标里 7 个提示词各出现两次、且 reps=2 ⇒ 148 张图不独立）。⛔ 不许改用 148。

    D_m = mean_{seed,rep}[ CLIP_处理(m) ] − mean_{seed,rep}[ CLIP_控制(m) ]

只读每份 JSON 的 `TRD` 行（`_CLIP_per` + `_mats`）——**控制臂 = `runs/trd_v10`，
处理臂 = 同配方再训 12000 步的检查点**，两臂同一批目标、同一批种子、同一条生成命令。

主检验 = **逐材质配对的符号翻转置换检验**（双侧，20000 次，rng 固定 12345；
`jzs_train` 无 scipy ⇒ 不用 t 分布、不做正态假设）。

| 判决 | 条件 |
|---|---|
| **`SCALE_HELPS_32`** | `p < 0.05` 且 `mean_D > 0` |
| **`SCALE_HURTS_32`** | `p < 0.05` 且 `mean_D < 0` |
| **`SCALE_NULL_32`** | `p >= 0.05` |

⚠⚠ **`SCALE_NULL_32` 的唯一许可读法**：「在这把尺子上，把训练量翻一倍带来的 32px CLIP
变化**小于本臂的 MDE**」。⛔ 不许读成"训练规模无用"、⛔ 不许读成"模型已训够"、
⛔ 不许读成"该转去加大模型"。本臂的 MDE 由 `--k` 决定，(M60) 功效计算
（`analysis/arch/m60_paired_power.py`，σ1 = 0.8480）给出 K=17 ⇒ **MDE ≈ 0.0996**
（＝(M58) 天花板 0.5127 的 19%）。判读器会用**控制臂实测的 σ1** 重算 MDE 并打印。

⚠ 本判决**只回答 32px CLIP 这一个问题**。⛔ 不许据此改 `eval/final_test.sh`
（那要另有"16px 不退步"的独立证据），⛔ 不许据此开判官臂（32px 准入条件② 未松）。

## 操作检验（任一不过即 VOID，按 OP1→OP2→OP3→OP4 顺序，⛔ 不补救不改判据）

| 检验 | 要求 | 不过则 |
|---|---|---|
| (OP1) 料齐 | 两臂各 `--k` 份 JSON；每份 `TRD._CLIP_per` 与 `_mats` 长度 = 148；不同材质数 = 67；两臂材质集合相同 | `VOID_NO_DATA` |
| (OP2) 新列自洽 | 每份 \|mean(`_CLIP_per`) − `CLIP`\| < 1e-4 | `VOID_PER_IMAGE_MISMATCH` |
| (OP3) 控制臂前 5 个种子复现 (M59) | 控制臂 seed 0..4 的 `TRD.CLIP` 与 `experiments/m59_pairedclip_s{S}.json` 逐个 \|差\| < 1e-3 | `VOID_CTRL_NOT_REPRODUCED` |
| (OP4) 假阳检验 | 控制臂**内部**对半分（种子按奇偶）的同一检验 p >= 0.01 | `VOID_PAIRED_FALSEPOS` |

⚑ (OP3) 把"控制臂就是 (M59) 那条 TRD 行"变成可证伪的（⚠ 并**披露**：控制臂前 5 个种子的
读数在本预注册之前就见过了 —— 它们就是 (M59) 的产物；处理臂在本文件提交时**不存在**）。
⚑ (OP4) 沿用 (M50) 的纪律：自己造经验零分布查假阳，⛔ 不当主判据的门（主判据的门在置换检验里）。

用法：
    python analysis/arch/m60_read_scale.py --dir /tmp --ctrl_tag m60_ctrl --trt_tag m60_more \
           --k 17 --m59dir experiments --out /tmp/m60_scale.json
    python analysis/arch/m60_read_scale.py --selftest
"""
import argparse
import json
import math
import os
import random

N_IMG_EXPECT = 148
N_MAT_EXPECT = 67
ROW = "TRD"
N_PERM = 20000
PERM_SEED = 12345
CEILING_M58 = 0.5127          # (M58) real_half - TRD_bar，只作分母/量级
SIGMA1_M60 = 0.8480           # (M60) 功效计算实测（控制臂 5 种子、10 个种子对）
Z80_TWOSIDED = 2.801586


def sign_test_p(pos, neg):
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def mean_sd(xs):
    n = len(xs)
    mu = sum(xs) / n
    if n < 2:
        return mu, 0.0
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return mu, math.sqrt(var)


def perm_p(d, n_perm=N_PERM, seed=PERM_SEED):
    """逐材质配对的符号翻转置换检验（双侧）。返回 (p, observed_mean)。"""
    obs = sum(d) / len(d)
    rng = random.Random(seed)
    hit = 0
    for _ in range(n_perm):
        s = 0.0
        for x in d:
            s += x if rng.random() < 0.5 else -x
        if abs(s / len(d)) >= abs(obs) - 1e-15:
            hit += 1
    return (hit + 1.0) / (n_perm + 1.0), obs


def per_material(per_img, mats):
    acc = {}
    for v, m in zip(per_img, mats):
        acc.setdefault(m, []).append(v)
    return {m: sum(v) / len(v) for m, v in acc.items()}


def load_arm(d, tag, k):
    """返回 ({seed: {mat: clip}}, {seed: row_mean}, problems)。"""
    by, rowmean, prob = {}, {}, []
    for s in range(k):
        p = os.path.join(d, "%s_s%d.json" % (tag, s))
        if not os.path.exists(p):
            prob.append((s, "missing"))
            continue
        with open(p, encoding="utf-8") as f:      # Windows 默认 GBK 会崩
            j = json.load(f)
        row = j.get(ROW, {})
        per, mats = row.get("_CLIP_per"), j.get("_mats")
        if not isinstance(per, list) or not isinstance(mats, list):
            prob.append((s, "no _CLIP_per/_mats"))
            continue
        if len(per) != N_IMG_EXPECT or len(mats) != N_IMG_EXPECT:
            prob.append((s, "len %d/%d" % (len(per), len(mats))))
            continue
        if abs(sum(per) / len(per) - row.get("CLIP", float("nan"))) >= 1e-4:
            prob.append((s, "per_image mismatch"))
            continue
        by[s] = per_material(per, mats)
        rowmean[s] = row.get("CLIP")
    return by, rowmean, prob


def sigma1_of(by):
    """由全部种子对的逐材质差反解单次生成噪声 sd。"""
    seeds = sorted(by)
    mats = sorted(set.intersection(*[set(by[s]) for s in seeds]))
    sds = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            d = [by[seeds[i]][m] - by[seeds[j]][m] for m in mats]
            sds.append(mean_sd(d)[1])
    if not sds:
        return None
    return (sum(sds) / len(sds)) / math.sqrt(2.0)


def arm_mean(by, mats, seeds=None):
    seeds = sorted(by) if seeds is None else sorted(seeds)
    return {m: sum(by[s][m] for s in seeds) / len(seeds) for m in mats}


def op3_offenders(ctrl_rows, m59rows, tol=1e-3):
    """(OP3)：控制臂 seed 0..4 的 TRD 行均值必须复现 (M59)。返回不合格清单。"""
    if not m59rows or len(m59rows) != 5:
        return None                      # None = 料不全，(OP3) 直接不过
    off = []
    for s, want in sorted(m59rows.items()):
        got = ctrl_rows.get(s)
        if got is None or want is None or abs(got - want) >= tol:
            off.append([s, want, got])
    return off


def analyse(ctrl, trt, k, op3_off=None, n_perm=N_PERM):
    res = {"k_requested": k, "n_perm": n_perm}
    ops = {}

    # ---- (OP1) 料齐 ----
    ok1 = (len(ctrl) == k and len(trt) == k and len(ctrl) > 1)
    mats_c = set.intersection(*[set(v) for v in ctrl.values()]) if ctrl else set()
    mats_t = set.intersection(*[set(v) for v in trt.values()]) if trt else set()
    ok1 = ok1 and len(mats_c) == N_MAT_EXPECT and mats_c == mats_t
    ops["OP1"] = {"ok": bool(ok1), "n_ctrl": len(ctrl), "n_trt": len(trt),
                  "n_mat_ctrl": len(mats_c), "n_mat_trt": len(mats_t),
                  "same_materials": mats_c == mats_t}
    if not ok1:
        res["ops"] = ops
        res["verdict"] = "VOID_NO_DATA"
        return res
    mats = sorted(mats_c)

    # ---- (OP2) 已在 load_arm 里逐份查过（不自洽的份不会进来）----
    ops["OP2"] = {"ok": True, "note": "checked per file at load time (tol 1e-4)"}

    # ---- (OP3) 控制臂前 5 个种子复现 (M59)（由调用方用行均值算好传进来）----
    ops["OP3"] = {"ok": op3_off is not None and len(op3_off) == 0,
                  "off": op3_off,
                  "note": "None = (M59) rows unavailable"}
    if not ops["OP3"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_CTRL_NOT_REPRODUCED"
        return res

    # ---- (OP4) 控制臂内部假阳检验（奇偶对半）----
    ev = [s for s in sorted(ctrl) if s % 2 == 0]
    od = [s for s in sorted(ctrl) if s % 2 == 1]
    a, b = arm_mean(ctrl, mats, ev), arm_mean(ctrl, mats, od)
    d_sham = [a[m] - b[m] for m in mats]
    p_sham, mean_sham = perm_p(d_sham, n_perm=n_perm)
    ops["OP4"] = {"ok": p_sham >= 0.01, "p": p_sham, "mean": mean_sham,
                  "even": ev, "odd": od}
    res["ops"] = ops
    if not ops["OP4"]["ok"]:
        res["verdict"] = "VOID_PAIRED_FALSEPOS"
        return res

    # ---- 主判据 ----
    mc, mt = arm_mean(ctrl, mats), arm_mean(trt, mats)
    d = [mt[m] - mc[m] for m in mats]
    p, obs = perm_p(d, n_perm=n_perm)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    res["main"] = {
        "n_materials": len(mats), "mean_D": obs, "p_perm": p,
        "pos": pos, "neg": neg, "p_sign": sign_test_p(pos, neg),
    }
    if p < 0.05:
        res["verdict"] = "SCALE_HELPS_32" if obs > 0 else "SCALE_HURTS_32"
    else:
        res["verdict"] = "SCALE_NULL_32"

    # ---- 只登记 ----
    s1 = sigma1_of(ctrl)
    mde = None
    if s1:
        sd_diff = s1 * math.sqrt(2.0 / k)
        mde = Z80_TWOSIDED * sd_diff / math.sqrt(len(mats))
    res["registered"] = {
        "sigma1_ctrl_measured": s1,
        "sigma1_m60_prereg": SIGMA1_M60,
        "mde_magnitude": mde,
        "mde_frac_of_ceiling": (mde / CEILING_M58) if mde else None,
        "ceiling_m58": CEILING_M58,
        "per_material_D_sd": mean_sd(d)[1],
    }
    return res


def selftest():
    """构造数据核算式与全部 VOID 分支，⛔ 不碰真实读数。"""
    ok = 0
    mats = ["m%03d" % i for i in range(N_MAT_EXPECT)]
    rng = random.Random(7)
    base = {m: 30.0 + rng.random() for m in mats}
    K = 6

    def mk(shift, sd=0.85, seed=0):
        r = random.Random(seed)
        return {s: {m: base[m] + shift + r.gauss(0, sd) for m in mats} for s in range(K)}

    m59 = {s: 33.0 for s in range(5)}

    # 1) 零效应 -> NULL
    r = analyse(mk(0.0, seed=1), mk(0.0, seed=2), K, op3_off=[], n_perm=2000)
    assert r["verdict"] == "SCALE_NULL_32", r["verdict"]; ok += 1
    assert r["ops"]["OP1"]["ok"] and r["ops"]["OP4"]["ok"]; ok += 1
    # 2) 大正效应 -> HELPS
    r = analyse(mk(0.0, seed=3), mk(+1.5, seed=4), K, op3_off=[], n_perm=2000)
    assert r["verdict"] == "SCALE_HELPS_32", r["verdict"]; ok += 1
    assert r["main"]["mean_D"] > 1.0; ok += 1
    # 3) 大负效应 -> HURTS
    r = analyse(mk(0.0, seed=5), mk(-1.5, seed=6), K, op3_off=[], n_perm=2000)
    assert r["verdict"] == "SCALE_HURTS_32", r["verdict"]; ok += 1
    # 4) 缺种子 -> VOID_NO_DATA
    c = mk(0.0, seed=7); c.pop(K - 1)
    r = analyse(c, mk(0.0, seed=8), K, op3_off=[], n_perm=200)
    assert r["verdict"] == "VOID_NO_DATA", r["verdict"]; ok += 1
    # 5) 材质集合不同 -> VOID_NO_DATA
    t = mk(0.0, seed=9)
    for s in t:
        t[s].pop(mats[0])
    r = analyse(mk(0.0, seed=10), t, K, op3_off=[], n_perm=200)
    assert r["verdict"] == "VOID_NO_DATA", r["verdict"]; ok += 1
    # 6) 没给 (M59) 行 -> VOID_CTRL_NOT_REPRODUCED
    r = analyse(mk(0.0, seed=11), mk(0.0, seed=12), K, op3_off=None, n_perm=200)
    assert r["verdict"] == "VOID_CTRL_NOT_REPRODUCED", r["verdict"]; ok += 1
    # 6b) 控制臂没复现 (M59) -> 同样 VOID
    r = analyse(mk(0.0, seed=11), mk(0.0, seed=12), K, op3_off=[[0, 33.0, 31.0]], n_perm=200)
    assert r["verdict"] == "VOID_CTRL_NOT_REPRODUCED", r["verdict"]; ok += 1
    # 6c) op3_offenders 本体
    assert op3_offenders({0: 1.0}, None) is None; ok += 1
    assert op3_offenders({s: 33.0 for s in range(5)}, {s: 33.0 for s in range(5)}) == []; ok += 1
    assert len(op3_offenders({s: 33.0 for s in range(5)}, {s: 33.0 + (s == 2) for s in range(5)})) == 1; ok += 1
    # 7) 置换检验：常数正差必然显著；全零差必然不显著
    p, o = perm_p([0.5] * N_MAT_EXPECT, n_perm=2000)
    assert p < 0.01 and abs(o - 0.5) < 1e-12; ok += 1
    p, o = perm_p([0.0] * N_MAT_EXPECT, n_perm=500)
    assert p > 0.9 and abs(o) < 1e-12; ok += 1
    # 8) 置换检验对称（符号翻转不改 p）
    d = [rng.gauss(0.2, 1.0) for _ in mats]
    p1, _ = perm_p(d, n_perm=3000)
    p2, _ = perm_p([-x for x in d], n_perm=3000)
    assert abs(p1 - p2) < 1e-12; ok += 1
    # 9) 符号检验对照 (M59) 已知值
    assert 0.2 < sign_test_p(39, 28) < 0.25; ok += 1
    assert sign_test_p(48, 19) < 1e-3; ok += 1
    # 10) MDE 随 K 下降、且 K=17 时与预注册值一致
    s1 = SIGMA1_M60
    m17 = Z80_TWOSIDED * s1 * math.sqrt(2.0 / 17) / math.sqrt(N_MAT_EXPECT)
    m5 = Z80_TWOSIDED * s1 * math.sqrt(2.0 / 5) / math.sqrt(N_MAT_EXPECT)
    assert abs(m17 - 0.0996) < 5e-4, m17; ok += 1
    assert abs(m5 - 0.1836) < 5e-4, m5; ok += 1
    assert m17 < m5; ok += 1
    # 11) per_material / mean_sd
    pm = per_material([1.0, 3.0, 5.0], ["a", "a", "b"])
    assert abs(pm["a"] - 2.0) < 1e-12 and abs(pm["b"] - 5.0) < 1e-12; ok += 1
    mu, sd = mean_sd([1.0, 2.0, 3.0])
    assert abs(mu - 2.0) < 1e-12 and abs(sd - 1.0) < 1e-12; ok += 1
    print("selftest %d/%d OK" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp")
    ap.add_argument("--ctrl_tag", default="m60_ctrl")
    ap.add_argument("--trt_tag", default="m60_more")
    ap.add_argument("--k", type=int, default=17)
    ap.add_argument("--m59dir", default="experiments")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    ctrl, ctrl_rows, pc = load_arm(a.dir, a.ctrl_tag, a.k)
    trt, trt_rows, pt = load_arm(a.dir, a.trt_tag, a.k)

    # (OP3) 的料：(M59) 前 5 个种子的 TRD 行均值
    m59 = {}
    for s in range(5):
        p = os.path.join(a.m59dir, "m59_pairedclip_s%d.json" % s)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                m59[s] = json.load(f).get(ROW, {}).get("CLIP")

    res = analyse(ctrl, trt, a.k, op3_off=op3_offenders(ctrl_rows, m59))
    res["load_problems"] = {"ctrl": pc, "trt": pt}
    res["ctrl_row_means"] = ctrl_rows
    res["trt_row_means"] = trt_rows
    res["m59_rows"] = m59

    # 落盘一律在打印之前
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M60) scale arm on paired CLIP@32px ==")
    for name in ("OP1", "OP2", "OP3", "OP4"):
        o = res.get("ops", {}).get(name)
        print("  %s: %s" % (name, o))
    if "main" in res:
        m = res["main"]
        print("main: n=%d  mean_D=%+.4f  p_perm=%.4f  pos/neg=%d/%d  p_sign=%.4f"
              % (m["n_materials"], m["mean_D"], m["p_perm"], m["pos"], m["neg"], m["p_sign"]))
        r = res["registered"]
        print("registered: sigma1=%.4f (prereg %.4f)  MDE=%.4f = %.0f%% of ceiling %.4f"
              % (r["sigma1_ctrl_measured"], r["sigma1_m60_prereg"], r["mde_magnitude"],
                 100 * r["mde_frac_of_ceiling"], r["ceiling_m58"]))
    print("VERDICT:", res["verdict"])
    if res["verdict"] == "SCALE_NULL_32":
        print("  【禁】只许读成: 效应 < MDE; 不许读成 '训练规模无用'")


if __name__ == "__main__":
    main()
