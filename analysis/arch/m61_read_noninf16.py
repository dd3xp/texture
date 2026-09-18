r"""(M61) 盲写判读器：(M60) 那个「再训一倍」的检查点在 **16px** 上有没有退步（**非劣性**判据）。

⛔ 本文件随预注册提交，**跑完一个字不许改**。⛔ 写它的时候 16px 的处理臂读数还不存在。

## 为什么是"非劣"不是"更好"

(M60) 判 `SCALE_HELPS_32`：同配方再训 12000 步，配对 CLIP@32px 涨 +0.1250（p=0.0114）。
(M60) 的分支表第一行写死：**先量同一检查点的 16px 有没有退步** —— 32px 涨而 16px 掉
＝ 换了配方而不是变强。⇒ 主判据是**非劣性**，必须先定等效边界 δ，⛔ 不许跑完再挑。

## 统计单位（零 GPU 事先量定，`eval/colour_task.py:targets("V_mat",16)`）

**材质，n = 125**（196 个 V_mat@16 目标分属 125 个材质：78 个各 1 张、23 个各 2 张、24 个各 3 张；
reps=2 ⇒ 每份 JSON 392 张图）。⛔ 不许改用 392 当分母（同材质多张不独立，(M59) 定死的纪律）。
⚠ 这个 125 与判官 `--set V_mat` 的 125 条目**是同一批材质**（16px 上每个材质都至少有 1 张参照瓦片；
32px 上只剩 67 个 ⇒ 那里 n=67）。

    D_m = mean_{seed,rep}[ CLIP_处理(m) ] − mean_{seed,rep}[ CLIP_控制(m) ]

控制臂 = `runs/trd_v10`，处理臂 = 同配方再训 12000 步的那个检查点（(M60) 的处理臂，
`/tmp/runs/trd_v10more_09180526`）。两臂同一批目标、同一批种子、同一条 `diag_decompose` 命令。

## 等效边界 δ（预注册公式，⛔ 数字由试点填、公式不许改）

    δ = (0.1250 / 0.5127) × ceiling16 = 0.243807 × ceiling16

分子分母都是**已发表的读数**：0.1250 = (M60) 32px 配对增益，0.5127 = (M58) 32px 天花板
（`real_half` − `TRD`）。⇒ δ ＝「把 32px 那次增益换算成*天花板占比*后，搬到 16px 的同一个占比」
＝ **换算成各自尺寸的天花板单位后的盈亏平衡点**。宣布非劣 ＝ 排除了「16px 亏掉的比 32px 赚到的还多」。

`ceiling16` = `real_half.CLIP` − `TRD.CLIP`，由**试点臂**（控制臂、种子 100..104）实测。
⚠⚠ **披露**：`real_half` 行只由参照集前后对半决定，**与模型、与种子都无关**（`diag_decompose.py:301`
是 `ref[:half]` vs `ref[half:]`，已在 (M58) 五份 JSON 上实测逐位相同）⇒ 它在 16px 上的值
**在写本文件之前就已知**：`experiments/m46_diag_decompose_v8_Vmat.json` 里 `real_half.CLIP`
= 33.69111633300781（n=98）。所以 `ceiling16` 唯一未知的一半是 `TRD.CLIP`@16px@v10。
⚠ 同一份 (M46) JSON 里 v8 的 TRD@16 = 33.49737548828125 ⇒ **v8 的 ceiling16 ≈ 0.1937 也已见过**。
⛔ 但 δ 必须用 **v10 试点实测**的 ceiling16，⛔ 不许改用 v8 那个数、⛔ 不许跑完换分母。

## 主检验（唯一下判的一条）

**逐材质配对的符号翻转置换检验**（20000 次，rng 固定 12345；`jzs_train` 无 scipy ⇒ 不做正态假设）：

- **非劣**（单侧，H0: mean_D = −δ）：对 `D_m + δ` 做符号翻转，
  `p_ni = P(perm_mean >= obs_mean(D+δ))`；`p_ni < 0.05` ⇒ 排除「亏损 ≥ δ」。
- **方向**（双侧，H0: mean_D = 0）：与 (M60) 逐字同一条检验。

| 判决 | 条件 |
|---|---|
| **`SCALE_HELPS_16`** | `p_ni < 0.05` 且 `p_two < 0.05` 且 `mean_D > 0` |
| **`NONINF_16`** | `p_ni < 0.05`（但不满足上一行） |
| **`SCALE_HURTS_16`** | `p_ni >= 0.05` 且 `p_two < 0.05` 且 `mean_D < 0` |
| **`INCONCLUSIVE_16`** | `p_ni >= 0.05` 且不满足上一行 |

⚠⚠ **`INCONCLUSIVE_16` 的唯一许可读法**：「这批料排除不了『16px 亏了 δ 或更多』，也没证到亏。」
⛔ 不许读成"16px 没事"、⛔ 不许读成"16px 坏了"、⛔ 不许事后放宽 δ 或加 K 重判
（要重问必须另行预注册，并把本轮读数作废）。
⚠ **`NONINF_16` 的唯一许可读法**：「在这把尺子上排除了『16px 的亏损 ≥ δ』」。
⛔ 不许读成"16px 没变化"（那是接受零假设）、⛔ 不许读成"可以改 `final_test.sh` 了"
（改主配置要回验证集重选＋另行预注册，(M18) 写死）。

## 操作检验（任一不过即 VOID，按 OP1→…→OP5 顺序，⛔ 不补救不改判据）

| 检验 | 要求 | 不过则 |
|---|---|---|
| (OP1) 料齐 | 两臂各 `--k` 份；每份 `TRD._CLIP_per` 与 `_mats` 长 **392**；材质数 **125**；两臂材质集合相同 | `VOID_NO_DATA` |
| (OP2) 新列自洽 | 每份 \|mean(`_CLIP_per`) − `CLIP`\| < 1e-4 | `VOID_PER_IMAGE_MISMATCH` |
| (OP3) 参照集同一 | **全部** JSON（两臂 + 试点）的 `real_half.CLIP` 逐位相同（tol 1e-9）且 `real_half.n` = 98 | `VOID_REF_SET_DIFFERS` |
| (OP4) 假阳检验 | 控制臂**内部**按种子奇偶对半的双侧检验 `p >= 0.01` | `VOID_PAIRED_FALSEPOS` |
| (OP5) 两臂确非同一模型 | 存在种子使 \|`TRD.CLIP`处理 − `TRD.CLIP`控制\| > 1e-6 | `VOID_SAME_MODEL` |

⚑ (OP3) 是本轮白拿的一条：`real_half` 与模型、与种子都无关 ⇒ 它一旦在两臂间不等，
说明两边根本没在同一批参照瓦片上评测（(M14) 那条"比多条臂前必须查 n/materials 相等"的机械版）。
⚑ (OP5) 挡的是"`--run` 打错指回同一个目录"——生成逐像素确定已九次实测 ⇒ 那样两臂会**逐位相同**。

**只登记、不下判**：逐材质符号检验；控制臂实测 σ1 与由它重算的 MDE；逐材质 `D` 的 sd；
δ 与 `ceiling16`；`D` 的单侧 95% 下界；控制臂对半自己做非劣检验的 `p_ni`（功效自查，
⚠ 不是门 —— ⛔ 不许因为它不过就改判据）。

用法：
    python analysis/arch/m61_read_noninf16.py --dir /tmp --ctrl_tag m61_ctrl --trt_tag m61_more \
           --k 20 --delta 0.0472 --ceiling16 0.1937 --pilot_tag m61_pilot --pilot_seeds 100 101 102 103 104 \
           --out /tmp/m61_noninf16.json
    python analysis/arch/m61_read_noninf16.py --selftest
"""
import argparse
import json
import math
import os
import random

N_IMG_EXPECT = 392            # 196 个 V_mat@16 目标 x reps 2
N_MAT_EXPECT = 125            # 零 GPU 事先量定（colour_task.targets("V_mat",16)）
N_FLOOR_EXPECT = 98           # real_half 的 n = 196 // 2
ROW = "TRD"
FLOOR_ROW = "real_half"
N_PERM = 20000
PERM_SEED = 12345
GAIN_M60_32 = 0.1250          # (M60) 32px 配对增益
CEILING_M58_32 = 0.5127       # (M58) 32px 天花板 real_half - TRD
DELTA_FRAC = GAIN_M60_32 / CEILING_M58_32     # 0.243807...
Z80_TWOSIDED = 2.801586       # z(0.975) + z(0.80)
Z80_ONESIDED = 2.486475       # z(0.95)  + z(0.80)


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
    """逐材质配对的符号翻转置换检验（双侧，H0: mean = 0）。返回 (p, observed_mean)。"""
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


def perm_p_noninf(d, delta, n_perm=N_PERM, seed=PERM_SEED):
    """非劣性检验（单侧）。H0: mean_D = -delta ⇒ 把 D 平移 +delta 后对称零假设成立。
    p = P(perm_mean >= obs_mean)，小 p ⇒ 排除「亏损 >= delta」。返回 (p, obs_shifted_mean)。"""
    sh = [x + delta for x in d]
    obs = sum(sh) / len(sh)
    rng = random.Random(seed)
    hit = 0
    for _ in range(n_perm):
        s = 0.0
        for x in sh:
            s += x if rng.random() < 0.5 else -x
        if s / len(sh) >= obs - 1e-15:
            hit += 1
    return (hit + 1.0) / (n_perm + 1.0), obs


def per_material(per_img, mats):
    acc = {}
    for v, m in zip(per_img, mats):
        acc.setdefault(m, []).append(v)
    return {m: sum(v) / len(v) for m, v in acc.items()}


def load_one(path):
    """返回 (per_material_dict, row_mean, floor_clip, floor_n, problem_or_None)。"""
    if not os.path.exists(path):
        return None, None, None, None, "missing"
    with open(path, encoding="utf-8") as f:        # Windows 默认 GBK 会崩
        j = json.load(f)
    row = j.get(ROW, {})
    per, mats = row.get("_CLIP_per"), j.get("_mats")
    fl = j.get(FLOOR_ROW, {})
    fc, fn = fl.get("CLIP"), fl.get("n")
    if not isinstance(per, list) or not isinstance(mats, list):
        return None, None, fc, fn, "no _CLIP_per/_mats"
    if len(per) != N_IMG_EXPECT or len(mats) != N_IMG_EXPECT:
        return None, None, fc, fn, "len %d/%d" % (len(per), len(mats))
    if abs(sum(per) / len(per) - row.get("CLIP", float("nan"))) >= 1e-4:
        return None, None, fc, fn, "per_image mismatch"
    return per_material(per, mats), row.get("CLIP"), fc, fn, None


def load_arm(d, tag, seeds):
    by, rowmean, floors, prob = {}, {}, {}, []
    for s in seeds:
        pm, rm, fc, fn, err = load_one(os.path.join(d, "%s_s%d.json" % (tag, s)))
        if err:
            prob.append((s, err))
            if fc is not None:
                floors[s] = (fc, fn)
            continue
        by[s], rowmean[s], floors[s] = pm, rm, (fc, fn)
    return by, rowmean, floors, prob


def sigma1_of(by):
    """由全部种子对的逐材质差反解单次生成噪声 sd（与 (M60) 逐字同一算法）。"""
    seeds = sorted(by)
    if len(seeds) < 2:
        return None
    mats = sorted(set.intersection(*[set(by[s]) for s in seeds]))
    sds = []
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            d = [by[seeds[i]][m] - by[seeds[j]][m] for m in mats]
            sds.append(mean_sd(d)[1])
    return (sum(sds) / len(sds)) / math.sqrt(2.0)


def arm_mean(by, mats, seeds=None):
    seeds = sorted(by) if seeds is None else sorted(seeds)
    return {m: sum(by[s][m] for s in seeds) / len(seeds) for m in mats}


def k_required(sigma1, delta, n_mat=N_MAT_EXPECT, z=Z80_ONESIDED):
    """单侧非劣检验、80% 功效（真值 D=0）所需的每臂种子数。"""
    if not sigma1 or delta <= 0:
        return None
    return 2.0 * (z * sigma1) ** 2 / (n_mat * delta ** 2)


def floor_offenders(floors, tol=1e-9):
    """(OP3)：所有 JSON 的 real_half.CLIP 必须逐位相同、n 必须是 98。返回不合格清单。"""
    vals = [(k, v[0], v[1]) for k, v in floors.items() if v and v[0] is not None]
    if not vals:
        return None                     # None = 一份也没读到，(OP3) 直接不过
    ref = vals[0][1]
    return [[k, c, n] for k, c, n in vals
            if n != N_FLOOR_EXPECT or abs(c - ref) > tol]


def analyse(ctrl, trt, k, delta, floor_off=None, ctrl_rows=None, trt_rows=None,
            n_perm=N_PERM, ceiling16=None):
    res = {"k_requested": k, "n_perm": n_perm, "delta": delta, "ceiling16": ceiling16,
           "delta_frac_of_ceiling": DELTA_FRAC}
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

    # ---- (OP2) 已在 load_one 里逐份查过（不自洽的份进不来）----
    ops["OP2"] = {"ok": True, "note": "checked per file at load time (tol 1e-4)"}

    # ---- (OP3) 参照集同一（real_half 与模型/种子无关 ⇒ 必须逐位相同）----
    ops["OP3"] = {"ok": floor_off is not None and len(floor_off) == 0,
                  "off": floor_off, "note": "None = no real_half row readable"}
    if not ops["OP3"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_REF_SET_DIFFERS"
        return res

    # ---- (OP4) 控制臂内部假阳检验（奇偶对半）----
    ev = [s for s in sorted(ctrl) if s % 2 == 0]
    od = [s for s in sorted(ctrl) if s % 2 == 1]
    a_, b_ = arm_mean(ctrl, mats, ev), arm_mean(ctrl, mats, od)
    d_sham = [a_[m] - b_[m] for m in mats]
    p_sham, mean_sham = perm_p(d_sham, n_perm=n_perm)
    ops["OP4"] = {"ok": p_sham >= 0.01, "p": p_sham, "mean": mean_sham,
                  "even": ev, "odd": od}
    if not ops["OP4"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_PAIRED_FALSEPOS"
        return res

    # ---- (OP5) 两臂确非同一模型 ----
    diffs = []
    if ctrl_rows and trt_rows:
        diffs = [abs(trt_rows[s] - ctrl_rows[s]) for s in sorted(ctrl_rows)
                 if s in trt_rows and ctrl_rows[s] is not None and trt_rows[s] is not None]
    ops["OP5"] = {"ok": bool(diffs) and max(diffs) > 1e-6,
                  "max_abs_row_diff": max(diffs) if diffs else None}
    res["ops"] = ops
    if not ops["OP5"]["ok"]:
        res["verdict"] = "VOID_SAME_MODEL"
        return res

    # ---- 主判据 ----
    mc, mt = arm_mean(ctrl, mats), arm_mean(trt, mats)
    d = [mt[m] - mc[m] for m in mats]
    p_ni, _ = perm_p_noninf(d, delta, n_perm=n_perm)
    p_two, obs = perm_p(d, n_perm=n_perm)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    res["main"] = {"n_materials": len(mats), "mean_D": obs,
                   "p_noninf": p_ni, "p_two": p_two,
                   "pos": pos, "neg": neg, "p_sign": sign_test_p(pos, neg)}
    if p_ni < 0.05:
        res["verdict"] = "SCALE_HELPS_16" if (p_two < 0.05 and obs > 0) else "NONINF_16"
    elif p_two < 0.05 and obs < 0:
        res["verdict"] = "SCALE_HURTS_16"
    else:
        res["verdict"] = "INCONCLUSIVE_16"

    # ---- 只登记 ----
    s1 = sigma1_of(ctrl)
    sd_d = mean_sd(d)[1]
    mde2 = Z80_TWOSIDED * s1 * math.sqrt(2.0 / k) / math.sqrt(len(mats)) if s1 else None
    mde1 = Z80_ONESIDED * s1 * math.sqrt(2.0 / k) / math.sqrt(len(mats)) if s1 else None
    p_sham_ni, _ = perm_p_noninf(d_sham, delta, n_perm=n_perm)
    res["registered"] = {
        "sigma1_ctrl_measured": s1,
        "mde_twosided": mde2,
        "mde_noninf_onesided": mde1,
        "mde_noninf_frac_of_delta": (mde1 / delta) if (mde1 and delta) else None,
        "per_material_D_sd": sd_d,
        "D_lower95_normal_approx": obs - 1.644854 * sd_d / math.sqrt(len(mats)),
        "ctrl_halves_p_noninf": p_sham_ni,
        "k_required_for_delta": k_required(s1, delta),
    }
    return res


def selftest():
    """构造数据核算式与全部 VOID 分支，⛔ 不碰真实读数。"""
    ok = 0
    mats = ["m%03d" % i for i in range(N_MAT_EXPECT)]
    rng = random.Random(7)
    base = {m: 33.0 + rng.random() for m in mats}
    K = 6
    DELTA = 0.20        # 选得比这组构造数据的 SE 大几倍，否则连真零效应也判不出非劣

    def mk(shift, sd=0.9, seed=0):
        r = random.Random(seed)
        return {s: {m: base[m] + shift + r.gauss(0, sd) for m in mats} for s in range(K)}

    rows_c = {s: 33.5 for s in range(K)}
    rows_t = {s: 33.6 for s in range(K)}

    # 1) 真零效应 + 足够功效 -> NONINF_16
    r = analyse(mk(0.0, seed=1), mk(0.0, seed=2), K, DELTA, floor_off=[],
                ctrl_rows=rows_c, trt_rows=rows_t, n_perm=2000)
    assert r["verdict"] == "NONINF_16", r["verdict"]; ok += 1
    assert all(r["ops"][o]["ok"] for o in ("OP1", "OP3", "OP4", "OP5")); ok += 1
    # 2) 大正效应 -> HELPS（非劣也必然过）
    r = analyse(mk(0.0, seed=3), mk(+1.0, seed=4), K, DELTA, floor_off=[],
                ctrl_rows=rows_c, trt_rows=rows_t, n_perm=2000)
    assert r["verdict"] == "SCALE_HELPS_16", r["verdict"]; ok += 1
    assert r["main"]["p_noninf"] < 0.05 and r["main"]["p_two"] < 0.05; ok += 1
    # 3) 大负效应 -> HURTS（非劣不过 + 双侧显著为负）
    r = analyse(mk(0.0, seed=5), mk(-1.0, seed=6), K, DELTA, floor_off=[],
                ctrl_rows=rows_c, trt_rows=rows_t, n_perm=2000)
    assert r["verdict"] == "SCALE_HURTS_16", r["verdict"]; ok += 1
    assert r["main"]["mean_D"] < 0; ok += 1
    # 4) 噪声大到既排不掉 -delta 又证不到亏 -> INCONCLUSIVE_16
    r = analyse(mk(0.0, sd=9.0, seed=13), mk(0.0, sd=9.0, seed=14), K, 0.001, floor_off=[],
                ctrl_rows=rows_c, trt_rows=rows_t, n_perm=2000)
    assert r["verdict"] == "INCONCLUSIVE_16", r["verdict"]; ok += 1
    # 5) 缺种子 -> VOID_NO_DATA
    c = mk(0.0, seed=7); c.pop(K - 1)
    r = analyse(c, mk(0.0, seed=8), K, DELTA, floor_off=[], ctrl_rows=rows_c,
                trt_rows=rows_t, n_perm=200)
    assert r["verdict"] == "VOID_NO_DATA", r["verdict"]; ok += 1
    # 6) 材质集合不同 -> VOID_NO_DATA
    t = mk(0.0, seed=9)
    for s in t:
        t[s].pop(mats[0])
    r = analyse(mk(0.0, seed=10), t, K, DELTA, floor_off=[], ctrl_rows=rows_c,
                trt_rows=rows_t, n_perm=200)
    assert r["verdict"] == "VOID_NO_DATA", r["verdict"]; ok += 1
    # 7) 参照集不同 -> VOID_REF_SET_DIFFERS（含"一份也没读到"）
    r = analyse(mk(0.0, seed=11), mk(0.0, seed=12), K, DELTA, floor_off=None,
                ctrl_rows=rows_c, trt_rows=rows_t, n_perm=200)
    assert r["verdict"] == "VOID_REF_SET_DIFFERS", r["verdict"]; ok += 1
    r = analyse(mk(0.0, seed=11), mk(0.0, seed=12), K, DELTA, floor_off=[[3, 33.0, 98]],
                ctrl_rows=rows_c, trt_rows=rows_t, n_perm=200)
    assert r["verdict"] == "VOID_REF_SET_DIFFERS", r["verdict"]; ok += 1
    # 8) 两臂逐位相同 -> VOID_SAME_MODEL
    same = mk(0.0, seed=15)
    r = analyse(same, {s: dict(v) for s, v in same.items()}, K, DELTA, floor_off=[],
                ctrl_rows=rows_c, trt_rows=dict(rows_c), n_perm=200)
    assert r["verdict"] == "VOID_SAME_MODEL", r["verdict"]; ok += 1
    r = analyse(mk(0.0, seed=16), mk(0.0, seed=17), K, DELTA, floor_off=[],
                ctrl_rows=None, trt_rows=None, n_perm=200)
    assert r["verdict"] == "VOID_SAME_MODEL", r["verdict"]; ok += 1

    # 9) floor_offenders 本体
    assert floor_offenders({}) is None; ok += 1
    assert floor_offenders({0: (34.0, 98), 1: (34.0, 98)}) == []; ok += 1
    assert len(floor_offenders({0: (34.0, 98), 1: (34.1, 98)})) == 1; ok += 1
    assert len(floor_offenders({0: (34.0, 98), 1: (34.0, 97)})) == 1; ok += 1

    # 10) 非劣检验的方向性：常数差 = -delta 时 p 应≈0.5 附近而非显著
    p, _ = perm_p_noninf([-0.05] * 40 + [-0.05] * 40, 0.05, n_perm=2000)
    assert p > 0.4, p; ok += 1
    p, _ = perm_p_noninf([0.0] * N_MAT_EXPECT, 0.05, n_perm=500)   # D=0 常数 ⇒ 平移后全 +delta
    assert p < 0.01, p; ok += 1
    p, _ = perm_p_noninf([-0.5] * N_MAT_EXPECT, 0.05, n_perm=500)  # 亏得远超 delta
    assert p > 0.99, p; ok += 1
    # 11) 双侧置换检验（与 (M60) 同一实现）
    p, o = perm_p([0.5] * N_MAT_EXPECT, n_perm=2000)
    assert p < 0.01 and abs(o - 0.5) < 1e-12; ok += 1
    p, o = perm_p([0.0] * N_MAT_EXPECT, n_perm=500)
    assert p > 0.9 and abs(o) < 1e-12; ok += 1
    d = [rng.gauss(0.2, 1.0) for _ in mats]
    p1, _ = perm_p(d, n_perm=3000)
    p2, _ = perm_p([-x for x in d], n_perm=3000)
    assert abs(p1 - p2) < 1e-12; ok += 1
    # 12) 符号检验对照 (M59)/(M60) 已知值
    assert 0.2 < sign_test_p(39, 28) < 0.25; ok += 1
    assert sign_test_p(45, 22) < 0.01; ok += 1
    # 13) delta 公式与 k_required 的量纲
    assert abs(DELTA_FRAC - 0.243807) < 1e-5, DELTA_FRAC; ok += 1
    assert k_required(0.9, 0.05) > k_required(0.9, 0.10); ok += 1
    assert k_required(0.9, 0.05) > k_required(0.45, 0.05); ok += 1
    assert k_required(None, 0.05) is None and k_required(0.9, 0.0) is None; ok += 1
    # n_mat 从 67 涨到 125 必然省种子（本轮相对 (M60) 白拿的那一半）
    assert k_required(0.9, 0.1, n_mat=125) < k_required(0.9, 0.1, n_mat=67); ok += 1
    # 14) per_material / mean_sd
    pm = per_material([1.0, 3.0, 5.0], ["a", "a", "b"])
    assert abs(pm["a"] - 2.0) < 1e-12 and abs(pm["b"] - 5.0) < 1e-12; ok += 1
    mu, sd = mean_sd([1.0, 2.0, 3.0])
    assert abs(mu - 2.0) < 1e-12 and abs(sd - 1.0) < 1e-12; ok += 1
    print("selftest %d/%d OK" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp")
    ap.add_argument("--ctrl_tag", default="m61_ctrl")
    ap.add_argument("--trt_tag", default="m61_more")
    ap.add_argument("--pilot_tag", default="m61_pilot")
    ap.add_argument("--pilot_seeds", type=int, nargs="*", default=[100, 101, 102, 103, 104])
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--ceiling16", type=float, default=None)
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.k is None or a.delta is None:
        ap.error("--k 与 --delta 必须显式给（由试点功效器 m61_power16.py 算出、写进账本后再传）")

    seeds = list(range(a.k))
    ctrl, ctrl_rows, fl_c, pc = load_arm(a.dir, a.ctrl_tag, seeds)
    trt, trt_rows, fl_t, pt = load_arm(a.dir, a.trt_tag, seeds)
    _, _, fl_p, pp = load_arm(a.dir, a.pilot_tag, a.pilot_seeds)

    floors = {}
    for tag, fl in (("c", fl_c), ("t", fl_t), ("p", fl_p)):
        for s, v in fl.items():
            floors["%s%d" % (tag, s)] = v

    res = analyse(ctrl, trt, a.k, a.delta, floor_off=floor_offenders(floors),
                  ctrl_rows=ctrl_rows, trt_rows=trt_rows, ceiling16=a.ceiling16)
    res["load_problems"] = {"ctrl": pc, "trt": pt, "pilot": pp}
    res["ctrl_row_means"] = ctrl_rows
    res["trt_row_means"] = trt_rows
    res["floor_clip"] = floors

    # 落盘一律在打印之前
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M61) non-inferiority at 16px, paired CLIP ==")
    print("  delta=%.4f (= %.4f x ceiling16=%s)" % (a.delta, DELTA_FRAC, a.ceiling16))
    for name in ("OP1", "OP2", "OP3", "OP4", "OP5"):
        print("  %s: %s" % (name, res.get("ops", {}).get(name)))
    if "main" in res:
        m = res["main"]
        print("main: n=%d  mean_D=%+.4f  p_noninf=%.4f  p_two=%.4f  pos/neg=%d/%d  p_sign=%.4f"
              % (m["n_materials"], m["mean_D"], m["p_noninf"], m["p_two"],
                 m["pos"], m["neg"], m["p_sign"]))
        r = res["registered"]
        print("registered: sigma1=%.4f  MDE_1side=%.4f (%.0f%% of delta)  D_low95=%+.4f  "
              "ctrl_halves_p_ni=%.4f  k_req=%.1f"
              % (r["sigma1_ctrl_measured"], r["mde_noninf_onesided"],
                 100 * r["mde_noninf_frac_of_delta"], r["D_lower95_normal_approx"],
                 r["ctrl_halves_p_noninf"], r["k_required_for_delta"]))
    print("VERDICT:", res["verdict"])
    if res["verdict"] == "INCONCLUSIVE_16":
        print("  【禁】只许读成: 既排不掉'亏 >= delta'也没证到亏; 不许放宽 delta 或加 K 重判")
    if res["verdict"] == "NONINF_16":
        print("  【禁】只许读成: 排除了'亏损 >= delta'; 不许读成'16px 没变化'、不许据此改 final_test.sh")


if __name__ == "__main__":
    main()
