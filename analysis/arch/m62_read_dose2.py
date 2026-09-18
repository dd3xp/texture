r"""(M62) 盲写判读器：训练预算的**第二次翻倍**在配对 CLIP@32px 上还涨不涨。

⛔ 本文件随预注册提交，**跑完一个字不许改**。⛔ 写它的时候处理臂（24000 步那个检查点）
还不存在 —— 它的训练在本文件提交之后才会挂起。

## 问的是什么（与 (M60) 的区别）

(M60) 判 `SCALE_HELPS_32`：`runs/trd_v10` 再训 **12000** 步 ⇒ 逐材质配对 CLIP +0.1250（p=0.0114）。
本轮问**下一档剂量**：同一个起点 `runs/trd_v10/last.pt`，同一条命令，只把 `--steps 12000`
换成 **24000**（＝在 v10 之后的训练量上再翻一倍）。

    控制臂 = v10 + 12000 步  =  (M60) 的处理臂（`/tmp/runs/trd_v10more_09180526`）
    处理臂 = v10 + 24000 步  =  本轮新训的检查点

⚑ 两臂**都是从 `runs/trd_v10/last.pt` 出发的单个续训周期**（同一条 lr 日程形状、同一批非默认键），
唯一的差别是周期长度 12000 vs 24000 ⇒ 这是一次干净的**剂量**对比。
⛔ 不是"再续一段"（那会变成第三次 warmup+decay，剂量含义不同）。

## 统计单位（与 (M58)/(M59)/(M60) 逐字相同）

**材质（提示词），n = 67**（32px 上 slug 数与提示词数恰好相同）。⛔ 不许改用 148 张当分母。

    D_m = mean_{seed,rep}[ CLIP_处理(m) ] − mean_{seed,rep}[ CLIP_控制(m) ]

## 主检验（唯一下判的一条）

**逐材质配对的符号翻转置换检验**（双侧，20000 次，rng 固定 12345；`jzs_train` 无 scipy）。
⚑ 检验本体、`load_arm`、`per_material`、`sigma1_of` 全部 **import (M60) 那份冻结判读器**
（`analysis/arch/m60_read_scale.py`，selftest 22/22、已在真实料上跑过两轮）——
⛔ 判据不重写（记忆里那条纪律：预检/新臂要 import 冻结判读器本体，别重写判据）。

| 判决 | 条件 |
|---|---|
| **`DOSE2_HELPS_32`** | `p < 0.05` 且 `mean_D > 0` |
| **`DOSE2_HURTS_32`** | `p < 0.05` 且 `mean_D < 0` |
| **`DOSE2_NULL_32`** | `p >= 0.05` |

⚠⚠ **`DOSE2_NULL_32` 的唯一许可读法**：「**第二次**翻倍在这把尺子上带来的 32px CLIP 变化
**小于本臂的 MDE**（K=28、σ1=0.8988 ⇒ 约 **0.0822**，＝ (M60) 那次增益 0.1250 的 **66%**）」。
⛔ 不许读成"训练规模无用"（(M60) 已判它有用）、⛔ 不许读成"模型训够了"、
⛔ 不许读成"该转去加大模型"（那是另一条臂、要另行预注册）、
⛔ 不许把本轮 `mean_D` 与 (M60) 的 +0.1250 连成"随剂量递减/递增"的曲线
（两个对比的控制臂不同、噪声相关，**不是同一把尺子上的两点**）。

⚠ 任一判决都**不授权**改 `eval/final_test.sh`、**不授权**开判官臂（32px 准入条件② 未松）。

## 操作检验（任一不过即 VOID，按 OP1→…→OP7 顺序，⛔ 不补救不改判据）

| 检验 | 要求 | 不过则 |
|---|---|---|
| (OP1) 料齐 | 两臂各 `--k` 份；每份 `_CLIP_per`/`_mats` 长 148；材质数 67；两臂材质集合相同 | `VOID_NO_DATA` |
| (OP2) 新列自洽 | 每份 \|mean(`_CLIP_per`) − `CLIP`\| < 1e-4（`load_arm` 里逐份查） | `VOID_PER_IMAGE_MISMATCH` |
| (OP3) 参照行逐位相同 | 全部 2K 份的 `real_half.CLIP` 是**同一个值**（逐位） | `VOID_REF_ROW_DIFFERS` |
| (OP4) 控制臂前 `--n_old` 份复现 (M60) | 与 `experiments/m60_scale.json` 的 `trt_row_means` 逐个 \|差\| < 1e-3 | `VOID_CTRL_NOT_REPRODUCED` |
| (OP5) 假阳检验 | 控制臂**内部**按种子奇偶对半的同一检验 p >= 0.01 | `VOID_PAIRED_FALSEPOS` |
| (OP6) 控制臂同源 | 控制臂**旧 `--n_old` 份 vs 新增份**的同一检验 p >= 0.01 | `VOID_CTRL_INHOMOGENEOUS` |
| (OP7) 实验确实做了 | 两臂行均值最大 \|差\| > 0（⛔ 不是逐位相同的同一批文件） | `VOID_ARMS_IDENTICAL` |

⚑ (OP3) 白拿：`real_half` 行只由参照集前后对半决定，与模型、与种子都无关
（(M61) 在 3 个检查点、38 个种子上实测逐位相同）⇒ 它携 0 比特、不破盲，却能抓到
"两臂跑的不是同一个尺寸/同一个参照集"。
⚑⚑ **(OP6) 是本轮新增的一条**：控制臂 28 份里有 17 份是**直接复用 (M60) 的产物**、
11 份是本轮新跑的。若新跑的那 11 份指错了检查点（(M61) 踩过两次的"料/路径"坑的近亲），
(OP4) 查不出来（它只看旧的 17 份）⇒ 必须另有一条查"控制臂内部是不是同一个模型"。
⚑ (OP7) 是 (M61) 立的纪律：**先证明"实验确实做了"**。本轮主判据是双侧检验，
两臂若因 bug 变成同一批文件会判成 `DOSE2_NULL_32` ＝ 一个有后果的"没效应"结论。

用法：
    python analysis/arch/m62_read_dose2.py --dir remote_tmp/m62 \
           --ctrl_tag m62_ctrl --trt_tag m62_dbl --k 28 --n_old 17 \
           --m60json experiments/m60_scale.json --out experiments/m62_dose2.json
    python analysis/arch/m62_read_dose2.py --selftest
"""
import argparse
import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m60_read_scale as M60          # ⛔ 冻结判据本体，只复用、不改

N_IMG_EXPECT = M60.N_IMG_EXPECT       # 148
N_MAT_EXPECT = M60.N_MAT_EXPECT       # 67
ROW = M60.ROW                         # "TRD"
REF_ROW = "real_half"
N_PERM = M60.N_PERM                   # 20000
CEILING_M58 = M60.CEILING_M58         # 0.5127
SIGMA1_M60_CTRL = 0.8988119024335479  # (M60) 控制臂实测；只用来算预注册那个 MDE
GAIN_M60 = 0.12499542587990321        # (M60) 第一次翻倍的增益，只登记、⛔ 不参与判据
Z80_TWOSIDED = M60.Z80_TWOSIDED


def mde_of(sigma1, k, n_mat=N_MAT_EXPECT):
    return Z80_TWOSIDED * sigma1 * math.sqrt(2.0 / k) / math.sqrt(n_mat)


def ref_rows(d, tag, k):
    """取每份 JSON 的 `real_half.CLIP`（(OP3) 用）。缺文件/缺键记 None。"""
    out = {}
    for s in range(k):
        p = os.path.join(d, "%s_s%d.json" % (tag, s))
        if not os.path.exists(p):
            out[s] = None
            continue
        with open(p, encoding="utf-8") as f:   # Windows 默认 GBK 会崩
            j = json.load(f)
        out[s] = j.get(REF_ROW, {}).get("CLIP")
    return out


def op4_offenders(ctrl_rows, m60_rows, n_old, tol=1e-3):
    """(OP4)：控制臂前 n_old 个种子必须复现 (M60) 处理臂的行均值。None = 料不全。"""
    if not m60_rows or len(m60_rows) < n_old:
        return None
    off = []
    for s in range(n_old):
        want = m60_rows.get(s)
        got = ctrl_rows.get(s)
        if got is None or want is None or abs(got - want) >= tol:
            off.append([s, want, got])
    return off


def analyse(ctrl, trt, k, n_old, ctrl_rows=None, trt_rows=None,
            refs=None, op4_off=None, n_perm=N_PERM):
    res = {"k_requested": k, "n_old": n_old, "n_perm": n_perm}
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

    # ---- (OP2) 已在 load_arm 里逐份查过 ----
    ops["OP2"] = {"ok": True, "note": "checked per file at load time (tol 1e-4)"}

    # ---- (OP3) 参照行逐位相同 ----
    vals = [v for v in (refs or {}).values()]
    ok3 = bool(vals) and len(vals) == 2 * k and all(v is not None for v in vals) \
        and len(set(vals)) == 1
    ops["OP3"] = {"ok": ok3, "n_seen": len(vals),
                  "distinct": sorted(set(v for v in vals if v is not None))[:5]}
    if not ok3:
        res["ops"] = ops
        res["verdict"] = "VOID_REF_ROW_DIFFERS"
        return res

    # ---- (OP4) 控制臂旧份复现 (M60) ----
    ops["OP4"] = {"ok": op4_off is not None and len(op4_off) == 0, "off": op4_off,
                  "note": "None = (M60) rows unavailable"}
    if not ops["OP4"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_CTRL_NOT_REPRODUCED"
        return res

    # ---- (OP5) 控制臂内部奇偶对半假阳检验 ----
    ev = [s for s in sorted(ctrl) if s % 2 == 0]
    od = [s for s in sorted(ctrl) if s % 2 == 1]
    a, b = M60.arm_mean(ctrl, mats, ev), M60.arm_mean(ctrl, mats, od)
    p5, m5 = M60.perm_p([a[m] - b[m] for m in mats], n_perm=n_perm)
    ops["OP5"] = {"ok": p5 >= 0.01, "p": p5, "mean": m5, "even": ev, "odd": od}
    if not ops["OP5"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_PAIRED_FALSEPOS"
        return res

    # ---- (OP6) 控制臂「旧 vs 新」同源 ----
    old = [s for s in sorted(ctrl) if s < n_old]
    new = [s for s in sorted(ctrl) if s >= n_old]
    if not old or not new:
        ops["OP6"] = {"ok": False, "p": None, "old": old, "new": new,
                      "note": "empty split"}
        res["ops"] = ops
        res["verdict"] = "VOID_CTRL_INHOMOGENEOUS"
        return res
    a, b = M60.arm_mean(ctrl, mats, old), M60.arm_mean(ctrl, mats, new)
    p6, m6 = M60.perm_p([a[m] - b[m] for m in mats], n_perm=n_perm)
    ops["OP6"] = {"ok": p6 >= 0.01, "p": p6, "mean": m6,
                  "n_old_seeds": len(old), "n_new_seeds": len(new)}
    if not ops["OP6"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_CTRL_INHOMOGENEOUS"
        return res

    # ---- (OP7) 实验确实做了 ----
    maxdiff = None
    if ctrl_rows and trt_rows:
        common = sorted(set(ctrl_rows) & set(trt_rows))
        ds = [abs(ctrl_rows[s] - trt_rows[s]) for s in common
              if ctrl_rows[s] is not None and trt_rows[s] is not None]
        maxdiff = max(ds) if ds else None
    ops["OP7"] = {"ok": maxdiff is not None and maxdiff > 0.0,
                  "max_row_diff": maxdiff}
    res["ops"] = ops
    if not ops["OP7"]["ok"]:
        res["verdict"] = "VOID_ARMS_IDENTICAL"
        return res

    # ---- 主判据 ----
    mc, mt = M60.arm_mean(ctrl, mats), M60.arm_mean(trt, mats)
    d = [mt[m] - mc[m] for m in mats]
    p, obs = M60.perm_p(d, n_perm=n_perm)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    res["main"] = {"n_materials": len(mats), "mean_D": obs, "p_perm": p,
                   "pos": pos, "neg": neg, "p_sign": M60.sign_test_p(pos, neg)}
    if p < 0.05:
        res["verdict"] = "DOSE2_HELPS_32" if obs > 0 else "DOSE2_HURTS_32"
    else:
        res["verdict"] = "DOSE2_NULL_32"

    # ---- 只登记（⛔ 一个都不参与判决）----
    s1 = M60.sigma1_of(ctrl)
    mde = mde_of(s1, k) if s1 else None
    d_row = None
    if ctrl_rows and trt_rows:
        cs = [v for v in ctrl_rows.values() if v is not None]
        ts = [v for v in trt_rows.values() if v is not None]
        if cs and ts:
            d_row = sum(ts) / len(ts) - sum(cs) / len(cs)
    res["registered"] = {
        "sigma1_ctrl_measured": s1,
        "sigma1_m60_prereg": SIGMA1_M60_CTRL,
        "mde_magnitude": mde,
        "mde_prereg_k28": mde_of(SIGMA1_M60_CTRL, k),
        "mde_frac_of_ceiling": (mde / CEILING_M58) if mde else None,
        "mde_frac_of_m60_gain": (mde / GAIN_M60) if mde else None,
        "ceiling_m58": CEILING_M58,
        "gain_m60_first_doubling": GAIN_M60,
        "per_material_D_sd": M60.mean_sd(d)[1],
        "D_rowmean_weighting": d_row,
        "ctrl_half_split_p": ops["OP5"]["p"],
    }
    return res


def selftest():
    """构造数据核算式与全部 VOID 分支，⛔ 不碰真实读数。"""
    ok = 0
    mats = ["m%03d" % i for i in range(N_MAT_EXPECT)]
    rng = random.Random(7)
    base = {m: 30.0 + rng.random() for m in mats}
    K, N_OLD = 8, 5

    def mk(shift, sd=0.85, seed=0, ks=K):
        r = random.Random(seed)
        return {s: {m: base[m] + shift + r.gauss(0, sd) for m in mats} for s in range(ks)}

    def rows(by, off=0.0):
        return {s: sum(by[s].values()) / len(by[s]) + off for s in by}

    def refs_ok(ks=K):
        return {("c", s): 33.5 for s in range(ks)} | {("t", s): 33.5 for s in range(ks)}

    def run(c, t, **kw):
        kw.setdefault("ctrl_rows", rows(c))
        kw.setdefault("trt_rows", rows(t, 0.7))
        kw.setdefault("refs", refs_ok())
        kw.setdefault("op4_off", [])
        kw.setdefault("n_perm", 2000)
        return analyse(c, t, K, N_OLD, **kw)

    # 1) 零效应 -> NULL；全部 7 条操作检验过
    r = run(mk(0.0, seed=1), mk(0.0, seed=2))
    assert r["verdict"] == "DOSE2_NULL_32", r["verdict"]; ok += 1
    assert all(r["ops"][k]["ok"] for k in ("OP1", "OP2", "OP3", "OP4", "OP5", "OP6", "OP7")); ok += 1
    # 2) 大正效应 -> HELPS
    r = run(mk(0.0, seed=3), mk(+1.5, seed=4))
    assert r["verdict"] == "DOSE2_HELPS_32" and r["main"]["mean_D"] > 1.0, r["verdict"]; ok += 1
    # 3) 大负效应 -> HURTS
    r = run(mk(0.0, seed=5), mk(-1.5, seed=6))
    assert r["verdict"] == "DOSE2_HURTS_32" and r["main"]["mean_D"] < -1.0, r["verdict"]; ok += 1
    # 4) 缺种子 -> VOID_NO_DATA
    c = mk(0.0, seed=7); c.pop(K - 1)
    assert run(c, mk(0.0, seed=8))["verdict"] == "VOID_NO_DATA"; ok += 1
    # 5) 材质集合不同 -> VOID_NO_DATA
    t = mk(0.0, seed=9)
    for s in t:
        t[s].pop(mats[0])
    assert run(mk(0.0, seed=10), t)["verdict"] == "VOID_NO_DATA"; ok += 1
    # 6) 参照行不同 -> VOID_REF_ROW_DIFFERS
    bad = refs_ok(); bad[("t", 0)] = 33.6
    assert run(mk(0.0, seed=11), mk(0.0, seed=12), refs=bad)["verdict"] == "VOID_REF_ROW_DIFFERS"; ok += 1
    # 6b) 参照行份数不够（缺文件）-> 同样 VOID
    short = {k2: v for k2, v in list(refs_ok().items())[:-1]}
    assert run(mk(0.0, seed=11), mk(0.0, seed=12), refs=short)["verdict"] == "VOID_REF_ROW_DIFFERS"; ok += 1
    # 6c) 参照行有 None -> 同样 VOID
    noneref = refs_ok(); noneref[("c", 1)] = None
    assert run(mk(0.0, seed=11), mk(0.0, seed=12), refs=noneref)["verdict"] == "VOID_REF_ROW_DIFFERS"; ok += 1
    # 7) (M60) 行不可得 / 没复现 -> VOID_CTRL_NOT_REPRODUCED
    assert run(mk(0.0, seed=13), mk(0.0, seed=14), op4_off=None)["verdict"] == "VOID_CTRL_NOT_REPRODUCED"; ok += 1
    assert run(mk(0.0, seed=13), mk(0.0, seed=14),
               op4_off=[[0, 33.0, 31.0]])["verdict"] == "VOID_CTRL_NOT_REPRODUCED"; ok += 1
    # 7b) op4_offenders 本体
    assert op4_offenders({0: 1.0}, None, 5) is None; ok += 1
    assert op4_offenders({s: 33.0 for s in range(5)}, {s: 33.0 for s in range(5)}, 5) == []; ok += 1
    assert len(op4_offenders({s: 33.0 for s in range(5)},
                             {s: 33.0 + (s == 2) for s in range(5)}, 5)) == 1; ok += 1
    assert len(op4_offenders({s: 33.0 for s in range(5)}, {s: 33.0 for s in range(3)}, 5) or []) == 0 \
        or op4_offenders({s: 33.0 for s in range(5)}, {s: 33.0 for s in range(3)}, 5) is None; ok += 1
    # 8) 控制臂奇偶对半真有差 -> VOID_PAIRED_FALSEPOS
    c = mk(0.0, seed=15)
    for s in c:
        if s % 2 == 1:
            for m in c[s]:
                c[s][m] += 2.0
    assert run(c, mk(0.0, seed=16))["verdict"] == "VOID_PAIRED_FALSEPOS"; ok += 1
    # 9) 控制臂旧/新不同源 -> VOID_CTRL_INHOMOGENEOUS
    # ⚠ 这里 n_old 取 4（K=8）＝ 旧/新两半各含两个奇、两个偶 ⇒ 污染在奇偶对半里相消，
    #   (OP5) 不会先触发。⚑ 顺带记一条：**(OP5) 与 (OP6) 不独立** —— 若旧/新划分与奇偶
    #   不平衡（如 n_old=5），同一个污染会先被 (OP5) 抓到（实测：本断言初稿就撞上了）。
    #   本轮真实取值 K=28 / n_old=17 ⇒ 新的 11 份里 6 偶 5 奇，**不平衡** ⇒ 真出问题时
    #   (OP5) 与 (OP6) 都可能报，⛔ 两条都是 VOID、不影响判决。
    c = mk(0.0, seed=17)
    for s in c:
        if s >= 4:
            for m in c[s]:
                c[s][m] += 2.0
    t = mk(0.0, seed=18)
    r = analyse(c, t, K, 4, ctrl_rows=rows(c), trt_rows=rows(t, 0.7),
                refs=refs_ok(), op4_off=[], n_perm=2000)
    assert r["verdict"] == "VOID_CTRL_INHOMOGENEOUS", r["verdict"]; ok += 1
    assert r["ops"]["OP5"]["ok"], "平衡划分下奇偶对半不该被旧/新差异触发"; ok += 1
    # 9b) n_old 覆盖全部种子（新的一份都没有）-> 同样 VOID
    cc = mk(0.0, seed=19)
    r = analyse(cc, mk(0.0, seed=20), K, K, ctrl_rows=rows(cc), trt_rows=rows(mk(0.0, seed=20), 0.7),
                refs=refs_ok(), op4_off=[], n_perm=500)
    assert r["verdict"] == "VOID_CTRL_INHOMOGENEOUS", r["verdict"]; ok += 1
    # 10) 两臂逐位相同 -> VOID_ARMS_IDENTICAL
    same = mk(0.0, seed=21)
    r = analyse(same, same, K, N_OLD, ctrl_rows=rows(same), trt_rows=rows(same),
                refs=refs_ok(), op4_off=[], n_perm=500)
    assert r["verdict"] == "VOID_ARMS_IDENTICAL", r["verdict"]; ok += 1
    # 11) MDE：随 K 下降；K=28 与预注册值一致；与 (M60) 的 K=17 对得上
    m28 = mde_of(SIGMA1_M60_CTRL, 28)
    m17 = mde_of(SIGMA1_M60_CTRL, 17)
    assert abs(m17 - 0.10551784) < 1e-6, m17; ok += 1          # (M60) 实跑那个 MDE
    assert abs(m28 - 0.0822) < 5e-4, m28; ok += 1
    assert m28 < m17; ok += 1
    assert 0.65 < m28 / GAIN_M60 < 0.67, m28 / GAIN_M60; ok += 1
    # 12) 冻结核确实来自 (M60)
    assert (N_IMG_EXPECT, N_MAT_EXPECT, ROW, N_PERM) == (148, 67, "TRD", 20000); ok += 1
    p, o = M60.perm_p([0.5] * N_MAT_EXPECT, n_perm=2000)
    assert p < 0.01 and abs(o - 0.5) < 1e-12; ok += 1
    p, o = M60.perm_p([0.0] * N_MAT_EXPECT, n_perm=500)
    assert p > 0.9 and abs(o) < 1e-12; ok += 1
    # 13) 只登记项：换加权与 σ1 都算得出来
    c, t = mk(0.0, seed=22), mk(+0.3, seed=23)
    r = analyse(c, t, K, N_OLD, ctrl_rows=rows(c), trt_rows=rows(t, 0.0),
                refs=refs_ok(), op4_off=[], n_perm=2000)
    assert r["registered"]["sigma1_ctrl_measured"] > 0; ok += 1
    assert r["registered"]["D_rowmean_weighting"] is not None; ok += 1
    assert abs(r["registered"]["mde_prereg_k28"] - mde_of(SIGMA1_M60_CTRL, K)) < 1e-12; ok += 1
    print("selftest %d/%d OK" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="remote_tmp/m62")
    ap.add_argument("--ctrl_tag", default="m62_ctrl")
    ap.add_argument("--trt_tag", default="m62_dbl")
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--n_old", type=int, default=17)
    ap.add_argument("--m60json", default="experiments/m60_scale.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    ctrl, ctrl_rows, pc = M60.load_arm(a.dir, a.ctrl_tag, a.k)
    trt, trt_rows, pt = M60.load_arm(a.dir, a.trt_tag, a.k)

    refs = {}
    for tag, key in ((a.ctrl_tag, "c"), (a.trt_tag, "t")):
        for s, v in ref_rows(a.dir, tag, a.k).items():
            refs[(key, s)] = v

    # (OP4) 的料：(M60) 处理臂的行均值（已发表，`experiments/m60_scale.json`）
    m60_rows = {}
    if os.path.exists(a.m60json):
        with open(a.m60json, encoding="utf-8") as f:
            for s, v in (json.load(f).get("trt_row_means") or {}).items():
                m60_rows[int(s)] = v

    res = analyse(ctrl, trt, a.k, a.n_old, ctrl_rows=ctrl_rows, trt_rows=trt_rows,
                  refs=refs, op4_off=op4_offenders(ctrl_rows, m60_rows, a.n_old))
    res["load_problems"] = {"ctrl": pc, "trt": pt}
    res["ctrl_row_means"] = ctrl_rows
    res["trt_row_means"] = trt_rows
    res["m60_trt_rows"] = m60_rows

    # 落盘一律在打印之前（非 GBK 字形会崩在写 JSON 之前）
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M62) second doubling on paired CLIP@32px ==")
    for name in ("OP1", "OP2", "OP3", "OP4", "OP5", "OP6", "OP7"):
        o = res.get("ops", {}).get(name)
        print("  %s: %s" % (name, o))
    if "main" in res:
        m = res["main"]
        print("main: n=%d  mean_D=%+.4f  p_perm=%.4f  pos/neg=%d/%d  p_sign=%.4f"
              % (m["n_materials"], m["mean_D"], m["p_perm"], m["pos"], m["neg"], m["p_sign"]))
        r = res["registered"]
        print("registered: sigma1=%.4f  MDE=%.4f (= %.0f%% of ceiling, %.0f%% of M60 gain %.4f)"
              % (r["sigma1_ctrl_measured"], r["mde_magnitude"],
                 100 * r["mde_frac_of_ceiling"], 100 * r["mde_frac_of_m60_gain"], GAIN_M60))
        print("registered: D_rowmean=%+.4f" % r["D_rowmean_weighting"])
    print("VERDICT:", res["verdict"])
    if res["verdict"] == "DOSE2_NULL_32":
        print("  【禁】只许读成: 第二次翻倍的效应 < MDE; 不许读成 '训练规模无用'/'训够了'")


if __name__ == "__main__":
    main()
