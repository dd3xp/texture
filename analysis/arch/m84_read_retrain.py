r"""(M84) 盲写判读器：重训复制臂（只换 --seed）在**配对 CLIP@32px** 上的判决。

⛔ 本文件在**处理臂训练尚未跑完、一份新读数都不存在**时写成（训练 22:14 UTC 起跑，
本文件写于 22:4x UTC，处理臂 0/28 份）；判决跑完一个字不许改。
判据、作废条件、操作检验、预测全部**先于本文件**写死在 `scripts/m84_w384b.sh` 头部
（commit 143970c，先于任何数据），本文件只做机械执行。

⛔ 判据算式一行不重写：`analysis/arch/m60_read_scale.py`（置换检验/符号检验/装料/σ1）与
`analysis/arch/m75_read_width.py`（`load_side_fields`/`read_run`/`cfg_diff_keys`/`rms`）
都是**冻结件**，本文件只 import。

## 本臂与 (M75) 的唯一差异

两臂 = `--seed 0`（(M75) 控制臂，已跑完）vs `--seed 1`（本臂），其余 52 键逐键相同
⇒ **构造零效应**（真值 D ≡ 0）⇒ 量的是「两臂各自从零训 44000 步」这种**实验形状**的噪声底。

    D_m = mean_{seed,rep}[ CLIP_{train-seed 1}(m) ] - mean_{seed,rep}[ CLIP_{train-seed 0}(m) ]
    MDE_retrain = 2.8016 * rms(D) / sqrt(67)          （(M62) 实测口径）

统计单位 = **材质（提示词），n = 67**（⛔ 不是 148 张）。

## 主判据（唯一下判的一条；⚠ 判的是 MDE，不是 p）

| 判决 | 条件 | 唯一许可读法 |
|---|---|---|
| **`FROMSCRATCH_TOO_BLUNT`** | `MDE_retrain >= 0.1250` | 「两臂各自从零训」这种形状连全项目最大的已知真效应（(M60) 的 +0.1250）都测不到 ⇒ 今后架构比较要么走**嵌套续训**（(M60) 形状），要么**每臂复制训练**。⛔ 不许读成"(M75) 的 null 不作数"、⛔ 不许读成"模型不稳定" |
| **`FROMSCRATCH_USABLE`** | `MDE_retrain <  0.1250` | 从零双臂的重训底噪不足以掩盖 (M60) 量级的效应。⛔ 不许读成"(M75) 的 null 因此更强"（那条臂自己的 MDE 0.2027 是实测值，本轮不改它） |

门槛 **0.1250** 冻结 ＝ (M60)「再训 12000 步」的增益 ＝ 全项目唯一量到过的真效应。

## 判据②（只登记 + 一条警告，⛔ 不参与主判决）

**(D1)** `p_perm < 0.05` ⇒ 附加标签 **`SHAM_FIRES`**：构造零效应臂上出现显著结果
⇒ 整条配对 CLIP 管线在真值为零处假阳 ⇒ 今后引 (M60)(M62)(M75) 必须同引此标签。
⛔ 反过来不成立：不触发**不**证明管线没问题（单次 5% 名义水平）。

## 作废条件（任一触发则整轮 VOID，⛔ 不许改判据救）

顺序与 (M75) 逐条相同，**只把 (V2) 的许可差异集从 `{out,d,heads}` 换成 `{out,seed}`**：

| 序 | 条件 | 判决 |
|---|---|---|
| 1 | (V3)/(OP1) 两臂读数份数 != K，或每份长度 != 148，或材质数 != 67，或两臂材质集合不同 | `VOID_NO_DATA` |
| 2 | (V4) 任一份读数出现 NaN | `VOID_NAN` |
| 3 | (V4)/(OP2) 任一份 `real_half.CLIP` != 34.1732177734375（逐位） | `VOID_REF_ROW` |
| 4 | (V1)/(OP4a) 任一臂 `log.json` 末步 != 44000 | `VOID_UNDERTRAINED` |
| 5 | (OP5) 两臂 `codebook_err` 不相同 | `VOID_DATA_MISMATCH` |
| 6 | (V2)/(OP4b) 两臂 `config.json` 除 {out, seed} 外有任何一个键不相同 | `VOID_CONFIG_DIFF` |
| 7 | (OP3) 控制臂**内部**按种子奇偶对半的同一检验 `p < 0.01`（假阳） | `VOID_PAIRED_FALSEPOS` |

⚠ (OP5) 故意排在 (V2) 之前（理由与 (M75) 同：`codebook_err` 本身就是 `config.json` 的键，
(V2) 会一并抓到，但"数据侧不同源"是更具体的诊断）。⛔ 这不是放宽：两条判决同为 VOID。
⚑ (OP5) 在本臂尤其硬：`kmeans` 的 `seed` 形参写死默认 0（`train_trd.py:51/69`）
⇒ codebook 与 `--seed` 无关 ⇒ 两臂 `codebook_err` 必须**逐位相同**。
⚑ 反向测试（selftest 写死）：`d`/`heads` 不同在本臂**必须**触发 `VOID_CONFIG_DIFF`
（(M75) 的白名单搬过来就会漏判）。

## 跑前三条预测（判决时逐条记分，⛔ 判决后不许改）

**(P1)** 判 `FROMSCRATCH_TOO_BLUNT`。 **(P2)** `SHAM_FIRES` 不触发。
**(P3)** 两臂 `log.json` val 末值之差 < 0.0623（(M57) 同配方换种子的 val 噪声下限 Dmax）。

## (OP1) 空跑的用法（料齐之前，⛔ 不传 --out、不落盘、不破盲）

    python analysis/arch/m84_read_retrain.py --dir remote_tmp --k 28 \
           --ctrl_tag m75_w384 --trt_tag m84_w384b --dry

⚠ (M62) 可迁移四：**两臂各验一次**（ctrl 过不蕴含 trt 过），且必须用**正式的 `--k`**。

判决当轮的完整命令（⚠ (M83) 教训：这条命令本身也要先逐字空跑一遍）：

    python analysis/arch/m84_read_retrain.py --selftest
    python analysis/arch/m84_read_retrain.py --dir remote_tmp --k 28 \
        --ctrl_tag m75_w384 --trt_tag m84_w384b \
        --ctrl_run remote_tmp/runs/trd_w384_09191345 \
        --trt_run  remote_tmp/runs/trd_w384b_09192215 \
        --out experiments/m84_retrain.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m60_read_scale import (  # noqa: E402  ⛔ 冻结判读器本体，只 import 不重写
    N_IMG_EXPECT, N_MAT_EXPECT, N_PERM, ROW, Z80_TWOSIDED,
    arm_mean, load_arm, mean_sd, perm_p, sigma1_of, sign_test_p,
)
from m75_read_width import (  # noqa: E402  ⛔ 同上
    REF_ROW_32, STEPS_EXPECT, cfg_diff_keys, load_side_fields, read_run, rms,
)

CFG_ALLOWED_DIFF_M84 = ("out", "seed")     # ⚠ 与 (M75) 的唯一差别
THR_MDE = 0.1250                           # (M60) 增益，门槛冻结
RMS_M75 = 0.5921                           # (M75) 宽度臂实测 rms(D)，只作分母
MDE_M75 = 0.2027                           # (M75) 实测 MDE，只登记
CEILING_M58 = 0.5127                       # 只作分母/量级
VAL_FLOOR_M57 = 0.0623                     # (M57) 同配方换种子的 val 噪声下限 Dmax


def last_val(path):
    """读一条臂 `log.json` 的 val 末值（(P3) 记分用）。缺文件返回 None。"""
    p = os.path.join(path, "log.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        log = json.load(f)
    if not isinstance(log, list):
        return None
    rows = [r for r in log if isinstance(r, dict) and r.get("step") is not None
            and r.get("val") is not None]
    if not rows:
        return None
    return max(rows, key=lambda r: r["step"])["val"]


def analyse(ctrl, trt, k, refs_ctrl, refs_trt, nans, cfg_c, cfg_t,
            last_c, last_t, val_c=None, val_t=None, n_perm=N_PERM):
    res = {"k_requested": k, "n_perm": n_perm}
    ops = {}

    # ---- 1. (V3)/(OP1) 料齐 ----
    mats_c = set.intersection(*[set(v) for v in ctrl.values()]) if ctrl else set()
    mats_t = set.intersection(*[set(v) for v in trt.values()]) if trt else set()
    ok1 = (len(ctrl) == k and len(trt) == k and len(ctrl) > 1
           and len(mats_c) == N_MAT_EXPECT and mats_c == mats_t)
    ops["OP1"] = {"ok": bool(ok1), "n_ctrl": len(ctrl), "n_trt": len(trt),
                  "n_mat_ctrl": len(mats_c), "n_mat_trt": len(mats_t),
                  "same_materials": mats_c == mats_t}
    if not ok1:
        res["ops"] = ops
        res["verdict"] = "VOID_NO_DATA"
        return res
    mats = sorted(mats_c)

    # ---- 2. (V4) NaN ----
    ops["V4_nan"] = {"ok": len(nans) == 0, "seeds_with_nan": nans}
    if nans:
        res["ops"] = ops
        res["verdict"] = "VOID_NAN"
        return res

    # ---- 3. (V4)/(OP2) 参照行逐位 ----
    bad_ref = sorted([[side, s, v] for side, dd in (("ctrl", refs_ctrl), ("trt", refs_trt))
                      for s, v in dd.items() if v != REF_ROW_32])
    ops["OP2"] = {"ok": len(bad_ref) == 0, "expect": REF_ROW_32, "offenders": bad_ref,
                  "n_checked": len(refs_ctrl) + len(refs_trt)}
    if bad_ref:
        res["ops"] = ops
        res["verdict"] = "VOID_REF_ROW"
        return res

    # ---- 4. (V1)/(OP4a) 末步 ----
    ops["OP4a_steps"] = {"ok": last_c == STEPS_EXPECT and last_t == STEPS_EXPECT,
                         "expect": STEPS_EXPECT, "ctrl": last_c, "trt": last_t}
    if not ops["OP4a_steps"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_UNDERTRAINED"
        return res

    # ---- 5. (OP5) codebook_err（⚠ 故意排在 (V2) 之前，理由见文件头）----
    ec = cfg_c.get("codebook_err") if cfg_c else None
    et = cfg_t.get("codebook_err") if cfg_t else None
    ops["OP5_codebook"] = {"ok": ec is not None and ec == et, "ctrl": ec, "trt": et}
    if not ops["OP5_codebook"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_DATA_MISMATCH"
        return res

    # ---- 6. (V2)/(OP4b) config 逐键：本臂只许 {out, seed} 不同 ----
    off = cfg_diff_keys(cfg_c, cfg_t, allowed=CFG_ALLOWED_DIFF_M84)
    seeds = [cfg_c.get("seed") if cfg_c else None, cfg_t.get("seed") if cfg_t else None]
    ops["OP4b_config"] = {"ok": off is not None and len(off) == 0,
                          "allowed_diff": list(CFG_ALLOWED_DIFF_M84),
                          "offenders": off, "seed_ctrl_trt": seeds,
                          "note": "None offenders = config.json 缺失"}
    if not ops["OP4b_config"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_CONFIG_DIFF"
        return res

    # ---- 7. (OP3) 控制臂内部假阳（奇偶对半）----
    ev = [s for s in sorted(ctrl) if s % 2 == 0]
    od = [s for s in sorted(ctrl) if s % 2 == 1]
    a, b = arm_mean(ctrl, mats, ev), arm_mean(ctrl, mats, od)
    d_sham = [a[m] - b[m] for m in mats]
    p_sham, mean_sham = perm_p(d_sham, n_perm=n_perm)
    ops["OP3_falsepos"] = {"ok": p_sham >= 0.01, "p": p_sham, "mean": mean_sham,
                           "even": ev, "odd": od}
    res["ops"] = ops
    if not ops["OP3_falsepos"]["ok"]:
        res["verdict"] = "VOID_PAIRED_FALSEPOS"
        return res

    # ---- 主判据：判的是 MDE，不是 p ----
    mc, mt = arm_mean(ctrl, mats), arm_mean(trt, mats)
    d = [mt[m] - mc[m] for m in mats]
    p, obs = perm_p(d, n_perm=n_perm)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    rms_d = rms(d)
    se = rms_d / math.sqrt(len(mats))
    mde = Z80_TWOSIDED * se
    res["main"] = {"n_materials": len(mats), "rms_D": rms_d,
                   "mde_retrain": mde, "threshold": THR_MDE,
                   "mean_D": obs, "p_perm": p, "pos": pos, "neg": neg,
                   "p_sign": sign_test_p(pos, neg)}
    res["verdict"] = "FROMSCRATCH_TOO_BLUNT" if mde >= THR_MDE else "FROMSCRATCH_USABLE"
    res["labels"] = ["SHAM_FIRES"] if p < 0.05 else []

    # ---- 只登记（跑前写死，⛔ 判决后不许增删）----
    s1 = sigma1_of(ctrl)
    mde_sigma1 = (Z80_TWOSIDED * s1 * math.sqrt(2.0 / k) / math.sqrt(len(mats))
                  if s1 else None)
    dval = (abs(val_t - val_c) if (val_c is not None and val_t is not None) else None)
    res["registered"] = {
        "rms_D_retrain": rms_d,
        "rms_ratio_vs_m75": rms_d / RMS_M75,
        "var_share_of_m75": (rms_d * rms_d) / (RMS_M75 * RMS_M75),
        "var_share_note": "加性分解假设未经检验 => 只登记、不下判",
        "mde_frac_of_ceiling": mde / CEILING_M58,
        "ceiling_m58": CEILING_M58,
        "mde_m75_width_arm": MDE_M75,
        "se_perm_null": se,
        "ci95_mean_D": [obs - 1.959964 * se, obs + 1.959964 * se],
        "per_material_D_sd": mean_sd(d)[1],
        "sigma1_ctrl_measured": s1,
        "mde_sigma1_formula_ONLY_FOR_AUDIT": mde_sigma1,
        "val_last_ctrl": val_c, "val_last_trt": val_t,
        "val_abs_diff": dval,
        "P3_val_diff_lt_floor": (None if dval is None else dval < VAL_FLOOR_M57),
        "val_floor_m57": VAL_FLOOR_M57,
    }
    return res


def selftest():
    """构造数据核算式与**全部** VOID 分支，⛔ 不碰任何真实读数。"""
    import random
    ok = 0
    mats = ["m%03d" % i for i in range(N_MAT_EXPECT)]
    rng = random.Random(11)
    base = {m: 30.0 + rng.random() for m in mats}
    K = 6

    def mk(shift, sd=0.85, seed=0):
        r = random.Random(seed)
        return {s: {m: base[m] + shift + r.gauss(0, sd) for m in mats} for s in range(K)}

    def refs(k=K, val=REF_ROW_32):
        return {s: val for s in range(k)}

    CFG = {"lr": 3e-4, "warmup": 1000, "p32": 0.5, "coarse": True, "p_coarse": 0.5,
           "steps": 44000, "batch": 256, "codebook_err": 7.26, "out": "/tmp/a",
           "d": 384, "heads": 6, "seed": 0}
    CFG2 = dict(CFG, out="/tmp/b", seed=1)

    def run(c, t, **kw):
        kw.setdefault("refs_ctrl", refs()); kw.setdefault("refs_trt", refs())
        kw.setdefault("nans", []); kw.setdefault("cfg_c", CFG); kw.setdefault("cfg_t", CFG2)
        kw.setdefault("last_c", STEPS_EXPECT); kw.setdefault("last_t", STEPS_EXPECT)
        kw.setdefault("n_perm", 2000)
        return analyse(c, t, K, **kw)

    # 1) 干净的一对 -> 七条操作检验全过
    r = run(mk(0.0, seed=1), mk(0.0, seed=2))
    for nm in ("OP1", "V4_nan", "OP2", "OP4a_steps", "OP4b_config",
               "OP5_codebook", "OP3_falsepos"):
        assert r["ops"][nm]["ok"], nm
    ok += 1
    assert r["verdict"] in ("FROMSCRATCH_TOO_BLUNT", "FROMSCRATCH_USABLE"); ok += 1
    # 1b) 判决只由 MDE 决定，与 p 无关：噪声大 -> TOO_BLUNT
    assert run(mk(0.0, sd=3.0, seed=31), mk(0.0, sd=3.0, seed=32))["verdict"] \
        == "FROMSCRATCH_TOO_BLUNT"; ok += 1
    # 1c) 噪声极小 -> USABLE（即便存在一个不大的真差）
    r = run(mk(0.0, sd=0.001, seed=33), mk(0.0, sd=0.001, seed=34))
    assert r["verdict"] == "FROMSCRATCH_USABLE", r["verdict"]; ok += 1
    assert r["main"]["mde_retrain"] < THR_MDE; ok += 1
    # 1d) ⚠ 冻结公式用的是 rms(D)（**含均值**，不是 sd）：噪声极小但偏移大 -> 仍 TOO_BLUNT
    #     这是 (M62) 实测口径的直接后果，⛔ 不许因此改公式（selftest 在见数据之前查出）
    r = run(mk(0.0, sd=0.001, seed=35), mk(+1.5, sd=0.001, seed=36))
    assert r["verdict"] == "FROMSCRATCH_TOO_BLUNT", r["verdict"]; ok += 1
    assert r["main"]["mean_D"] > 1.0 and abs(r["main"]["rms_D"] - 1.5) < 0.01; ok += 1
    # 1e) 门槛是 >=：恰好等于 0.1250 判 TOO_BLUNT
    need = THR_MDE * math.sqrt(N_MAT_EXPECT) / Z80_TWOSIDED       # 使 rms(D) 恰好达标
    assert abs(Z80_TWOSIDED * need / math.sqrt(N_MAT_EXPECT) - THR_MDE) < 1e-12; ok += 1
    # 2) (D1) 标签：构造零效应处的假阳 -> SHAM_FIRES
    r = run(mk(0.0, sd=0.001, seed=37), mk(+0.005, sd=0.001, seed=38))
    assert r["labels"] == ["SHAM_FIRES"], r["labels"]; ok += 1
    assert r["main"]["p_perm"] < 0.05; ok += 1
    r = run(mk(0.0, seed=39), mk(0.0, seed=40))
    assert r["labels"] == [] or r["main"]["p_perm"] < 0.05; ok += 1
    # 2b) 标签不参与主判决：显著 + 噪声极小 -> 仍判 USABLE 并挂 SHAM_FIRES
    r1 = run(mk(0.0, sd=0.001, seed=41), mk(+0.005, sd=0.001, seed=42))
    assert r1["verdict"] == "FROMSCRATCH_USABLE" and r1["labels"] == ["SHAM_FIRES"], \
        (r1["verdict"], r1["labels"]); ok += 1
    # 3) 缺种子 -> VOID_NO_DATA
    c = mk(0.0, seed=7); c.pop(K - 1)
    assert run(c, mk(0.0, seed=8))["verdict"] == "VOID_NO_DATA"; ok += 1
    # 4) 材质集合不同 -> VOID_NO_DATA
    t = mk(0.0, seed=9)
    for s in t:
        t[s].pop(mats[0])
    assert run(mk(0.0, seed=10), t)["verdict"] == "VOID_NO_DATA"; ok += 1
    # 5) NaN -> VOID_NAN（排在参照行之前）
    assert run(mk(0.0, seed=13), mk(0.0, seed=14), nans=[2],
               refs_trt=refs(val=1.0))["verdict"] == "VOID_NAN"; ok += 1
    # 6) 参照行 -> VOID_REF_ROW（两臂各测一次 + 逐位）
    assert run(mk(0.0, seed=15), mk(0.0, seed=16),
               refs_ctrl=refs(val=34.17321777))["verdict"] == "VOID_REF_ROW"; ok += 1
    assert run(mk(0.0, seed=15), mk(0.0, seed=16),
               refs_trt=refs(val=34.0))["verdict"] == "VOID_REF_ROW"; ok += 1
    assert run(mk(0.0, seed=15), mk(0.0, seed=16))["ops"]["OP2"]["ok"]; ok += 1
    assert REF_ROW_32 != 34.17321777 and REF_ROW_32 == 34.1732177734375; ok += 1
    # 7) 欠训 -> VOID_UNDERTRAINED（两臂各测一次 + 缺 log.json）
    assert run(mk(0.0, seed=17), mk(0.0, seed=18), last_c=43000)["verdict"] == "VOID_UNDERTRAINED"; ok += 1
    assert run(mk(0.0, seed=17), mk(0.0, seed=18), last_t=22000)["verdict"] == "VOID_UNDERTRAINED"; ok += 1
    assert run(mk(0.0, seed=17), mk(0.0, seed=18), last_t=None)["verdict"] == "VOID_UNDERTRAINED"; ok += 1
    # 8) codebook_err 不同 -> VOID_DATA_MISMATCH，且排在 (V2) 之前
    assert run(mk(0.0, seed=21), mk(0.0, seed=22),
               cfg_t=dict(CFG2, codebook_err=7.30))["verdict"] == "VOID_DATA_MISMATCH"; ok += 1
    assert run(mk(0.0, seed=21), mk(0.0, seed=22),
               cfg_t=dict(CFG2, codebook_err=7.30, p32=0.3))["verdict"] == "VOID_DATA_MISMATCH"; ok += 1
    assert run(mk(0.0, seed=21), mk(0.0, seed=22), cfg_t=None)["verdict"] == "VOID_DATA_MISMATCH"; ok += 1
    assert run(mk(0.0, seed=21), mk(0.0, seed=22), cfg_c=None)["verdict"] == "VOID_DATA_MISMATCH"; ok += 1
    # 9) config 逐键 -> VOID_CONFIG_DIFF
    r = run(mk(0.0, seed=19), mk(0.0, seed=20), cfg_t=dict(CFG2, p32=0.3))
    assert r["verdict"] == "VOID_CONFIG_DIFF", r["verdict"]; ok += 1
    assert r["ops"]["OP4b_config"]["offenders"] == [["p32", 0.5, 0.3]]; ok += 1
    assert run(mk(0.0, seed=19), mk(0.0, seed=20),
               cfg_t=dict(CFG2, newkey=1))["verdict"] == "VOID_CONFIG_DIFF"; ok += 1
    # 9b) ⚠ 反向测试：(M75) 的白名单在本臂必须**失效** —— d/heads 不同要被抓
    assert run(mk(0.0, seed=19), mk(0.0, seed=20),
               cfg_t=dict(CFG2, d=512))["verdict"] == "VOID_CONFIG_DIFF"; ok += 1
    assert run(mk(0.0, seed=19), mk(0.0, seed=20),
               cfg_t=dict(CFG2, heads=8))["verdict"] == "VOID_CONFIG_DIFF"; ok += 1
    # 9c) 而 seed / out 不同**不**触发（本臂唯一许可的两处）
    assert cfg_diff_keys(CFG, CFG2, allowed=CFG_ALLOWED_DIFF_M84) == []; ok += 1
    assert cfg_diff_keys(CFG, dict(CFG2, seed=99), allowed=CFG_ALLOWED_DIFF_M84) == []; ok += 1
    assert cfg_diff_keys(CFG, dict(CFG2, d=512), allowed=CFG_ALLOWED_DIFF_M84) != []; ok += 1
    # 10) (OP3) 假阳分支
    c = mk(0.0, seed=23)
    for s in c:
        if s % 2 == 1:
            c[s] = {m: v + 2.0 for m, v in c[s].items()}
    assert run(c, mk(0.0, seed=24))["verdict"] == "VOID_PAIRED_FALSEPOS"; ok += 1
    # 11) 判决顺序：多条同时成立时先触发者胜出
    assert run(c, mk(0.0, seed=29), last_c=1, cfg_t=None, nans=[0])["verdict"] == "VOID_NAN"; ok += 1
    assert run(c, mk(0.0, seed=29), last_c=1, cfg_t=None)["verdict"] == "VOID_UNDERTRAINED"; ok += 1
    # 12) MDE 口径 = (M62) 实测式，⛔ 不是 σ1 式
    assert abs(rms([0.2] * N_MAT_EXPECT) - 0.2) < 1e-12; ok += 1
    assert abs(rms([3.0, 4.0]) - math.sqrt(12.5)) < 1e-12; ok += 1
    assert abs(Z80_TWOSIDED * 0.2 / math.sqrt(67) - 2.801586 * 0.2 / math.sqrt(67)) < 1e-15; ok += 1
    # 12b) 对 (M75) 已发表的实测值对账：rms 0.5921 -> MDE 0.2027
    assert abs(Z80_TWOSIDED * RMS_M75 / math.sqrt(N_MAT_EXPECT) - MDE_M75) < 5e-4; ok += 1
    # 12c) 登记项：比值与方差份额自洽
    r = run(mk(0.0, seed=25), mk(0.0, seed=26))
    reg = r["registered"]
    assert abs(reg["rms_ratio_vs_m75"] - reg["rms_D_retrain"] / RMS_M75) < 1e-12; ok += 1
    assert abs(reg["var_share_of_m75"] - reg["rms_ratio_vs_m75"] ** 2) < 1e-12; ok += 1
    lo, hi = reg["ci95_mean_D"]
    assert lo < r["main"]["mean_D"] < hi; ok += 1
    assert reg["sigma1_ctrl_measured"] > 0 and reg["mde_sigma1_formula_ONLY_FOR_AUDIT"] > 0; ok += 1
    # 12d) σ1 版只登记：它与主判据用的 MDE 不是同一个数
    assert reg["mde_sigma1_formula_ONLY_FOR_AUDIT"] != r["main"]["mde_retrain"]; ok += 1
    # 13) (P3) 记分位
    r = run(mk(0.0, seed=27), mk(0.0, seed=28), val_c=6.952580193064057, val_t=6.99)
    assert r["registered"]["P3_val_diff_lt_floor"] is True; ok += 1
    r = run(mk(0.0, seed=27), mk(0.0, seed=28), val_c=6.95, val_t=7.30)
    assert r["registered"]["P3_val_diff_lt_floor"] is False; ok += 1
    assert run(mk(0.0, seed=27), mk(0.0, seed=28))["registered"]["P3_val_diff_lt_floor"] is None; ok += 1
    # 14) 冻结件行为
    p, o = perm_p([0.5] * N_MAT_EXPECT, n_perm=2000)
    assert p < 0.01 and abs(o - 0.5) < 1e-12; ok += 1
    p, o = perm_p([0.0] * N_MAT_EXPECT, n_perm=500)
    assert p > 0.9; ok += 1
    # 15) 常量与预注册逐位一致
    assert (N_IMG_EXPECT, N_MAT_EXPECT) == (148, 67); ok += 1
    assert STEPS_EXPECT == 44000 and CFG_ALLOWED_DIFF_M84 == ("out", "seed"); ok += 1
    assert THR_MDE == 0.1250 and VAL_FLOOR_M57 == 0.0623 and ROW == "TRD"; ok += 1
    assert N_PERM == 20000; ok += 1
    print("selftest %d/%d OK" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="remote_tmp")
    ap.add_argument("--ctrl_tag", default="m75_w384")
    ap.add_argument("--trt_tag", default="m84_w384b")
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--ctrl_run", default="")
    ap.add_argument("--trt_run", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true",
                    help="only run (OP1): print counts/materials, no main test, no write")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    ctrl, ctrl_rows, pc = load_arm(a.dir, a.ctrl_tag, a.k)
    trt, trt_rows, pt = load_arm(a.dir, a.trt_tag, a.k)

    if a.dry:
        # ⛔ 不读任何 CLIP 值、不落盘 ⇒ 不破盲
        for nm, by, prob in (("ctrl/" + a.ctrl_tag, ctrl, pc), ("trt/" + a.trt_tag, trt, pt)):
            ms = set.intersection(*[set(v) for v in by.values()]) if by else set()
            print("  (OP1) %-18s n=%2d/%d  n_mat=%d  problems=%d"
                  % (nm, len(by), a.k, len(ms), len(prob)))
            if prob:
                print("        problems:", prob[:6])
        mc = set.intersection(*[set(v) for v in ctrl.values()]) if ctrl else set()
        mt = set.intersection(*[set(v) for v in trt.values()]) if trt else set()
        print("  same_materials=%s   (dry run: no main test, nothing written)"
              % (mc == mt and bool(mc)))
        return

    refs_c, nan_c = load_side_fields(a.dir, a.ctrl_tag, a.k)
    refs_t, nan_t = load_side_fields(a.dir, a.trt_tag, a.k)
    cfg_c, last_c = read_run(a.ctrl_run) if a.ctrl_run else (None, None)
    cfg_t, last_t = read_run(a.trt_run) if a.trt_run else (None, None)
    val_c = last_val(a.ctrl_run) if a.ctrl_run else None
    val_t = last_val(a.trt_run) if a.trt_run else None

    res = analyse(ctrl, trt, a.k, refs_c, refs_t, sorted(set(nan_c) | set(nan_t)),
                  cfg_c, cfg_t, last_c, last_t, val_c=val_c, val_t=val_t)
    res["load_problems"] = {"ctrl": pc, "trt": pt}
    res["ctrl_row_means"] = ctrl_rows
    res["trt_row_means"] = trt_rows

    # 落盘一律在打印之前（非 GBK 字形会在打印时崩）
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M84) retrain replicate arm: train-seed 1 vs 0, paired CLIP@32px ==")
    for name in ("OP1", "V4_nan", "OP2", "OP4a_steps", "OP4b_config",
                 "OP5_codebook", "OP3_falsepos"):
        o = res.get("ops", {}).get(name)
        if o is not None:
            print("  %-13s: %s" % (name, o))
    if "main" in res:
        m, r = res["main"], res["registered"]
        print("main: n=%d  rms_D=%.4f  MDE_retrain=%.4f  (threshold %.4f)"
              % (m["n_materials"], m["rms_D"], m["mde_retrain"], m["threshold"]))
        print("      mean_D=%+.4f  p_perm=%.4f  pos/neg=%d/%d  p_sign=%.4f"
              % (m["mean_D"], m["p_perm"], m["pos"], m["neg"], m["p_sign"]))
        print("registered: rms/rms(M75 0.5921)=%.3f  var_share=%.3f  MDE=%.0f%% of ceiling %.4f"
              % (r["rms_ratio_vs_m75"], r["var_share_of_m75"],
                 100 * r["mde_frac_of_ceiling"], r["ceiling_m58"]))
        print("            CI95=[%+.4f,%+.4f]  val_last ctrl/trt=%s/%s  P3=%s"
              % (r["ci95_mean_D"][0], r["ci95_mean_D"][1],
                 r["val_last_ctrl"], r["val_last_trt"], r["P3_val_diff_lt_floor"]))
    print("VERDICT:", res["verdict"], "labels:", res.get("labels"))
    if res["verdict"] == "FROMSCRATCH_TOO_BLUNT":
        print("  【禁】只许读成: 从零双臂这种形状测不到 0.1250 量级的效应;"
              " 不许读成 '(M75) 的 null 不作数' / '模型不稳定'")
    if res["verdict"] == "FROMSCRATCH_USABLE":
        print("  【禁】不许读成 '(M75) 的 null 因此更强'(那条臂的 MDE 0.2027 是实测值)")
    if "SHAM_FIRES" in res.get("labels", []):
        print("  【禁】SHAM_FIRES 只登记; 不参与主判决; 反过来不成立")


if __name__ == "__main__":
    main()
