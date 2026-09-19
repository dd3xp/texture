r"""(M75) 盲写判读器：尺寸臂（宽度 d 384→512）在**配对 CLIP@32px** 上的判决。

⛔ 本文件在**两臂训练尚未跑完、一份读数都不存在**时写成；判决跑完一个字不许改。
判据、作废条件、操作检验、预测全部**先于本文件**写死在 `scripts/m75_w384.sh` 头部
（commit 048116d，先于任何数据），本文件只做机械执行。

## 判据（唯一下判的一条）

    D_m = mean_{seed,rep}[ CLIP_{d=512}(m) ] − mean_{seed,rep}[ CLIP_{d=384}(m) ]

统计单位 = **材质（提示词），n = 67**（⛔ 不是 148 张；沿用 (M59)(M60)(M62) 同一口径：
74 个 V_mat@32 目标里 7 个提示词各出现两次、且 reps=2 ⇒ 148 张图不独立）。
只读每份 JSON 的 `TRD` 行（`_CLIP_per` + `_mats`），两臂各 K 个种子。
主检验 = 逐材质配对的符号翻转置换检验（双侧，20000 次，rng 固定 12345），
**import 冻结的 `analysis/arch/m60_read_scale.py` 本体**（⛔ 不重写判据算式）。

| 判决 | 条件 |
|---|---|
| **`WIDTH_HELPS_32`** | `mean_D > 0` 且 `p_perm < 0.05` |
| **`WIDTH_HURTS_32`** | `mean_D < 0` 且 `p_perm < 0.05`（只登记，⛔ 不许读成"该变窄"——本轮没有 d<384 的臂） |
| **`WIDTH_NULL_32`** | `p_perm >= 0.05` |

⚠⚠ **`WIDTH_NULL_32` 的唯一许可读法**＝「在这把尺子、这个预算、这个统计单位上，
d 384→512 的变化**小于本臂 MDE**」。⛔ 不许读成"容量轴关闭"、⛔ 不许读成"模型够大了"、
⛔ 不许读成"该去加数据"。

⚠ **MDE 一律按 (M62) 实测口径** `MDE = 2.8016 · rms(D) / sqrt(n_mat)`
（＝符号翻转置换检验零分布 sd 的精确值 × 双侧 80% 功效 z）。
⛔ **禁用已被 (M62) 判错的 σ1 公式** `σ1·sqrt(2/K)`（它假设 D 的全部离散都是生成噪声，
实测偏小 1.10–1.85 倍）；本判读器仍把 σ1 版**只登记**出来供对账，⛔ 它不参与任何判决。

⚠ 本臂标定为**探索臂**（(M67) 三层准入门未满足、用户未豁免，理由见 m75_w384.sh 头部）。

## 作废条件（任一触发则整轮 VOID，⛔ 不许改判据救）

按**固定顺序**检查，先触发者即判决：

| 序 | 条件 | 判决 |
|---|---|---|
| 1 | (V3)/(OP1) 两臂读数份数 != K，或每份长度 != 148，或材质数 != 67，或两臂材质集合不同 | `VOID_NO_DATA` |
| 2 | (V4) 任一份读数出现 NaN | `VOID_NAN` |
| 3 | (V4)/(OP2) 任一份 `real_half.CLIP` != 34.1732177734375（逐位） | `VOID_REF_ROW` |
| 4 | (V1)/(OP4a) 任一臂 `log.json` 末步 != 44000 | `VOID_UNDERTRAINED` |
| 5 | (OP5) 两臂 `codebook_err` 不相同 | `VOID_DATA_MISMATCH` |
| 6 | (V2)/(OP4b) 两臂 `config.json` 除 {out, d, heads} 外有任何一个键不相同 | `VOID_CONFIG_DIFF` |
| 7 | (OP3) 控制臂**内部**按种子奇偶对半的同一检验 `p < 0.01`（假阳） | `VOID_PAIRED_FALSEPOS` |

⚠ **(OP5) 故意排在 (V2) 之前**：`codebook_err` 本身就是 `config.json` 的一个键
（`train_trd.py:454`），(V2) 会把它当成"普通的配置键不同"⇒ 两条同时触发。
排序由 `--selftest` 在**任何数据存在之前**查出并写死，取更具体的诊断
（`VOID_DATA_MISMATCH` 指明"数据侧不同源"，比笼统的 `VOID_CONFIG_DIFF` 信息量大）。
⛔ 这不是放宽：两条都是作废，判决同为 VOID，只影响 VOID 的**标签**。

⚑ (V2) 是可机械执行的：`train_trd.py:454` 的 `config.json` ＝ `vars(args) | {codebook_err}`，
两条启动脚本除 `{out, d, heads}` 外逐字相同（已在 (M75) 开臂时 `md5sum` + `ps` 逐字复核）。
⚑ (OP5) 的道理：codebook 由数据构建、在 `seed_for_training` 之前 ⇒ 与宽度无关
⇒ 两臂 `codebook_err` 必须逐位相同，不同即说明数据侧不同源。
⚑ (OP3) 沿用 (M50)/(M60) 纪律：自己造经验零分布查假阳，⛔ 不当主判据的门。

## (OP1) 空跑的用法（料齐之前）

    python analysis/arch/m75_read_width.py --dir remote_tmp --k 28 \
           --ctrl_tag m75_w384 --trt_tag m75_w512 --dry

`--dry` 只跑到 (OP1) 就停、只打印份数/材质集合，**不进主检验、不落盘** ⇒ 不破盲。
⚠ (M62) 可迁移四：**两臂各验一次**——ctrl 与 trt 是两条不同管道，ctrl 过不蕴含 trt 过；
且必须用**正式的 `--k`**（小 `--k` 会真进主检验＝提前看到读数、自毁盲判）。

用法：
    python analysis/arch/m75_read_width.py --selftest
    python analysis/arch/m75_read_width.py --dir remote_tmp --k 28 \
        --ctrl_tag m75_w384 --trt_tag m75_w512 \
        --ctrl_run remote_tmp/trd_w384_09191345 --trt_run remote_tmp/trd_w512_09191345 \
        --out experiments/m75_width.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m60_read_scale import (  # noqa: E402  ⛔ 冻结判读器本体，只 import 不重写
    N_IMG_EXPECT, N_MAT_EXPECT, N_PERM, ROW, Z80_TWOSIDED,
    arm_mean, load_arm, mean_sd, perm_p, sign_test_p,
)

REF_ROW_32 = 34.1732177734375      # (M61)(OP3) 32px `real_half.CLIP`，与模型/种子无关
STEPS_EXPECT = 44000
CFG_ALLOWED_DIFF = ("out", "d", "heads")
CFG_FIVE_KEYS = ("lr", "warmup", "p32", "coarse", "p_coarse", "steps")   # (M62) 5 键清单
CEILING_M58 = 0.5127               # 只作分母/量级
GAIN_M60 = 0.1250                  # (M60) "再训 12000 步"的增益，(P2) 的比较对象


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def load_side_fields(d, tag, k):
    """只取 (V4)/(OP2) 需要的两样：`real_half.CLIP` 与「本份里有没有 NaN」。

    ⛔ 不碰 TRD 行的任何读数 ⇒ 空跑时调用它也不破盲。
    """
    refs, nans = {}, []
    for s in range(k):
        p = os.path.join(d, "%s_s%d.json" % (tag, s))
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:       # Windows 默认 GBK 会崩
            j = json.load(f)
        refs[s] = j.get("real_half", {}).get("CLIP")
        per = j.get(ROW, {}).get("_CLIP_per") or []
        if any(isinstance(v, float) and math.isnan(v) for v in per):
            nans.append(s)
        if isinstance(refs[s], float) and math.isnan(refs[s]):
            nans.append(s)
    return refs, sorted(set(nans))


def read_run(path):
    """读一条臂的 `config.json` 与 `log.json` 末步。缺文件返回 None（⇒ 对应 VOID）。"""
    cfg, last = None, None
    pc = os.path.join(path, "config.json")
    pl = os.path.join(path, "log.json")
    if os.path.exists(pc):
        with open(pc, encoding="utf-8") as f:
            cfg = json.load(f)
    if os.path.exists(pl):
        with open(pl, encoding="utf-8") as f:
            log = json.load(f)
        steps = [r.get("step") for r in log if isinstance(r, dict) and r.get("step") is not None] \
            if isinstance(log, list) else []
        last = max(steps) if steps else None
    return cfg, last


def cfg_diff_keys(a, b, allowed=CFG_ALLOWED_DIFF):
    """(V2)：返回「除 allowed 外取值不同（或只在一侧出现）」的键，排序后返回。"""
    if a is None or b is None:
        return None
    keys = set(a) | set(b)
    off = []
    for kk in sorted(keys):
        if kk in allowed:
            continue
        if a.get(kk, "\0missing") != b.get(kk, "\0missing"):
            off.append([kk, a.get(kk), b.get(kk)])
    return off


def analyse(ctrl, trt, k, refs_ctrl, refs_trt, nans, cfg_c, cfg_t,
            last_c, last_t, n_perm=N_PERM):
    res = {"k_requested": k, "n_perm": n_perm}
    ops = {}

    # ---- 1. (V3)/(OP1) 料齐（load_arm 已逐份查过 148 长度与 per_image 自洽）----
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
    bad_ref = sorted([[side, s, v] for side, d in (("ctrl", refs_ctrl), ("trt", refs_trt))
                      for s, v in d.items() if v != REF_ROW_32])
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

    # ---- 5. (OP5) codebook_err ----
    # ⚠ 故意排在 (V2) 之前：codebook_err 也是 config.json 的键，(V2) 会一并抓到，
    #   但 "数据侧不同源" 是更具体的诊断。排序由 selftest 在数据存在之前定死。
    ec = cfg_c.get("codebook_err") if cfg_c else None
    et = cfg_t.get("codebook_err") if cfg_t else None
    ops["OP5_codebook"] = {"ok": ec is not None and ec == et, "ctrl": ec, "trt": et}
    if not ops["OP5_codebook"]["ok"]:
        res["ops"] = ops
        res["verdict"] = "VOID_DATA_MISMATCH"
        return res

    # ---- 6. (V2)/(OP4b) config 逐键 ----
    off = cfg_diff_keys(cfg_c, cfg_t)
    five = {kk: [cfg_c.get(kk) if cfg_c else None, cfg_t.get(kk) if cfg_t else None]
            for kk in CFG_FIVE_KEYS}
    ops["OP4b_config"] = {"ok": off is not None and len(off) == 0,
                          "allowed_diff": list(CFG_ALLOWED_DIFF),
                          "offenders": off, "five_keys": five,
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

    # ---- 主判据 ----
    mc, mt = arm_mean(ctrl, mats), arm_mean(trt, mats)
    d = [mt[m] - mc[m] for m in mats]
    p, obs = perm_p(d, n_perm=n_perm)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    res["main"] = {"n_materials": len(mats), "mean_D": obs, "p_perm": p,
                   "pos": pos, "neg": neg, "p_sign": sign_test_p(pos, neg)}
    if p < 0.05:
        res["verdict"] = "WIDTH_HELPS_32" if obs > 0 else "WIDTH_HURTS_32"
    else:
        res["verdict"] = "WIDTH_NULL_32"

    # ---- 只登记 ----
    se = rms(d) / math.sqrt(len(mats))
    mde = Z80_TWOSIDED * se
    res["registered"] = {
        "mde_m62_measured": mde,                     # ⚑ 唯一许可引用的 MDE
        "mde_frac_of_ceiling": mde / CEILING_M58,
        "ceiling_m58": CEILING_M58,
        "se_perm_null": se,
        "ci95_mean_D": [obs - 1.959964 * se, obs + 1.959964 * se],
        "per_material_D_sd": mean_sd(d)[1],
        "abs_mean_D_lt_gain_m60": abs(obs) < GAIN_M60,   # (P2) 的记分
        "gain_m60": GAIN_M60,
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
           "d": 384, "heads": 6}
    CFG2 = dict(CFG, out="/tmp/b", d=512, heads=8)

    def run(c, t, **kw):
        kw.setdefault("refs_ctrl", refs()); kw.setdefault("refs_trt", refs())
        kw.setdefault("nans", []); kw.setdefault("cfg_c", CFG); kw.setdefault("cfg_t", CFG2)
        kw.setdefault("last_c", STEPS_EXPECT); kw.setdefault("last_t", STEPS_EXPECT)
        kw.setdefault("n_perm", 2000)
        return analyse(c, t, K, **kw)

    # 1) 零效应 -> NULL，且七条检验全过
    r = run(mk(0.0, seed=1), mk(0.0, seed=2))
    assert r["verdict"] == "WIDTH_NULL_32", r["verdict"]; ok += 1
    for nm in ("OP1", "V4_nan", "OP2", "OP4a_steps", "OP4b_config", "OP5_codebook", "OP3_falsepos"):
        assert r["ops"][nm]["ok"], nm
    ok += 1
    # 2) 大正效应 -> HELPS
    r = run(mk(0.0, seed=3), mk(+1.5, seed=4))
    assert r["verdict"] == "WIDTH_HELPS_32", r["verdict"]; ok += 1
    assert r["main"]["mean_D"] > 1.0; ok += 1
    # 3) 大负效应 -> HURTS
    r = run(mk(0.0, seed=5), mk(-1.5, seed=6))
    assert r["verdict"] == "WIDTH_HURTS_32", r["verdict"]; ok += 1
    assert r["main"]["mean_D"] < -1.0; ok += 1
    # 4) 缺种子 -> VOID_NO_DATA
    c = mk(0.0, seed=7); c.pop(K - 1)
    assert run(c, mk(0.0, seed=8))["verdict"] == "VOID_NO_DATA"; ok += 1
    # 5) 材质集合不同 -> VOID_NO_DATA
    t = mk(0.0, seed=9)
    for s in t:
        t[s].pop(mats[0])
    assert run(mk(0.0, seed=10), t)["verdict"] == "VOID_NO_DATA"; ok += 1
    # 6) NaN -> VOID_NAN（且排在参照行之前）
    r = run(mk(0.0, seed=13), mk(0.0, seed=14), nans=[2], refs_trt=refs(val=1.0))
    assert r["verdict"] == "VOID_NAN", r["verdict"]; ok += 1
    # 7) 参照行不对 -> VOID_REF_ROW（两臂各测一次）
    assert run(mk(0.0, seed=15), mk(0.0, seed=16),
               refs_ctrl=refs(val=34.17321777)) ["verdict"] == "VOID_REF_ROW"; ok += 1
    assert run(mk(0.0, seed=15), mk(0.0, seed=16),
               refs_trt=refs(val=34.0))["verdict"] == "VOID_REF_ROW"; ok += 1
    # 7b) 逐位相等才算过（浮点必须一模一样）
    assert run(mk(0.0, seed=15), mk(0.0, seed=16))["ops"]["OP2"]["ok"]; ok += 1
    assert REF_ROW_32 != 34.17321777; ok += 1
    # 8) 欠训 -> VOID_UNDERTRAINED（两臂各测一次 + 缺 log.json）
    assert run(mk(0.0, seed=17), mk(0.0, seed=18), last_c=43000)["verdict"] == "VOID_UNDERTRAINED"; ok += 1
    assert run(mk(0.0, seed=17), mk(0.0, seed=18), last_t=22000)["verdict"] == "VOID_UNDERTRAINED"; ok += 1
    assert run(mk(0.0, seed=17), mk(0.0, seed=18), last_t=None)["verdict"] == "VOID_UNDERTRAINED"; ok += 1
    # 9) config 多出一个键不同 -> VOID_CONFIG_DIFF
    bad = dict(CFG2, p32=0.3)
    r = run(mk(0.0, seed=19), mk(0.0, seed=20), cfg_t=bad)
    assert r["verdict"] == "VOID_CONFIG_DIFF", r["verdict"]; ok += 1
    assert r["ops"]["OP4b_config"]["offenders"] == [["p32", 0.5, 0.3]]; ok += 1
    # 9b) 只在一侧出现的键也要被抓
    assert run(mk(0.0, seed=19), mk(0.0, seed=20),
               cfg_t=dict(CFG2, newkey=1))["verdict"] == "VOID_CONFIG_DIFF"; ok += 1
    # 9c) 缺 config.json -> 同样 VOID（此时 (OP5) 先触发，因为 codebook_err 也没了）
    assert run(mk(0.0, seed=19), mk(0.0, seed=20), cfg_t=None)["verdict"] == "VOID_DATA_MISMATCH"; ok += 1
    assert run(mk(0.0, seed=19), mk(0.0, seed=20), cfg_c=None)["verdict"] == "VOID_DATA_MISMATCH"; ok += 1
    # 9d) 允许差的三个键不触发
    assert cfg_diff_keys(CFG, CFG2) == []; ok += 1
    assert cfg_diff_keys(CFG, dict(CFG2, d=768)) == []; ok += 1     # d 在白名单里
    # 10) codebook_err 不同 -> VOID_DATA_MISMATCH
    r = run(mk(0.0, seed=21), mk(0.0, seed=22), cfg_t=dict(CFG2, codebook_err=7.30))
    assert r["verdict"] == "VOID_DATA_MISMATCH", r["verdict"]; ok += 1
    # 10b) 排序写死：codebook_err 与另一个键同时不同 -> 取更具体的 DATA_MISMATCH
    r = run(mk(0.0, seed=21), mk(0.0, seed=22), cfg_t=dict(CFG2, codebook_err=7.30, p32=0.3))
    assert r["verdict"] == "VOID_DATA_MISMATCH", r["verdict"]; ok += 1
    # 10c) 只有非 codebook 键不同时才落到 CONFIG_DIFF
    r = run(mk(0.0, seed=21), mk(0.0, seed=22), cfg_t=dict(CFG2, p32=0.3))
    assert r["verdict"] == "VOID_CONFIG_DIFF", r["verdict"]; ok += 1
    # 11) 假阳分支：控制臂奇偶被人为劈开 -> VOID_PAIRED_FALSEPOS
    c = mk(0.0, seed=23)
    for s in c:
        if s % 2 == 1:
            c[s] = {m: v + 2.0 for m, v in c[s].items()}
    assert run(c, mk(0.0, seed=24))["verdict"] == "VOID_PAIRED_FALSEPOS"; ok += 1
    # 12) MDE 口径：(M62) 实测式，⛔ 不是 σ1 式
    d = [0.2] * N_MAT_EXPECT
    assert abs(rms(d) - 0.2) < 1e-12; ok += 1
    assert abs(rms([3.0, 4.0]) - math.sqrt(12.5)) < 1e-12; ok += 1
    mde = Z80_TWOSIDED * rms(d) / math.sqrt(N_MAT_EXPECT)
    assert abs(mde - 2.801586 * 0.2 / math.sqrt(67)) < 1e-12; ok += 1
    # 12b) 对 (M62) 已发表的实测值对账：K=28 那格 MDE = 0.1525
    assert abs(2.801586 * (0.1525 * math.sqrt(67) / 2.801586) / math.sqrt(67) - 0.1525) < 1e-9; ok += 1
    # 12c) MDE 恒为正、随 rms 单调
    r1 = run(mk(0.0, seed=25), mk(0.0, seed=26))
    assert r1["registered"]["mde_m62_measured"] > 0; ok += 1
    lo, hi = r1["registered"]["ci95_mean_D"]
    assert lo < r1["main"]["mean_D"] < hi; ok += 1
    # 12d) (P2) 记分位：|mean_D| 与 (M60) 增益比
    assert run(mk(0.0, seed=27), mk(+1.5, seed=28))["registered"]["abs_mean_D_lt_gain_m60"] is False; ok += 1
    assert r1["registered"]["abs_mean_D_lt_gain_m60"] is True; ok += 1
    # 13) 常数正差必然显著 / 全零差必然不显著（冻结件 perm_p 的行为）
    p, o = perm_p([0.5] * N_MAT_EXPECT, n_perm=2000)
    assert p < 0.01 and abs(o - 0.5) < 1e-12; ok += 1
    p, o = perm_p([0.0] * N_MAT_EXPECT, n_perm=500)
    assert p > 0.9; ok += 1
    # 14) 判决顺序：多个作废条件同时成立时，先触发的那个胜出
    r = run(c, mk(0.0, seed=29), last_c=1, cfg_t=None, nans=[0])
    assert r["verdict"] == "VOID_NAN", r["verdict"]; ok += 1
    r = run(c, mk(0.0, seed=29), last_c=1, cfg_t=None)
    assert r["verdict"] == "VOID_UNDERTRAINED", r["verdict"]; ok += 1
    # 15) 常量与预注册逐位一致
    assert (N_IMG_EXPECT, N_MAT_EXPECT) == (148, 67); ok += 1
    assert STEPS_EXPECT == 44000 and CFG_ALLOWED_DIFF == ("out", "d", "heads"); ok += 1
    print("selftest %d/%d OK" % (ok, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="remote_tmp")
    ap.add_argument("--ctrl_tag", default="m75_w384")
    ap.add_argument("--trt_tag", default="m75_w512")
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--ctrl_run", default="")
    ap.add_argument("--trt_run", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--dry", action="store_true",
                    help="只跑到 (OP1) 就停：只打印份数/材质集合，不进主检验、不落盘")
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
        print("  same_materials=%s   (dry run: 未进主检验、未落盘)" % (mc == mt and bool(mc)))
        return

    refs_c, nan_c = load_side_fields(a.dir, a.ctrl_tag, a.k)
    refs_t, nan_t = load_side_fields(a.dir, a.trt_tag, a.k)
    cfg_c, last_c = read_run(a.ctrl_run) if a.ctrl_run else (None, None)
    cfg_t, last_t = read_run(a.trt_run) if a.trt_run else (None, None)

    res = analyse(ctrl, trt, a.k, refs_c, refs_t, sorted(set(nan_c) | set(nan_t)),
                  cfg_c, cfg_t, last_c, last_t)
    res["load_problems"] = {"ctrl": pc, "trt": pt}
    res["ctrl_row_means"] = ctrl_rows
    res["trt_row_means"] = trt_rows

    # 落盘一律在打印之前（非 GBK 字形会在打印时崩）
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)

    print("== (M75) width arm d=512 vs d=384 on paired CLIP@32px ==")
    for name in ("OP1", "V4_nan", "OP2", "OP4a_steps", "OP4b_config",
                 "OP5_codebook", "OP3_falsepos"):
        o = res.get("ops", {}).get(name)
        if o is not None:
            print("  %-13s: %s" % (name, o))
    if "main" in res:
        m, r = res["main"], res["registered"]
        print("main: n=%d  mean_D=%+.4f  p_perm=%.4f  pos/neg=%d/%d  p_sign=%.4f"
              % (m["n_materials"], m["mean_D"], m["p_perm"], m["pos"], m["neg"], m["p_sign"]))
        print("registered: MDE(M62 measured)=%.4f = %.0f%% of ceiling %.4f   CI95=[%+.4f,%+.4f]"
              % (r["mde_m62_measured"], 100 * r["mde_frac_of_ceiling"],
                 r["ceiling_m58"], r["ci95_mean_D"][0], r["ci95_mean_D"][1]))
    print("VERDICT:", res["verdict"])
    if res["verdict"] == "WIDTH_NULL_32":
        print("  【禁】只许读成: 效应 < MDE; 不许读成 '容量轴关闭'/'模型够大了'/'该去加数据'")
    if res["verdict"] == "WIDTH_HURTS_32":
        print("  【禁】只登记; 不许读成 '该变窄'(本轮没有 d<384 的臂)")


if __name__ == "__main__":
    main()
