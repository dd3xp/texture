r"""(M88) 给 (M85) 分叉计划里「每臂复制训练」那一格定价 —— 盲期写死，⛔ 不碰 (M84) 任何读数。

## 为什么这一格非定价不可

(M85) 第四节的分叉表写着：若判 `FROMSCRATCH_TOO_BLUNT`，「今后架构臂**只许**走嵌套续训
或每臂复制训练」。这两条出口对**场上唯一剩下的架构候选（尺寸臂 d=512）不是等价的**：

- 嵌套续训（(M60) 形状）对宽度**构造上不可用** —— (M62) 已判 `SIZE_ARM_CONFOUNDED`：
  改宽度没有检查点可续，`--init_from` 一用就把 32000 步血统混进来。
- ⇒ 若 `FROMSCRATCH_TOO_BLUNT` 触发，尺寸臂的**唯一合法形状就是每臂复制训练**，
  而账本登记的那个"~22 GPU 小时"是**按 R=1 的从零形状**估的 ⇒ **标价作废**。

本文件只做一件事：把 `MDE(R, K)` 与 GPU 小时写成**判决日可以直接代入的两条式子**，
并在盲期跑出参数化价目表。⚠⚠ **定价不授权开臂**（(M62) 同款声明）：(M67) 三层准入门
原样未满足、用户未豁免，三选一原样挂起。

## 盲态披露

写于处理臂读数 **0/28**、训练 ~36000/44000 步时。⛔ 本文件没有读过 `m84_w384b_s*.json`
任何一份，也没有 import 任何会装料的函数；`--mde_base/--f` 是**命令行标量**，
判决日才由已落盘的判决产物代入。⛔ 判决出来后本文件一个字不许改。

## 两条式子

设判决日实测的 `MDE_retrain`（K=28、n_mat=67、(M62) 实测口径）为 `MDE_28`，
把逐材质 D 的方差在 K=28 处劈成两项：

    Var(D) = V_read(K=28) + V_drift
    f^2    = V_drift / Var(D)            # 重训漂移占的方差份额

- `V_read` 是读数种子带来的，∝ 1/K（两臂各 K 份）；
- `V_drift` 是**同配置两次独立从零训练之间**的差，它对 K 不敏感（(M62) 劈半相关 r=+0.700
  量到的正是这种"可复现的逐材质结构"），但每臂各训 R 次再平均 ⇒ ∝ 1/R。

⇒                MDE(R, K) = MDE_28 * sqrt( (1-f^2) * 28/K  +  f^2 / R )

⚠⚠ 三条必须同引的保留：
1. **`f` 不是本臂的登记项** —— 判决日要另跑一次 (M62) 的冻结诊断 `m62_mde_audit.audit_round`
   （它返回 `irreducible_per_material_sd_c` 与 `sd_D_empirical`，f = c / sd_emp），
   而那是**判决落盘之后**才许跑的（(M62) 先例）。
2. **`V_drift ∝ 1/R` 未经检验** —— 它假设不同训练种子的漂移在材质上独立同分布。
   本项目只会有 R=1 的一个实测点（(M84) 自己）⇒ 这是**外推**，与 (M62) 的"对数线性外推"
   同一类保留。⛔ 不许当实测。
3. 真值恒零的臂上 `rms` 含均值（(M85) 第二节）⇒ `MDE_28` 里混进了 `mean_D` 的偏移；
   若 `SHAM_FIRES` 同时触发，本文件的 `f` 分解**不可解读**，必须停在那里。

## 价目（GPU 小时；⚠ 全是实测常数，出处写在代码里）

    train_h(R) = R * (steps * s_per_step_ctrl + steps * s_per_step_trt) / 3600
    read_h(R,K) = R * K * (min_per_read_ctrl + min_per_read_trt) / 60
    total = train_h + read_h

⚑ 读数可三卡并行 ⇒ 墙钟 ≈ 读数 GPU 小时 / 3；训练本项目一直是单卡串行。
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m60_read_scale import Z80_TWOSIDED  # noqa: E402  ⛔ 冻结件，只 import

K_BASE = 28                  # (M84)/(M75)/(M62) 三臂共用的读数种子数
N_MAT = 67                   # 32px 逐材质统计单位
TARGET = 0.1250              # (M60) 增益＝全项目唯一量到过的真效应＝(M84) 冻结门槛
STEPS = 44000                # (M75)/(M84) 两臂的从零单阶段步数

# —— 实测常数（出处＝docs/arch_progress.md，⛔ 不是外推）——
S_PER_STEP = {"d384": 0.296, "d512": 0.406}      # w512 13:50→18:48 UTC ＝ 4h58m ⇒ 0.406
MIN_PER_READ = {"d384": 9.8, "d512": 10.0}       # 三卡并行实测 ~9.8 / ~10 分钟每份
MEM_MIB_READ = {"d384": 17989, "d512": 22400}    # 只登记：能并行几条流的硬上限

ARMS = {
    "width":    ("d384", "d512"),   # 尺寸臂：控制 d=384 vs 处理 d=512
    "same384":  ("d384", "d384"),   # 同宽度的任何从零双臂（如剂量轴重做）
}


def mde_at(mde_base, f, R, K):
    """MDE(R,K) = MDE_28 * sqrt((1-f^2)*K_BASE/K + f^2/R)。f 为漂移方差份额的平方根。"""
    if not (0.0 <= f <= 1.0):
        raise ValueError("f must be in [0,1]")
    if R < 1 or K < 1:
        raise ValueError("R,K >= 1")
    return mde_base * math.sqrt((1.0 - f * f) * K_BASE / K + f * f / R)


def price(R, K, arm="width"):
    """一条 (R,K) 配置的 GPU 小时；返回 (训练, 读数, 合计)。"""
    c, t = ARMS[arm]
    train_h = R * STEPS * (S_PER_STEP[c] + S_PER_STEP[t]) / 3600.0
    read_h = R * K * (MIN_PER_READ[c] + MIN_PER_READ[t]) / 60.0
    return train_h, read_h, train_h + read_h


def cheapest(mde_base, f, target=TARGET, arm="width", r_max=8,
             k_grid=(8, 12, 16, 20, 24, 28, 40, 56)):
    """在 (R,K) 网格里找**满足 MDE<=target 的最便宜配置**；够不到返回 None。

    ⚠ K 只取 4 的倍数：(OP3) 要按种子奇偶对半、(M62) 诊断要四等分。
    """
    best = None
    for R in range(1, r_max + 1):
        for K in k_grid:
            if K % 4:
                continue
            m = mde_at(mde_base, f, R, K)
            if m > target:
                continue
            cost = price(R, K, arm)[2]
            if best is None or cost < best["gpu_hours"]:
                best = {"R": R, "K": K, "mde": m, "gpu_hours": cost,
                        "train_h": price(R, K, arm)[0], "read_h": price(R, K, arm)[1]}
    return best


def grid_table(arm="width", f_grid=(0.0, 0.3, 0.5, 0.7, 0.85, 1.0),
               ratio_grid=(1.0, 1.2, 1.5, 2.0, 3.0)):
    """参数化价目表：横轴 f，纵轴 `MDE_28 / TARGET` 的比值（⚠ 全是假设值，不是读数）。"""
    rows = []
    for ratio in ratio_grid:
        mde_base = TARGET * ratio
        for f in f_grid:
            best = cheapest(mde_base, f, arm=arm)
            rows.append({
                "mde_base_over_target": ratio, "f": f,
                "reachable": best is not None,
                "R": None if best is None else best["R"],
                "K": None if best is None else best["K"],
                "gpu_hours": None if best is None else round(best["gpu_hours"], 2),
            })
    return rows


def selftest():
    ok = 0

    def chk(cond, msg):
        nonlocal ok
        assert cond, msg
        ok += 1

    # —— 式子的四个极限 ——
    chk(abs(mde_at(1.0, 0.0, 1, K_BASE) - 1.0) < 1e-12, "R=1,K=28 must be identity")
    chk(abs(mde_at(1.0, 0.0, 9, K_BASE) - 1.0) < 1e-12, "f=0 => R 无关")
    chk(abs(mde_at(1.0, 1.0, 1, 9999) - 1.0) < 1e-12, "f=1 => K 无关")
    chk(abs(mde_at(1.0, 0.0, 1, 4 * K_BASE) - 0.5) < 1e-12, "f=0 => K 翻 4 倍买 2 倍")
    chk(abs(mde_at(1.0, 1.0, 4, K_BASE) - 0.5) < 1e-12, "f=1 => R 翻 4 倍买 2 倍")
    # 混合情形的逐位算术
    want = math.sqrt((1 - 0.36) * 28 / 14 + 0.36 / 2)
    chk(abs(mde_at(1.0, 0.6, 2, 14) - want) < 1e-12, "mixed case arithmetic")
    # —— 单调性 ——
    chk(mde_at(1.0, 0.5, 2, 28) < mde_at(1.0, 0.5, 1, 28), "R 单调降")
    chk(mde_at(1.0, 0.5, 1, 56) < mde_at(1.0, 0.5, 1, 28), "K 单调降")
    chk(mde_at(1.0, 0.9, 1, 56) > mde_at(1.0, 0.9, 2, 28), "漂移主导时 R 比 K 值钱")
    chk(mde_at(1.0, 0.2, 1, 56) < mde_at(1.0, 0.2, 2, 28), "读数主导时 K 比 R 值钱")
    # —— 参数校验（反向测试）——
    for bad in [(-0.1, 1, 28), (1.1, 1, 28)]:
        try:
            mde_at(1.0, *bad)
            chk(False, "f 越界必须抛")
        except ValueError:
            chk(True, "f 越界抛了")
    try:
        mde_at(1.0, 0.5, 0, 28)
        chk(False, "R=0 必须抛")
    except ValueError:
        chk(True, "R=0 抛了")
    # —— 价目的逐位算术 ——
    tr, rd, tot = price(1, 28, "width")
    chk(abs(tr - 44000 * (0.296 + 0.406) / 3600.0) < 1e-9, "train_h 算术")
    chk(abs(rd - 28 * (9.8 + 10.0) / 60.0) < 1e-9, "read_h 算术")
    chk(abs(tot - (tr + rd)) < 1e-12, "合计")
    chk(abs(tot - 17.82) < 0.02, "(M75) 那条臂的实测标价 ≈17.8 GPU 小时")
    chk(price(2, 28, "width")[2] > 2 * price(1, 14, "width")[2] * 0.9, "R 翻倍主要买在训练侧")
    chk(price(1, 28, "same384")[2] < price(1, 28, "width")[2], "同宽度更便宜")
    chk(abs(price(3, 16, "width")[2] - 3 * price(1, 16, "width")[2]) < 1e-9, "对 R 线性")
    # —— cheapest 的正确性：返回的配置必须真达标，且网格里没有更便宜的达标配置 ——
    for mb, f in [(0.20, 0.4), (0.18, 0.7), (0.30, 0.2), (0.1249, 0.99)]:
        best = cheapest(mb, f)
        if best is not None:
            chk(best["mde"] <= TARGET + 1e-12, "返回的配置真达标")
            brute = None
            for R in range(1, 9):
                for K in (8, 12, 16, 20, 24, 28, 40, 56):
                    if mde_at(mb, f, R, K) <= TARGET:
                        c = price(R, K, "width")[2]
                        brute = c if brute is None or c < brute else brute
            chk(abs(brute - best["gpu_hours"]) < 1e-9, "确实是网格最小")
    # 够不到的情形：f=1 且 MDE_28 > 8 倍 target ⇒ R<=8 无解
    chk(cheapest(TARGET * 9, 1.0) is None, "够不到必须返回 None")
    chk(cheapest(TARGET * 0.9, 0.0)["R"] == 1, "本来就达标 => R=1")
    chk(cheapest(TARGET * 0.9, 0.0)["K"] == 24, "f=0 时 K 不能降：28/K 把 MDE 拉回阀上")
    # 目标越严越贵（同一 f 下）
    a, b = cheapest(0.20, 0.5), cheapest(0.20, 0.5, target=0.09)
    chk(b is None or b["gpu_hours"] >= a["gpu_hours"], "目标越严越贵")
    # 网格表形状
    rows = grid_table()
    chk(len(rows) == 30, "价目表 5x6 格")
    chk(all(r["reachable"] for r in rows if r["mde_base_over_target"] == 1.0), "1.0 倍必可达")
    print("selftest %d/%d OK" % (ok, ok))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--mde_base", type=float, default=None,
                    help="判决日实测的 MDE_retrain（K=28）。⛔ 盲期别填。")
    ap.add_argument("--f", type=float, default=None,
                    help="漂移方差份额的平方根 = c / sd_emp（(M62) 冻结诊断给）。")
    ap.add_argument("--arm", default="width", choices=sorted(ARMS))
    ap.add_argument("--target", type=float, default=TARGET)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    out = {
        "constants": {"K_BASE": K_BASE, "N_MAT": N_MAT, "TARGET": TARGET,
                      "STEPS": STEPS, "Z80_TWOSIDED": Z80_TWOSIDED,
                      "s_per_step": S_PER_STEP, "min_per_read": MIN_PER_READ,
                      "mem_mib_read": MEM_MIB_READ},
        "baseline_price_R1_K28": {k: round(v, 3) for k, v in
                                  zip(("train_h", "read_h", "gpu_hours"), price(1, 28, a.arm))},
        "arm": a.arm,
        "grid_hypothetical": grid_table(a.arm),
    }
    if a.mde_base is not None and a.f is not None:
        best = cheapest(a.mde_base, a.f, target=a.target, arm=a.arm)
        out["measured"] = {"mde_base": a.mde_base, "f": a.f, "target": a.target,
                           "cheapest": best}
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
