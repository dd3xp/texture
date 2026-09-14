"""零 API 复核所有已发表的判官数字：从逐对记录重算，和账本里写的值逐个对照。

`eval/judge_pairs.py` 把每一对的判定（'A'/'B'/'inconsistent'/null）落盘，所以胜率、二项 p、
Jeffreys 区间、弃样数**全都可以从净克隆重算**，一次 API 都不用。账本里的数字写死在下面的
EXPECT 里，对不上就退 1 —— 之后每一轮改账本里的判官数字，都必须让这个脚本继续通过。

    python analysis/arch/recheck_judge.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from exact import binom_test, jeffreys     # noqa: E402

# 文件名 -> (A 胜, 有效, 弃样, API 失败)；数字抄自 docs/arch_progress.md
EXPECT = {
    # 正式测试（E-mat 272 材质，预注册 56f1801，7 条臂，只跑一次）
    "judge_full_TRD16_vs_B2_16.json":                     (90, 190, 82, 0),
    "judge_full_TRD16_vs_B7_16.json":                    (131, 208, 64, 0),
    "judge_full_TRD16c_rr4_vs_B2_16.json":               (114, 199, 73, 0),
    "judge_full_C_TRD16_E_mat_vs_C_B2_E_mat_16.json":    (114, 189, 83, 0),
    "judge_full_C_TRD16_E_mat_vs_C_B7_E_mat_16.json":    (117, 193, 79, 0),
    "judge_full_TRD24_rr4_vs_B2_24.json":                 (94, 188, 84, 0),
    "judge_full_TRD32_rr4_vs_B2_32.json":                 (83, 202, 70, 0),
    # 验证集：CLIP-FD 汇率曲线上的四个操作点对 B7（预注册 e600f21）
    "judge_full_nf8_name_vs_B7val_c1.5_16_V_mat.json":     (47, 100, 25, 0),
    "judge_full_nf8_name_rr2_vs_B7val_c1.5_16_V_mat.json": (55,  96, 29, 0),
    "judge_full_nf8_name_rr4_vs_B7val_c1.5_16_V_mat.json": (59,  93, 32, 0),
    "judge_full_nf8_xpal_vs_B7val_c1.5_16_V_mat.json":     (56,  92, 33, 0),
    # 验证集：配对的那一比（预注册 919d8a7）
    "judge_full_nf8_name_rr4_vs_nf8_name_16_rr4_diff_V_mat.json": (49, 68, 24, 0),
    # 验证集 32px：温度与代次各自单独拆开（预注册 cebc506 / ae7f65e），全部同材质直接对判
    "judge_full_v10x32_t60_vs_v11dx_direct_32_V_mat.json":  (32, 74, 51, 0),
    "judge_full_v11dx_t60_vs_v11dx_direct_32_V_mat.json":   (21, 80, 45, 0),
    "judge_full_v10x_direct_vs_v11dx_direct_32_V_mat.json": (46, 75, 50, 0),
    # 验证集 32px：推理侧（预注册 20b188b）与训练侧（按包均衡 γ=0.5，预注册 a6b9742）
    "judge_full_v10x100r16_rr4_vs_v10x100_rr4_32_V_mat.json": (25, 78, 47, 0),
    "judge_full_pb05x100_rr4_vs_v10x100_rr4_32_V_mat.json":   (36, 73, 52, 0),
    # 测试集 16px 消融（预注册 83a6257，`eval/ablation_16.sh`）；参照臂 TRD16c_rr4 同材质直接对判
    "judge_full_TRD16c_rr4_vs_AB_pal_rr4_16.json":           (126, 190, 82, 0),
    "judge_full_TRD16c_rr4_vs_AB_ex_rr4_16.json":             (87, 175, 97, 0),
    "judge_full_TRD16c_rr4_vs_AB_ct_rr4_16.json":             (86, 154, 118, 0),
    # B3（SD-piXL）12 材质子集（预注册 cf9cd80，eval/b3_subset.sh）。"TRD16c" 是未重排的原始样本，不是 rr4
    "judge_full_TRD16_vs_B3_16_sdpixl_subset.json":          (11, 12, 0, 0),
    "judge_full_TRD16c_vs_B3_16_sdpixl_subset.json":         (10, 10, 2, 0),
    "judge_full_B2_vs_B3_16_sdpixl_subset.json":              (8,  9, 3, 0),
    "judge_full_B7_vs_B3_16_sdpixl_subset.json":             (10, 10, 2, 0),
    # 测试集：tier_anchor 的锚臂（预注册 8ec5446，eval/tier_anchor.sh）。
    # 绝对胜率只报不判（判据 (5)：B7 在 24/32 从未调过 CFG），进判据的是梯度。
    "judge_full_TRD24_rr4_vs_B7_24.json":                    (144, 208, 64, 0),
    "judge_full_TRD32_rr4_vs_B7_32.json":                    (116, 188, 84, 0),
    # 同一预注册的 B1 腿：B1@24 试点没过门 -> 判据 (1) 整对梯度作废，这条 full 只作水平记录、不入判据
    "judge_full_TRD32_rr4_vs_B1_32.json":                    (105, 186, 86, 0),
}

# 试点 -> (真题可解, n 真题, 空对照可解, n 空对照, 是否过门槛)。
# 消融表里 AB_xm 与 AB_rr 两条臂**没有 full**，判读全压在试点上，所以试点也要能零 API 复算。
PILOT_EXPECT = {
    "judge_pilot_TRD16c_rr4_vs_AB_pal_rr4_16.json": (12, 15, 2, 5, True),
    "judge_pilot_TRD16c_rr4_vs_AB_ex_rr4_16.json":  (10, 15, 3, 5, True),
    "judge_pilot_TRD16c_rr4_vs_AB_xm_rr4_16.json":   (9, 15, 1, 5, False),
    "judge_pilot_TRD16c_rr4_vs_AB_ct_rr4_16.json":  (11, 15, 3, 5, True),
    "judge_pilot_TRD16c_rr4_vs_TRD16c_16.json":      (6, 15, 1, 5, False),
    "judge_pilot_TRD16_vs_B3_16_sdpixl_subset.json":  (11, 12, 0, 5, True),
    "judge_pilot_TRD16c_vs_B3_16_sdpixl_subset.json": (10, 12, 3, 5, True),
    "judge_pilot_B2_vs_B3_16_sdpixl_subset.json":     (11, 12, 0, 5, True),
    "judge_pilot_B7_vs_B3_16_sdpixl_subset.json":     (10, 12, 0, 5, True),
    # tier_anchor（预注册 8ec5446）：B1@24 不过门 -> 按判据 (1) B1 那一对梯度作废，没有 full
    "judge_pilot_TRD24_rr4_vs_B7_24.json":            (11, 15, 1, 5, True),
    "judge_pilot_TRD32_rr4_vs_B7_32.json":            (12, 15, 2, 5, True),
    "judge_pilot_TRD24_rr4_vs_B1_24.json":             (9, 15, 3, 5, False),
    "judge_pilot_TRD32_rr4_vs_B1_32.json":            (10, 15, 0, 5, True),
    # nod32（预注册 7e96e2c）：不过门 -> 判据 (1) 不报胜负、不下结论，没有 full
    "judge_pilot_nod32x100_rr4_vs_ctrl0x100_rr4_32_V_mat.json": (9, 15, 1, 5, False),
    # b2_canvas（预注册 92fc3c7）：不过门 -> 判据 (1) 不报胜负，没有 full。(H_B2) 的"值多少"仍未量到
    "judge_pilot_B2_vs_B2up16_32.json":                (9, 15, 2, 5, False),
}


def main():
    bad = []
    for name, exp in EXPECT.items():
        p = ROOT / "experiments" / name
        if not p.exists():
            bad.append(f"{name}: 文件不在（净克隆里应当有，见 .gitignore 的豁免）")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        r = d["records"]
        got = (sum(x["verdict"] == "A" for x in r),
               sum(x["verdict"] in ("A", "B") for x in r),
               sum(x["verdict"] == "inconsistent" for x in r),
               sum(x["verdict"] is None for x in r))
        # 同时确认脚本当时写下的汇总字段与逐对记录自洽
        summary = (d["a_wins"], d["decided"], d["inconsistent"], d["api_fail"])
        w, t = got[0], got[1]
        lo, hi = jeffreys(w, t)
        mark = "OK "
        if got != exp:
            bad.append(f"{name}: 重算 {got} != 账本 {exp}")
            mark = "**对不上**"
        elif summary != got:
            bad.append(f"{name}: 文件里的汇总 {summary} 与逐对记录 {got} 不自洽")
            mark = "**不自洽**"
        print(f"{mark} {d['tag']:44s} {w}/{t} = {w / t:.0%}  p={binom_test(w, t):.3g}  "
              f"[{lo:.0%},{hi:.0%}]  弃 {got[2]}  API 失败 {got[3]}")
    for name, exp in PILOT_EXPECT.items():
        p = ROOT / "experiments" / name
        if not p.exists():
            bad.append(f"{name}: 文件不在（净克隆里应当有，见 .gitignore 的豁免）")
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        r = d["records"]
        real = [x for x in r if x["kind"] == "real"]
        null = [x for x in r if x["kind"] == "null"]
        got = (sum(x["resolved"] for x in real), len(real),
               sum(x["resolved"] for x in null), len(null),
               bool(d["pass"]))
        # 判官如果在试点里泄露过胜负，白名单就漏了，这里顺手钉死
        leak = sorted(set().union(*(set(x) for x in r)) - {"pair", "kind", "material", "answered", "resolved"})
        mark = "OK "
        if leak:
            bad.append(f"{name}: 试点记录里出现了不该有的字段 {leak}")
            mark = "**泄露**"
        elif got != exp:
            bad.append(f"{name}: 重算 {got} != 账本 {exp}")
            mark = "**对不上**"
        elif (got[0] / got[1] >= d["min_rate"] and got[0] / got[1] > got[2] / got[3]) != got[4]:
            bad.append(f"{name}: 门槛判读与 min_rate={d['min_rate']} 不自洽")
            mark = "**不自洽**"
        print(f"{mark} 试点 {d['tag']:38s} 真题 {got[0]}/{got[1]} = {got[0] / got[1]:.0%}  "
              f"空对照 {got[2]}/{got[3]} = {got[2] / got[3]:.0%}  门槛 {d['min_rate']:.0%}  "
              f"{'过' if got[4] else '不过'}")
    if bad:
        print("\n以下对不上：")
        for b in bad:
            print(" -", b)
        return 1
    print(f"\n{len(EXPECT)} 条臂 + {len(PILOT_EXPECT)} 条试点全部与账本一致（零 API 复算）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
