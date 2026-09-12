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
    if bad:
        print("\n以下对不上：")
        for b in bad:
            print(" -", b)
        return 1
    print(f"\n{len(EXPECT)} 条臂全部与账本一致（零 API 复算）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
