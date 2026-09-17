#!/usr/bin/env bash
# (P1) 论文支撑第一轮：**判官对检索基线 B5** —— 全项目 44 条臂里从来没有一条对 B5。
# 预注册：本脚本先提交再跑，只跑一次。零 GPU（B5 落盘是 CPU 检索，图都已在盘）。
#
# ===== 为什么这是最该先补的一条 =====
# 16px 正式测试表（`experiments/final_E_mat_16.json`）：B5 的分布指标**全面最好**
#   B5  KID 2.69 / FID 43.0 / FD  45.3 / CLIP 32.95（最低）/ seam 0.998
#   TRD16       8.80 /      43.2 /      82.3 /      34.60 /      0.979
# 原因是构造性的：**B5 返回的就是真人画的原图**，所以分布距离必然最小。
# → 审稿人必问"你的检索基线 FID 更低，凭什么说你更好"。我们的答案有两半，而**两半都还没量过**：
#   (a) 它与材质名的匹配最差（CLIP 最低）；
#   (b) 它**做不到"配色跟随指定区域"**——只能原样搬一张已有瓦片。而那正是 `GOAL.md` 的任务。
# 文献侧的依据（`docs/paper_genre_survey.md` §七之三）：SD-πXL 用 bilinear 反例证明
# "没有一个指标直接衡量像素化质量"——同理，B5 在分布指标上赢**不构成**它更好。
#
# ===== 三条臂（全部 16px、E_mat 272 材质、每材质第 0 张）=====
#   A1  TRD16c_rr4 vs B5        主判据臂（= 正式测试 56f1801 的 TRD16c 配置，一个字未改）
#   A2  TRD16      vs B5        单样本配置（与已发表的 TRD16 那条腿同口径）
#   A3  C_TRD16_E_mat vs C_B5_E_mat   **区域颜色任务**（= GOAL.md 的任务，两边同一目标色）
#
# ===== 判据（跑前写死，⛔ 一个字不许放宽）=====
#  ① **免门（`--no_gate`）**：⚠ 不是"放宽"——去掉筛子不可能引入筛选偏倚，且本轮**不使用任何旧试点数字**
#     （依据 `557a50e`：65% 那道试点门实测误杀约 29%，b2_canvas 就是被它杀掉的健康臂）。
#     事后可解率检验由 full 自带的 `resolve_rate` / `p_vs_floor` 给（地板 21/118 = 17.8%）。
#  ② **胜负判据**：某臂算"TRD 胜 B5" = 二项 p < 0.05 且方向为胜；算"输" = p < 0.05 且方向为输；
#     其余写"未测到差异"，⛔ 不许写成"持平"。
#  ③ **可解率**：`p_vs_floor < 0.05` 且 `resolve_rate > 17.8%` 才允许解读胜负；否则该臂记
#     `UNRESOLVABLE`、不报胜负（⚠ 这是**事后**检验，不影响是否开跑）。
#  ④ **族级 CI 必报**：跑完把三条臂登进 `analysis/arch/recheck_judge.py` 的 EXPECT，
#     再跑 `analysis/arch/judge_cluster_sweep.py` 取去末词族级 CI；**二项 CI 只当下界**（(M17)/(M19) 纪律）。
#  ⑤ **操作检验（任一不过则该臂作废）**：(OP1) B5 落盘图可复现（`build_b5.py --verify` 逐像素零不一致）；
#     (OP2) 每臂可比对材质数 ≥ 250；(OP3) `api_fail / 可比对数 ≤ 0.05`；
#     (OP4) 同材质两张图逐像素相同的比例 ≤ 25/272（两边几乎同图时判官只能瞎答）。
#  ⑥ **跑前方向预测（照录，跑完无论对错都登账）**：A1 预测**胜**（分布指标输、判官赢，正是我们要的反差）；
#     A2 预测**未测到差异或小胜**（单样本配置对 B2 那条腿本来就是 47%）；A3 预测**大胜**（B5 无法跟随区域色）。
#     ⚠ 账在此：本项目跑前预测的战绩是**赢少输多**，这三条大概率至少错一条。
#  ⑦ **不动的东西**：`final_test.sh`、`judge_pairs.py`、`colour_task.py` 的判据与默认值、
#     `UNITS_PER_TILE`、32px 准入条件①②③（本轮全在 16px，不受②限制）、已下的任何判决。
#  ⑧ ⛔ **本轮不授权任何配置变更**，无论结果如何。
# 产物写 /tmp（`/mnt/data` 长期贴满），跑完 scp 回本机入库。凭据只从环境变量读。
#
# 起法：tmux new-session -d -s arch_b5j "VLM_BASE_URL=... VLM_API_KEY=... PY=<python> bash eval/b5_judge.sh"
set -u
P=${PY:-python}
OUT=${OUT:-/tmp/judge_b5}
mkdir -p "$OUT"

# ---- 料：B5 落盘（CPU，零 GPU），并当场做 (OP1) 可复现检验 ----
$P eval/build_b5.py --set E_mat --size 16 --n 4 || exit 1
$P eval/build_b5.py --set E_mat --size 16 --n 4 --verify || { echo "OP1 不过：B5 落盘不可复现"; exit 1; }
# ---- 料：区域颜色任务的 B5 目录（与其他方法同一目标色、同一 labshift 残差校正）----
$P eval/build_colour_dirs.py --set E_mat --methods B5 || exit 1

J() { $P eval/judge_pairs.py full "$@" --size 16 --no_gate --outdir "$OUT"; }
J --a TRD16c_rr4      --b B5
J --a TRD16           --b B5
J --a C_TRD16_E_mat   --b C_B5_E_mat
echo B5J_DONE
