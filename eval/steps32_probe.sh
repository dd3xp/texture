#!/usr/bin/env bash
# 32px 结构崩塌的第一个候选机制：**采样步数没有跟着 token 数一起放大**。
#
# 背景（`analysis/arch/scale_diag.py`，本轮）：同 67 个材质上，TRD 在 16px 与真人
# 打平（结构门 49% vs 55% p=0.31、各向异性 0.326 vs 0.287 p=0.29），在 32px 上
# 却塌到 门 28% vs 真人 64%（p=6e-7）、各向异性 0.130 vs 0.534（p=3e-7）。
# 预注册的"尺度漂移"假设**已被否决**：过门瓦片的主周期比值 16->32 是 2.00（正比），
# 真人 2.50、B2 1.75 —— 画得出结构的时候尺度是对的，问题是**大多数时候画不出结构**。
# 数据也不是原因：v11d 已经用 1877 张 32px 真人瓦片、p32=0.7 训过，照样塌。
#
# 机制假设：吸收态离散扩散每步把一批格子从掩码里放出来，**同一步放出的格子是条件独立抽的**。
# 16px 是 256 格 / 24 步 ≈ 11 格每步；32px 是 1024 格 / 24 步 ≈ 43 格每步。
# 一次独立抽 43 个格子抽不出砖缝这种长程一致的结构 → 输出趋于各向同性的糊。
# 若如此，**把步数按格子数成比例放大（24 -> 96）就该把结构找回来，且不用重训**。
#
# 判读规则（跑之前写死，量具 = analysis/arch/scale_diag.py 同一套门与口径）：
#  1. **一致性检查**：steps=24 这一臂是 v11dx_direct 的复跑（同 run/ckpt/cfg/pal_mode/种子），
#     它的 32px 结构门必须落在 v11dx_direct 的 28% ±10pp 内。不落在里面说明管线对不上，
#     本轮所有臂一律不读。
#  2. **主判据**：steps=96 的结构门比 steps=24 高 **>=10pp** 且两比例检验 p<0.05
#     → "采样步数是 32px 的一条活杠杆"。否则记为不成立，机制假设当轮关掉。
#  3. 次判据（不单独定论）：各向异性中位数的 MW 检验、以及 24/96/192 的剂量-反应是否单调。
#  4. ⚠ 各向异性**不是优化目标**（拿它当目标必然拍平/条纹化）。本轮只用它探测机制；
#     真要换配置，必须先过**同材质直接对判**的判官臂（下一轮），再谈测试集。
#
# 只写 /tmp：`/mnt/data` 已 100% 满（0 字节可写），本轮不往仓库目录落任何文件。
#   tmux new-session -d -s arch_s32 "bash eval/steps32_probe.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/tmp/steps32
mkdir -p $OUT

# 与 scripts/v11d_train_eval.sh:25 逐字相同，只加 --steps / --out
G="$P eval/gen_trd.py --run runs/trd_v11d --ckpt last.pt --set V_mat --bs 8 --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --size 32 --out $OUT"
for S in 24 96 192; do
  $G --steps $S --tag s$S || exit 1
done
echo STEPS32_DONE
