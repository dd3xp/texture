#!/usr/bin/env bash
# 位置偏置补"格子单位"的局部性特征（--bias_cells 1 2 4）——唯一变量，控制组是 v11d。
#
# 依据（预注册 8e364ae，结果见 experiments/scale_prior.json）：TRD 里没有绝对位置，空间结构只能
# 由 `model/trd.py::ToroidalBias` 表达，而它只看**归一化**环面偏移 d/n，且 16px 与 32px **共用一张表**
# —— 等于写死了"结构随画布缩放"。真人训练瓦片不同意：相关长度 L 在两档**按格子对齐**
# （L32/L16 = 1.011，95% CI [0.960, 1.177]，判据 √2；同材质名子集 0.986）。
# 训练包里 16px 瓦片比 32px 多 6.8 倍（2732 vs 403）→ 冲突由 16px 的先验赢 → 32px 拿到错误的空间频率，
# 这正是已记录的"TRD 在 32px 上满屏单像素噪点、没有大尺度结构"。
# 补的特征是 exp(-|d_cells|/s)，s = 1/2/4 格：任何**环绕后格子偏移**的函数在 n 环面上自动 n 周期
# （谐波若按固定格子周期写，在 n=24 上就不是 24 周期了，会破坏可平铺性——所以没用谐波），
# 已自检 16/24/32 三档循环平移不变。旧检查点的权重照抄、新列置零 → 第 0 步偏置函数与 v10 逐元素相同。
#
# **控制组 = v11d**：配方逐字相同（同 init、同步数、同数据、同超参），唯一差别是这三维特征。
#   v11d 已发表（docs/arch_progress.md 2026-09-12）：16px KID 6.2 / FID 64.0 / CLIP 34.77；
#   32px N=100+4选1 KID 24.8 / FID 109.1 / FD 426 / CLIP 35.05；32px 直接生成判官 vs B2 = 24/91 = 26%。
#
# 判据（写死，跑前提交）：
#   ① **16px 不许退步**：KID 不高于 v11d 的 6.2 + 1.0，否则这条改动作废、不再往下评。
#   ② 32px 的看点是**指标同向改善**：KID / FID / FD 三项里至少两项优于 v11d 同配置。
#   ③ ①②都过才授权下一轮做判官对判（本脚本**不问判官**，避免用胜率挑配置）。
#   ④ 不改 `final_test.sh`、不改任何默认值；`--bias_cells` 默认空 = 旧行为一个字不变。
#
# 起法（凭据只从环境变量给，不落盘）：
#   tmux new-session -d -s arch_cell "bash scripts/trd_cell_train_eval.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
RUN=/tmp/runs/trd_cell_09150215          # /mnt/data 满 → 检查点写 /tmp（scripts/sync_remote_tmp.sh 会拉回）
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-6} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_cell

$P -u model/train_trd.py --out $RUN --steps 12000 --lr 1.5e-4 --warmup 500 \
  --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
  --extra --extra_file train_extra_packs_only.json+train_64to32.json \
  --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
  --bias_cells 1 2 4 \
  --sizes 16 32 --p32 0.7 --save_at 6000 12000 > /tmp/trd_cell.txt 2>&1 || exit 1

# 生成口径逐字照搬控制组：16px 与 *_direct 用 --n 2（v11d 脚本的口径），
# N=100 + 4 选 1 用 --n 4（v11dx100_rr4 的口径）——比谁都不能换尺子。
G="$P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --bs 8 --cfg 1.5 --pal_mode retrieve --xmodal"
$G --size 16 --n 2 --tag cellx || exit 1               # 判据①：16px 是目前领先 B2 的那一档
for S in 32 24; do
  $G --size $S --n 2 --tag cellx_direct || exit 1
  $G --size $S --n 4 --ret_nname 100 --tag cellx100 || exit 1
  $P eval/rerank.py --src cellx100 --set V_mat --size $S --n 4 || exit 1
done
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v10x v11dx cellx --out $RUN/eval_cell_Vmat_16.json
for S in 32 24; do
  $P eval/run_eval.py --set V_mat --size $S \
     --methods B1val B2val v11dx_direct v11dx100_rr4 cellx_direct cellx100_rr4 --out $RUN/eval_cell_Vmat_$S.json
done
echo CELL_DONE
