#!/usr/bin/env bash
# (M58) 32px 的 CLIP 缺口拆解：同一条命令跑多个种子 = 跨种子经验零分布（(M50) 那条纪律）。
# 判据一个字不在本文件里定：全文在 docs/arch_progress.md 的 (M58) 预注册节，
# 判读器 analysis/arch/m58_read_clipdecomp.py（盲写、--selftest 13/13）。
#
# 起法（远程；卡号与 PATH 都写在本文件里，tmux 两个都不继承）：
#   tmux new-session -d -s m58a "bash /tmp/m58_decomp.sh 6 0 1 2"
#   tmux new-session -d -s m58b "bash /tmp/m58_decomp.sh 7 3 4"
# 第一个参数 = 卡号，其余 = 种子列表。
set -u
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

for S in "$@"; do
  # --bs 8 是 (M53) 逼出来的（32px 上 bs=32 吃满 44.5GB）；--floor_reps 用默认 1（本轮不读 KID）
  $PY -u eval/diag_decompose.py --run runs/trd_v10 --size 32 --xmodal --reps 2 --bs 8 \
      --seed "$S" --out /tmp/m58_clipdecomp_s"$S".json || { echo M58_ABORT_SEED_"$S"; exit 4; }
done

echo M58_DONE_GPU"$GPU"
