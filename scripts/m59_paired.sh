#!/usr/bin/env bash
# (M59) 与 (M58) **逐字同一条命令**，只多一个 --per_image（额外落逐图 CLIP，不改任何计算）。
# 于是同一批读数既能复现 (M58) 的行均值（= (OP3) 复现检验），又能做逐材质配对检验。
# 判据一个字不在本文件里定：全文在 docs/arch_progress.md 的 (M59) 预注册节，
# 判读器 analysis/arch/m59_read_pairedclip.py（盲写、--selftest 16/16）。
#
# 起法（远程；卡号与 PATH 都写在本文件里，tmux 两个都不继承）：
#   tmux new-session -d -s m59a "bash /tmp/m59_paired.sh 6 0 1 2"
#   tmux new-session -d -s m59b "bash /tmp/m59_paired.sh 7 3 4"
# 第一个参数 = 卡号，其余 = 种子列表。
set -u
GPU=$1; shift
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

for S in "$@"; do
  $PY -u eval/diag_decompose.py --run runs/trd_v10 --size 32 --xmodal --reps 2 --bs 8 \
      --per_image \
      --seed "$S" --out /tmp/m59_pairedclip_s"$S".json || { echo M59_ABORT_SEED_"$S"; exit 4; }
done

echo M59_DONE_GPU"$GPU"
