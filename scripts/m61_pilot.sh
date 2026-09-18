#!/bin/bash
# (M61) 试点：控制臂 runs/trd_v10 在 16px 上出 5 份逐图 CLIP 读数（种子 100..104）。
# ⛔ 这 5 份只用来定 sigma1 / ceiling16 / delta / K，一个数都不进判据（判据臂种子是 0..K-1）。
# 卡号写在本文件里（tmux 不继承 ssh 环境）；python 走绝对路径（PATH 同样不继承）。
set -e
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
for S in 100 101 102 103 104; do
  echo "=== seed $S start $(date -u +%H:%M:%S) ==="
  $PY -u eval/diag_decompose.py --run runs/trd_v10 --size 16 --xmodal --reps 2 \
      --per_image --seed "$S" --out "/tmp/m61_pilot_s$S.json"
  echo "=== seed $S done $(date -u +%H:%M:%S) ==="
done
echo M61_PILOT_DONE
