#!/usr/bin/env bash
# (M53) 32px 差距拆解（预注册 ac4da39，判据写在任何读数之前）。零判官、零 API。
# 卡号写在本文件里：tmux 不继承 ssh 环境，CUDA_VISIBLE_DEVICES 传不进去。
set -e
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=6
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
$PY -u eval/diag_decompose.py --run runs/trd_v10 --size 32 --xmodal \
  --reps 2 --seed 0 --floor_reps 9 --out /tmp/m53_decompose32.json >> /tmp/m53_decomp32.txt 2>&1
echo M53_DONE >> /tmp/m53_decomp32.txt
