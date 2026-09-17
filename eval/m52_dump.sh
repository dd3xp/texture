#!/bin/bash
# (M52) 把 (M49)/(M50) seed0 那次拆解的各行瓦片按判官要的目录结构落盘，零 API。
# 卡号写死在文件里：tmux 不继承 ssh 环境（记忆里踩过两次 OOM）。
set -e
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_m52
cd /mnt/data/kw/RoundSquisheen/texture
/mnt/data/kw/anaconda3/envs/jzs_train/bin/python -u eval/diag_decompose.py \
  --run runs/trd_v8 --ckpt last.pt --cfg 1.5 --reps 2 --seed 0 \
  --xmodal --rerank \
  --dump /tmp/m52_dump \
  --out /tmp/m52_decompose_s0.json
echo "M52_DUMP_DONE"
