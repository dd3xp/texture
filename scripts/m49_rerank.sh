#!/bin/bash
# (M49) 调色板重排器的可达读数：同一个 5 张候选集，换三种「看不到目标」的挑法。
# 卡号必须写在本文件里（tmux 不继承 ssh 环境）；python 必须绝对路径（PATH 也不继承）。
set -x
cd /mnt/data/kw/RoundSquisheen/texture
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_m49
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
$PY -u eval/diag_decompose.py --run runs/trd_v8 --ckpt last.pt --cfg 1.5 \
  --reps 2 --seed 0 --xmodal --rerank \
  --out /tmp/m49_rerank_v8_Vmat.json
echo "EXIT=$?"
