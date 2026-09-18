#!/bin/bash
# (M61) 判据臂：16px 逐图 CLIP 读数。用法
#   bash /tmp/m61_arm.sh <GPU> <ctrl|more> <seed> [seed ...]
# 卡号由第一个参数带进来后 export 在本文件里（tmux 不继承 ssh 环境）；python 走绝对路径。
# 读数命令与 (M58)/(M59)/(M60) 逐字同一条，只换 --run / --size / --seed / --out。
set -e
GPU="$1"; ARM="$2"; shift 2
case "$ARM" in
  ctrl) RUN=runs/trd_v10 ;;
  more) RUN=/tmp/runs/trd_v10more_09180526 ;;
  *) echo "bad arm: $ARM"; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
for S in "$@"; do
  echo "=== $ARM seed $S start $(date -u +%H:%M:%S) gpu=$GPU ==="
  $PY -u eval/diag_decompose.py --run "$RUN" --size 16 --xmodal --reps 2 \
      --per_image --seed "$S" --out "/tmp/m61_${ARM}_s$S.json"
  echo "=== $ARM seed $S done $(date -u +%H:%M:%S) ==="
done
echo "M61_${ARM}_GPU${GPU}_DONE"
