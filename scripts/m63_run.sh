#!/bin/bash
# (M63) GATE_CEILING_32 一次性读数。日志与产物都落 /tmp（/mnt/data 常年贴满）。
exec >> /tmp/m63.txt 2>&1
set -e
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton
cd /mnt/data/kw/RoundSquisheen/texture
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
echo "=== m63 start $(date -u) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"
$PY -u analysis/arch/m63_gate_ceiling.py --selftest
$PY -u analysis/arch/m63_gate_ceiling.py --out /tmp/m63_gate_ceiling.json
echo "=== m63 done $(date -u)"
