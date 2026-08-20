#!/usr/bin/env bash
# SD-piXL baseline 探针：回答"给它纯棕色 + 'wood'，够不够？"
#
# 四个 run 并行占 GPU 4-7（0-3 被别人占满了）。设计上回答两个问题：
#   P1 vs P2 / P3 vs P4  —— 同一输入、同一 prompt，只改像素预算。
#                           若 16x16 结果就是 64x64 的糊版本而非重新设计的 motif，
#                           则 resolution-conditioning 这个立论成立。
#   P3 / P4              —— 纯色形状放白底上，看轮廓是否守得住。
#
# 用法: bash baselines/sdpixl/run_probe.sh
set -euo pipefail

SDPIXL="/mnt/data/kw/RoundSquisheen/pixel/SD-piXL"
PROJ="/mnt/data/kw/RoundSquisheen/texture"
ASSETS="$PROJ/baselines/sdpixl/assets"
CONFIG="$PROJ/baselines/sdpixl/configs/probe_wood.yaml"
LOGDIR="$PROJ/experiments/sdpixl_probe"
PY="/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python"

mkdir -p "$LOGDIR"

# 计算节点无外网；模型已在 ~/.cache/huggingface 里，强制离线以跳过 HEAD 校验
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# id  gpu  input           size  prompt
PROBES=(
  "p1_flat16   4  flat_brown.png   16   wood texture, wooden planks"
  "p2_flat64   5  flat_brown.png   64   wood texture, wooden planks"
  "p3_crate32  6  crate_brown.png  32   a wooden crate, wood grain"
  "p4_crate128 7  crate_brown.png  128  a wooden crate, wood grain"
)

cd "$SDPIXL"

for probe in "${PROBES[@]}"; do
    read -r id gpu img size prompt <<<"$probe"
    # prompt 是行尾剩余部分，read 已把它整段放进 $prompt

    echo "[launch] $id  gpu=$gpu  size=${size}x${size}  img=$img  prompt='$prompt'"

    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" main.py \
        -c "$CONFIG" \
        --input_image "$ASSETS/$img" \
        --prompt "$prompt" \
        --size "$size,$size" \
        > "$LOGDIR/$id.log" 2>&1 &

    echo "$!" > "$LOGDIR/$id.pid"
    sleep 5   # 错开启动，避免同时抢 HF 缓存
done

echo
echo "四个 run 已启动。跟踪:"
echo "  tail -f $LOGDIR/p1_flat16.log"
echo "  ls $SDPIXL/workdir/probe_wood/"
