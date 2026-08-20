#!/usr/bin/env bash
# 固定输入、固定 prompt、固定调色板，只扫像素预算。
#
# 这是探针实验（run_probe.sh）的修正版，改掉了它的两个设计问题：
#   1. 探针里 flat 输入配 16/64、crate 输入配 32/128，分辨率和输入类型混淆了，
#      跨配对的比较（比如 64 vs 32）说明不了问题。这里四档全用同一张 crate 输入。
#   2. 探针用的 wood8 调色板没有中性色，白色背景被就近映射成最浅的棕，
#      纹理漫进背景时看不出来。这里换成 wood9bg，背景有专用色，越界即可判。
#
# 代价：因为调色板不同（9 色 vs 8 色），本轮的绝对数值不能直接和探针那轮对比。
#
# 用法: bash baselines/sdpixl/run_sweep.sh
set -euo pipefail

SDPIXL="/mnt/data/kw/RoundSquisheen/pixel/SD-piXL"
PROJ="/mnt/data/kw/RoundSquisheen/texture"
ASSETS="$PROJ/baselines/sdpixl/assets"
CONFIG_SRC="$PROJ/baselines/sdpixl/configs/probe_wood.yaml"
LOGDIR="$PROJ/experiments/sdpixl_sweep"
PY="/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python"

INPUT="crate_brown.png"
PROMPT="a wooden crate, wood grain"
PALETTE="$ASSETS/wood9bg.hex"

mkdir -p "$LOGDIR"

# SD-piXL 把结果目录建在 config 文件旁边，所以先把 config 拷进 experiments/
CONFIG="$LOGDIR/sweep_wood.yaml"
cp "$CONFIG_SRC" "$CONFIG"

# 计算节点无外网；模型已在 ~/.cache/huggingface 里，强制离线以跳过 HEAD 校验
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DIFFUSERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

# size:gpu
ARMS=("16:4" "32:5" "64:6" "128:7")

cd "$SDPIXL"

for arm in "${ARMS[@]}"; do
    size="${arm%%:*}"
    gpu="${arm##*:}"
    id="s${size}"

    echo "[launch] $id  gpu=$gpu  size=${size}x${size}  input=$INPUT  palette=$(basename "$PALETTE")"

    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" main.py \
        -c "$CONFIG" \
        --input_image "$ASSETS/$INPUT" \
        --prompt "$PROMPT" \
        --palette "$PALETTE" \
        --size "$size,$size" \
        > "$LOGDIR/$id.log" 2>&1 &

    echo "$!" > "$LOGDIR/$id.pid"
    sleep 5
done

echo
echo "四档已启动 (16/32/64/128)，同一输入同一 prompt。"
echo "  结果: $LOGDIR/sweep_wood/"
