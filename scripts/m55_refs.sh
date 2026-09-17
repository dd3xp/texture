#!/usr/bin/env bash
# (M55) 第一步：渲 SDXL 参考图并存 CLIP 图像嵌入（`docs/arch_plan.md` §1 条件 2，从未跑过）。
# 用法：bash scripts/m55_refs.sh <shard 0|1>
#
# ⚠ 规矩（记忆里踩过的坑，逐条对应）：
#   - 卡号写在本文件里（tmux 不继承 ssh 的 CUDA_VISIBLE_DEVICES，静默回落到默认卡 -> OOM）
#   - python 写绝对路径（tmux 不继承 PATH，否则 `command not found` + set -e 秒退、日志只剩一行）
#   - 产物写 /tmp（/mnt/data 97% 满）；目录名 m55_refs 已加进 sync_remote_tmp.sh 白名单
#   - --extra 必须与训练的 --extra_file 同一个文件，否则材质覆盖不全、训练死在 train_trd.py:383
set -e
SHARD=${1:?需要 shard 号}
case "$SHARD" in
  0) export CUDA_VISIBLE_DEVICES=6 ;;
  1) export CUDA_VISIBLE_DEVICES=7 ;;
  *) echo "shard 只能是 0 或 1"; exit 2 ;;
esac
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
OUT=/tmp/m55_refs
mkdir -p "$OUT"
$PY -u baselines/render_refs.py \
    --extra train_extra_packs_only.json \
    --k 2 --steps 28 --shard "$SHARD" --nshards 2 --out "$OUT"
touch "$OUT/shard$SHARD.done"
echo "M55_REFS_SHARD${SHARD}_DONE"
