#!/usr/bin/env bash
# (M62) 读数臂：32px 逐图 CLIP。用法
#   bash /tmp/m62_arm.sh <GPU> <ctrl|dbl> <seed> [seed ...]
# 读数命令与 (M58)/(M59)/(M60)/(M61) 逐字同一条，只换 --run / --seed / --out。
# 判据一个字不在本文件里定：全文在 docs/arch_progress.md 的 (M62) 预注册节，
# 判读器 analysis/arch/m62_read_dose2.py（盲写、--selftest 30/30）。
#
# ⚠ 卡号由第一个参数带进来后 export 在本文件里（tmux 不继承 ssh 环境）；python 走绝对路径。
# ⚠ 控制臂 seed 0..16 不在这里跑：直接复用 (M60) 的 m60_more_s{0..16}.json（预注册第四节）。
# ⚠ (M61) 教训：脚本自己声明该在哪台机器跑。
set -e
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M62_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
GPU="$1"; ARM="$2"; shift 2
case "$ARM" in
  ctrl) RUN=/tmp/runs/trd_v10more_09180526 ;;   # = (M60) 的处理臂 = v10 + 12000 步
  dbl)  RUN=/tmp/runs/trd_v10dbl_09181524 ;;    # = 本轮新训     = v10 + 24000 步
  *) echo "M62_BAD_ARM_$ARM"; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

for S in "$@"; do
  echo "=== $ARM seed $S start $(date -u +%H:%M:%S) gpu=$GPU ==="
  $PY -u eval/diag_decompose.py --run "$RUN" --size 32 --xmodal --reps 2 --bs 8 \
      --per_image \
      --seed "$S" --out /tmp/m62_"$ARM"_s"$S".json
  echo "=== $ARM seed $S done $(date -u +%H:%M:%S) ==="
done
echo "M62_${ARM}_GPU${GPU}_DONE"
