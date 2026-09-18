#!/usr/bin/env bash
# (M60) 读数：与 (M59) `scripts/m59_paired.sh` **逐字同一条 diag_decompose 命令**，
# 只有 `--run` 与 `--out` 两处按臂变化（这正是预注册第六节允许的两处）。
# 控制臂 = runs/trd_v10（(M59) 那条 TRD 行）；处理臂 = 同配方再训 12000 步的新检查点。
# 判据一个字不在本文件里定：全文在 docs/arch_progress.md 的 (M60) 预注册节，
# 判读器 analysis/arch/m60_read_scale.py（盲写、--selftest 22/22）。
#
# 起法（远程；卡号与 PATH 都写在本文件里，tmux 两个都不继承）：
#   tmux new-session -d -s m60c "bash /tmp/m60_eval.sh 2 ctrl 5 6 7 8 9 10 11 12 13 14 15 16"
#   tmux new-session -d -s m60m "bash /tmp/m60_eval.sh 7 more 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
# 第一个参数 = 卡号，第二个 = 臂（ctrl|more），其余 = 种子列表。
# ⚠ 控制臂 seed 0..4 不在这里跑：直接拷 experiments/m59_pairedclip_s{0..4}.json（预注册第五节）。
set -u
GPU=$1; shift
ARM=$1; shift
case "$ARM" in
  ctrl) RUN=runs/trd_v10 ;;
  more) RUN=/tmp/runs/trd_v10more_09180526 ;;
  *) echo "M60_BAD_ARM_$ARM"; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

for S in "$@"; do
  $PY -u eval/diag_decompose.py --run "$RUN" --size 32 --xmodal --reps 2 --bs 8 \
      --per_image \
      --seed "$S" --out /tmp/m60_"$ARM"_s"$S".json || { echo M60_ABORT_"$ARM"_SEED_"$S"; exit 4; }
done

echo M60_DONE_"$ARM"_GPU"$GPU"
