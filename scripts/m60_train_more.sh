#!/usr/bin/env bash
# (M60) 候选 G 第一档「更久」：runs/trd_v10 再训 12000 步（= 把 v10 自己的训练预算翻一倍）。
# 预注册见 docs/arch_progress.md 2026-09-18 13:30 那节第五节；配方逐键与 v10 相同，
# v10 的 13 个非默认键全部显式重传，v10 之后新增的参数一个都不传（留默认 = 旧行为）。
#
# ⚠ 卡号写在本文件里（tmux 不继承 ssh 环境，CUDA_VISIBLE_DEVICES 传不进去）。
# ⚠ PATH 也不继承 → python 必须写绝对路径。
# ⚠ 产物全写 /tmp（/mnt/data 长期贴满）；TRITON_CACHE_DIR 同理，否则 atexit 里才崩。
set -e
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

OUT=/tmp/runs/trd_v10more_09180526
mkdir -p "$OUT" /tmp/triton

$PY -u model/train_trd.py \
  --out "$OUT" \
  --init_from runs/trd_v10/last.pt \
  --steps 12000 \
  --batch 256 \
  --lr 1.5e-4 \
  --warmup 500 \
  --sizes 16 32 \
  --p32 0.5 \
  --batch32 64 \
  --bias_freqs 8 \
  --bias_hidden 128 \
  --level_emb \
  --pal_aug 0.3 \
  --pal_smooth 0.1 \
  --extra \
  --n_ex 4 \
  --extra_file train_extra_packs_only.json \
  --coarse \
  --p_coarse 0.5 \
  --save_at 4000 8000 12000

echo M60_TRAIN_DONE_GPU7
