#!/bin/bash
# (M75) 尺寸臂 —— 处理臂：d=512 / heads=8（head_dim 仍为 64），从零单阶段 44000 步
# 判据、作废条件、操作检验、跑前预测**全部写死在 scripts/m75_w384.sh 头部**，本文件不重复。
# 与控制臂的唯一差异：--d 384→512、--heads 6→8（耦合改动，已在控制臂头部披露）。
# (M62) 实测：混训 batch 256 的 ~22GB 里权重+Adam 只 0.53GB ⇒ 几乎全是激活 ⇒ 激活 ∝ d
#   ⇒ d=512 预计 ~29GB。⛔ d=768（~44GB）塞不下，且硬塞只能动 (M41) 禁改的 --batch。
set -e
exec >> /tmp/m75_w512.txt 2>&1
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_m75_w512
cd /mnt/data/kw/RoundSquisheen/texture
echo "=== m75_w512 start $(date -u) on gpu $CUDA_VISIBLE_DEVICES"
/mnt/data/kw/anaconda3/envs/jzs_train/bin/python -u model/train_trd.py \
  --out /tmp/runs/trd_w512_09191345 \
  --seed 0 --reseed_after_build \
  --steps 44000 --lr 3e-4 --warmup 1000 --p32 0.5 --coarse --p_coarse 0.5 \
  --d 512 --heads 8 \
  --batch 256 --batch32 64 --wd 0.05 --drop 0.1 --depth 12 --codes 512 \
  --p_text_drop 0.1 --p_color_drop 0.5 --eval_every 1000 --sizes 16 32 \
  --bias_freqs 8 --bias_hidden 128 --level_emb --pal_aug 0.3 --pal_smooth 0.1 \
  --extra --extra_file train_extra_packs_only.json --n_ex 4 --p_ex_drop 0.3 \
  --n_domains 2 --save_at 22000 44000
echo "=== m75_w512 done $(date -u)"
echo "M75_W512_DONE"
