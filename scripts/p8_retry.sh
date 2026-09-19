#!/usr/bin/env bash
# (P8) 补跑器：把主脚本里因**别人占卡**而 OOM 的臂串行补上。
# ⚠ 这是**运维包装，不是判据变更**：训练与生成命令与 `scripts/p8_bias_ablation.sh` 里的
#   `train_arm` 逐字相同（同 STAMP、同超参、同成对配方 --seed 0 --reseed_after_build）。
#   ⛔ 不许在这里改任何超参；已有 `last.pt` 的臂**跳过**，不重训（重训会换掉已产出的臂 = 偷偷换料）。
# 背景：2026-09-19 首发时 GPU 6 被别人的作业占到 64G，A1 当场 CUDA OOM；八张卡里只有一张空 >25G。
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_p8
STAMP=09190900
NEED=${NEED:-25000}          # 需要的空闲显存 MiB

wait_gpu() {                  # 打印一张空闲显存 >= NEED 的卡号；没有就等
  while :; do
    G=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
        | awk -F', ' -v need=$NEED '{free=$3-$2; if (free>=need) {print $1; exit}}')
    [ -n "${G:-}" ] && { echo "$G"; return; }
    sleep 120
  done
}

train_arm() {                 # $1=臂名  $2...=额外参数（与主脚本逐字相同）
  NAME=$1; shift
  RUN=/tmp/runs/trd_p8${NAME}_$STAMP
  if [ -f "$RUN/last.pt" ]; then echo "[$NAME] 已有 last.pt，跳过"; return 0; fi
  GPU_ID=$(wait_gpu)
  echo "[$NAME] 用 GPU $GPU_ID 开训 $(date -u +%H:%M)"
  CUDA_VISIBLE_DEVICES=$GPU_ID $P -u model/train_trd.py --out $RUN --steps 12000 --lr 1.5e-4 --warmup 500 \
    --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
    --extra --extra_file train_extra_packs_only.json+train_64to32.json \
    --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
    --seed 0 --reseed_after_build "$@" \
    --sizes 16 32 --p32 0.7 --save_at 12000 > /tmp/p8_$NAME.txt 2>&1 || { echo "[$NAME] 训练失败"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_ID $P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --size 16 --bs 8 \
    --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --tag p8${NAME}x || { echo "[$NAME] 生成失败"; return 1; }
  echo "[$NAME] DONE $(date -u +%H:%M)"
}

while tmux ls 2>/dev/null | grep -q '^arch_p8:'; do sleep 120; done   # 等主脚本退出，避免抢同一张卡
train_arm C
train_arm A1 --bias_off
train_arm A2 --bias_wrap none
train_arm A3 --bias_wrap broken
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v11dx p8Cx p8A1x p8A2x p8A3x \
   --out /tmp/p8_eval_Vmat_16.json || exit 1
echo P8_ALL_DONE
