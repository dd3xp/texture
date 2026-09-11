#!/usr/bin/env bash
# v11 = v10 的配方 + 固定相位的粗网格条件（--coarse_phase fixed）。
# 依据：analysis 探针（runs/probe_v10/probe_coarse.json）——v10 确实在读粗网格
# （val32 结构交叉熵 off 2.133 → on 1.842，喂错瓦片的粗网格反而升到 2.347），
# 但随机相位让"这 2×2 块里有一格是这个色阶"成了四选一的歧义；推理端 gen_trd 的
# ix=[0,0,1,1,…] 本来就是相位 (0,0)，训练对齐过去后粗网格才是硬约束。
# 排在 v10 的评测之后跑，不抢 7 号卡。
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
while ps -eo args | grep -q "[v]10eval_tmp.sh"; do sleep 120; done

export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$P -u model/train_trd.py --out runs/trd_v11 --steps 12000 --lr 1.5e-4 --warmup 500 \
  --bias_freqs 8 --bias_hidden 128 --level_emb --pal_aug 0.3 --pal_smooth 0.1 \
  --sizes 16 32 --p32 0.5 --extra --extra_file train_extra_packs_only.json --n_ex 4 \
  --coarse --coarse_phase fixed --init_from runs/trd_v8/last.pt --save_at 6000 12000 \
  > experiments/trd_v11.txt 2>&1 || exit 1

# 学到了没有：同一把探针对准 v11（"on" 模式本来就用相位 (0,0)，与 v11 的训练口径一致）
$P -u model/train_trd.py --out runs/probe_v11 --bias_freqs 8 --bias_hidden 128 --level_emb \
  --pal_aug 0.3 --pal_smooth 0.1 --sizes 16 32 --extra --extra_file train_extra_packs_only.json \
  --n_ex 4 --coarse --init_from runs/trd_v11/last.pt --probe_coarse runs/trd_v11/last.pt \
  > experiments/probe_v11.txt 2>&1

G="$P eval/gen_trd.py --run runs/trd_v11 --ckpt last.pt --set V_mat --bs 8 --n 2 --cfg 1.5 --pal_mode retrieve --xmodal"
$G --size 16 --tag v11x || exit 1          # 16px 是目前领先 B2 的那一档，得确认没被改坏
for S in 32 24; do
  $G --size $S --cascade 0 --tag v11x_casc || exit 1
done
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v8x v10x v11x --out experiments/eval_v11_Vmat_16.json
$P eval/run_eval.py --set V_mat --size 32 --methods B1val B2val v8x v10x_casc v11x_casc --out experiments/eval_v11_Vmat_32.json
$P eval/run_eval.py --set V_mat --size 24 --methods B1val B2val v8x v10x_casc v11x_casc --out experiments/eval_v11_Vmat_24.json
echo V11_DONE
