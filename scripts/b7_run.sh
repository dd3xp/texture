#!/usr/bin/env bash
# B7（同数据从零训练的扩散 UNet）：训练 → 验证集 16/24/32 + 区域颜色任务 → 指标表
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=runs/b7_ddpm_09120300
$P -u baselines/ddpm_unet.py train --out $RUN --steps 40000 > experiments/b7_ddpm_09120300.txt 2>&1 || exit 1
for S in 16 32 24; do $P baselines/ddpm_unet.py gen --run $RUN --set V_mat --size $S --n 2 --tag B7val || exit 1; done
$P baselines/ddpm_unet.py gen --run $RUN --set V_mat --size 16 --colour_task --tag B7val || exit 1
$P eval/run_eval.py --set V_mat --size 16 --methods B1val B2val B7val v8x --out experiments/eval_b7_Vmat_16.json
$P eval/run_eval.py --set V_mat --size 32 --methods B1val B2val B7val v10x_direct --out experiments/eval_b7_Vmat_32.json
$P eval/run_eval.py --set V_mat --size 24 --methods B1val B2val B7val v10x_direct --out experiments/eval_b7_Vmat_24.json
$P eval/colour_task.py --set V_mat --methods B2+labshift B7val B7val+labshift v8xc+labshift --out experiments/colour_Vmat_b7.json
echo B7_DONE
