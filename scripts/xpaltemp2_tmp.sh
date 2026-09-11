#!/usr/bin/env bash
# 软典型度扫描第二个种子 + 补两个温度。
# 第一轮（seed 0）里 FD 对温度**非单调**（hard 104.2 → 0.01 是 92.1 → 0.03 是 96.6 → 名字检索 81.9），
# 而 KID 的自带标准差就有 1.6 —— 125 个材质 / 196 张参照上这些差多半有一半是噪声。
# 换种子重跑同样四个点，看 0.01 相对 hard 的 FD 优势（12 个点）是否复现。
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

G="$P eval/gen_trd.py --run runs/trd_v8 --ckpt last.pt --set V_mat --size 16 --n 2 --bs 16 --cfg 1.5 --pal_mode retrieve --seed 1"
$G --xmodal --tag s1_hard || exit 1
$G --tag s1_name || exit 1
for T in 0.01 0.02; do
  $G --xmodal --xpal_temp $T --tag s1_$T || exit 1
done
$P eval/run_eval.py --set V_mat --size 16 \
   --methods s1_hard s1_name s1_0.01 s1_0.02 \
   --out experiments/eval_xpaltemp_s1_Vmat.json
echo XPALTEMP_S1_DONE
