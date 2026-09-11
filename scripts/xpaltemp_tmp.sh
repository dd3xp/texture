#!/usr/bin/env bash
# 软典型度调色板（--xpal_temp）在验证集上的扫描。
#
# 为什么做这个：正式测试里 TRD16 输给 B7 的唯一一项是 FD-DINOv2（82.3 vs 59.3）。
# 验证集消融（arch_progress 2026-09-12"跨模态检索拆开看"）已定位到**全部 FD 代价与全部
# CLIP 收益都来自调色板那一半**：只对调色板做跨模态 CLIP 34.79 / FD 106.5，只对范例做
# 33.51 / 82.5，名字检索（不挑）33.47 / 81.9。原实现是**硬取**：在 n_name=30 个同名候选里
# 取图文相似度最高的 topk=5 再均匀抽一个 → 系统性偏向"典型"配色，生成集合比画师窄。
# 软典型度把它换成按 softmax(相似度/温度) 在 30 个候选上抽：温度→0 等于原来的硬取，
# 温度很大等于名字检索。两个端点的 FD 差 22 个点、CLIP 差 1.35 —— 本扫描看中间是否有
# 同时压住 B7（val FD 86.2、KID 6.0、FID 65.9、CLIP 33.03）四项的操作点。
#
# 只跑验证集（V_mat），测试集不碰。
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

G="$P eval/gen_trd.py --run runs/trd_v8 --ckpt last.pt --set V_mat --size 16 --n 2 --bs 16 --cfg 1.5 --pal_mode retrieve"
$G --xmodal --tag xpt_hard || exit 1                      # 端点 A：原 TRD16（硬取前 5）
$G --tag xpt_name || exit 1                               # 端点 B：名字检索（v8f）
for T in 0.003 0.01 0.03; do
  $G --xmodal --xpal_temp $T --tag xpt_$T || exit 1
done
$P eval/run_eval.py --set V_mat --size 16 \
   --methods B2val B7val_c1.5 xpt_hard xpt_name xpt_0.003 xpt_0.01 xpt_0.03 \
   --out experiments/eval_xpaltemp_Vmat.json
echo XPALTEMP_DONE
