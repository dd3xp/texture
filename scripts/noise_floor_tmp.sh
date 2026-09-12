#!/usr/bin/env bash
# 验证集的噪声下限 + 重排操作点的误差棒（见 eval/noise_floor.py）。
# 两个端点配置各出每材质 8 张：跨模态硬取（= 正式测试的 TRD16）与名字检索（= v8f）。
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

G="$P eval/gen_trd.py --run runs/trd_v8 --ckpt last.pt --set V_mat --size 16 --n 8 --bs 16 --cfg 1.5 --pal_mode retrieve --seed 7"
$G --xmodal --tag nf8_xpal || exit 1
$G --tag nf8_name || exit 1

$P eval/noise_floor.py --set V_mat --size 16 --methods nf8_name nf8_xpal --k 8 \
   --subsets 1 2 4 --n_draws 12 --out experiments/noise_floor_Vmat.json || exit 1
$P eval/noise_floor.py --set V_mat --size 16 --methods nf8_name nf8_xpal --k 8 \
   --subsets 2 4 --n_draws 12 --rerank --out experiments/noise_floor_rr_Vmat.json || exit 1
echo NOISE_FLOOR_DONE
