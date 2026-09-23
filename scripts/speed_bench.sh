#!/usr/bin/env bash
# (P7) 速度表：正式测试的原配置原命令，独占一张 A100，端到端 wall-clock。
#   TRD16c_rr4 : gen_trd（272 材质 × 4）+ rerank 4 选 1，含模型/CLIP 加载，总时间 / 272
#   B7         : ddpm_unet gen（272 × 4，DDIM 50 步），含加载，总时间 / 272 / 4（每张）
#   B2         : SDXL 1024 × 4 渲染 + auto_crop + 降采样量化，--limit 20，
#                按脚本自带逐材质计时（不含加载，去掉第 1 个热身材质）
#   B3         : 已有实测（16px 中位 4.44 h/张），不重跑
# 输出全在 /tmp/speed，不碰 experiments/baselines 下的正式产物（B7 另用 tag SPD_B7，跑完删）。
set -e
cd /mnt/data/kw/RoundSquisheen/texture
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-4}
O=/tmp/speed; mkdir -p $O
now() { date +%s.%N; }
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader > $O/gpu_before.txt

t0=$(now)
$P eval/gen_trd.py --run runs/trd_v8 --ckpt last.pt --set E_mat --cfg 1.5 --pal_mode retrieve --xmodal \
   --size 16 --n 4 --bs 16 --ret_nname 100 --tag SPD_TRD16c --out $O > $O/trd_gen.log 2>&1
t1=$(now)
$P eval/rerank.py --src SPD_TRD16c --set E_mat --size 16 --n 4 --root $O > $O/trd_rr.log 2>&1
t2=$(now)
echo "TRD gen=$(echo "$t1-$t0"|bc) rerank=$(echo "$t2-$t1"|bc)" | tee $O/trd_time.txt

t0=$(now)
$P baselines/ddpm_unet.py gen --run runs/b7_ddpm_09120300 --set E_mat --size 16 --n 4 --cfg 1.5 --tag SPD_B7 > $O/b7.log 2>&1
t1=$(now)
echo "B7 total=$(echo "$t1-$t0"|bc)" | tee $O/b7_time.txt
rm -rf experiments/baselines/SPD_B7

$P baselines/sdxl_baselines.py --set E_mat --n 4 --sizes 16 --limit 20 --out $O/b2 > $O/b2.log 2>&1
echo done > $O/DONE
