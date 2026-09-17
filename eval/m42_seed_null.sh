#!/usr/bin/env bash
# =====================================================================================
# (M42) 给 (M39) 那把零 API pixfrac 筛子做一次**构造性零对照**。**本轮零 API：本文件里
# 没有任何判官调用，也不读 VLM 凭据** —— 判决只由 `analysis/arch/m42_read_seed_null.py`
# （盲写于任何一张图存在之前，判据全文在它的 docstring 里）给出。
#
# 为什么是这一轮的活：(M41) 起筛子被**当门用**并第一次挡下一小时 API，可它的锚点只有 n=3、
# 从没独立验证过，而门的固有代价是「说 NOGO 的那次也不会给它评分」⇒ 靠开臂永远攒不到验证。
# 唯一一对**只换种子**的重训（trd_seed{1,2}_09161230，config 只差 out/seed、同码本、同为
# 12000 步、同一个 --init_from）真效应按构造为 0 ⇒ 它们之间的 pixfrac 就是这把尺子在
# 「跨 run 重训」这一类比较上的**纯噪声上限**。这是零 API 能买到的那一半验证。
#
# 出图配方**逐字照抄** (M37)(M38)(M40)(M41) 用的 TRD16c：16px / V_mat / --cfg 1.5 /
# --pal_mode retrieve --xmodal --ret_nname 100 / --n 4 + rerank.py --n 4。
# 唯一变量 = `--run`。⛔ 跑完不许改配方、不许挪锚点（D37/D38/D40 写死在判读器里）、
# ⛔ 不许换尺子（(M39)：同一份数据 mae 给的次序相反 —— 换尺子取顺眼结论＝p-hacking）。
#
# 用法（远程，GPU 显存看 memory.free 不看 utilization；16px 生成约 5.2GB）：
#   CUDA_VISIBLE_DEVICES=3 bash eval/m42_seed_null.sh
# =====================================================================================
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
O=${OUT:-/tmp/m42seed}
MET=${MET:-/tmp/m42_seed_null.json}
RA=${RA:-/tmp/runs/trd_seed1_09161230}
RB=${RB:-/tmp/runs/trd_seed2_09161230}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_m42
mkdir -p "$O"

"$P" -c 'import numpy, PIL, torch, sys; sys.exit(0)' || { echo "[m42] python 探针失败: $P"; exit 1; }
# 判读器先自检再干活（账本：盲写判读器必须先跑 --selftest）
"$P" analysis/arch/m42_read_seed_null.py --selftest || { echo "[m42] 判读器自检不过 -> 停"; exit 1; }
md5sum "$RA/codebook.npy" "$RB/codebook.npy" runs/trd_v8/codebook.npy

COMMON="--ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --out $O"
echo "[m42] A: eval/gen_trd.py --run $RA $COMMON --tag seed1"
echo "[m42] B: eval/gen_trd.py --run $RB $COMMON --tag seed2"
$P eval/gen_trd.py --run "$RA" $COMMON --tag seed1 || exit 1
$P eval/gen_trd.py --run "$RB" $COMMON --tag seed2 || exit 1
for t in seed1 seed2; do
  $P eval/rerank.py --src $t --set V_mat --size 16 --n 4 --root "$O" || exit 1
done

$P analysis/arch/m42_read_seed_null.py --root "$O" --ra "$RA" --rb "$RB" --out "$MET"
RC=$?
# 次要指标只登记、⛔ 不当证据（账本：分布指标在这几档基本全瞎）
$P eval/run_eval.py --set V_mat --size 16 --root "$O" --out /tmp/m42_gen16.json \
   --methods seed1_rr4 seed2_rr4 seed1 seed2 || true
echo "M42_SEED_NULL_DONE rc=$RC"
