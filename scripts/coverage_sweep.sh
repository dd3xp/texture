#!/usr/bin/env bash
# 覆盖率 vs 选择：同一个 v4 检查点出 32 张/材质，CLIP-B/16 在前 n 张里挑一张（n 嵌套），
# 看判官胜率随 n 怎么走。n=4 那一档已知在结构性材质（HI）上 60%，单张只有 29%——
# 如果继续往真人的 84% 爬，缺的就是"挑"而不是"生成"；如果早早封顶，缺的是模型本身。
# ⚠ 预算不对等（B2 是 best-of-4），所以这是诊断，不是能拿去比的方法。
# 用法：bash scripts/coverage_sweep.sh   （等 v7 的 val_full 跑完再开始，不抢卡）
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
LOG=experiments/valfull_v7.txt
while ! grep -q VALFULL_DONE $LOG 2>/dev/null; do sleep 120; done
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-7} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=v4ct20x32
$P eval/gen_trd.py --run runs/trd_v4 --ckpt last.pt --set V_mat --size 16 --bs 16 --n 32 \
   --cfg 1.5 --pal_mode retrieve --choice_temp 20 --tag $TAG || exit 1
# n=1（就是第 0 张）必须自己跑一遍：已公布的单张 29% / 四选一 60% 是**旧检索**下量的，
# 而这一批用的是现在的默认（色数跟检索走），两端必须在同一条件下才连得成一条曲线。
for N in 1 4 8 16 32; do $P eval/rerank.py --src $TAG --set V_mat --n $N || exit 1; done
$P eval/run_eval.py --set V_mat --methods B2val v4_ret_ct20 ${TAG}_rr1 ${TAG}_rr4 ${TAG}_rr8 ${TAG}_rr16 ${TAG}_rr32 \
   --out experiments/eval_coverage_Vmat.json
$P eval/periodicity.py --methods REALval v4_ret_ct20 ${TAG}_rr4 ${TAG}_rr32
for A in ${TAG}_rr32 ${TAG}_rr8 ${TAG}_rr1; do
  $P eval/judge_pairs.py pilot --a $A --b B2val --set V_mat || continue
  $P eval/judge_pairs.py full --a $A --b B2val --set V_mat
done
echo COVERAGE_DONE
