#!/usr/bin/env bash
# 正式测试（E-mat，出结果前定死的 272 个测试材质 / 857 张测试真人瓦片）。**只在验证集上定完配置后跑一次。**
#
#   RUN=runs/trd_vX CKPT=last.pt PAL=retrieve CT=20 CFG=1.5 TAG=TRDfinal bash eval/final_test.sh
#
# 产出：
#   experiments/baselines/<TAG>/{16,24,32}/      每材质 4 张（分布指标取第 0 张，多样性用 4 张）
#   experiments/baselines/<TAG>_rr4/{16,24,32}/  同 4 张里 CLIP-B/16 挑一张（与 B2 的 best-of-4 同预算；评测用 B/32）
#   experiments/colour/<TAG>/16/                 区域颜色任务（每张测试真人瓦片一个目标）
#   experiments/final_E_mat_{16,24,32}.json、experiments/final_colour_E_mat.json
# 判官（正反两问、先试点）另跑：eval/judge_pairs.py pilot/full --a <TAG>_rr4 --b B2 --set E_mat
set -e
: "${RUN:?}" "${TAG:?}"
CKPT=${CKPT:-last.pt}; PAL=${PAL:-retrieve}; CT=${CT:-20}; CFG=${CFG:-1.5}
P=${PY:-python}
export HF_HUB_OFFLINE=1
for S in 16 24 32; do
  BS=32; [ "$S" = 32 ] && BS=4; [ "$S" = 24 ] && BS=8
  $P eval/gen_trd.py --run "$RUN" --ckpt "$CKPT" --set E_mat --size $S --n 4 --bs $BS --cfg "$CFG" \
     --pal_mode "$PAL" --choice_temp "$CT" --tag "$TAG"
  $P eval/rerank.py --src "$TAG" --set E_mat --size $S --n 4
done
$P eval/gen_trd.py --run "$RUN" --ckpt "$CKPT" --set E_mat --size 16 --bs 32 --cfg "$CFG" \
   --pal_mode "$PAL" --choice_temp "$CT" --colour_task --tag "$TAG"
$P eval/run_eval.py --set E_mat --size 16 --methods B1 B2 B4 B5 "$TAG" "${TAG}_rr4" --out experiments/final_E_mat_16.json
$P eval/run_eval.py --set E_mat --size 24 --methods B1 B2 B4 "$TAG" "${TAG}_rr4" --out experiments/final_E_mat_24.json
$P eval/run_eval.py --set E_mat --size 32 --methods B1 B2 B4 "$TAG" "${TAG}_rr4" --out experiments/final_E_mat_32.json
$P eval/colour_task.py --set E_mat --methods B1+labshift B1+recolor B2+labshift B2+recolor B5+labshift \
   "$TAG" "$TAG+labshift" --out experiments/final_colour_E_mat.json
echo FINAL_TEST_DONE
