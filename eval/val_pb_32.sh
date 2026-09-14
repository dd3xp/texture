#!/usr/bin/env bash
# 32px 按包均衡采样（γ=0.5，从 v10 接着训 8000 步）的验证：训完自动出图 → 4 选 1 → 与现行 32px 配置同材质直接对判。
# 预注册：本脚本先提交再跑。挂在服务器上排队，不依赖任何会话在线。
#
#   新：/tmp/runs/trd_v10pb05_09141130（--pack_balance 0.5，批 192/48）
#   旧：v10x100_rr4（v10，跨模态 N=100，CLIP-B/16 4 选 1 = 正式测试 TRD32 的配置）
# 判据（写死）：
#   ① 先试点（15 真题 + 5 同图空对照），可解率 ≥65% 且高于空对照才跑全量；
#   ② 新配置算"更好" = 全量胜率二项 p < 0.05 且方向为胜 → 下一步试 γ=1；
#      平或输 → 按包均衡这条路关闭（γ=1 只在 ① 通过且方向为胜时才试）；
#   ③ 换 32px 主配置再测测试集 = 对测试集的第二次观察，须另行预注册、只跑一次。
# 大文件写 /tmp（由 scripts/sync_remote_tmp.sh 同步回本机）；小图写 experiments/baselines（几 MB）。
set -u
P=${PY:-python}
RUN=/tmp/runs/trd_v10pb05_09141130
OUT=/tmp/judge_pb05
mkdir -p "$OUT"
while tmux ls 2>/dev/null | grep -q arch_pb05; do sleep 60; done
[ -f "$RUN/last.pt" ] || { echo "no checkpoint"; exit 1; }
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton
export CUDA_VISIBLE_DEVICES=${GPU:-6}
$P eval/gen_trd.py --run "$RUN" --ckpt last.pt --set V_mat --size 32 --bs 8 --n 4 --cfg 1.5 \
   --pal_mode retrieve --xmodal --ret_nname 100 --tag pb05x100 || exit 1
$P eval/rerank.py --src pb05x100 --set V_mat --size 32 --n 4 || exit 1
$P eval/run_eval.py --set V_mat --size 32 --methods B1val B2val v10x100_rr4 pb05x100_rr4 --out "$OUT/eval_pb05_Vmat_32.json"
$P eval/judge_pairs.py pilot --a pb05x100_rr4 --b v10x100_rr4 --set V_mat --size 32 --outdir "$OUT" && \
$P eval/judge_pairs.py full  --a pb05x100_rr4 --b v10x100_rr4 --set V_mat --size 32 --outdir "$OUT"
echo PB05_DONE
