#!/usr/bin/env bash
# 32px 训练侧第二条（也是**最后一条**）臂：按包均衡 **γ=1.0** vs 现配置。验证集 V_mat，同材质直接对判。
# 预注册：本脚本先提交再跑。
#
# 承 `eval/val_judge_pb05.sh`（预注册 a6b9742）：γ=0.5 判官 **36/73 = 49%，p=1，[38%,61%]** —— 打平。
# 那份预注册的判据 (3) 写着"打平 -> γ=0.5 关闭，最多再试一档 γ=1（须另行预注册）"，本文件即是。
# γ=0.5 把两个近乎平涂的大包（`dungeonsoup` 220 + `macrotex` 186，占 32px 池 641 张的 63%）压到约 25%，
# γ=1.0 压到约 8%（各包等概率）。⚠ γ=1 的风险是一批 48 张要摊到 24 个包、小包只有 1–6 张 → 重复采样。
#
# 训练（已起，tmux `arch_pb10`，GPU 2，日志 `/tmp/trd_v10pb10.txt`）：
#   /tmp/runs/trd_v10pb10_09141300 = runs/trd_v10/last.pt 接着训 8000 步，
#   **除 `--pack_balance 1.0` 外与 pb05 逐字相同**（batch 192/48、p32 0.5、lr 1.5e-4、warmup 500）
#   → 两条臂之间 γ 是唯一的差别。
#
# **预注册判据（跑之前写死）**：
#  (1) 试点（15 真题 + 5 对两边同图的空对照）真题可解率 >= 65% 且高于空对照，才跑 full。
#  (2) 主判据：A（pb10）胜率的二项双侧 p < 0.05 才算判出方向。
#  (3) **方向预测**：预测 pb10 胜。**打平或输 -> "调训练池采样权重"整条线关闭**，
#      不再试第三个 γ，也不再用别的权重（按包等概率已是这族的端点）。
#  (4) 混杂同 pb05：pb10 = v10 + 8000 步 + 均衡，多训的 8000 步没有对照 →
#      **赢了才谈换主配置，且换之前必须先跑 γ=0 的对照臂**（v10 接着训 8000 步，其余一字不改）。
#  (5) 这只动验证集。据此换 32px 主配置再测测试集 = 对测试集的第二次观察，须另行预注册、只跑一次。
#  (6) ⚠ 判官压缩效应：正结果作下界；负结果只能说"没测到大效应"。
#
# 产物全写 /tmp（`/mnt/data` 随时会被别人写满），跑完 scp 回本机入库。凭据只从环境变量读。
# 起法：tmux new-session -d -s arch_pb10j "VLM_BASE_URL=... VLM_API_KEY=... CUDA_VISIBLE_DEVICES=<空卡>
#        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PY=<jzs_train python> bash /tmp/val_judge_pb10.sh > /tmp/vjpb10.txt 2>&1"
# ⚠ 生成要约 10GB 显存：GPU 3 只剩 9.6GB 时实测 OOM。
set -u
P=${PY:-python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
RUN=${RUN:-/tmp/runs/trd_v10pb10_09141300}
O=${OUT:-/tmp/gen32pb10}
J=${JOUT:-/tmp/judge_pb10}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
mkdir -p "$O" "$J"

$P eval/gen_trd.py --run "$RUN" --ckpt last.pt --set V_mat --size 32 --n 4 --bs 8 \
   --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --tag pb10x100 --out "$O" || exit 1
$P eval/rerank.py --src pb10x100 --set V_mat --size 32 --n 4 --root "$O" || exit 1
cp -r "$REPO/experiments/baselines/v10x100_rr4" "$O/" || exit 1

$P eval/judge_pairs.py pilot --a pb10x100_rr4 --b v10x100_rr4 --set V_mat --size 32 \
   --root "$O" --outdir "$J" \
  && $P eval/judge_pairs.py full --a pb10x100_rr4 --b v10x100_rr4 --set V_mat --size 32 \
   --root "$O" --outdir "$J"
echo VJ_PB10_DONE
