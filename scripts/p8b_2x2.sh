#!/usr/bin/env bash
# (P8b) 2×2：偏置周期性 × 循环平移增广。预注册：本脚本与 `analysis/arch/p8b_read.py` 先提交再跑，只跑一次。
#
# ===== 为什么不是原计划的"偏置开关 × roll" =====
# ① TRD 没有绝对位置。`--bias_off` 后网格注意力对格子排列置换不变 ⇒ 产物没有空间结构，
#    接缝与内部一样"乱"，|ratio−1| 自然≈0 ⇒ (P8) A1 的 BIAS_NULL 是尺子在无结构图上失明，
#    不是偏置不承重（A1 KID 8.4→26.9 同时发生）。偏置开关 × roll 两格都测不到接缝。
# ② torus 偏置下模型对循环平移严格等变（论文 Proposition），循环平移后的样本损失逐字相同 ⇒
#    roll 增广对 torus 臂在数学上近乎空操作。
# ⇒ 能检验"构造上可平铺"的对照是 **{torus, none} × {roll, no_roll}**：
#    none = 同参数量、不折回的普通相对偏置（(P8) A2），它不等变，接缝要靠 roll 从数据里学。
#
# ===== 四臂（全部按 v8 配方从头训练 20k 步，⛔ 不从任何 checkpoint 初始化）=====
#   TR  torus + roll      （= 交付配方）
#   T0  torus + no_roll
#   NR  none  + roll      （= (P8) A2 的从头训练版）
#   N0  none  + no_roll
# 从头训练是必须的：(P8) 各臂从 v10 微调，v10←v8 是带 roll 训的 ⇒ 关 roll 的微调臂会继承无缝能力。
#
# ===== 判据（跑前写死，判读器逐字实现；⛔ 一个字不许放宽）=====
# 尺子：16px V_mat 产物的 |ratio−1|（(P8) 同一把 `dev`，(P3) 已验证），每臂 125 材质各取第 0 张。
# worse(X,Y) := 中位 |ratio−1| 满足 X > Y + 0.05 **且** 逐材质配对符号检验 p<0.05（X 更差占多数）。
#   K1 = worse(N0, NR)   非周期偏置没有 roll 就出接缝（尺子在这一格看得见接缝的前提）
#   K2 = worse(N0, T0)   同样没有 roll，torus 无缝而 none 有缝
#   K3 = not worse(T0, TR)  torus 去掉 roll 不变差
# 判决：
#   K2 且 K3            -> `BY_CONSTRUCTION`   摘要"构造上保住可平铺"由本实验坐实
#   K2 且 非K3          -> `TORUS_PARTIAL`      torus 承重但 roll 仍有贡献
#   非K2 且 K1          -> `AUG_CARRIES`        可平铺由增广承载 ⇒ 摘要那句必须改
#   非K1 且 非K2        -> `RULER_BLIND`        N0 也看不出缝 ⇒ 本实验不能归因，⛔ 不许读成任何一方赢
# 质量读数（不参与判决）：四臂 16px V_mat KID 对着地板 4.76 读；N0 若出缝且 KID 超地板变差，照实登记。
# 操作检验：四臂 config.json 只允许 bias_wrap / no_roll / out 不同；四臂都有 last.pt；每臂 125 材质齐。
# ⛔ 不跑测试集 E_mat，⛔ 不改 final_test.sh 的任何默认值，⛔ 不调判官 API。
#
# 起法：tmux new-session -d -s arch_p8b "bash scripts/p8b_2x2.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_p8b
STAMP=${STAMP:-09240430}

$P analysis/arch/p8_selftest.py || { echo "自检未过，拒绝起跑"; exit 1; }

train_arm() {          # $1=臂名  $2=GPU  $3...=差异参数
  NAME=$1; GPU_ID=$2; shift 2
  RUN=/tmp/runs/trd_p8b${NAME}_$STAMP
  CUDA_VISIBLE_DEVICES=$GPU_ID $P -u model/train_trd.py --out $RUN --steps 20000 --lr 3e-4 --warmup 1000 \
    --extra --extra_file train_extra_packs_only.json --n_ex 4 \
    --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
    --sizes 16 32 --p32 0.3 --seed 0 --reseed_after_build --save_at 20000 "$@" \
    > /tmp/p8b_$NAME.txt 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$GPU_ID $P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --size 16 --bs 8 \
    --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --tag p8b${NAME}x >> /tmp/p8b_$NAME.txt 2>&1 || return 1
}

( train_arm TR ${G1:-0}                              && echo TR_DONE ) &
( train_arm T0 ${G2:-1} --no_roll                    && echo T0_DONE ) &
( train_arm NR ${G3:-2} --bias_wrap none             && echo NR_DONE ) &
( train_arm N0 ${G4:-3} --bias_wrap none --no_roll   && echo N0_DONE ) &
wait

$P eval/run_eval.py --set V_mat --size 16 --methods B2val p8bTRx p8bT0x p8bNRx p8bN0x \
   --out /tmp/p8b_eval_Vmat_16.json || exit 1
$P analysis/arch/p8b_read.py --stamp $STAMP --eval /tmp/p8b_eval_Vmat_16.json --out /tmp/p8b_read.json
echo P8B_DONE
