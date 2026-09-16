#!/usr/bin/env bash
# (M29) 把位置偏置表**按画布解绑**的重训 —— 唯一变量是 `--bias_size_cond`，控制组是 v11d。
#
# ===== 为什么花这笔钱（四块前提现已齐全，全部预注册过）=====
#   (M25) 3b0af3e：真人 32px 除"随画布缩放"的主周期外，**另有一支固定 4 像素的成分**
#         （C(4)=+0.010、C(12)=+0.011，包级 CI [+0.0066,+0.0178] 不含 0），比主成分小 5-8 倍。
#   (M26) a8ced52：这张表吃的是**归一化**偏移 u=d/n，而两档在**同一个 u** 上要求**相反的符号**
#         （u=0.375：16px −0.0303 [−.0494,−.0155] 是谷，32px +0.0111 [+.0049,+.0156] 是峰，47 包 LOPO 零翻侧）
#         -> 一张只吃 u 的表物理上装不下两档。
#   (M27) 5ec5556：那张表**是产物周期的把手**（24px 未训练画布，C_24(6)=+0.0416 族级 [+0.0168,+0.0757]）。
#   (M28) 67b8a04：**在训练过的 32px 上也是**（只换除数，ΔC(4)=+0.0375 族级 [+0.0244,+0.0528]，
#         空操作臂 272/272 逐字节相同）。
#   (M13) 4fa50da：**加性**的格子局部性支 `--bias_cells` 什么也没测到 -> 只死了那一种形式，
#         "替换/削弱归一化谐波支"仍是 (M26)/(M27)/(M28) 唯一授权的动作。
#
# ===== 改的是什么 =====
#   `model/trd.py::ToroidalBias(size_cond=True)`：MLP 的隐层经一层**零初始化的 FiLM**
#   （输入只有标量 s=n/32）做仿射调制 -> 同一个 u 在不同画布上可以给出不同（乃至反号）的偏置。
#   ⛔ **不用"固定格子周期的谐波"**：sin(2πf·wrap(d)/P) 在 n 不是 P 的整数倍时于折回点 d=n/2 不连续
#      -> n=24 上破坏可平铺（(M13) 的脚本里已写死这条，本轮不推翻）。
#   FiLM 只改每个隐单元的仿射系数，偏置仍只是 wrap(d)/n 的函数 -> 平移等变/可平铺**按构造**保留；
#   s 是连续标量 -> 没训过的 n=24 落在 16 与 32 之间**插值**（不是外推到未定义的 one-hot）。
#   本机自检（跑前、CPU、零 GPU）：
#     (S1) 零初始化 -> 16/24/32 三档偏置与旧路径 **max|Δ|=0.0**（第 0 步逐元素相同）；
#     (S2) FiLM 非零后同一个 u=0.25 上两档给出不同的值（解绑确实生效，gap=0.43）；
#     (S3) FiLM 非零下三档循环平移 max|Δ| 分别 8.9e-08 / 3.8e-06 / 1.3e-07，
#          与**旧路径**的 6.0e-08 / 1.3e-06 / 6.0e-08 同量级 = 浮点噪声，可平铺性未被破坏。
#
# ===== 判据（跑前写死、随本脚本一起提交；⛔ 一个字不许放宽）=====
#  ① **守门（交付不许退步）**：16px KID <= v11d 的 6.2 + 1.0 = **7.2**（与 (M13) ① 逐字相同）。
#     不过 -> 这条改动作废，**不再往下评** ②。
#  ② **主判据（结构；对着 (M26) 点名的"小的那一支"画）**：
#     真人 32px 有那支固定 4 像素的成分，而 v10/v11d 的产物**没继承**
#     （(M28) 控制臂 C(4)=−0.0020、C(12)=+0.0002）。解绑若真的解开了 (M26) 的符号冲突，
#     新臂的 32px 产物应当长出来：
#       **ΔC(4) = C_new(4) − C_v11d(4) 的族级 95% CI 下界 > 0**
#       **且** C_new(4) 自身的族级 CI 下界 > 0（合取保护，直接引 (M28)/(M27) 实测的空总体标定：
#       纯衰减总体上 C(4) 恒为负 -> 正的绝对 C(4) 凸性造不出来）。
#     ⚠ **C(12) 只作 (R3) 描述性读数，不进判据**（只用 d=4 一个位置 = 不做多重比较）。
#  ③ **操作检验（跑前写死；任一条不过则 ② 作废，不下判）**：
#     (OP1) 新臂 A_hat(1) >= 控制臂的 **0.8 倍** —— (M28) 的干预臂 A_hat(1) 从 0.166 掉到 0.083
#           正是"这一臂分布外"的证据；近邻相关塌了的话梳齿是退化产物，不算学到。
#     (OP2) 近纯色比例 flat_frac <= 控制臂 + 0.05（模型没垮成纯色）。
#     (OP3) 配对 >= 500 对、>= 150 族；LOFO 零翻侧才不加 `_FRAGILE` 后缀。
#  ④ **次要（不单独授权任何事）**：32px 的 KID/FID/FD 三项里至少两项优于 v11d 同配置
#     （与 (M13) ② 逐字相同）。⛔ **不许拿 ④ 替 ② 下判**（(M14) 的教训：显著的次要指标不能替未判定的主判据）。
#  ⑤ **判决**：
#     ①过 且 ②过 且 ③全过            -> **`UNBIND_WORKS`**：解绑让模型学到了真人 32px 那支被漏掉的成分。
#     ①过 且 ΔC(4) 族级 CI **上界 < +0.0066**（真人那支效应量的包级 CI 下界）-> **`UNBIND_NULL`**：
#                                        解绑到位了也补不出那一支 -> (M9) 这条线判死、别再投钱。
#     其余                              -> **`UNDECIDED`**（⛔ 不许读成任何一边）。
#  ⑥ **识别检验（(M26)/(M27) 点名要求，必须在看本轮任何真数据前跑过）**：复用
#     `analysis/arch/comb_shift.py` 已验证过的三条合成总体（周期 8->4 判"搬家"、两臂同周期判"没动"、
#     纯衰减不下判），另加本轮特有的**阳性对照 (ID4)**：同一条分析代码路径在 (M28) 已存的
#     `/tmp/gen32_m28/{m28_ctl,m28_over}` 上必须复现出 ΔC(4)=+0.0375 那个量级的"搬家"判决。
#  ⑦ **⛔ 本轮零判官、零 API**。即使判出 `UNBIND_WORKS`，32px 准入条件（6c64796）②
#     "不开任何 32px 新臂"**仍然挡着判官对判** —— 那需要**另行**解除，不由本轮授权。
#     本脚本**不问判官**，避免用胜率挑配置。
#  ⑧ **不动的东西**：`final_test.sh`、`UNITS_PER_TILE`、`judge_pairs.py`、任何默认值；
#     `--bias_size_cond` 默认关 = 旧行为一个字不变。
#
# ⚠ 本脚本**只出料，不下判**：ΔC(4) 由 `analysis/arch/comb_learned.py` 在下一轮算
#    （该脚本须先过 ⑥ 的四条识别检验才准碰本轮产物）。判据已随本脚本提交、先于任何数据。
# ⚠ 产物写 /tmp（/mnt/data 贴满）；gen32* 落在 scripts/sync_remote_tmp.sh 的同步列表里。
#
# 起法：tmux new-session -d -s arch_scond "GPU=6 bash scripts/trd_sizecond_train_eval.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
RUN=/tmp/runs/trd_scond_09161230
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-6} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_scond

# ---- 训练：配方与 v11d/(M13) 逐字相同，唯一差别是 --bias_size_cond ----
$P -u model/train_trd.py --out $RUN --steps 12000 --lr 1.5e-4 --warmup 500 \
  --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
  --extra --extra_file train_extra_packs_only.json+train_64to32.json \
  --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
  --bias_size_cond \
  --sizes 16 32 --p32 0.7 --save_at 6000 12000 > /tmp/trd_scond.txt 2>&1 || exit 1

# ---- ① 守门：16px（生成口径逐字照搬控制组）----
G="$P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --bs 8 --cfg 1.5 --pal_mode retrieve --xmodal"
$G --size 16 --n 2 --tag scondx || exit 1
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v10x v11dx scondx \
   --out $RUN/eval_scond_Vmat_16.json || exit 1

# ---- ② 主判据的料：E_mat 32px 两臂（新臂 + v11d 控制臂），口径逐字照搬 (M28) ----
OUT=/tmp/gen32_m29
C="$P -u eval/gen_trd.py --set E_mat --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --bs 8 --out $OUT --n 2"
$C --run $RUN            --ckpt last.pt --size 32 --tag m29_new   || exit 1
$C --run runs/trd_v11d   --ckpt last.pt --size 32 --tag m29_ctl   || exit 1
# (R3) 描述性：(M27)/(M28) 明写的前置读数 —— 改表之后 24px 的梳齿位置会不会变（⛔ 不进判据）
$C --run $RUN            --ckpt last.pt --size 24 --tag m29_new24 || exit 1
$C --run runs/trd_v11d   --ckpt last.pt --size 24 --tag m29_ctl24 || exit 1

# ---- ④ 次要指标：32/24 两档，方法列表逐字照搬 (M13) ----
for S in 32 24; do
  $G --size $S --n 2 --tag scondx_direct || exit 1
  $G --size $S --n 4 --ret_nname 100 --tag scondx100 || exit 1
  $P eval/rerank.py --src scondx100 --set V_mat --size $S --n 4 || exit 1
  $P eval/run_eval.py --set V_mat --size $S \
     --methods B1val B2val v11dx_direct v11dx100_rr4 scondx_direct scondx100_rr4 \
     --out $RUN/eval_scond_Vmat_$S.json || exit 1
done
echo SCOND_DONE
