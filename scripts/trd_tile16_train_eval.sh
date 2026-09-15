#!/usr/bin/env bash
# 32px 训练流里加一条增广：**16px 真人瓦片 2×2 平铺**（--p_tile16 0.5）——唯一变量，控制组是 v11d。
#
# 依据（全部是账本里已成立的事实，本轮不新测）：
#   1. (M9)/(M14-C)：真人瓦片的相关结构**按格子对齐**、不随画布缩放（只引方向，不引 L/R 数值）。
#      16px 真人瓦片 2×2 平铺出来的 32px 图，局部（格子尺度）统计**就是真人的**，
#      而且因为周期 16 整除 32，它在构造上仍然可平铺 → 是合法的 32px 训练样本。
#   2. 32px 缺的是**材质覆盖**不是张数（`6e2a95f`：32px 只覆盖 36% 测试材质、16px 覆盖 68%；
#      当时的结论是"合规供给不存在"所以关闭了**换数据来源**这条线）。本轮不换来源：
#      平铺用的是**同一批已在训的干净许可 16px 数据**，把 32px 那一路的材质覆盖补到 16px 的水平。
#      ⚠ 因此本轮的前提**不是**"32px 数据不够"（那条已被 24px 零数据打平 B2 反证掉），
#      而是"32px 那一路从没见过**跨材质**的真人格子尺度统计"。
#   3. (M13) 把格子尺度先验写进偏置（加性）**什么也没测到**，两种解释分不开：
#      (a) 错设不是原因；(b) 是原因但模型没有压力去用那组特征。本轮走的是另一条路——
#      不改架构、不动可平铺性，只在**数据侧**给那个先验加压力。模型参数与 v11d **逐位相同**。
#
# 控制组 = v11d，配方**逐字相同**（同 --init_from runs/trd_v10/last.pt、同 12000 步、同数据、
#   同超参、同 --coarse/--n_ex/--bias_freqs/--pal_aug 等），**唯一差别就是 --p_tile16 0.5**。
#   v11d 已发表（V_mat，(M13) 表）：16px KID 6.198 / FID 63.97 / FD 109.17 / CLIP 34.769；
#   32px _direct   KID 26.008 / FID 114.31 / FD 416.59 / CLIP 34.324；
#   32px 100_rr4   KID 24.758 / FID 109.09 / FD 425.79 / CLIP 35.051。
#   ⚑ --p_tile16 默认 0 且用短路求值不消耗随机数 → 旧配方逐位复现，`final_test.sh` 一个字不变。
#
# ===== 判据（跑前写死，与本文件一起提交；本脚本不问判官，不用胜率挑配置）=====
#   ① **16px 不许退步**：t16x 的 KID ≤ 6.198 + 1.0 = 7.2（与 (M13) 判据①逐字相同）。
#      不过 → 这条改动作废、不再往下评。
#   ② **32px 要真的动**（比 (M13) 严，跑前写死，消除"哪个同配置"的口径自由度）：
#      `_direct` 与 `100_rr4` **两个配置都要满足**——
#        (a) KID/FID/FD 三项里**至少一项**的改善幅度**超过 m=1 噪声下限**（KID 4.8 / FID 6.0 / FD 20）；
#        (b) 三项里**没有任何一项**的退步幅度超过噪声下限。
#      只要有一个配置不满足 (a)(b) 就判②不通过。全部落在下限内 → 照 (M13) 读作"**什么也没测到**"，
#      ⛔ 不许读成"更差"，也不许挑一个配置报。
#   ③ **退化检验**（`analysis/arch/tile2_degeneracy.py`，零 GPU/API）：实验组生成的 32px 图里
#      "四象限逐像素相同"的比例 **> 20%** 即判**退化**——哪怕②过了也**不授权**任何下一步，
#      因为那等于模型把 32px 塌成了"16px 复制两遍"（而 e4 已证原生 32px 优于放大的 16px）。
#      控制组同测作参照（预期 ≈0%）。
#   ④ ①②③全过**才**授权下一轮谈判官对判（仍受 32px 准入条件约束，本轮不碰）。
#   ⑤ 不改 `final_test.sh`、不改任何默认值；`--p_tile16` 默认 0 = 旧行为一个字不变。
#   ⑥ 24px 的 CLIP 与 16px 的 FID/FD/CLIP 只**报告**、不入任何判据。
#
# 起法：tmux new-session -d -s arch_tile16 "bash scripts/trd_tile16_train_eval.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
RUN=/tmp/runs/trd_tile16_09151300          # /mnt/data 只剩 87G → 检查点写 /tmp（sync_remote_tmp.sh 拉回）
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_tile16

$P -u model/train_trd.py --out $RUN --steps 12000 --lr 1.5e-4 --warmup 500 \
  --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
  --extra --extra_file train_extra_packs_only.json+train_64to32.json \
  --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
  --p_tile16 0.5 \
  --sizes 16 32 --p32 0.7 --save_at 6000 12000 > /tmp/trd_tile16.txt 2>&1 || exit 1

# 生成口径逐字照搬控制组（(M13) 那份）：16px 与 *_direct 用 --n 2；N=100 + 4 选 1 用 --n 4。
G="$P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --bs 8 --cfg 1.5 --pal_mode retrieve --xmodal"
$G --size 16 --n 2 --tag t16x || exit 1                 # 判据①：16px 是目前领先 B2 的那一档
for S in 32 24; do
  $G --size $S --n 2 --tag t16x_direct || exit 1
  $G --size $S --n 4 --ret_nname 100 --tag t16x100 || exit 1
  $P eval/rerank.py --src t16x100 --set V_mat --size $S --n 4 || exit 1
done
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v10x v11dx t16x --out $RUN/eval_tile16_Vmat_16.json
for S in 32 24; do
  $P eval/run_eval.py --set V_mat --size $S \
     --methods B1val B2val v11dx_direct v11dx100_rr4 t16x_direct t16x100_rr4 --out $RUN/eval_tile16_Vmat_$S.json
done
$P analysis/arch/tile2_degeneracy.py --tags t16x_direct t16x100_rr4 v11dx_direct v11dx100_rr4 \
   --size 32 --out $RUN/tile2_degeneracy.json
echo TILE16_DONE
