#!/usr/bin/env bash
# (P8) 环面相对注意力偏置的消融 —— 摘要里"用不增参数的环面偏置在构造上保住可平铺"那句的唯一支撑。
# 预注册：本脚本与 `analysis/arch/p8_selftest.py` 先提交再跑，只跑一次。
#
# ===== 为什么必须做 =====
# ① 摘要（`docs/paper_draft.md`）里最像方法贡献的一句，至今**零消融**；
# ② 体裁调研（`docs/paper_genre_survey.md` §四）：参数量配平的 drop-in 互换是本族**最低门槛**，
#    不做会被打；且 **全族没有一篇做过"去掉构造性机制就失败"的反证**（唯一剂量-应答是 FabricDiffusion
#    的 TexTile 0.47→0.62）→ 这是便宜且无人占领的位置；
# ③ 审稿人必问的 "为什么不直接用 circular padding"（TileGen 那条）要靠这组臂 + B7 一起答。
#
# ===== 跑前自检已过（`p8_selftest.py`，5/5）=====
# ⚑ **自检抓到一个会让整轮白跑的问题**：原计划的"非环面"臂＝去掉折回那两行，实测与环面臂**逐位相同**
#   —— 因为 `sin/cos(2πf·d)` 本身以 1 为周期，折回在数学上是多余的。
#   ⇒ **可平铺性来自谐波编码，不是来自折回**（这句本身要写进论文）。故非环面臂改为换编码：
#   非周期的幂特征 [d^k, |d|^k]，**维数与谐波编码逐位相同（4×freqs）→ 参数量一个字不变**（(S5) 三臂均 1638）。
# ⚑ (S3) 循环平移等变：torus 5.96e-08（等变）／none 6.7e-02／broken 2.7e-01（都被破坏）＝对照有区分度。
#
# ===== 四条臂（唯一变量＝偏置；配方与 v11d 逐字相同，成对配方 --seed 0 --reseed_after_build）=====
#   C   （控制）  默认              环面谐波偏置
#   A1  --bias_off                 整块网格-网格偏置置零（参数仍在，不参与）
#   A2  --bias_wrap none           非周期幂特征，参数量相同
#   A3  --bias_wrap broken         按错误周期 1.3n 折回（构造性保证被故意打破）
# ⚠ 2×2（偏置 × 随机 roll 增广，`--no_roll`）留作 (P8b)，**另行预注册**；本轮只跑主四条。
#
# ===== 判据（跑前写死，⛔ 一个字不许放宽）=====
#  ① **性质检验（主判据，对着"构造上保可平铺"那句画）**：16px 产物的接缝比值偏离 |ratio−1|（理想 1，
#     两侧都差，(P3) 已证该口径对平涂/抹边免疫）。判 **`BIAS_IS_LOAD_BEARING`** 需要
#     **A1、A2、A3 三条全部**的 |ratio−1| 中位数 > C 的中位数 + 0.05，**且**逐材质配对符号检验 p < 0.05；
#     若只有部分臂变差 → **`BIAS_PARTIAL`**（照实报是哪几条）；若没有一条变差 → **`BIAS_NULL`**
#     ⇒ ⛔ 摘要里"构造上保住可平铺"那句必须改写成"我们采用了环面编码"，**不许再声称它承重**。
#  ② **质量守门（读数，不单独授权）**：16px KID 的差值必须对着 **max(重训漂移 1.087, 采样下限 4.76)=4.76**
#     读；⛔ 小于 4.76 的差值一律记"什么也没测到"，⛔ 不许说某臂"更差"。
#  ③ **操作检验（任一不过 ⇒ 该臂作废）**：(OP1) 四臂 `config.json` 逐键比对，只允许 `bias_wrap`/`bias_off`
#     不同；(OP2) 四臂都跑满 12000 步且 `last.pt` 存在；(OP3) 生成口径四臂逐字相同；
#     (OP4) 每臂 V_mat 125 材质齐。
#  ④ **跑前方向预测（照录，跑完无论对错都登账）**：A1 最差；A3 次差（接缝被明确打破）；
#     A2 与 C 的差**小于** A3 与 C 的差（幂特征虽非周期，但仍是平滑的相对偏置）。
#     ⚠ 账在此：本项目跑前预测**赢少输多**，(P1) 刚错过一条。
#  ⑤ ⛔ 本轮**零判官、零 API**；⛔ 不授权任何配置变更；⛔ 不动 `final_test.sh` 与任何默认值
#     （`--bias_wrap` 默认 torus、`--bias_off`/`--no_roll` 默认关 = 旧行为逐位不变，(S1) 已验）。
# 产物写 /tmp（`/mnt/data` 长期贴满）。
#
# 起法：tmux new-session -d -s arch_p8 "GPU=2 GPU_B=6 bash scripts/p8_bias_ablation.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=/tmp/triton_p8
STAMP=09190900

$P analysis/arch/p8_selftest.py || { echo "自检未过，拒绝开跑"; exit 1; }

train_arm() {          # $1=臂名  $2=GPU  $3...=额外参数
  NAME=$1; GPU_ID=$2; shift 2
  RUN=/tmp/runs/trd_p8${NAME}_$STAMP
  CUDA_VISIBLE_DEVICES=$GPU_ID $P -u model/train_trd.py --out $RUN --steps 12000 --lr 1.5e-4 --warmup 500 \
    --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
    --extra --extra_file train_extra_packs_only.json+train_64to32.json \
    --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
    --seed 0 --reseed_after_build "$@" \
    --sizes 16 32 --p32 0.7 --save_at 12000 > /tmp/p8_$NAME.txt 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$GPU_ID $P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --size 16 --bs 8 \
    --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --tag p8${NAME}x || return 1
}

( train_arm C  ${GPU:-2}                    && echo C_DONE  ) &
( train_arm A1 ${GPU_B:-6} --bias_off       && echo A1_DONE ) &
wait
( train_arm A2 ${GPU:-2} --bias_wrap none   && echo A2_DONE ) &
( train_arm A3 ${GPU_B:-6} --bias_wrap broken && echo A3_DONE ) &
wait

$P eval/run_eval.py --set V_mat --size 16 --methods B2val v11dx p8Cx p8A1x p8A2x p8A3x \
   --out /tmp/p8_eval_Vmat_16.json || exit 1
echo P8_TRAIN_DONE
