#!/usr/bin/env bash
# (M31) 标定「同配置重训漂移」—— (M30) `RNG_DIVERGES` 明确授权的那一件事，也是解开架构侧堵点的前提。
#
# ===== 为什么必须先花这笔钱 =====
# (M29) 判 `GATE_FAILED`：新臂 16px KID 8.316 vs 控制臂 v11d 6.198，差 +2.118 > 门槛 +1.0。
# (M30) 证明两臂其实是**两次独立训练**（`nn.Linear` 构造从 CPU 全局 RNG 抽数，取批用同一条流
# -> 第 0 步就分叉、200/200 步批次下标不同），而"同配置重训"的漂移量本项目**从未标定过**
# （`eval/noise_floor.py` 只量采样噪声，不含训练随机性）。
# -> 在标定它之前，**任何**"改架构 + 重训 + 对控制臂守门"的预注册都可能在量噪声。本轮就标定它。
#
# ===== 两条新臂（唯一变量 = --seed；配方与 runs/trd_v11d 逐字相同）=====
#   D1  --seed 1   /tmp/runs/trd_seed1_<stamp>
#   D2  --seed 2   /tmp/runs/trd_seed2_<stamp>
# 控制臂 = 已存的 runs/trd_v11d（= seed 0，那一行原本写死 0）。
# `--seed` 本轮新加，**默认 0 -> 旧路径逐字节不变**（`model/train_trd.py:277`）。
#
# ===== 判据（跑前写死，随本脚本一起提交；⛔ 一个字不许放宽）=====
#  ① **操作检验**（任一条不过 -> `OPS_FAILED`，不下判）：
#     (OP1) 两条新臂的 `config.json` 与 v11d 逐键比对，**只允许 `seed` 这一个键不同**
#           （后加的默认键 `bias_cells=[] / p_tile16=0.0 / pack_balance=0.0 / bias_size_cond=False` 视为相同）；
#     (OP2) 三条臂都真的跑到 12000 步且 `last.pt` 存在；
#     (OP3) 生成/评测口径与 (M29)① 逐字相同（16px `--n 2`，`run_eval.py --methods B2val v10x v11dx <新臂>`）。
#  ② **主统计量 = 三条臂两两之间 16px KID 的绝对差**（seed0-1、seed0-2、seed1-2 共 3 个），
#     记 **Dmax = max|ΔKID|**。⚠ n=3 个差值，**只报区间不做检验**（⛔ 不许对 3 个数算 p 值）。
#  ③ **判决（只约束今后的预注册，⛔ 不重新裁决任何已下的判决）**：
#     Dmax >= 1.0  -> **`DRIFT_EXCEEDS_GATE`**：±1.0 的守门落在漂移之内
#                     -> 今后守门必须画在 **max(Dmax, 噪声下限 4.8)** 之外，或改用同种子成对设计。
#     Dmax <  1.0  -> **`DRIFT_WITHIN_GATE`**：±1.0 守门在漂移之外，今后可续用（仍受采样下限约束）。
#  ④ ⛔⛔ **本轮不重开 (M29)**：无论 Dmax 多大，(M29) 的 `GATE_FAILED` 与"`--bias_size_cond` 作废、
#     默认留关"**一个字不改**。理由：那是**决策**（跑前写死的门挡下了改动），而"这条改动让 16px 变差"
#     这个**推断**本项目从来没有下过（(M29) 当轮就写明 +2.118 落在噪声下限内、只能读"被守门挡下"）。
#     ⛔ 不许把本轮结果读成"(M29) 其实通过了"、⛔ 不许据此重跑或换个解绑形式再试。
#  ⑤ **次要（报告项，不授权任何事）**：三条臂的 FID/FD/CLIP，以及 32px 的 KID/FID/FD（口径同 (M29)④）。
#     ⚠ 它们各自的漂移也一并报出来，供今后画守门用；⛔ 不许拿来给任何配置排序。
#  ⑥ **不动的东西**：`final_test.sh`、`UNITS_PER_TILE`、`judge_pairs.py`、`noise_floor.py`、
#     任何默认值（`--seed` 默认 0 = 旧行为）、32px 准入条件①②③、已下的任何判决。
#  ⑦ ⛔ **本轮零判官、零 API**（漂移是指标层面的量，不问判官；问判官等于用胜率挑配置）。
#
# 起法：tmux new-session -d -s arch_drift "GPU=6 GPU2=2 bash scripts/m31_drift_calib.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_drift
STAMP=09161230

train_one() {                                  # $1=seed  $2=gpu
  RUN=/tmp/runs/trd_seed$1_$STAMP
  CUDA_VISIBLE_DEVICES=$2 $P -u model/train_trd.py --out $RUN --steps 12000 --lr 1.5e-4 --warmup 500 \
    --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
    --extra --extra_file train_extra_packs_only.json+train_64to32.json \
    --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
    --seed $1 \
    --sizes 16 32 --p32 0.7 --save_at 6000 12000 > /tmp/trd_seed$1.txt 2>&1 || return 1
  CUDA_VISIBLE_DEVICES=$2 $P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --size 16 --bs 8 \
    --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --tag seed$1x || return 1
  for S in 32 24; do
    CUDA_VISIBLE_DEVICES=$2 $P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --size $S --bs 8 \
      --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --tag seed$1x_direct || return 1
  done
}

train_one 1 ${GPU:-6} || { echo "seed1 FAILED"; exit 1; }
train_one 2 ${GPU2:-${GPU:-6}} || { echo "seed2 FAILED"; exit 1; }

# 评测：16px 是判据②的料，32/24 是判据⑤的报告项
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v10x v11dx seed1x seed2x \
   --out /tmp/eval_drift_Vmat_16.json || exit 1
for S in 32 24; do
  $P eval/run_eval.py --set V_mat --size $S --methods B2val v11dx_direct seed1x_direct seed2x_direct \
     --out /tmp/eval_drift_Vmat_$S.json || exit 1
done
echo DRIFT_DONE
