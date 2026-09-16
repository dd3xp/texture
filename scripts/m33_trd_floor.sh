#!/usr/bin/env bash
# (M33) 把采样噪声下限从 **v8 检索基线** 搬到 **TRD 臂自己** 身上。
#
# ===== 为什么要花这笔钱 =====
# 硬规矩①：守门必须画在噪声之外。但全项目一路在引的两个下限
#   m=1: KID 4.76 / m=4: KID 2.70（`experiments/noise_floor_Vmat.json`）
# 是在 **nf8_name / nf8_xpal（v8 检索基线）** 上量的 —— 那是**另一个方法族**。
# 今后每一条架构候选都是 **TRD 臂**。若 TRD 臂更吵，这道门就一直画得比真噪声紧
# （＝本项目第四次"守门画在噪声里"）；若更静，我们白白丢掉了判读力。
# 本轮把它在 TRD 臂上实测一次。顺带补上账本明写的缺口：**m=4 口径的重训漂移未知**
# （(M31) 的 Dmax=1.087 只对 m=1 有效 -> 账本⛔ 不许写"2.70 vs 1.087"）。
#
# ===== 三条臂（唯一变量仍是 --seed，与 (M31) 同一批）=====
#   seed0  runs/trd_v11d                  -> tag nfs0
#   seed1  /tmp/runs/trd_seed1_09161230   -> tag nfs1
#   seed2  /tmp/runs/trd_seed2_09161230   -> tag nfs2
# 生成配方与 `scripts/m31_drift_calib.sh:55` **逐字相同**，只改两处：
#   `--n 2` -> `--n 8`（噪声下限要求每材质 k=8 张）、`--out /tmp/nf_arms`（/mnt/data 随时会满）。
# ⛔ 不重训任何东西；三条 last.pt 都是 (M31) 已产出的、一个字节不动。
#
# ===== 判据（跑前写死，随本脚本一起提交；⛔ 一个字不许放宽）=====
#  ① **操作检验**（任一条不过 -> `OPS_FAILED`，不下判）：
#     (OP1) 三条臂 `config.json` 逐键比对，只允许 `seed` 不同（豁免表照抄 m31_read_drift）；
#     (OP2) 三条臂在 /tmp/nf_arms/<tag>/16 下各 **1000 png = 125 材质 x 8 张**，且每材质齐整 8 张；
#     (OP3) noise_floor 的 JSON 里三条臂齐全、m 键含 {1,2,4,8}，且 m<8 时 `draws` == 12。
#     ⚠ 判读器必须另报"已查几项"——`bad` 天然为空时不许读成"通过"（(M31) 盲写期抓到的 bug (a)）。
#  ② **主统计量 = TRD 臂上的 KID 守门门槛**：
#     每条臂在每个 m 下有 `sd_a(m)`（12 次抽样的标准差）。项目判据的门槛是 `2*sqrt(sd_a^2+sd_b^2)`。
#     记 **thr_max(m) = 三个臂对里的最大值**（保守取法，主判据），
#     并同时报 `thr_mean(m) = 2*sqrt(2)*mean(sd)`（次要）。
#  ③ **判决**（比较对象：v8 上的 m=1 KID **4.76**；m=4 一并报但不参与判决标签）：
#     r = thr_max(1) / 4.76
#       r > 1.10 -> **`FLOOR_UNDERSTATED`**：过去的门比真噪声紧
#                   -> 今后守门下限改用本轮 TRD 实测值（用哪个 m 就用哪个 m 的 thr_max）。
#                   ⛔ 不重开任何已下判决：(M29) 的 +2.118 本来就落在 4.8 之内，下限变大只会更落在里面。
#       r < 0.90 -> **`FLOOR_OVERSTATED`**：v8 的数偏保守 -> **仍可继续引用**（保守方向是安全的）；
#                   今后可改用 TRD 数把门收紧，但新臂预注册里**必须显式写明用的是哪一个数**。
#                   ⛔⛔ 不许回头用更松的门去复审任何**已判**的臂（(M29)/(M31) 判决一个字不改）。
#       否则     -> **`FLOOR_AGREES`**：2.70/4.76 沿用，并首次获得"在 TRD 臂上也成立"的凭据。
#  ④ **报告项 = m=4 口径的重训漂移**：三条臂 KID 均值两两绝对差的最大值 `Dmax(m)`，m=1,2,4 各一个。
#     ⚠⚠ 本轮的 `Dmax(1)` **不是** (M31) 的 1.087：估计量不同（这里每臂是 12 次抽样的**均值**，
#     (M31) 是 `--n 2` 里固定取第 0 张的**单次**读数）-> ⛔ 不许说"复现/未复现 (M31)"、
#     ⛔ 不许拿本轮数字去替换 (M31) 判决里的任何读数。
#     ⚠ n=3 个差值，**只报区间不做检验**（⛔ 不许对 3 个数算 p 值）。
#     用途（只此一条）：今后在 m=4 上守门时，门画在 `max(thr_max(4), Dmax(4))` 之外；
#     **除非**用 (M32) 成对配方（--init_from + --reseed_after_build）把漂移项直接消掉。
#  ⑤ ⛔ **本轮不比较任何两条配置的优劣**——三条臂只差 `--seed`，本来就无优劣可言。
#     ⛔ 不授权任何新臂（每条新臂仍要自己的预注册）、⛔ 不放宽 32px 准入条件①②③。
#  ⑥ ⛔ **零判官、零 API**（噪声下限是指标层面的量，不问判官）。
#  ⑦ **不动的东西**：`noise_floor.py`、`gen_trd.py`、`run_eval.py`、`train_trd.py`、`final_test.sh`、
#     `UNITS_PER_TILE`、`judge_pairs.py`、任何默认值、已下的任何判决
#     （含 (M29) `GATE_FAILED`、(M31) `DRIFT_EXCEEDS_GATE`、(M32) `PAIRED_OK`）。
#
# 起法：tmux new-session -d -s arch_floor "GPU=2 bash scripts/m33_trd_floor.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_floor
G=${GPU:-2}
OUTDIR=/tmp/nf_arms                       # ⛔ 不写 /mnt/data：14T 已 100%

gen_one() {                               # $1=tag  $2=run目录
  CUDA_VISIBLE_DEVICES=$G $P eval/gen_trd.py --run $2 --ckpt last.pt --set V_mat --size 16 --bs 8 \
    --n 8 --cfg 1.5 --pal_mode retrieve --xmodal --tag $1 --out $OUTDIR || return 1
}

gen_one nfs0 runs/trd_v11d                || { echo "nfs0 FAILED"; exit 1; }
gen_one nfs1 /tmp/runs/trd_seed1_09161230 || { echo "nfs1 FAILED"; exit 1; }
gen_one nfs2 /tmp/runs/trd_seed2_09161230 || { echo "nfs2 FAILED"; exit 1; }

CUDA_VISIBLE_DEVICES=$G $P -u eval/noise_floor.py --set V_mat --size 16 \
  --methods nfs0 nfs1 nfs2 --root $OUTDIR --k 8 --subsets 1 2 4 \
  --out /tmp/m33_noise_floor_Vmat_16.json || exit 1
echo FLOOR_DONE
