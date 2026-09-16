#!/usr/bin/env bash
# (M35) 给位置偏置表加一支**固定像素周期**的梳齿 —— 唯一变量是 `--bias_pix 4`，控制组是 v11d。
#
# ===== 为什么是这一支（不是 (M29) 的解绑、也不是 (M13) 的加性格子支）=====
#   已发表的四块前提照旧：(M25) 3b0af3e 真人 32px 另有一支固定 4 像素成分；
#   (M26) a8ced52 表只吃 u=d/n，而**同一个 u 上两档要求相反符号**；
#   (M27) 5ec5556 / (M28) 67b8a04 那张表**确实是产物周期的把手**（24px 外推 + 训练过的 32px 干预）。
#   (M26)/(M27)/(M28) 反复写死的唯一授权动作 = "**另行预注册『替换/削弱归一化谐波支』的重训**"。
#
#   (M29)（4cdab69）用的是**按画布解绑**（零初始化 FiLM，输入 s=n/32）：被 16px 守门挡下，
#   `--bias_size_cond` 作废。⛔ 本轮**不重开 (M29)、不复审它的判决、不换一种解绑形式再试**：
#   本轮这一支**没有任何画布输入**，16/24/32 共用同一组权重 —— 它不是解绑，是给表**补上
#   归一化坐标表达不了的那一类函数**（"每 4 像素一道"，而不是"每张画布 k 个周期"）。
#   (M13)（4fa50da）的 `--bias_cells` 是 exp(-|d|/s) 的**衰减**特征，**表达不了梳齿** -> 那一种形式死了，
#   本支是周期特征，专门表达梳齿。
#
# ===== 改的是什么 =====
#   `model/trd.py::ToroidalBias(pix_periods=(4,))`：MLP 的输入**追加 4 维**
#   sin/cos(2*pi*w/4)（w = 环绕后的**有符号像素**偏移，y/x 各两维）。
#   * 可平铺 / 平移等变：P 整除 n 时该特征关于 w 以 n 为周期 -> 折回点 d=±n/2 单值连续。
#     P=4 整除本项目全部画布 16/24/32 -> 正是 (M29) 写死的"P 不整除 n 就不连续"的**例外**；
#     构造器对每个用到的 n 断言 n % P == 0（不整除时当场响，自检 (S4) 已验）。
#   * 零初始化：`train_trd.py` 的 `init_from` 里"旧列照抄、新列置零"那条现成代码路径
#     （原为 `--bias_cells` 写的）-> 第 0 步的偏置函数与源检查点**逐元素相同**。
#   本机跑前自检（CPU、零 GPU、`analysis/arch/pix_selfcheck.py`，5/5 全过，产物 /tmp/m35_selfcheck.json）：
#     (S1) 新列置零时 16/24/32 三档偏置与旧路径 **max|delta| = 0.0**（逐元素相同）；
#     (S2) 新列非零时，同一个 u=0.375 上 16px(d=6) 与 32px(d=12) 的增量**反号**
#          （+0.134 / -0.186）—— (M26) 那个冲突，不靠任何画布输入就能给出两个符号。
#          ⛔ (S2) 只判反号：MLP 带 GELU，哪一边为正由学出来的权重定，跑前指定方向等于凭空加要求。
#     (S3) 新列非零时三档"行列同步循环平移"残差 1.5e-07 / 3.2e-06 / 1.2e-07，与旧路径
#          6.0e-08 / 1.3e-06 / 6.0e-08 同量级 = 浮点噪声，可平铺性未破坏；
#     (S4) 折回点单值：P=4 时 |2 sin(pi n / P)| < 2e-15（三档），反对照 P=5 为 1.18/1.18/1.90
#          且断言确实会响（⛔ 不拿 g[n/2] 与 g[n-n/2] 比 —— 偶数 n 下那是同一个元素 = 什么也没量）；
#     (S5) 只多 4*hidden = 512 个参数。
#   另：`init_from` 的整条重映射在本机跑通，三档偏置表 max|delta| = 0.0，missing/unexpected 均为空。
#
# ===== 动机读数（**已发表**的 (M25)/(M26) 数字；⛔ 非本轮判据、⛔ 不许当结果引）=====
#   单独一个 P=4 的余弦在 d≡0 (mod 4) 为 +1、在 d≡2 (mod 4) 为 -1。真人局部对比 C(d)：
#     16px  d=2 -0.054  d=4 +0.056  d=6 -0.030  d=8 +0.073
#     32px  d=2 -0.052  d=4 +0.010  d=6 -0.022  d=8 +0.052
#           d=10 -0.016 d=12 +0.011 d=14 -0.019 d=16 +0.081
#   两档共 12 个偶数位置**全部同号** -> 一张共用的表加这一支就能同时服务两档：
#   u=0.375 在 16px 是 d=6（不是 4 的倍数 -> 压低）、在 32px 是 d=12（是 -> 抬高）。
#   ⚠ 这是**选候选的理由**，不是证据；本轮的证据只能是下面 ① ② ③ 的读数。
#
# ===== 判据（跑前写死、随本脚本一起提交；⛔ 一个字不许放宽）=====
#  ① **守门（交付不许退步）**：16px KID(m=2) 不得比控制臂 v11d 高出 **thr(2) = 2.419** 以上。
#     * 口径**只此一种**（(M33) 第五节的守门配方，`0bbba84`）：两臂都 `--n 2` 生成、
#       `run_eval.py --all_samples` 评测、两臂 `materials` 相同；门 2.419 = (M33) 在**三条 TRD seed 臂**
#       上实测的 `thr_max(2)`，同一把尺子。漂移项 `Dmax(2)=1.162` < 2.419 -> 采样项主导。
#     * ⛔ **这不是把 (M29) 的门放宽**：(M29) `GATE_FAILED` 判决、那道 ±1.0 的门、`--bias_size_cond`
#       作废 —— **一个字不动**，⛔ 本轮不许回头用 2.419 复审任何已判的臂（(M33) 预注册里就写死了）。
#       本轮是**新臂**，按 (M30)/(M33) 授权的"守门画在实测噪声之外"画门。
#     * 不过 -> **`GATE_FAILED`**，`--bias_pix` 作废、默认留关，②③**不下判**。
#     * 同时**报告**（⛔ 非判据）：m=1 口径的同一个差值，供与 (M29) 的 +2.118 并列登记。
#  ② **主判据（结构；对着 (M26) 点名的"小的那一支"画，与 (M29)② 逐字相同）**：
#       **ΔC(4) = C_new(4) - C_v11d(4) 的族级 95% CI 下界 > 0**
#       **且** C_new(4) 自身的族级 CI 下界 > 0（合取保护：纯衰减总体上 C(4) 恒为负，
#       正的绝对 C(4) 凸性造不出来 —— (M27)/(M28) 已实测标定）。
#     ⚠ **只用 d=4 一个位置**（不做多重比较）；C(12) 只作 (R3) 描述性读数，不进判据。
#  ③ **操作检验（跑前写死；任一条不过则 ② 作废，不下判）**：
#     (OP1) 新臂 A_hat(1) >= 控制臂的 0.8 倍；(OP2) flat_frac <= 控制臂 + 0.05；
#     (OP3) 配对 >= 500 对、>= 150 族；LOFO 零翻侧才不加 `_FRAGILE` 后缀。
#     (OP4) 两臂 16px 的 `materials` 必须相等（`run_eval.py:100` 的 ok 是各方法各算各的）。
#  ④ **次要（⛔ 不单独授权任何事、⛔ 不许替 ② 下判）**：32px 的 KID/FID/FD 三项里至少两项优于 v11d。
#  ⑤ **判决**（②③由 `analysis/arch/comb_learned.py` 出，那把尺子的识别检验 (M29) 已全过、本轮
#     **一个字不改**；它的 `UNBIND_WORKS`/`UNBIND_NULL` 在本轮读作"梳齿支**学到/学不到**那支成分"）：
#     ①过 且 ②过 且 ③全过                      -> **`PIXCOMB_WORKS`**
#     ①过 且 ΔC(4) 族级 CI 上界 < +0.0066       -> **`PIXCOMB_NULL`**：表补上了这类函数也补不出那一支
#                                                  -> 这条架构线判死、别再投钱。
#     其余                                       -> **`UNDECIDED`**（⛔ 不许往任何一边读）。
#  ⑥ **识别检验**：②③那把尺子的四条（ID1 长出来判 WORKS / ID2 没长出来判 NULL / ID3 退化判 OPS_FAILED /
#     ID4 在 (M28) 已存产物上复现 ΔC(4)=+0.0375）由 `comb_learned.py` 每次运行时重跑，**不过就不看真数据**。
#     ①那把尺子（`analysis/arch/m35_read_gate.py`）先 `--selftest` 再上真数据。
#  ⑦ **⛔ 本轮零判官、零 API**。即使 ② 判过，32px 准入条件（6c64796）②"不开任何 32px 判官新臂"
#     **仍然挡着对判**，那需要**另行**解除。本脚本不问判官，避免用胜率挑配置。
#  ⑧ **不动的东西**：`final_test.sh`、`UNITS_PER_TILE`、`judge_pairs.py`、`comb_learned.py`、
#     `noise_floor.py`、`sync_remote_tmp.sh`、任何默认值；`--bias_pix` 默认空 = 旧行为一个字不变。
#  ⑨ **不用 (M32) 的成对配方**（`--reseed_after_build`）：它要求控制臂也重训一次（v11d 是旧路径训的），
#     而 (M33) 已实测**漂移项不是主噪声源**（m=2：Dmax 1.162 vs thr 2.419）-> 那一小时买不到判读力。
#     因此本轮**不附** `paired_rng.py` 读数（那条要求只挂在**用**成对配方的预注册上）。
#
# ⚠ 本脚本**只出料**，①由 `m35_read_gate.py` 判、②③由 `comb_learned.py` 判。
# ⚠ 产物全写 /tmp（/mnt/data 97% 满）；命名已核对落在 scripts/sync_remote_tmp.sh 的白名单里
#    （runs/ / gen32* / trd_*.txt / eval_*Vmat*.json / m3[0-9]_*.json）。
#
# 起法：tmux new-session -d -s arch_pix "GPU=3 bash scripts/trd_pixcomb_train_eval.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
RUN=/tmp/runs/trd_pix4_09162048
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-3} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_pix

# ---- 训练：配方与 v11d/(M13)/(M29) 逐字相同，唯一差别是 --bias_pix 4 ----
$P -u model/train_trd.py --out $RUN --steps 12000 --lr 1.5e-4 --warmup 500 \
  --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
  --extra --extra_file train_extra_packs_only.json+train_64to32.json \
  --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
  --bias_pix 4 \
  --sizes 16 32 --p32 0.7 --save_at 6000 12000 > /tmp/trd_pix4.txt 2>&1 || exit 1

# ---- ① 守门：16px，生成口径逐字照搬控制组；两臂都是 --n 2，评测加 --all_samples（m=2 门 2.419）----
G="$P eval/gen_trd.py --run $RUN --ckpt last.pt --set V_mat --bs 8 --cfg 1.5 --pal_mode retrieve --xmodal"
$G --size 16 --n 2 --tag pixx || exit 1
$P eval/run_eval.py --set V_mat --size 16 --all_samples --methods v11dx pixx \
   --out $RUN/eval_pix_Vmat_16_m2.json || exit 1
# 报告项（⛔ 非判据）：同两臂的 m=1 读数，供与 (M29) 的 +2.118 并列登记
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v10x v11dx pixx \
   --out $RUN/eval_pix_Vmat_16.json || exit 1

# ---- ② 主判据的料：E_mat 32px 两臂（新臂 + v11d 控制臂），口径逐字照搬 (M28)/(M29) ----
OUT=/tmp/gen32_m35
C="$P -u eval/gen_trd.py --set E_mat --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --bs 8 --out $OUT --n 2"
$C --run $RUN          --ckpt last.pt --size 32 --tag m35_new   || exit 1
$C --run runs/trd_v11d --ckpt last.pt --size 32 --tag m35_ctl   || exit 1
# (R3) 描述性：(M27)/(M28) 明写的前置读数 —— 改表之后 24px 的梳齿应从 d=6 移到 d=4/8（⛔ 不进判据）
$C --run $RUN          --ckpt last.pt --size 24 --tag m35_new24 || exit 1
$C --run runs/trd_v11d --ckpt last.pt --size 24 --tag m35_ctl24 || exit 1

# ---- ④ 次要指标：32/24 两档，方法列表逐字照搬 (M13)/(M29) ----
for S in 32 24; do
  $G --size $S --n 2 --tag pixx_direct || exit 1
  $G --size $S --n 4 --ret_nname 100 --tag pixx100 || exit 1
  $P eval/rerank.py --src pixx100 --set V_mat --size $S --n 4 || exit 1
  $P eval/run_eval.py --set V_mat --size $S \
     --methods B1val B2val v11dx_direct v11dx100_rr4 pixx_direct pixx100_rr4 \
     --out $RUN/eval_pix_Vmat_$S.json || exit 1
done
echo PIX_DONE
