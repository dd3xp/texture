#!/usr/bin/env bash
# 消融表（16px，测试集 E_mat，272 材质 / 857 张真人瓦片）。**预注册：本文件先提交再跑，跑完不许改判据。**
#
# 参照臂 = `TRD16c_rr4`（v8 + 跨模态检索 N=100 + CLIP-B/16 4 选 1），即 `eval/final_test.sh` 里
# 与 B2 同预算的那条主配置；它**已经在测试集上出过图**（`experiments/baselines/TRD16c_rr4/16/` 272 张），
# 本脚本不重出，直接拿来当对照。
#
# ⚠ 这**确实是**在测试集上多看了一眼（判官那步尤其是）。之所以还做：消融的用途是**解释**已定的配置，
# 不是挑配置。**跑前写死**：无论结果如何，`final_test.sh` 的主配置**一个字不改**；
# 想按消融结果换配置，必须回验证集重选、再另行预注册一次测试集观察。
#
# 每条臂只从参照臂里**去掉一个部件**，其余逐字相同（cfg 1.5、n 4、bs 16、种子默认 0）：
#   AB_pal  调色板记忆库关   `--pal_mode model`（模型自出调色板，不检索真人调色板）
#   AB_ex   结构范例关       `--no_ex`
#   AB_xm   跨模态检索关     去掉 `--xmodal` 与 `--ret_nname`（退回纯按材质名检索，N=30）
#   AB_ct   解码顺序换回惯例 `--choice_temp 4.5`（MaskGIT 默认；我们用 20）
#   AB_rr   4 选 1 关        = 已有的 `TRD16c` 取第 0 张，**无需重出**
#
# ⚠ **环面位置编码那条臂做不了，本轮不做**（不是忘了）：`runs/trd_v1` 的 `bias_freqs=1` 确实是单频，
# 但它与 v2 还差着 `bias_hidden` 64/128、`level_emb` false/true、`pal_aug` 0/0.3、`pal_smooth` 0/0.1、
# `sizes` [16]/[16,32] —— **五处混杂**，拿 v1 当"单频臂"会把另外四项的功劳算到位置编码头上。
# 要做只能另训一版「v8 除 bias_freqs=1 外一字不改」，那是一次完整训练，另行预注册。
#
# **预注册判据（跑之前写死）**
# (1) 表格口径：CLIP-B/32 与 KID/FID 全部用 `eval/run_eval.py`（分布指标取每材质第 0 张，与已发表口径一致）。
#     噪声下限（`eval/noise_floor.py`，m=1）：KID 4.8 / FID 6.0 / CLIP 0.41。**差值小于下限的一律读作"没测到"**，
#     不许因为方向对就说某部件有用。
# (2) 判官是主尺子（CLIP 只用来定序，FD 已退出优化目标）：**五条臂全部**与参照臂做同材质配对判官，
#     顺序写死为 AB_pal → AB_ex → AB_xm → AB_ct → AB_rr，**不看第 (1) 步结果再决定跑哪几条**（防挑臂）。
#     每条先试点：15 对真题 + 5 对两边同图的空对照，真题跨序一致率 >= 65% 且高于空对照才跑全量；
#     不过门槛就记"判官分辨不了这个部件"，不是"这个部件没用"。
# (3) 单条臂的判读：参照臂胜率二项双侧 p < 0.05 才算该部件承重。**五条臂不做多重比较校正**，
#     故任何单条 0.01 < p < 0.05 的结果都只写作"提示性"，不进摘要。
# (4) 方向预测（先写下来）：AB_pal、AB_xm 参照臂胜（这两条在验证集上选配置时效应最大）；
#     AB_ct 参照臂胜（验证集 KID 12.8→5.5）；AB_ex、AB_rr 无预测。
# (5) 判官压缩效应：正结果作下界，负结果只能说"没测到大效应"。
# (6) 成本上限：五条臂 × (20 试点 + 至多 272 全量) 次 API。任一条臂 API 失败率 > 25% 就停下记账，不补跑。
#
# 产物全写 /tmp（`/mnt/data` 100% 满）；跑完把 JSON scp 回本机入库。凭据只从环境变量读。
# 起法：tmux new-session -d -s arch_abl "VLM_BASE_URL=... VLM_API_KEY=... CUDA_VISIBLE_DEVICES=<空卡>
#        PY=<jzs_train python> bash /tmp/ablation_16.sh > /tmp/abl16.txt 2>&1"
# 分两段：不给 STAGE 或 STAGE=gen 只出图算指标（零 API）；STAGE=judge 只跑判官（零 GPU 之外的训练）。
set -u
P=${PY:-python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
O=${OUT:-/tmp/abl16}
J=${JOUT:-/tmp/judge_abl16}
STAGE=${STAGE:-gen}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
mkdir -p "$O" "$J"

V8=runs/trd_v8
G="$P eval/gen_trd.py --run $V8 --ckpt last.pt --set E_mat --size 16 --n 4 --bs 16 --cfg 1.5 --out $O"

if [ "$STAGE" = gen ]; then
  # 参照臂与 AB_rr 直接复用已有产物，不重出
  for d in TRD16c TRD16c_rr4; do cp -r "$REPO/experiments/baselines/$d" "$O/" || exit 1; done

  $G --pal_mode model    --xmodal --ret_nname 100 --tag AB_pal || exit 1
  $G --pal_mode retrieve --xmodal --ret_nname 100 --no_ex --tag AB_ex  || exit 1
  $G --pal_mode retrieve                          --tag AB_xm  || exit 1
  $G --pal_mode retrieve --xmodal --ret_nname 100 --choice_temp 4.5 --tag AB_ct || exit 1
  for t in AB_pal AB_ex AB_xm AB_ct; do
    $P eval/rerank.py --src $t --set E_mat --size 16 --n 4 --root "$O" || exit 1
  done
  $P eval/run_eval.py --set E_mat --size 16 --root "$O" --out /tmp/abl16_metrics.json \
     --methods TRD16c_rr4 AB_pal_rr4 AB_ex_rr4 AB_xm_rr4 AB_ct_rr4 TRD16c || exit 1
  echo ABL16_GEN_DONE
fi

if [ "$STAGE" = judge ]; then
  for A in AB_pal_rr4 AB_ex_rr4 AB_xm_rr4 AB_ct_rr4 TRD16c; do
    $P eval/judge_pairs.py pilot --a TRD16c_rr4 --b "$A" --set E_mat --size 16 --root "$O" --outdir "$J" \
      && $P eval/judge_pairs.py full --a TRD16c_rr4 --b "$A" --set E_mat --size 16 --root "$O" --outdir "$J"
  done
  echo ABL16_JUDGE_DONE
fi
