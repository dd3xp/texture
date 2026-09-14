#!/usr/bin/env bash
# 32px 架构/训练制度臂：**把 32px 训练数据整个撤掉**（算力等量换成 16px），看 32px 输出是变好还是变差。
# 预注册：本脚本先提交再跑，跑完不许改判据。只动**验证集** V_mat。
#
# ---------------------------------------------------------------- 为什么是这条臂
# 现在 32px 只剩一条成立的事实：判官判 TRD 输 B2（41%，p=0.014，测试集）。它从未被解释过。
# 已全否的杠杆：推理侧八条（步数/加 32px 数据/重排/温度/检查点/代次/级联/Gibbs 精修）、
# 训练侧按包均衡 γ=0.5（打平）与 γ=1（试点不过门）。"换数据来源"这条 2026-09-14 也关了（供给不存在）。
# 账本因此写死了下一条臂的准入条件：**必须先有一个能解释"24px（零 32px 数据）为何比 32px（641 张）好"
# 的假设**，不许再拿"数据不够"起臂。
#
# 本臂的假设（与那个准入条件正面对上）：**32px 训练数据是净负的**。
#  - 直接反证据：24px 一张训练数据都没有，判官对 B2 打平（94/188 = 50%）；
#    32px 有 641 张，判官对 B2 输（83/202 = 41%）。
#  - 同向旁证：我们唯一一次**给 32px 加数据**（v11d，1236 张 64→32），32px 不但没好，反而更差；
#    而 `analysis/arch/data32_supply.py` 已证明那 1236 张里 91% 来自 `Sharpik__sharpnet_textures`
#    一个包，把 32px 池的有效包数（1/HHI）从 4.3 压到 2.2。
#  - 机制猜想：`model/trd.py:54-68` 的环面偏置只吃**按画布归一化**的偏移 (dy/n, dx/n)
#    ⇒ 架构本身是尺度等变的，16px 学到的结构先验本就能原样搬到 32px（24px 走的就是这条路）。
#    641 张几乎全来自两三个近乎平涂的大包 → 32px 那半个训练预算学到的是"这几个包长什么样"，
#    把本来够用的尺度等变先验**改坏了**。
# ⚠ "撤掉数据"和"加数据/调采样权重"是**反方向**的操作，不是又一个旋钮：
#    它测的是这批数据的**符号**（净正还是净负），这是此前从未被观察过的量。
#
# ---------------------------------------------------------------- 两条臂（唯一变量 = 32px 数据在不在）
#   A  nod32  ：从 runs/trd_v10/last.pt 接着训 8000 步，`--sizes 16`（一个 32px 样本都不喂）
#   B  ctrl0  ：从同一个检查点接着训 8000 步，`--sizes 16 32 --p32 0.5 --batch32 48`（其余逐字相同）
# 其余超参（lr 1.5e-4、batch 192、warmup 500、d/depth/heads、coarse、n_ex 4、extra_file）两臂一致，
# 与 `/tmp/runs/trd_v10pb05_09141130`（γ=0.5 那条臂）的续训口径相同。
# B 同时**补上了 `eval/val_judge_pb05.sh` 判据 (4) 点名要的 γ=0 对照臂**（v10 接着训 8000 步，其余不改），
# 此前一直缺这条对照。
#
# ⚠ 三处必须说清楚的口径限制（跑之前写下）：
#  (i)  **算力等量替换**，不是"只删数据"：A 的 8000 步全是 16px，B 是约一半 16px 一半 32px。
#       所以测的是"这半个训练预算花在 32px 数据上值不值"，正是实务上要做的那个取舍。
#  (ii) `eval/gen_trd.py:208` 的 `PaletteMemory(dev)` 默认 `sizes=(16,32)`，**直接读数据集、与 run 无关**
#       → 两臂的调色板检索库**完全一样**，A 照样拿得到 32px 真人调色板。
#       本臂撤掉的只是 32px 数据对**结构**的训练贡献。（16px 消融表已证承重的是调色板库，此处刻意不动它。）
#  (iii) 码本：`train_trd.py:270` 在有 `--init_from` 时直接载入源 run 的 codebook.npy
#       → 两臂码本逐位相同，不是混杂项。
#
# ---------------------------------------------------------------- 预注册判据（跑之前写死）
#  (1) 先试点（15 真题 + 5 对两边同图的空对照）：真题可解率 >= 65% 且高于空对照，才跑 full。
#      不过门槛 = **判官在这个比较上没有分辨力**，不报胜负、不下任何结论。
#      （这台仪器的地板是 16%，n=90，`analysis/arch/pilot_gate_audit.py`。）
#  (2) 主判据：A（nod32）胜率的二项双侧 p < 0.05 才算判出方向。
#  (3) **方向预测（写在跑之前）**：预测 **A 胜**。
#      - 显著胜 -> "32px 训练数据净负"在验证集上成立。**仍不换主配置**：下一步是另行预注册、
#        只跑一次的测试集观察（A vs B2 @32，E_mat）。
#      - 打平   -> 这条线关闭。**不许再试中间剂量**（p32=0.25 之类）——那就退回旋钮了。
#      - 显著输 -> 假设被证伪，32px 数据是承重的；这条线当场关闭，并且账本里
#        "24px 零数据反证"那条要降级为"与本臂结果不一致，机制未知"。
#  (4) **16px 不许退步**（它是唯一领先 B2 的一档）：两臂的 16px 验证损失一并报出来。
#      这是**报告要求，不是 32px 判据的闸门**——即便 A 的 16px 损失更差，(2)(3) 照常判读，
#      但 A 想进一步换主配置就必须先在 16px 上另立一条臂。
#  (5) **val32 损失不是判据，而且两臂根本没法比**：`--sizes 16` 时 `val32 = []`，
#      `train_trd.py:503` 直接把 log.json 的 `val32` 记成 null → A 这一栏是空的。
#      即便补算也不该当判据：val32 自己就被单个包主导（`analysis/arch/ref_pack_audit.py`）。
#  (6) **不看结构门/各向异性**：门不给 TRD 自家配置排序（温度轴上与判官反向 24pp），
#      它的"真人 32px"参照又被证明是一个包。本臂**只用判官**。
#  (7) 判官压缩：正结果作下界；零结果只能说"没测到大效应"。
#  (8) 操作检验（两条都要满足，否则本臂作废）：
#      - A 的 `config.json` 里 `"sizes": [16]`，且 log.json 每条记录的 `"val32"` 都是 null
#        （`train_trd.py:268/503`：`--sizes 16` 时 32px 的训练与验证池都是空的）；
#      - B 的 `config.json` 里 `"sizes": [16, 32]`、`"p32": 0.5`，log.json 的 `val32` 是数字。
#      两份 config.json 除 sizes/p32/batch32/out 外必须逐字段相同。
#
# ---------------------------------------------------------------- 运行约束
# `/mnt/data` 100% 满 → 检查点、生成图、判定 JSON **全部写 /tmp**（`scripts/sync_remote_tmp.sh` 会拉回本机）。
# GPU 全被占（含我们自己的 arch_b3 / arch_b3rev 两个 SD-piXL）→ 本脚本**自己排队**：
# 轮询到某张卡空出 MINFREE MiB 才开训，且 B3 在跑时**绝不碰 GPU 2 / 6**（别把它 12 小时的活 OOM 掉）。
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
STAMP=${STAMP:-$(date +%m%d%H%M)}
A=/tmp/runs/trd_nod32_$STAMP
B=/tmp/runs/trd_ctrl0_$STAMP
O=${OUT:-/tmp/gen32nod}
J=${JOUT:-/tmp/judge_nod32}
MINFREE=${MINFREE:-30000}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$O" "$J" /tmp/runs

pick_gpu() {
  local skip=""
  if ps -eo args | grep -q "[r]un_subset.py"; then skip="2 6"; fi
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits \
    | tr -d ' ' | while IFS=, read -r i used tot; do
        case " $skip " in *" $i "*) continue;; esac
        [ $((tot - used)) -ge "$MINFREE" ] && { echo "$i"; break; }
      done
}

wait_gpu() {
  local g=""
  while :; do
    g=$(pick_gpu)
    [ -n "$g" ] && { echo "$g"; return 0; }
    sleep 300
  done
}

COMMON="--init_from runs/trd_v10/last.pt --steps 8000 --batch 192 --lr 1.5e-4 --warmup 500 \
 --bias_freqs 8 --bias_hidden 128 --level_emb --pal_aug 0.3 --pal_smooth 0.1 \
 --extra --extra_file train_extra_packs_only.json --n_ex 4 --coarse --p_coarse 0.5"

G=$(wait_gpu); echo "[nod32] 训练 A 用 GPU $G"
CUDA_VISIBLE_DEVICES=$G $P -u model/train_trd.py --out "$A" --sizes 16 $COMMON || exit 1

G=$(wait_gpu); echo "[nod32] 训练 B 用 GPU $G"
CUDA_VISIBLE_DEVICES=$G $P -u model/train_trd.py --out "$B" --sizes 16 32 --p32 0.5 --batch32 48 $COMMON || exit 1

# 生成协议与正式测试的 TRD32 逐字相同：v10 谱系、跨模态检索 N=100、cfg 1.5、4 选 1
G=$(wait_gpu); echo "[nod32] 生成用 GPU $G"
export CUDA_VISIBLE_DEVICES=$G
for R in "$A:nod32x100" "$B:ctrl0x100"; do
  RUN=${R%%:*}; TAG=${R##*:}
  $P eval/gen_trd.py --run "$RUN" --ckpt last.pt --set V_mat --size 32 --n 4 --bs 8 \
     --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --tag "$TAG" --out "$O" || exit 1
  $P eval/rerank.py --src "$TAG" --set V_mat --size 32 --n 4 --root "$O" || exit 1
done

$P eval/judge_pairs.py pilot --a nod32x100_rr4 --b ctrl0x100_rr4 --set V_mat --size 32 \
   --root "$O" --outdir "$J" \
  && $P eval/judge_pairs.py full --a nod32x100_rr4 --b ctrl0x100_rr4 --set V_mat --size 32 \
   --root "$O" --outdir "$J"
echo VJ_NOD32_DONE
