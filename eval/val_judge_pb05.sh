#!/usr/bin/env bash
# 32px 训练侧第一条臂：**按包均衡采样 γ=0.5** vs 现配置。验证集 V_mat，同材质直接对判。
# 预注册：本脚本先提交再跑。
#
# 背景（docs/arch_progress.md 2026-09-12 / 09-14 各节）：
#  - 32px 现在唯一成立的事实是**判官判 TRD 输 B2（41%，p=0.014）**；"真人 32px 有结构而 TRD 塌了"
#    那个说法已经撤回（那个参照组 89% 来自同一个平铺包）。结构门**不给 TRD 自家配置排序**
#    （温度轴上与判官反向 24pp），所以本臂**只用判官，不看门**。
#  - 推理侧六条杠杆全否（步数/加 32px 数据/重排/温度/检查点/代次），上一轮又否了级联与 Gibbs 精修
#    → 只剩训练侧。32px 训练池 641 张 / 24 包，`dungeonsoup` 220 + `macrotex` 186 两个近乎平涂的大包占 63%；
#    `train_trd.py --pack_balance 0.5` 把 32px 采样权重取 ∝ 包大小^-0.5，两者占比降到约 25%。
#  - 被比的 A：`/tmp/runs/trd_v10pb05_09141130`（从 runs/trd_v10/last.pt 接着训 8000 步，γ=0.5，
#    16px 验证损失回到与 v10 相同的 6.869，val32 5.255→5.203）。
#  - 生成协议与 B 逐字相同（v10，跨模态 N=100，CLIP-B/16 4 选 1，cfg 1.5，`--ckpt last.pt`）；
#    `--ckpt last.pt` 的口径已单独验过对 32px 无影响（ckpt32_probe，四对全不显著）。
#
# **预注册判据（跑之前写死）**：
#  (1) 先试点（15 真题 + 5 对两边同图的空对照），真题可解率 >= 65% 且高于空对照，才跑 full；
#      不过门槛 = **判官在这个比较上没有分辨力**，不报胜负、不作任何结论。
#  (2) 主判据：A（pb05）胜率的二项双侧 p < 0.05 才算判出方向。
#  (3) **方向预测（写在跑之前）**：预测 pb05 胜。
#      - 显著胜 -> 按包均衡成立，但**还不能换主配置**，见 (4)；
#      - 打平 -> γ=0.5 这一档关闭；**最多再试一档 γ=1**（须另行预注册），那一档再不成立则
#        "调训练池采样权重"整条线关闭；
#      - 显著输 -> 整条线当场关闭，不试 γ=1。
#  (4) **混杂，必须记住**：pb05 = v10 + 8000 步 + 按包均衡，**多训的 8000 步没有对照**。
#      旧证据（ckpt32_probe：相隔 10000 步的 best/last 四对全不显著）指向"多训不改 32px"，
#      但那是用结构门量的，而门已被证明不给配置排序 -> **只要 pb05 赢了，换主配置之前必须先跑
#      一条 γ=0 的对照臂（v10 接着训 8000 步，其余一字不改）**。
#  (5) 这只动验证集。据此换 32px 主配置再测测试集 = 对测试集的第二次观察，须另行预注册、只跑一次。
#  (6) ⚠ 判官压缩效应：正结果作下界；负结果只能说"没测到大效应"。
#
# `/mnt/data` 只剩十几 G 且随时会被别人写满 -> 生成、重排、判定 JSON **全部写 /tmp**，
# 判官的两个目录都放在 /tmp/gen32pb 下（把仓库里的 B 拷过去），跑完 scp 回本机入库。
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
set -u
P=${PY:-python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
RUN=${RUN:-/tmp/runs/trd_v10pb05_09141130}
O=${OUT:-/tmp/gen32pb}
J=${JOUT:-/tmp/judge_pb05}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
mkdir -p "$O" "$J"

$P eval/gen_trd.py --run "$RUN" --ckpt last.pt --set V_mat --size 32 --n 4 --bs 8 \
   --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --tag pb05x100 --out "$O" || exit 1
$P eval/rerank.py --src pb05x100 --set V_mat --size 32 --n 4 --root "$O" || exit 1
cp -r "$REPO/experiments/baselines/v10x100_rr4" "$O/" || exit 1

$P eval/judge_pairs.py pilot --a pb05x100_rr4 --b v10x100_rr4 --set V_mat --size 32 \
   --root "$O" --outdir "$J" \
  && $P eval/judge_pairs.py full --a pb05x100_rr4 --b v10x100_rr4 --set V_mat --size 32 \
   --root "$O" --outdir "$J"
echo VJ_PB05_DONE
