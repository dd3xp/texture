#!/usr/bin/env bash
# (M39) **零 API 的跑前筛子**：把下一个候选杠杆 `--ret_nname` 的"产物差别"放到一把已经用判官标定过的尺子上，
# 再决定值不值得花 API。预注册：本文件（连同判读器 analysis/arch/m39_diff_meter.py）先提交再跑，跑完不许改判据。
# **本轮零训练、零 API**，只有三次 16px 生成。
#
# ================================================================ 一、为什么是这条、为什么先筛不先判
# (M38) 结账时写下了一条**可复用的跑前判据**（账本 2026-09-17 03:15 「下一步」第 2 条）：
#   「若挑战者与现行配置的产物差别小于"检查点那一档"（step 3000 vs 20000），判官在 n≈125 材质上
#     很可能又是 `_NULL`。下一轮选杠杆时先估这个差别（可以先零 API 地量两臂逐像素差异 / 分布指标差，
#     再决定值不值得花 API）。」
# 本文件就是**兑现那句话的第一次**：先花 GPU 不花 API，量出差别，再按预注册的门决定下一轮开不开判官臂。
#
# 为什么候选是 `--ret_nname`：
#  (a) 它是 `final_test.sh` 里第三个"当年按分布指标定、从没被现行判官验过"的格子
#      （`final_test.sh:19,21,23`：TRD16 用 30 / TRD16c 用 100 / TRD24 用 300 / TRD32 用 100 —— 同一条管线
#      上四个不同的数，本身就说明它是**调出来的**而不是推出来的）。
#  (b) 它是**唯一已证承重的部件**上的旋钮：(M18) `AB_pal` 判定四个部件里只有「调色板记忆库」承重
#      （126/190=66%，族级 [.574,.747]，`W1_robust`）。`gen_trd.py:108,161` 里 `--ret_nname` 正是
#      "先按名字取这么多条候选，再按图文相似度挑调色板"的那个 N => 它直接决定承重部件吃到什么。
#  (c) 零训练：纯推理参数，模型在盘。
#
# ⚠ **本轮不问好坏，只问"有没有差别"**。⛔ 本文件产出的任何数字都**不是**质量证据、
#   ⛔ 不许用来说哪个 N 更好、⛔ 不许写进任何主张。它只有一个用途：**决定下一轮花不花 API**。
# ⛔ 不碰 32px 准入条件①②③④，不碰任何已下的判决（含 (M36) `PIXCOMB_NULL`、(M37) `CKPT_LATE_WINS`、
#   (M38) `CFG_NULL`），⛔ `final_test.sh` 一个字不动（本轮**没有任何**改它的授权）。
#
# ================================================================ 二、设计（三条臂，唯一变量是 --ret_nname）
#   ret100 = 现行 TRD16c 的那个 N（基准，也是**复现锚点**，见 (OP-D)）
#   ret30  = TRD16 用的那个 N（挑战者一）
#   ret300 = TRD24 用的那个 N（挑战者二）
# 其余逐字相同，照抄 `final_test.sh` 的 TRD16c 配方 + (M38) 的 COMMON：
#   --run runs/trd_v8 --ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --pal_mode retrieve --xmodal --cfg 1.5
#   + `eval/rerank.py --n 4`。种子默认 0，三臂相同。
# ⚠ 必须是验证集 V_mat（这是挑配置的前戏，测试集只许最后看一次）。
#
# ================================================================ 三、预注册判据（跑之前写死；判读器同时盲写）
# 尺子 d(A,B) = 125 个材质上「A、B 的 `_rr4` 瓦片逐像素不同的比例」的均值（0..1，与灰度无关、可跨臂比）。
# 三个锚点全部由**已判过的臂**提供，判读器 `analysis/arch/m39_diff_meter.py` 一次算齐：
#   D37 = d(ck3k_rr4, ck20k_rr4)   <- (M37) 判官判**显著**（26/79=32.9%，族级 [0.217,0.438]，`W1_robust`）
#   D38 = d(cfg25_rr4, cfg15_rr4)  <- (M38) 判官判 **`CFG_NULL`**（42/70=60%，族级 [0.4827,0.6970] 含 0.5）
#   D0  = d(ret100_rr4, cfg15_rr4) <- **零点**：两条臂的命令逐字相同（cfg15 那条本来就是 --ret_nname 100 --cfg 1.5）
#
#  (C1) **标定门一**：必须 D37 > D38。否则这把尺子和判官的两次判决**不同向** -> `METER_UNCALIBRATED`，
#       本轮**不出 GO/NO-GO**，只记录"尺子不成立"。（这是本轮唯一可能自证失败的地方，写在前面。）
#  (C2) **标定门二（零点）**：必须 D0 <= D38/10。若管线不确定、同样命令跑出不同图，则 d 里混着
#       与变量无关的噪声 -> 同样 `METER_UNCALIBRATED`。⚑ 顺带这是对"生成是否确定性"的一次实测记录。
#  (V)  两条挑战者各自判（c ∈ {ret30, ret300}，d_c = d(c_rr4, ret100_rr4)）：
#       - d_c >= D37            -> **`SCREEN_GO(c)`**：产物差别不小于"检查点那一档" => **授权下一轮**
#                                  为 c 另行预注册一条 16px/V_mat 判官臂（⛔ 授权仅限"可以去预注册"，
#                                  ⛔ 不代表 c 更好、⛔ 不许跳过预注册直接跑）。
#       - d_c <= D38            -> **`SCREEN_NOGO(c)`**：差别不大于"CFG 那一档"，而那一档判官已实测判不出
#                                  => ⛔ 本轮起不许在 c 上花 API，除非另行论证。
#       - D38 < d_c < D37       -> **`SCREEN_AMBIGUOUS(c)`**：两锚点之间，**默认不花 API**。
#  (P)  **两条都 GO 时只许提拔一条**：取 d 大的那条。这是**设计选择不是证据**，且下一轮仍须完整预注册。
#  (5)  **方向预测（跑前写下）**：预测 **两条都 `SCREEN_GO`**。理由：`--ret_nname` 改的是承重部件
#       （调色板记忆库）吃到的候选池，池子大小 30 vs 100 vs 300 差 3.3 倍 => 多半换出不同调色板 =>
#       整块瓦片的颜色都变，逐像素差别应当大于"同一调色板下改 CFG"。**若出现 NOGO，这条预测当场记为被推翻**，
#       并且要连带承认：`final_test.sh` 里那四个不同的 N **在产物上根本分不开**。
#  (6)  操作检验（任一条不过，本轮作废）：(OP1) 三个 tag 各 500 张、三个 `_rr4` 各 125 张；
#       (OP2) 三条命令逐字打印备查，唯一差异是 `--ret_nname`；(OP-D) 零点门见 (C2)；
#       (OP5) `run_eval.py` 三臂的 `n`/`materials` 相等。
#  (7)  次要报告项（⛔ 非判据）：三臂的 16px KID/FID/FD/CLIP，对照 (M33) 的 m=1 下限
#       KID 4.656 / FID 6.0 / FD 20 / CLIP 0.41，小于下限一律读作"没测到"。⛔ 不许拿它挑 N。
#  (8)  本轮不改任何默认值：`gen_trd.py`/`rerank.py`/`run_eval.py`/`judge_pairs.py`/`final_test.sh` 全不动。
#
# ================================================================ 四、运行约束
# `/mnt/data` 长期贴满 -> 产物全写 `/tmp`；指标 JSON 显式命名 `/tmp/m39_retnn16.json`、
# 判读 JSON `/tmp/m39_retnn_screen.json`，落在 `scripts/sync_remote_tmp.sh` 的 `m3[0-9]_*.json` 白名单里。
# 选卡只看 `memory.free`（16px 生成实占约 5GB）。起法（命令必须先写进文件再跑）：
#   tmux new-session -d -s arch_retnn "bash /tmp/run_retnn.sh"
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
O=${OUT:-/tmp/retnn16}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
mkdir -p "$O"

COMMON="--run runs/trd_v8 --ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --pal_mode retrieve --xmodal --cfg 1.5 --out $O"

"$P" -c 'import numpy, PIL, torch, sys; sys.exit(0)' || { echo "[retnn16] python 探针失败: $P"; exit 1; }
# (OP2)：三条命令原样打印，唯一差异必须是 --ret_nname
for N in 30 100 300; do echo "[retnn16] eval/gen_trd.py --ret_nname $N $COMMON --tag ret$N"; done
for N in 30 100 300; do
  $P eval/gen_trd.py --ret_nname $N $COMMON --tag ret$N || exit 1
done
for N in 30 100 300; do
  $P eval/rerank.py --src ret$N --set V_mat --size 16 --n 4 --root "$O" || exit 1
done
# (OP1)：张数
$P - "$O" <<'PYEOF' || exit 1
import sys, pathlib
root = pathlib.Path(sys.argv[1]); bad = []
for t, want in [("ret30", 500), ("ret100", 500), ("ret300", 500),
                ("ret30_rr4", 125), ("ret100_rr4", 125), ("ret300_rr4", 125)]:
    n = len(list((root / t / "16").glob("*.png")))
    print("[OP1] %-11s %d 张 (应为 %d)" % (t, n, want))
    if n != want:
        bad.append("OP1:%s=%d" % (t, n))
print("[操作检验] 已查 %d 项，problems=%d %s" % (6, len(bad), bad))
sys.exit(1 if bad else 0)
PYEOF
$P eval/run_eval.py --set V_mat --size 16 --root "$O" --out /tmp/m39_retnn16.json \
   --methods ret30_rr4 ret100_rr4 ret300_rr4 ret30 ret100 ret300 || exit 1
echo RETNN16_GEN_DONE
