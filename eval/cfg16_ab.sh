#!/usr/bin/env bash
# (M38) 第二条"用现行判官复查一个从没被判官验过的配置选择"的臂：**主配方的 CFG 到底该取多少**
# （16px、验证集 V_mat、同材质直接对判）。预注册：本文件先提交再跑，跑完不许改判据。零训练。
#
# ================================================================ 一、为什么是这条
# (M37) 刚刚证明：`final_test.sh` 里一个从未被现行判官验过的选择（检查点）**是可以被判出显著差别的**，
# 而且当年替我们做这个选择的那把尺子（val 损失）指的是**反方向**。同一张表上还坐着第二个选择，
# 它的来历比检查点那条更脆：
#
#  (a) **CFG=1.5 是在 v2 上按 FID 差 1–2 分选的**（账本 2026-09-11「v2 最终配置」一节，原话
#      "**FID 最好，KID/FD 接近最好**"）。那张表 step20000 的四档是：
#        CFG 1.0 / 1.25 / 1.5 / 2.0 -> KID 24.9 / 28.6 / 25.7 / 26.1、FID 82.2 / 83.2 / 81.0 / 83.1、
#        FD 112.4 / 107.9 / 105.9 / 106.6、CLIP 32.56 / 32.54 / 32.50 / 32.29。
#      对照 (M33) 在 TRD 臂上实测的 m=1 噪声下限（KID **4.656** / FID **6.0** / FD **20** / CLIP **0.41**），
#      **这四档两两之差没有一项过得了下限** => 当年那次选择**整个发生在噪声里**。
#  (b) **选它的那个模型不是出货的那个**。v2 的训练数据后来被判为**被污染**（账本原话：v2/v3 不能上测试集），
#      而且 v2 期**还没有跨模态检索**（`--xmodal`，v7 才加）。现在出货的是 **v8 + xmodal + ret_nname 100**。
#  (c) **在出货那套检索下，账上唯一一次 1.5 vs 2.5 的对照反而偏向 2.5**（2026-09-11「跨模态检索」一节）：
#        v7+xmodal CFG 1.5 -> KID 7.9 / FID 67.2 / FD 97.6 / CLIP 34.74
#        同上      CFG 2.5 -> KID 7.3 / FID 66.3 / FD 89.4 / CLIP 34.73
#      三项分布指标都更好 —— 但**三项之差也全在噪声下限之下**（0.6 / 0.9 / 8.2），所以它**什么也没证明**。
#      而更早那次**名字检索**（无 xmodal）的扫描方向相反（1.5 -> 2.5：KID 8.8->11.2、FID 67.0->71.6、FD 84.1->91.9，
#      同样**全在下限之下**）。两次方向相反、两次都测不到 => 这是一个**真正未定**的岔路口，不是已判的。
#  (d) **上界是画过的**：同一张表里 CFG **3.5 / 5.0** 的 KID 是 18.0 / 23.6，对 1.5 的 8.8 差 9.2 / 14.8，
#      **远过下限** => 高 CFG 走到 3.5 以上是真的差。所以本臂只问 **2.5**，⛔ 不扫 3.5/5。
#  (e) 代价最低：两条臂都只是**推理参数**，模型在盘 => **零训练**；若 2.5 胜，受益的是主配方本身。
#
# ⚠⚠ **与「推理侧八条旋钮已全否」那条禁令的关系（照 (M37) 的写法，说清楚不回避）**：
#   * 那八条（步数/加 32px 数据/重排/温度/检查点/代次…）是**作为 32px 缺口的杠杆**被否的，
#     账本原话是"别再在 641 张的池子里调旋钮"。本臂**不问 32px**：出图、判官、判据全在 16px 上。
#   * CFG 那一条当年的判决用的是**分布指标**，而上面 (a)(c) 已逐项核对：**那些差值全在噪声下限之下**
#     => 它对"人类/判官偏好"从未成立过，和检查点那条同型。
#   * ⛔ **无论本臂结果如何，都不许把它搬到 24/32px 上**（32px 准入条件④：判官尺子没校准）。
#   * ⛔ 本臂**不碰** 32px 准入条件①②③一个字，也不碰任何已下的判决（含 (M36) `PIXCOMB_NULL`、
#     (M37) `CKPT_LATE_WINS`）。
#
# ================================================================ 二、设计（两条臂，唯一变量是 --cfg）
#   A = cfg25_rr4   `--cfg 2.5`（挑战者）
#   B = cfg15_rr4   `--cfg 1.5`（= 现行主配方 `final_test.sh` 的 TRD16c 那行）
# 其余**逐字相同**，照抄 `final_test.sh` 的 TRD16c 配方：
#   --run runs/trd_v8 --ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --pal_mode retrieve --xmodal --ret_nname 100
#   + `eval/rerank.py --n 4`（CLIP-B/16 四选一，与 B2 的 best-of-4 同预算）
# 种子默认 0，两臂相同。⚑ `--ckpt last.pt` 是 (M37) 刚判过的那一侧，**不是**本臂的变量。
#
# ⚠ **必须是验证集 V_mat**：这是一次**挑配置**，测试集只许最后看一次（`final_test.sh` 那一次，至今未跑）。
# ⚠ 口径限制（跑前写下）：V_mat 125 材质 -> full 有效 n 约 75-85。n=80 时双侧 p<0.05 需 **62.5%** 胜率
#   => **本臂只测得到「大效应」**；判不出方向**不等于**两档 CFG 一样好（见判据 (4) 的 `CFG_NULL` 读法）。
#
# ================================================================ 三、预注册判据（跑之前写死）
#  (1) **不设试点门**，可解率由 full 自己量（先例：`eval/b2_canvas2.sh`，`--no_gate` 必须显式传）。
#      `judge_pairs.py` 报 `resolve_rate` 与 `p_vs_floor`（FLOOR = 21/118 = 17.8%）。
#      若 `p_vs_floor >= 0.05` 或 `resolve_rate <= FLOOR` -> **`VOID_UNRESOLVABLE`：不报胜负、不下结论**
#      （⛔ 不许记成"打平"）。
#  (2) **主判据 = 判官**（判官是主尺子；CLIP 只用来定序，FD 已退出优化目标）：
#      A=`cfg25_rr4` 对 B=`cfg15_rr4` 的胜率，按 (M17) 立的规矩报**族级 (U1) CI**
#      （新 full 先登进 `analysis/arch/recheck_judge.py:EXPECT`，再跑族级自助）。
#      裸二项 p 只能当**下界**报，⛔ 不许单独拿它下判。
#  (3) **次要报告项（⛔ 非判据、⛔ 不许替 (2) 下判）**：`run_eval.py` 的 16px KID/FID/FD/CLIP 四项，
#      两臂之差对照 (M33) 的 m=1 下限 KID **4.656** / FID **6.0** / FD **20** / CLIP **0.41**。
#      差值小于下限的一律读作"**没测到**"。⛔ 不许因为方向对就说某档 CFG 更好。
#      ⚑ (M37) 已记在案：这四把尺子会互相打架，判官站 CLIP/FD 侧、与 FID 反向 —— 本轮**只登记、不解释**。
#  (4) **四种结果的读法全部写在前面，事后不许再编第五种**：
#      - A 的族级 CI **下界 > 0.5** -> **`CFG_HIGH_WINS`**：主配方的 CFG 选低了。
#        **授权（仅此一项、仅此一行）**：把 `eval/final_test.sh` 里 **TRD16c 那一条**生成命令的 CFG
#        从 1.5 改成 2.5（该行现在靠 `$G` 里的 `--cfg 1.5`，改法 = 给该行显式加 `--cfg 2.5`）。
#        ⛔ TRD16、TRD24、TRD32、颜色任务、B7 **一个字不动** —— 那要各自另一条臂、另行预注册。
#      - A 的族级 CI **上界 < 0.5** -> **`CFG_LOW_WINS`**：现行 1.5 **首次**被现行判官验证，
#        账上第二个"靠噪声选出来的配置"就此填上。⛔ `final_test.sh` 不改。
#      - 族级 CI **含 0.5** -> **`CFG_NULL`**：只等于"**在 n≈80 上没测到 >=62.5% 的效应**"。
#        ⛔ 不许记成"打平"、⛔ 不许据此说"CFG 无所谓"、⛔ 不许据此改任何默认值。
#      - 可解率不过地板 -> 见 (1)。
#  (5) **方向预测（写在跑之前）**：预测 **`CFG_NULL`**，次可能 `CFG_HIGH_WINS`。
#      理由：上面 (c) 的两次对照方向相反且都在噪声里，没有任何一侧有真凭据；而 1.5 与 2.5 的产物
#      在人眼尺度上的差别多半小于检查点那一档（3000 步 vs 20000 步）。**若 `CFG_LOW_WINS`，
#      这条预测当场记为被推翻**，并且要连带承认：v2 那次"在噪声里选出来的 1.5"**碰巧选对了**。
#  (6) **操作检验（任一条不过，本臂作废）**：
#      (OP1) 两个 tag 目录各 500 张（125 材质 x n=4）、两个 `_rr4` 目录各 125 张；
#      (OP2) 两臂命令逐字相同，唯一差异是 `--cfg`（脚本把两条命令原样打印出来备查）；
#      (OP3) `api_fail == 0`；
#      (OP4) 两臂 `_rr4` 产物**逐像素相同的材质数 <= 25/125**（若两边几乎同图，判官只能瞎答，
#            这时 (2) 的胜率不可解读 -> 作废）；
#      (OP5) `run_eval.py` 里两臂 `n` 与 `materials` 必须相等（`run_eval.py:100` 的 ok 是各算各的）。
#  (7) **本轮不改任何默认值**：`judge_pairs.py` 的 `min_rate=0.65` 一个字不动（`--no_gate` 显式传），
#      `gen_trd.py`/`rerank.py`/`run_eval.py`/`train_trd.py`/`UNITS_PER_TILE` 全不动。
#  (8) 成本上限：GPU 两次 16px 生成（各约 500 张，参照 359 张/分，实占约 5GB 显存）+ 判官至多 125 对 x 2 序。
#      `api_fail` 率 > 25% 就停下记账、不补跑。
#
# ================================================================ 四、运行约束
# `/mnt/data` 长期贴满 -> 产物全写 `/tmp`（`--out`/`--root`/`--outdir`），结果 scp 回本机入库。
# `scripts/sync_remote_tmp.sh` 的 `judge_*` 与 `m3[0-9]_*.json` 模式覆盖本轮产物；
# 指标 JSON 显式命名为 `/tmp/m38_cfg16.json` 以落在白名单里。
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
# 起法（两段；不给 STAGE 或 STAGE=gen 只出图算指标 = 零 API）：
#   tmux new-session -d -s arch_cfg "bash /tmp/run_cfg.sh"   # 命令必须先写进文件再跑
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
O=${OUT:-/tmp/cfgab16}
J=${JOUT:-/tmp/judge_cfgab16}
STAGE=${STAGE:-gen}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
mkdir -p "$O" "$J"

V8=runs/trd_v8
COMMON="--run $V8 --ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --pal_mode retrieve --xmodal --ret_nname 100 --out $O"

if [ "$STAGE" = gen ]; then
  "$P" -c 'import numpy, PIL, torch, sys; sys.exit(0)' || { echo "[cfgab16] python 探针失败: $P"; exit 1; }
  # (OP2)：两条命令原样打印，唯一差异必须是 --cfg
  echo "[cfgab16] A: eval/gen_trd.py --cfg 2.5 $COMMON --tag cfg25"
  echo "[cfgab16] B: eval/gen_trd.py --cfg 1.5 $COMMON --tag cfg15"
  $P eval/gen_trd.py --cfg 2.5 $COMMON --tag cfg25 || exit 1
  $P eval/gen_trd.py --cfg 1.5 $COMMON --tag cfg15 || exit 1
  for t in cfg25 cfg15; do
    $P eval/rerank.py --src $t --set V_mat --size 16 --n 4 --root "$O" || exit 1
  done
  # (OP1)+(OP4)：张数与"两边是不是几乎同图"
  $P - "$O" <<'PYEOF' || exit 1
import sys, pathlib, numpy as np
from PIL import Image
root = pathlib.Path(sys.argv[1])
bad = []
for t, want in [("cfg25", 500), ("cfg15", 500), ("cfg25_rr4", 125), ("cfg15_rr4", 125)]:
    n = len(list((root / t / "16").glob("*.png")))
    print("[OP1] %-10s %d 张 (应为 %d)" % (t, n, want))
    if n != want:
        bad.append("OP1:%s=%d" % (t, n))
same = 0
tot = 0
for p in sorted((root / "cfg25_rr4" / "16").glob("*.png")):
    q = root / "cfg15_rr4" / "16" / p.name
    if not q.exists():
        bad.append("OP4:missing:%s" % p.name)
        continue
    tot += 1
    if np.array_equal(np.asarray(Image.open(p)), np.asarray(Image.open(q))):
        same += 1
print("[OP4] 两臂逐像素相同的材质 %d/%d（门槛 <=25）" % (same, tot))
if same > 25:
    bad.append("OP4:same=%d" % same)
print("[操作检验] 已查 %d 项，problems=%d %s" % (5, len(bad), bad))
sys.exit(1 if bad else 0)
PYEOF
  $P eval/run_eval.py --set V_mat --size 16 --root "$O" --out /tmp/m38_cfg16.json \
     --methods cfg25_rr4 cfg15_rr4 cfg25 cfg15 || exit 1
  echo CFGAB16_GEN_DONE
fi

if [ "$STAGE" = judge ]; then
  [ -n "${VLM_BASE_URL:-}" ] && [ -n "${VLM_API_KEY:-}" ] || { echo "[cfgab16] 缺 VLM 凭据"; exit 1; }
  while ps -eo args | grep -q '[j]udge_pairs.py'; do
    echo "[cfgab16] 别的判官在跑，300s 后再看"; sleep 300
  done
  $P eval/judge_pairs.py full --no_gate --a cfg25_rr4 --b cfg15_rr4 \
     --set V_mat --size 16 --root "$O" --outdir "$J" || exit 1
  echo CFGAB16_JUDGE_DONE
fi
