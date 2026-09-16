#!/usr/bin/env bash
# (M37) 重新选杠杆后的第一条臂：**主配方到底该用哪个检查点**（16px、验证集 V_mat、同材质直接对判）
# 预注册：本文件先提交再跑，跑完不许改判据。零训练（两个检查点都已在盘）。
#
# ================================================================ 一、为什么是这条
# (M36) `PIXCOMB_NULL` 之后，(M26)/(M27)/(M28) 那个"唯一授权动作"已无未试过的廉价形式，
# 账本写死「下一轮第一件事是**重新选杠杆**」。选中这条的理由是三条硬事实：
#
#  (a) **主配方站在过拟合曲线的错误一侧**。`runs/trd_v8/log.json`（16px 主配置的那次训练）：
#      val 在 **step 3000 见底 6.0647**，之后**单调上升**到 step 20000 的 **6.7671**（+11.6%）。
#      `eval/final_test.sh` 里三条 V8 生成命令全都显式写着 `--ckpt last.pt` = **step 20000**。
#      （`gen_trd.py:73` 的默认值其实是 `best.pt`，是交付脚本把它改掉的。）
#      同一现象在 v10 上也有：最低 6.5426@2000 -> 最终 6.8690@12000。
#  (b) **这条纪律从来没被现行判官验过**。它是 v2/v3 期定的（`812ce00`），当年**只用结构门**判过，
#      而结构门后来被证明**排反序** => "检查点不影响人类偏好"这句话至今只是账上一条备注
#      （账本原话："其实没被判官验过，只记账"）。
#  (c) **代价最低、覆盖最广**：两个检查点都已在盘 => **零训练**；若早停这侧胜，
#      受益的是**每一个尺寸、每一条今后的臂**和正式测试本身。
#
# ⚠⚠ **与「推理侧八条旋钮已全否」那条禁令的关系，说清楚不回避**：
#   * 那八条（步数/加 32px 数据/重排/温度/**检查点**/代次…）是**作为 32px 缺口的杠杆**被否的，
#     账本原话是"**别再在 641 张的池子里调旋钮**"。本臂**不问 32px**：出图、判官、判据全在 16px 上。
#   * 检查点那一条的原始判决用的是**已被证伪的结构门**（见 (b)），对"人类/判官偏好"从未成立。
#   * ⛔ **无论本臂结果如何，都不许把它搬到 32px 上当杠杆**；32px 要用得另行预注册，
#     且仍受 32px 准入条件④（不开任何 32px 判官新臂）约束。
#   * ⛔ 本臂**不碰** 32px 准入条件①②③一个字，也不碰任何已下的判决。
#
# ================================================================ 二、设计（两条臂，唯一变量是 --ckpt）
#   A = ck3k_rr4   `runs/trd_v8/best.pt`  = **step 3000**（val 最低点；mtime 09:37 与 3000 步的时刻对得上）
#   B = ck20k_rr4  `runs/trd_v8/last.pt`  = **step 20000**（= 现行主配方 `final_test.sh` 用的那个）
# 其余**逐字相同**，照抄 `final_test.sh` 的 TRD16c 配方：
#   --set V_mat --size 16 --n 4 --bs 16 --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100
#   + `eval/rerank.py --n 4`（CLIP-B/16 四选一，与 B2 的 best-of-4 同预算）
# 种子默认 0，两臂相同。
#
# ⚠ **必须是验证集 V_mat**：这是一次**挑配置**，测试集只许最后看一次（`final_test.sh` 那一次）。
# ⚠ 口径限制（跑前写下）：V_mat 125 材质 -> full 有效 n 约 75-85。n=80 时双侧 p<0.05 需 **62.5%** 胜率
#   => **本臂只测得到「大效应」**；判不出方向**不等于**两个检查点一样好（见判据 (4) 的 `CKPT_NULL` 读法）。
#
# ================================================================ 三、预注册判据（跑之前写死）
#  (1) **不设试点门**，可解率由 full 自己量（先例：`eval/b2_canvas2.sh`，`--no_gate` 必须显式传）。
#      `judge_pairs.py` 报 `resolve_rate` 与 `p_vs_floor`（FLOOR = 21/118 = 17.8%）。
#      若 `p_vs_floor >= 0.05` 或 `resolve_rate <= FLOOR` -> **`VOID_UNRESOLVABLE`：不报胜负、不下结论**
#      （与 nod32 同类的"什么都没测到"，⛔ 不许记成"打平"）。
#  (2) **主判据 = 判官**（判官是主尺子；CLIP 只用来定序，FD 已退出优化目标）：
#      A=`ck3k_rr4` 对 B=`ck20k_rr4` 的胜率，按 (M17) 立的规矩报**族级 (U1) CI**
#      （新 full 先登进 `analysis/arch/recheck_judge.py:EXPECT`，再跑 `analysis/arch/judge_cluster_sweep.py`）。
#      裸二项 p 只能当**下界**报，⛔ 不许单独拿它下判。
#  (3) **次要报告项（⛔ 非判据、⛔ 不许替 (2) 下判）**：`run_eval.py` 的 16px KID/FID/FD/CLIP 四项，
#      两臂之差对照 (M33) 在 TRD 臂上实测的噪声下限 **thr(1)=4.656**（m=1 口径，取每材质第 0 张）。
#      差值小于下限的一律读作"**没测到**"。⛔ 不许因为方向对就说某个检查点更好。
#  (4) **四种结果的读法全部写在前面，事后不许再编第五种**：
#      - A 的族级 CI **下界 > 0.5** -> **`CKPT_EARLY_WINS`**：主配方站错了边。
#        **授权**（仅此一项）：把 `eval/final_test.sh` 里**三条 V8 命令**的 `--ckpt last.pt` 改成 `best.pt`。
#        ⛔ V10 那三行（24/32px）**一个字不动** —— 那要另一条臂、另行预注册。
#      - A 的族级 CI **上界 < 0.5** -> **`CKPT_LATE_WINS`**：现行纪律**首次**被现行判官验证，
#        账上那个洞就此填上。⛔ `final_test.sh` 不改。
#      - 族级 CI **含 0.5** -> **`CKPT_NULL`**：只等于"**在 n≈80 上没测到 >=62.5% 的效应**"。
#        ⛔ 不许记成"打平"、⛔ 不许据此说"检查点无所谓"、⛔ 不许据此改任何默认值。
#      - 可解率不过地板 -> 见 (1)。
#  (5) **方向预测（写在跑之前）**：预测 **`CKPT_LATE_WINS` 或 `CKPT_NULL`** ——
#      v2/v3 期的观察是 best.pt 采样**反而更差**（尽管那次用的是后来被推翻的尺子），
#      且掩码生成模型的验证 CE 与采样质量本就常常脱钩。**若 A 胜，这条预测当场记为被推翻**，
#      并且要连带承认：我们一年里所有 `last.pt` 臂都拿了一个次优起点。
#  (6) **操作检验（任一条不过，本臂作废）**：
#      (OP1) 两个 tag 目录各 500 张（125 材质 x n=4）、两个 `_rr4` 目录各 125 张；
#      (OP2) 两臂命令逐字相同，唯一差异是 `--ckpt`（脚本把两条命令原样打印出来备查）；
#      (OP3) `api_fail == 0`；
#      (OP4) 两臂 `_rr4` 产物**逐像素相同的材质数 <= 25/125**（若两边几乎同图，判官只能瞎答，
#            这时 (2) 的胜率不可解读 -> 作废）；
#      (OP5) `run_eval.py` 里两臂 `n` 与 `materials` 必须相等（`run_eval.py:100` 的 ok 是各算各的）。
#  (7) **本轮不改任何默认值**：`judge_pairs.py` 的 `min_rate=0.65` 一个字不动（`--no_gate` 显式传），
#      `gen_trd.py`/`rerank.py`/`run_eval.py`/`train_trd.py`/`UNITS_PER_TILE` 全不动。
#  (8) 成本上限：GPU 两次 16px 生成（各约 500 张，参照 359 张/分）+ 判官至多 125 对 x 2 序。
#      `api_fail` 率 > 25% 就停下记账、不补跑。
#
# ================================================================ 四、运行约束
# `/mnt/data` 长期贴满 -> 产物全写 `/tmp`（`--out`/`--root`/`--outdir`），结果 scp 回本机入库。
# `scripts/sync_remote_tmp.sh` 的 `judge_*` 与 `m3[0-9]_*.json` 模式覆盖本轮产物；
# 指标 JSON 显式命名为 `/tmp/m37_ckpt16.json` 以落在白名单里。
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
# 起法（两段；不给 STAGE 或 STAGE=gen 只出图算指标 = 零 API）：
#   tmux new-session -d -s arch_ck "bash /tmp/ckpt16_ab.sh"   # 命令必须先写进文件再跑
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
O=${OUT:-/tmp/ckab16}
J=${JOUT:-/tmp/judge_ckab16}
STAGE=${STAGE:-gen}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
mkdir -p "$O" "$J"

V8=runs/trd_v8
COMMON="--run $V8 --set V_mat --size 16 --n 4 --bs 16 --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --out $O"

if [ "$STAGE" = gen ]; then
  "$P" -c 'import numpy, PIL, torch, sys; sys.exit(0)' || { echo "[ckab16] python 探针失败: $P"; exit 1; }
  # (OP2)：两条命令原样打印，唯一差异必须是 --ckpt
  echo "[ckab16] A: eval/gen_trd.py --ckpt best.pt $COMMON --tag ck3k"
  echo "[ckab16] B: eval/gen_trd.py --ckpt last.pt $COMMON --tag ck20k"
  $P eval/gen_trd.py --ckpt best.pt $COMMON --tag ck3k  || exit 1
  $P eval/gen_trd.py --ckpt last.pt $COMMON --tag ck20k || exit 1
  for t in ck3k ck20k; do
    $P eval/rerank.py --src $t --set V_mat --size 16 --n 4 --root "$O" || exit 1
  done
  # (OP1)+(OP4)：张数与"两边是不是几乎同图"
  $P - "$O" <<'PYEOF' || exit 1
import sys, pathlib, numpy as np
from PIL import Image
root = pathlib.Path(sys.argv[1])
bad = []
for t, want in [("ck3k", 500), ("ck20k", 500), ("ck3k_rr4", 125), ("ck20k_rr4", 125)]:
    n = len(list((root / t / "16").glob("*.png")))
    print("[OP1] %-10s %d 张 (应为 %d)" % (t, n, want))
    if n != want:
        bad.append("OP1:%s=%d" % (t, n))
same = 0
tot = 0
for p in sorted((root / "ck3k_rr4" / "16").glob("*.png")):
    q = root / "ck20k_rr4" / "16" / p.name
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
  $P eval/run_eval.py --set V_mat --size 16 --root "$O" --out /tmp/m37_ckpt16.json \
     --methods ck3k_rr4 ck20k_rr4 ck3k ck20k || exit 1
  echo CKAB16_GEN_DONE
fi

if [ "$STAGE" = judge ]; then
  [ -n "${VLM_BASE_URL:-}" ] && [ -n "${VLM_API_KEY:-}" ] || { echo "[ckab16] 缺 VLM 凭据"; exit 1; }
  while ps -eo args | grep -q '[j]udge_pairs.py'; do
    echo "[ckab16] 别的判官在跑，300s 后再看"; sleep 300
  done
  $P eval/judge_pairs.py full --no_gate --a ck3k_rr4 --b ck20k_rr4 \
     --set V_mat --size 16 --root "$O" --outdir "$J" || exit 1
  echo CKAB16_JUDGE_DONE
fi
