#!/usr/bin/env bash
# =====================================================================================
# (M43) 预注册：**补充训练数据量** —— 2026-09-11 那次把 80% 补充数据扔掉的选择，
# 第一次交给现行判官。本文件在任何一张图存在之前提交 = 预注册。
# ⛔ 跑完不许改判据、不许改配方重跑、不许挪门槛、不许换尺子。
#
# ------------------------------------------------------------------- (1) 为什么是这一格
# 主线还是那句「**当年是谁、用哪把尺子替我们选的，那个差值过得了噪声下限吗**」。
# (M42) 顺带审出的这一格剂量最大：现行 `--extra_file train_extra_packs_only.json` = **2433 张**，
# 同目录 `train_extra.json`（含模组）= **12241 张**；而 16px 基础训练池（`dataset_k16.json`
# 5979 条里的 train 部分）只有几千条 ⇒ 换成含模组的那份，**训练池约翻 2.5 倍**。
# 当年选掉它的全部证据是 v7 vs v8 的 m=1 分布指标：KID 2.7 / FID 5.3 / FD 6.6 / CLIP 0.08，
# **4/4 在 (M33) 的 m=1 噪声下限（4.656/6.0/20/0.41）之下**，FD 那项方向还是反的；
# 其中 KID 与 FD 两项**比 (M42) 量到的纯重训噪声（KID 3.92 / FD 7.04）还小**。
# ⇒ 「模组数据没帮助」这句话从没被任何过得了噪声的尺子支持过，判官更没问过（判官晚于它出现）。
#
# ------------------------------------------------------------------- (2) 臂与唯一变量（以及**不回避**的混杂）
#   A = data7：`runs/trd_v7` 的 **step_20000.pt**（训练池含模组 + `--domain` 来源条件）
#   B = data8：`runs/trd_v8` 的 **step_20000.pt**（= 现行 16px 主配置的权重；与 `last.pt`
#              **逐张量相同**，已实测 152/152 `torch.equal`）
# 出图命令唯一差异 = `--run`（两臂都显式写 `--ckpt step_20000.pt`）。
#
# ⚠⚠ **这不是一条干净的数据臂，本轮不假装它是。** 两个 run 的 config 共差 6 键：
#   `out`（无意义）、`save_at`/`steps`（30000 vs 20000 —— **已由“两臂都取 step_20000”这一步消除**，
#   所以 (M37) `CKPT_LATE_WINS` 那个混杂**不在本比较里**）、`domain`(True/False)、
#   `extra_file`（v7 没有这个键 ＝ 当年默认的 `train_extra.json` 含模组 / v8 = packs_only）、
#   `codebook_err`（码本由训练数据建 ⇒ 两份 codebook.npy 必然不同）。
# ⇒ 真正剩下的差异是 **{含模组的训练池 + 来源条件} 整体**，外加它必然带来的**不同码本**。
# ⛔ 因此本轮无论判成什么，**都只授权"采纳"层面的话（16px 该发哪一个配置），
#    ⛔ 一律不许读成"模组数据本身有用/没用"**（那是机制问题，要另开一条成对臂）。
# ⚠ v7 带 `--domain`，而 `gen_trd.py --dom_w` 默认 0.0 —— 已按 (M42) 第三节的要求**先核实**：
#   `model/trd.py:353` 在 `dom is None` 时喂 `torch.zeros_like(k)`，而 `model/train_trd.py:124`
#   规定 **0 = 材质包**（1 = 模组、2 = SDXL）⇒ 默认出图路径确实把来源钉在"材质包"那一侧，
#   臂是有意义的。（`eval/gen_trd.py:107` 的 `--dom_w` 本轮不传，保持默认 0.0。）
# ⚠ 32px 侧对模组的判决（(M24)：按名字筛完只剩 229 张）是**另一件事**，两边不许互当证据。
#
# ------------------------------------------------------------------- (3) 出图配方（逐字照抄，⛔ 不许调）
# 与 (M37)(M38)(M39)(M40)(M41)(M42) 完全相同的 TRD16c：
#   16px / V_mat / `--cfg 1.5` / `--pal_mode retrieve --xmodal --ret_nname 100` / `--n 4`
#   + `eval/rerank.py --n 4`；判官 `--no_gate`（(M19) 之后试点门已废）。
# ⚠ `--pal_mode retrieve` 的调色板记忆库与 run 无关（`PaletteMemory(dev)` 只吃真人瓦片），
#   但检索到的调色板要**用各自 run 的码本**量化 ⇒ 本轮两臂的差别里含一份**换码本**的颜色位移。
#   这是"换训练数据"的必然产物、不是实现瑕疵，**但它使本轮的 pixfrac 与三个锚点不同类**（见 (5)）。
#
# ------------------------------------------------------------------- (4) 操作检验（任一不过 ⇒ 判 VOID，不开判官）
# (OP1) 两臂各 500 张（125 材质 x n=4）、各 125 张 `_rr4`。
# (OP2) 两份 config 的差异键 ⊆ {out, save_at, steps, domain, extra_file, codebook_err}；
#       且 **必须**同时包含 `domain` 与 `extra_file` —— 否则两臂根本不是我们以为的那两个模型。
# (OP3) 两个检查点自报的 `step` 都 == 20000（消除 (M37) 那个混杂的**唯一**凭据）。
# (OP4) 两份 codebook.npy 的 md5 **不同** ＝ 训练池确实不同（正向对照；⛔ 相同就说明拿错了 run）。
# (OP5) 逐像素相同的材质 <= 25/125（(M37)(M38)(M40) 已用的同一道门）。
# (OP6) 判官 `api_fail == 0`；开跑前先 curl 探针一次（(M40) 的教训：失效 key 会静默烧掉一小时；
#       ⚠ 账本：严格模式现在命中两把 key，要取 `tail -1` 那把）。
#
# ------------------------------------------------------------------- (5) 零 API 筛子：当门用（省钱），但**不改它的锚点**
# 三锚点逐字照抄，⛔ 一个字不许改：D37 = 0.8173（判官显著）/ D38 = 0.6608（`CFG_NULL`）/
# D40 = 0.6431（`GEN_NULL`）。(M42) 的构造性零对照 D_seed = 0.6643 **只登记**（AMBIGUOUS 授权的
# 唯一一句话就是"不改门、不挪锚点"），本轮照办。
# 门（出图之后、花 API 之前算一次，写进代码执行：非 `SCREEN_GO` 就 `exit 3`、零 API）：
#     d >= 0.8173            -> `SCREEN_GO`         ：开判官臂
#     d <= 0.6608            -> `SCREEN_NOGO`       ：不开判官、不花 API，判 `DATA_TOO_SMALL`
#     0.6608 < d < 0.8173    -> `SCREEN_AMBIGUOUS`  ：同样不开判官，记账走人
# ⚠⚠ **本轮的 pixfrac 与三个锚点不同类**：三个锚点都是**同一份码本**内的比较，本轮多一份
#   换码本的颜色位移（见 (3)）。这句话**跑前写死**，两个方向都要照它读：
#   - 若 `SCREEN_GO` 且判官**显著** ⇒ 记作筛子的**首个独立 GO 验证**，但**必须同引这条不同类的注意事项**。
#   - 若 `SCREEN_GO` 而判官 `_NULL` ⇒ 按 (M41) §5 立下的规矩 **当场撤回筛子**，⛔ 不许因为
#     "本轮不同类"或"功效不足"给它找台阶 —— 那条规矩是写死的，本轮照单执行。
#
# ------------------------------------------------------------------- (6) 判官臂与主判据（只在 `SCREEN_GO` 时执行）
# 主判据 = **族级 (U1) CI**（`judge_cluster.analyse`，新臂先登进 `recheck_judge.EXPECT`）：
#     CI 不含 0.5 且偏 A -> `DATA_V7_WINS`：**16px 该发的是含模组那一支** ⇒ 授权（且仅授权）
#         另行预注册一次 `final_test.sh` 换配置，外加一条干净的机制臂（同码本成对续训）。
#         ⛔ 本轮不许直接改 `final_test.sh` 一个字。
#     CI 不含 0.5 且偏 B -> `DATA_V8_WINS`：2026-09-11 那次选择**第一次被现行判官验证**
#         （在"采纳"这个层面）。⛔ 仍不许读成"模组数据没用"（见 (2) 的混杂）。
#     CI 含 0.5           -> `DATA_NULL`：**在该 n 上没测到 >= X% 的效应**（X 按实得 n 精确二项现算）。
#         ⛔ 不许读成"没差别"、⛔ 不许读成"一样好"、⛔ 不许靠加臂碰运气。
# 先决条件：可解率对地板 21/118 显著，否则 `VOID_UNRESOLVABLE`。
# ⚠⚠ 功效硬上限（(M41) `power_curve.py`）：V_mat 分母 125，可解率拉满 MDE 也只 62.4%，
#   实际 80% 功效的 MDE ≈ **67.6%**；引本轮任何 `_NULL` 必须同引这一句。
# 次要报告项（⛔ 非判据，thr(1) = KID 4.656 / FID 6.0 / FD 20 / CLIP 0.41）：`run_eval.py` 四项。
#
# ------------------------------------------------------------------- (7) 跑前预测（写死，跑完照录，推翻了就照录推翻）
# (P1) 筛子 **`SCREEN_GO`（d >= 0.8173）**。理由：两条**从零各自独立训满 20000 步**的 run，
#      训练池差 2.5 倍、码本还不同 ⇒ 差别应当超过 (M42) 那个"只换种子"的构造性零对照 D_seed=0.6643，
#      而且换码本这一份颜色位移几乎逐张都在。
#      ⚠ 账本写死的反面证据：**跑前 pixfrac 预测已连错两次、且方向相反**
#      （(M41) 猜 >=0.8173 实得 0.3493；(M42) 猜 >=0.8173 实得 0.6643）⇒ 这条预测**不值钱**，
#      它存在的意义只是不许事后改口。
# (P2) 若开成判官臂：预测 **`DATA_V7_WINS`**。理由 = 剂量：训练池 2.5 倍是本项目量过的最大一次改动，
#      而当年扔掉它的四把尺子全在噪声下限之下、其中两项比纯重训噪声还小。
#      ⚠ 反面：(M38)(M40) 连着两轮 `_NULL`、可解率 63.2->56.0->50.4 在掉，MDE 67.6% 很高。
# (P3) 次要分布指标：预测 `_rr4` 那四格里**至少 1 格**过 thr(1)（(M41) 是 0/8、(M39) 是 4/16）。
#      ⛔ 它不是判据，过不过都不改任何判决。
#
# ------------------------------------------------------------------- (8) 不许动的东西
# ⛔ `eval/final_test.sh`、`eval/judge_pairs.py`（活件）、`judge_cluster*.py`、
#    筛子三锚点与门槛、`UNITS_PER_TILE`、任何默认值、32px 准入条件①②③④、已下的任何判决。
# ⛔ 结构门 / 各向异性永远不许当优化目标或挑配置的依据。
# ⛔ 本轮不训练任何模型（两个检查点都已在 `runs/`）。
#
# 用法（远程）：
#   STAGE=gen   bash eval/m43_data_ab.sh
#   STAGE=judge bash eval/m43_data_ab.sh     # 自己会读 meter 的 screen，非 GO 就 exit 3
# =====================================================================================
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
O=${OUT:-/tmp/m43ab}
J=${JOUT:-/tmp/judge_m43ab}
MET=${MET:-/tmp/m43_meter.json}
STAGE=${STAGE:-gen}
RA=${RA:-runs/trd_v7}
RB=${RB:-runs/trd_v8}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_m43
mkdir -p "$O" "$J"

COMMON="--ckpt step_20000.pt --set V_mat --size 16 --n 4 --bs 16 --cfg 1.5 \
 --pal_mode retrieve --xmodal --ret_nname 100 --out $O"

if [ "$STAGE" = gen ]; then
  "$P" -c 'import numpy, PIL, torch, sys; sys.exit(0)' || { echo "[m43] python 探针失败: $P"; exit 1; }
  # 盲写判读器先自检再干活（账本：判读器必须先跑 --selftest）
  "$P" analysis/arch/m43_read_data.py --selftest || { echo "[m43] 判读器自检不过 -> 停"; exit 1; }
  md5sum "$RA/codebook.npy" "$RB/codebook.npy"
  # 两条命令原样打印：唯一差异必须是 --run
  echo "[m43] A: eval/gen_trd.py --run $RA $COMMON --tag data7"
  echo "[m43] B: eval/gen_trd.py --run $RB $COMMON --tag data8"
  $P eval/gen_trd.py --run "$RA" $COMMON --tag data7 || exit 1
  $P eval/gen_trd.py --run "$RB" $COMMON --tag data8 || exit 1
  for t in data7 data8; do
    $P eval/rerank.py --src $t --set V_mat --size 16 --n 4 --root "$O" || exit 1
  done
  # (OP1)-(OP5) + 筛子：全部由盲写判读器算并落盘
  $P analysis/arch/m43_read_data.py screen --root "$O" --ra "$RA" --rb "$RB" --out "$MET"
  RC=$?
  # 次要指标只登记、⛔ 不当证据
  $P eval/run_eval.py --set V_mat --size 16 --root "$O" --out /tmp/m43_gen16.json \
     --methods data7_rr4 data8_rr4 data7 data8 || true
  echo "M43_GEN_DONE rc=$RC"
  exit $RC
fi

if [ "$STAGE" = judge ]; then
  # 门写进代码执行（账本：没人读的退出码不是门）
  NP=$("$P" -c "import json,sys;m=json.load(open(sys.argv[1],encoding='utf-8'));print(len(m['problems']) if 'problems' in m else 99)" "$MET") || exit 1
  echo "[m43] 操作检验 problems 数 = $NP（99 = meter 里连这个键都没有，同样作废）"
  [ "$NP" = 0 ] || { echo "[m43] 【禁】(OP1)-(OP5) 有 $NP 项不过 -> 按 (4) 本臂 VOID，不开判官、不花 API"; exit 4; }
  S=$("$P" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('screen'))" "$MET") || exit 1
  echo "[m43] 筛子判决 = $S"
  [ "$S" = SCREEN_GO ] || { echo "[m43] 【禁】非 SCREEN_GO -> 按 (5) 不开判官臂、不花 API，记账走人"; exit 3; }
  [ -n "${VLM_BASE_URL:-}" ] && [ -n "${VLM_API_KEY:-}" ] || { echo "[m43] 缺 VLM 凭据"; exit 1; }
  curl -s -o /dev/null -w '[m43] key 探针 HTTP %{http_code}\n' \
    -H "Authorization: Bearer $VLM_API_KEY" "$VLM_BASE_URL/models"
  while ps -eo args | grep -q '[j]udge_pairs.py'; do
    echo "[m43] 别的判官在跑，300s 后再看"; sleep 300
  done
  $P eval/judge_pairs.py full --no_gate --a data7_rr4 --b data8_rr4 \
     --set V_mat --size 16 --root "$O" --outdir "$J" || exit 1
  echo M43_JUDGE_DONE
fi
