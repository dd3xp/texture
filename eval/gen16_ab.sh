#!/usr/bin/env bash
# (M40) 第三条"用现行判官复查一个从没被判官验过的配置选择"的臂：**16px 主配方到底该用哪一代模型**
# （v8 还是 v10；16px、验证集 V_mat、同材质直接对判）。预注册：本文件先提交再跑，跑完不许改判据。**零训练**。
#
# ================================================================ 一、为什么是这条
# (M37) 立下、(M38) 沿用的那个招式是：`final_test.sh` 里每一个配置选择都要问
# 「**当年是谁、用哪把尺子替我们选的，那个差值过得了噪声下限吗**」。
# 已经问过两格：检查点（(M37) `CKPT_LATE_WINS`，现行做法首次被验证）、CFG（(M38) `CFG_NULL`，仍未验证）。
# 这一格是**同一张表上分量最重的一格**：出货的 16px 结果用哪一代权重。
#
#  (a) **这个选择当年是按分布指标定的，而那四把尺子这一档全部瞎**。账本 2026-09-12「v10（粗网格条件）结果」
#      那张表，16px 一行（v8x / v10x）：
#        KID 5.2 / 7.9、FID 61.9 / 65.8、FD 104.2 / 109.2、CLIP 34.82 / 34.85
#      逐项对 (M33) 在 TRD 臂上实测的 m=1 噪声下限（KID **4.656** / FID **6.0** / FD **20** / CLIP **0.41**）：
#        |ΔKID| 2.7 < 4.656、|ΔFID| 3.9 < 6.0、|ΔFD| 5.0 < 20、|ΔCLIP| 0.03 < 0.41
#      => **四项一个都没过下限**。当年账本里那句"v10 16px 略退（KID 5.2→7.9）"就是**整个发生在噪声里**的
#      一次选择 —— 与 (M38) 在 CFG 上当场复现的情形同型，只是这次被它决定的是**权重本身**。
#  (b) **两代模型是严格嵌套的，变量干净**：`runs/trd_v10/config.json` 写死
#      `init_from = runs/trd_v8/last.pt`、`steps 12000`、`coarse true`、`p_coarse 0.5`、`p32 0.5`
#      （v8：`steps 20000`、`p32 0.3`、无 coarse）。两份 `codebook.npy` 的 md5 相同
#      （实测 8dd2a303b547584a1013a2cb116d23be，见 (OP3)），`extra_file` 相同
#      => v10 = **v8 再训 12000 步**（多学的那部分里 16px 也有份：账本原话"16px 瓦片也做 8→16 的训练"）。
#  (c) **(M37) 刚刚证明"再多训一段"这件事在这条管线上是能被判出来的**：v8 内部 step3000 vs step20000,
#      晚的显著胜（26/79=32.9%，族级 [0.217,0.438]，`W1_robust`），而当年替我们挑检查点的 val 曲线
#      **指的是反方向**。v10 的 val 曲线同样难看（账本：`runs/trd_v10` 最佳 val 6.5426 @ step 2000，
#      之后单调变差）—— 但 (M37) 已经把"val 触底=该收手"这条读法钉死为**在这件事上给错方向**
#      ⇒ v10 的 val 曲线**不构成**反对它的证据，这一格至今**真正未定**。
#  (d) **它满足 (M38) 留下的跑前判据的精神**：挑战者与现行配置的差别应当**不小于**"检查点那一档"——
#      这里换的是**权重**（12000 步 + 新条件嵌入），不是一个推理标量。(M39) 已实测生成管线逐像素确定，
#      所以两臂之间的差别全部来自权重。⚑ 本轮会顺带把这把尺子的读数记下来（见 (9)），
#      **但读数不做门**，理由见 (9)。
#  (e) 代价最低：两份权重都在盘上（`runs/trd_v8/last.pt`、`runs/trd_v10/last.pt`）=> **零训练**。
#
# ⚠⚠ **与「推理侧八条旋钮含代次已全否」那条禁令的关系（照 (M37)(M38) 的写法，说清楚不回避）**：
#   * 那八条是**作为 32px 缺口的杠杆**被否的（账本原话"别再在 641 张的池子里调旋钮"），
#     其中"代次"那一条具体指的是 32px 上的 `v10x_direct vs v11dx_direct`（46/75，`EXPECT` 有案）。
#     **本臂不问 32px**：出图、判官、判据全在 16px 上，问的是 16px 主配方该用哪一代。
#   * ⛔ **无论本臂结果如何，都不许把它搬到 24/32px 上**（32px 准入条件④：那边的判官尺子没校准）。
#   * ⛔ 本臂**不碰** 32px 准入条件①②③一个字，也不碰任何已下的判决
#     （含 (M36) `PIXCOMB_NULL`、(M37) `CKPT_LATE_WINS`、(M38) `CFG_NULL`、(M39) `SCREEN_AMBIGUOUS`）。
#   * ⛔ 本臂**不是**在问"32px 用 v10 对不对"，也**不许**被读成对 v11/v11d 的任何评价。
#
# ================================================================ 二、设计（两条臂，唯一变量是 --run）
#   A = gen10_rr4   `--run runs/trd_v10`（挑战者：v8 再训 12000 步的那一代）
#   B = gen8_rr4    `--run runs/trd_v8` （= 现行主配方 `final_test.sh:20` 的 TRD16c 那行）
# 其余**逐字相同**，照抄 `final_test.sh` 的 TRD16c 配方（也与 (M38)/(M39) 的基准臂逐字相同）：
#   --ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100
#   + `eval/rerank.py --n 4`（CLIP-B/16 四选一，与 B2 的 best-of-4 同预算）
# 种子默认 0，两臂相同。⚑ `--ckpt last.pt` 是 (M37) 刚判过的那一侧、`--cfg 1.5` 是 (M38) 未能判动的那一侧，
# 两者都**不是**本臂的变量。
#
# ⚠ **v10 的粗网格条件在本臂里是关着的，这是设计不是疏漏**：`eval/gen_trd.py:275` 只有 `--cascade` 且
#   `size != 16` 时才喂粗网格；本臂不传 `--cascade`、尺寸是 16 => `model/trd.py:339` 走
#   `cz = full_like(grid, K_MAX)` 的空条件分支，而那正是 v10 训练时 **50%（`p_coarse 0.5`）** 见过的模式。
#   所以这是"v10 在它自己训练过的无粗网格模式下、按现行 16px 配方出图"，**不是**把它绑住了一只手。
#   ⛔ 但也因此，本臂**不许**被读成对"粗网格条件本身"的评价。
#
# ⚠ **必须是验证集 V_mat**：这是一次**挑配置**，测试集只许最后看一次（`final_test.sh` 那一次，至今未跑）。
# ⚠ 口径限制（跑前写下）：V_mat 125 材质 -> full 有效 n 约 70-85（(M37) 79、(M38) 70）。
#   n=79 时双侧 p<0.05 需约 **62%** 胜率 => **本臂只测得到"大效应"**；判不出方向**不等于**两代一样好
#   （见判据 (4) 的 `GEN_NULL` 读法）。
#
# ================================================================ 三、预注册判据（跑之前写死）
#  (1) **不设试点门**，可解率由 full 自己量（先例：`eval/b2_canvas2.sh`，`--no_gate` 必须显式传）。
#      `judge_pairs.py` 报 `resolve_rate` 与 `p_vs_floor`（FLOOR = 21/118 = 17.8%）。
#      若 `p_vs_floor >= 0.05` 或 `resolve_rate <= FLOOR` -> **`VOID_UNRESOLVABLE`：不报胜负、不下结论**
#      （⛔ 不许记成"打平"）。
#  (2) **主判据 = 判官**（判官是主尺子；CLIP 只用来定序，FD 已退出优化目标）：
#      A=`gen10_rr4` 对 B=`gen8_rr4` 的胜率，按 (M17) 立的规矩报**族级 (U1) CI**
#      （新 full 先登进 `analysis/arch/recheck_judge.py:EXPECT`，再跑族级自助）。
#      裸二项 p 只能当**下界**报，⛔ 不许单独拿它下判。
#  (3) **次要报告项（⛔ 非判据、⛔ 不许替 (2) 下判）**：`run_eval.py` 的 16px KID/FID/FD/CLIP 四项，
#      两臂之差对照 (M33) 的 m=1 下限 KID **4.656** / FID **6.0** / FD **20** / CLIP **0.41**。
#      差值小于下限的一律读作"**没测到**"（(M14) 的规矩：差 0.0002 也是没过）。
#      ⚑ 这一格特别要紧：当年就是这四把尺子替我们选的 v8 —— 本轮**在今天的配方上再量一次**，
#      看它们是不是照旧全瞎。⛔ 但无论它们说什么，都不许拿来替代 (2)。
#  (4) **四种结果的读法全部写在前面，事后不许再编第五种**：
#      - A 的族级 CI **下界 > 0.5** -> **`GEN_V10_WINS`**：16px 主配方当年选错了代次。
#        **授权（仅此一项、仅此一行）**：把 `eval/final_test.sh` 里 **TRD16c 那一条**生成命令的
#        `--run $V8` 改成 `--run $V10`。
#        ⛔ TRD16（`--tag TRD16` 那两行）、TRD24、TRD32、颜色任务、B7 **一个字不动**
#        —— 那要各自另一条臂、另行预注册。⛔ 也不许顺手改 `gen_trd.py` 的任何默认值。
#      - A 的族级 CI **上界 < 0.5** -> **`GEN_V8_WINS`**：现行 v8 **首次**被现行判官验证，
#        账上第三个"靠噪声选出来的配置"就此填上，且与 (M37) 一起构成"再多训一段不总是更好"的边界。
#        ⛔ `final_test.sh` 不改。
#      - 族级 CI **含 0.5** -> **`GEN_NULL`**：只等于"**在 n≈79 上没测到 >=62% 的效应**"。
#        ⛔ 不许记成"打平"、⛔ 不许据此说"代次无所谓"、⛔ 不许据此改任何默认值。
#      - 可解率不过地板 -> 见 (1)。
#  (5) **方向预测（写在跑之前）**：预测 **`GEN_V10_WINS`**，次可能 `GEN_NULL`。
#      理由：(M37) 在这条管线上刚测到"同一支再多训一段、晚的显著胜"，而 v10 正是 v8 再训 12000 步
#      （且 16px 也在训），当年判它"略退"的四把尺子逐项都在噪声下限之下。
#      **若 `GEN_V8_WINS`，这条预测当场记为被推翻**，并且要连带承认两件事：
#      ①当年那次在噪声里做的选择**碰巧选对了**；②(M37) 的"晚的赢"**不能外推到跨 run 的续训**
#      —— 差别很可能来自 v10 把 32px 批次比例从 0.3 提到 0.5（16px 反而少练了）。
#  (6) **操作检验（任一条不过，本臂作废）**：
#      (OP1) 两个 tag 目录各 500 张（125 材质 x n=4）、两个 `_rr4` 目录各 125 张；
#      (OP2) 两臂命令逐字相同，唯一差异是 `--run`（脚本把两条命令原样打印出来备查）；
#      (OP3) 两份 `codebook.npy` 的 md5 必须相同（否则权重与调色板码对不上，整臂无意义）；
#      (OP4) 两臂 `_rr4` 产物**逐像素相同的材质数 <= 25/125**（若两边几乎同图，判官只能瞎答，
#            这时 (2) 的胜率不可解读 -> 作废）；
#      (OP5) `run_eval.py` 里两臂 `n` 与 `materials` 必须相等（`run_eval.py:100` 的 ok 是各算各的）；
#      (OP6) `api_fail == 0`（判官阶段）。
#  (7) **本轮不改任何默认值**：`judge_pairs.py` 的 `min_rate=0.65` 一个字不动（`--no_gate` 显式传），
#      `gen_trd.py`/`rerank.py`/`run_eval.py`/`train_trd.py`/`model/trd.py`/`UNITS_PER_TILE` 全不动。
#      唯一允许的仓库改动是 `scripts/sync_remote_tmp.sh` 的白名单加 `m4[0-9]_*.json`
#      （(M39) 的教训：新产物命名必须落在同步列表里；⚠ 加完仍须手动 scp 补拉一次，增量同步不回溯）。
#  (8) 成本上限：GPU 两次 16px 生成（各约 500 张，参照 359 张/分，实占约 5GB 显存）+ 判官至多 125 对 x 2 序。
#      `api_fail` 率 > 25% 就停下记账、不补跑。
#  (9) **(M39) 那把 pixfrac 尺子在本轮只记录、⛔ 不做门**，理由写在前面免得事后看着像挑读法：
#      (M39) 实测它**分辨力不足**（三个非零读数全挤在 0.66-0.82，两锚点只隔 0.157，两条挑战者都掉进
#      "两者之间"），它自己的结论就是"这把筛子既提拔不了也否不掉任何东西"。本轮按 (M39) 结账时给的
#      两条出路里的第二条走：**直接按"值不值"论证去预注册判官臂**（理由 (a)-(e)），
#      筛子读数降格为**跑前预测的记分项**：跑前预测 **d(gen10,gen8) >= D37 = 0.8173**
#      （换权重的差别不小于换检查点）。⚠ 这条预测**不影响任何判决**，对错都只记在账本上。
#      ⚑ 附带好处：无论本臂判成什么，它都给那把两锚点的尺子添上**第三个已判过的锚点**
#      —— 正是 (M39) 结账时说"下次要筛，锚点 >= 3 个"所缺的那一个。
#
# ================================================================ 四、运行约束
# `/mnt/data` 长期贴满 -> 产物全写 `/tmp`（`--out`/`--root`/`--outdir`），结果 scp 回本机入库。
# 指标 JSON 显式命名 `/tmp/m40_gen16.json`、尺子读数 `/tmp/m40_gen16_meter.json`；
# 判官 JSON 落 `/tmp/judge_gen16ab/`（`judge_*` 已在白名单）。
# 选卡只看 `memory.free`（16px 生成实占约 5GB；⛔ 别看 utilization）。
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
# 起法（两段；不给 STAGE 或 STAGE=gen 只出图算指标 = 零 API）。命令必须**先写进文件再跑**：
#   tmux new-session -d -s arch_gen16 "bash /tmp/run_gen16.sh"
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
O=${OUT:-/tmp/gen16ab}
J=${JOUT:-/tmp/judge_gen16ab}
STAGE=${STAGE:-gen}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton
mkdir -p "$O" "$J"

V8=runs/trd_v8
V10=runs/trd_v10
COMMON="--ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --out $O"

if [ "$STAGE" = gen ]; then
  "$P" -c 'import numpy, PIL, torch, sys; sys.exit(0)' || { echo "[gen16ab] python 探针失败: $P"; exit 1; }
  # (OP3)：码本 md5 必须相同
  md5sum $V8/codebook.npy $V10/codebook.npy
  A=$(md5sum $V8/codebook.npy | cut -d' ' -f1)
  B=$(md5sum $V10/codebook.npy | cut -d' ' -f1)
  [ "$A" = "$B" ] || { echo "[OP3] 【禁】码本 md5 不同 $A != $B -> 本臂作废"; exit 1; }
  echo "[OP3] 码本 md5 相同: $A"
  # (OP2)：两条命令原样打印，唯一差异必须是 --run
  echo "[gen16ab] A: eval/gen_trd.py --run $V10 $COMMON --tag gen10"
  echo "[gen16ab] B: eval/gen_trd.py --run $V8 $COMMON --tag gen8"
  $P eval/gen_trd.py --run $V10 $COMMON --tag gen10 || exit 1
  $P eval/gen_trd.py --run $V8  $COMMON --tag gen8  || exit 1
  for t in gen10 gen8; do
    $P eval/rerank.py --src $t --set V_mat --size 16 --n 4 --root "$O" || exit 1
  done
  # (OP1)+(OP4)+判据 (9) 的尺子读数；外加一条白拿的确定性复核（见下）
  $P - "$O" <<'PYEOF' || exit 1
import sys, pathlib, json
import numpy as np
from PIL import Image
root = pathlib.Path(sys.argv[1]); bad = []; checked = 0
for t, want in [("gen10", 500), ("gen8", 500), ("gen10_rr4", 125), ("gen8_rr4", 125)]:
    n = len(list((root / t / "16").glob("*.png")))
    print("[OP1] %-11s %d 张 (应为 %d)" % (t, n, want))
    checked += 1
    if n != want:
        bad.append("OP1:%s=%d" % (t, n))

def compare(da, db):
    """返回 (n, 逐像素全同数, pixfrac 均值, mae 均值)；缺目录返回 None。"""
    da, db = pathlib.Path(da), pathlib.Path(db)
    if not da.is_dir() or not db.is_dir():
        return None
    fr, ae, ident, n = [], [], 0, 0
    for f in sorted(da.glob("*.png")):
        g = db / f.name
        if not g.exists():
            continue
        x = np.asarray(Image.open(f).convert("RGB")).astype(np.int32)
        y = np.asarray(Image.open(g).convert("RGB")).astype(np.int32)
        d = np.abs(x - y)
        nd = int((d.reshape(-1, 3).sum(axis=-1) > 0).sum())
        fr.append(nd / d.reshape(-1, 3).shape[0]); ae.append(float(d.mean()) / 255.0)
        ident += (nd == 0); n += 1
    return (n, ident, float(np.mean(fr)) if n else None, float(np.mean(ae)) if n else None)

m = compare(root / "gen10_rr4" / "16", root / "gen8_rr4" / "16")
meter = {"pair": "gen10_rr4 vs gen8_rr4"}
if m is None or m[0] == 0:
    bad.append("OP4:no_data"); checked += 1
else:
    n, ident, pf, mae = m
    meter.update({"n": n, "identical": ident, "pixfrac": pf, "mae": mae})
    print("[OP4] 两臂逐像素相同的材质 %d/%d（门槛 <=25）" % (ident, n))
    print("[判据9-记录项] pixfrac=%.4f mae=%.4f （跑前预测 >= D37 0.8173；不做门）" % (pf, mae))
    checked += 1
    if ident > 25:
        bad.append("OP4:same=%d" % ident)

# 白拿的确定性复核：gen8 与 (M38) cfg15 / (M39) ret100 的配方逐字相同，若那两份还在 /tmp 就该 125/125 同图。
# (M31) 的坑：缺数据必须明说"没查"，⛔ 不许拿空集冒充"查过没事"。
det = {}
for name, other in [("cfg15", "/tmp/cfgab16/cfg15_rr4/16"), ("ret100", "/tmp/retnn16/ret100_rr4/16")]:
    r = compare(root / "gen8_rr4" / "16", other)
    if r is None or r[0] == 0:
        det[name] = "not_checked(缺目录)"
        print("[确定性复核] %-7s 未查（%s 不在）" % (name, other))
    else:
        det[name] = {"n": r[0], "identical": r[1], "pixfrac": r[2]}
        print("[确定性复核] %-7s 同图 %d/%d" % (name, r[1], r[0]))
        checked += 1
        if r[1] != r[0]:
            bad.append("DET:%s=%d/%d" % (name, r[1], r[0]))
meter["determinism_vs_prior_arms"] = det
meter["problems"] = bad
meter["n_checked"] = checked
pathlib.Path("/tmp/m40_gen16_meter.json").write_text(
    json.dumps(meter, ensure_ascii=False, indent=1), encoding="utf-8")
print("[操作检验] 已查 %d 项，problems=%d %s" % (checked, len(bad), bad))
print("-> /tmp/m40_gen16_meter.json")
sys.exit(1 if bad else 0)
PYEOF
  $P eval/run_eval.py --set V_mat --size 16 --root "$O" --out /tmp/m40_gen16.json \
     --methods gen10_rr4 gen8_rr4 gen10 gen8 || exit 1
  echo GEN16AB_GEN_DONE
fi

if [ "$STAGE" = judge ]; then
  [ -n "${VLM_BASE_URL:-}" ] && [ -n "${VLM_API_KEY:-}" ] || { echo "[gen16ab] 缺 VLM 凭据"; exit 1; }
  while ps -eo args | grep -q '[j]udge_pairs.py'; do
    echo "[gen16ab] 别的判官在跑，300s 后再看"; sleep 300
  done
  $P eval/judge_pairs.py full --no_gate --a gen10_rr4 --b gen8_rr4 \
     --set V_mat --size 16 --root "$O" --outdir "$J" || exit 1
  echo GEN16AB_JUDGE_DONE
fi
