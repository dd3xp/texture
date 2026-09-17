#!/usr/bin/env bash
# =====================================================================================
# (M41) 第二阶段：出图 → 零 API 筛子（**当门用**）→ 只在 SCREEN_GO 时开判官臂
# 判据全文在 `eval/m41_clip_loss_ab.sh`（第一阶段，已提交 = 预注册）。本文件同样在
# **任何一张图存在之前**提交；⛔ 跑完不许改配方、不许挪门槛、不许换尺子。
#
# 配方逐字照抄 (M37)(M38)(M40) 用的 TRD16c：16px / V_mat / --cfg 1.5 /
# --pal_mode retrieve --xmodal --ret_nname 100 / --n 4 + rerank.py --n 4 / --no_gate。
# 唯一变量 = `--run`（A = 带训练侧 CLIP 损失的续训，B = 同配方不带）。
#
# ⚠⚠ 门写进代码里执行：judge 阶段**读** meter JSON 的 screen 字段，
#    不是 `SCREEN_GO` 就直接退出、一次 API 都不发 —— 这样"谁在什么时候执行这条读法"是写死的
#    （账本的教训：盲写的读法必须连执行者一起写死，否则三轮没人查表）。
# =====================================================================================
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
REPO=${REPO:-/mnt/data/kw/RoundSquisheen/texture}
TAG=${TAG:?必须传 TAG（与第一阶段同一个）}
O=${OUT:-/tmp/m41ab}
J=${JOUT:-/tmp/judge_m41ab}
STAGE=${STAGE:-gen}
MET=${MET:-/tmp/m41_meter.json}
cd "$REPO"
export HF_HUB_OFFLINE=1 TRITON_CACHE_DIR=/tmp/triton_m41
mkdir -p "$O" "$J"

RA=runs/trd_clipw_$TAG          # A：加了训练侧 CLIP 对齐损失
RB=runs/trd_ctl_$TAG            # B：同配方控制臂
COMMON="--ckpt last.pt --set V_mat --size 16 --n 4 --bs 16 --cfg 1.5 --pal_mode retrieve --xmodal --ret_nname 100 --out $O"

if [ "$STAGE" = gen ]; then
  "$P" -c 'import numpy, PIL, torch, sys; sys.exit(0)' || { echo "[m41] python 探针失败: $P"; exit 1; }
  # 两臂都是从 v8 续训 ⇒ 码本必须与 v8 逐字节相同（否则调色板码对不上，(M41) 的 A−B 不可解读）
  md5sum runs/trd_v8/codebook.npy $RA/codebook.npy $RB/codebook.npy
  H=$(md5sum runs/trd_v8/codebook.npy | cut -d' ' -f1)
  for d in $RA $RB; do
    [ "$(md5sum $d/codebook.npy | cut -d' ' -f1)" = "$H" ] || { echo "[OP] 【禁】$d 码本与 v8 不同 -> 本臂作废"; exit 1; }
  done
  echo "[OP] 三份码本 md5 相同: $H"
  # (OP1)(OP2) 两条命令原样打印，唯一差异必须是 --run
  echo "[m41] A: eval/gen_trd.py --run $RA $COMMON --tag clipw"
  echo "[m41] B: eval/gen_trd.py --run $RB $COMMON --tag ctl"
  $P eval/gen_trd.py --run $RA $COMMON --tag clipw || exit 1
  $P eval/gen_trd.py --run $RB $COMMON --tag ctl   || exit 1
  for t in clipw ctl; do
    $P eval/rerank.py --src $t --set V_mat --size 16 --n 4 --root "$O" || exit 1
  done
  $P - "$O" "$MET" "$RA" "$RB" <<'PYEOF' || exit 1
import sys, pathlib, json
import numpy as np
from PIL import Image
root, out, ra, rb = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
bad, checked = [], 0
for t, want in [("clipw", 500), ("ctl", 500), ("clipw_rr4", 125), ("ctl_rr4", 125)]:
    n = len(list((root / t / "16").glob("*.png")))
    print("[OP1] %-10s %d 张 (应为 %d)" % (t, n, want))
    checked += 1
    if n != want:
        bad.append("OP1:%s=%d" % (t, n))


def compare(da, db):
    """(n, 逐像素全同数, pixfrac 均值, mae 均值)；缺目录返回 None。"""
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


# ---- 训练侧操作检验 (OP2)(OP3)(OP4)：两臂的 log.json 与 config.json
D37, D38 = 0.8173, 0.6608            # ⛔ 锚点写死，不许改
meter = {"pair": "clipw_rr4 vs ctl_rr4", "anchors": {"D37": D37, "D38": D38, "D40": 0.6431}}
try:
    ca = json.load(open(pathlib.Path(ra) / "config.json", encoding="utf-8"))
    cb = json.load(open(pathlib.Path(rb) / "config.json", encoding="utf-8"))
    diff = sorted(k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k))
    allowed = {"out", "clip_loss", "clip_w", "clip_bs", "codebook_err"}
    print("[OP1-cfg] 两臂 config 差异键: %s" % diff)
    meter["cfg_diff"] = diff
    checked += 1
    if set(diff) - allowed:
        bad.append("OP1cfg:%s" % (sorted(set(diff) - allowed),))
    la = json.load(open(pathlib.Path(ra) / "log.json", encoding="utf-8"))
    lb = json.load(open(pathlib.Path(rb) / "log.json", encoding="utf-8"))
    # (OP2) 成对性：step 0 的 loss_grid / loss_pal 必须相同（同一批、同一套掩码）
    d0 = max(abs(la[0]["loss_grid"] - lb[0]["loss_grid"]), abs(la[0]["loss_pal"] - lb[0]["loss_pal"]))
    print("[OP2] step0 loss_grid/loss_pal 最大差 %.3e（门槛 1e-5）" % d0)
    meter["op2_step0_maxdiff"] = d0
    checked += 1
    if not (d0 < 1e-5):
        bad.append("OP2:step0diff=%.3e" % d0)
    # (OP3) 损失确实在被优化：clip_cos 首两点均值 < 末两点均值
    cs = [r["clip_cos"] for r in la if "clip_cos" in r]
    meter["clip_cos_first2"] = float(np.mean(cs[:2])) if len(cs) >= 4 else None
    meter["clip_cos_last2"] = float(np.mean(cs[-2:])) if len(cs) >= 4 else None
    checked += 1
    if len(cs) < 4:
        bad.append("OP3:no_clip_cos")
    else:
        print("[OP3] clip_cos 首两点 %.4f -> 末两点 %.4f（只确认损失通了，⛔ 非效应证据）"
              % (meter["clip_cos_first2"], meter["clip_cos_last2"]))
        if not (meter["clip_cos_last2"] > meter["clip_cos_first2"]):
            bad.append("OP3:clip_cos_not_up")
    # (OP4) 都跑满 6000 步
    meter["last_step"] = {"A": la[-1]["step"], "B": lb[-1]["step"]}
    meter["val_last"] = {"A": la[-1]["val"], "B": lb[-1]["val"]}
    print("[OP4] 末步 A=%d B=%d；末 val A=%.4f B=%.4f（⛔ val 非判据，见判据 (P3)）"
          % (la[-1]["step"], lb[-1]["step"], la[-1]["val"], lb[-1]["val"]))
    checked += 1
    if la[-1]["step"] != 6000 or lb[-1]["step"] != 6000:
        bad.append("OP4:steps=%d/%d" % (la[-1]["step"], lb[-1]["step"]))
except FileNotFoundError as e:
    bad.append("OPtrain:missing(%s)" % e.filename)
    checked += 1
    print("[OP训练侧] 缺文件 %s -> 已查 %d 项、记 problem，⛔ 不当作'查过没事'" % (e.filename, checked))

# ---- (OP5) + 判据 (5) 的门
m = compare(root / "clipw_rr4" / "16", root / "ctl_rr4" / "16")
if m is None or m[0] == 0:
    bad.append("OP5:no_data"); checked += 1
    meter["screen"] = "VOID_NO_DATA"
    print("[OP5] 【禁】没量到任何一对图 -> screen=VOID_NO_DATA（⛔ 这不是'没事'）")
else:
    n, ident, pf, mae = m
    meter.update({"n": n, "identical": ident, "pixfrac": pf, "mae": mae})
    print("[OP5] 两臂逐像素相同的材质 %d/%d（门槛 <=25）" % (ident, n))
    checked += 1
    if ident > 25:
        bad.append("OP5:same=%d" % ident)
    meter["screen"] = ("SCREEN_GO" if pf >= D37 else
                       "SCREEN_NOGO" if pf <= D38 else "SCREEN_AMBIGUOUS")
    print("[判据5-门] pixfrac=%.4f mae=%.4f  D38=%.4f < d < D37=%.4f ?  -> %s"
          % (pf, mae, D38, D37, meter["screen"]))
    print("[判据5-门] 跑前预测 (P1) 是 d >= %.4f = SCREEN_GO" % D37)
meter["problems"] = bad
meter["n_checked"] = checked
out.write_text(json.dumps(meter, ensure_ascii=False, indent=1), encoding="utf-8")
print("[操作检验] 已查 %d 项，problems=%d %s" % (checked, len(bad), bad))
print("-> %s" % out)
sys.exit(1 if bad else 0)
PYEOF
  $P eval/run_eval.py --set V_mat --size 16 --root "$O" --out /tmp/m41_gen16.json \
     --methods clipw_rr4 ctl_rr4 clipw ctl || exit 1
  echo M41_GEN_DONE
fi

if [ "$STAGE" = judge ]; then
  # 门：不是 SCREEN_GO 就一次 API 都不发
  # 门 0（2026-09-17 补，A 臂尚在训练、一张图都不存在时）：预注册第一阶段 §4 写的是
  # 「(OP1)–(OP5) 任一不过 ⇒ 判 VOID」，但**没有任何代码读它**：gen 阶段确实会因 problems 非空
  # 而 exit 1，可看门狗是换行接的两条命令（不是 &&）⇒ 退出码没人接，judge 阶段照样开跑
  # ＝一条已被预注册判作废的臂仍会烧掉一小时 API。补这一条只是**执行**已写死的作废规则：
  # ⛔ 它只会更严、且是单向的（只能少花 API、不能让任何臂变成"过"），⛔ 不是改判据、⛔ 不是挪门槛、
  # ⛔ 没碰筛子的三个锚点。账本的老教训：**没人读的退出码不是门**。
  NP=$("$P" -c "import json,sys;m=json.load(open(sys.argv[1],encoding='utf-8'));print(len(m['problems']) if 'problems' in m else 99)" "$MET") || exit 1
  echo "[m41] 操作检验 problems 数 = $NP（99 = meter 里连这个键都没有，同样作废）"
  [ "$NP" = 0 ] || { echo "[m41] 【禁】(OP1)-(OP5) 有 $NP 项不过 -> 按预注册 §4 本臂 VOID，不开判官、不花 API"; exit 4; }
  S=$("$P" -c "import json,sys;print(json.load(open(sys.argv[1],encoding='utf-8')).get('screen'))" "$MET") || exit 1
  echo "[m41] 筛子判决 = $S"
  [ "$S" = SCREEN_GO ] || { echo "[m41] 【禁】非 SCREEN_GO -> 按预注册不开判官臂、不花 API，记账走人"; exit 3; }
  [ -n "${VLM_BASE_URL:-}" ] && [ -n "${VLM_API_KEY:-}" ] || { echo "[m41] 缺 VLM 凭据"; exit 1; }
  # (M40) 的教训：失效 key 会静默烧掉一小时 -> 先探针
  curl -s -o /dev/null -w '[m41] key 探针 HTTP %{http_code}\n' \
    -H "Authorization: Bearer $VLM_API_KEY" "$VLM_BASE_URL/models"
  while ps -eo args | grep -q '[j]udge_pairs.py'; do
    echo "[m41] 别的判官在跑，300s 后再看"; sleep 300
  done
  $P eval/judge_pairs.py full --no_gate --a clipw_rr4 --b ctl_rr4 \
     --set V_mat --size 16 --root "$O" --outdir "$J" || exit 1
  echo M41_JUDGE_DONE
fi
