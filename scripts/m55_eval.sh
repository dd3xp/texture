#!/usr/bin/env bash
# (M55) 第三步：训练收尾后自动出图 + 算指标，供预注册第四节的判据读。
# 判据一个字不在本文件里定：全文在 docs/arch_progress.md 的 (M55) 第四节（预注册 `ed75a9c`），
# 判读器 `analysis/arch/m55_read_refs.py`（盲写、`--selftest` 11/11）。
#
# 起法（远程，卡号写在本文件里）：
#   tmux new-session -d -s m55_ev "bash /tmp/m55_eval.sh"
#
# ⚠ 规矩（记忆里踩过的坑，逐条对应）：
#   - 卡号/PATH 都写进本文件（tmux 既不继承 CUDA_VISIBLE_DEVICES 也不继承 PATH）
#   - 产物全写 /tmp（/mnt/data 97% 满）；JSON 名落在 sync_remote_tmp.sh 的 m5[0-9]_*.json 白名单里
#   - 等的是**进程/标记串**，只用 grep -q 的退出码，不拿 stdout 比字符串
#   - TRITON_CACHE_DIR 指 /tmp，否则 deepspeed 在 atexit 里写表失败、活干完了才崩
#
# (OP4)：两臂生成命令**除 `--run` 外逐字相同**（`--refs` 也照给 v10 臂 —— `gen_trd.py:147`
# 只在 `model.ref_proj is not None` 时才读它，v10 没有这一路，所以给了等于没给）。
set -u
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export REFRUN=${REFRUN:-/tmp/runs/trd_refs_09180330}
OUT=/tmp/m55_gen
LOG=/tmp/m55_train.txt

# 1) 等训练收尾（上限 6 小时；只看标记串在不在）
for i in $(seq 1 360); do
  grep -q M55_TRAIN_DONE "$LOG" && break
  grep -q M55_TRAIN_ABORT_NO_REFS "$LOG" && { echo M55_EVAL_ABORT_NO_TRAIN; exit 3; }
  sleep 60
done
grep -q M55_TRAIN_DONE "$LOG" || { echo M55_EVAL_ABORT_TIMEOUT; exit 3; }

# 2) (OP2)：新检查点确实带参考图这一路，否则本轮白跑
$PY - <<'EOF' || { echo M55_EVAL_ABORT_OP2; exit 3; }
import json, os, sys
c = json.load(open(os.environ.get("REFRUN", "/tmp/runs/trd_refs_09180330") + "/config.json"))
print("OP2 config refs =", c.get("refs"), "ref_dim =", c.get("ref_dim"))
sys.exit(0 if c.get("refs") else 1)
EOF

# 3) 出图 + 算指标：32px（主判据）-> 16px（护栏）-> 24px（次要），先把要下判的那档跑出来
G="$PY -u eval/gen_trd.py --ckpt last.pt --set V_mat --bs 8 --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --refs /tmp/m55_refs --out $OUT"
for S in 32 16 24; do
  $G --run "$REFRUN"     --size $S --tag m55refs || { echo M55_EVAL_ABORT_GEN_REFS_$S; exit 4; }
  $G --run runs/trd_v10  --size $S --tag m55v10  || { echo M55_EVAL_ABORT_GEN_V10_$S;  exit 4; }
  $PY -u eval/run_eval.py --set V_mat --size $S --methods m55refs m55v10 \
      --root "$OUT" --out /tmp/m55_eval_Vmat_$S.json || { echo M55_EVAL_ABORT_EVAL_$S; exit 5; }
  echo "M55_EVAL_SIZE${S}_DONE"
done
echo M55_EVAL_DONE
