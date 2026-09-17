#!/bin/bash
# (M52) 上界臂：pal_real（oracle 调色板）vs pal_xmodal（正式配置），同一张网格。
# 凭据只从 tmux -e 注入的环境变量读，不落盘。先探针（2 次调用），探针不过就不烧 full。
set -e
export HF_HUB_OFFLINE=1
cd /mnt/data/kw/RoundSquisheen/texture
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
P=/tmp/judge_pilot_pal_real_vs_pal_xmodal_16_V_mat.json
rm -f "$P"
$PY -u eval/judge_pairs.py pilot --a pal_real --b pal_xmodal --size 16 --set V_mat \
  --root /tmp/m52_dump --outdir /tmp --n_pilot 1 --n_null 0 --min_rate 0.0 || true
$PY - "$P" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
ok = d["n_real"] == 1
print("PROBE_OK" if ok else "PROBE_FAIL", flush=True)
sys.exit(0 if ok else 3)
PYEOF
$PY -u eval/judge_pairs.py full --a pal_real --b pal_xmodal --size 16 --set V_mat \
  --no_gate --root /tmp/m52_dump --outdir /tmp
echo "M52_JUDGE_DONE"
