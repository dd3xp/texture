#!/usr/bin/env bash
# (M55) 操作检验 (OP1) 的**跑前**版本：render_refs 的提示词集合必须覆盖 train_trd 要的每一个材质。
# 少一个就会死在 train_trd.py:383。零 GPU（--list_only 不加载 SDXL）。
set -e
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
for f in scripts/m55_refs.sh scripts/m55_train.sh; do tr -d '\r' < "$f" > /tmp/m55x && mv /tmp/m55x "$f"; done
$PY -u baselines/render_refs.py --extra train_extra_packs_only.json --list_only
$PY - <<'PYEOF'
import sys
sys.path.insert(0, "model"); sys.path.insert(0, "eval"); sys.path.insert(0, "baselines")
from tiles_data import load
from prompts import prompt_words
from render_refs import all_prompts
EX = "train_extra_packs_only.json"
mats = set()
for size, split, ex in ((16, "train", EX), (16, "val", False), (16, "test", False),
                        (32, "train", EX), (32, "val", False)):
    mats |= {s["material"] for s in load(size, split, extra=ex)}
need = {" ".join(prompt_words(m)) or m for m in mats}      # train_trd.py:384 的键
have = set(all_prompts(EX))
miss = need - have
print(f"train_trd 需要 {len(need)} 个提示词；render_refs 会产出 {len(have)} 个；缺 {len(miss)}")
print("OP1_PRECHECK_PASS" if not miss else f"OP1_PRECHECK_FAIL {sorted(miss)[:5]}")
PYEOF
