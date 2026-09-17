#!/usr/bin/env bash
# (M55) 第二步：等两个渲染分片齐了，用 v10 的配方 + `--refs` 训实验臂。
# 配方逐键抄自 runs/trd_v10/config.json（(M41) 的"逐键对一遍"），只多 --refs/--p_ref_drop，
# 外加 (M32) 的 --reseed_after_build（成对配方；对照 = 已有的 runs/trd_v10，不重训）。
set -e
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
REFS=/tmp/m55_refs
OUT=/tmp/runs/trd_refs_09180330

# 等料：只看文件在不在（退出码判据，不比字符串 —— (M44) 踩过 `grep -c` 打印 0 的坑）。
# 上限 6 小时；到点没齐就 exit 3，别无限等。
for i in $(seq 1 360); do
  if [ -f "$REFS/shard0.done" ] && [ -f "$REFS/shard1.done" ]; then break; fi
  sleep 60
done
if [ ! -f "$REFS/shard0.done" ] || [ ! -f "$REFS/shard1.done" ]; then
  echo "M55_TRAIN_ABORT_NO_REFS"; exit 3
fi

mkdir -p /tmp/runs
$PY -u model/train_trd.py --out "$OUT" \
    --steps 12000 --batch 256 --lr 1.5e-4 --warmup 500 \
    --sizes 16 32 --p32 0.5 --batch32 64 \
    --bias_freqs 8 --bias_hidden 128 --level_emb --pal_aug 0.3 --pal_smooth 0.1 \
    --extra --extra_file train_extra_packs_only.json \
    --coarse --p_coarse 0.5 \
    --init_from runs/trd_v8/last.pt --reseed_after_build \
    --refs "$REFS" --p_ref_drop 0.3 \
    --save_at 6000 12000
echo "M55_TRAIN_DONE $OUT"
