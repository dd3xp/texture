#!/usr/bin/env bash
# `--p_tile16` 那轮的操作检验：集合（E_mat/V_mat）× extra 文件（v10 的 / v11d 起真正在用的）2×2 盘点。
# 零 GPU、零 API，只读训练 split 的 JSON。判据写在 analysis/arch/pool32_coverage.py 文件头（跑前写死）。
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
X=train_extra_packs_only.json+train_64to32.json
OUT=${1:-/tmp/pool32_coverage_2x2.txt}
{
  echo "##### [1] E_mat + 旧 extra（v10 用的，= 6e2a95f 逐字复现）"
  $P analysis/arch/pool32_coverage.py --set E_mat
  echo; echo "##### [2] E_mat + 训练实际 extra（v11d 起）"
  $P analysis/arch/pool32_coverage.py --set E_mat --extra "$X"
  echo; echo "##### [3] V_mat + 旧 extra"
  $P analysis/arch/pool32_coverage.py --set V_mat
  echo; echo "##### [4] V_mat + 训练实际 extra（= --p_tile16 那轮真正在量的口径）"
  $P analysis/arch/pool32_coverage.py --set V_mat --extra "$X"
} > "$OUT" 2>&1
echo "wrote $OUT"
