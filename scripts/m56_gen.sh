#!/usr/bin/env bash
# (M56) 出图：同检查点、同 seed、除 `--refs` 外逐字相同的三臂。
# 判据一个字不在本文件里定：全文在 docs/arch_progress.md 的 (M56) 节，
# 判读器 analysis/arch/m56_read_refcond.py（盲写、--selftest 18/18）。
#
# 起法（远程；卡号与 PATH 都写在本文件里，tmux 两个都不继承）：
#   tmux new-session -d -s m56_gen "bash /tmp/m56_gen.sh"
set -u
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
REFRUN=/tmp/runs/trd_refs_09180330
OUT=/tmp/m56_gen

# 0) 臂 A = (M55) 已出的图，原样搬过来（命令与本轮三臂逐字相同，只差 --tag）
for S in 32 16; do
  mkdir -p "$OUT/m56refs/$S"
  cp /tmp/m55_gen/m55refs/$S/*.png "$OUT/m56refs/$S/" || { echo M56_ABORT_NO_ARM_A_$S; exit 3; }
done

# 1) 错配参考嵌入 + 操作检验数
$PY -u scripts/m56_shuffle_refs.py --src /tmp/m55_refs --out /tmp/m56_refs_shuf \
    --stats /tmp/m56_refstats.json --set V_mat || { echo M56_ABORT_SHUFFLE; exit 3; }

G="$PY -u eval/gen_trd.py --run $REFRUN --ckpt last.pt --set V_mat --bs 8 --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --out $OUT"

# 2) 臂 B（错配）：主判据那档先跑
$G --refs /tmp/m56_refs_shuf --size 32 --tag m56shuf || { echo M56_ABORT_GEN_SHUF_32; exit 4; }
# 3) 臂 A'（原样重跑）＝ 确定性操作检验，只在 32px
$G --refs /tmp/m55_refs      --size 32 --tag m56rerun || { echo M56_ABORT_GEN_RERUN_32; exit 4; }
# 4) 16px（只登记，不下判）
$G --refs /tmp/m56_refs_shuf --size 16 --tag m56shuf || { echo M56_ABORT_GEN_SHUF_16; exit 4; }

echo M56_GEN_DONE
