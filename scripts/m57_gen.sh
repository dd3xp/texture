#!/usr/bin/env bash
# (M57) 出图：在**有 32px 真人瓦片的训练材质**（T32b，240 个）上出 v10 的 32px 与 16px 产物。
# 判据一个字不在本文件里定：全文在 docs/arch_progress.md 的 (M57) 预注册节，
# 判读器 analysis/arch/m57_read_canvas.py（盲写、--selftest 26/26）。
#
# 起法（远程；卡号与 PATH 都写在本文件里，tmux 两个都不继承）：
#   tmux new-session -d -s m57_gen "bash /tmp/m57_gen.sh"
set -u
export CUDA_VISIBLE_DEVICES=6
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
OUT=/tmp/m57_gen

# 0) 提示词集：T32b 必须已在 eval/prompt_sets_train.json 里且与重算一致（--check 只校验、不写）
$PY -u analysis/arch/m57_build_t32set.py --check || { echo M57_ABORT_SET; exit 3; }

# 1) 配方＝(M55) 那条 v10 臂逐字相同，只改 --set / --out / --tag
#    （唯一省掉的是 --refs：v10 没有 ref_proj，gen_trd.py:147 那一路根本不读它）
G="$PY -u eval/gen_trd.py --run runs/trd_v10 --ckpt last.pt --set T32b --bs 8 --n 2 --cfg 1.5 --pal_mode retrieve --xmodal --out $OUT"

$G --size 32 --tag m57t32 || { echo M57_ABORT_GEN_32; exit 4; }   # 主判据那档
$G --size 16 --tag m57t32 || { echo M57_ABORT_GEN_16; exit 4; }   # 作废条件 3（尺子非平凡）

echo M57_GEN_DONE
