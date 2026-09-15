#!/usr/bin/env bash
# (M28) 预注册 67b8a04，全文＝analysis/arch/comb_shift.py 的 docstring。
# 问题：(M27) 判出的"归一化表是产物周期的把手"只在**未训练的 24px** 上成立过，而它自己写死了
#       "⛔ 不许外推成 32px 上表也是把手"。可重训的整条理由链恰恰压在 32px 上。
# 干预：同一个检查点、权重一个字不改，只在推理期把偏置表的唯一输入 u=wrap(d)/32 换成 wrap(d)/16
#       （`--bias_n 16`）。H_table -> 梳齿搬到 d=4；H_content -> 不动。
#
# 三臂（提示词、种子、调色板、范例、批次全同，唯一变量是 --bias_n）：
#   m28_ctl   控制臂（--bias_n 不给 = 关）
#   m28_over  干预臂（--bias_n 16）
#   m28_noop  空操作臂（--bias_n 32 = 换成它本来的值）→ (OP1) 必须与 m28_ctl 逐字节相同
#
# ⛔ 零判官 / 零 API / 零训练 / 零配置改动；`--bias_n` 默认 0，`final_test.sh` 一个字没动。
# 产物写 /tmp（/mnt/data 长期贴满）；目录名以 gen32 开头 = 落在 scripts/sync_remote_tmp.sh 的同步列表里。
#
# 起法：tmux new-session -d -s arch_m28 "bash scripts/m28_comb_shift.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_m28
OUT=/tmp/gen32_m28

G="$P -u eval/gen_trd.py --run runs/trd_v10 --ckpt last.pt --set E_mat --size 32 --cfg 1.5 \
   --pal_mode retrieve --xmodal --ret_nname 100 --bs 8 --out $OUT"

$G --n 2 --tag m28_ctl               || exit 1
$G --n 2 --tag m28_over --bias_n 16  || exit 1
$G --n 1 --tag m28_noop --bias_n 32  || exit 1

$P -u analysis/arch/comb_shift.py --gen_root $OUT --out /tmp/comb_shift.json || exit 1
echo M28_DONE
