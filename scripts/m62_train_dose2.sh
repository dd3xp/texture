#!/usr/bin/env bash
# (M62) 训练预算的第二次翻倍：runs/trd_v10 的续训周期从 12000 拉到 24000。
# 预注册见 docs/arch_progress.md 2026-09-18 23:30 那节第四节。
# ⚑ 本文件 = scripts/m60_train_more.sh 逐字复制，只改三处：--steps / --out / --save_at。
#   v10 的 13 个非默认键原样重传；v10 之后新增的参数一个都不传（留默认 = 旧行为）。
#
# ⚠ 卡号写在本文件里（tmux 不继承 ssh 环境，CUDA_VISIBLE_DEVICES 传不进去）。
# ⚠ PATH 也不继承 → python 必须写绝对路径。
# ⚠ 产物全写 /tmp（/mnt/data 长期贴满）；TRITON_CACHE_DIR 同理，否则 atexit 里才崩。
# ⚠ (M61) 教训：脚本自己声明该在哪台机器跑，别让"在本机敲了一遍"造出假象。
set -e
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M62_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
# 日志必须落文件（探针 grep 的就是这个文件的哨兵）；在脚本里 exec 重定向 ⇒ 起法不用带 '>>'。
exec >> /tmp/trd_v10dbl.txt 2>&1
export CUDA_VISIBLE_DEVICES=2
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

OUT=/tmp/runs/trd_v10dbl_09181524
mkdir -p "$OUT" /tmp/triton

$PY -u model/train_trd.py \
  --out "$OUT" \
  --init_from runs/trd_v10/last.pt \
  --steps 24000 \
  --batch 256 \
  --lr 1.5e-4 \
  --warmup 500 \
  --sizes 16 32 \
  --p32 0.5 \
  --batch32 64 \
  --bias_freqs 8 \
  --bias_hidden 128 \
  --level_emb \
  --pal_aug 0.3 \
  --pal_smooth 0.1 \
  --extra \
  --n_ex 4 \
  --extra_file train_extra_packs_only.json \
  --coarse \
  --p_coarse 0.5 \
  --save_at 6000 12000 18000 24000

echo M62_TRAIN_DONE_GPU2
