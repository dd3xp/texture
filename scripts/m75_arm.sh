#!/usr/bin/env bash
# (M75) 读数臂：32px 逐图 CLIP。用法
#   bash /tmp/m75_arm.sh <GPU> <w384|w512> <seed> [seed ...]
# 读数命令与 (M58)/(M59)/(M60)/(M61)/(M62) **逐字同一条**，只换 --run / --seed / --out。
# 判据一个字不在本文件里定：全文写死在 scripts/m75_w384.sh 头部（commit 048116d，
# 先于任何数据），判读器 analysis/arch/m75_read_width.py（盲写、--selftest 41/41）。
#
# ⚠ 卡号由第一个参数带进来后 export 在本文件里（tmux 不继承 ssh 环境）；python 走绝对路径。
# ⚠ (M61) 教训：脚本自己声明该在哪台机器跑。
# ⚠ 输出命名 m75_* 已加进 scripts/sync_remote_tmp.sh 白名单（/tmp 开机即清空）。
# ⚠ d=512 的读数显存高于 (M62) 实测的 17.98GB ⇒ 挑卡前先量一份（预注册第六节）。
set -e
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M75_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
GPU="$1"; ARM="$2"; shift 2
case "$ARM" in
  w384) RUN=/tmp/runs/trd_w384_09191345 ;;   # 控制臂 d=384/heads=6，从零单阶段 44000 步
  w512) RUN=/tmp/runs/trd_w512_09191345 ;;   # 处理臂 d=512/heads=8，同上
  *) echo "M75_BAD_ARM_$ARM"; exit 2 ;;
esac
# 日志必须落文件：只打到 tmux 面板的话，脚本一退出会话就没了，哨兵跟着蒸发（(M60) 踩过）。
# ⚑ 在脚本里 exec 重定向 ⇒ 起法不用带 '>>'，省掉 ssh→tmux→bash -c 那层引号（$VAR 会被吃掉）。
exec >> "/tmp/m75_${ARM}_gen${GPU}.txt" 2>&1
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

for S in "$@"; do
  # 幂等跳过守卫：只认「真跑完才存在」的证据（(P8) 教训——别认会被中途覆写的文件）
  if [ -s /tmp/m75_"$ARM"_s"$S".json ]; then
    echo "=== $ARM seed $S SKIP (already exists) ==="
    continue
  fi
  echo "=== $ARM seed $S start $(date -u +%H:%M:%S) gpu=$GPU ==="
  $PY -u eval/diag_decompose.py --run "$RUN" --size 32 --xmodal --reps 2 --bs 8 \
      --per_image \
      --seed "$S" --out /tmp/m75_"$ARM"_s"$S".json
  echo "=== $ARM seed $S done $(date -u +%H:%M:%S) ==="
done
echo "M75_${ARM}_GPU${GPU}_DONE"
