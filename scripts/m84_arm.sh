#!/usr/bin/env bash
# (M84) 读数臂：32px 逐图 CLIP。用法
#   bash /tmp/m84_arm.sh <GPU> <seed> [seed ...]
# 读数命令与 (M58)/(M59)/(M60)/(M61)/(M62)/(M75) **逐字同一条**，只换 --run / --seed / --out。
# ⛔ 判据一个字不在本文件里定：全文写死在 scripts/m84_w384b.sh 头部（先于任何数据），
#    判读器 analysis/arch/m84_read_retrain.py（盲写、另行提交）。
# ⚠ 本臂只跑**处理臂**（seed 1 那个 run）；控制臂 28 份直接复用 (M75) 的 m75_w384_s*.json。
# ⚠ 卡号由第一个参数带进来后 export 在本文件里（tmux 不继承 ssh 环境）；python 走绝对路径。
# ⚠ (M61) 教训：脚本自己声明该在哪台机器跑。
# ⚠ 输出命名 m84_* 已加进 scripts/sync_remote_tmp.sh 白名单（/tmp 开机即清空）。
set -e
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M84_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
GPU="$1"; shift
RUN=/tmp/runs/trd_w384b_09192215     # d=384/heads=6、从零单阶段 44000 步、--seed 1
# 日志必须落文件：只打到 tmux 面板的话，脚本一退出会话就没了，哨兵跟着蒸发（(M60) 踩过）。
exec >> "/tmp/m84_w384b_gen${GPU}.txt" 2>&1
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton
PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture

for S in "$@"; do
  # 幂等跳过守卫：只认「真跑完才存在」的证据（(P8) 教训——别认会被中途覆写的文件）
  if [ -s /tmp/m84_w384b_s"$S".json ]; then
    echo "=== w384b seed $S SKIP (already exists) ==="
    continue
  fi
  echo "=== w384b seed $S start $(date -u +%H:%M:%S) gpu=$GPU ==="
  $PY -u eval/diag_decompose.py --run "$RUN" --size 32 --xmodal --reps 2 --bs 8 \
      --per_image \
      --seed "$S" --out /tmp/m84_w384b_s"$S".json
  echo "=== w384b seed $S done $(date -u +%H:%M:%S) ==="
done
echo "M84_W384B_GPU${GPU}_DONE"
