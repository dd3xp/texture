#!/usr/bin/env bash
# (M78) 读数启动守卫 + 排队器：等某一臂**真的训完**、且卡上真有显存，再去调 m75_arm.sh。
#   bash /tmp/m75_wait_and_go.sh <w384|w512> <GPU> <need_MiB> <seed> [seed ...]
#
# ⛔ 本文件不含任何判据：判据全文写死在 scripts/m75_w384.sh 头部（commit 048116d）；
#    读数命令一个字都不在这里，仍由 scripts/m75_arm.sh（(M76) 冻结）发出。
# ⚠ (M77) 第五节的守卫：train_trd.py:633 每次记日志就覆写 last.pt，而读数走 --ckpt 默认的
#   last.pt ⇒ 若在训练途中启动，会读到**中途检查点**，而 (V1) 查 log.json 查不出来。
#   ⇒ 这里只认「只有真跑完才存在」的两样证据（(P8) 教训）：
#     ① $RUN/step_44000.pt（train_trd.py:634 只在 step in save_at 时写）
#     ② 训练日志末尾的哨兵 M75_W384_DONE / M75_W512_DONE（python 正常退出后才打印）
#   两样都齐 ⇒ 此刻的 last.pt 就是 44000 步那个 state（:633 与 :634 同一轮写出）。
# ⚠ ⛔ 不读日志里任何一行指标（只 grep 哨兵、只看文件存在）⇒ 携 0 比特，不破盲。
# ⚠ (M61) 教训：脚本自己声明该在哪台机器跑。
# ⚠ 挑卡只看 memory.free（(M35)）；一份 32px 读数 (M62) 实测 17.98GB（d=384 口径）。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M75_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
ARM="$1"; GPU="$2"; NEED="$3"; shift 3
case "$ARM" in
  w384) RUN=/tmp/runs/trd_w384_09191345; SENT=M75_W384_DONE ;;
  w512) RUN=/tmp/runs/trd_w512_09191345; SENT=M75_W512_DONE ;;
  *) echo "M75_BAD_ARM_$ARM"; exit 2 ;;
esac
LOG="/tmp/m75_${ARM}.txt"
FIRST="$1"                      # 同一张卡上多条流各写各的日志，别互相盖
exec >> "/tmp/m75_wait_${ARM}_gpu${GPU}_s${FIRST}.txt" 2>&1
echo "=== wait_and_go $ARM gpu=$GPU need=$NEED delay=${DELAY:-0} seeds=$* start $(date -u)"
# DELAY：同卡第二条流用。⚠ 必须睡在「训练已结束」**之后**（睡在开头等于没睡：
# 训练还要跑一小时，延时早就过完了，两条流照样在同一分钟一起起来）。
DELAYED=0
while :; do
  if [ -f "$RUN/step_44000.pt" ] && grep -q "^${SENT}\$" "$LOG"; then
    if [ "$DELAYED" = 0 ] && [ "${DELAY:-0}" != 0 ]; then
      echo "$(date -u +%H:%M:%S) train_done=1 sleeping ${DELAY}s (同卡第二条流)"
      sleep "$DELAY"; DELAYED=1
    fi
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    echo "$(date -u +%H:%M:%S) train_done=1 free=${FREE} need=${NEED}"
    if [ "$FREE" -ge "$NEED" ]; then
      echo "$(date -u +%H:%M:%S) LAUNCH m75_arm.sh $GPU $ARM $*"
      exec bash /tmp/m75_arm.sh "$GPU" "$ARM" "$@"
    fi
  else
    echo "$(date -u +%H:%M:%S) train_done=0"
  fi
  sleep 60
done
