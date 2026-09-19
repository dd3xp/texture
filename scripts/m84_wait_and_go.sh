#!/usr/bin/env bash
# (M84) 读数启动守卫 + 排队器：等训练**真的跑完**、且卡上真有显存，再去调 m84_arm.sh。
#   bash /tmp/m84_wait_and_go.sh <GPU> <need_MiB> <seed> [seed ...]
#
# ⛔ 本文件不含任何判据：判据全文写死在 scripts/m84_w384b.sh 头部；读数命令一个字都不在
#    这里，仍由 scripts/m84_arm.sh 发出。
# ⚠ (M77) 第五节的守卫：train_trd.py:633 每次记日志就覆写 last.pt，而读数走 --ckpt 默认的
#   last.pt ⇒ 若在训练途中启动，会读到**中途检查点**，而 (V1) 查 log.json 查不出来。
#   ⇒ 这里只认「只有真跑完才存在」的两样证据（(P8) 教训）：
#     ① $RUN/step_44000.pt（train_trd.py:634 只在 step in save_at 时写）
#     ② 训练日志末尾的哨兵 M84_W384B_DONE（python 正常退出后才打印）
# ⚠ ⛔ 不读日志里任何一行指标（只 grep 哨兵、只看文件存在）⇒ 携 0 比特，不破盲。
# ⚠ (M61) 教训：脚本自己声明该在哪台机器跑。
# ⚠ 挑卡只看 memory.free（(M35)）；一份 32px 读数 (M62)/(M78) 实测 17,989 MiB（d=384 口径）。
# ⚠ (M79)：本守卫失败时的行为是**继续等**，不是退出 ⇒ 最坏情况是慢（多条流串行），
#   ⛔ 不是"静默少料而判读器照常出判决"。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M84_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
GPU="$1"; NEED="$2"; shift 2
RUN=/tmp/runs/trd_w384b_09192215
SENT=M84_W384B_DONE
LOG=/tmp/m84_w384b.txt
FIRST="$1"                      # 同一张卡上多条流各写各的日志，别互相盖
exec >> "/tmp/m84_wait_gpu${GPU}_s${FIRST}.txt" 2>&1
echo "=== wait_and_go gpu=$GPU need=$NEED seeds=$* start $(date -u)"
while :; do
  if [ -f "$RUN/step_44000.pt" ] && grep -q "^${SENT}\$" "$LOG"; then
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    echo "$(date -u +%H:%M:%S) train_done=1 free=${FREE} need=${NEED}"
    if [ "$FREE" -ge "$NEED" ]; then
      echo "$(date -u +%H:%M:%S) LAUNCH m84_arm.sh $GPU $*"
      exec bash /tmp/m84_arm.sh "$GPU" "$@"
    fi
  else
    echo "$(date -u +%H:%M:%S) train_done=0"
  fi
  sleep 60
done
