#!/usr/bin/env bash
# (M78) 一次性起 4 条被守卫的读数流（命令写进文件再跑：ssh→tmux→bash -c 三层引号会吃 $VAR）。
#   bash /tmp/m75_go.sh
# 每条流都先等**它那一臂**真的训完（step_44000.pt + 哨兵）再动手，见 m75_wait_and_go.sh。
# 卡：w384→GPU7（训完约 45.6G free）、w512→GPU2（约 49.9G free）。
# 同卡两条流：第二条 DELAY=1800 且门槛更高，等第一条把显存占上去之后再判断还塞不塞得下。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M75_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
W=/tmp/m75_wait_and_go.sh
A="0 1 2 3 4 5 6 7 8 9 10 11 12 13"
B="14 15 16 17 18 19 20 21 22 23 24 25 26 27"
tmux new-session -d -s m75r_384a "bash $W w384 7 20000 $A"
tmux new-session -d -s m75r_384b "DELAY=1800 bash $W w384 7 20000 $B"
tmux new-session -d -s m75r_512a "bash $W w512 2 20000 $A"
tmux new-session -d -s m75r_512b "DELAY=1800 bash $W w512 2 26000 $B"
tmux ls | grep '^m75r_'
echo M75_GO_DONE
