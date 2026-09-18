#!/bin/bash
# (M61) 判据臂启动器：两臂各 K=28 份（种子 0..27），拆成 4 条流。
#   GPU2: ctrl 0..13 / ctrl 14..27      GPU7: more 0..13 / more 14..27
# 每份 ~9.4 分钟 / 19.6GB ⇒ 每卡 2 条并发 ≈ 39GB，两张卡都够。
# ⚠ 日志必须落文件（tmux 面板一退就没，末行哨兵才复核得了）。
set -e
A="0 1 2 3 4 5 6 7 8 9 10 11 12 13"
B="14 15 16 17 18 19 20 21 22 23 24 25 26 27"
tmux new-session -d -s m61cA "bash /tmp/m61_arm.sh 2 ctrl $A >> /tmp/m61_ctrlA.txt 2>&1"
tmux new-session -d -s m61cB "bash /tmp/m61_arm.sh 2 ctrl $B >> /tmp/m61_ctrlB.txt 2>&1"
tmux new-session -d -s m61mA "bash /tmp/m61_arm.sh 7 more $A >> /tmp/m61_moreA.txt 2>&1"
tmux new-session -d -s m61mB "bash /tmp/m61_arm.sh 7 more $B >> /tmp/m61_moreB.txt 2>&1"
echo M61_LAUNCHED
