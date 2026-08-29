#!/usr/bin/env bash
# 自动推进训练：把长训练切成 8 分钟的小段接力，直到跑满目标轮数。
#
# 为什么要这样：这些共享机器上单次训练超过 10–13 分钟会被静默杀掉
# （无报错、日志干净截断，而 2 分钟的跑能正常收尾）。
# 所以不追求"一次跑完"，改成**每段自己保存、下一段从 last.pt 接上**。
# 脚本本身在 tmux 里跑，不依赖任何 ssh 会话存活——
# cron 在负责人离线时不触发，脚本放在机器上才可靠。
#
# 用法（在 kw 上）：
#   tmux new-session -d -s auto 'bash scripts/auto_train.sh <tag> <目标轮数> [额外参数...]'
# 进度看 experiments/model/<tag>/history.json 和 /tmp/auto_<tag>.log

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TAG="${1:?需要 tag}"
TARGET="${2:?需要目标轮数}"
shift 2
PY=/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python
LOG=/tmp/auto_${TAG}.log
OUT=experiments/model/${TAG}

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$LOG"; }

say "=== 开始：tag=$TAG 目标 $TARGET 轮，参数: $* ==="

for seg in $(seq 1 60); do          # 最多 60 段，够跑到任何合理的轮数
    done_ep=0
    if [ -f "$OUT/history.json" ]; then
        done_ep=$($PY -c "
import json,sys
try:
    h=json.load(open('$OUT/history.json'))['history']
    print(h[-1]['epoch'] if h else 0)
except Exception: print(0)
" 2>/dev/null || echo 0)
    fi

    if [ "$done_ep" -ge "$TARGET" ]; then
        say "已达目标（$done_ep >= $TARGET 轮），结束"
        break
    fi

    RESUME=""
    [ -f "$OUT/last.pt" ] && RESUME="--resume"
    say "第 $seg 段：从第 $((done_ep+1)) 轮起 $RESUME"

    $PY -u model/train.py --tag "$TAG" --epochs "$TARGET" --max-minutes 8 \
        $RESUME "$@" >> "$LOG" 2>&1

    new_ep=$($PY -c "
import json
try:
    h=json.load(open('$OUT/history.json'))['history']
    print(h[-1]['epoch'] if h else 0)
except Exception: print(0)
" 2>/dev/null || echo 0)

    # 没有推进说明卡住了，别无限空转
    if [ "$new_ep" -le "$done_ep" ]; then
        say "本段没有推进（$done_ep → $new_ep），停止以免空转"
        break
    fi
    sleep 5
done

BEST=$($PY -c "
import json
h=json.load(open('$OUT/history.json'))
print(f\"最佳验证 {h['best_val']:.4f} @ep{h['best_epoch']}\")
" 2>/dev/null || echo '读取失败')
say "=== 结束：$BEST ==="
