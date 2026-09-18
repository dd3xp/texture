#!/bin/bash
# (M61) zero-GPU health check for the four running arms. Read-only.
# Guard: run locally by mistake and every probe silently reports "no data / all dead"
# with exit code 0 and a healthy M61_HEALTH_DONE sentinel. Must run on the GPU node.
case "$(hostname)" in
  a100-node*) ;;
  *) echo "M61_HEALTH_WRONG_HOST: $(hostname); use: ssh emnlp bash /tmp/m61_health.sh" >&2; exit 4 ;;
esac
echo "=== pids ==="
for p in 3240851 3240857 3240863 3240869; do
  if [ -d /proc/$p ]; then echo "pid $p ALIVE"; else echo "pid $p GONE"; fi
done
echo "=== per-arm counts ==="
echo -n "ctrl: "; ls /tmp/m61_ctrl_s*.json 2>/dev/null | wc -l
echo -n "more: "; ls /tmp/m61_more_s*.json 2>/dev/null | wc -l
echo "=== which seeds ==="
ls /tmp/m61_ctrl_s*.json /tmp/m61_more_s*.json 2>/dev/null | tr '\n' ' '
echo
echo "=== progress (blind: only '=== seed N start/done ===' lines, never metric rows) ==="
for f in /tmp/m61_ctrlA.txt /tmp/m61_ctrlB.txt /tmp/m61_moreA.txt /tmp/m61_moreB.txt; do
  echo "== $f"; grep -h '^=== ' "$f" 2>/dev/null | tail -n 2
done
echo "=== gpu 2 / 7 ==="
nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader | sed -n '3p;8p'
echo "=== disk /tmp ==="
df -h /tmp | tail -1
echo "=== sizes ==="
ls -l /tmp/m61_ctrl_s*.json /tmp/m61_more_s*.json 2>/dev/null | awk '{print $5}' | sort -u | tr '\n' ' '
echo
echo "M61_HEALTH_DONE"
