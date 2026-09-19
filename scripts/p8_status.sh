#!/usr/bin/env bash
# (P8) 只读健康探针：不改任何东西，只报四条臂的状态。
# ⚠ 必须在 GPU 机上跑（(M61) 教训：本机跑会四个 pid 全 GONE、假报"实验全崩"）。
case "$(hostname)" in a100-node*) ;; *) echo "WRONG_HOST $(hostname)"; exit 4;; esac
cd /mnt/data/kw/RoundSquisheen/texture
STAMP=09190900
echo "=== now(UTC) $(date -u +%F_%H:%M) ==="
for A in C A1 A2 A3; do
  RUN=/tmp/runs/trd_p8${A}_$STAMP
  CK=no; [ -f "$RUN/last.pt" ] && CK=yes
  STEP=$(grep -c '^step ' /tmp/p8_$A.txt 2>/dev/null || true)
  LAST=$(grep -o 'step [0-9]*' /tmp/p8_$A.txt 2>/dev/null | tail -n 1 || true)
  ERR=$(grep -ci 'out of memory\|Traceback' /tmp/p8_$A.txt 2>/dev/null || true)
  GEN=$(ls /tmp/gen/p8${A}x/16 2>/dev/null | wc -l)
  GEN2=$(ls experiments/baselines/p8${A}x/16 2>/dev/null | wc -l)
  echo "[$A] last.pt=$CK  loglines=$STEP  last='$LAST'  err=$ERR  gen_tmp=$GEN gen_exp=$GEN2"
done
echo "=== procs ==="
ps -eo pid,etime,args | grep -E 'p8_|train_trd|gen_trd' | grep -v grep
echo "=== gpu free MiB ==="
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1": "$3-$2}'
echo P8_STATUS_DONE
