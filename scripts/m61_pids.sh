#!/bin/bash
# (M61) 挂起复核：列出四条流「脚本本体」的 pid 与逐字 cmdline，并核 GPU 占用。
# ⚠ 等的必须是 m61_arm.sh 的 pid，不是里面那个 python（(M57) 踩过）。
for p in $(pgrep -f 'bash /tmp/m61_arm.sh'); do
  echo -n "pid=$p cmd="
  tr '\0' ' ' < /proc/$p/cmdline
  echo
done
echo "--- gpu ---"
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
echo "--- free ---"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
