#!/usr/bin/env bash
# (M75) 只读健康探针：两条训练臂还活着没有、各跑到第几份/第几步。
#   bash /tmp/m75_health.sh
# ⚠ (M61) 教训一：探针必须自己声明该在哪台机器跑（本机敲一遍会造出"全崩了"的假象：
#   pid 全 GONE、日志 No such file、退出码仍 0 且照打哨兵 = "成功的打印≠数据到手"）。
# ⚠ (M61) 教训二：⛔ 不许 tail 原始日志。train_trd.py:629 每条进度行是 JSON、
#   step 与 loss/val **混在同一行** ⇒ 这里只数**行数**、⛔ 一个指标都不打印。
#   进度 ≈ (记录数 − 1) × eval_every(1000) 步，够判断"卡没卡住"，且携 0 比特读数。
# ⚠ (M57)：看的必须是「脚本本体」的 pid，不是 python 的。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M75_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
echo "host=$(hostname) now_utc=$(date -u +%H:%M:%S)"

# ---- 阶段一：训练（脚本本体 pid 见 docs/arch_progress.md (M75) 第四节）----
alive=0
for pa in "3715077:w384" "3715073:w512"; do
  p="${pa%%:*}"; nm="${pa##*:}"
  if [ -d "/proc/$p" ]; then echo "train $nm pid $p ALIVE"; alive=$((alive + 1))
  else echo "train $nm pid $p GONE"; fi
done
for nm in w384 w512; do
  f="/tmp/m75_$nm.txt"
  if [ -s "$f" ]; then
    echo "train $nm: records=$(grep -c '^{' "$f") sentinel=$(grep -c "^M75_${nm^^}_DONE\$" "$f") oom=$(grep -ci 'out of memory\|Traceback' "$f")"
  else
    echo "train $nm: NO_LOG"
  fi
done
echo "train_alive=$alive/2"

# ---- 阶段二：读数（料齐才判；份数就是唯一要看的东西）----
for nm in w384 w512; do
  echo "read $nm: json=$(ls /tmp/m75_${nm}_s*.json 2>/dev/null | wc -l)/28"
done
echo M75_HEALTH_DONE
