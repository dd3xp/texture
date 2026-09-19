#!/usr/bin/env bash
# (M84) 只读健康探针：训练臂还活着没有、跑到第几步；三条读数守卫还在不在、出了几份料。
#   bash /tmp/m84_health.sh
# ⚠ (M61) 教训一：探针必须自己声明该在哪台机器跑（本机敲一遍会造出"全崩了"的假象：
#   pid 全 GONE、日志 No such file、退出码仍 0 且照打哨兵 = "成功的打印≠数据到手"）。
# ⚠ (M61) 教训二：⛔ 不许 tail 原始日志。train_trd.py:629 每条进度行是 JSON、
#   step 与 loss/val **混在同一行** ⇒ 这里只数**行数**、⛔ 一个指标都不打印。
#   进度 ≈ (记录数 − 1) × eval_every(1000) 步，够判断"卡没卡住"，且携 0 比特读数。
# ⚠ (M57)：看的必须是「脚本本体」的 pid，不是里面那个 python 的。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M84_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
echo "host=$(hostname) now_utc=$(date -u +%H:%M:%S)"

# ---- 阶段一：训练（脚本本体 pid 见 docs/arch_progress.md (M84)）----
if [ -d /proc/3905171 ]; then echo "train w384b pid 3905171 ALIVE"; else echo "train w384b pid 3905171 GONE"; fi
f=/tmp/m84_w384b.txt
if [ -s "$f" ]; then
  echo "train w384b: records=$(grep -c '^{' "$f") sentinel=$(grep -c '^M84_W384B_DONE$' "$f") oom=$(grep -ci 'out of memory\|Traceback' "$f")"
else
  echo "train w384b: NO_LOG"
fi
echo "ckpt44000=$([ -f /tmp/runs/trd_w384b_09192215/step_44000.pt ] && echo 1 || echo 0)"

# ---- 阶段二：读数守卫（失败时的行为是继续等，不是退出 ⇒ GONE 才是异常）----
for pa in "3905477:gpu2" "3905482:gpu6" "3905488:gpu7"; do
  p="${pa%%:*}"; nm="${pa##*:}"
  if [ -d "/proc/$p" ]; then echo "guard $nm pid $p ALIVE"; else echo "guard $nm pid $p GONE"; fi
done

# ---- 阶段三：料（料齐才判；份数就是唯一要看的东西）----
echo "read w384b: json=$(ls /tmp/m84_w384b_s*.json 2>/dev/null | wc -l)/28"
echo M84_HEALTH_DONE
