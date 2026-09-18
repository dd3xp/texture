#!/usr/bin/env bash
# (M62) 只读健康探针：处理臂三张卡还活着没有、各跑到第几份。
#   bash /tmp/m62_health.sh
# ⚠ (M61) 教训一：探针必须自己声明该在哪台机器跑（本机敲一遍会造出"全崩了"的假象）。
# ⚠ (M61) 教训二：⛔ 不许 tail 原始日志（指标与进度混排，会把读数端上来）——
#   这里只数结构化进度行 '^=== ' 与哨兵，**从不打印任何指标**。
# ⚠ (M57)：等的必须是「脚本本体」的 pid，不是 python 的。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M62_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
echo "host=$(hostname) now_utc=$(date -u +%H:%M:%S)"
alive=0
for p in 3345427 3345431 3345438; do
  if [ -d "/proc/$p" ]; then
    echo "pid $p ALIVE"
    alive=$((alive + 1))
  else
    echo "pid $p GONE"
  fi
done
for g in 2 6 7; do
  f="/tmp/m62_dbl_gen$g.txt"
  if [ -s "$f" ]; then
    echo "gpu$g: started=$(grep -c '^=== dbl seed .* start' "$f") done=$(grep -c '^=== dbl seed .* done' "$f") sentinel=$(grep -c '^M62_dbl_GPU._DONE$' "$f")"
  else
    echo "gpu$g: NO_LOG"
  fi
done
echo "alive=$alive/3"
echo M62_HEALTH_DONE
