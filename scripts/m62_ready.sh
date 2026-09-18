#!/usr/bin/env bash
# (M62) 只读探针：料齐没齐。用法
#   bash /tmp/m62_ready.sh train   # 训练跑完没有
#   bash /tmp/m62_ready.sh ctrl    # 控制臂新增的 11 份（seed 17..27）齐没齐
#   bash /tmp/m62_ready.sh dbl     # 处理臂 28 份（seed 0..27）齐没齐
#
# 判"齐没齐"**只用退出码**（(M44)：wc -l 在"没有"时照样打印 0，拿 stdout 比字符串会误判）：
#   0 = 齐（份数够 + 哨兵在）  1 = 份数不够  2 = 哨兵缺  3 = 用法错  4 = 跑错机器
# ⚠ (M61) 教训一：探针必须自己声明该在哪台机器跑，否则在本机敲一遍会造出"全崩了"的假象。
# ⚠ (M61) 教训二：给盲判臂做健康检查，**探针不许 tail 原始日志** —— 指标与进度混排，
#   tail 会把读数端上来。这里只 grep 结构化进度行 '^=== ' 与哨兵，**从不打印指标**。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M62_WRONG_HOST_$(hostname)"; exit 4 ;;
esac

STAGE="${1:-}"
case "$STAGE" in
  train)
    grep -q '^M62_TRAIN_DONE_GPU2$' /tmp/trd_v10dbl.txt 2>/dev/null || exit 2
    exit 0
    ;;
  ctrl|dbl)
    if [ "$STAGE" = ctrl ]; then FIRST=17; LAST=27; WANT=11; else FIRST=0; LAST=27; WANT=28; fi
    n=0
    for S in $(seq "$FIRST" "$LAST"); do
      [ -s "/tmp/m62_${STAGE}_s$S.json" ] && n=$((n+1))
    done
    # 只打印结构化进度（份数 + '=== done' 行数），⛔ 从不打印任何指标
    echo "${STAGE}_json=$n/$WANT done_lines=$(grep -ch "^=== $STAGE seed .* done" /tmp/m62_${STAGE}_gen*.txt 2>/dev/null | paste -sd+ | bc 2>/dev/null || echo 0)"
    [ "$n" -eq "$WANT" ] || exit 1
    logs=$(ls /tmp/m62_${STAGE}_gen*.txt 2>/dev/null | wc -l)
    sent=$(grep -l "^M62_${STAGE}_GPU._DONE$" /tmp/m62_${STAGE}_gen*.txt 2>/dev/null | wc -l)
    [ "$logs" -ge 1 ] && [ "$sent" -eq "$logs" ] || exit 2
    exit 0
    ;;
  *)
    echo "usage: m62_ready.sh train|ctrl|dbl"
    exit 3
    ;;
esac
