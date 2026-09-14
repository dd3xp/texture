#!/usr/bin/env bash
# 把服务器 /tmp 里本项目的产物增量拉回本机 remote_tmp/（不进 git）。
# 为什么：/mnt/data 满了，训练检查点、SD-piXL 工作目录、判官结果都改写到服务器 /tmp；
# 而服务器 tmpfiles 规则是 `D /tmp`——**开机即清空**（共享账号的其他人也可能删）。
# 由 scripts/cron.ps1 每 30 分钟调用一次（在"主会话活着就让路"的判断之前，所以总会跑）。
# 增量：只打包上次同步时刻之后改过的文件（GNU tar --newer-mtime），检查点每千步重写一次就会重新拉。
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="$ROOT/remote_tmp"
STAMP="$DST/.last_sync"
mkdir -p "$DST"
SINCE=$(( $(cat "$STAMP" 2>/dev/null || echo 120) - 120 ))   # 留 2 分钟余量防两台机器时钟偏差
NOW=$(date +%s)
# 本项目在服务器 /tmp 下的东西（显式列出，别碰共享账号里别人的文件）
PATHS='runs sdpixl_runs* judge_* gen32* abl* speed*.json trd_*.txt b3*.txt vj*.txt nod32*.txt tanchor*.txt b2canvas*.txt'
ssh -o ConnectTimeout=30 -o ServerAliveInterval=30 emnlp \
  "cd /tmp && ls -d $PATHS 2>/dev/null | xargs -r tar cf - --newer-mtime=@$SINCE --exclude='*.lock' --exclude='.trainlock' 2>/dev/null" \
  | tar xf - -C "$DST" 2>/dev/null
st=("${PIPESTATUS[@]}")
if [ "${st[0]}" = 0 ]; then
  echo "$NOW" > "$STAMP"
  echo "[$(date '+%m-%d %H:%M')] sync ok since @$SINCE -> $(du -sh "$DST" | cut -f1)"
else
  echo "[$(date '+%m-%d %H:%M')] sync FAILED (ssh=${st[0]} tar=${st[1]}), stamp unchanged"
fi
