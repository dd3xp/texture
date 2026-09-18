#!/usr/bin/env bash
# (M60) 读数阶段的取料器：把 34 份 JSON 集齐到一个**全新**目录、逐份 md5 与远程对拍，再跑冻结的判读器。
#
# 为什么要有它：上一轮已经拆掉"在服务器上跑判读器＝静默 VOID_CTRL_NOT_REPRODUCED"的陷阱
# （远程没有 experiments/m59_*.json、也没有判读器本体）。剩下的最后一个**静默**失败模式是
#   「scp 少传一份 / 传了半截 ⇒ 判读器读到旧拷贝或短文件，却打印出一个看着正常的判决」。
# 本脚本让那种情况**退出码非零**，而不是打印一个可信的假判决。
#
# ⛔ 判据一个字不改：最后一条命令与 docs/arch_progress.md (M60) 预注册第五节逐字相同
#   （--k 17 / --ctrl_tag m60_ctrl / --trt_tag m60_more / --m59dir experiments）。
# ⛔ 本脚本**只在本机跑**（远程缺 m59 产物与判读器本体，见 (M60) 那一节）。
#
# 用法：bash scripts/m60_fetch.sh [目标目录]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-$ROOT/remote_tmp/m60_read}"
K=17

# 目标目录必须是全新的：故意不加 -p，已存在就失败 —— 杜绝混进上一轮 m60_precheck/ 里的旧拷贝。
mkdir "$DST"

NAMES=()
for s in $(seq 5 $((K - 1))); do NAMES+=("m60_ctrl_s$s.json"); done
for s in $(seq 0 $((K - 1))); do NAMES+=("m60_more_s$s.json"); done
REMOTE=""
for n in "${NAMES[@]}"; do REMOTE="$REMOTE /tmp/$n"; done

# 1) 远程料必须齐才拉（⚠ 用退出码判断，不拿 stdout 比字符串 —— (M44) 那条老坑）
if ! ssh emnlp "for f in $REMOTE; do test -s \"\$f\" || exit 1; done"; then
  echo "[m60_fetch] 远程 29 份还没齐（或某份是空文件）—— 不拉、不判。"; rmdir "$DST"; exit 2
fi

# 2) 拉那 29 份；控制臂 s0..s4 按预注册直接拷 (M59) 产物
scp -q emnlp:"$REMOTE" "$DST/"
for s in 0 1 2 3 4; do
  cp "$ROOT/experiments/m59_pairedclip_s$s.json" "$DST/m60_ctrl_s$s.json"
done

# 3) 逐份 md5 对拍（只对拉回来的 29 份；m59 那 5 份是本机已入库的原件）
# ⚠ 实测坑：Windows git-bash 的 md5sum 走二进制模式，打的是 `hash *name`，而 Linux 端是 `hash  name`
#   ⇒ 不归一化的话哈希全对也会 diff 失败（＝第 3 道闸误杀真数据）。两边都只取 hash+basename。
norm() { awk '{h=$1; n=$2; sub(/^\*/, "", n); print h, n}' | sort; }
ssh emnlp "cd /tmp && md5sum ${NAMES[*]}" | norm > "$DST/.remote.md5"
(cd "$DST" && md5sum "${NAMES[@]}") | norm > "$DST/.local.md5"
if ! diff -q "$DST/.remote.md5" "$DST/.local.md5" > /dev/null; then
  echo "[m60_fetch] md5 不一致（传输不完整）：" ; diff "$DST/.remote.md5" "$DST/.local.md5" || true
  exit 3
fi

# 4) 目录里必须正好 34 份（多一份少一份都停手）
n=$(ls "$DST"/m60_ctrl_s*.json "$DST"/m60_more_s*.json | wc -l)
if [ "$n" -ne $((2 * K)) ]; then echo "[m60_fetch] 目录里有 $n 份，要 $((2 * K)) 份"; exit 4; fi
echo "[m60_fetch] 34 份齐、29 份 md5 与远程逐字一致 -> $DST"

# 5) 冻结的判读器（⛔ K、加权、判据、判读器一个字不许改）
python "$ROOT/analysis/arch/m60_read_scale.py" --dir "$DST" \
  --ctrl_tag m60_ctrl --trt_tag m60_more --k $K \
  --m59dir "$ROOT/experiments" --out "$ROOT/experiments/m60_scale.json"
