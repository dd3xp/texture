#!/usr/bin/env bash
# 把本地代码同步到 emnlp (8x A100 80GB)。
# 只推代码，不推 data/ 和 experiments/ —— 那些留在远端。
#
# 优先用 rsync；本地 Git Bash 通常没有 rsync，此时回退到 tar over ssh。
# 回退路径不做删除同步（远端多出来的文件不会被清理），这是有意的保守选择。
#
# 用法: bash scripts/sync_to_emnlp.sh [--dry-run]

set -euo pipefail

REMOTE_HOST="emnlp"
REMOTE_DIR="/mnt/data/kw/RoundSquisheen/texture"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

EXCLUDES=(
    '.git' '__pycache__' '.venv' 'data' 'experiments'
    '*.ckpt' '*.pt' '*.pth' '*.safetensors' '.ipynb_checkpoints'
)

cd "$LOCAL_DIR"

if command -v rsync >/dev/null 2>&1; then
    args=(-avz --delete)
    [[ $DRY_RUN -eq 1 ]] && args+=(--dry-run)
    for e in "${EXCLUDES[@]}"; do args+=(--exclude "$e"); done
    rsync "${args[@]}" "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"
else
    echo "[info] 未找到 rsync，回退到 tar over ssh（不做删除同步）"
    tar_args=(czf -)
    for e in "${EXCLUDES[@]}"; do tar_args+=(--exclude="$e"); done

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] 会传输以下文件："
        tar "${tar_args[@]}" . | tar tzf - | grep -v '/$'
    else
        ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_DIR'"
        tar "${tar_args[@]}" . | ssh "$REMOTE_HOST" "tar xzf - -C '$REMOTE_DIR'"
    fi
fi

echo
echo "已同步到 $REMOTE_HOST:$REMOTE_DIR"
