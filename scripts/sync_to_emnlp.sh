#!/usr/bin/env bash
# 把本地代码同步到 emnlp (8x A100 80GB)。
# 只推代码，不推 data/ 和 experiments/ —— 那些留在远端。
# 用法: bash scripts/sync_to_emnlp.sh [--dry-run]

set -euo pipefail

REMOTE_HOST="emnlp"
REMOTE_DIR="/mnt/data/kw/RoundSquisheen/texture"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXTRA_ARGS=()
if [[ "${1:-}" == "--dry-run" ]]; then
    EXTRA_ARGS+=(--dry-run)
    echo "[dry-run] 只显示会同步什么，不实际传输"
fi

rsync -avz --delete "${EXTRA_ARGS[@]}" \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '.venv/' \
    --exclude 'data/' \
    --exclude 'experiments/' \
    --exclude '*.ckpt' --exclude '*.pt' --exclude '*.pth' --exclude '*.safetensors' \
    --exclude '.ipynb_checkpoints/' \
    "$LOCAL_DIR/" "$REMOTE_HOST:$REMOTE_DIR/"

echo
echo "已同步到 $REMOTE_HOST:$REMOTE_DIR"
echo "  ssh $REMOTE_HOST 'cd $REMOTE_DIR && ls'"
