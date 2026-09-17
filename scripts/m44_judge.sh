#!/usr/bin/env bash
# (M44) 判官阶段启动链。⛔ 凭据**不落盘**：VLM_BASE_URL / VLM_API_KEY 由
# `tmux new-session -e ...` 注入（tmux 3.2a 支持），本文件里一个字都没有。
# ⚠ 判据全文在 `eval/m44_judge_nogate.sh`（预注册，写于任何判官读数存在之前）。
cd /mnt/data/kw/RoundSquisheen/texture
bash eval/m44_judge_nogate.sh >> /tmp/m44_judge.txt 2>&1
echo "M44_JUDGE_EXIT=$?" >> /tmp/m44_judge.txt
