#!/bin/bash
# (M41) 第二阶段直跑（2026-09-17 10:1x UTC+8）。两臂训练均已跑满 6000 步、last.pt 齐，
# 所以**不需要看门狗**；v2 看门狗因 tmux 前缀匹配（has-session -t arch_m41 命中自身所在的
# arch_m41wd2）死等了 36 分钟，已 kill。本文件 = v2 看门狗去掉那个 while 循环，其余逐字相同：
# 同样的 TAG/OUT/JOUT/MET/CUDA/TRITON、同样的 gen->rc->judge 单向接法。
# ⛔ 没有改 eval/m41_gen_judge.sh 一个字（md5 会在启动前复核）。
cd /mnt/data/kw/RoundSquisheen/texture
export PY=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
export TAG=09162316 OUT=/tmp/m41ab JOUT=/tmp/judge_m41ab MET=/tmp/m41_meter.json
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/tmp/triton_m41
export VLM_BASE_URL=http://113.45.39.247:3001
export VLM_API_KEY=__KEY__
STAGE=gen bash eval/m41_gen_judge.sh >> /tmp/m41_gen.txt 2>&1
G=$?
echo "GEN rc=$G" >> /tmp/m41_gen.txt
if [ "$G" -ne 0 ]; then
  echo "[run2] GEN rc=$G 非 0 -> 按预注册 §4 不进判官阶段，零 API" >> /tmp/m41_judge.txt
  exit 0
fi
STAGE=judge bash eval/m41_gen_judge.sh >> /tmp/m41_judge.txt 2>&1
echo "JUDGE rc=$?" >> /tmp/m41_judge.txt
