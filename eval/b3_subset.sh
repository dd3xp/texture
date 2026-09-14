#!/usr/bin/env bash
# B3（SD-piXL）12 材质子集对比。预注册：本脚本与 analysis/arch/b3_subset_stats.py 先提交再跑，只跑一次。
#
# 子集 = eval/sdpixl_subset.json（固定种子 2026 预先从 E_mat 抽的 12 个材质，B3 每材质 1 张、默认 1 万步）。
# 臂（全部 16px、每材质第 0 张，瓦片全部已在盘上 -> 零生成）：
#   S1  TRD16   vs B3   —— 主判据臂（= 正式测试 56f1801 的 TRD16 配置，未改）
#   S2  TRD16c  vs B3   —— 正式测试的 TRD16c（N=100 + CLIP-B/16 4 选 1）
#   S3  B2      vs B3   —— 参照：SD-piXL 是否胜过它所基于的"大模型+降采样"
#   S4  B7      vs B3   —— 参照：同数据像素扩散 UNet
# 判据（写死）：
#   ① 每臂先试点（12 真题全用 + 5 同图空对照），可解率 >=65% 且高于空对照才跑全量；不过门槛的臂不报胜负。
#   ② n=12：双侧二项 p<0.05 需要有效对里至少 10/12 同向。S1 算"TRD 胜 B3" = 全量 p<0.05 且方向为胜；
#      否则只能写"n=12 下未测到差异"，**不许写成"与 B3 持平"**（n=12 的最小可测效应约 83%）。
#   ③ CLIP-B/32 分数（与 run_eval 同口径）：12 材质配对，报均值与符号检验，描述性、不作主判据。
#   ④ 成本：B3 每张 GPU 小时取自出图日志（共享卡上测得，照实报）；TRD 的速度等独占卡另测（speed.json）。
#   ⑤ 本比较不授权任何配置变更。
# 输出写 /tmp（/mnt/data 满），跑完 scp 回本机入库。凭据只从环境变量读。
set -u
P=${PY:-python}
OUT=${OUT:-/tmp/judge_b3}
mkdir -p "$OUT"
SUB=eval/sdpixl_subset.json
J() { $P eval/judge_pairs.py pilot "$@" --subset $SUB --n_pilot 12 --outdir "$OUT" && \
      $P eval/judge_pairs.py full  "$@" --subset $SUB --outdir "$OUT"; }
J --a TRD16  --b B3 --size 16
J --a TRD16c --b B3 --size 16
J --a B2     --b B3 --size 16
J --a B7     --b B3 --size 16
$P analysis/arch/b3_subset_stats.py --judge_dir "$OUT" --out "$OUT/b3_subset_clip.json"
echo B3SUB_DONE
