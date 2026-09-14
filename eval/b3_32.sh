#!/usr/bin/env bash
# B3（SD-piXL）32px：同一 12 材质子集出图 → 与 TRD32 同材质对判。预注册：本脚本先提交再跑，判官只跑一次。
#
# 为什么要跑：16px 上 SD-piXL 从第 0 步起就是平涂（SDXL 参考图的细格被 16x16 双线性初始化平均掉，
# 1 万步 SDS 也没长出结构）→ 所有基线都大胜它（experiments/judge_b3/）。32px 更接近它的工作区间，
# 而 32px 恰是 TRD 输 B2 的那一档 → 必须知道 SD-piXL 在这一档是否强于 TRD。
# 出图：与 16px 完全同一套设置（configs/texture_text.yaml、默认 1 万步、调色板取 B1/32 第 0 张实际用色）。
# 臂（32px、每材质第 0 张）：
#   T1  TRD32_rr4 vs B3  —— 主判据臂（= 正式测试 56f1801 的 TRD32 配置，未改）
#   T2  B2        vs B3  —— 参照
#   T3  B7        vs B3  —— 参照
# 判据（写死，同 eval/b3_subset.sh）：
#   ① 每臂先试点（12 真题 + 5 同图空对照），可解率 >=65% 且高于空对照才跑全量；不过门槛不报胜负。
#   ② T1 "TRD 胜 B3" = 全量双侧二项 p<0.05 且方向为胜；"TRD 输 B3" = p<0.05 且方向为输；
#      其余一律写"n=12 下未测到差异"，不许写成持平。
#   ③ CLIP-B/32 配对分数描述性、不作主判据；④ 本比较不授权任何配置变更。
# 出图约 5 GPU 小时/张（共享卡），单卡 12 张 ≈ 2.5 天；GPU 由 GPU=... 指定，第二个进程可用 REV=1 从另一头跑。
set -u
P=${PY:-python}
OUT=${OUT:-/tmp/judge_b3_32}
mkdir -p "$OUT"
SUB=eval/sdpixl_subset.json
${SDPIXL_RUNNER_PY:-$P} -u baselines/sdpixl/run_subset.py --size 32 --gpus ${GPU:-7} --work /tmp/sdpixl_runs32 \
    ${REV:+--reverse}
[ -z "${REV:-}" ] || exit 0          # 只让正序那个进程去判，避免两个进程各判一遍
cnt() { ls experiments/baselines/B3/32/*_0.png 2>/dev/null | wc -l; }
# 倒序进程（tmux arch_b3_32rev）若还在做最后几张，等它
while [ "$(cnt)" -lt 12 ] && tmux ls 2>/dev/null | grep -q arch_b3_32rev; do sleep 300; done
n=$(cnt)
[ "$n" -ge 12 ] || { echo "B3/32 只有 $n/12，不判"; exit 0; }
J() { $P eval/judge_pairs.py pilot "$@" --subset $SUB --n_pilot 12 --outdir "$OUT" && \
      $P eval/judge_pairs.py full  "$@" --subset $SUB --outdir "$OUT"; }
J --a TRD32_rr4 --b B3 --size 32
J --a B2        --b B3 --size 32
J --a B7        --b B3 --size 32
$P analysis/arch/b3_subset_stats.py --size 32 --judge_dir "$OUT" --out "$OUT/b3_32_clip.json"
echo B3_32_DONE
