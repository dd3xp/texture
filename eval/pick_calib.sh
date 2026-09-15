#!/usr/bin/env bash
# 判官校准：量"分不出时是扔硬币还是照位置挑"。预注册 = 本脚本与 analysis/arch/pick_decomp.py，跑之前提交。
#
# 为什么跑：环不闭合，不可加压在 e3 判出率 51%（另三条 70–74%）上。刺激那条解释已排除（830b2d2）。
# 剩下的两个机制对胜率的修正**方向相反**，而已落盘的数据分不开它们——judge_pairs.py:78 把
# "挑了哪个位置"扔了。本次只补这一位，**不产生胜率、不登账成臂**。
#
# 这不是"开一条 32px 新臂"（准入条件第 3 条）：不新增任何方法/配置，只校准尺子本身，
# 正是 6c64796 写下的下一步"弄清同一台仪器判出率为何能差 23pp"。
#
# 判据与无自由参数的点预测（e3 74.1% / e4 64.8%，(M-a) 都是 50%）见 analysis/arch/pick_decomp.py。
# 规模：2 边 × 136 对 × 2 问 = 544 次调用，与前两轮同量级。种子固定且两边同一个 → 同一批材质，配对对比。
# 产物 /tmp/judge_picks/（盘满规矩），名字落在 sync_remote_tmp.sh 的 judge_* 里。
set -u
P=${PY:-python}
OUT=${OUT:-/tmp/judge_picks}
mkdir -p "$OUT"
N=${N:-136}
$P eval/judge_picks.py --a B2         --b B2up16   --size 32 --n "$N" --outdir "$OUT"
$P eval/judge_picks.py --a TRD32_rr4  --b TRD16cup --size 32 --n "$N" --outdir "$OUT"
$P analysis/arch/pick_decomp.py
echo PICK_CALIB_DONE
