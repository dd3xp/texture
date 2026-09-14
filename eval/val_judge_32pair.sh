#!/usr/bin/env bash
# 32px 配置的同材质直接对判（验证集 V_mat，125 材质）。预注册：本脚本先提交再跑。
#
# 背景：正式测试判官 32px 判 TRD 输 B2（41%，p=0.014），唯一站得住的 32px 事实；
# 结构门已证明不能给 TRD 自家配置排序（docs/arch_progress.md 2026-09-12/14）→ 配置比较只认同材质直接对判。
# 三张图目录都已在验证集上生成过（v10，跨模态 N=100，CLIP-B/16 4 选 1），零新生成：
#   v10x100_rr4     直接 32px（= 正式测试 TRD32 的配置）
#   v10x100c_rr4    由粗到细级联（16px 结构 → 粗网格条件 → 32px）
#   v10x100r16_rr4  直接 32px + 块 Gibbs 精修（16 轮 × 15%，温度 0.5）
# 判据（写死）：
#   ① 每臂先试点（15 真题 + 5 同图空对照），可解率 ≥65% 且高于空对照才跑全量，不过门槛的臂不报胜负；
#   ② 某变体算"优于直接生成" = 全量胜率的二项 p < 0.05 且方向为胜；
#   ③ 任何据此换 32px 主配置再测测试集 = 对测试集的第二次观察，须另行预注册、只跑一次。
# 输出写 /tmp（/mnt/data 满），跑完 scp 回本机入库。凭据只从环境变量读。
P=${PY:-python}
OUT=${OUT:-/tmp/judge_32pair}
mkdir -p "$OUT"
J() { $P eval/judge_pairs.py pilot "$@" --outdir "$OUT" && $P eval/judge_pairs.py full "$@" --outdir "$OUT"; }
J --a v10x100c_rr4   --b v10x100_rr4 --set V_mat --size 32
J --a v10x100r16_rr4 --b v10x100_rr4 --set V_mat --size 32
echo VJ32_DONE
