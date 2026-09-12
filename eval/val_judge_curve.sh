#!/usr/bin/env bash
# 验证集上沿"CLIP <-> FD 汇率曲线"移动时，判官怎么走。
#
# 为什么跑这个（docs/arch_progress.md 2026-09-12 01:40 UTC 那节的下一步 1）：
# 噪声下限那轮量出来，项目里已知的两个杠杆（调色板典型度、4 选 1 重排）落在**同一条**
# CLIP-FD 交换曲线上：CLIP 每涨约 1.1，FD 涨约 16-22，换在哪一步都一样。
# 分布指标已经被这条曲线绑死，**判官是唯一还没被绑住的尺子**——正式测试里 TRD16
# （CLIP 34.6 / FD 82.3）判官胜 B7 63%，而 B7 的 FD 好得多。所以问：曲线上哪个点判官最喜欢？
#
# 四个操作点（全部对 B7val_c1.5，V_mat 125 材质，16px；括号里是噪声下限那轮量的 m=1 口径数字）：
#   nf8_name       名字检索、单张      CLIP 33.51  FD  88.8
#   nf8_name_rr2   名字检索 + 2 选 1   CLIP 34.15  FD  92.8
#   nf8_name_rr4   名字检索 + 4 选 1   CLIP 34.66  FD 104.7
#   nf8_xpal       跨模态硬取、单张    CLIP 34.59  FD 110.6   <- 与正式测试的 TRD16 同配置
# 前三个是同一个杠杆的剂量梯度，第四个与第三个 CLIP 相同而杠杆不同 -> 顺便检验
# "两个杠杆是同一笔买卖"这句话在判官这把尺子上是否也成立。
#
# **预注册判据（跑之前写死，事后不许改）**：
#  (1) 每个臂都先试点（15 真题 + 5 两边同图的空对照，只量两序一致率），
#      可解率 >= 65% 且高于空对照才跑 full。不过门槛的臂**不报胜负**。
#  (2) 两个臂算"不同"的唯一判据：Jeffreys 95% 区间**不重叠**。四个臂区间全重叠
#      = "判官分不出曲线上的位置" -> 这条曲线对判官也不构成取舍，配置要靠别的理由定。
#  (3) 一致性检查：nf8_xpal 就是正式测试里 TRD16 的配置，测试集上对 B7 是 63% [56%,69%]。
#      若它在验证集上落到 [56%,69%] 之外，说明验证集判官与测试集判官口径不一致，
#      **本轮所有臂一律不得用来定配置**（只当探索记录）。
#  (4) 任何配置变更要动测试集 = 对测试集的第二次观察，必须另行预注册后只跑一次。
#
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
set -u
P=${PY:-python}
cd "$(dirname "$0")/.."

# 正式测试的判官（预注册、只跑一次）还在跑时不许上 API：并行会给那次运行带来 API 失败。
while ps -eo args | grep -q '[f]inal_judge.sh'; do
  echo "[wait] final_judge.sh 还在跑，60s 后再看"; sleep 60
done

# 重排操作点：从 nf8_name 的 8 张里按 CLIP B/16 挑（评测/判官用的是别的模型，不自评）
$P eval/rerank.py --src nf8_name --set V_mat --n 2
$P eval/rerank.py --src nf8_name --set V_mat --n 4

J() { $P eval/judge_pairs.py pilot "$@" && $P eval/judge_pairs.py full "$@"; }
for A in nf8_name nf8_name_rr2 nf8_name_rr4 nf8_xpal; do
  J --a "$A" --b B7val_c1.5 --set V_mat --size 16
done
echo VAL_JUDGE_CURVE_DONE
