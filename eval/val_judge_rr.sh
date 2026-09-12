#!/usr/bin/env bash
# 曲线两端**直接对判**：4 选 1 重排的样本 vs 同一材质的单张样本（验证集，16px）。
#
# 为什么（承 `eval/val_judge_curve.sh` 那轮的结果）：四个操作点各自对 B7 的胜率是
#   名字检索单张 47% [37,57] -> 2 选 1 57% [47,67] -> 4 选 1 63% [53,73]（跨模态硬取 61% [51,70]）
# 方向单调、跟着 CLIP 走、与 FD 反着走，但**预注册判据 ②（Jeffreys 区间不重叠）一条都没满足**
# ——每臂只有约 95 个有效对，区间宽 ±10pp，而全曲线只铺开 16pp。
# 各自对第三方（B7）比是**非配对**的，白扔掉"同一材质"这个配对信息。
# 这里改成同材质直接对判，同样的 API 预算下灵敏度高得多。
#
# 只判**重排真的换了样本**的材质（`eval/rr_diff_subset.py`）：挑中第 0 张时两边逐像素相同，
# 判官只能瞎答。筛选只看 CLIP 分数，与判官胜负无关。
#
# **预注册判据（跑之前写死）**：
#  (1) 先试点（15 真题 + 5 两边同图空对照），可解率 >= 65% 且高于空对照才跑 full。
#  (2) 主判据：`nf8_name_rr4` 在这些材质上的胜率，二项双侧 p < 0.05 才算判出方向。
#  (3) **方向预测（写在跑之前）**：预测 rr4 胜。若 rr4 **打平或输**，
#      则"判官跟着 CLIP 走"这个读法当场撤回，上一轮那条单调梯度记为噪声。
#  (4) 这只动验证集。据此换配置再测测试集 = 对测试集的第二次观察，必须另行预注册。
#
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
set -u
P=${PY:-python}
cd "$(dirname "$0")/.."

$P eval/rr_diff_subset.py --src nf8_name --rr nf8_name_rr4 --set V_mat --out eval/rr4_diff.json
$P eval/judge_pairs.py pilot --a nf8_name_rr4 --b nf8_name --set V_mat --size 16 --subset eval/rr4_diff.json \
  && $P eval/judge_pairs.py full --a nf8_name_rr4 --b nf8_name --set V_mat --size 16 --subset eval/rr4_diff.json
echo VAL_JUDGE_RR_DONE
