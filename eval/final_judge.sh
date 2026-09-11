#!/usr/bin/env bash
# 正式测试的判官（E-mat），在 final_test.sh 之后跑。比较对象在运行前提交 = 预注册。
# 协议不变：每组先试点（15 对真题 + 5 对两边同图的空对照，只量两序一致率），过门槛（65% 且高于空对照）才跑全量。
# 凭据只从环境变量 VLM_BASE_URL / VLM_API_KEY 读。
P=${PY:-python}
$P eval/build_colour_dirs.py --set E_mat --methods TRD16 B7 B2
J() { $P eval/judge_pairs.py pilot "$@" && $P eval/judge_pairs.py full "$@"; }
J --a TRD16       --b B2 --set E_mat --size 16          # 名字→纹理，主配置 vs 最强 SDXL 管线
J --a TRD16       --b B7 --set E_mat --size 16          # vs 同数据扩散 UNet
J --a TRD16c_rr4  --b B2 --set E_mat --size 16          # CLIP 取向配置 vs B2（同 4 选 1 预算）
J --a C_TRD16_E_mat --b C_B2_E_mat --set E_mat --size 16   # 区域颜色任务（同目标色）
J --a C_TRD16_E_mat --b C_B7_E_mat --set E_mat --size 16
J --a TRD24_rr4   --b B2 --set E_mat --size 24
J --a TRD32_rr4   --b B2 --set E_mat --size 32
echo FINAL_JUDGE_DONE
