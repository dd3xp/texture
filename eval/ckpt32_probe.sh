#!/usr/bin/env bash
# 32px 缺口的第三条线索：**采样用的检查点已经过拟合了 10000 多步，从来没人从 best.pt 采过样。**
#
# 零 GPU 的发现（只读 `experiments/trd_v10.txt` / `trd_v11d.txt`，本轮）：
#   v10 ：最佳 val **6.5426 @ step 2000**，之后单调变差到 6.869 @ 12000（val32 5.074 -> 5.255）
#   v11d：最佳 val **6.3259 @ step 1000**，之后单调变差到 6.712 @ 12000（val32 4.965 -> 5.200）
# 而两支 32px 的生成命令（`scripts/t32_tmp.sh:3`、`scripts/v11d_train_eval.sh:25`）都写着
# **`--ckpt last.pt`**（`eval/gen_trd.py` 的默认反而是 `best.pt`）——
# 也就是说库里所有 32px 瓦片都来自**验证损失最差的那一版权重**，两支各多训了 10000/11000 步。
# 这解释得通"32px 大多数瓦片没有结构"：过拟合的掩码预测器把每格都推向材质的平均色。
#
# ⚠ 但 val 损失这把尺子**自己就不追踪 32px 结构**：v11d 的 val/val32 两项都比 v10 好
# （6.326 vs 6.543、4.965 vs 5.074），结构门却反过来（v11d 24% vs v10 33%）。
# 所以这里**只当筛选**，判据是结构门/各向异性；真要换配置必须再过同材质直接对判的判官臂
# （与 `eval/judge32_temp.sh` 同规矩：结构门不是人类偏好，不许当优化目标）。
#
# 设计 = 2(代次) x 2(检查点) x 2(温度) 的一半：`last.pt` 那 4 格库里已经有
# （`v10x_direct` / `v10x32_t60` / `v11dx_direct` / `v11dx_t60`），本脚本只补 `best.pt` 那 4 格。
# 其余参数与各自的 `last.pt` 版**逐字相同**（`--bs 8 --cfg 1.5 --pal_mode retrieve --xmodal`），
# 唯一另外的差别是 `--n 1`（省 GPU；门是按瓦片算的比率，不受每材质样本数影响）。
#
# **预注册判据（跑之前写死）**：
#  (1) 一致性检查：四个新配置都必须**出满 125 张**，否则本轮不读。
#  (2) 主判据：四对 best vs last（同代次同温度）里，**至少一对**结构门 +>=10pp 且两比例 p<0.05
#      -> 才认为"检查点选择是 32px 的一条活杠杆"，下一轮给它一条判官臂。
#  (3) 若四对全部 <10pp 或不显著 -> **当场关掉这条线索**，记为否定结果：
#      过拟合不是 32px 塌陷的主因，`--ckpt last.pt` 只是个无害的历史口径。
#  (4) ⚠ 次判据（各向异性）只作描述，**不许**单独据它换配置（条纹化伪影也会抬高它）。
#  (5) 全部在验证集 V_mat 上；测试集一次不碰。
#
# ⚠ `/mnt/data` 0 字节可用 -> 瓦片全部写 `/tmp/ckpt32`，统计在本机用
#   `analysis/arch/scale_diag.py` 重算（scp 回瓦片）。
set -u
P=${PY:-/mnt/data/kw/anaconda3/envs/jzs_train/bin/python}
O=${OUT:-/tmp/ckpt32}
cd "${REPO:-$(dirname "$0")/..}"      # 盘满时脚本只能放 /tmp 跑，这时用 REPO= 指仓库
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# deepspeed 在 **atexit** 里写 triton 自动调优表；盘满时它 Errno 28 -> 退出码非零 ->
# 活干完了才崩、还把 `&&` 链上后面的配置全带掉（本轮实测：第一个配置出满 125 张后整链停）。
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$O/triton}

G="$P eval/gen_trd.py --ckpt best.pt --set V_mat --size 32 --bs 8 --n 1 --cfg 1.5 --pal_mode retrieve --xmodal --out $O"
$G --run runs/trd_v10  --tag v10b_direct            || exit 1
$G --run runs/trd_v10  --temp 0.6 --tag v10b_t60    || exit 1
$G --run runs/trd_v11d --tag v11db_direct           || exit 1
$G --run runs/trd_v11d --temp 0.6 --tag v11db_t60   || exit 1
echo CKPT32_PROBE_DONE
