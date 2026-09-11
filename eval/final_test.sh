#!/usr/bin/env bash
# 正式测试（E-mat：出结果前定死的 272 个测试材质 / 857 张测试真人瓦片）。**只跑一次。**
# 配置全部在验证集上定（docs/arch_progress.md 2026-09-12 各节），本文件在运行前提交 = 预注册，跑完不许改配置重跑。
#
#   TRD16    v8，跨模态检索（N=30），单张不挑                    —— 名字→纹理主配置
#   TRD16c   v8，跨模态检索（N=100）+ CLIP-B/16 4 选 1（与 B2 同预算）—— 名字→纹理 CLIP 取向
#   TRD24    v10，跨模态检索（N=300）+ 4 选 1
#   TRD32    v10，跨模态检索（N=100）+ 4 选 1
#   颜色任务 v8，跨模态检索（调色板按区域颜色 + 典型度）+ labshift 残差校正
#   B7       同数据扩散 UNet（runs/b7_ddpm_09120300），CFG 1.5（验证集调）
# 评测用 CLIP-B/32；分布指标每材质第 0 张；32px 测试真人参照只有 2 张 → 只报无参照指标。
set -e
P=${PY:-python}
export HF_HUB_OFFLINE=1
V8=runs/trd_v8; V10=runs/trd_v10
G="$P eval/gen_trd.py --ckpt last.pt --set E_mat --cfg 1.5 --pal_mode retrieve --xmodal"
$G --run $V8 --size 16 --n 4 --bs 16 --tag TRD16
$G --run $V8 --size 16 --n 4 --bs 16 --ret_nname 100 --tag TRD16c
$P eval/rerank.py --src TRD16c --set E_mat --size 16 --n 4
$G --run $V10 --size 24 --n 4 --bs 8 --ret_nname 300 --tag TRD24
$P eval/rerank.py --src TRD24 --set E_mat --size 24 --n 4
$G --run $V10 --size 32 --n 4 --bs 8 --ret_nname 100 --tag TRD32
$P eval/rerank.py --src TRD32 --set E_mat --size 32 --n 4
$G --run $V8 --size 16 --bs 16 --colour_task --tag TRD16
for S in 16 24 32; do
  $P baselines/ddpm_unet.py gen --run runs/b7_ddpm_09120300 --set E_mat --size $S --n 4 --cfg 1.5 --tag B7
done
$P baselines/ddpm_unet.py gen --run runs/b7_ddpm_09120300 --set E_mat --size 16 --colour_task --cfg 1.5 --tag B7
$P eval/run_eval.py --set E_mat --size 16 --methods B1 B2 B4 B5 B7 TRD16 TRD16c_rr4 --out experiments/final_E_mat_16.json
$P eval/run_eval.py --set E_mat --size 24 --methods B1 B2 B4 B7 TRD24_rr4 --out experiments/final_E_mat_24.json
$P eval/run_eval.py --set E_mat --size 32 --methods B1 B2 B4 B7 TRD32_rr4 --out experiments/final_E_mat_32.json
$P eval/colour_task.py --set E_mat --methods B1+labshift B1+recolor B2+labshift B2+recolor B5+labshift B7+labshift \
   TRD16 TRD16+labshift --out experiments/final_colour_E_mat.json
echo FINAL_TEST_DONE
