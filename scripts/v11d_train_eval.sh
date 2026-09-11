#!/usr/bin/env bash
# v11d = v10 接着训 + 64→32 众数降采样补的 1236 张 32px 真人瓦片（32px 训练瓦片 641 → 1877），p32 0.7。
#
# 为什么另起一个名字：2026-09-12 两个并行会话各自启了一个叫 "v11" 的训练，**都写 runs/trd_v11**
# （一个是固定相位粗网格、从 v8 接着训，见 scripts/v11_launch_tmp.sh；一个是本脚本这个补数据的），
# last.pt / best.pt / step_6000.pt / log.json 每 1000 步互相覆盖一次，谁的权重都说不清。
# 这里用独立的 --out / 日志 / 评测 JSON / 标签；train_trd.py 也加了 .trainlock，再撞会直接拒绝启动。
#
# 起法（凭据只从环境变量给，不落盘）：
#   tmux new-session -d -s arch_v11d "VLM_BASE_URL=... VLM_API_KEY=... bash scripts/v11d_train_eval.sh"
set -u
P=/mnt/data/kw/anaconda3/envs/jzs_train/bin/python
cd /mnt/data/kw/RoundSquisheen/texture
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

$P -u model/train_trd.py --out runs/trd_v11d --steps 12000 --lr 1.5e-4 --warmup 500 \
  --init_from runs/trd_v10/last.pt --coarse --p_coarse 0.5 --n_ex 4 \
  --extra --extra_file train_extra_packs_only.json+train_64to32.json \
  --level_emb --bias_freqs 8 --bias_hidden 128 --pal_aug 0.3 --pal_smooth 0.1 \
  --sizes 16 32 --p32 0.7 --save_at 6000 12000 > experiments/trd_v11d.txt 2>&1 || exit 1

# 另一条链（v11_launch_tmp.sh）也在 7 号卡上跑生成/评测，等它腾出显存再评
while ps -eo args | grep -qE "[g]en_trd.py|[r]un_eval.py"; do sleep 60; done

G="$P eval/gen_trd.py --run runs/trd_v11d --ckpt last.pt --set V_mat --bs 8 --n 2 --cfg 1.5 --pal_mode retrieve --xmodal"
$G --size 16 --tag v11dx || exit 1                 # 16px 是目前领先 B2 的那一档，确认没被改坏
for S in 32 24; do
  $G --size $S --tag v11dx_direct || exit 1
  $G --size $S --temp 0.6 --tag v11dx_t60 || exit 1
done
$P eval/run_eval.py --set V_mat --size 16 --methods B2val v8x v10x v11dx --out experiments/eval_v11d_Vmat_16.json
$P eval/run_eval.py --set V_mat --size 32 --methods B1val B2val v10x_direct v11dx_direct v11dx_t60 --out experiments/eval_v11d_Vmat_32.json
$P eval/run_eval.py --set V_mat --size 24 --methods B1val B2val v10x_direct v11dx_direct v11dx_t60 --out experiments/eval_v11d_Vmat_24.json

# 判官：32px 上 v10 级联已经"分不出"，真人 vs B2 也分不出 → 这一问只在补了数据后仍显著输时才有信息
$P eval/judge_pairs.py pilot --a v11dx_direct --b B2val --set V_mat --size 32 &&
  $P eval/judge_pairs.py full --a v11dx_direct --b B2val --set V_mat --size 32
echo V11D_DONE
