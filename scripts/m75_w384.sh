#!/bin/bash
# (M75) 尺寸臂 —— 控制臂：d=384（现行宽度），从零单阶段 44000 步
#
# ============================ 预注册判据（先于任何数据提交） ============================
# 本臂与 scripts/m75_w512.sh 构成一对「只差宽度」的架构臂。判据、作废条件、预测全部
# 写死在本文件头部；docs/arch_progress.md 同轮登记同一份文本。
#
# 【为什么现在开】(M67) 判 WIDTH_PRIOR_UNOBTAINABLE：宽度的「预期效应量」在本项目内
#   拿不到（17 次生成器训练只有一种架构签名 d=384/depth=12/heads=6/codes=512，宽度一次
#   没被变过）⇒ 按 (M44)/(M54)/(M64) 三层准入门，这条臂开不了。(M67) 同时判定那道门
#   已经「筛掉全部候选」＝架构侧永久停工。
#   ⚠⚠ 本轮**明确披露**：三层准入门**未被满足**，用户也**未**给出豁免（已挂起 6 轮无回应）。
#   开臂依据是 GOAL.md 本身 —— 防跑偏规则 4「结果不好就如实写不好，然后改架构再训，
#   不许转去做别的事来填满时间」、规则 3「进展必须能从 git log 看出来：大部分提交应落在
#   model/…且附带训练日志」，以及用户原话「设置好定时检测继续推进」。
#   在此之前已连续 5 轮零 GPU、零训练 ⇒ 继续等待本身就是规则 4 禁止的行为。
#   ⇒ 本臂**标定为探索臂**：null 的唯一许可读法写死在下面 (C2)，⛔ 不得读成"容量轴关闭"。
#
# 【消掉 (M62) SIZE_ARM_CONFOUNDED 的办法】(M62) 挂起尺寸臂的理由是：v10 由
#   `--init_from runs/trd_v8/last.pt` 续训而来 ⇒ 改宽度无检查点可续，与控制臂差的不只是
#   宽度，还有 32000 步血统 + 一套无判据可选的两段配方。(M62) 第三节把"那一堆自由度"
#   收窄成 **5 键清单**（lr / warmup / p32 / coarse+p_coarse / steps）。
#   本轮的解法＝**两臂都从零单阶段训 44000 步**（＝v10 血统的总步数 20000+12000+12000），
#   5 键各取唯一值、两臂逐键相同 ⇒ 两臂之间**只剩宽度一个差异**。
#     lr 3e-4（v8 段＝从零那一段的值）／warmup 1000（同上）／p32 0.5（v10 段，＝读数仪器
#     所在的那一档）／coarse true + p_coarse 0.5（v10 段）／steps 44000。
#   ⚠ 5 键的取值对"宽度比较是否成立"不起作用（两臂相同），只影响结论的可推广范围。
#
# 【耦合改动，必须披露】d=512 不能整除 heads=6（trd.py:267 的 view 会崩）⇒ 处理臂
#   heads 6→8。两臂 head_dim 因此都恰为 64（384/6 = 512/8 = 64）＝把"每个头多宽"这一维
#   钉死，只放开"几个头/总宽度"。⛔ 这是一条**耦合改动**，不是纯净的单变量。
#
# 【--reseed_after_build】train_trd.py:261 的帮助原文："开 = 两条只差架构的臂看到逐位
#   相同的数据"。两臂都开 ⇒ 参数量不同导致的 RNG 流错位被消掉，训练数据顺序逐位相同。
#
# 【读数（另行在判读器里冻结，本文件只定判据）】
#   仪器＝配对 CLIP@32px：eval/diag_decompose.py --size 32 --xmodal --bs 8 --reps 2
#   --per_image --ckpt last.pt，两臂各 K=28 个种子（与 (M60)(M62) 同口径）；
#   统计单位＝**逐材质 n=67**（⛔ 不是 148 张），配对置换检验（import 冻结的
#   analysis/arch/m60_read_scale.py 的 arm_mean/perm_p）。
#
# 【主判据】D = CLIP(d=512) − CLIP(d=384)，逐材质配对：
#   (C1) mean_D > 0 且置换 p < 0.05            ⇒ **WIDTH_HELPS_32**
#   (C2) 否则                                   ⇒ **WIDTH_NULL_32**
#        ⚠⚠ (C2) 的**唯一许可读法**＝「在这把尺子、这个预算、这个统计单位上，
#        d 384→512 的变化小于本臂 MDE」。⛔ 不许读成"容量轴关闭"、⛔ 不许读成
#        "模型够大了"、⛔ 不许读成"该去加数据"。MDE 按 (M62) 实测口径
#        2.8016·rms(D)/sqrt(67) 计算，⛔ 禁用已被 (M62) 判错的 σ1 公式。
#   (C3) mean_D < 0 且置换 p < 0.05            ⇒ **WIDTH_HURTS_32**（只登记，
#        ⛔ 不许读成"该变窄"——本轮没有 d<384 的臂）。
#
# 【作废条件（任一触发则整轮 VOID，⛔ 不许改判据救）】
#   (V1) 任一臂未训满 44000 步（OOM/被杀/log.json 末步 < 44000）。
#   (V2) 两臂 config.json 除 {out, d, heads} 外有任何一个键不相同。
#   (V3) 两臂读数份数不等，或材质集合不同（same_materials=False）。
#   (V4) 任一臂读数出现 NaN，或 real_half.CLIP 不等于 34.1732177734375（(M61)(OP3) 参照行）。
#
# 【操作检验】
#   (OP1) 料在判读器眼里：**两臂各验一次**（(M62) 可迁移四），用正式 --k 空跑、不传 --out。
#   (OP2) real_half.CLIP 逐位命中 34.1732177734375（与模型/种子无关的参照行）。
#   (OP3) 控制臂内部奇偶对半，mean_D ≈ 0 且 p 不显著（不假阳）。
#   (OP4) 两臂 log.json 末步都是 44000，且 config.json 的 5 键逐键相同。
#   (OP5) 两臂数据顺序逐位相同的间接证据：两臂 codebook_err 相同（codebook 由数据构建，
#         在 seed_for_training 之前 ⇒ 与宽度无关）。
#
# 【跑前方向预测（3 条，判决时逐条记分）】
#   (P1) 判 WIDTH_NULL_32（押自己不想要的结果）。
#   (P2) |mean_D| < 0.1250（＝(M60) 那条"再训 12000 步"的增益），即宽度买到的比预算翻倍少。
#   (P3) 两臂 val 末值都比 runs/trd_v10 的 6.869 更差（单阶段 44000 步没有 v8→v10 的两段血统）。
#
# 【本臂不做什么】⛔ 不碰 final_test.sh、⛔ 不开判官臂、⛔ 不动 (h)、⛔ 不改任何活件、
#   ⛔ 不改 eval/*、⛔ 不动 V_mat/E_mat、⛔ 零 API。
# =======================================================================================
set -e
exec >> /tmp/m75_w384.txt 2>&1
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/tmp/triton_m75_w384
cd /mnt/data/kw/RoundSquisheen/texture
echo "=== m75_w384 start $(date -u) on gpu $CUDA_VISIBLE_DEVICES"
/mnt/data/kw/anaconda3/envs/jzs_train/bin/python -u model/train_trd.py \
  --out /tmp/runs/trd_w384_09191345 \
  --seed 0 --reseed_after_build \
  --steps 44000 --lr 3e-4 --warmup 1000 --p32 0.5 --coarse --p_coarse 0.5 \
  --d 384 --heads 6 \
  --batch 256 --batch32 64 --wd 0.05 --drop 0.1 --depth 12 --codes 512 \
  --p_text_drop 0.1 --p_color_drop 0.5 --eval_every 1000 --sizes 16 32 \
  --bias_freqs 8 --bias_hidden 128 --level_emb --pal_aug 0.3 --pal_smooth 0.1 \
  --extra --extra_file train_extra_packs_only.json --n_ex 4 --p_ex_drop 0.3 \
  --n_domains 2 --save_at 22000 44000
echo "=== m75_w384 done $(date -u)"
echo "M75_W384_DONE"
