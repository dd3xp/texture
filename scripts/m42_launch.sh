#!/usr/bin/env bash
# (M42) 启动链：零 API、零训练，只出图 + 量 pixfrac + 盲写判读器给判决。
# ⚠ 卡号**写死在文件里**：tmux 不继承 ssh 的环境变量（账本老坑），`CUDA_VISIBLE_DEVICES=2 tmux
# new-session ...` 传不进去，会静默回落到默认卡。⚠ 显存看 memory.free（(M35)）：16px 生成
# 带 `--xmodal`（CLIP 图塔 + 检索池）实测 **>9GB**，9.2GB 空闲的 GPU 3 上 OOM 过一次 -> 用 GPU 2（29GB）。
cd /mnt/data/kw/RoundSquisheen/texture
export CUDA_VISIBLE_DEVICES=2
bash eval/m42_seed_null.sh >> /tmp/m42_seed.txt 2>&1
echo "M42_LAUNCH_EXIT=$?" >> /tmp/m42_seed.txt
