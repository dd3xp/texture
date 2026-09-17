#!/usr/bin/env bash
# (M43) 启动链：零训练、零 API（judge 阶段自己会读筛子，非 SCREEN_GO 就 exit 3）。
# ⚠ 卡号**写死在文件里**：tmux 不继承 ssh 的环境变量（账本老坑），
#   `CUDA_VISIBLE_DEVICES=2 tmux new-session ...` 传不进去，会静默回落到默认卡。
# ⚠ 显存只看 memory.free（(M35)）：16px 生成带 `--xmodal`（CLIP 图塔 + 检索池）实测 >9GB
#   ⇒ 挑卡按 >12GB 算。
cd /mnt/data/kw/RoundSquisheen/texture
export CUDA_VISIBLE_DEVICES=2
STAGE=gen bash eval/m43_data_ab.sh >> /tmp/m43_gen.txt 2>&1
echo "M43_LAUNCH_EXIT=$?" >> /tmp/m43_gen.txt
