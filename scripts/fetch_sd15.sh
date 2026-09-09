#!/bin/bash
# 取 SD 1.5 的 fp16 权重，只取 diffusers 管线必需的文件（约 2.0G）。
#
# B24（`analysis/paired/second_generator.py`）要用的第二个生成器。
# GPU 机器离线且磁盘吃紧，所以流程是：**在能联网的机器上跑本脚本 -> scp 过去 -> 跑完删掉**。
#   bash scripts/fetch_sd15.sh /tmp/sd15
#   scp -r /tmp/sd15 <gpu>:<某处>/models/
#   python analysis/paired/second_generator.py --model <某处>/models/sd15
#   ssh <gpu> rm -rf <某处>/models/sd15     # 用完还空间，那台盘是共享且已满
# 不装 huggingface_hub，直接 curl，避免往本机环境加依赖。
set -u
DST="$1"
REPO="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main"
FILES="
model_index.json
scheduler/scheduler_config.json
text_encoder/config.json
text_encoder/model.fp16.safetensors
tokenizer/merges.txt
tokenizer/special_tokens_map.json
tokenizer/tokenizer_config.json
tokenizer/vocab.json
unet/config.json
unet/diffusion_pytorch_model.fp16.safetensors
vae/config.json
vae/diffusion_pytorch_model.fp16.safetensors
feature_extractor/preprocessor_config.json
"
mkdir -p "$DST"
fail=0
for f in $FILES; do
  mkdir -p "$DST/$(dirname "$f")"
  if [ -s "$DST/$f" ]; then echo "已有 $f"; continue; fi
  code=$(curl -sL --retry 4 --retry-delay 3 -w '%{http_code}' -o "$DST/$f" "$REPO/$f")
  sz=$(stat -c %s "$DST/$f" 2>/dev/null || echo 0)
  if [ "$code" != "200" ] || [ "$sz" -lt 100 ]; then
    echo "**失败** $f (HTTP $code, $sz bytes)"; fail=1
  else
    echo "OK $f ($((sz/1024))KB)"
  fi
done
echo "---"
du -sh "$DST"
[ "$fail" = 0 ] && echo "全部就绪" || echo "**有文件失败，勿上传**"
