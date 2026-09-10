# 新架构进度账本

> 每一轮在末尾**追加**一节，只写：做了什么 / 新架构 vs 基线指标表 / 什么在跑 / 下一步。
> 目标见 `GOAL.md`，计划见 `docs/arch_plan.md`。

## 2026-09-11 起步（交互会话）

**做了什么**
- 写下永久目标 `GOAL.md`，重写 `scripts/cron_prompt.md`（旧的审计/论文队列冻结，旧版存档
  `docs/cron_prompt_old_2026-09-11.md`）。
- 调研指标与基线，写 `docs/arch_plan.md`。
- 读旧模型代码定位失败原因：绝对位置编码（不平移等变）、封闭词表材质 ID（521 个）、
  6 层 256 维只训 ≤12 分钟。战绩 D6 40:60、A4 17:83 输给朴素降采样。
- 数据层 `model/tiles_data.py`（亮度秩规范化，自检像素级可逆）。
- 服务器离线可用：torch 2.8 / torchvision 0.23 / transformers 4.51 / diffusers 0.35；
  `openai/clip-vit-base-patch32`、`facebook/dinov2-small`、`nerijs/pixel-art-xl`（HF 缓存）。
  缺：scipy、lpips、torch-fidelity、Inception-FID 权重 → 本机下载 scp，指标自实现。
- 许可：数据集全是 ContentDB 许可干净的包，已排除 Minecraft 系，可训练。

**指标表**：本轮无（M1 评测框架尚未完成）。

**在跑**：无。

**下一步**：M1 —— `eval/metrics.py`（KID/FID、FD-DINOv2、CLIP score、LPIPS 多样性、平铺性）
+ 参照集 + 天花板/地板自检。
