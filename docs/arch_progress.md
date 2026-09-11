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

## 2026-09-11 上午（交互会话，续）

**做了什么**
- **M1 完成**：`eval/metrics.py`（KID/FID、FD-DINOv2、CLIP、LPIPS 多样性、平铺比）。自检通过：
  真人 val 对 test = KID 2.96e-3 / FID 39.1 / FD-DINOv2 43.8，打乱像素、纯色、噪声全部远差于它。
  ⚠ CLIP 只看语义/颜色不看结构（打乱像素 34.0 反比真人 33.4 高），结构靠 KID/FID/FD-DINOv2。
  ⚠ 平铺比只能揪出"接缝比内部更断"，分不出噪声（噪声 0.996、真人 1.04）。
- **评测提示词集定死**（`eval/prompt_sets.json`，出结果前提交）：E-mat 272 材质 / 参照 857 张；
  E-all 447 / 1265。
- **M2 基线**：`baselines/sdxl_baselines.py`（B1 SDXL+降采样、B2 现有管线、B4 像素画 LoRA）；
  `baselines/sdpixl/run_subset.py`（B3 SD-piXL，**实测 3h22m/张**，只跑固定 12 材质子集
  `eval/sdpixl_subset.json`，默认步数不降）；B5 检索在 `eval/run_eval.py` 里现场构造。
- **M3 新架构 TRD v1**：`model/trd.py` + `model/train_trd.py`（33.1M 参数，512 色码本误差 7.2/255）。
  首次开训因注意力偏置按批次复制而极慢（>10 分钟/千步），已修（偏置广播、PAD 槽不屏蔽）。
- 评测执行器 `eval/gen_trd.py` / `eval/run_eval.py`（分布指标每方法每材质只取第 0 张，n 相同）。

**早期探针指标**（E-mat 16px，**B1 只出了 59/272，FID 不可比，看 KID**）

| 方法 | KID×10³ | FD-DINOv2 | CLIP | LPIPS 多样性 |
| --- | --- | --- | --- | --- |
| B5 检索（真人瓦片，参照线） | 9.0 | 38.9 | 33.3 | 0.368 |
| B1 SDXL+降采样（59 个） | 64.4 | 291 | 34.8 | 0.207 |

SDXL+降采样离真人分布很远——这是新架构要打的缺口。检索返回的就是真人瓦片，分布指标天然好，
但不能生成新纹理、不能跟随区域颜色，报告时作参照线并注明。

**在跑（服务器 tmux）**
- `arch_b12`（GPU 6）：B1/B2，272 材质，写 `experiments/baselines/B1|B2/`，日志 `experiments/baselines_b12.txt`
- `arch_b4`（GPU 7）：B4 像素画 LoRA，日志 `experiments/baselines_b4.txt`
- `arch_trd_v1`（GPU 6）：TRD v1，2 万步，`runs/trd_v1/`，日志 `experiments/trd_v1.txt`；
  第 1000 步 train 4.57 / val 7.41（val 不带颜色条件、且是没见过的包，口径比 train 难，看趋势）

**下一步（按顺序）**
1. `arch_b12` 跑完后（B3 的调色板取自 B1 第 0 张，所以必须等它）：在**空出来的卡**上起 B3：
   `tmux new-session -d -s arch_b3 "cd <项目目录> && SDPIXL_DIR=<pixel 项目下的 SD-piXL 目录> SDPIXL_PY=<SD-piXL 环境的 python> HF_HUB_OFFLINE=1 <jzs_train python> -u baselines/sdpixl/run_subset.py --gpus <卡号> >> experiments/b3.txt 2>&1"`
   （具体路径见 `baselines/sdpixl/run_sweep.sh` 的 SDPIXL/PY 两个变量；每张 19.5GB，12 张约 40 GPU 小时，
   可给两块卡 `--gpus 6 7`）。
2. `arch_trd_v1` 跑完后：`python eval/gen_trd.py --run runs/trd_v1` 出 E-mat 每材质 4 张，
   然后 `python eval/run_eval.py --methods B1 B2 B4 B5 TRD_trd_v1_cfg2.0_t1.0` 出**第一张完整对比表**，
   记进本账本。看 `runs/trd_v1/val_samples.png` 目视结构材质（砖、木板）是否还出噪点。
3. 按结果改架构（v2 候选：大模型渲染嵌入作条件、32px 数据联合训练、CFG/温度扫描），回到 2。

## 2026-09-11 中午（交互会话，续）：v1 诊断 → v2

**v1 诊断**（`runs/trd_v1`，第 3000 步）：
- 颜色语义对（砖红/紫、木板棕黄、石灰、沙淡黄、树皮棕）——文本条件在颜色上起作用。
- **结构完全没有**：石砖/砖/木板没有砖缝和板条，只有斑点或近乎纯色（`experiments/diag_last.png`）。
  **旧模型的失败重演了。**
- 曲线：结构损失 2.03→1.96 基本不动；调色板损失 2.54→0.81 在背答案；验证 7.41→10.12 一路涨
  （拆开看：val_grid 2.31→2.21，val_pal 5.1→7.9，涨的是调色板）。

**原因（我的设计缺陷）**：模型没有绝对位置，"往上看 4 行"只能靠环面偏置表达，而 v1 只喂了
**基频** sin/cos，偏置天生偏向"越近越相关"的平滑衰减 = 斑点，表达不了"每 4 格一道砖缝"。
另：秩的含义随色阶数 k 变；调色板在背训练包的色值。

**v2**（`runs/trd_v2`，GPU 7，tmux `arch_trd_v2`，日志 `experiments/trd_v2.txt`）：
`--bias_freqs 8 --bias_hidden 128 --level_emb --pal_aug 0.3 --pal_smooth 0.1 --sizes 16 32`
- 8 次谐波：16 格环面上任意函数可精确表示，砖缝周期能表达；
- 归一化色阶嵌入：秩 r → round(15r/(k-1))，"最亮/最暗"跨 k 共享；
- 调色板颜色抖动（色相 ±18°、亮度 ±15%，每批重编码，平均色条件同步重算）+ 标签平滑；
- 16 与 32 混训（32px 测试集只有 6 张 → 32px 以 val 156 张为参照，选检查点只看 16px val）。
v1 继续跑完作消融对照。所有新开关默认 = v1 行为，旧检查点经 `model_from_args` 照常加载。

**任务工具**：`tools/paint_region_trd.py`（区域色作平均色条件，配色原生跟随；环面 → 无接缝）。
第 2000 步 v1 实测：颜色跟随了、平铺无缝，但"wooden planks"是斑点——正是 v2 要修的。

**下一步**
1. v2 到 ~3000 步：跑同样的诊断（给 stone brick / brick / wood planks / cobble 采样），
   **结构出来了没有**是 v2 的第一个判据。没出来 → 先查采样（MaskGIT 置信度解码对需要全局协调的
   周期结构可能不利：试降 choice_temp、增步数），再查模型（遮一半真砖图看能否补全砖缝）。
2. `arch_b12` 跑完 → 在 GPU 6 起 B3 SD-piXL（见上一节第 1 步）。
3. v1/v2 训完 → `eval/gen_trd.py` + `eval/run_eval.py` 出完整对比表。
