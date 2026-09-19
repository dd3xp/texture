# 标题与摘要（占位稿 v0，2026-09-18）

⚠ **占位性质**：GOAL.md 未改，本文件只是把题目和摘要先钉下来，方便后续实验对着它补料。
⚠ **纪律**：摘要里每一句都必须在 §三 的证据表里有出处；⛔ 没出处的话不许留在摘要里。
⚠ 标着 `[待补]` 的数字是**还没跑出来**的，⛔ 不许先填一个"大概"。

## 一、标题候选（推荐第 1 个）

1. **Native Generation of Tileable Pixel-Art Textures from a Material Name and a Target Colour**
   —— 直说任务与两个条件；"Native" 对上我们与"渲染再降采样"的分野。
2. Tile-Native Pixel Art: Retrieval-Augmented Discrete Generation of Low-Resolution Textures
   —— 把方法放进标题（检索增强 + 离散生成），但"Tile-Native"是新造词，有风险。
3. Small Canvases, Real Constraints: Generating Tileable 16×16 Textures under Palette and Colour Conditions
   —— 图形学味更重，ICLR 侧偏文艺。

⚠ 标题里**不要**出现 "SOTA"、"first"（检索非穷尽，只能在正文写"据我们所知未见"）。

## 二、摘要（英文占位稿，约 200 词）

> Game and mod textures are authored natively at tiny canvases: a 16×16 tile is not a
> downscaled photograph but a grid in which every pixel is a design decision, the palette
> holds at most a handful of slots, and the tile must stay continuous across its own wrap
> boundary because it is used tiled. We study the task of painting a solid-colour region of
> an image with a tileable pixel-art texture given only a material name and the region's
> colour, and we generate it **natively at the target resolution** rather than rendering
> large and downsampling. Our model decodes palette-index and luminance-rank tokens in
> parallel, places a parameter-free toroidal relative attention bias on the grid, and
> conditions on palettes retrieved from a memory of artist palettes. Ablating that bias
> costs 18.5 KID, and replacing its periodic encoding with non-periodic features of
> identical parameter count still costs 7.6, so what carries the model is the periodic
> form itself; tileability, by contrast, is carried by cyclic-shift augmentation on tiles
> that are themselves tileable. Against render-and-downsample pipelines, a same-data pixel-space diffusion
> baseline, a published score-distillation pixel-art method, and direct retrieval of artist
> tiles, our 16×16 outputs are preferred by a de-biased pairwise judge in [待补：B5 臂] of N
> head-to-head comparisons, while costing seconds instead of GPU-hours. Getting these
> comparisons to mean anything required repairing the rulers: we show that
> render-and-downsample baselines collect a structural gain purely from a larger canvas,
> that a widely used tiling score awards a perfect value to a flat fill, and that a
> pairwise judge's resolvable-rate floor is far below chance.

**⛔ 摘要里刻意没写的三件事**（写了就站不住）：
- ⛔ 不写"三档全面领先"：24px 打平、32px 的比较尺子未校准（环不闭合 p=0.0074、B2 画布增益 78%、
  族级 CI [0.315,0.502] 含 0.5）→ 这些进 Limitations，不进摘要。
- ⛔ 不写"首个"或"SOTA"。
- ⛔ 不写"比真人更可平铺"（(P3) 已证那是参照带构造缺陷）。

### 中文对照（给自己看，不入稿）

游戏与模组材质天生画在极小画布上：16×16 不是照片缩小，而是每个像素都是设计决定、调色板只有十几个槽、
且因为要平铺使用所以必须在自己的环绕边界上保持连续。我们研究"只给材质名与区域颜色，把一块纯色区域
画成可平铺像素材质"这个任务，并且**在目标分辨率上原生生成**，而不是渲染大图再降采样。模型并行解码
调色板索引与亮度秩两路 token，在网格上加一个**不增加参数**的环面相对注意力偏置，并以从艺术家调色板库
检索到的调色板作为条件。去掉该偏置 KID 掉 18.5，把它的周期编码换成**同参数量**的非周期编码仍掉 7.6，
⇒ 承重的是**周期形式本身**；而可平铺性则由**循环平移增广 + 本身可平铺的训练数据**承载。对照渲染降采样管线、同数据像素空间扩散、已发表的分数蒸馏像素画方法、
以及直接检索艺术家瓦片，我们的 16×16 产物在去偏两两判官下被偏好 [待补]，而成本是秒级而非 GPU 小时级。
要让这些比较有意义，先得修尺子：我们指出渲染降采样基线**仅因画布更大**就白拿结构增益、一个被广泛使用的
可平铺分数会给**纯色填充满分**、而两两判官的可解率地板远低于掷硬币。

## 三、证据表（摘要每句 → 出处；⛔ 出处为空的句子不许留）

| 摘要断言 | 出处 | 状态 |
|---|---|---|
| 16px 是原生创作尺寸、每像素是设计决定、调色板极小 | 数据集统计（`dataset_k16.json`：中位 16 色 / 2172 张用满 16 槽）+ AIIDE 2022 的 Pixel Art Characterization | ✅ |
| 瓦片要在环绕边界连续 | 任务定义 + `tile_seam_ratio` 口径 | ✅ |
| 原生生成 vs 渲染降采样 | B1/B2/B4 三列 | ✅ |
| 并行解码调色板索引 + 亮度秩 | 方法 | ✅ |
| 环面偏置不增参数；**周期形式对质量承重**；可平铺由增广+数据承载 | (P8) `BIAS_NULL`（性质）+ 质量读数：A1 置零 +18.49 KID、A2 同参数量非周期 +7.55、地板 4.76（`experiments/p8_read.json`）；⛔ **不许再写「构造上保住可平铺」** | ✅（可平铺那半待 (P8b) 2×2 坐实） |
| 检索增强调色板记忆 | 16px 消融 AB_pal 126/190=66% p=8e-06、族级 [.574,.747] `W1_robust` | ✅ |
| 判官偏好我们（对 B2 / B7 / SD-πXL / B5） | vs B7 63.0%（`W1_robust`）、区域上色任务 vs B2 60.3% / vs B7 60.6%（均 `W1_robust`）、vs SD-πXL 11/12 p=0.0064、**vs B5 58.9% p=0.0136 族级 [.502,.668] `W1_robust`（P1 已跑完）**；**vs B2 16px 57.3% 必须同引族级 CI [0.4999,0.641] 含 0.5**；⛔ **区域上色任务 vs B5 = 53.4% p=0.367 未测到差异，不许混进「多数场次」** | ✅ |
| 秒级 vs GPU 小时级 | SD-πXL 实测 16px 中位 4.44 h / 32px 3.45 h；我们的精确数字 [待补]（P7 等独占卡） | ◑ 待补 |
| 降采样基线仅因画布更大就白拿增益 | B2@32 胜 B2@16 = 107/138 = 78% p=5.5e-11，族级 `W1_robust`；构造性前提 `side=4.5×per` 与目标尺寸无关 | ✅ |
| 广泛使用的可平铺分数给纯色满分 | (P3) `A_const` 在 TS 口径上 0.000 vs 最好真方法 5.31（`experiments/seam_calib_E_mat_16.json`） | ✅ |
| 判官可解率地板远低于掷硬币 | 去重 118 道空对照 21/118 = 17.8% | ✅ |

## 四、接下来要补的料（对着摘要缺口）

1. **P1** 判官对 B5 三条臂 → 填"被偏好 N/M"那句（在跑）。
2. **P7** 速度表 → 填"秒级"的精确数字（等独占卡；要含 4 选 1 重排开销）。
3. **P8** 环面偏置消融（关掉 / 非环面 / 故意破坏 / 与随机 roll 增强 2×2）→ 支撑"构造上保可平铺 + 不增参数"那句。
4. **P2** AB_xm / AB_rr 免门重问 → 把消融表填满（等判官空闲）。
