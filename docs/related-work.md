# 文献调研

调研日期：2026-08-20

## 摘要

pixel art 的学术工作几乎全部集中在 **SIGGRAPH Asia 一条线**上（2018 / 2022 / 2024 隔届必有），
分成两个几乎不相交的族。**两族之间是空的**——没有人做"结构已定，只生成材质"这个中间层。
这是本项目要占的位置。

---

## 族 A：Pixelization（照片 → 像素画），已成熟

| 工作 | 出处 | 要点 |
| --- | --- | --- |
| Depixelizing Pixel Art | Kopf & Lischinski, SIGGRAPH 2011, TOG 30(4) | 反向问题：像素画 → 分辨率无关矢量表示。解决对角邻接歧义。 |
| Pixelated Image Abstraction | Gerstner, DeCarlo, Alexa, Finkelstein, Gingold, Nealen. NPAR 2012；期刊版 C&G 2013 | superpixel + 调色板**联合优化**。有正式用户研究和专家访谈。 |
| Deep Unsupervised Pixelization | SIGGRAPH Asia 2018 | 无监督，可控 cell size。用参考像素画正则化 cell 结构。 |
| Make Your Own Sprites: Aliasing-Aware and Cell-Controllable Pixelization | TOG 41(6), SIGGRAPH Asia 2022 (Cardiff) | 把 pixelization 解耦为 cell-aware 和 aliasing-aware 两阶段。 |
| Structure-Aware Pixel Art Scaling via Block Size Detection | MDPI Appl. Sci. 2026 | 像素画放缩，block size 检测。 |

**共同点**：全部关心 "一个 cell 有多大"，即**几何尺度**。
**没有任何一篇问**："在这个尺度下纹理应该长成什么样"，即**语义尺度**。
这个区分是本项目 contribution statement 的核心。

## 族 B：Sprite 生成，做得较弱

| 工作 | 出处 | 要点 |
| --- | --- | --- |
| Generating Pixel Art Character Sprites using GANs | arXiv 2208.06413 (2022) | 角色 sprite 换朝向。 |
| On the Challenges of Generating Pixel Art Character Sprites Using GANs | AIIDE 2022 | 同上，负面结果分析。 |
| A Missing Data Imputation GAN for Character Sprite Generation | arXiv 2409.10721 (2024) | 缺失朝向补全。 |
| Pixel art character generation as an image-to-image translation problem using GANs | Graphical Models (ScienceDirect, 2024) | Pix2Pix 架构改造。 |
| Pixel VQ-VAE | arXiv 2203.12130 (2022) | 像素画的离散表示学习。 |
| PixDiff-PIG: Palette-Informed Diffusion for Pixel Art Generation | 2026（**会议档次不明，仅在 ResearchGate 找到，需核实**） | k-means 调色板 + OKLab + TinyUNet DDPM + BLIP caption。42k sprite 数据集。 |

**数据规模是这一族的瓶颈**：Tiny Hero 只有 **912 对**，扩展版 14,202 对，PixDiff-PIG 的 42k 是目前最大。
另有 DiffusionDB-pixelart（CC0, 2000 张），但是 SD 生成的而非真人绘制，作为训练数据质量存疑。

---

## 两个直接撞车的工作

### 威胁 1：SD-πXL（SIGGRAPH Asia 2024）— 最高优先级

- Binninger & Sorkine-Hornung (ETH Zurich)
- 项目页 https://igl.ethz.ch/projects/sd-pixl/ ｜ 代码 https://github.com/AlexandreBinninger/SD-piXL ｜ arXiv 2410.06236
- 本地已 clone：`emnlp:/mnt/data/kw/RoundSquisheen/pixel/SD-piXL`

**输入**：prompt + 可选图像做空间条件 + 任意 H×W + 任意 n 色调色板。
**方法**：score distillation sampling + 可微图像生成器，Gumbel-softmax 从调色板采样。
**应用**：十字绣、拼豆、乐高等 fabrication 场景。

这几乎是本项目任务的超集。审稿人第一个问题必然是：
> 给 SD-πXL 喂一张纯棕色图 + prompt "wood"，为什么不够？

#### 实测结果（2026-08-20）

复现设置见 `baselines/sdpixl/`。四个 run 并行跑在 emnlp GPU 4-7，各 10001 步：
纯棕色满幅图 + `"wood texture, wooden planks"` 出 16×16 与 64×64；
纯棕色圆角矩形置于白底 + `"a wooden crate, wood grain"` 出 32×32 与 128×128。
调色板为 8 色木质色阶，色相与输入棕色一致。
对照图 `experiments/sdpixl_probe/montage_final.png`。

原先预判三条，实测**两条成立、一条不成立**。

**① 边界保持不住 —— 成立，且比预期严重。**

输入里那片平整背景是单一颜色，跑完后：

| 输出 | 全图用色 | 外圈（背景）用色 | 外圈主色占比 |
| --- | --- | --- | --- |
| 16×16 | 1 / 8 | 1 | 100% |
| 32×32 | 8 / 8 | 3 | 52% |
| 64×64 | 8 / 8 | 6 | — （满幅纹理，无背景） |
| 128×128 | 8 / 8 | 8 | 40% |

32×32 的背景被横向条带啃穿，128×128 的背景变成满屏噪点、主色只剩 40%。
纹理不受区域约束地漫过整幅画布。对本任务（sprite 轮廓神圣）这是硬伤。

另外 128×128 的"箱子"被 SDXL 先验拽成了带透视的三维开口箱，
而不是一张平面 sprite——语义也漂了。

**② 慢 —— 成立，且成本与输出分辨率无关。**

| 输出 | 墙钟时间 |
| --- | --- |
| 16×16 | 203 min |
| 32×32 | 201 min |
| 64×64 | 200 min |
| 128×128 | 200 min |

四者几乎相同：开销由 SDXL 的前向/反向主导，与像素数无关。
也就是说**给小 sprite 不会更便宜**，恰好和像素画工具需要的性质相反。
（另有 8/16 的独立 run：16×16 花 3h46m，19.48 GB 显存。）

**③ 只会产出"量化过的照片"、没有像素画惯例 —— 不成立，需要修正。**

64×64 跑满 10000 步后收敛出了**相当像样的竖向木板纹理**：硬边、限定调色板、
板宽有变化。中途（step 2000）看还是噪声，5000 步开始成条，10000 步成形。
32×32 也长出了可辨认的、带横向板条和边框的木箱。
在这两个分辨率上 SD-πXL 确实能给出可用的木质纹理，
早先"只是噪声"的判断是基于 20% 进度的快照，判早了。

保留意见：它用的是实色色带，没有把**抖动当作明暗过渡**来用——
而这正是像素画表达色阶的核心手法。所以"缺少像素画惯例"这一点部分成立，
但不能作为主要卖点。

**④ 意外发现：16×16 完全崩溃。**

这是最强的一条。10001 步、3h23m 之后，16×16 的输出是**一块纯色**——
8 色调色板只用掉 1 色。SDXL 的先验工作在 1024²，
当可微生成器只有 256 个像素可动时，蒸馏梯度给不出任何连贯结构。
**崩溃恰好发生在定义像素画的那个分辨率区间。**

**⑤ 分辨率是噪声/语义漂移轴，不是 motif 重设计轴。**

同输入同 prompt 下：16×16 全平，64×64 竖板条；32×32 平面板条箱，
128×128 三维透视箱。变化不是"艺术家为像素预算重新设计母题"，
而是模型容量变化导致的先验漂移，且不可控。
这正面支持了本项目的立论——现有最强方法根本没把分辨率当成设计维度。

#### 干净扫描：修正上面的结论（2026-08-21）

上面那轮探针有两个设计问题：flat 输入配 16/64、crate 输入配 32/128，
分辨率和输入类型混淆；调色板没有中性色，白背景被映射成浅棕，纹理越界看不出来。

修正后重跑（`baselines/sdpixl/run_sweep.sh`）：**同一张 crate 输入、同一 prompt、
同一 wood9bg 调色板（8 色木质色阶 + 纯白背景色），只扫 16/32/64/128**。
对照图 `experiments/sdpixl_sweep/montage_sweep.png`。
注意调色板从 8 色变 9 色，本轮绝对数值不能与上一轮直接比。

| 输出 | 全图用色 | 外圈用色 | 外圈主色占比 | 背景白色残留（初始 44%） |
| --- | --- | --- | --- | --- |
| 16×16 | 3 / 9 | 1 | 100% | **0%** |
| 32×32 | 9 / 9 | 2 | 75% | 28% |
| 64×64 | 9 / 9 | 1 | 100% | **35%** |
| 128×128 | 9 / 9 | 9 | 41% | **0%** |

**必须撤回的一条：16×16 崩溃不成立。**

探针那轮 16×16 输出一块纯色，我当时把它当成最强的 motivation。干净扫描里
16×16 画出了**可辨认的、带竖板条的木箱**。也就是说那次崩溃是
"flat 满幅输入 + wood texture 这个 prompt" 特有的，**不是分辨率本身的性质**。
这条不能写进论文。

**仍然成立、且现在有了单变量证据的一条：平整区域在两端都保不住，
而且是两种不同的失败。**

背景白色残留随分辨率呈 0% → 28% → 35% → 0%，两头归零：

- **16×16**：物体涨满整幅画布，背景被整片重绘成实心棕（外圈 100% 单色但不是白）。
  失败方式是**吞掉背景**。
- **128×128**：背景变成满屏噪点（外圈用满 9 色，主色仅 41%）。
  失败方式是**噪声淹没背景**。
- 中段 32/64 保住了部分背景，64×64 的外圈甚至是 100% 单色（纯白），
  但物体周围仍有一圈浅棕"阴影"晕开。

**64×64 是甜点区，现在是干净的单变量结论。**
它同时拿到最高的背景保留率和最整齐的外圈，视觉上也最像可用的木箱贴图。
而像素画常用的 16–32 档恰好落在甜点区之外。

**motif 确实随分辨率变，但是先验漂移而非重新设计。**
16×16 是正面的板条面板；32/64 是正面带边框的板条箱；
128×128 变成**带透视的三维开口箱**，不再是平面 sprite。
变化方向由模型容量决定，用户无法控制——这正面支持本项目立论，
但论据是"不可控"，不是"崩溃"。

**成本与分辨率无关，第二次验证。**
本轮四档用时 199 / 199 / 199 / 202 分钟，与输出像素数无关。

#### 对项目的结论

**项目不需要重新设计，可以继续。** 经干净扫描修正后，可用的 motivation 按强度排序：

1. **平整区域在两端都保不住，且是两种不同的失败**（16 吞掉背景，128 噪声淹没背景）。
   有单变量证据和可量化指标（背景色残留率 0% / 28% / 35% / 0%）。
2. **成本与输出分辨率完全脱钩**（两轮共八个 run，均约 200 分钟）。
   意味着给小 sprite 不会更便宜，与像素画工具的需求相反。
3. **motif 随分辨率漂移且不可控**（16 平面板条 → 64 正面箱 → 128 三维透视箱）。
   论据是"不可控"，不是"做不到"。

两条不能用：
- **"16×16 崩溃"已撤回**，那是特定输入+prompt 的产物，不是分辨率的性质。
- **"不像像素画"不要当主要论据**——中等分辨率上它做得比预期好。

### 威胁 2：TU Delft 本科论文（2025-06）

- *Procedural texturing for pixel art: Making pixel art resemble real materials*
- Francisco Siqueira Carneiro da Cunha Neto，BSc CS&E, TU Delft
- 导师：Elmar Eisemann, Petr Kellnhofer, Mathijs Molenaar
- https://repository.tudelft.nl/file/File_08448c65-f780-421c-8c7b-f2d9a64b6631

**任务定义与本项目一致。** 但方法有硬伤：需要三个输入——
源图 + 二值 mask + **手工制作的元素 sprite sheet（8 个 45° 朝向变体）**。
即纹理素材需人先画好，算法只负责摆放。运行 <1s。

对本项目的意义：
- 证明该问题被认真的图形学研究者（Eisemann 组）认为值得做
- 本科毕业论文，不占顶会坑位，是完美的 related work + baseline
- 其核心限制"必须手工提供纹理元素"正是本项目要消灭的东西

---

## 确认为空白的两块地

### 1. 语义层面的 resolution conditioning

查了 PiD、PixelFlow、NoiseShift 等所有 resolution-conditioned 生成工作，
**全部是关于计算效率和多尺度采样**，没有一篇研究
"艺术家在不同像素预算下会重新设计 motif"。

窗口正在关闭：PixDiff-PIG 的 future work 明确写了
"multi-scale diffusion for higher-resolution assets"，说明有人在往这看但还没做。

### 2. 像素画专用评估指标

这个领域**所有人都在用 FID**（PixDiff-PIG、GAN 系列全都是）。
但 FID 在 16×16、8 色调色板的图上基本是坏的——Inception 特征在该分布上没有意义。
**没有任何人提出过 pixel-art 专用指标。**

可提出的方向：
- 调色板一致性（palette adherence）
- 网格对齐度（FFT 上检测非整数周期能量）
- 抖动模式统计与真人作品的分布距离
- 边缘锐度分布

一个正经的 benchmark + 有人类研究支撑的指标，本身在 CVPR 就是可发表的贡献。

---

## 投稿建议

| 会议 | 评估 | 理由 |
| --- | --- | --- |
| **SIGGRAPH Asia** | **最优** | 2018/2022/2024 连着三届有 pixel art 论文，几乎隔届必有，2026 该轮到。血统和审稿口味完全对得上。 |
| **CVPR** | 可行 | 须打包成 resolution-conditioned 生成 + benchmark + metric 三件套，且要有可推广的 method claim。 |
| **ICLR** | 偏低 | 除非离散约束生成做成通用方法，并在 pixel art 之外的任务上验证。 |

## 风险

- **版权**：pixel art 数据多为有版权的游戏素材。近年顶会对数据集来源审查收紧。
  建议主数据集用 itch.io / OpenGameArt 的 CC0 素材（量少但干净），程序化生成做增广。
- **数据规模**：整个领域最大公开数据集才 42k，且质量参差。

## 调研可信度说明

TU Delft PDF 与 arXiv 全文本机取不到（网络限制）。
SD-πXL 与该本科论文的细节来自搜索摘要 + 项目页 + 代码库描述，**非原文精读**。
SD-πXL 的确切限制条件需读原文确认。
PixDiff-PIG 仅在 ResearchGate 找到，会议档次不明，可能是低层次 symposium。
