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

**必须实测回答。** 预判的三个失败点（**待验证**）：

1. **全图优化**，不保证纯色区域边界完整——而本任务中边界是神圣的（sprite 轮廓）。
2. **per-image 优化，慢**，非前馈，做不了交互工具。
3. score distillation 从 SD 蒸馏出的是"量化过的照片"，而非**像素画惯例**——
   真正的像素画有抖动阶梯、有意的色相偏移（暗部偏冷/偏紫）、锯齿边缘的固定处理法。
   这些是艺术传统，不在 SD 的先验里。

若这三点成立 → 它们就是论文的 motivation figure。**若不成立 → 项目需要重新设计。**

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
