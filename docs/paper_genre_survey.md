# 同类方法论文的体裁调研（写作结构 + 实验清单 + 我们的缺口）

用途：为"方法论文"定稿服务（定于 2026-09-18：**一篇、方法为主、评测发现作分析章节**）。
⚠ 本文件记的是**别人的做法**与**我们的缺口**，不产生任何本项目的判决。
⚠ **OpenReview 全站在本环境 403（challenge verification）** → 所有"审稿人原话/评分"一律 **UNVERIFIED**；
下文关于"审稿期待"的话是从论文结构与其自我防御性写法反推的**推断**，不是引语。
⚠ 五路调研里两路（掩码生成/检索增强/架构偏置族 + 可平铺先例）已回；像素画族与纹理材质族重跑中。

## 一、结构（实测，非猜测）

**硬规则**（已核 ICLR 2026 AuthorGuide / 2027 AuthorGuidelines）：投稿正文 **9 页**，rebuttal 与
camera-ready 放宽 10 页；参考文献与附录不计页；Ethics（≤1 页）与 Reproducibility 不计页、排正文末引用前；
**ICLR 2027 起 AI-use statement 强制**。

实测配比（正文一律顶满）：

| | Intro | Method | Experiments | Limitations |
|---|---|---|---|---|
| Meissonic（ICLR 2025, 10p） | ~2 | ~5 | ~2.5 | 0 |
| ImageRAG（ICLR 2026, 10p） | ~1.5 | ~2.5 | ~4（消融 1、人评 0.7） | ~0.8，**独立编号 §5** |
| MaskGIT / MAGE（CVPR, 8p） | ~1.5 | 2–3 | 4–4.5 | 附录 |

**本文建议配比（9 页）**：Intro 1.5 ／ Related Work 0.75（可整节移附录，Meissonic 即如此）／
Method 3 ／ Experiments 3（主表 .75 + 消融 1 + 判官 .75 + 性质检验 .5）／ Limitations 0.5 ／ Conclusion 0.25。

**可照抄的骨架 = ALiBi（ICLR 2022）**：**全文没有 Ablation 节**，消融被前置成 §2
"Current Approaches Do Not Extrapolate Efficiently" —— 方法登场之前先用一节把归因钉死。
对本文＝先证"现有做法（零/反射 padding、非环形相对偏置、事后拼接）在 wrap 方向做不到什么"，再出环形偏置。
次选骨架：Token-Critic §2 Background（两个 Challenges 子节把前作形式化并逐条点缺陷，Related Work 压到 §5）。

**消融位置两种成例**：iRPE 放 §4.2（SOTA **之前**，叙事＝"哪个因素真的重要"）；
Swin 放 §4.4（SOTA **之后**，当偏置只是系统一环时）。大体量消融外移附录是常态（MAGE 正文只留 Analysis）。
**Limitations 独立编号节、排 Conclusion 前**是 2026 ICLR 现行惯例（ImageRAG §5）。
**贡献 bullet 有无都能过审**（MaskGIT 全文 contribut 命中 0）。

## 二、指标换了代（选主判据直接相关）

- 2022–23：FID / IS / Precision / Recall（MaskGIT 另有 CAS、NLL）
- 2024 MaskBit：把 **gFID / rFID** 在符号层面分开，弃 Prec/Rec
- 2025 Meissonic：**明文弃用 FID 与 CLIP**（"limited relevance to visual aesthetics"），改报 HPSv2 / GenEval / MPS
- 六篇**无人**报 sFID、density/coverage

**速度报告的强制项 = 采样步数（6/6 全报，3 篇做成主表一列）**，其次参数量（5/6）；
wall-clock 仅 3 篇且单位各异（Muse 秒/批 + 硬件；Meissonic 秒 + 脚注声明排除 VAE 与 text encoder；
MaskGIT 报比值 "30–48×" + 一个绝对 anchor）。
**基线行数 4→26 行都能过审**，由"跟一个方法争"还是"跟一个榜争"决定。

**三条表格纪律**：① MaskGIT 用记号标死每个数字的来源等级（自己重训／取自原文／估算）；
② Muse 用记号隔离"在评测集上微调过"的对手；③ ⚑ **MaskBit 自建 "MaskGIT (w/ our VQGAN+)" 中间基线，
把 tokenizer 增益与生成器增益分开归因** —— 本文三部件（掩码解码 / 检索调色板 / 环形偏置）必须有等价物。

## 三、检索增强这一族的核心判据（我们直接踩在这条线上）

七种控制手段在 RDM / Re-Imagen / kNN-Diffusion / RetrieveGAN / ImageRAG 上的使用情况：

- **最低门槛：no-retrieval 臂必须同参数同算力、且进主表**（kNN-Diffusion 32.8 vs 12.5 进主表；
  RDM 用同参同算力的 inversion model）。⚑ 我们已有：16px 消融的 **AB_pal 126/190=66% p=8e-06**。
- ⚑⚑ **最有说服力的单表 = RDM 的 2×2 库交叉互换**：权重冻住只换检索库，性能随库内容变
  → 直接封住"是不是参数里记住了"。**我们零成本可做（只推理）。**
- k 扫描、库大小扫描：四篇里三篇做。我们已有 free-k / topk / n_name 等扫描的历史产物。
- **disjointness 要覆盖三层**：训练时 query 是否从自己的 kNN 剔出／评测集是否在库里／库与训练集是否重叠。
  我们有训练侧去污染（structural twins），**但"调色板库与评测集的关系"从未写清楚**。
- **检索开销是必答题**（RDM 报 0.95ms 取 20-NN@20M + 2GB/1M；kNN-Diffusion 报 FAISS 配置与 7s 推理）。
- ⚑⚑ **两个全族空档 = 低成本加分项**：**五篇没有一篇做 random-retrieval 对照臂**（喂随机邻居而非最近邻）；
  **五篇没有一篇做量化 copy/记忆检测**（生成图 vs 检索图的距离分布 + 真实图 null）。两者我们都能便宜做。

## 四、"这个架构部件就是它 work 的原因"——六种证据（强→弱）

量化背景：AblationBench（arXiv 2507.08038，从约 89,100 条 ICLR 2023–2025 审稿构建）：
**48% 的投稿至少收到一条含 "ablat" 的意见；平均每篇 2 条建议的消融；59% 修改式 / 41% 移除式。**

1. **参数量配平的 drop-in 互换**（不做会被打）。G-CNN 把滤波器数除以 √|G| "so as to keep the number of
   parameters approximately fixed"；ALiBi 同架构同超参**连随机种子都不换**；Swin 四档（无/绝对/绝对+相对/两者）。
   ⚑ 我们的环形偏置改的是偏置项、参数量几乎不变 —— **这一点要在正文写出来，是天然优势**。
   ⚠ 但按本项目 (M30)：**零初始化新模块 ≠ 控制组**（nn.Linear 构造抽 CPU 全局 RNG）→
   "其余一切固定包括种子"这句话**只有在标定过同配置重训漂移（(M31) Dmax=1.087）之后才敢写**。
2. **"by construction" 的性质检验，与质量指标分表分列**（本领域标准操作）：
   - Tiled Diffusion 的 **Tiling Score (TS)** ＝接缝处与固定偏移处像素绝对差均值的**最大值**，
     并设 ⚑ **"Swapped VT2I" 伪 ground-truth 当 TS 的噪声地板**（＝本项目 noise_floor.py 纪律的同构物）。
   - **TexTile（CVPR 2024，arXiv 2403.12961）＝可微的第三方可平铺性度量**；其摘要即"现有指标互不相关、
     严重阻碍该领域进展"，下游分类错误率 TexTile 0.064 vs FID 0.568 / GS 0.694 / MSID 0.797。
     它自己承认 human perceptual validation 不在范围内 → ⚑ **这是"性质尺子不测质量、
     质量交给判官"这一分工的正当性来源。**
   - 等变文献通行做法：**初始化时、用随机张量、逐层量相对等变误差** → 对本文＝
     **环绕接缝误差要在初始化时（随机 token）先测一遍**证明是结构给的，训练后再测一遍。
3. **同一互换在多规模/多数据集/多任务上重复**（ALiBi 3 语料×3 规模；Swin 3 任务；iRPE 分类+检测）。
   **单分辨率单数据集的单点提升是本族最容易被打的形态。** ⚑ 我们的替代轴＝
   **训练尺寸 → 更大画布平铺的外推**（ALiBi 式免费外推，不需要新判官臂，绕开 32px 准入条件②）。
4. **理论命题 + 该命题的实测验证**（RoPE 给长程衰减命题并实测衰减曲线）。
   对本文＝环面平移等变写成两行命题 + 数值误差。
5. **算力对齐的配对比较**（Token-Critic 因自己每步两次前向，**把基线步数加倍再比**，并在表里标出来）。
6. **拆分归因的中间基线**（MaskBit 那一行）。正交格（Muse 的 2×2、MaskBit Table 3a）
   **四格全报含最差那格**才是强形式；累加 roadmap 的代价是顺序依赖、单项不可归因。

⚠ 实操纪律（两篇明说）：**消融用更短训练配方可接受，但必须在题注交代**
（MaskBit "a shorter training schedule has been used for this study"；MAGE 消融 400 vs 主结果 1600 epoch）。
⚑ **消融要带种子噪声**：iRPE 报 "80.99 ± 0.16" + "trained and evaluated by three times"。

## 五、判官/人类偏好评测的呈现（2025–2026 录用论文）

⚑ **反直觉但重要**：在**类条件/离散生成**这条线上，人工盲比**不是准入要求**
（MaskGIT / MAGE / Token-Critic / MaskBit / MAR 全部零人评）；跨到**文本条件/美学**线才变主判据，
而此时同行对协议严谨度的**实际**要求很低（Meissonic 用一句话 + GPT-4o 判官就过了 ICLR 2025）。

可抄的约定：

- **Dynamic CFG（ICLR 2026）**：胜率 + **95% CI，显著者表中加下划线**；平局记 "one or none"；平均 2–3 人/题。
- **EditScore（ICLR 2026）**：⚑ **判官非确定性用 self-ensemble**：K 次独立前向聚合标量分，
  **Avg@4 0.763 vs 单次 0.703**；基准构建"每题两名标注者、只有排序完全一致的条目才进基准"。
  → **正好治本项目"57 题 14 翻面"。**
- **GenArena（2026-02 预印本）**：双向一致性（两个方向都选同一张才算偏好）、强制选择禁平局、
  跨方向冲突算法性判 Tie、Bradley-Terry MLE → Elo、报 **Krippendorff α（0.8628）** 与对人类榜的 Spearman 0.86；
  结论 **pairwise 普遍优于 pointwise（+11.4~+25.4pp）**。
- **Muse（ICML 2023）**：1650 prompt × 5 人、匿名 + 左右随机、含 indifferent 选项、≥3 人共识才计票，
  结果三分（70.6/25.4/4）。⛔ 无 CI、无 p 值、无 κ。
  ⚑⚑ **附录 A.5 明确拒绝跑"哪张更真实"并给理由**（mode collapse 会在该题上占便宜）
  → **"不跑会让自己占便宜的评测"的可引先例。**
- **Otani et al.（CVPR 2023）普查 37 篇**：只有 20 篇做人评、只有 4 篇披露每样本评分次数、**无一篇报 IAA**；
  建议 3 名标注者/图、4 个模型要得出结论性排名需 >100 条 prompt。
- 预印本《How to Correctly Report LLM-as-a-Judge Evaluations》(arXiv 2511.21140)：报比例 CI、
  已知判官错误率时做 **Rogan-Gladen 去偏**、披露混淆矩阵而非只报准确率、明说平局处理、报反序结果。

⚑⚑ **结论：本项目的判官协议（两序去偏 + 弃不一致、空对照定地板 17.8%、免门 p_vs_floor、
族级 CI、W1_robust 分级、试点门误杀率实测）比这个体裁里任何一篇都严。** 这不是义务，
是**可以主动写出来卖的方法论卖点**；呈现时套已录用的约定（胜率 + 95% 族级 CI + 下划线标显著、
双向一致性、Avg@K、Krippendorff α 与对 CLIP/人的相关系数）。

## 六、负结果/零结果的成文惯例（九种，可直接借句式）

制度层面：NeurIPS Checklist 明写 "authors should be rewarded rather than punished for being up front
about the limitations"；NeurIPS 2026 把 Negative Results 列为正式贡献类型；ICML 2024 有立场论文
*Position: Embracing Negative Results in ML*（arXiv 2406.03980）。⛔ 但**没有** "What didnt work"
附录节的成型惯例。

1. **一句话、不给数字、转成设计优势**（Token-Critic §3.3："we find this does not yield a better result.
   In fact, by ignoring the previous mask, Token-Critic has the ability to correct previously sampled
   tokens…"）—— 注意负结果放在 **Method 节**而非实验节。
2. ⚑ **明说主判据这把尺子在此处钝，并同时补两把尺子**（MAGE 的 FID/裁剪解释；
   ImageRAG "these metrics gauge only coarse semantic similarity… we additionally report GPTScores…
   user studies"）→ **本文"自动指标提升不大"就用这个模板。**
3. **两把尺子并排印出反向结果**（MaskGIT FID/IS vs NLL；MAGE FID vs linear-probe；
   Muse 重建 FID vs 生成 FID 排序不同；MaskBit 明说加中间特征损失改善重建但**损害生成**）
   → 本项目 **FD 与判官反向**正是这一类，**是加分项不是弱点**。
4. **"更多不一定更好"的非单调结论**（MaskGIT 迭代步数有 sweet spot；Muse 24 步 8.03 vs 256 步 8.58；
   MaskBit 分组数与 bit 数都是 U 形）。
5. **主动去测边界，证明技巧不普遍适用**（MaskBit 把 2 组套到 VQGAN+ 上 gFID 2.12→4.4 并解释机制）。
6. **报"落选设计"的具体数字**（kNN-Diffusion 附录三种条件化 18.3/22.4/34.1；ALiBi 可训练斜率
   "slowed down the training speed by 3%" 且 "did not yield strong extrapolation results"、乘法式 "degraded performance"）。
7. **用 N/A 表示"任务直接崩掉"，据此得必要性论断**（MAGE σ=0 时 FID>50 记 N/A →
   "a variable masking ratio is necessary"；MAR 直接 L2 "leads to a disastrous FID score (>>100)" + 机制解释）。
8. **任务相关的部分失效照报**（Swin 绝对位置 +0.4% 分类但 −0.2 AP / −0.6 mIoU；
   iRPE "negligible gap in classification. However, in object detection…"）
   → ⚑ **本项目"32px 不缺调色板部件""某部件只在某档有效"完全适用 iRPE 这个句式。**
9. **自己给机制故事留口子**（最高级）：ALiBi 附录 B 用 S=1 滑窗**自证"可能并没有在用更长上下文"**，
   把主张缩到能承受的那层；RoPE §4.5.5 "there lacks of thorough explanations on why it converges faster"；
   MaskBit Limitations 直接质疑 FID 效度并说明为何仍用它。
   → ⚑⚑ **本项目「环不闭合」与 (M24) 缺机制，按这个先例写成附录自我反驳 + future work，
   ⛔ 不是硬凑第四个机制。**

⚑ **Limitations 最佳模板 = MaskBit 附录 H 三条**：①未探索的组合（给算力理由）；
②**只在一个数据集/分辨率上训 → 因此我不做某一类比较**（把范围限制转成自律而非道歉）；
③**主判据本身的效度质疑 + 为何仍然用它**。加 ImageRAG §5 的三个加粗小标题式结构。

## 七、与本文最贴的四个先例（必须引、必须定位）

- **TileGen**（Zhou et al., SIGGRAPH Asia 2022）：StyleGAN 变体，**靠 circular padding 保证输出永远可平铺**
  ＝"tileability by construction"的经典先例 → ⚠ **审稿人一定问"为何不直接用 circular padding"，必须有对照。**
  ⚑ 我们**已经有了**：B7（同数据像素扩散 UNet）**本来就用 circular padding**，只需在正文把这层定位写出来。
  （TileGen 详细结构 UNVERIFIED：ACM DL 403。）
- **Tiled Diffusion**（Madar & Fried, CVPR 2025，arXiv 2412.15185）：结构 Intro→RW→Method→§4 Evaluation
  （4.2 内嵌消融）→§5 Applications→**§6 Limitations**→Conclusion；自定义 TS；基线 5 行含伪 GT 噪声地板；
  消融两条。**无人工评测、无速度报告。**
- **Local Padding in Patch-Based GANs**（arXiv 2309.02340）：**改动就在 padding 层**，与本文同型
  （摘要逐字："the implementation of local padding in the state-of-the-art super-resolution models
  effectively eliminates tiling artifacts"）。章节/指标 UNVERIFIED。
- **TexTile**（CVPR 2024）：见 §四.2；是可引的第三方可平铺性尺子。
- ⚑ **环形/周期相对注意力偏置本身未找到已发表先例**（最近者＝arXiv 2606.22945，2026-06 预印本，
  Coordinate-Transformed RoPE 做平铺控制，无场地）。**这是新颖性的正面信号，但检索非穷尽**，
  ⛔ 不许写成"首个"，只能写"据我们所知未见"。

## 七之二、纹理/材质族 18 篇的实测统计（第二路调研，2026-09-18）

⚠ **先纠三处会影响引文的错误**：① arXiv 2308.11408 是 **MatFuse**（CVPR 2024），**不是** MatFusion；
真 MatFusion＝Sartor & Peers, SIGGRAPH Asia 2023，arXiv **2406.06539**，DOI 10.1145/3610548.3618194。
② **Infinite Texture（arXiv 2405.08210）无被接收证据 → 按 preprint 引**（作者是 UW/Google 系，不是 Adobe）。
③ **Local Padding（arXiv 2309.02340）是 preprint-only**（机构库登记为 Working paper › Preprint），无 venue 无 DOI。

**出现率（分母＝实抽 18 篇）**：定性图阵 **18/18**；具名基线 ≥3 个 18/18；Limitations 17/18；
至少一个自动指标 16/18（TileGen 与 TEXTure **零指标零表格**照样中 SIGGRAPH 线）；消融 16/18；runtime 14/18；
**人评 9/18**（子领域分裂：mesh 文本纹理化 5/5 必做、SVBRDF/材质 0/4、可平铺线 3/7）；
分布型指标（FID/KID/SIFID）10/18；**可平铺的量化指标只有 5/18，且全在 2024 之后**；
**为自造指标画地板/对照臂 2/18**。

⚑⚑ **四个"全族零"**（这决定了我们方法学章节的定位）：
**置信区间 / 显著性检验 / p 值 0/18；评分者一致性统计 0/18；每对双序 + 用时门 + 需真比较的注意力检查 +
剔除不一致 0/18（去偏一律只到"随机顺序"这一层）；同配置重训漂移或采样噪声下限标定 0/18。**
→ 本项目的纪律**严于惯例**，是卖点；但**没有别人的做法可引来替我们的口径背书**，方法学必须自证。
⚠ 反过来也别被"文献都不做地板"诱导去放弃地板。

**⚠ FID/KID 在这个体裁已事实分叉、跨论文不可直接比**：物体级固化为 "FID + KID×10⁻³ @ 20 固定视角 @512²"
（Text2Tex / Paint3D）；TexFusion 自建 SD2-depth 参考分布只报 FID；SceneTex 改报 CLIP score + IS 不报 FID/KID；
Tiled Diffusion 以 VT2I 为参照算 FID；MatSynth 只在 renderings 上算。
→ **写"我们 FID 比 X 好"必须同引参考分布与渲染/视角口径。**

### "tileability by construction" 的三条路线与证据门槛

| 路线 | 代表 | 证据形态 | 有量化平铺指标吗 |
|---|---|---|---|
| **A 架构环绕**（circular/toroidal） | TileGen（奠基）、FabricDiffusion、Tiled Diffusion 的 AT 基线 | TileGen＝机制断言 + 平铺渲染图 + **"用非平铺数据训练仍出无缝"的准对照**（把"数据是否平铺"这个混淆因子挑开）；**零证明零指标** | TileGen 无；**FabricDiffusion 唯一量化** |
| **B 推理侧不变性**（noise rolling） | ControlMat、Structured Pattern Expansion、Tiled Diffusion | ControlMat 纯视觉；另两篇带指标 | **唯一普遍带指标的路线** |
| **C 边界条件构造**（受约束 inpainting） | Content-aware Tile Generation（TOG 2024） | **唯一带形式化附录**（Wang/Dual Wang packing 论证） | 借用 TexTile |

⚑⚑ **三条对本文最要紧的事实**：
1. **2024 年后门槛抬升**：TileGen 2022 年零指标就过了 SIGGRAPH Asia；TexTile 之后新论文基本都报一个平铺数。
   → 投 ICLR 按 2024 后标准：**机制论证 + 至少一个平铺量化 + 消融**。
2. **"by construction" 已被文献自己限定**：Tiled Diffusion 把 circular padding 定性为
   "inherently tileable **but restricted to self-tiling scenarios with no rotations**"，
   并在 Limitations 里**自认 latent 旋转 ≠ 像素旋转、所以它的构造性"必然"不成立**。
   → ⚠ **我们的平铺性若依赖离散化/量化步骤（低分辨率尤其容易），那一步必须在正文显式交代**，
   ⛔ 不许靠 "by construction" 一笔带过。
3. ⚑⚑ **没有任何一篇做"去掉构造性机制就失败"的完整反证**：Content-aware 无"去掉边界条件"消融、无故意错配边界的反例；
   **FabricDiffusion 的 TexTile 0.47（去掉环绕 padding）→ 0.62（加上）是全领域唯一一条剂量-应答**。
   → **这是一个便宜且无人占领的位置：给环形偏置做"关掉它"的消融 + "故意破坏它"的对照臂。**

### 两种可直接抄的"给自己的指标定标"做法（全族只有这 2 例）

- **Tiled Diffusion 的双端锚点**：坏端＝普通 T2I（TS≈0.29，定义上不平铺）；
  好端＝**Swapped VT2I**（把左右/上下两半互换 → 接缝处必然完美，TS≈0.03）。
  ⚠ 只有点估计、**无离散度无 CI 无检验**，"significantly" 无统计量；且 Swapped 的 CLIP 掉到 0.14
  → 该锚点**只能校准尺度、不能当质量参照**。
  ⚠ 其 TS 定义式为 `TS_pair = (1/h)Σ_y |I₁(x₁,y) − I₂(x₂,y)|`，"三次测量"的聚合方式**论文自相矛盾**
  （定义句写 maximum、后文写 average，两句都实际存在）→ **引用时必须点出这处不一致，别替它选一个**。
- ⚑ **Structured Pattern Expansion 的"参照带"**：把**已知为真可平铺的语料**算出分布当参照带
  （其方法 TexTile **63.93% ± 8.24%** vs 数据集内真平铺 **62.25% ± 14.04%**），论证"我落在带内"
  而不是"我比对手高"。⚠ 参照带自己的 σ 比它方法的还大；仍无 CI 无检验。
  → **我们有真人瓦片，这个形态零成本可做，且比"比对手高"更稳。**

### TexTile 引用时的两条硬约束

- 定义：先 **2×2 平铺**（`I_tiled ← tile(I,(2,2))`，论文称这点重复次数足够），再
  `TexTile = 1/(1+exp(−λ·M(I_tiled)))`，**λ=0.25**；训练 384²、推理 512²；骨干＝预训练 ConvNext + 两个线性自注意力残差块。
- ⚠ **它给的是分类性能（err 0.064 / Acc 0.982 / F1 0.983 / AUC 0.997），不是尺度地板**；
  且**它从未与人类感知做相关性验证**（原文自认 human perceptual validation 不在范围内）
  → ⛔ **不能写成"TexTile 已被人类验证"**。它自己也承认失败案例与"人类对重复性的感知没有牢固理解"。
- 它当损失用时的增益：Heitz 0.431→0.781、SinFusion 0.388→0.798；训练成本 6 小时 / 单张 RTX 3060。

### 颜色条件的度量惯例（我们"配色跟随区域"直接对口）

**Color Alignment in Diffusion（CVPR 2025，arXiv 2503.06746）** 自造两个指标，**是双向 Chamfer 距离**：
`CD-A = Σ_p min_q ‖x̂₀[p] − c[q]‖²₂ / |x̂₀|`（accuracy：生成像素贴合给定色值的程度）、
`CD-C = Σ_q min_p ‖x̂₀[p] − c[q]‖²₂ / |c|`（completeness：指定颜色被覆盖的比例）。
⚠ **这个领域没有用 ΔE / CIEDE2000 的惯例**，而我们的区域上色任务用的是 **ΔE76**
→ **建议两套都报**（ΔE76 保留感知口径，CD-A/CD-C 换取可比性），零 GPU。
它的 runtime 报法也可抄：**以 vanilla diffusion = 1.00× 为基准的相对倍数表**（回避"你卡更好"的质疑）。

### 速度表：两种最完整的报法（建议抄）

- **ControlMat 式：分辨率 × 时间 × 显存三列**（3 s@512² / 18 s@1024²(12GB) / 43 s@2048²(18GB) / 350 s@4096²(20GB)，A10G 24GB）。
- **Content-aware 式：拆到原子算子**（"40 ms per U-Net evaluation × 40 steps ⇒ **1.7 s per tile
  (including VAE encoding/decoding)**"，再给端到端 81 瓦片 140 s、带 4 候选拒绝 12 min）。
  ⚑ 它明确交代"含 VAE 编解码"与"候选重采样的额外开销"——**这两项最常被人偷偷漏掉，而我们用 4 选 1 重排，必须照实计入**。
- **必须一起声明**：采样步数与 sampler、分辨率、是否含 VAE 编解码、**是否含候选重排/拒绝重试**、显存峰值。
- 训练成本惯例＝GPU 数×型号×小时 + iterations/epochs + batch。**全族无人报能耗/金钱/吞吐/多次运行方差**
  → 报了就是加分且无人可比。
- ⚠ 坏范例（可反向引用）：Paint3D 与 **Tiled Diffusion 一个 runtime 数字都不报**；
  TEXTure 表里 5/32/46 min 与正文 "19 through 45 minutes" **自相矛盾**。

### ⚑⚑ 单分辨率评测：**18 篇里 0 篇为它给出正面论证**

只有三种姿态：①**写进 Limitations 自认局限**（MatFusion 逐字 "MatFusion is currently limited to
256×256 resolution SVBRDFs."；TileGen 把 512² 写进 Limitations）；②**沉默**（最多数，且没人在评审里被逼着解释）；
③**把"分辨率无关"翻成贡献从而绕过辩护**（SD-πXL 贡献第一条即 "works at any resolution" 并展示 24²–192²；
ControlMat / Structured Pattern 把多档时间显存全给出来 → 多分辨率本身成了卖点）。
⚑ 另一条可抄的实操防御＝**Text2Tex 的做法：用两个数据集，并让每个数据集配它自己领域原生的基线**
（把"换数据集就得换对手"做实，使每张表内部自洽）。
→ **对本文的判断：想找"别人怎么为单分辨率开脱"——没有先例可引。只有两条合法出路：
(i) 写进 Limitations（MatFusion 措辞可直接改写照用）；(ii) 把跨分辨率能力转成主张并把三档都报出来。
而我们"三档从来不在同一把尺子上"这个已知问题，文献里既无先例也无掩护** ——
选 (ii) 必须先把尺子统一，选 (i) 则要接受把主张范围缩到 16px。**当前决定＝(i)。**

### 其他可直接借用的做法

- ⚑ **MatFusion 的 "Adversarial (retrained)" 行**：**用自己的训练集重训对手**，把"数据优势"这个混淆因子挑掉。
  ⚑ **我们已经有了**：B7 就是同数据从零训练的像素扩散 UNet —— **正文要把这层写成刻意的设计，不是顺手**。
- ⚑ **Infinite Texture 把消融放在基线比较之前**（先证部件必要、再证赢）；
  ⚑ **SceneTex 让消融行也带人评分**（消融与基线同表）＝把"部件承重"讲得最硬的排版。
- **ControlMat 把"大图 + 可平铺"抬成独立一级章节（§5）** → 若主打可平铺，这是有先例的强动作。
- **MatFusion 的 Limitations 是全批最诚实、最值得模仿的一段**（逐字）："it has trouble generating
  pixel-perfect reproductions. **Hence, MatFusion does not necessary produce the lowest errors on
  pixel-based metrics.**" → ⚑ 本项目"FD 输 B7、但判官赢"完全可以照这个句式写。
- **SD-πXL 的成本与协议**（我们最近的参照）：**1.5 小时 / RTX4090 / 6000 步**，自列为 limitation；
  定量用 150 张；感知研究 56 人、**排名式**、三轴（semantic accuracy / fidelity / aesthetic appeal），
  **无检验、无一致性、无注意力检查**；⚠ 三个轴**没有公式级定义**（这是它的弱点，不可抄）。
  它对低分辨率的姿态＝把"任意分辨率"写成贡献，并用"优化式、无需数据集"绕开数据依赖。

## 八、对照本项目：缺口清单（按性价比排序）

| # | 要做的 | 为什么（对应上文） | 成本 |
|---|---|---|---|
| 1 | **判官臂：TRD vs B5（检索基线）@16px** + **区域上色任务 vs B5** | B5 在 16px 分布指标上完胜（KID 2.69 / FID 43.0 / FD 45.3，因为它返回真人原图），而 44 条臂里**没有一条对 B5** → 评审必问。它 CLIP 最低（32.95）且**做不到配色跟随区域**＝我们最强论点却从未量过 | 仅 API，图已在盘 |
| 2 | **调色板库 2×2 交叉互换**（冻权重只换库内容） | §三：本族最有说服力的单表（RDM） | 仅推理 |
| 3 | **random-retrieval 对照臂**（喂随机调色板而非最近邻） | §三：**全族空档**，低成本加分 | 仅推理 |
| 4 | **copy/记忆量化检测**（生成 vs 检索 vs 真实 null 的距离分布） | §三：**全族空档**；同时回答"是不是把库里的图抄出来" | 零 GPU 为主 |
| 5 | **速度表**：采样步数 + 参数量 + wall-clock（标注硬件、是否含 text encoder） | §二：步数是 6/6 强制项；我们对 SD-piXL 有 3.45–4.44 GPU 小时实测 | 半天 |
| 6 | **性质检验分表**：环绕接缝误差（**初始化时随机 token 先测一遍** + 训练后）＋ TS 式指标 + 伪 GT 噪声地板 | §四.2：本领域标准操作；我们已有 tile_seam_ratio 但**没有噪声地板臂、没有初始化时的测量** | 零 GPU / 少量 |
| 7 | **TexTile 第三方可平铺性分数**（我们 + 全部基线） | §四.2：可引的第三方尺子，且把"性质 vs 质量"的分工写正当 | 需跑其预训练模型 |
| 8 | **偏置 × 环形增强的 2×2**（有/无随机 roll 增强 × 有/无环形偏置） | §四.3 + G-CNN 的 CIFAR10/CIFAR10+：回答"随机 roll 增强不就够了"的唯一答案，**零判官臂** | 2 次训练 |
| 9 | **外推轴**：训练尺寸 → 更大画布平铺 | §四.3：替代"多分辨率"轴，**绕开 32px 准入条件②** | 仅推理 |
| 10 | **架构瘦身**（砍掉消融里无效的三个部件后重验 16px） | §四.6 中间基线纪律；把"负面消融"从弱点变卖点 | 1–2 次训练 |
| 11 | **消融带 ±std**（同配方重复 3 次） | §四 的 iRPE 纪律；我们已有 (M31) Dmax=1.087 可直接引 | 已部分具备 |
| 12 | **disjointness 三层写清**（调色板库 vs 评测集 vs 训练集） | §三：必答项，我们只做了训练侧去污染 | 零成本（查 + 写） |

第二路调研（纹理/材质族）追加的六条：

| # | 要做的 | 为什么 | 成本 |
|---|---|---|---|
| 13 | **"关掉环形偏置" + "故意破坏它"两条对照臂** | §七之二：**全族没有一篇做过构造性机制的反证**，FabricDiffusion 的 0.47→0.62 是唯一剂量-应答 → 便宜且无人占领的位置 | 1 次训练 + 推理 |
| 14 | **接缝指标的双端锚点**：坏端＝非平铺生成、好端＝**两半互换**（接缝处定义上完美） | §七之二：全族只有 2 例给自己的指标定标，这是其中一例的形态 | 零 GPU |
| 15 | **"真人参照带"**：在已知可平铺的真人瓦片上算同一指标，论证"我落在带内" | §七之二：比"我比对手高"更稳；我们有真人瓦片 → 零成本 | 零 GPU |
| 16 | **CD-A / CD-C 双向 Chamfer 颜色指标**（与现有 ΔE76 并报） | §七之二：该领域**没有用 ΔE 的惯例**，CVPR 2025 那篇的 Chamfer 口径才是可比的 | 零 GPU |
| 17 | **速度表照 ControlMat + Content-aware 两种报法**：分辨率×时间×显存，并拆到原子算子，**显式计入 4 选 1 重排的开销** | §七之二：候选重采样开销是最常被偷偷漏掉的两项之一，而我们正用 4 选 1 | 与第 5 条合并 |
| 18 | **把 B7 写成"用我们数据重训的对手"**（MatFusion 的 retrained 行） | §七之二：这是挑掉"数据优势"混淆因子的最佳实践，我们已具备却没这么写 | 零成本（写） |

⛔ 上表是**候选清单，不是授权**：每条真要跑仍须按本项目规矩单独预注册（判据先于数据）。
⚠ 第 1 条涉及 16px 判官新臂（**不**受 32px 准入条件②限制）；第 9 条特意设计成不需要新判官臂。
