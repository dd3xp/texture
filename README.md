# texture

像素画纹理合成 (Pixel Art Texture Synthesis)。

## 任务

输入一张平涂纯色图（例如纯棕色区域）+ 目标材质语义（例如 `wood`）+ 目标像素预算 `N`，
输出在该区域上绘制了该材质纹理的像素画图像。

核心研究假设：**低分辨率下的纹理不是高分辨率纹理的降采样，而是艺术家的重新设计。**
16x16 的木头（两条深色横线 + 角上的钉子）和 128x128 的木头（连续年轮曲线）
是两个不同的 motif，不是同一 motif 的两个采样率。

因此任务定义为 **resolution-conditioned pixel art texture synthesis**，
而非传统的 scale-invariant texture synthesis。

## 硬约束

生成结果必须同时满足像素画的四条硬约束：

1. **固定小调色板**（通常 4~16 色）
2. **整数网格对齐**（cell 边界不能有非整数周期能量）
3. **零抗锯齿**（边缘是硬的）
4. **抖动承担明暗过渡**（dithering 是有结构的高频，不是噪声）

第 4 条是主要技术难点：扩散模型输出连续 RGB，事后量化会摧毁抖动结构，
因为任何去噪先验都会把抖动当噪声抹掉。

## 开发流程

本机（无 GPU）负责开发、调试、小规模验证；需要 GPU 时同步到 `emnlp`（8x A100 80GB）运行。

```bash
# 同步到 GPU 机器
bash scripts/sync_to_emnlp.sh

# 远端工作目录
ssh emnlp
cd /mnt/data/kw/RoundSquisheen/texture
```

注意：`emnlp` 的 `/mnt/data` 剩余空间紧张（约 300G），大规模数据集下载前先确认。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/texture/` | 主代码包 |
| `scripts/` | 同步、启动、数据处理脚本 |
| `experiments/` | 实验配置与产出（不入库） |
| `data/` | 数据集（不入库） |
| `docs/` | 文献调研、设计文档 |
| `notebooks/` | 探索性分析 |

## 当前状态

阶段：**文献调研完成，进入 baseline 验证。**

见 [docs/related-work.md](docs/related-work.md)。下一步三件事按优先级：

1. 跑 SD-πXL（SIGGRAPH Asia 2024），喂纯色图 + `"wood"`，确认它在哪失败。
   这是唯一能否决整个项目的实验。代码已在 `emnlp:/mnt/data/kw/RoundSquisheen/pixel/SD-piXL`。
2. 做 motivation figure：同一材质在 16/32/64/128 像素下真人画的版本，并排对比降采样结果。
3. 读 TU Delft 2025 本科论文全文，精确化其限制。
