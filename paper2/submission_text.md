# OpenReview 提交用纯文本

> 摘要截止 2026-09-18，正文 2026-09-25。
> 由 `paper/main.tex` 机器抽取；改了 tex 就重跑 `python scripts/extract_submission_text.py`。

## Title

When There Is No Right Answer: Evaluation Failure and a Sampling-Scale Fix for Low-Resolution Pixel-Art Textures

## Abstract

Low-resolution pixel-art texture generation exposes a blind spot in how generative models are evaluated. When the output is a 16x16 grid of discrete colour cells, independent artists drawing the same material agree on 9.8% of cells against 8.6% expected under a paired null (50,753 artist pairs) — there is no unique right answer. We show this single fact breaks both mainstream automatic evaluation routes. Per-pixel metrics measure marginal fit rather than quality. And VLM judges fail in ways that neither standard judge diagnostic detects: on the same human-labelled set, two judges agree with the human on identical fractions of items (66.7%), yet one compresses an 87% effect to 62% (p=0.14) and erases its stratification while the other preserves both; a third inverts the human conclusion outright. Self-consistency is no better: the judge that erases the stratification is the most self-consistent of the three. Under human blind comparison a trivial downsampling baseline beats a purpose-built masked-prediction model 83:17, and is indistinguishable from artist-drawn tiles under both arbiters (43% by a human, 45% by the validated judge). The bottleneck is not model capacity but a structural-unit convention mismatch: the generator draws ~25 units per tile regardless of canvas while artists draw ~3.2, measured independently from 5,979 artist tiles and 125 artist sources and reproduced on a second generator. Cropping to the detected dominant period repairs this on the 45–58% of prompts where a period is found, and manipulating the unit count produces a monotone dose–response in the fix's benefit (88%, 54%, 17% at 27.7, 13.8 and 9.4 units). On the rest, where artists draw per-cell colour rather than structural units, we close four candidate routes and find none that works. We pre-registered and falsified a pure sampling-limit account along the way; reporting that failure is what selects the surviving mechanism.

---

标题 112 字符；摘要 298 词 / 1934 字符。
