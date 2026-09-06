# 无头轮次的常驻指令

你是被 Windows 计划任务每 30 分钟拉起的**一个全新会话**，没有上一轮的记忆。
工作目录 `C:\Codes\texture`。干**一轮**有实质推进的活，然后提交推送、退出。

## 目标

ICLR 2027（摘要 2026-09-18，正文 2026-09-25）。
任务：**纯色图 + 材质 → 低分辨率像素画纹理**，分辨率 16 / 24 / 32。

## 每轮固定开头（不要跳过）

1. 读 `docs/loop.md`（循环协议、已排除路线、轮次记录）
2. 读 `docs/paper.md`（主张 ↔ 证据对照表，**空格子就是待办**）
3. `git log --oneline -8` 看最近做了什么
4. 检查远端任务：
   `ssh -o ConnectTimeout=25 emnlp "cd /mnt/data/kw/RoundSquisheen/texture && tmux ls; tail -3 experiments/crop_scale_study.txt"`

## 优先级

1. **远端任务掉了就重启**（tmux 会话消失但结果文件里没有完成标志）
2. **结果出来了就拉回本地、写进 `docs/`、提交推送**
3. **按 `docs/paper.md` 的待补清单做下一个实验**
4. 没有明确待办时：补 `docs/related-work.md` 的文献调研

## 硬性约束

- **GPU 只用 emnlp**（`ssh emnlp`），路径 `/mnt/data/kw/RoundSquisheen/texture`。
  **不要动 kw**（别人在用），不要动 `RoundSquisheen` 下的其他目录。
- 环境用 `/mnt/data/kw/anaconda3/envs/jzs_train/bin/python`（有可用的 diffusers）；
  纯分析可用 `SD-piXL` 环境。**不要往任何共享环境里装包。**
- 用 **PowerShell 工具**做 `git push`（Bash 工具连不上 Windows 凭据管理器）。
- 提交信息用英文，结尾加
  `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`。
- **不要重训模型**——A4 已证明那条路输给平凡基线 83:17。
- **不要把目视判断当结论**写进 `docs/paper.md` 的"✅ 已有"，标 ⚠ 并注明待盲比。
- API 密钥只从环境变量读，**不要写进任何文件**。

## 已排除的路线（不要重走，理由见 loop.md）

从零训练掩码预测模型 / 手工结构先验 / 配对监督 / 逐像素指标做判据 /
VLM 判官做结论（只能粗筛）/ 结构描述子尺子做质量代理。

## 收尾（每轮必做）

- 把这一轮做了什么追加到 `docs/loop.md` 的轮次记录
- `git add -A && git commit && git push`（用 PowerShell 工具 push）
- 若发现需要用户决策的事，写进 `docs/loop.md` 的「待用户」小节，不要空等
