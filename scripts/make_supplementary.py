#!/usr/bin/env python3
"""生成匿名化 supplementary 代码包（ICLR 双盲，第 34 轮）。

流程：git HEAD 导出干净副本 → 删除内部日志/运维目录（docs/、scripts/）→
按显式清单脱敏服务器路径、主机名、用户名、环境名 → 全树扫描验证零泄露 →
打 zip 到 paper/supplementary.zip（产物不入库）。

原则（吸取 Bash 双反斜杠折叠与静默 sed 的教训）：
- 每条替换 assert 命中次数，多改、少改、没改都直接报错；
- 脱敏只发生在导出副本，仓库工作树一个字节不动；
- 最后用禁词全树扫描兜底，命中即非零退出。
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True).strip())
OUT = ROOT / "paper" / "supplementary.zip"

# 整目录移除：内部研究日志与服务器运维脚本（含身份信息，不属复现所需）
DROP = ["docs", "scripts"]

README_NOTE = (
    "> **Anonymized supplementary code.** Internal research logs (`docs/`) "
    "and server-ops scripts (`scripts/`) are omitted from this release; "
    "all model, analysis, and figure code plus the committed data snapshot "
    "are included.\n\n"
)

README_WORKFLOW_OLD = """本机（无 GPU）负责开发、调试、小规模验证；需要 GPU 时同步到 `emnlp`（8x A100 80GB）运行。

```bash
# 同步到 GPU 机器
bash scripts/sync_to_emnlp.sh

# 远端工作目录
ssh emnlp
cd /mnt/data/kw/RoundSquisheen/texture
```

注意：`emnlp` 的 `/mnt/data` 剩余空间紧张（约 300G），大规模数据集下载前先确认。"""

README_WORKFLOW_NEW = ("本机（无 GPU）负责开发、调试、小规模验证；"
                       "需要 GPU 时同步到 GPU 服务器（8×A100 80GB）运行。")

STRUCT_OLD = """    # emnlp 上只有 shenhao_h3 自己的守护脚本，它只杀 h3_serve_* 会话与
    # 自身路径下的 sglang，匹配不到本项目；也读不到 OOM 记录（无权限）。
    # 被杀的任务多在 kw 上，那台当时不可达。所以下面是**对现象的应对**，"""

STRUCT_NEW = """    # 服务器上他人的守护脚本只杀其自身会话与路径下的进程，匹配不到
    # 本项目；也读不到 OOM 记录（无权限）。所以下面是**对现象的应对**，"""

# (相对路径, 旧串, 新串, 期望命中次数)
REPLACEMENTS: list[tuple[str, str, str, int]] = [
    ("README.md", "# 有 GPU（emnlp）：从材质名一路做到成图",
     "# 有 GPU：从材质名一路做到成图", 1),
    ("README.md", README_WORKFLOW_OLD, README_WORKFLOW_NEW, 1),

    ("baselines/sdpixl/run_sweep.sh",
     'SDPIXL="/mnt/data/kw/RoundSquisheen/pixel/SD-piXL"',
     'SDPIXL="/path/to/SD-piXL"', 1),
    ("baselines/sdpixl/run_sweep.sh",
     'PROJ="/mnt/data/kw/RoundSquisheen/texture"',
     'PROJ="/path/to/this-repo"', 1),
    ("baselines/sdpixl/run_sweep.sh",
     'PY="/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python"',
     'PY="python"', 1),
    ("baselines/sdpixl/run_probe.sh",
     'SDPIXL="/mnt/data/kw/RoundSquisheen/pixel/SD-piXL"',
     'SDPIXL="/path/to/SD-piXL"', 1),
    ("baselines/sdpixl/run_probe.sh",
     'PROJ="/mnt/data/kw/RoundSquisheen/texture"',
     'PROJ="/path/to/this-repo"', 1),
    ("baselines/sdpixl/run_probe.sh",
     'PY="/mnt/data/kw/anaconda3/envs/SD-piXL/bin/python"',
     'PY="python"', 1),

    ("baselines/sdpixl/configs/probe_wood.yaml",
     'palette: "/mnt/data/kw/RoundSquisheen/texture/baselines/sdpixl/assets/wood8.hex"',
     'palette: "baselines/sdpixl/assets/wood8.hex"', 1),
    ("baselines/sdpixl/configs/probe_wood_cn09.yaml",
     'palette: "/mnt/data/kw/RoundSquisheen/texture/baselines/sdpixl/assets/wood8.hex"',
     'palette: "baselines/sdpixl/assets/wood8.hex"', 1),

    ("tools/paint_region.py",
     "（需要 GPU 与 diffusers，只在 emnlp 上跑）",
     "（需要 GPU 与 diffusers）", 1),
    ("model/generate.py",
     "# 待 kw 恢复后用画廊比", "# 待 GPU 机器恢复后用画廊比", 1),
    ("analysis/structure_grain/struct_metric.py", STRUCT_OLD, STRUCT_NEW, 1),
    ("analysis/exact.py",
     "服务器上有两个环境：`jzs_train` 有可用的 diffusers 但**没有 scipy**，",
     "服务器上有两个环境：一个有可用的 diffusers 但**没有 scipy**，", 1),
    ("analysis/paired/crop_res5_eval.py",
     "jzs_train 环境无 scipy：", "服务器环境无 scipy：", 1),
    ("analysis/paired/crop_fewunits_eval.py",
     "jzs_train 无 scipy：", "服务器环境无 scipy：", 1),
    ("analysis/paired/crop_scale_study.py",
     "（jzs_train 环境无 scipy）", "（服务器环境无 scipy）", 1),
    ("analysis/paired/fig_qualitative.py",
     "在 emnlp 上渲染", "在 GPU 服务器上渲染", 1),
    ("analysis/paired/fig_qualitative.py",
     "先在 emnlp 上跑", "先在 GPU 服务器上跑", 1),
    ("analysis/annotate/spotcheck_crop_key.py",
     "需在 emnlp GPU 上、", "需在 GPU 服务器上、", 1),
]

# 禁词：任何文本文件命中即失败（sk- 为 API key 前缀模式）
BANNED = [re.compile(p) for p in (
    r"/mnt/data", r"RoundSquisheen", r"emnlp", r"shenhao",
    r"jzs_train", r"anaconda3", r"113\.45\.39\.247",
    r"sk-[A-Za-z0-9]{20,}",
)]
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".pt", ".pth",
              ".npz", ".npy", ".zip", ".woff", ".woff2", ".ico"}


def export_head(dst: Path) -> None:
    tar_bytes = subprocess.check_output(["git", "archive", "HEAD"], cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(tar_bytes)) as tf:
        tf.extractall(dst, filter="data")


def sanitize(dst: Path) -> None:
    for d in DROP:
        shutil.rmtree(dst / d)
    for rel, old, new, n in REPLACEMENTS:
        f = dst / rel
        text = f.read_text(encoding="utf-8")
        hits = text.count(old)
        assert hits == n, f"{rel}: 期望 {n} 次命中，实得 {hits}：{old[:60]!r}"
        f.write_text(text.replace(old, new), encoding="utf-8")
    readme = dst / "README.md"
    readme.write_text(README_NOTE + readme.read_text(encoding="utf-8"),
                      encoding="utf-8")


def verify(dst: Path) -> list[str]:
    problems = []
    for f in sorted(dst.rglob("*")):
        if not f.is_file() or f.suffix.lower() in BINARY_EXT:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for pat in BANNED:
            for m in pat.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                problems.append(f"{f.relative_to(dst)}:{line}: {m.group()!r}")
    return problems


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="supp_"))
    try:
        root = tmp / "supplementary"
        root.mkdir()
        export_head(root)
        sanitize(root)
        problems = verify(root)
        if problems:
            print("泄露扫描命中，拒绝打包：")
            print("\n".join(problems[:40]))
            return 1
        OUT.unlink(missing_ok=True)
        n_files = 0
        with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(root.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(tmp))
                    n_files += 1
        print(f"OK: {OUT}（{n_files} 文件，"
              f"{OUT.stat().st_size / 1e6:.1f} MB），泄露扫描零命中")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
