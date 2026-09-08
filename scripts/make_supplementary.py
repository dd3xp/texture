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

STRUCT_OLD = """    # emnlp 上只有 shenhao_h3 自己的守护脚本，它只杀 h3_serve_* 会话与
    # 自身路径下的 sglang，匹配不到本项目；也读不到 OOM 记录（无权限）。
    # 被杀的任务多在 kw 上，那台当时不可达。所以下面是**对现象的应对**，"""

STRUCT_NEW = """    # 服务器上他人的守护脚本只杀其自身会话与路径下的进程，匹配不到
    # 本项目；也读不到 OOM 记录（无权限）。所以下面是**对现象的应对**，"""

# (相对路径, 旧串, 新串, 期望命中次数)
REPLACEMENTS: list[tuple[str, str, str, int]] = [
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
    # 中文项目 README（含陈旧研究状态）整个换成审稿人可读的英文复现指南。
    # 指南从导出副本取（即 HEAD 版本）——脚本必须在 commit 之后跑，
    # 与"提交后重打 supplementary"的既有流程一致。
    guide = dst / "paper" / "supplementary_README.md"
    assert guide.is_file(), "paper/supplementary_README.md 不在 HEAD 里（先 commit 再打包）"
    (dst / "README.md").write_text(guide.read_text(encoding="utf-8"),
                                   encoding="utf-8")
    guide.unlink()


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
