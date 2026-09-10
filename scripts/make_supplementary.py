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

import fnmatch
import os
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

# 整目录移除：内部研究日志（含服务器状态与身份信息，不属复现所需）
DROP = ["docs"]

# scripts/ 是运维目录（cron、同步、打包自身），整体不发。例外是复现指南
# 明确指给审稿人的那几个——指南提到却不在包里，审稿人按图索骥必然落空。
SCRIPTS_KEEP = {"batch_pack.py", "fetch_sd15.sh"}

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
    kept = set()
    for f in sorted((dst / "scripts").iterdir()):
        if f.is_file() and f.name in SCRIPTS_KEEP:
            kept.add(f.name)
        elif f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()
    assert kept == SCRIPTS_KEEP, f"scripts/ 保留清单对不上：实得 {sorted(kept)}"
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


# 复现指南里写的每一条路径，都必须在包内真的存在。此前无人检查这件事：
# 指南是照着**仓库**写的，而包是仓库删掉 docs/ 与 scripts/ 之后的样子，
# 两边一漂移，审稿人就会照着指南去找一个不存在的文件。
GUIDE_PATH = re.compile(
    r"[A-Za-z0-9_][A-Za-z0-9_/.*-]*\.(?:py|sh|json|csv|html|md|tex|png|bib)\b")


def check_guide_paths(dst: Path) -> list[str]:
    text = (dst / "README.md").read_text(encoding="utf-8")
    files = {f.relative_to(dst).as_posix()
             for f in dst.rglob("*") if f.is_file()}
    names = {f.rsplit("/", 1)[-1] for f in files}
    problems = []
    tokens = sorted(set(GUIDE_PATH.findall(text)))
    assert tokens, "指南里一条路径都没抽到，正则坏了"
    for tok in tokens:
        # 带 / 的按包内相对路径解析，裸文件名按 basename 解析（指南两种都用）。
        pool = files if "/" in tok else names
        ok = (any(fnmatch.fnmatch(p, tok) for p in pool) if "*" in tok
              else tok in pool)
        if not ok:
            problems.append(f"README.md 指向包内不存在的路径：{tok}")
    return problems


# 指南把下面这些命令当作"审稿人可以自己跑"的自检推出去，它们各自有明确的
# 通过/不通过契约（非零退出即不通过）。上面那个 check_guide_paths 只验了
# **路径存在**，没人验过**跑起来过不过**——于是包里可以带着一个自己跑不过的
# 自检出门，等于把反证材料一并交上去。这里在**包内副本**上真跑一遍。
# 只列不依赖 GPU / 网络 / 未入库数据的那几个。
SELFCHECKS = ["analysis/paired/recheck_gpu_claims.py"]


def run_selfchecks(dst: Path) -> list[str]:
    guide = (dst / "README.md").read_text(encoding="utf-8")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    problems = []
    for rel in SELFCHECKS:
        assert rel in guide, f"{rel} 不在复现指南里，自检清单已过期"
        r = subprocess.run([sys.executable, rel], cwd=dst,
                           capture_output=True, env=env)
        if r.returncode:
            out = (r.stdout + r.stderr).decode("utf-8", errors="replace")
            problems.append(f"{rel} 退出码 {r.returncode}：\n{out.strip()}")
    return problems


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
        dangling = check_guide_paths(root)
        if dangling:
            print("复现指南有断链，拒绝打包：")
            print("\n".join(dangling))
            return 1
        failing = run_selfchecks(root)
        if failing:
            print("包内自检自己跑不过，拒绝打包"
                  "（先把正文/脚本对不上的那处修好，别把反证一起交出去）：")
            print("\n\n".join(failing))
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
