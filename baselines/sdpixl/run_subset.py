"""B3：SD-piXL（Binninger & Sorkine-Hornung, SIGGRAPH Asia 2024）在固定子集上出图。

SD-piXL 是分数蒸馏优化，实测每张 16×16 约 3 小时 22 分（1 万步，1.2 s/步，19.5GB）。
全量 272 个材质要 ~900 GPU 小时，做不到，所以只跑 `eval/sdpixl_subset.json` 里
**固定种子（2026）预先选定**的 12 个材质。**用默认步数，不降**——降步数对它不公平。

- 纯文本模式（`configs/texture_text.yaml`：无输入图，关 ControlNet；SD-piXL 先用 SDXL
  出参考图初始化，再优化）。
- 提示词：与 B1 同一模板。
- 调色板：取该材质 B1 第 0 张 16px 瓦片的实际用色（≤12 色）——与 B1 同一套颜色，
  比的是"同样的颜色下谁画得好"。
- 结果：`final_argmax.png`（256×256）最近邻缩回 16×16，存 `experiments/baselines/B3/16/<slug>_0.png`。

路径通过环境变量给（本脚本不写死服务器路径）：
    SDPIXL_DIR=<SD-piXL 代码目录>  SDPIXL_PY=<其 conda 环境的 python>
用法：
    python baselines/sdpixl/run_subset.py --gpus 7        # 按 GPU 串行，已完成的自动跳过
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TMPL = ("pixel art, {p}, top-down seamless tileable game texture, "
        "flat lighting, no shadows, orthographic, chunky large pixels")


def palette_from_b1(slug, size=16):
    t = np.asarray(Image.open(ROOT / f"experiments/baselines/B1/{size}/{slug}_0.png").convert("RGB"))
    cols = np.unique(t.reshape(-1, 3), axis=0)
    # SD-piXL 的 load_hex 读 "rrggbb"（不带 #），与仓库原有的 assets/*.hex 同格式
    return ["%02x%02x%02x" % tuple(int(v) for v in c) for c in cols]


def run_one(item, gpu, size, work):
    slug = item["material"].rsplit(".", 1)[0]
    dst = ROOT / f"experiments/baselines/B3/{size}/{slug}_0.png"
    if dst.exists():
        return "skip"
    d = work / slug
    d.mkdir(parents=True, exist_ok=True)
    lock = d / ".lock"                    # 多个进程（不同 GPU）并行跑同一子集时，别重复做同一张
    if lock.exists():
        return "locked (another worker)"
    lock.write_text(str(gpu))
    pal = d / "palette.hex"
    pal.write_text("\n".join(palette_from_b1(slug, size)) + "\n")
    cfg = d / "config.yaml"
    shutil.copy(ROOT / "baselines/sdpixl/configs/texture_text.yaml", cfg)
    # PYTHONNOUSERSITE=1：共享账号的用户级 site-packages（~/.local）里有另一套 torch，
    # 会遮住 SD-piXL 环境自己的 torch 2.4 / torchvision 0.19，导致
    # "operator torchvision::nms does not exist"（2026-09-11 实测 12/12 秒挂）。
    # 只让本进程忽略它，不改动任何共享的东西。
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", DIFFUSERS_OFFLINE="1", PYTHONNOUSERSITE="1")
    cmd = [os.environ["SDPIXL_PY"], "main.py", "-c", str(cfg),
           "--prompt", TMPL.format(p=item["prompt"]), "--palette", str(pal),
           "--size", f"{size},{size}"]
    t0 = time.time()
    with open(d / "run.log", "w") as log:
        rc = subprocess.call(cmd, cwd=os.environ["SDPIXL_DIR"], env=env, stdout=log, stderr=log)
    outs = sorted(d.glob("**/final_argmax.png"), key=lambda p: p.stat().st_mtime)
    if rc != 0 or not outs:
        return f"FAILED rc={rc} (see {d / 'run.log'})"
    im = Image.open(outs[-1]).convert("RGB").resize((size, size), Image.NEAREST)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst)
    return f"ok {(time.time() - t0) / 3600:.2f} h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--work", type=Path, default=ROOT / "experiments/sdpixl_runs")
    ap.add_argument("--reverse", action="store_true", help="倒序处理（第二个进程从另一头开始）")
    a = ap.parse_args()
    for k in ("SDPIXL_DIR", "SDPIXL_PY"):
        if k not in os.environ:
            raise SystemExit(f"需要环境变量 {k}")
    items = json.loads((ROOT / "eval/sdpixl_subset.json").read_text(encoding="utf-8"))["items"]
    if a.reverse:
        items = items[::-1]
    # 按 GPU 轮转分配，每块 GPU 串行（一张图 19.5GB，别和人抢显存）
    from concurrent.futures import ThreadPoolExecutor
    queues = {g: items[i::len(a.gpus)] for i, g in enumerate(a.gpus)}

    def worker(g):
        for it in queues[g]:
            r = run_one(it, g, a.size, a.work)
            print(f"[gpu{g}] {it['prompt']:<28} {r}", flush=True)

    with ThreadPoolExecutor(len(a.gpus)) as ex:
        list(ex.map(worker, a.gpus))
    print("done")


if __name__ == "__main__":
    main()
