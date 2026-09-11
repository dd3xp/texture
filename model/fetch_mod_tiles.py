"""扩训练数据：从 Luanti ContentDB 的**模组**（type=mod）里收真人画的方块贴图。

原数据集只抓了材质包（type=txp，105 个），而 ContentDB 上有 3000+ 个模组，每个模组都自带原创
方块贴图（`textures/*.png`），文件名即材质名，与材质包同一套命名。TRD 在 16px 上第 6000 步就开始过拟合，
训练集去污染后只有 ~3200 张，数据是瓶颈。

许可与排除（与 `analysis/premise/fetch_packs.py` / `docs/dataset.md` 同一口径）：
- 只收媒体许可匹配 PERMISSIVE 的（CC0 / CC-BY / CC-BY-SA / MIT / Apache / LGPL / GPL / Unlicense），NC、ND 一律不收。
- **Minecraft 系一律不碰**：名字、标题、简介里出现 minecraft / mineclon / voxelibre / mcl / faithful /
  pixel perfection / sphax 的模组整个跳过（这些多是 Minecraft 贴图的移植）。
- 与任何 val/test 材质包**同作者**的模组跳过（防画风泄漏）。
- 贴图筛选与原数据集相同：DENY 名单、方形 16/32、不透明（透明像素 ≤2%）、原生色 ≥3。
与 val/test 的结构去重在 `tiles_data.load(decontam=True)` 里统一做（和材质包补的瓦片同一条规则）。

本机跑（要联网；代理用环境变量 HTTPS_PROXY），只存筛过的小 PNG，不存 zip：
    python model/fetch_mod_tiles.py --out data/mods
然后 scp 到服务器，`build_extra_train.py --mods data/mods` 并进 train_extra.json。
"""
import argparse
import io
import json
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis" / "dataset"))
sys.path.insert(0, str(ROOT / "analysis" / "premise"))
from build_tiles import DENY           # noqa: E402
from fetch_packs import PERMISSIVE     # noqa: E402

API = "https://content.luanti.org/api/packages/"
DL = "https://content.luanti.org/packages/{author}/{name}/download/"
MC = re.compile(r"minecraft|mineclon|voxelibre|\bmcl|mcl_|faithful|pixel.?perfection|sphax", re.I)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "texture-research-dataset/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def held_authors():
    ds = json.loads((ROOT / "data/tiles/dataset_k16.json").read_text(encoding="utf-8"))
    return {s["pack"].split("__")[0].lower() for s in ds["samples"] if s["split"] != "train"}


def detail(p):
    try:
        return json.loads(get(f"{API}{p['author']}/{p['name']}/"))
    except Exception as e:
        return {"error": str(e)}


def harvest(d, out: Path):
    """下载一个模组的 zip，只留合格贴图。返回留下的文件数。"""
    dst = out / f"{d['author']}__{d['name']}"
    if (dst / ".done").exists():
        return len(list(dst.glob("*.png")))
    try:
        blob = get(DL.format(author=d["author"], name=d["name"]), timeout=180)
        z = zipfile.ZipFile(io.BytesIO(blob))
    except Exception as e:
        return f"download failed: {e}"
    kept = 0
    dst.mkdir(parents=True, exist_ok=True)
    for info in z.infolist():
        fn = info.filename
        if not fn.lower().endswith(".png") or "/textures/" not in "/" + fn.lower() or info.file_size > 64_000:
            continue
        name = fn.rsplit("/", 1)[-1]
        if DENY.search(name) or (dst / name).exists():
            continue
        try:
            im = Image.open(io.BytesIO(z.read(info)))
            w, h = im.size
            if w != h or w not in (16, 32):
                continue
            if im.mode in ("RGBA", "LA", "P") or "transparency" in im.info:
                if (np.asarray(im.convert("RGBA"))[..., 3] < 255).mean() > 0.02:
                    continue
            a = np.asarray(im.convert("RGB"))
        except Exception:
            continue
        if len(np.unique(a.reshape(-1, 3), axis=0)) < 3:
            continue
        Image.fromarray(a).save(dst / name)
        kept += 1
    (dst / ".done").write_text(str(kept))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/mods")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cat_path = a.out / "catalogue.json"
    if cat_path.exists():
        cat = json.loads(cat_path.read_text(encoding="utf-8"))
    else:
        lst = json.loads(get(API + "?type=mod"))
        print(f"模组 {len(lst)} 个，拉详情…", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            dets = list(ex.map(detail, lst))
        cat = [{"author": p["author"], "name": p["name"], "title": d.get("title", ""),
                "short": d.get("short_description", ""), "media_license": d.get("media_license") or "",
                "downloads": d.get("downloads", 0), "error": d.get("error")} for p, d in zip(lst, dets)]
        cat_path.write_text(json.dumps(cat, ensure_ascii=False, indent=0), encoding="utf-8")
    ha = held_authors()
    ok, why = [], {"licence": 0, "minecraft": 0, "held_author": 0, "error": 0}
    for d in cat:
        if d.get("error"):
            why["error"] += 1
        elif not PERMISSIVE.match(d["media_license"]):
            why["licence"] += 1
        elif MC.search(" ".join([d["name"], d["title"], d["short"]])):
            why["minecraft"] += 1
        elif d["author"].lower() in ha:
            why["held_author"] += 1
        else:
            ok.append(d)
    print(f"可用 {len(ok)} / {len(cat)}；排除 {why}", flush=True)
    res = {}

    def job(d):
        r = harvest(d, a.out)
        res[f"{d['author']}__{d['name']}"] = {"kept": r, "media_license": d["media_license"]}
        return r

    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(job, ok)):
            if i % 100 == 0:
                print(f"[{i}/{len(ok)}] 累计贴图 {sum(v['kept'] for v in res.values() if isinstance(v['kept'], int))}",
                      flush=True)
    (a.out / "manifest.json").write_text(json.dumps(res, ensure_ascii=False, indent=0), encoding="utf-8")
    n = sum(v["kept"] for v in res.values() if isinstance(v["kept"], int))
    print(f"完成：{sum(1 for v in res.values() if isinstance(v['kept'], int) and v['kept'])} 个模组贡献 {n} 张贴图")


if __name__ == "__main__":
    main()
