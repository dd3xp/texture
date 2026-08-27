"""从 Luanti ContentDB 抓取许可干净的材质包，按分辨率分组。

设计意图见 docs/premise.md。要点：不追求"同一作者出多个分辨率"（查过，
那种包基本不存在，ContentDB 上的多分辨率标签是虚的），而是每个分辨率
各取多位作者的包，让"作者风格"从组间混淆变量变成组内方差。

只收许可明确可用的包。Minecraft 系材质包一律不碰——Faithful 的许可
明确禁止用于训练神经网络，其余（Sphax 等）更严。
"""

import argparse
import json
import re
import urllib.request
from pathlib import Path

API = "https://content.luanti.org/api/packages/"
DL = "https://content.luanti.org/packages/{author}/{name}/download/"

# 允许的媒体许可。NC（禁商用）和 ND（禁改）一律排除，
# 因为研究用途下游不确定，宁可保守。
PERMISSIVE = re.compile(
    r"^(CC0|CC-BY-\d|CC-BY-SA-\d|MIT|Apache-2\.0|Unlicense|LGPL|GPL-\d)", re.I
)
RES_TAG = re.compile(r"^(\d+)px$")

# 排除名单：这些包虽然许可干净，但内容不是真的材质。
# broken_textures 是故意做的"缺失材质"占位包（紫黑棋盘格），
# 混进来会污染每一种材质的统计。
EXCLUDE = {"broken_textures", "mystic_stones_progress_bar",
           "better_banner_shields_texture", "q3a_style_crosshairs"}


def get(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def catalogue() -> list[dict]:
    """拉全部材质包详情（列表接口不含许可和标签）。"""
    out = []
    for p in get(API + "?type=txp"):
        try:
            d = get(f"{API}{p['author']}/{p['name']}/")
        except Exception:
            continue
        ml = d.get("media_license") or ""
        if not PERMISSIVE.match(ml):
            continue
        if p["name"].lower() in EXCLUDE:
            continue
        res = sorted(int(t[:-2]) for t in d.get("tags", []) if RES_TAG.match(t))
        if not res:
            continue
        out.append({
            "author": p["author"], "name": p["name"], "title": d.get("title"),
            "media_license": ml, "res": res, "downloads": d.get("downloads", 0),
        })
    return out


def fetch(pack: dict, dest: Path) -> Path | None:
    z = dest / f"{pack['author']}__{pack['name']}.zip"
    if z.exists():
        return z
    url = DL.format(**pack)
    try:
        urllib.request.urlretrieve(url, z)
        return z
    except Exception as e:
        print(f"  ! {pack['name']}: {e}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/contentdb"))
    ap.add_argument("--low", type=int, default=16, help="低分辨率组的分辨率标签")
    ap.add_argument("--high", type=int, nargs="+", default=[128, 256],
                    help="高分辨率组，可给多个")
    ap.add_argument("--n-per-group", type=int, default=20)
    ap.add_argument("--all-res", action="store_true",
                    help="D1 用：按分辨率标签分组抓取全部许可干净的包，不截断")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cat = catalogue()
    print(f"许可可用且带分辨率标签的包: {len(cat)}")
    (args.out / "catalogue.json").write_text(json.dumps(cat, indent=1, ensure_ascii=False))

    if args.all_res:
        groups = {f"res{r}": [p for p in cat if r in p["res"]] for r in (16, 32, 64)}
    else:
        groups = {
            "low": [p for p in cat if args.low in p["res"] and max(p["res"]) <= args.low],
            "high": [p for p in cat if any(h in p["res"] for h in args.high)],
        }

    manifest = {}
    for g, packs in groups.items():
        packs = sorted(packs, key=lambda p: -p["downloads"])
        if not args.all_res:
            packs = packs[: args.n_per_group]
        d = args.out / g
        d.mkdir(exist_ok=True)
        print(f"\n== {g} 组: {len(packs)} 个 ==")
        got = []
        for p in packs:
            z = fetch(p, d)
            if z:
                print(f"  {p['title'][:34]:<34} {p['media_license']:<14} {z.stat().st_size//1024}KB")
                got.append({**p, "zip": str(z)})
        manifest[g] = got

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"\n清单写入 {args.out/'manifest.json'}")


if __name__ == "__main__":
    main()
