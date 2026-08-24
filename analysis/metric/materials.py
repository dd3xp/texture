"""测试集用的材质清单，两个脚本共用。

从 pairs.json 里取覆盖度够（>=5 个高分辨率源）且是真正可平铺材质面的条目。
排除植物、工具、图标——它们不是"材质"，放进来会污染 CLIP 的零样本分类。
"""

import json
import re
from pathlib import Path

# 允许的材质词根。只保留表面材质，不要 sapling/leaves/tool/sign 这类
ROOTS = [
    "wood", "stone", "cobble", "brick", "sand", "gravel", "tree", "dirt",
    "grass", "snow", "ice", "clay", "obsidian", "desert", "junglewood",
    "pine_wood", "acacia_wood", "aspen_wood", "stone_brick", "mossycobble",
    "sandstone", "coral", "silver_sand", "permafrost", "moss", "steelblock",
    "copperblock", "bronzeblock", "tinblock", "goldblock", "diamondblock",
    "meselamp", "glass", "bookshelf", "coalblock", "ironblock",
]
DENY = re.compile(r"sapling|leaves|tool|sign|ladder|torch|item|_top$|seed|"
                  r"bush|shrub|fern|papyrus|grass_\d|dry_grass_\d|"
                  r"lump|ingot|slot|_side$|door|rail|chest")


def prompt_for(name: str) -> str:
    """从文件名派生 CLIP / SDXL 用的自然说法。"""
    b = name.replace("default_", "").replace(".png", "")
    special = {
        "cobble": "cobblestone", "mossycobble": "mossy cobblestone",
        "tree": "tree bark", "obsidian": "obsidian rock",
        "dirt": "dirt soil", "wood": "wood planks",
        "stone_brick": "stone brick wall", "brick": "brick wall",
        "desert_stone_brick": "desert stone brick wall",
        "desert_cobble": "desert cobblestone",
        "junglewood": "jungle wood planks",
        "pine_wood": "pine wood planks",
        "acacia_wood": "acacia wood planks",
        "aspen_wood": "aspen wood planks",
        "silver_sand": "silver sand",
    }
    return special.get(b, b.replace("_", " "))


def load(pairs_path: Path = Path("data/contentdb/pairs.json"),
         min_high: int = 5) -> dict[str, str]:
    pairs = json.loads(Path(pairs_path).read_text())
    out = {}
    for k, v in pairs.items():
        base = k.replace("default_", "").replace(".png", "")
        if DENY.search(k):
            continue
        if not any(base == r or base.startswith(r) for r in ROOTS):
            continue
        if len(v["high"]) < min_high:
            continue
        out[k] = prompt_for(k)
    return out
