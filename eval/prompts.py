"""评测提示词集（**在任何方法出结果之前定死**，2026-09-11）。

两套，都只来自 `split=test`（12 个训练时没见过的包）：

  E-all  全部测试材质（447 个）。参照集 = 全部 1265 张测试真人瓦片。
         衡量"像不像真人方块纹理"的整体分布。
  E-mat  名字里含下面 MATERIAL_WORDS 之一的测试材质 —— 任务本身（"材质名 → 纹理"）
         的主表。机器正面、活塞、控制器这类不是材质，不进 E-mat。

提示词 = 去掉模组命名空间前缀后的词（`default_desert_stone_brick` → `desert stone brick`），
所有方法（新架构与全部基线）拿到**同一串文字**。
MATERIAL_WORDS 与 NAMESPACES 是写死的规则，**不许看了结果再改**。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from tiles_data import load, clean_name        # noqa: E402

NAMESPACES = {"default", "mcl", "core", "xdecor", "nc", "moreblocks", "mesecons", "jeija",
              "carts", "farming", "stairs", "walls", "doors", "flowers", "bucket", "fire",
              "tnt", "vessels", "beds", "boats", "screwdriver", "xpanes", "dye", "technic",
              "homedecor", "ethereal", "caverealms", "moreores", "basic", "materials"}

MATERIAL_WORDS = {
    "stone", "cobble", "cobblestone", "brick", "bricks", "sandstone", "sand", "gravel",
    "dirt", "grass", "clay", "wool", "ice", "snow", "permafrost", "obsidian", "granite",
    "diorite", "andesite", "marble", "slate", "basalt", "deepslate", "netherrack", "tuff",
    "wood", "planks", "plank", "log", "bark", "tree", "leaves", "moss", "mossy",
    "ore", "coal", "iron", "gold", "copper", "diamond", "mese", "tin", "silver",
    "quartz", "glass", "concrete", "terracotta", "mud", "soil", "lava", "water",
    "coral", "cactus", "bamboo", "hay", "straw", "mushroom", "sponge", "bone",
    "metal", "steel", "bronze", "block", "tile", "tiles", "rock", "desert", "savanna",
}


def prompt_words(material: str) -> list[str]:
    return [w for w in clean_name(material) if w not in NAMESPACES]


def is_material(material: str) -> bool:
    return any(w in MATERIAL_WORDS for w in prompt_words(material))


def build():
    test = load(16, "test")
    mats = sorted({s["material"] for s in test})
    eall = [{"material": m, "prompt": " ".join(prompt_words(m)) or " ".join(clean_name(m))}
            for m in mats]
    emat = [e for e in eall if is_material(e["material"])]
    ref_mat = sum(1 for s in test if is_material(s["material"]))
    out = {"E_all": eall, "E_mat": emat,
           "ref_counts": {"E_all": len(test), "E_mat": ref_mat},
           "rule": "split=test; NAMESPACES stripped; E_mat = any MATERIAL_WORDS in prompt words"}
    p = ROOT / "eval/prompt_sets.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"E-all {len(eall)} 材质（参照 {len(test)} 张）；"
          f"E-mat {len(emat)} 材质（参照 {ref_mat} 张） -> {p.relative_to(ROOT)}")
    print("E-mat 样例:", [e["prompt"] for e in emat[:12]])
    print("被排除样例:", [e["prompt"] for e in eall if not is_material(e["material"])][:12])


def build_val():
    """验证集版本（V-all / V-mat），**只用于调参**（CFG、采样温度、选检查点）。
    与 E 集同一套规则；单独存文件，E 集（`prompt_sets.json`，出结果前已定死）不动。"""
    val = load(16, "val")
    mats = sorted({s["material"] for s in val})
    vall = [{"material": m, "prompt": " ".join(prompt_words(m)) or " ".join(clean_name(m))}
            for m in mats]
    vmat = [e for e in vall if is_material(e["material"])]
    out = {"V_all": vall, "V_mat": vmat,
           "ref_counts": {"V_all": len(val), "V_mat": sum(1 for s in val if is_material(s["material"]))},
           "rule": "split=val; same rule as E sets; for tuning only"}
    p = ROOT / "eval/prompt_sets_val.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"V-all {len(vall)}；V-mat {len(vmat)}（参照 {out['ref_counts']['V_mat']} 张） -> {p.relative_to(ROOT)}")


def load_set(name):
    """E_* 从 prompt_sets.json、V_* 从 prompt_sets_val.json 读。返回 (提示词列表, 参照 split)。"""
    f = "prompt_sets_val.json" if name.startswith("V_") else "prompt_sets.json"
    return json.loads((ROOT / "eval" / f).read_text(encoding="utf-8"))[name],         ("val" if name.startswith("V_") else "test")


if __name__ == "__main__":
    import sys as _s
    build_val() if "--val" in _s.argv else build()
