#!/usr/bin/env python
"""(M57) 输入构造：把"有 32px 真人瓦片的训练材质"做成一个提示词集 `T32b`。零 GPU、零 API、零判官。

为什么需要它：(M23)（`e0ebf16`）结账时留下一条**明写"须预注册才能测"的提案**（原文）：
  「控制组产物 `v11dx_direct` MAD 中位 **9.47** < 训练池 **18.15** ＝模型在这把尺子上比它的训练料
    更靠平铺侧。**两者材质集合不同、混杂 → ⛔ 不许当结果引**；要测必须同材质配对并预注册判据。」
同材质配对需要"模型在**有 32px 真人瓦片的那些材质**上的产物"，而现成的产物都在 V_mat（验证材质）上，
V_mat 的 32px 参照又恰是 (M22) 判 `REF_PERIODIC_FRAGILE` 的那个病态池（83/113 逐像素就是平铺）
⇒ 必须另出一批图。本脚本只负责**定下那批图的提示词**，不看任何读数。

口径（跑之前定死，⛔ 事后不许改）：
  - 只用 **base 池**＝`load(32, "train", extra=False)`（`dataset_k16.json` 本体，403 张 / 240 材质 / 19 包）。
    ⛔ 不含 `train_64to32.json`：那是 64px 降采样，**它的像素周期是预处理产物**，正好污染要量的量
    （(M23) 原文就把它单独拎出来说"要分开量"）。
    ⛔ 也不含 `train_extra_packs_only.json`：那份只在服务器上（gitignore），本集要能在本机复现。
  - 条目 = **材质名**（不按提示词去重）：同一材质名可出现在多个包里（240 个名字里 110 个跨包），
    而 `gen_trd.py` 按 `material` 存文件名 ⇒ 用材质名当主键才配得上。
    ⚠ 少数不同材质名会得到同一串提示词（条件完全相同），无害，只是多算几张。
  - 提示词规则 = `eval/prompts.py` 的 `prompt_words`，与 E/V/T_all **同一套**，一个字没改。

写法：**只往 `eval/prompt_sets_train.json` 里加 `T32b` 这一个键**，其余键（`T_all`）按原字节校验，
⛔ 一个字不改（`load_set` 按首字母 "T" 读这个文件，所以只能加在这里）。

用法：
    python analysis/arch/m57_build_t32set.py            # 写入并打印
    python analysis/arch/m57_build_t32set.py --check     # 只校验已有的 T32b 与重算一致
"""
import argparse
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "eval"))
from tiles_data import load           # noqa: E402
from prompts import prompt_words      # noqa: E402

SET_NAME = "T32b"
F = ROOT / "eval/prompt_sets_train.json"


def build():
    s = load(32, "train", extra=False)
    mats = sorted({x["material"] for x in s})
    entries = [{"material": m, "prompt": " ".join(prompt_words(m))} for m in mats]
    bad = [e for e in entries if not e["prompt"].strip()]
    if bad:
        raise SystemExit(f"空提示词 {len(bad)} 个：{bad[:3]}")
    packs = Counter(x["pack"] for x in s)
    return entries, {"n_tiles": len(s), "n_materials": len(mats), "n_packs": len(packs),
                     "top_packs": packs.most_common(5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    entries, info = build()
    d = json.loads(F.read_text(encoding="utf-8"))
    print(f"base 32px train：{info['n_tiles']} 张 / {info['n_materials']} 材质 / {info['n_packs']} 包")
    print("前 5 大包：", info["top_packs"])
    print(f"{SET_NAME}：{len(entries)} 条；样例 {[e['prompt'] for e in entries[:6]]}")
    print("提示词重复（不同材质名同一串）：",
          sum(c - 1 for c in Counter(e["prompt"] for e in entries).values() if c > 1))

    if a.check:
        old = d.get(SET_NAME)
        same = old == entries
        print(f"--check：文件里的 {SET_NAME} 与重算{'一致' if same else '不一致'}")
        return 0 if same else 1

    if SET_NAME in d and d[SET_NAME] != entries:
        raise SystemExit(f"{SET_NAME} 已存在且内容不同 —— 拒绝覆盖（跑前口径不许改）")
    keys_before = [k for k in d if k != SET_NAME]
    before = {k: d[k] for k in keys_before}
    d[SET_NAME] = entries
    assert all(d[k] == before[k] for k in keys_before), "其余键被动到了"
    F.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {F.relative_to(ROOT)}（键：{list(d.keys())}，其余键逐条校验未动）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
