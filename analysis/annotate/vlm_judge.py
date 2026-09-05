"""用大模型当判官——**先对着人的标注验证，验证通过才用**。

本项目的核心教训：任何新尺子都要先对着人的判断验一遍。
`struct_metric.py` 三条自定判据全过，却和人的偏好**反向**（A4）。
大模型判官同样是一把尺子，不能直接信。

好在验证不需要新标注：手上有 A4（72 对）与 B2（27 对）的人工标签，
把同样的图、同样的问题喂给模型，算一致率。

密钥只从环境变量读，不写入任何文件。
"""

import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

Q = ("下面是两张 16x16 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质？只回答 A 或 B，不要解释。")
Q_CHECK = ("下面是两张 16x16 的像素画材质贴图。\n"
           "哪一张更清晰、更像正常的像素画（而不是模糊的）？只回答 A 或 B。")


def load_items(html: Path):
    s = html.read_text(encoding="utf-8")
    return json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", s, re.S).group(1))


def upscale_b64(b64s: str, px: int = 256) -> str:
    im = Image.open(io.BytesIO(base64.b64decode(b64s))).convert("RGB")
    im = im.resize((px, px), Image.NEAREST)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(model: str, prompt: str, imgs: list[str], base: str, key: str,
        retries: int = 3) -> str | None:
    content = [{"type": "text", "text": prompt}]
    for b in imgs:
        content.append({"type": "image_url",
                        "image_url": {"url": "data:image/png;base64," + b}})
    body = {"model": model, "max_tokens": 8, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    for a in range(retries):
        try:
            r = requests.post(base.rstrip("/") + "/v1/chat/completions",
                              headers={"Authorization": "Bearer " + key},
                              json=body, timeout=120)
            if r.status_code != 200:
                time.sleep(2 + 3 * a)
                continue
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            time.sleep(2 + 3 * a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-swap", dest="swap", action="store_false",
                    help="关闭正反两问去偏（默认开）")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    base = os.environ.get("VLM_BASE_URL")
    key = os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    items = load_items(args.html)
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    if args.limit:
        rows = rows[:args.limit]

    agree = tot = 0
    chk_ok = chk_tot = 0
    left_pick = incons = 0
    recs = []
    for r in rows:
        i = int(r["idx"])
        if i >= len(items) or r["choice"] == "tie":
            continue
        it = items[i]
        prompt = Q_CHECK if it["kind"] == "check" else Q.format(label=it["label"])
        prompt += "\n（第一张是 A，第二张是 B）"
        L, R = upscale_b64(it["limg"]), upscale_b64(it["rimg"])

        def one(a, b):
            o = ask(args.model, prompt, [a, b], base, key)
            if o is None:
                return None
            u = o.upper()
            return "first" if u.startswith("A") else ("second" if u.startswith("B")
                                                     else None)

        f1 = one(L, R)
        if f1 is None:
            continue
        pick = "left" if f1 == "first" else "right"
        if args.swap:
            # **正反各问一次治位置偏好**：实测两个模型选左都在 67% 上下，
            # 不去偏的话一致率主要反映的是位置偏好而非判断。
            f2 = one(R, L)
            if f2 is None:
                continue
            pick2 = "right" if f2 == "first" else "left"
            if pick != pick2:
                incons += 1
                continue
        left_pick += pick == "left"
        chosen = it[pick]
        recs.append({"idx": i, "kind": it["kind"], "human": r["chosen"],
                     "vlm": chosen, "material": it["material"]})
        if it["kind"] == "check":
            chk_tot += 1
            chk_ok += chosen == "good"
        else:
            tot += 1
            agree += chosen == r["chosen"]
        print(f"  [{len(recs)}/{len(rows)}] {it['material'][:26]:<26} "
              f"人={r['chosen']:<9} 模型={chosen:<9}"
              f"{'✓' if chosen == r['chosen'] else '✗'}", flush=True)

    print(f"\n模型 {args.model}")
    print(f"  有效对 {tot}，与人一致 {agree} = {agree/max(tot,1):.1%}")
    print(f"  注意力检查 {chk_ok}/{chk_tot}")
    print(f"  选左比例 {left_pick/max(len(recs),1):.1%}（应接近 50%）")
    if args.swap:
        print(f"  正反不一致而弃用 {incons} 对"
              f"（{incons/max(incons+len(recs),1):.0%}，越高说明判断越不稳）")
    print("\n判读：一致率显著高于 50% -> 这把尺子可用，能替人扩大规模；")
    print("      接近 50% -> 它和人不是一回事，不能用它代替盲比。")
    if args.out:
        args.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1))
        print(f"逐条结果写入 {args.out}")


if __name__ == "__main__":
    main()
