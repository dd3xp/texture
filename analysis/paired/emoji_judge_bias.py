"""判官的位置偏好是否也出现在另一个领域（emoji）。

B6 在纹理任务上量到 gemini 选左 67%（文献报告约 5%）。
若这是纹理特有的，就只能作为领域内观察；
若在 emoji 上同样出现，则是该类判官的普遍问题。

**位置偏好不需要人工标注**：只看选左比例，期望值 50%。
用同一批 emoji 的两套独立设计构成对，问"哪个更像 X"。
"""

import argparse, base64, io, itertools, os, time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

NAMES = {
    "1f600": "grinning face", "1f603": "smiling face", "1f604": "smiling face with open mouth",
    "1f609": "winking face", "1f60a": "smiling face with smiling eyes", "1f60d": "smiling face with heart eyes",
    "1f61c": "winking face with tongue", "1f620": "angry face", "1f622": "crying face",
    "1f62d": "loudly crying face", "1f631": "face screaming in fear", "1f634": "sleeping face",
    "1f383": "jack-o-lantern", "1f384": "christmas tree", "1f385": "santa claus",
    "1f386": "fireworks", "1f388": "balloon", "1f389": "party popper",
    "1f34e": "red apple", "1f34f": "green apple", "1f350": "pear", "1f351": "peach",
    "1f352": "cherries", "1f353": "strawberry", "1f345": "tomato", "1f346": "eggplant",
    "1f680": "rocket", "1f682": "locomotive", "1f684": "high-speed train",
    "1f30d": "globe showing europe-africa", "1f311": "new moon", "1f315": "full moon",
    "1f408": "cat", "1f415": "dog", "1f418": "elephant", "1f419": "octopus",
}
SETS = ["noto", "twemoji", "openmoji"]


def b64(f, px=256):
    im = Image.open(f).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    im = Image.alpha_composite(bg, im).convert("RGB").resize((px, px), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def ask(model, prompt, imgs, base, key, retries=3):
    content = [{"type": "text", "text": prompt}]
    for b in imgs:
        content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + b}})
    body = {"model": model, "max_tokens": 8, "temperature": 0,
            "messages": [{"role": "user", "content": content}]}
    for a in range(retries):
        try:
            r = requests.post(base.rstrip("/") + "/v1/chat/completions",
                              headers={"Authorization": "Bearer " + key}, json=body, timeout=120)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        time.sleep(2 + 3 * a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-3.1-pro-preview")
    ap.add_argument("--cache", type=Path, default=Path("data/emoji"))
    args = ap.parse_args()
    base, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要 VLM_BASE_URL / VLM_API_KEY")

    left = tot = swap_consistent = swap_tot = 0
    for code, name in NAMES.items():
        fs = {s: args.cache / f"{s}_{code}.png" for s in SETS}
        fs = {s: f for s, f in fs.items() if f.exists() and f.stat().st_size > 0}
        if len(fs) < 2:
            continue
        for a, b in itertools.combinations(sorted(fs), 2):
            q = (f"下面是两个「{name}」的 emoji 设计。\n"
                 f"哪一个更好地表现了「{name}」？只回答 A 或 B，不要解释。\n"
                 f"（第一张是 A，第二张是 B）")
            A, B = b64(fs[a]), b64(fs[b])
            o1 = ask(args.model, q, [A, B], base, key)
            if not o1:
                continue
            p1 = "left" if o1.upper().startswith("A") else ("right" if o1.upper().startswith("B") else None)
            if p1 is None:
                continue
            tot += 1; left += p1 == "left"
            o2 = ask(args.model, q, [B, A], base, key)
            if o2:
                p2 = "right" if o2.upper().startswith("A") else "left"
                swap_tot += 1; swap_consistent += (p1 == p2)
            if tot % 10 == 0:
                print(f"  [{tot}] 选左 {left/tot:.0%}", flush=True)
    import sys as _s; _s.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from exact import binom_test  # 无依赖，见 analysis/exact.py
    print(f"\n模型 {args.model}   emoji 对 {tot}")
    print(f"  选左比例 {left/max(tot,1):.1%}   （期望 50%）"
          f"  二项 p={binom_test(left, tot):.3g}")
    print(f"  正反一致 {swap_consistent}/{swap_tot} = {swap_consistent/max(swap_tot,1):.0%}")
    print(f"\n对照 · 纹理任务（B6）：gemini 选左 67%")
    print("判读：emoji 上也明显偏离 50% -> 位置偏好非纹理特有。")


if __name__ == "__main__":
    main()
