"""用**经过验证的**判官判一份盲比任务（无人工标签时）。

只有通过验证的判官才该这样用。验证口径见 B6/B15/B17：
在人工已标注的 A4 上，判官须**复现结论**（方向 + 显著性 + 分层顺序），
而不只是逐条一致率高。

  gemini-3.1-pro  62% 不显著、分层抹平  -> 仅粗筛
  gpt-5.6-sol     42% 方向倒转          -> 不可用
  claude-opus-5   73% p=0.008、分层顺序保住 -> **可用于粗粒度大效应**

即便如此它仍压缩效应（人 86% -> 73%，分层 7% -> 25%），
故结果须标明为**下界**，且不得用于精确幅度或细分层结论。

**方向性限定（重要）**：压缩会把任何真实差异推向 50%，
即**压缩本身就在制造零结果**。因此该判官

- **可以**支撑"存在大效应"（观测值是下界）；
- **不可以**确认"两者不可区分"这类零结论——
  那正是压缩会伪造的东西；
- 但**可以证伪零结论**：若它在压缩之下**仍**判出显著差异，
  说明真实差异只会更大。单向检验，能证伪不能证实。
"""

import argparse, base64, io, json, os, re, time
from pathlib import Path

import requests
from PIL import Image

Q = ("下面是两张 {n}x{n} 的像素画材质贴图，材质是「{label}」。\n"
     "哪一张更像这个材质、更像一张能用的游戏贴图？只回答 A 或 B，不要解释。\n"
     "（第一张是 A，第二张是 B）")
Q_CHECK = ("下面是两张像素画贴图。哪一张更清晰、更像正常的像素画"
           "（而不是模糊的）？只回答 A 或 B。\n（第一张是 A，第二张是 B）")


def up(b64s, px=256):
    im = Image.open(io.BytesIO(base64.b64decode(b64s))).convert("RGB").resize((px, px), Image.NEAREST)
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
                              headers={"Authorization": "Bearer " + key}, json=body, timeout=180)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        time.sleep(2 + 3 * a)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    base, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要 VLM_BASE_URL / VLM_API_KEY")

    s = args.html.read_text(encoding="utf-8")
    items = json.loads(re.search(r"const ITEMS = (\[.*?\]);\n", s, re.S).group(1))
    recs, inc = [], 0
    for i, it in enumerate(items):
        q = (Q_CHECK if it["kind"] == "check"
             else Q.format(n=args.size, label=it["label"]))
        L, R = up(it["limg"]), up(it["rimg"])
        o1 = ask(args.model, q, [L, R], base, key)
        o2 = ask(args.model, q, [R, L], base, key)
        if not o1 or not o2:
            continue
        p1 = "left" if o1.upper().startswith("A") else "right"
        p2 = "right" if o2.upper().startswith("A") else "left"
        if p1 != p2:                       # 正反不一致 -> 弃用（去位置偏好）
            inc += 1
            continue
        recs.append({"idx": i, "kind": it["kind"], "material": it["material"],
                     "stratum": it.get("stratum"), "chosen": it[p1]})
        if len(recs) % 5 == 0:
            print(f"  [{len(recs)}] 已判", flush=True)
    args.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1))

    real = [r for r in recs if r["kind"] == "real"]
    chk = [r for r in recs if r["kind"] == "check"]
    ok = sum(1 for r in chk if r["chosen"] == "good")
    w = sum(1 for r in real if r["chosen"] == "after")
    n = len(real)
    print(f"\n判官 {args.model}   有效 {n} 对（正反不一致弃用 {inc}）")
    print(f"  注意力检查 {ok}/{len(chk)}")
    if n:
        print(f"  **裁剪后胜 {w}/{n} = {w/n:.0%}**   p={binom_test(w, n):.3g}")
    print("\n注：该判官在 A4 上把 86% 压成 73%，故此处数字是**下界**；"
          "不得用于精确幅度或细分层结论。")


if __name__ == "__main__":
    main()
