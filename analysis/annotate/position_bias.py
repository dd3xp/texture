"""干净口径地量位置偏好，替换早期那个来路不明的「67% 选左」。

早期记录（plan.md）说「首轮实测两个模型选左都在 67% 上下」，但那一轮用的是
B2（人自己就分不出来的集合），且没留下逐条原始回答。而 A4 上**去偏后**三个
判官选左都只有 34–38.5%（人 45.1%），方向与 67% 相反——两者不可能同时成立。

本脚本对每一对问两次（原序、反序），把两次的**原始回答**都记下来，给出：
  - 位置偏好 = P(选第一张)，在全部 2N 次提问上统计，中性值 50%；
  - 换序一致率，与 WebDevJudge（83.5–89.6%）同口径可比。
判官只看图，不看人的标签，故本脚本不需要 CSV 的 choice 列。
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_judge import load_items, upscale_b64, ask, Q, Q_CHECK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    # 单次调用实测 gemini 约 89 秒；串行 144 次要 3.5 小时，必须并发。
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    base, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")

    items = load_items(a.html)
    if a.limit:
        items = items[:a.limit]

    def job(t):
        i, it = t
        prompt = (Q_CHECK if it["kind"] == "check" else Q.format(label=it["label"]))
        prompt += "\n（第一张是 A，第二张是 B）"
        L, R = upscale_b64(it["limg"]), upscale_b64(it["rimg"])

        def one(x, y):
            o = ask(a.model, prompt, [x, y], base, key)
            if o is None:
                return None
            u = o.upper()
            return "first" if u.startswith("A") else ("second" if u.startswith("B") else None)

        f1, f2 = one(L, R), one(R, L)          # 原序、反序
        if f1 is None or f2 is None:
            return None
        return {"idx": i, "kind": it["kind"], "material": it["material"],
                "order1": f1, "order2": f2,
                "pick1": "left" if f1 == "first" else "right",
                "pick2": "right" if f2 == "first" else "left"}

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        recs = [r for r in ex.map(job, list(enumerate(items))) if r]

    pairs = len(recs)
    asked = 2 * pairs
    first = sum((r["order1"] == "first") + (r["order2"] == "first") for r in recs)
    o1 = sum(r["order1"] == "first" for r in recs)
    consistent = sum(r["pick1"] == r["pick2"] for r in recs)

    print(f"\n模型 {a.model}   有效对 {pairs}")
    print(f"  位置偏好 P(选第一张) = {first}/{asked} = {first/max(asked,1):.1%}（中性 50%）")
    print(f"    其中仅看原序 = {o1}/{pairs} = {o1/max(pairs,1):.1%}"
          f"   ← 与 emoji 脚本同口径；该口径把内容偏好混进来，仅供对照")
    print(f"  换序一致率 = {consistent}/{pairs} = {consistent/max(pairs,1):.1%}"
          f"   （WebDevJudge 各判官 83.5–89.6%）")
    a.out.write_text(json.dumps(recs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"逐条原始回答写入 {a.out}")


if __name__ == "__main__":
    main()
