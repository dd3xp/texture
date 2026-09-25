#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""模拟评审小组：五个**不同厂商**的模型各审两遍，逐份落盘，⛔ 不做任何分数平均。

为什么跨厂商是硬要求：本项目已实测同一批数据换厂商的判官会**逐条倒转**结论（B15：一个厂商把
人的 78% 读成 42%），且"自洽率高 ≠ 判得对"。⇒ 单模型的评审意见没有参考价值；
本脚本的产物是**去重后的缺点清单 + 谁提到了它**，分数只登记、不聚合。

⚠ 用法上的两条纪律：
  ① 凭据只从环境变量读（`VLM_BASE_URL` / `VLM_API_KEY`），⛔ 不落盘、不进命令行；
  ② 开跑前先对**每个模型**做一次最小补全探针 —— 2026-09-17 踩过：网关列着某个模型、认证也过，
     但补全稳定报上游错误，结果 272 次调用全废（详见 memory 的 vlm-credentials 条）。

    VLM_BASE_URL=... VLM_API_KEY=... python scripts/ai_review.py --paper paper2.txt --out /tmp/aireview
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

MODELS = ["claude-opus-5", "gpt-5.6-sol", "gemini-3.8-flash", "grok-4.5", "deepseek/deepseek-v4-pro"]

FORM = """You are reviewing a submission to ICLR 2027. Read the paper below and write a review in the
conference's own format. Be a demanding but fair reviewer: your job is to find the weaknesses an area
chair would want to know about, not to be kind.

Return ONLY a JSON object with exactly these keys:
{
 "summary": "<3-5 sentences: what the paper claims and does>",
 "strengths": ["<one point each>", ...],
 "weaknesses": [{"point": "<the objection, concretely>",
                 "kind": "evidence|novelty|presentation|scope|reproducibility",
                 "severity": "blocking|major|minor",
                 "what_would_fix_it": "<what the authors could add or change>"}, ...],
 "questions": ["<question to the authors>", ...],
 "soundness": <1-4>, "presentation": <1-4>, "contribution": <1-4>,
 "rating": <1-10>, "confidence": <1-5>,
 "would_you_champion_it": true|false
}
Rules: at least 4 weaknesses; be specific about which claim or table you mean; if a number looks
unsupported say so and name it; do not invent content that is not in the paper. JSON only, no prose
outside it.

=== PAPER ===
"""


def ask(model, prompt, base, key, max_tokens=16000, retries=3):   # 推理型要留足余量
    body = {"model": model, "max_tokens": max_tokens, "temperature": 1.0,
            "messages": [{"role": "user", "content": prompt}]}
    for a in range(retries):
        try:
            r = requests.post(base.rstrip("/") + "/v1/chat/completions",
                              headers={"Authorization": "Bearer " + key}, json=body, timeout=600)
            if r.status_code != 200:
                print(f"    HTTP {r.status_code}: {r.text[:160]}", flush=True)
                time.sleep(5 + 10 * a)
                continue
            msg = r.json()["choices"][0]["message"]
            # 推理型模型（deepseek-v4-pro / glm-5 / gpt-5.6-sol 等）会先花 token 推理：
            # token 不够时 content 为空、内容在 reasoning_content 里 ⇒ 两个都要看，否则会误判"模型不可用"。
            return msg.get("content") or msg.get("reasoning_content") or ""
        except Exception as e:                      # noqa: BLE001
            print(f"    异常 {type(e).__name__}: {e}", flush=True)
            time.sleep(5 + 10 * a)
    return None


def probe(model, base, key):
    """开跑前的最小补全探针：只 GET /models 会放过"模型在列表里但上游坏了"这种故障。"""
    # ⚠ max_tokens 必须给够：给 8 时 deepseek-v4-pro / gemini-3.8-flash 都被误判为不可用（2026-09-25 实测）。
    out = ask(model, "Reply with the single word OK.", base, key, max_tokens=200, retries=2)
    return bool(out and out.strip())


def parse_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:                                        # 常见毛病：尾随逗号
            return json.loads(re.sub(r",(\s*[}\]])", r"\1", m.group(0)))
        except json.JSONDecodeError:
            return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/aireview"))
    ap.add_argument("--runs", type=int, default=2, help="每个模型审几遍（两遍用来看模型自身的不稳定）")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--max_tokens", type=int, default=16000,
                    help="⚠ 网关按 prompt+max_tokens **预扣**额度：贵的模型给太大会直接 402（2026-09-25 实测 grok-4.5）")
    a = ap.parse_args()
    base, key = os.environ.get("VLM_BASE_URL"), os.environ.get("VLM_API_KEY")
    if not base or not key:
        raise SystemExit("需要环境变量 VLM_BASE_URL 与 VLM_API_KEY")
    a.out.mkdir(parents=True, exist_ok=True)
    paper = a.paper.read_text(encoding="utf-8", errors="replace")
    prompt = FORM + paper

    alive = []
    for m in a.models:
        ok = probe(m, base, key)
        print(f"探针 {m:<28} {'OK' if ok else '**不可用，跳过**'}", flush=True)
        if ok:
            alive.append(m)
    if not alive:
        raise SystemExit("没有可用模型（网关或上游故障）")

    done, fail = 0, []
    for m in alive:
        for run in range(1, a.runs + 1):
            tag = m.replace("/", "_") + f"_r{run}"
            p = a.out / f"review_{tag}.json"
            if p.exists():
                print(f"{tag} 已有，跳过", flush=True)
                continue
            print(f"审稿中 {tag} …", flush=True)
            t0 = time.time()
            raw = ask(m, prompt, base, key, max_tokens=a.max_tokens)
            js = parse_json(raw)
            rec = {"model": m, "run": run, "seconds": round(time.time() - t0, 1),
                   "parsed": js is not None, "review": js, "raw": None if js else raw}
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            if js:
                done += 1
                print(f"  完成 {rec['seconds']}s  rating={js.get('rating')} "
                      f"soundness={js.get('soundness')} 缺点 {len(js.get('weaknesses') or [])} 条", flush=True)
            else:
                fail.append(tag)
                print(f"  **解析失败**（原文已存）", flush=True)
    print(f"\n可用模型 {len(alive)}/{len(a.models)}；成功 {done} 份；解析失败 {fail or '无'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
