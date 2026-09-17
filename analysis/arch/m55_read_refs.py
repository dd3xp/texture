#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M55) 判读器：按预注册第四节的判据逐字映射（**盲写**：写于两臂任何指标存在之前）。

判据不是本文件定的，全文在 `docs/arch_progress.md` 的 (M55) 第四节（预注册提交 `ed75a9c`）：
  主判据（32px、V_mat、run_eval 的 CLIP）
    REFS_HELP_32  : refs 臂 CLIP - v10 臂 CLIP > 0.41（m=1 门槛），**且**护栏不触发
    REFS_NULL_32  : 差 <= 0.41
  护栏（非循环）：16px 的 KID_x1e3 同时变差超过 4.656（(M33) m=1 门槛）-> REFS_TRADEOFF
  操作检验 (OP3)：两臂 `n` 与 `materials` 相等（(M14)：不等就不许比）

⛔ 本文件不新增判据、不放宽门槛。数据缺失时**不下判**，另报"已查几项"
（(M31) 那条坑：不许拿空集冒充"量过没事"）。
⚠ 打印里只用 ASCII 与常见汉字（Windows 控制台 GBK）；落盘一律在打印之前。
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLIP_THR = 0.41        # eval/noise_floor.py m=1
KID_THR = 4.656        # (M33) m=1


def read_arm(path, key):
    """返回 (row, 说明)。缺文件/缺键都返回 None，调用方负责记账。"""
    if path is None:
        return None, "未给路径"
    p = Path(path)
    if not p.exists():
        return None, f"缺文件 {p}"
    rows = json.loads(p.read_text(encoding="utf-8"))
    if key not in rows:
        return None, f"{p.name} 里没有键 {key}（有 {sorted(rows)}）"
    return rows[key], "ok"


def judge(a32, b32, a16, b16):
    """逐字映射预注册第四节。返回 (verdict, detail)。"""
    d = {}
    if a32 is None or b32 is None:
        return "NO_DATA_MAIN", d
    # (OP3)：两臂必须同 n 同 materials
    d["n"] = [a32.get("n"), b32.get("n")]
    d["materials"] = [a32.get("materials"), b32.get("materials")]
    if d["n"][0] != d["n"][1] or d["materials"][0] != d["materials"][1]:
        return "VOID_OP3_UNCOMPARABLE", d
    d["clip"] = [a32.get("CLIP"), b32.get("CLIP")]
    if d["clip"][0] is None or d["clip"][1] is None:
        return "NO_DATA_MAIN", d
    d["d_clip"] = d["clip"][0] - d["clip"][1]
    d["clip_thr"] = CLIP_THR
    if d["d_clip"] <= CLIP_THR:                      # ⛔ 差 0.0002 也是没过（(M14)）
        return "REFS_NULL_32", d
    # 主判据已过 -> 必须先能读护栏，否则不下判
    if a16 is None or b16 is None:
        return "NO_DATA_GUARDRAIL", d
    d["kid16"] = [a16.get("KID_x1e3"), b16.get("KID_x1e3")]
    if d["kid16"][0] is None or d["kid16"][1] is None:
        return "NO_DATA_GUARDRAIL", d
    d["d_kid16"] = d["kid16"][0] - d["kid16"][1]     # KID 越小越好 -> 正数 = 变差
    d["kid_thr"] = KID_THR
    if d["d_kid16"] > KID_THR:
        return "REFS_TRADEOFF", d
    return "REFS_HELP_32", d


def selftest():
    def row(clip, kid=10.0, n=125, mats=125):
        return {"CLIP": clip, "KID_x1e3": kid, "n": n, "materials": mats}
    cases = [
        # (a32, b32, a16, b16, 期望)
        (row(35.0), row(34.0), row(35.0, 10.0), row(34.0, 10.0), "REFS_HELP_32"),
        (row(34.5), row(34.324), row(35.0), row(34.8), "REFS_NULL_32"),          # 0.176 < 0.41
        (row(0.41), row(0.0), row(35.0), row(34.8), "REFS_NULL_32"),            # 恰好 0.41 = 不过
        (row(34.733), row(34.324), row(35.0), row(34.8), "REFS_NULL_32"),        # 0.409 差一点也是不过
        (row(34.735), row(34.324), row(35.0), row(34.8), "REFS_HELP_32"),        # 0.411 刚过
        (row(35.0), row(34.0), row(35.0, 16.0), row(34.0, 10.0), "REFS_TRADEOFF"),   # KID 差 6.0
        (row(35.0), row(34.0), row(35.0, 14.0), row(34.0, 10.0), "REFS_HELP_32"),    # KID 差 4.0 < 门槛
        (row(35.0), row(34.0, n=70, mats=70), None, None, "VOID_OP3_UNCOMPARABLE"),
        (row(35.0), row(34.0), None, None, "NO_DATA_GUARDRAIL"),
        (None, row(34.0), None, None, "NO_DATA_MAIN"),
        (row(35.0), {"n": 125, "materials": 125}, None, None, "NO_DATA_MAIN"),   # 32px 缺 CLIP 列
    ]
    bad = 0
    for i, (a32, b32, a16, b16, want) in enumerate(cases):
        got, _ = judge(a32, b32, a16, b16)
        ok = got == want
        bad += not ok
        print(f"  [{i}] {'ok ' if ok else 'BAD'} want={want} got={got}")
    print(f"selftest {len(cases)-bad}/{len(cases)}")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json32", help="32px 的 run_eval 输出（两臂可在同一个文件里）")
    ap.add_argument("--json32b", help="v10 臂另存一个文件时给这个")
    ap.add_argument("--json16")
    ap.add_argument("--json16b")
    ap.add_argument("--json24", help="次要读数，只登记不下判")
    ap.add_argument("--json24b")
    ap.add_argument("--a", default="TRD_refs", help="实验臂在 JSON 里的方法键")
    ap.add_argument("--b", default="TRD_v10", help="对照臂（v10）的方法键")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments" / "m55_refs_verdict.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)

    checked = {}
    arms = {}
    for tag, pa, pb in (("32", a.json32, a.json32b or a.json32),
                        ("16", a.json16, a.json16b or a.json16),
                        ("24", a.json24, a.json24b or a.json24)):
        ra, na = read_arm(pa, a.a)
        rb, nb = read_arm(pb, a.b)
        arms[tag] = (ra, rb)
        checked[f"{tag}px_{a.a}"] = na
        checked[f"{tag}px_{a.b}"] = nb

    verdict, detail = judge(*arms["32"], *arms["16"])
    sec = {}
    for tag in ("16", "24"):
        ra, rb = arms[tag]
        if ra and rb and ra.get("CLIP") is not None and rb.get("CLIP") is not None:
            sec[f"d_clip_{tag}"] = ra["CLIP"] - rb["CLIP"]
        for col in ("KID_x1e3", "FID", "FD_DINOv2", "LPIPS_div", "tile_seam_ratio"):
            if ra and rb and ra.get(col) is not None and rb.get(col) is not None:
                sec[f"d_{col}_{tag}"] = ra[col] - rb[col]

    res = {"round": "M55", "arm_a": a.a, "arm_b": a.b, "verdict": verdict,
           "detail": detail, "secondary_registered_only": sec, "checked": checked,
           "n_checked": len(checked), "n_missing": sum(v != "ok" for v in checked.values())}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")   # 落盘在打印之前

    print(f"(M55) 判读器 —— 判据来自预注册 ed75a9c，本文件不新增判据")
    print(f"  已查 {res['n_checked']} 项，其中缺 {res['n_missing']} 项：")
    for k, v in checked.items():
        print(f"    {k}: {v}")
    print(f"  detail: {json.dumps(detail, ensure_ascii=False)}")
    print(f"  次要读数（只登记不下判）: {json.dumps(sec, ensure_ascii=False)}")
    print(f"\n  判决 = {verdict}")
    if verdict.startswith("NO_DATA"):
        print("  【禁】数据不全，未下判；不许把缺数据读成 REFS_NULL_32")
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
