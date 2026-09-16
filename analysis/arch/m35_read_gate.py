#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M35) 判据① 的判读器：固定像素周期梳齿支那一臂有没有过 16px 守门？

预注册全文 = `scripts/trd_pixcomb_train_eval.sh` 的头部 + `docs/arch_progress.md` 的 (M35) 条目，
判据随那个脚本一起提交、**先于任何数据**。本文件只实现判据①与 (OP4)，【禁】一个字不许放宽。
②③由 `analysis/arch/comb_learned.py` 判（那把尺子一个字不改）。

判据①（(M33) 第五节的守门配方，`0bbba84`）：
    两臂都 `--n 2` 生成、`run_eval.py --all_samples` 评测（= m=2 口径），
    新臂 KID 不得比控制臂 v11d 高出 **thr(2) = 2.419** 以上。
    2.419 = (M33) 在三条 TRD seed 臂上实测的 thr_max(2)；漂移项 Dmax(2)=1.162 < 它，采样项主导。
    【禁】不许拿这个门回头复审任何已判的臂（(M29) GATE_FAILED 一个字不动）。

(OP4)：两臂 `materials` 必须相等（`run_eval.py:100` 的 ok 是**每个方法各算各的**）。

报告项（【禁】非判据）：同两臂的 m=1 读数（默认取第 0 张），供与 (M29) 的 +2.118 并列登记。

本项目踩过的坑，这里逐条堵住：
  * JSON 还不存在时【禁】打印"通过"—— 另报 `n_checked`，`checked_ok` 为空不算通过；
  * 键名在真 JSON 上验过：**`KID_x1e3`** 不是 `KID`；`None` 与"不过门"在同一个布尔里长得一样；
  * 落盘一律在打印之前；打印里只用 ASCII 与常用汉字（非 GBK 字形会让 Windows 控制台崩在写 JSON 之前）。

跑法：
    python analysis/arch/m35_read_gate.py --selftest          # 先跑这个
    python analysis/arch/m35_read_gate.py --m2 <RUN>/eval_pix_Vmat_16_m2.json \
        --m1 <RUN>/eval_pix_Vmat_16.json --new pixx --ctl v11dx --out /tmp/m35_gate.json
"""
import argparse
import json
import sys
from pathlib import Path

THR2 = 2.419        # (M33) 三条 TRD seed 臂实测 thr_max(m=2)，【禁】不许改
DRIFT2 = 1.162      # (M33) Dmax(2)，报告用：< THR2 = 采样项主导
KEY = "KID_x1e3"    # 在真 JSON 上验过的键名（不是 "KID"）


def read_arm(js, name):
    """返回 (KID, materials, 出了什么问题)。缺一不可 —— 缺了就是 None，【禁】当成通过。"""
    if js is None:
        return None, None, "JSON 不存在"
    if name not in js:
        return None, None, f"JSON 里没有方法 {name}"
    row = js[name]
    if KEY not in row:
        return None, None, f"{name} 里没有键 {KEY}"
    v = row[KEY]
    if v is None:
        return None, row.get("materials"), f"{name} 的 {KEY} 是 null"
    return float(v), row.get("materials"), ""


def judge(m2_js, m1_js, new, ctl):
    out = {"thr2": THR2, "drift2": DRIFT2, "key": KEY, "new": new, "ctl": ctl,
           "checked": [], "problems": []}
    k_new, mat_new, e1 = read_arm(m2_js, new)
    k_ctl, mat_ctl, e2 = read_arm(m2_js, ctl)
    for e in (e1, e2):
        if e:
            out["problems"].append(e)
    out["m2"] = {"new_KID": k_new, "ctl_KID": k_ctl,
                 "materials_new": mat_new, "materials_ctl": mat_ctl}
    if k_new is not None and k_ctl is not None:
        out["checked"].append("m2_KID")
        out["m2"]["delta"] = k_new - k_ctl
    # (OP4) 分母相等
    op4 = (mat_new is not None and mat_ctl is not None and mat_new == mat_ctl)
    if mat_new is not None and mat_ctl is not None:
        out["checked"].append("OP4_materials")
    else:
        out["problems"].append("拿不到 materials，(OP4) 未检验")
    out["OP4_pass"] = bool(op4)

    # 报告项（非判据）：m=1
    k1n, _, e3 = read_arm(m1_js, new)
    k1c, _, e4 = read_arm(m1_js, ctl)
    out["m1_report"] = {"new_KID": k1n, "ctl_KID": k1c,
                        "delta": (None if (k1n is None or k1c is None) else k1n - k1c),
                        "note": "报告项，非判据；与 (M29) 的 +2.118 同为 m=1 口径"}
    for e in (e3, e4):
        if e:
            out["problems"].append("m1(报告项): " + e)

    out["n_checked"] = len(out["checked"])
    if out["m2"].get("delta") is None or not op4:
        out["verdict"] = "VOID_NO_DATA"          # 【禁】数据缺失不许读成通过
    elif out["m2"]["delta"] <= THR2:
        out["verdict"] = "GATE_OK"
    else:
        out["verdict"] = "GATE_FAILED"
    return out


def load(p):
    p = Path(p) if p else None
    if p is None or not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def selftest():
    """每条分支各造一次数据，确认判读器真的会在该响的时候响。"""
    cases = []
    mk = lambda kid, mats=125: {"KID_x1e3": kid, "materials": mats}          # noqa: E731

    # 1. 刚好在门内 -> 过
    js = {"pixx": mk(6.198 + 2.418), "v11dx": mk(6.198)}
    cases.append(("门内", judge(js, js, "pixx", "v11dx")["verdict"], "GATE_OK"))
    # 2. 刚好越门 -> 不过（0.0002 也是没过）
    js = {"pixx": mk(6.198 + 2.4192), "v11dx": mk(6.198)}
    cases.append(("越门", judge(js, js, "pixx", "v11dx")["verdict"], "GATE_FAILED"))
    # 3. 新臂更好 -> 过
    js = {"pixx": mk(5.0), "v11dx": mk(6.198)}
    cases.append(("更好", judge(js, js, "pixx", "v11dx")["verdict"], "GATE_OK"))
    # 4. JSON 不存在 -> VOID（【禁】拿"没量"冒充"量过没事"）
    cases.append(("无数据", judge(None, None, "pixx", "v11dx")["verdict"], "VOID_NO_DATA"))
    # 5. 少一条臂 -> VOID
    js = {"v11dx": mk(6.198)}
    cases.append(("缺新臂", judge(js, js, "pixx", "v11dx")["verdict"], "VOID_NO_DATA"))
    # 6. KID 是 null -> VOID（不是"不过门"）
    js = {"pixx": {"KID_x1e3": None, "materials": 125}, "v11dx": mk(6.198)}
    cases.append(("KID为null", judge(js, js, "pixx", "v11dx")["verdict"], "VOID_NO_DATA"))
    # 7. 键名写错（KID 而非 KID_x1e3）-> VOID
    js = {"pixx": {"KID": 6.0, "materials": 125}, "v11dx": mk(6.198)}
    cases.append(("键名错", judge(js, js, "pixx", "v11dx")["verdict"], "VOID_NO_DATA"))
    # 8. (OP4) 分母不等 -> VOID
    js = {"pixx": mk(6.0, 120), "v11dx": mk(6.198, 125)}
    cases.append(("分母不等", judge(js, js, "pixx", "v11dx")["verdict"], "VOID_NO_DATA"))
    # 9. 无数据时 n_checked 必须是 0（"已查几项"这条要真的报出来）
    r = judge(None, None, "pixx", "v11dx")
    cases.append(("无数据时已查项数", r["n_checked"], 0))
    # 10. 报告项缺失不影响主判决
    js2 = {"pixx": mk(6.5), "v11dx": mk(6.198)}
    r = judge(js2, None, "pixx", "v11dx")
    cases.append(("报告项缺失", r["verdict"], "GATE_OK"))
    cases.append(("报告项为空", r["m1_report"]["delta"], None))
    # 11. 常量没被改动
    cases.append(("thr2", THR2, 2.419))
    cases.append(("键名常量", KEY, "KID_x1e3"))

    bad = [(k, got, want) for k, got, want in cases if got != want]
    for k, got, want in cases:
        print(f"  [{'ok ' if got == want else 'BAD'}] {k}: got={got} want={want}")
    print(f"自测 {len(cases) - len(bad)}/{len(cases)} 通过")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--m2", default="", help="--all_samples 的 16px 评测 JSON（m=2 口径，判据①）")
    ap.add_argument("--m1", default="", help="默认口径的 16px 评测 JSON（m=1，报告项）")
    ap.add_argument("--new", default="pixx")
    ap.add_argument("--ctl", default="v11dx")
    ap.add_argument("--out", default="/tmp/m35_gate.json")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = judge(load(a.m2), load(a.m1), a.new, a.ctl)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")   # 先落盘
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"已查 {out['n_checked']} 项；问题 {len(out['problems'])} 条")
    print(f"==== (M35) 判据① -> {out['verdict']} ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
