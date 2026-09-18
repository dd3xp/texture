"""(M62) 零 GPU 预检：在**处理臂还不存在**时，把控制臂那半边能问的操作检验提前问掉。

只问三条，每一条都**携带 0 比特的 ctrl-vs-trt 信息** ⇒ 不破盲：
  (OP2) 每份 JSON 内部自洽（|mean(_CLIP_per) − CLIP| < 1e-4）—— 文件内部性质；
  (OP3) `real_half` 参照行逐位相同 —— 参照行只由参照集前后对半决定，与模型、与种子无关
        ((M61) 已在 3 个检查点、38 个种子上实测)；
  (OP4) 控制臂前 n_old 份复现 (M60) 已发表的 `trt_row_means` —— 比的是两份**已发表**的控制臂读数。

⛔ 不问 (OP5)/(OP6)：它们要走 `analyse()`，而 `analyse()` 的 (OP1) 要求 `len(ctrl)==k`
   ⇒ 想提前问就得传小 `--k` ＝ 真的进主检验 ＝ 自毁盲判（(M61) 写死的纪律）。
⛔ 不重写任何判据：全部 import 冻结的 `m62_read_dose2` / `m60_read_scale` 本体。
⛔ 不打印任何 TRD 行均值（(M61) 教训二：探针不许把读数端上来）。

用法：python analysis/arch/m62_op3_precheck.py [--dir remote_tmp/m62] [--k 28] [--n_old 17]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m62_read_dose2 as M62          # 冻结判读器本体
import m60_read_scale as M60          # M62 的冻结核


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="remote_tmp/m62")
    ap.add_argument("--ctrl_tag", default="m62_ctrl")
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--n_old", type=int, default=17)
    ap.add_argument("--m60json", default="experiments/m60_scale.json")
    a = ap.parse_args()

    by, rowmean, prob = M60.load_arm(a.dir, a.ctrl_tag, a.k)
    present = sorted(by)
    missing = [s for s, why in prob if why == "missing"]
    hard = [p for p in prob if p[1] != "missing"]

    refs = M62.ref_rows(a.dir, a.ctrl_tag, a.k)
    seen = {s: v for s, v in refs.items() if v is not None}
    distinct = sorted(set(seen.values()))

    m60_rows = {}
    if os.path.exists(a.m60json):
        with open(a.m60json, encoding="utf-8") as f:
            for s, v in (json.load(f).get("trt_row_means") or {}).items():
                m60_rows[int(s)] = v
    op4 = M62.op4_offenders(rowmean, m60_rows, a.n_old)

    old_ok = [s for s in present if s < a.n_old]
    new_ok = [s for s in present if s >= a.n_old]
    print("ctrl loaded=%d/%d  (old %d/%d, new %d/%d)  missing=%s"
          % (len(present), a.k, len(old_ok), a.n_old,
             len(new_ok), a.k - a.n_old, missing))
    print("(OP2) non-missing problems = %s" % (hard,))
    print("(OP3) distinct real_half.CLIP over %d loaded files = %s"
          % (len(seen), distinct))
    print("(OP4) offenders over old %d = %s"
          % (a.n_old, "N/A (M60 rows unavailable)" if op4 is None else len(op4)))

    bad = bool(hard) or len(distinct) != 1 or op4 is None or len(op4) > 0
    print("M62_CTRL_PRECHECK_%s" % ("FAIL" if bad else "CLEAN"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
