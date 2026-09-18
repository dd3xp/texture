"""(M61) 零 GPU 预检：只问 (OP3) —— real_half 参照行是否逐位相同、n 是否为 98。

⚠ 不进主检验、不读 TRD/more 的任何差值：`real_half` 是**参照行**，两臂按构造同一，
   携带 0 比特的 ctrl-vs-more 信息 ⇒ 盲判不受影响。
⚠ 复用冻结的判读器本体（import 其 load_arm / floor_offenders），本脚本不重写任何判据。
"""
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "m61", "analysis/arch/m61_read_noninf16.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

D = sys.argv[1] if len(sys.argv) > 1 else "remote_tmp/m61"
seeds = list(range(28))
bad = 0
for tag in ("m61_ctrl", "m61_more"):
    by, rowmean, floors, prob = m.load_arm(D, tag, seeds)
    off = m.floor_offenders(floors)
    vals = sorted(set(v[0] for v in floors.values() if v and v[0] is not None))
    ns = sorted(set(v[1] for v in floors.values() if v and v[0] is not None))
    hard = [p for p in prob if p[1] != "missing"]
    print("%-10s loaded=%2d  floors=%2d  OP3_offenders=%s" % (tag, len(by), len(floors), off))
    print("           distinct real_half.CLIP = %s   n = %s" % (vals, ns))
    print("           non-missing problems    = %s" % (hard,))
    if off is None or len(off) > 0 or hard:
        bad += 1
print("N_FLOOR_EXPECT=%s  N_IMG_EXPECT=%s  N_MAT_EXPECT=%s" %
      (m.N_FLOOR_EXPECT, m.N_IMG_EXPECT, m.N_MAT_EXPECT))
print("OP3_PRECHECK_%s" % ("FAIL" if bad else "CLEAN"))
