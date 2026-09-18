"""(M62) 零 GPU 预检之二：控制臂 28 份齐了，把 (OP5)/(OP6) 也提前问掉。

为什么现在能问、而且**不破盲**：
  (OP5) 控制臂**内部**奇偶对半的同一检验；
  (OP6) 控制臂**内部**「旧 17 vs 新 11」的同一检验。
两条的输入都**只有控制臂**，一个 trt 数字都不碰 => 对 ctrl-vs-trt 这个主对比携 0 比特信息。
(与上一轮 (OP3) 那次同构，且更强：那条靠"参照行与模型无关"，这两条压根没有 trt 这一侧。)

上一轮写死的"问不了"的理由有两条，现在都消失了：
  (a) 走 analyse() 要 len(ctrl)==k，处理臂为空会先被 (OP1) 挡回 VOID_NO_DATA
      => 本脚本**不调用 analyse()**，改为逐字复用它内部那两段所依赖的**冻结具名函数**
         M60.arm_mean / M60.perm_p，⛔ 不重写任何判据、⛔ 不传小 --k；
  (b) 控制臂新料还差 3 份 => 现已 28/28 齐。

⚠⚠ 纪律（写在看见任何 p 之前）：本脚本的结果**不许**成为改 K、改判据、改加权、
   改 n_old 或"补救"的理由。两条都不过即 VOID，如实记账。
⛔ 不打印任何 TRD 行均值，⛔ 不读 m62_dbl_*。

用法：python analysis/arch/m62_op56_precheck.py [--dir remote_tmp/m62] [--k 28] [--n_old 17]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m62_read_dose2 as M62          # 冻结判读器本体（只借常量/加载）
import m60_read_scale as M60          # M62 的冻结核：arm_mean / perm_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="remote_tmp/m62")
    ap.add_argument("--ctrl_tag", default="m62_ctrl")
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--n_old", type=int, default=17)
    a = ap.parse_args()

    ctrl, _rowmean, prob = M60.load_arm(a.dir, a.ctrl_tag, a.k)
    if len(ctrl) != a.k or prob:
        print("ctrl loaded=%d/%d problems=%s" % (len(ctrl), a.k, prob))
        print("M62_OP56_PRECHECK_NOT_READY")
        return 2

    # (OP1) 的材质口径：与 analyse() 同一行，只是这里没有 trt 一侧
    mats_c = set.intersection(*[set(v) for v in ctrl.values()])
    if len(mats_c) != M62.N_MAT_EXPECT:
        print("n_mat_ctrl=%d (expect %d)" % (len(mats_c), M62.N_MAT_EXPECT))
        print("M62_OP56_PRECHECK_NOT_READY")
        return 2
    mats = sorted(mats_c)

    # ---- (OP5) 以下四行逐字复制自 m62_read_dose2.analyse 的 (OP5) 段，⛔ 未作任何修改 ----
    ev = [s for s in sorted(ctrl) if s % 2 == 0]
    od = [s for s in sorted(ctrl) if s % 2 == 1]
    a5, b5 = M60.arm_mean(ctrl, mats, ev), M60.arm_mean(ctrl, mats, od)
    p5, m5 = M60.perm_p([a5[m] - b5[m] for m in mats], n_perm=M62.N_PERM)

    # ---- (OP6) 以下四行逐字复制自 m62_read_dose2.analyse 的 (OP6) 段，⛔ 未作任何修改 ----
    old = [s for s in sorted(ctrl) if s < a.n_old]
    new = [s for s in sorted(ctrl) if s >= a.n_old]
    a6, b6 = M60.arm_mean(ctrl, mats, old), M60.arm_mean(ctrl, mats, new)
    p6, m6 = M60.perm_p([a6[m] - b6[m] for m in mats], n_perm=M62.N_PERM)

    ok5, ok6 = p5 >= 0.01, p6 >= 0.01
    print("ctrl loaded=%d/%d  n_mat=%d" % (len(ctrl), a.k, len(mats)))
    print("(OP5) even=%d odd=%d  p=%.4f  mean=%+.6f  ok=%s"
          % (len(ev), len(od), p5, m5, ok5))
    print("(OP6) old=%d new=%d  p=%.4f  mean=%+.6f  ok=%s"
          % (len(old), len(new), p6, m6, ok6))
    print("M62_OP56_PRECHECK_%s" % ("CLEAN" if (ok5 and ok6) else "FAIL"))
    return 0 if (ok5 and ok6) else 1


if __name__ == "__main__":
    sys.exit(main())
