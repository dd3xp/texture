"""16px 人工盲比的页面构建器：交付配置 `TRD16c_rr4` vs 最强基线 `B2`（论文 §6 主表的那一对）。

构建、注意力检查、左右配平、显示框修正与验钥**全部复用** `build_study_h32.py`（import，一字未改），
本文件只换臂名、尺寸、随机种子与输出路径。16px 与 32px 的 `panel()` 同为 192×392，显示框修正同样适用。

--- 判据（造页面时写下并 commit，采数据之前不改）---
与 `build_study_h32.py` 逐条相同，只把前缀换成 HUMAN16：
  两序一致（decided）的对里 `TRD16c_rr4` 胜率对 0.5 做二项检验；
  >50% 且 p<0.05 -> `HUMAN16_TRD_WINS`；<50% 且 p<0.05 -> `HUMAN16_B2_LEADS`；其余 -> `HUMAN16_TIE`（⛔ 不许读成打平）。
  作废条件 (V1)(V2)(V3) 同 32px 页。
**标注者**：四位作者**各自独立**完成一遍、各自导出 CSV（先不讨论），再另行协商出一致标签。
  主判据按**每人各自**与**协商一致**两种口径都报；另报四人间 Fleiss κ（两序一致的对上）。
  ⛔ 不许只报对 TRD 有利的那个口径。

跑法：
    python analysis/annotate/build_study_h16.py
    python analysis/annotate/build_paired_orders.py experiments/annotate/study_h16.html --n-same 5
    python analysis/annotate/build_study_h16.py --audit experiments/annotate/study_h16_v2.html
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_study_h32 as h  # noqa: E402

h.ARM_A, h.ARM_B = "TRD16c_rr4", "B2"
ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    defaults = ["--size", "16", "--seed", "16",
                "--out", str(ROOT / "experiments/annotate/study_h16.html")]
    sys.argv = sys.argv[:1] + defaults + sys.argv[1:]
    h.main()
