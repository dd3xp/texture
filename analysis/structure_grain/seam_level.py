"""缝的档位：15 百分位这个常数对不对？

`learn_prior` 把缝色取成整幅图索引的 15 百分位——一个拍脑袋的常数。
但**缝在哪些行**是检出来的，直接量那些位置真人实际用的档位即可。

若两者差得多，说明缝画得过深或过浅，而这会直接影响出图观感
（缝太浅看不见，太深会像黑线切割）。
不需要 GPU。
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model"))
from structural_prior import learn_prior                            # noqa: E402


def main():
    ds = json.loads(Path("data/tiles/dataset_k16.json").read_text())
    bymat = {}
    for s in ds["samples"]:
        if s["size"] == 16 and s["split"] == "train":
            bymat.setdefault(s["material"], []).append(s)

    print(f"{'材质':<32}{'现行(15百分位)':>15}{'缝行实测':>11}"
          f"{'面实测':>9}{'缝-面':>8}")
    print("-" * 78)
    d15, dact, gaps = [], [], []
    for m in sorted(bymat):
        tr = bymat[m]
        if len(tr) < 4:
            continue
        raw = [np.frombuffer(bytes.fromhex(s["idx"]), np.uint8).reshape(16, 16)
               for s in tr]
        pr = learn_prior(raw, [len(s["palette"]) for s in tr], material=m)
        if not pr["rows"]:
            continue
        seam_obs, face_obs = [], []
        for t, s in zip(raw, tr):
            nk = max(len(s["palette"]) - 1, 1)
            m_ = np.zeros(16, bool)
            m_[pr["rows"]] = True
            seam_obs.append(np.median(t[m_]) / nk)
            face_obs.append(np.median(t[~m_]) / nk)
        so, fo = float(np.median(seam_obs)), float(np.median(face_obs))
        d15.append(pr["seam"])
        dact.append(so)
        gaps.append(so - fo)
        if len(d15) <= 12:
            print(f"{m:<32}{pr['seam']:>15.3f}{so:>11.3f}{fo:>9.3f}{so-fo:>8.3f}")
    print(f"\n{len(d15)} 个有横缝的材质：")
    print(f"  现行常数(15百分位) 中位 {np.median(d15):.3f}")
    print(f"  缝行实测档位       中位 {np.median(dact):.3f}")
    print(f"  缝比面暗多少       中位 {np.median(gaps):+.3f}")
    print(f"  两者绝对差         中位 {np.median(np.abs(np.array(d15)-np.array(dact))):.3f}")
    over = int((np.array(d15) < np.array(dact)).sum())
    print(f"  现行常数**比实测更暗**的材质: {over}/{len(d15)}")
    print("\n判读：常数明显偏暗 -> 缝画成了黑线，该改成实测档位")


if __name__ == "__main__":
    main()
