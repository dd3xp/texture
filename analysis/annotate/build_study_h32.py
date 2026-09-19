"""(h) 32px 人工盲比的**页面构建器**：交付配置 `TRD32_rr4` vs 最强基线 `B2`。

⚠⚠ **本脚本只造页面，不采集数据。** 跑不跑、跑多少对，是 (M64)/(M66)/(M67) 摆给用户的决定
（账本 `docs/arch_progress.md` 三候选表的第一行）。本文件与它生成的 HTML 在用户点头之前
**不发出、不标注**。

为什么只剩这一把尺子
--------------------
32px 上 `GOAL.md` 的四族自动指标已全部判完，没有一把能认证"领先基线"：
  * 分布指标（KID/FID/FD）：(M53) 地板太吵 `VOID_FLOOR_TOO_NOISY`；(M64) 更判死
    ——32px 测试集的真人参照恒为 0，且 64px 也补不出来（`NO_32PX_TEST_REFS`）。
  * 图文相似度（配对 CLIP@32px）：(M58)–(M62) 能定向，但 (M63) `GATE_CEILING_BELOW_GOAL`
    ——**真人瓦片自己的 CLIP 就比 B2 低 0.7762＝门槛的 1.89×** ⇒ 只能定向、不能认证。
  * 可平铺性：(M65) `VOID_RULER_BLIND`，按包自助区间盖住了不可平铺对照。
  * 多样性：(M66) `DIV_ANCHOR_UNDECIDED`，且**交付形状 4 选 1 ⇒ 对 `TRD32_rr4` 与 `B2`
    构造上就没有数**。
  * VLM 判官：32px 校准从未过关（(M9) 期连真人 vs B2 都分不出），且 (M44) 的功效硬上限
    仍在 ⇒ ⛔ 不解禁。
人工盲比**免疫于"参照集荒"**：它比的是"哪张更像这个材质名"，不需要任何真人参照瓦片。

--- 判据（造页面时写下并 commit，采数据之前不改；采数据之后一个字不改）---

**统计单位**：一对 = 一个 E_mat 测试材质。每对问两次（左右互换），**两答不一致即弃用**
（`reduce_paired_orders.py`，与本项目全部 44 条判官臂同一条规矩）。

**主判据**：在两序一致（decided）的对里，`TRD32_rr4` 的胜率对 0.5 做二项检验。
  * 胜率 >50% 且 p<0.05 -> `HUMAN32_TRD_WINS`：32px 上**人认为我们领先最强基线**。
    这是全项目唯一能下这个结论的仪器。
  * 胜率 <50% 且 p<0.05 -> `HUMAN32_B2_LEADS`：**这是按已有读数预期的结果**
    （判官 32px 41% 我们输，方向已登记，⚠ p 值在 (M17) 被撤销、只引方向）。
    它的价值不是"又输了一次"，而是**第一次给出"差多远"的可信估计**，
    并把这把尺子立成今后任何一版的验收标准。
  * 其余 -> `HUMAN32_TIE`：**在这个样本量上**分不出。⛔ 不许读成"打平了"。

**作废条件（任一触发，该次标注整份作废，⛔ 不许只删掉出错的那几题）**：
  (V1) 注意力检查错 >1 个。两类检查都算：`check` = 清晰 vs 重糊（应选清晰那张）、
       `check_same` = **两边同一张图**（应按"分不出"）。后者是 (M30) 的教训——
       重糊检查点得再快也不会错，抓不到"没真比较"。
  (V2) decided 对数 < 60。(M44) 的功效表：80% 功效检出 65% 需 n=90 **decided**；
       decided 太少则本轮只报"没测到"，⛔ 不许把不显著读成平局。
       ⚠ 补跑剩下的材质**需要用户再次点头**（那是更多人工时间），⛔ 不许自动扩样。
  (V3) `check_same` 上按"分不出"的比例 < 50%（＝这个人在两张一模一样的图上仍在瞎选边），
       等于仪器地板塌了。

**次判据（描述性，⛔ 不参与判定）**：
  * decided/n ＝ 人的可解率。与判官在 32px 的 ~41–54% 并排登记，⛔ 不许互相换算。
  * 左右选择计数（`chosen` 对 `left`/`right`）——(M30) 那次 71% 点左边就是这么抓到的。
  * 逐材质结果留档，供将来任何一版做同材质配对比较。

**盲**：页面只显示材质名，不显示哪边是哪个方法；`left`/`right` 字段记的是臂名，
标注者看不到（它在 CSV 里，标注结束才导出）。

**不重新生成任何像素**：两臂的 272 张 32px 瓦片是 `final_test.sh` 早已产出的交付物，
直接从远程 `experiments/baselines/{B2,TRD32_rr4}/32/` 取。渲染用的是
`eval/judge_pairs.py:panel`（**import，一字未改**）⇒ 人与 VLM 判官看到的是**同一种刺激**
（192px 最近邻单张 + 下方 3×3 平铺），两台仪器的差异不会被"图不一样"污染。

--- 补注 (M74)：显示几何修正（**在采集任何数据之前**提交，只改显示、判据一个字未动）---

`analysis/arch/m74_panel_geometry.py` 量出：共用模板 `task_template.html` 的
`img{width:320px;height:320px}` 是**按方形图写的**——此前七张标注页喂进去的都是方图
（16×16 / 24×24 / 384×384，实测纵横比畸变恰好 **1.000**），唯独本页喂的是 `panel()` 的
**192×392 竖长面板** ⇒ 横向拉 320/192、纵向压 320/392 ⇒ **纵横比畸变 392/192 ≈ 2.042**，
方形纹素被显示成约 2:1 的扁矩形。

⇒ 本文件只给**自己这一页**把显示框换成**等比**的 288×588（= 192×392 的 1.5 倍，整数、
每个纹素恰好 9×9 显示像素）。⛔ 不动共用模板（其余构建器喂的是方图，现状正确）、
⛔ 不动 `eval/judge_pairs.py:panel`（活件；47 条已发表判官臂靠它逐字不变才可比）、
⛔ 不动上面任何一条判据 / 作废条件 / 抽样 / 随机种子。
⚠ 这不是"改了刺激"：VLM 判官拿到的一直是未畸变的 192×392 PNG，本修正是把**人这一侧
恢复**到上面登记的"同一种刺激"，不是偏离它。
⚠ 验钥比的是面板字节、与 CSS 无关 ⇒ 不受影响，但修完必须重跑一次（实测仍 360/360、0 对不上）。
⚠⚠ **只改本文件不够**：真正发给人的是两序页 `study_h32_v2.html`，而
`build_paired_orders.py:emit` 是**从模板重建**的 ⇒ 第一次修完 v2 仍读 2.042。
那个文件自己的注释早就写过同一个坑（问题句被静默还原成模板默认，`study_spread` 上抓到过）
⇒ 按它既有的"从源页继承"写法补上显示框的继承（**对喂方图的页面逐字节无变化**，
其 `--selftest` 13→28 项与两条反向测试全过）。⚑ 可迁移：**「从模板重建」的下游构建器，
会把上游每一处逐页定制静默还原；同一个坑在同一个文件里第二次发生。**

跑法（纯本地，零 GPU 零 API）：
    python analysis/annotate/build_study_h32.py                 # 抽样 90 对
    python analysis/annotate/build_study_h32.py --n 0           # 全量 272 对
两步：本脚本出单次呈现页，再由已审计的 `build_paired_orders.py` 转成两序页。
最后**必须**验钥（把页面里每块面板重算一遍，确认它真是它自称的那条臂）：
    python analysis/annotate/build_study_h32.py --audit experiments/annotate/study_h32_v2.html
"""
import argparse
import base64
import hashlib
import io
import json
import random
import re
import sys
from pathlib import Path

from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
from judge_pairs import first_tile, panel          # noqa: E402  活件，只 import 不改
from prompts import load_set                       # noqa: E402

ARM_A = "TRD32_rr4"        # 我们交付的配置
ARM_B = "B2"               # 最强基线


def audit(page: Path, size: int, set_name: str) -> int:
    """验钥：页面里每一块面板都必须与它自称的那条臂重算出的字节完全一致。

    `build_paired_orders.verify()` 只保证"图来自源页面、左右确实换过"；它**证明不了**
    标成 `TRD32_rr4` 的那块真是 TRD 的瓦片。而键一旦错位，判决会整体倒转。
    """
    items = json.loads(re.search(r"const ITEMS = (\[.*?\]);",
                                 page.read_text(encoding="utf-8"), re.S).group(1))
    slug = {e["prompt"]: e["material"].rsplit(".", 1)[0] for e in load_set(set_name)[0]}
    exp = {}
    for arm in (ARM_A, ARM_B):
        d = ROOT / "experiments/baselines" / arm / str(size)
        for prompt, s in slug.items():
            t = first_tile(d, s)
            if t is not None:
                exp[(arm, prompt)] = hashlib.sha256(panel(t).encode()).hexdigest()

    n = bad = 0
    for it in items:
        if it["kind"] != "real":
            continue
        assert it["material"] in slug, f"材质不在 {set_name} 里：{it['material']}"
        for side, key in (("left", "limg"), ("right", "rimg")):
            n += 1
            if hashlib.sha256(it[key].encode()).hexdigest() != exp[(it[side], it["material"])]:
                bad += 1
    mats = {it["material"] for it in items if it["kind"] == "real"}
    print(f"验钥 {page.name}：真题面板 {n} 块，对不上 {bad} 块；材质 {len(mats)} 个")
    assert bad == 0, "页面里的面板与它自称的臂对不上"
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=None,
                    help="不建页，改为验钥：逐块面板重算并比对字节")
    ap.add_argument("--n", type=int, default=90,
                    help="抽多少对（0 = 全部 272）。90 来自 (M44) 功效表：80% 功效检出 65%")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--set", default="E_mat")
    ap.add_argument("--n-check", type=int, default=5, help="清晰 vs 重糊 的注意力检查数")
    ap.add_argument("--seed", type=int, default=32)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "experiments/annotate/study_h32.html")
    a = ap.parse_args()
    if a.audit:
        raise SystemExit(audit(a.audit, a.size, a.set))

    rng = random.Random(a.seed)
    da = ROOT / "experiments/baselines" / ARM_A / str(a.size)
    db = ROOT / "experiments/baselines" / ARM_B / str(a.size)

    pairs, missing = [], 0
    for e in load_set(a.set)[0]:
        slug = e["material"].rsplit(".", 1)[0]
        ta, tb = first_tile(da, slug), first_tile(db, slug)
        if ta is None or tb is None:
            missing += 1
            continue
        pairs.append((e["prompt"], ta, tb))
    n_all = len(pairs)
    if a.n and a.n < n_all:
        pairs = [pairs[i] for i in sorted(rng.sample(range(n_all), a.n))]

    items, sides = [], [0, 0]
    for label, ta, tb in pairs:
        imgs = {ARM_A: panel(ta), ARM_B: panel(tb)}
        first = sides[0] <= sides[1]              # 交付臂在左的次数逐格配平
        sides[0 if first else 1] += 1
        l, r_ = (ARM_A, ARM_B) if first else (ARM_B, ARM_A)
        items.append({"material": label, "label": label, "kind": "real", "struct": 0.0,
                      "stratum": f"{a.size}px", "left": l, "right": r_,
                      "limg": imgs[l], "rimg": imgs[r_]})
    n_real = len(items)

    # 注意力检查一：清晰 vs 同一张图重糊。（检查二 check_same 由 build_paired_orders.py 加）
    for it in rng.sample(items, min(a.n_check, len(items))):
        good_im = Image.open(io.BytesIO(base64.b64decode(it["limg"]))).convert("RGB")
        blur_im = (good_im.resize((72, 147), Image.NEAREST)
                   .filter(ImageFilter.GaussianBlur(6))
                   .resize(good_im.size, Image.BILINEAR))

        def enc(im):
            buf = io.BytesIO()
            im.save(buf, "PNG")
            return base64.b64encode(buf.getvalue()).decode()

        gg = {"good": enc(good_im), "blur": enc(blur_im)}
        l, r_ = ("good", "blur") if rng.random() < 0.5 else ("blur", "good")
        items.append({"material": it["material"], "label": it["label"], "kind": "check",
                      "struct": 0.0, "stratum": it["stratum"], "left": l, "right": r_,
                      "limg": gg[l], "rimg": gg[r_]})

    rng.shuffle(items)
    assert sides[0] == sides[1] or abs(sides[0] - sides[1]) == 1, "左右没配平"
    print(f"{ARM_A} vs {ARM_B} @{a.size}px（{a.set}）：可比对 {n_all}，缺图跳过 {missing}")
    print(f"取用 {n_real} 对 + 注意力检查 {len(items) - n_real} 道 = {len(items)} 次呈现（单序）")
    print(f"左右平衡（{ARM_A} 在左 / 在右）：{sides[0]} / {sides[1]}")

    tpl = (Path(__file__).parent / "task_template.html").read_text(encoding="utf-8")
    assert "__ITEMS__" in tpl and "MIN_MS" in tpl and "r.pair" in tpl, "模板缺件"
    # 补注 (M74)：只给本页把 CSS 显示框换成等比的 288×588（= 192×392 的 1.5 倍）。
    # 共用模板一个字不动（其余构建器喂的是方图，320×320 对它们是正确的）。
    old_box, new_box = "img{width:320px;height:320px", "img{width:288px;height:588px"
    assert tpl.count(old_box) == 1, f"模板显示框不是预期的那一处：{tpl.count(old_box)}"
    tpl = tpl.replace(old_box, new_box)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(tpl.replace("__ITEMS__", json.dumps(items, ensure_ascii=False)),
                     encoding="utf-8")
    print(f"写入 {a.out}  ({a.out.stat().st_size // 1024} KB)")
    print("下一步（两序页）：python analysis/annotate/build_paired_orders.py "
          f"{a.out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
