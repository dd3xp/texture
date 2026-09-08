"""正文到底占几页？——按**页面文字**判定，不看标签页码。

为什么需要这个脚本：`main.aux` 里的 `endoflimit` 标签放在最后一个限制段**之后**，
若正文恰好在第 9 页底结束，标签会落到第 10 页，看起来像超页。
本会话已因此两次白删正文（更早还有 `endofmain` 那次——它甚至在复现性声明之后，
而声明与参考文献都不计入 ICLR 的 9 页）。

判定方式：抽每页文字，找出**最后一个含正文内容的页**。
正文结束的标志是 Limitations 的最后一段；其后是复现性声明、AI 使用声明、
参考文献、附录，这些都不计入页数限制。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "paper/main.pdf"
LIMIT = 9                      # ICLR 2027 正文页数上限

# 正文最后一段的特征词，改了限制节最后一段就要同步改这里
LAST_BODY = "Two generators"     # 限制节最后一段的特征词，改那段就要同步改这里
# 这些之后的内容不计入页数
AFTER = ("REPRODUCIBILITY", "REFERENCES", "AI USE")


def main():
    try:
        from pypdf import PdfReader
    except ImportError:
        raise SystemExit("需要 pypdf（本地有；远端 GPU 机器上没有，故此检查在本地做）")
    if not PDF.exists():
        raise SystemExit(f"缺 {PDF}")
    r = PdfReader(str(PDF))
    pages = [(r.pages[i].extract_text() or "") for i in range(len(r.pages))]

    body_end = None
    after_start = None
    for i, t in enumerate(pages):
        if LAST_BODY.lower() in t.lower():
            body_end = i + 1
        u = t.upper()
        if after_start is None and any(k in u for k in AFTER):
            after_start = i + 1

    print(f"总页数 {len(pages)}")
    print(f"正文最后一段（“{LAST_BODY}”）出现在第 {body_end} 页")
    print(f"不计页数的内容（声明/参考文献）自第 {after_start} 页起")
    if body_end is None:
        raise SystemExit("找不到正文末段特征词——改过限制节就要更新 LAST_BODY")
    ok = body_end <= LIMIT
    print(f"\n正文占 {body_end} 页，上限 {LIMIT} -> {'通过' if ok else '**超页**'}")
    if not ok:
        print("  真的超了才删正文。删之前先确认不是浮动图落到正文末页——")
        print("  本项目出现过两次：图被推到 Limitations 中间，删字治不了。")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
