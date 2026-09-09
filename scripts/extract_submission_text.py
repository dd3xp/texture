"""从 main.tex 抽出标题与摘要的纯文本，供 OpenReview 粘贴。

摘要截止（2026-09-18）比正文早一周，那一步只需要标题 + 摘要纯文本。
手工去 LaTeX 容易漏（`\\%`、`{,}`、行内数学、`---`），所以机器抽。

改了 main.tex 就重跑：`python scripts/extract_submission_text.py`
输出 `paper/submission_text.md`，并在末尾报残留的 LaTeX 命令（应为空）。
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper/main.tex"
OUT = ROOT / "paper/submission_text.md"
BS = chr(92)


def clean(t: str) -> str:
    # **先保护转义百分号**：`re.sub(r"%.*", "", t)` 会从 `\%` 里的 % 开始吞掉整行，
    # 第一版因此把摘要里 "9.8\% of cells against 8.6\%..." 整段吃没了。
    SENT = "@@PCT@@"
    t = t.replace(BS + "%", SENT)
    t = re.sub(r"%.*", "", t)                       # 真注释
    t = t.replace(SENT, "%")
    t = t.replace(BS + BS, " ")                     # 强制换行
    t = re.sub(BS * 2 + r"(?:emph|texttt|textbf|textit)\{([^{}]*)\}", r"\1", t)
    t = re.sub(BS * 2 + r"[a-zA-Z]+\{([^{}]*)\}", r"\1", t)
    t = t.replace(BS + "&", "&").replace(BS + "_", "_")
    t = t.replace("{,}", ",").replace("{=}", "=")
    t = t.replace(BS + "sim", "~").replace(BS + "times", "x")
    t = re.sub(r"\$([^$]*)\$", r"\1", t)            # 行内数学
    t = t.replace("---", "—").replace("--", "–")
    t = t.replace("``", '"').replace("''", '"')
    t = re.sub(r"[{}]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    s = TEX.read_text(encoding="utf-8")
    m = re.search(BS * 2 + r"title\{(.*?)\}" + BS * 2 + r"s*" + BS * 2 + r"n", s, re.S)
    if not m:                                        # 退回到第一个平衡花括号
        i = s.index(BS + "title{") + 7
        depth, j = 1, i
        while depth:
            depth += (s[j] == "{") - (s[j] == "}")
            j += 1
        m_title = s[i:j - 1]
    else:
        m_title = m.group(1)
    abst = s[s.index(BS + "begin{abstract}") + len(BS + "begin{abstract}"):
             s.index(BS + "end{abstract}")]

    T, A = clean(m_title), clean(abst)
    OUT.write_text(
        "# OpenReview 提交用纯文本\n\n"
        "> 摘要截止 2026-09-18，正文 2026-09-25。\n"
        "> 由 `paper/main.tex` 机器抽取；改了 tex 就重跑 "
        "`python scripts/extract_submission_text.py`。\n\n"
        "## Title\n\n" + T + "\n\n## Abstract\n\n" + A + "\n\n"
        f"---\n\n标题 {len(T)} 字符；摘要 {len(A.split())} 词 / {len(A)} 字符。\n",
        encoding="utf-8")

    left = sorted(set(re.findall(BS * 2 + r"[a-zA-Z]+", T + A)))
    print(f"标题：{T}")
    print(f"\n摘要：{len(A.split())} 词 / {len(A)} 字符")
    print(f"残留 LaTeX 命令：{left or '无'}")
    print(f"写入 {OUT}")
    sys.exit(1 if left else 0)


if __name__ == "__main__":
    main()
