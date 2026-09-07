"""检查论文的交叉引用与文献引用是否配对。

起因：`Figure~\ref{fig:gradient}` 里的 `\r` 被当成回车吃掉，PDF 里印成
"Figure~ ef{fig:gradient}"。因为 `ef{...}` 不是未定义命令，pdflatex 不报错，
三遍编译加 `grep undefined` 全部干净——只有人眼或本脚本能发现。
"""
import re
from pathlib import Path

BS = chr(92)
TEX = Path("paper/main.tex")
BIB = Path("paper/refs.bib")


def main():
    s = TEX.read_text(encoding="utf-8")
    b = BIB.read_text(encoding="utf-8")
    labels = set(re.findall(BS + BS + r"label\{([^}]+)\}", s))
    refs = set(re.findall(BS + BS + r"ref\{([^}]+)\}", s))
    cites = set()
    for grp in re.findall(BS + BS + r"cite[tp]?\{([^}]+)\}", s):
        cites |= {x.strip() for x in grp.split(",")}
    bib = set(re.findall(r"@\w+\{([^,]+),", b))

    print("引用了但没有 label:", sorted(refs - labels) or "无")
    print("定义了但没被引用:", sorted(labels - refs) or "无")
    print("引用了但 bib 里没有:", sorted(cites - bib) or "无")
    print("bib 里有但没被引用:", sorted(bib - cites) or "无")

    # 被吃掉的反斜杠：Figure~/Table~/Section 后面必须紧跟命令
    bad = re.findall(r"(?:Figure|Table|Section)~(?!" + BS + BS + r")", s)
    print(f"Figure~/Table~ 后面没跟命令的: {len(bad)} 处")
    # 常见命令去掉反斜杠后残留的词，出现在行首即可疑
    stubs = ("ef{", "abel{", "aragraph{", "extbf{", "extit{", "mph{",
             "ho{", "ightarrow", "imes", "ec{", "ext{")
    for i, line in enumerate(s.splitlines(), 1):
        for st in stubs:
            if line.startswith(st):
                print(f"  第 {i} 行疑似丢反斜杠: {line[:60]}")
        # 行中版本：`\t`/`\r` 被折叠成控制符后，残词前面是制表符或非字母。
        # 实例：`2.7\times` 变成 "2.7<TAB>imes"，编译零警告，PDF 印出 "imes"。
        if "\t" in line:
            print(f"  第 {i} 行含制表符（\\t 被折叠的指纹）: {line[:60]!r}")
        for st in stubs:
            for m in re.finditer(re.escape(st), line):
                j = m.start()
                if j > 0 and not (line[j - 1].isalpha() or line[j - 1] == BS):
                    print(f"  第 {i} 行行中疑似丢反斜杠({st}): {line[max(0, j - 20):j + 20]!r}")


if __name__ == "__main__":
    main()
