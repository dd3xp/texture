"""抓「用了但没定义」的名字——GPU/API 脚本的潜伏 NameError。

为什么需要：这些脚本本地跑不起来（缺 GPU、缺 API 凭据），`ast.parse` 只能
验语法，函数体里引用一个不存在的名字要等到远端跑到那一行才炸。
本项目已因此栽过一次（`crop_scale_study.py` 的 `binom_p` 半截改名，
`adc341b` 引入，直到实验跑到统计段才暴露）。

做法：对每个模块，收集所有绑定（import / def / class / 赋值 / 形参 / for /
with / except / 推导式），再遍历所有 `Load` 上下文的 Name，报出既不在绑定里、
也不是内建的那些。作用域不做精确建模——**宁可多报**，逐条人工过一眼即可。
"""
import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["analysis", "scripts", "tools"]
SKIP = {"__pycache__"}


def bound_names(tree: ast.AST) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                out.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ar = n.args
                for a in (list(ar.args) + list(ar.posonlyargs) + list(ar.kwonlyargs)):
                    out.add(a.arg)
                for a in (ar.vararg, ar.kwarg):
                    if a:
                        out.add(a.arg)
        elif isinstance(n, ast.Lambda):
            ar = n.args
            for a in (list(ar.args) + list(ar.posonlyargs) + list(ar.kwonlyargs)):
                out.add(a.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out.update(n.names)
    return out


def check(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    known = bound_names(tree) | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    bad = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in known:
            bad.setdefault(n.id, n.lineno)
    return bad


def main():
    files = []
    for t in TARGETS:
        for p in (ROOT / t).rglob("*.py"):
            if not any(s in p.parts for s in SKIP):
                files.append(p)
    total = 0
    for p in sorted(files):
        try:
            bad = check(p)
        except SyntaxError as e:
            print(f"**语法错误** {p.relative_to(ROOT)}: {e}")
            total += 1
            continue
        if bad:
            total += len(bad)
            rel = p.relative_to(ROOT)
            for name, line in sorted(bad.items(), key=lambda kv: kv[1]):
                print(f"**未定义** {rel}:{line}  {name}")
    print(f"\n扫了 {len(files)} 个文件，可疑 {total} 处"
          f"{'' if total else ' —— 全净'}")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
