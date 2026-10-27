"""为 docs 中的行间公式添加每篇文档独立的 MathJax 编号。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
TAG_RE = re.compile(r"\\tag\{\d+\}")
ANY_TAG_RE = re.compile(r"\\tag\b")


def number_equations(text: str, name: str) -> tuple[str, int]:
    """保留原有内容，只在独立的 $$ 公式结束前插入或更新编号。"""
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    fence: tuple[str, int] | None = None
    math_start: int | None = None
    count = 0
    newline = "\r\n" if "\r\n" in text else "\n"

    for line_number, line in enumerate(lines, start=1):
        fence_match = FENCE_RE.match(line)
        if fence_match is not None and math_start is None:
            marker = fence_match.group(1)
            if fence is None:
                fence = (marker[0], len(marker))
            elif marker[0] == fence[0] and len(marker) >= fence[1]:
                if not fence_match.group(2).strip():
                    fence = None

        if fence is not None or line.strip() != "$$":
            output.append(line)
            continue

        if math_start is None:
            math_start = len(output)
            output.append(line)
            continue

        body = output[math_start + 1 :]
        if not body:
            raise ValueError(f"{name}:{line_number}: 空公式块")

        count += 1
        if TAG_RE.fullmatch(body[-1].strip()):
            output[-1] = f"\\tag{{{count}}}{newline}"
        elif any(ANY_TAG_RE.search(body_line) for body_line in body):
            raise ValueError(f"{name}:{line_number}: 公式中已有非末行或自定义编号")
        else:
            output.append(f"\\tag{{{count}}}{newline}")

        output.append(line)
        math_start = None

    if math_start is not None:
        raise ValueError(f"{name}: 未闭合的 $$ 公式块")
    return "".join(output), count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="仅检查编号是否连续，不修改文件")
    args = parser.parse_args()

    docs_dir = Path(__file__).resolve().parents[1] / "docs"
    changed: list[str] = []
    total = 0
    for path in sorted(docs_dir.glob("*.md")):
        original = path.read_text(encoding="utf-8")
        updated, count = number_equations(original, str(path))
        total += count
        if updated != original:
            changed.append(path.name)
            if not args.check:
                path.write_text(updated, encoding="utf-8")

    print(f"检查 {total} 个公式；需更新 {len(changed)} 篇文档")
    for name in changed:
        print(name)
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
