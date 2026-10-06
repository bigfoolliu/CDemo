#!/usr/bin/env python3
"""从 Make 的预演输出生成 clangd 编译数据库；不执行构建或训练。"""

import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "--always-make", "all",
         "build/tests/snake_ai_test", "build/tests/logger_test"],
        cwd=root, check=True, capture_output=True, text=True,
    )
    entries = []
    for line in result.stdout.splitlines():
        arguments = shlex.split(line)
        sources = [arg for arg in arguments if arg.endswith(".c")]
        if not sources or "-o" not in arguments:
            continue
        arguments[0] = shutil.which(arguments[0]) or arguments[0]
        # 多文件链接命令拆成每个翻译单元的记录，保留实际编译选项。
        for source in sources:
            entries.append({
                "directory": str(root),
                "file": str(root / source),
                "arguments": [arg for arg in arguments
                              if arg not in sources or arg == source],
            })
    if not entries:
        raise SystemExit("未找到编译命令，请检查 Makefile。")
    destination = root / "build" / "compile_commands.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
    print(f"已生成 {destination}（{len(entries)} 个翻译单元）")


if __name__ == "__main__":
    main()
