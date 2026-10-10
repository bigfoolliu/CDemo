#!/usr/bin/env python3
"""从 Make 的预演输出生成 clangd 编译数据库；不执行构建或训练。"""

import json
from pathlib import Path
import shlex
import shutil
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphics", action="store_true", help="加入依赖 SDL2 的图形窗口编译命令")
    options = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    targets = ["all", "build/tests/snake_ai_test", "build/tests/logger_test", "build/tests/graphics_test", "build/tests/particles_test"]
    if options.graphics:
        targets.extend(["build/graphics/demo", "build/tests/graphics_window_test",
                        "build/graphics/particles_demo", "build/tests/particles_window_test"])
    result = subprocess.run(
        ["make", "--no-print-directory", "--dry-run", "--always-make", *targets],
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
    # 可选窗口与核心测试共用翻译单元，保留每个源文件第一条有效命令。
    entries = list({entry["file"]: entry for entry in reversed(entries)}.values())
    if not entries:
        raise SystemExit("未找到编译命令，请检查 Makefile。")
    destination = root / "build" / "compile_commands.json"
    if not options.graphics and destination.exists():
        # 普通示例任务不要求 SDL；保留已显式生成的窗口记录，避免提示突然失效。
        previous = json.loads(destination.read_text())
        optional_files = {str(root / "src" / "graphics" / "demo.c"),
                          str(root / "tests" / "graphics_window_test.c"),
                          str(root / "src" / "graphics" / "particles_demo.c"),
                          str(root / "src" / "graphics" / "graphics_sdl.c"),
                          str(root / "tests" / "particles_window_test.c")}
        entries.extend(entry for entry in previous if entry["file"] in optional_files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(entries, ensure_ascii=False, indent=2) + "\n")
    print(f"已生成 {destination}（{len(entries)} 个翻译单元）")


if __name__ == "__main__":
    main()
