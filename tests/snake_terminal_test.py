"""伪终端集成测试：无第三方依赖，不修改用户历史文件。"""

import os
from pathlib import Path
import pty
import select
import signal
import subprocess
import tempfile
import termios
import time


def wait_for(master, output, text):
    deadline = time.monotonic() + 5
    while text.encode() not in output:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"未看到：{text}"
        if select.select([master], [], [], remaining)[0]:
            output.extend(os.read(master, 65536))


def run_case(path, interrupt):
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    process = subprocess.Popen(
        ["./build/games/snake", str(path)],
        stdin=slave, stdout=slave, stderr=slave,
    )
    output = bytearray()
    try:
        wait_for(master, output, "历史最高：8")
        assert not termios.tcgetattr(slave)[3] & termios.ICANON
        if interrupt:
            process.send_signal(signal.SIGINT)
        else:
            wait_for(master, output, "死亡！1 秒后自动重开。")
            wait_for(master, output, "新一局开始！")
            os.write(master, b"\x1b")
            time.sleep(0.02)
            os.write(master, b"[A")
            start = len(output)
            deadline = time.monotonic() + 3
            while output[start:].count(b"\x1b[2J") < 2:
                assert time.monotonic() < deadline
                if select.select([master], [], [], 0.5)[0]:
                    output.extend(os.read(master, 65536))
            frames = output[start:].split(b"\x1b[2J")[1:]
            rows = [next(i for i, line in enumerate(frame.splitlines()) if b"@" in line)
                    for frame in frames[:2]]
            assert rows[1] == rows[0] - 1, "方向键后应持续向上移动"
            os.write(master, b"q")
        process.wait(timeout=3)
        assert process.returncode == 0
        restored = termios.tcgetattr(slave)
        # macOS 可在恢复规范输入时设置内核的待重处理标志 PENDIN。
        restored[3] &= ~getattr(termios, "PENDIN", 0)
        original[3] &= ~getattr(termios, "PENDIN", 0)
        assert restored == original, f"退出必须恢复终端：{original!r} -> {restored!r}"
        scores = path.read_text().splitlines()
        assert scores[:3] == ["3", "8", "5"], "必须保留旧记录"
        assert len(scores) == (4 if interrupt else 5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)


with tempfile.TemporaryDirectory(prefix="cdemo-snake-") as directory:
    for interrupt in (False, True):
        path = Path(directory) / f"scores-{interrupt}.log"
        path.write_text("3\n8\n5\n")
        run_case(path, interrupt)
    bad = subprocess.run(
        ["./build/games/snake", f"{directory}/missing/scores.log"], capture_output=True,
    )
    assert bad.returncode != 0

print("终端集成测试通过：自动重开、方向键、得分持久化、退出与终端恢复。")
