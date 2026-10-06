"""短训练可重复性、CLI、模型校验和 AI 终端演示；数据只写临时目录。"""

import csv
import os
from pathlib import Path
import pty
import re
import select
import signal
import subprocess
import tempfile
import termios
import time

PROGRAM = Path("build/games/snake_ai").resolve()


def run(directory, *arguments):
    return subprocess.run([str(PROGRAM), *arguments], cwd=directory,
                          capture_output=True, text=True, timeout=30)


def wait_for(master, output, text):
    deadline = time.monotonic() + 5
    while text.encode() not in output:
        remaining = deadline - time.monotonic()
        assert remaining > 0, f"未看到 {text}"
        if select.select([master], [], [], remaining)[0]:
            output.extend(os.read(master, 65536))


def play_case(directory, model, interrupt):
    master, slave = pty.openpty()
    original = termios.tcgetattr(slave)
    process = subprocess.Popen([str(PROGRAM), "play", "--model", str(model)],
                               cwd=directory, stdin=slave, stdout=slave, stderr=slave)
    output = bytearray()
    try:
        wait_for(master, output, "AI 控制方向")
        assert not termios.tcgetattr(slave)[3] & termios.ICANON
        if interrupt:
            process.send_signal(signal.SIGINT)
        else:
            wait_for(master, output, "AI 死亡，一秒后重开。")
            wait_for(master, output, "AI 新一局开始。")
            os.write(master, b"q")
        process.wait(timeout=3)
        assert process.returncode == 0
        restored = termios.tcgetattr(slave)
        for attributes in (restored, original):
            attributes[3] &= ~getattr(termios, "PENDIN", 0)
        assert restored == original
        records = list(Path(directory).glob("sessions/snake-ai/run-*/ai_scores.log"))
        assert records
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)
        os.close(slave)


with tempfile.TemporaryDirectory(prefix="cdemo-ai-integration-") as directory:
    root = Path(directory)
    # 人工得分哨兵，训练、评估和演示均不得写入。
    human = root / "snake_scores.log"
    human.write_text("99\n")
    sessions = []
    for _ in range(2):
        result = run(directory, "train", "--seconds", "30", "--steps", "1500", "--seed", "42")
        assert result.returncode == 0, result.stderr
        match = re.search(r"运行目录：([^\n\r]+)", result.stdout)
        assert match
        sessions.append(root / match.group(1))
    assert sessions[0] != sessions[1]
    for filename in ("best.model", "final.model"):
        assert (sessions[0] / filename).read_bytes() == (sessions[1] / filename).read_bytes()
    rows = []
    for session in sessions:
        with (session / "metrics.csv").open() as file:
            metrics = list(csv.DictReader(file))
        for row in metrics:
            del row["elapsed"]
        rows.append(metrics)
        assert (session / "config.txt").exists()
        assert (root / "logs/snake-ai" / session.name / "report.txt").exists()
    assert rows[0] == rows[1]
    assert sum(int(row["steps"]) for row in rows[0]) == 1500
    result = run(directory, "eval", "--model", str(sessions[0] / "best.model"))
    assert result.returncode == 0 and "模型评估" in result.stdout
    # 常数 Q 模型总选择直行，用于确定性地验证碰撞和重开。
    model = root / "straight.model"
    header = "CDEMO_SNAKE_DQN 1 11 64 64 3 1 20 10\n"
    model.write_text(header + "0\n" * 5123)
    play_case(directory, model, False)
    play_case(directory, model, True)
    for content in ("wrong version\n", header + "0\n", header + "nan\n" * 5123,
                    header + "0\n" * 5123 + "extra\n"):
        model.write_text(content)
        result = run(directory, "eval", "--model", str(model))
        assert result.returncode != 0
    for arguments in (("train", "--seconds", "0"), ("train", "--seed", "-1"),
                      ("train", "--steps"), ("eval",), ("bad",)):
        assert run(directory, *arguments).returncode != 0
    assert human.read_text() == "99\n"

print("AI 集成测试通过：短训练可重复、独立评估、模型校验、自动游戏/重开、Q/Ctrl+C、终端恢复和历史隔离。")
