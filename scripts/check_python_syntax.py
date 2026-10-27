"""检查仓库 Python 文件是否能被当前解释器解析，不生成 __pycache__。"""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIRS = (ROOT / "src", ROOT / "tests", ROOT / "scripts")


def main() -> None:
    """解析源码、测试和开发脚本中的 Python 文件。

    :return: 无。
    """
    sources = sorted(
        path
        for directory in SOURCE_DIRS
        for path in directory.rglob("*.py")
        if "__pycache__" not in path.parts
    )

    for source in sources:
        code = source.read_text(encoding="utf-8")
        compile(code, str(source), "exec")

    print("Python 语法检查通过：{} 个文件。".format(len(sources)))


if __name__ == "__main__":
    main()
