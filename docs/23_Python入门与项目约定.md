# Python 入门与本项目约定

## 1. 为什么同时学习 C 与 Python

本仓库用 C 逐步实现底层数据结构、数值计算和算法细节；用 Python 快速处理数据、运行实验、绘图，并在后续专题中连接成熟的机器学习工具。两种语言围绕相同概念学习，代码不必逐项重复：先用 C 看清核心机制，再用 Python 简化实验或比较结果。

## 2. 项目目录

- `src/` 保存 C 示例；`src/python/` 保存可单独运行的 Python 教学示例。
- `src/python/ml/` 按编号保存 `src/ml/` 机器学习案例的 Python 对照；GMM Python 实现位于 `src/python/gmm_em.py`。
- `scripts/` 保存仓库维护工具；`tests/` 保存 C 测试和 Python 集成脚本。
- Python 示例先使用标准库，不要求安装第三方包。主题需要额外依赖时，应单独说明用途和安装方法。
- Python 示例用 `snake_case` 命名变量和函数，类名使用 `CapWords`；导入模块时不执行训练、写文件等副作用。

第一个示例 [basics.py](../src/python/basics.py) 展示类型注解（type hints）、列表、函数、异常处理和 `__main__` 入口。

## 3. 机器学习案例的跨语言对照

`src/python/ml/01_...py` 至 `28_...py` 与同编号 C 文件对应；`29_gmm_em.c` 对应 `gmm_em.py`。建议先运行 C 版本观察显式数据结构和循环，再运行 Python 版本对照相同概念。简单算法尽量复用固定数据和核心步骤；部分高级 Python 示例刻意缩小模型、数据或训练范围，或只演示关键前向计算，阅读时应结合各自专题文档了解完整 C 实现。

使用 `python3 src/python/ml/01_linear_regression.py` 单独运行一个案例，或从仓库根目录运行 `make python` 顺序运行全部 Python 教学示例。

## 4. 运行与检查

在仓库根目录运行：

```sh
make python       # 运行 src/python/ 中的学习示例
make run          # 运行 C 示例和 Python 学习示例
make lint-python  # 检查仓库内 Python 文件的语法
make lint         # 执行 C 严格语法检查和 Python 语法检查
```

也可以直接运行单个示例：

```sh
python3 src/python/basics.py
```

Python 是解释型语言，无需编译步骤。`make` 默认仍负责构建 C 可执行文件；运行 Python 示例请使用 `make python` 或 `make run`。

## 5. 可选依赖与虚拟环境

当前示例仅依赖 Python 标准库。后续某个专题确实需要 NumPy、Matplotlib 或机器学习框架时，先创建本地虚拟环境（virtual environment）：

```sh
python3 -m venv .venv
source .venv/bin/activate
```

只为该专题记录并安装必要依赖，不把 `.venv/` 或模型缓存提交到版本控制。没有激活虚拟环境时，C 构建和标准库 Python 示例仍可使用。

## 6. VS Code 使用

安装仓库推荐的 Python 扩展，选择当前可用的 Python 3 解释器。按 `F5` 并选择 `Python: 基础示例` 可设置断点；“终端 → 运行任务”中也可以选择 `Python: 运行学习示例` 或 `Python: 语法检查`。C 代码继续使用 clangd 和 CodeLLDB 配置。
