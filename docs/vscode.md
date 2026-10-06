# VS Code 开发配置

在 VS Code 中直接打开仓库根目录，确保终端能运行 `cc`、`make`、`python3` 和 `clangd`。本机 Apple Clang 17 支持项目使用的 `-std=c23`；其他编译器可能使用 `-std=c2x`，由 Makefile 自动探测。接受标准参数不意味着实现了所有 C23 特性。

## 扩展与代码提示

在扩展面板搜索 `@recommended`，安装 `clangd、CodeLLDB 和 Makefile Tools`。clangd 提供补全、跳转、诊断和格式化；CodeLLDB 提供断点和变量观察；Makefile Tools 方便查看构建目标。配置已关闭 Microsoft C/C++ 扩展的 IntelliSense，避免两套诊断重复运行。

首次运行 `make compile-commands`，或者按 `Cmd+Shift+B` 构建全部。编译数据库位于 `build/compile_commands.json`，通过 Make 的预演命令生成，不会运行示例。它记录实际 C 标准、线程参数和各翻译单元，让编辑器理解多文件模块及测试代码。新增源文件、修改编译参数或切换编译器后重新生成；例如 `make CC=clang compile-commands`。切换正常构建工具链前仍需 `make clean`。

如果提示找不到 clangd，在用户设置中指定 `clangd.path`。没有补全时检查数据库是否存在，然后执行命令面板的 `clangd: Restart language server`。编译器自身的诊断仍以 `make lint` 为准。

## 构建与测试

通过“终端 → 运行任务”选择全部构建、调试构建、严格检查、普通测试或 AI 集成测试。默认构建快捷键为 `Cmd+Shift+B`（Windows/Linux 为 `Ctrl+Shift+B`）。普通构建不会自动开始训练，测试任务可能产生较多终端输出。

`make debug` 在 `build/debug/` 编译全部学习示例及 AI 单元测试，使用 `-O0 -g3`，便于逐行观察变量。正常 AI 构建继续使用 `-O2`，长期训练使用 `make snake-ai` 的产物。两类生成物分目录，`make clean` 清理调试程序和编译数据库。

## 断点与运行

在“运行和调试”中选择线性回归、人工贪吃蛇、AI 短训练或 AI 单元测试，然后按 F5。启动前自动构建调试版本。点击行号左侧设置断点，F10 单步跳过，F11 进入函数，在变量或监视面板观察参数、损失和数组。macOS 的功能键可能需要同时按 Fn。

游戏在集成终端运行，以支持方向键和终端输入；先点击终端获取焦点。正常退出优先使用游戏的 `q` 键，人工游戏停止调试时也配置了 SIGTERM 恢复终端。若异常退出后终端状态失常，可运行 `reset`。断点暂停会暂停整个实时游戏。

AI 调试训练限制为 1500 步或 30 秒，适合观察流程，不代表能训练出高分模型。模型与实验产物仍存于 `sessions/snake-ai/`，日志位于 `logs/`；线性回归日志为 `logs/linear_regression.log`。所有配置都以仓库根目录为工作目录。

其他示例可复制一个启动配置，将 `program` 改成对应的 `build/debug/...` 路径。多文件示例使用 `build/debug/modules_demo`，不要单独编译模块入口。

## 格式化

`.clang-format` 规定四空格缩进、函数左花括号另起一行，并保留 include 顺序。保存时不自动格式化，避免学习过程中出现大范围改动；使用“格式化文档”手动整理当前 C 文件，或运行 `clang-format -i src/目标文件.c`。只格式化本次修改涉及的文件，中文注释需继续解释契约、边界和失败行为。
