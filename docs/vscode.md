# VS Code 开发配置

在 VS Code 中直接打开仓库根目录，确保终端能运行 `cc`、`make`、`python3` 和 `clangd`。本机 Apple Clang 17 支持项目使用的 `-std=c23`；其他编译器可能使用 `-std=c2x`，由 Makefile 自动探测。接受标准参数不意味着实现了所有 C23 特性。

## 扩展与代码提示

在扩展面板搜索 `@recommended`，安装 `clangd、CodeLLDB 和 Makefile Tools`。clangd 提供补全、跳转、诊断和格式化；CodeLLDB 提供断点和变量观察；Makefile Tools 方便查看构建目标。配置已关闭 Microsoft C/C++ 扩展的 IntelliSense，避免两套诊断重复运行。

首次运行 `make compile-commands`，或者按 `Cmd+Shift+B` 构建全部。编译数据库位于 `build/compile_commands.json`，通过 Make 的预演命令生成，不会运行示例。它记录实际 C 标准、线程参数和各翻译单元，让编辑器理解多文件模块及测试代码。新增源文件、修改编译参数或切换编译器后重新生成；例如 `make CC=clang compile-commands`。切换正常构建工具链前仍需 `make clean`。

调试构建任务也会先生成编译数据库。仓库的 `.clangd` 为 C 源文件明确指定 C23，即使数据库被 `make clean` 删除，仍能识别 `bool`、`true` 和 `nullptr` 等关键字。不要为了消除编辑器报错而改写教学示例的新语法。

如果提示找不到 clangd，在用户设置中指定 `clangd.path`。没有补全或 `bool` 被标红时检查数据库是否存在，然后执行命令面板的 `clangd: Restart language server`。编译器自身的诊断仍以 `make lint` 为准。

## 构建与测试

通过“终端 → 运行任务”选择全部构建、调试构建、严格检查、普通测试、运行标准库示例或 AI 集成测试。默认构建快捷键为 `Cmd+Shift+B`（Windows/Linux 为 `Ctrl+Shift+B`）。普通构建不会自动开始训练，测试任务可能产生较多终端输出。

`make debug` 在 `build/debug/` 编译全部学习示例及 AI 单元测试，使用 `-O0 -g3`，便于逐行观察变量。正常 AI 构建继续使用 `-O2`，长期训练使用 `make snake-ai` 的产物。两类生成物分目录，`make clean` 清理调试程序和编译数据库。

## 断点与运行

标准库专区提供十个 `C: stdlib/对应文件.c` 入口，与普通示例一样使用调试版本；学习步骤和接口速查见 [常用标准库学习指南](c_standard_library.md)。

在“运行和调试”下拉列表中选择与 C 文件对应的入口，然后按 F5。配置已覆盖 `src/` 根目录、`basics/`、`advanced/` 和 `ml/` 的全部独立示例，以及多文件模块、人工贪吃蛇、AI 短训练和 AI 单元测试。启动前自动构建调试版本，`stopOnEntry` 默认为 `false`，仅在设置的断点或运行异常处暂停。短小示例可能立即结束，请先设置断点；需要启动后立即暂停时可临时改为 `true`。

调试控制台中的 `Console is in 'commands' mode`、`Launching` 和 `Launched process` 是 CodeLLDB 的正常提示，程序输出显示在集成终端中。在该控制台求值时使用 `?count`，也可直接使用变量或监视面板。启用 `stopOnEntry` 后，启动停顿可能显示 `signal SIGSTOP`；这与程序崩溃不同，按 F5 可继续执行。相关行为见 [CodeLLDB 使用手册](https://github.com/vadimcn/codelldb/blob/master/MANUAL.md)。

例如打开 `src/basics/06_pointers.c`，选择 `C: basics/06_pointers.c` 再按 F5。仅打开文件不会自动切换调试入口，运行前检查下拉列表。点击行号左侧设置断点，F10 单步跳过，F11 进入函数，在变量或监视面板观察参数、损失和数组。macOS 的功能键可能需要同时按 Fn。

游戏在集成终端运行，以支持方向键和终端输入；先点击终端获取焦点。正常退出优先使用游戏的 `q` 键，人工游戏停止调试时也配置了 SIGTERM 恢复终端。若异常退出后终端状态失常，可运行 `reset`。断点暂停会暂停整个实时游戏。

AI 调试训练限制为 1500 步或 30 秒，适合观察流程，不代表能训练出高分模型。模型与实验产物仍存于 `sessions/snake-ai/`，日志位于 `logs/`；线性回归日志为 `logs/linear_regression.log`。所有配置都以仓库根目录为工作目录。

调试 `src/modules/main.c` 或 `arithmetic.c` 时选择 `C: 多文件模块（modules）`，该入口使用共同链接的 `build/debug/modules_demo`。头文件没有独立入口，请在调用它的示例中设置断点。后续新增示例可复制一个启动配置，将 `program` 改成对应的 `build/debug/...` 路径。

## 格式化

`.clang-format` 规定四空格缩进、函数左花括号另起一行，并保留 include 顺序。保存时不自动格式化，避免学习过程中出现大范围改动；使用“格式化文档”手动整理当前 C 文件，或运行 `clang-format -i src/目标文件.c`。只格式化本次修改涉及的文件，中文注释需继续解释契约、边界和失败行为。

## 粒子效果调试

“运行和调试”选择 `C: 粒子喷泉演示`，按 F5。启动前自动执行图形编译数据库任务和 `make particles-debug`，使用 SDL 头文件路径与 `-O0 -g3`，便于逐行观察粒子结构体；生成物位于 `build/debug/graphics/particles_demo`。

在 `src/graphics/particles.c` 的 `particles_step` 内设断点检查速度、位置和寿命。在窗口中用 Space 暂停，N 前进一步，R 重置。调试器断点暂停整个进程；窗口暂停只停止模拟，仍处理输入。任务 `C: 测试粒子核心` 无需窗口或 SDL。完整说明见 [粒子教程](particles.md)。

## 康威生命游戏调试

选择 `C: 康威生命游戏` 按 F5，启动任务先更新包含 SDL 参数的编译数据库，再运行 `make life-debug`，程序位于 `build/debug/graphics/life_demo`。在 `life_step` 中观察邻居数量、两张数组与 current 索引；Space 暂停演化，N 推进一代，暂停时左键编辑。任务 `C: 测试生命游戏核心` 不依赖 SDL。规则及操作见 [生命游戏教程](life.md)。

## BMP 导出调试

选择 `C: BMP 图片导出` 按 F5，任务先生成普通编译数据库，再构建 `build/debug/graphics/bmp_demo`，无需 SDL。可在 `gfx_bmp_write` 中检查文件头、BGR 字节和行填充。已有三个窗口调试入口均支持按 S 保存；导出路径位于 `exports/`，详细说明见 [BMP 教程](bmp_export.md)。

## 固定块内存池调试

选择 `C: 固定块内存池` 按 F5，沿用普通编译数据库与 `make debug`，程序位于 `build/debug/advanced/06_memory_pool`。在头文件的分配和归还函数中设断点，观察 free_head、in_use、allocated 数组与块地址；通过实例跟踪 LIFO 地址复用。任务 `C: 测试内存池` 运行专门断言。原理和生命周期约束见 [内存池教程](memory_pool.md)。
