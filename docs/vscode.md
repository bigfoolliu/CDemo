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

`.clang-format` 使用四空格缩进、100 列行宽：函数左花括号另起一行，控制流、结构体和枚举的左花括号跟随条件或声明。分支和循环内部语句仍分行；短初始化和参数列表紧凑排列，超长才换行。最多保留一行连续空行，并保留 include 顺序及手工编排的中文注释，让代码清晰但不过于松散。

保存时不自动格式化；在命令面板选择“格式化文档”手动整理当前 C 文件。也可先运行 `clang-format src/目标文件.c` 在终端预览，再用 `clang-format -i src/目标文件.c` 写回。只整理本次修改涉及的代码，避免无关文件的大范围变动。

格式化工具不能代替教学代码的组织：每条声明只声明一个变量、仅在明显逻辑阶段之间留空行、为控制流补齐花括号，以及将 `@brief`、每个 `@param`、`@return` 逐行说明，仍需主动遵守 [仓库排版规范](../AGENTS.md)。工具不会自动判断逻辑分段，也不会消除所有人为展开的初始化；已有尾逗号可能让短初始化保持多行，不要仅为强制换行添加尾逗号。中文注释继续解释契约、边界和失败行为。

## 粒子效果调试

“运行和调试”选择 `C: 粒子喷泉演示`，按 F5。启动前自动执行图形编译数据库任务和 `make particles-debug`，使用 SDL 头文件路径与 `-O0 -g3`，便于逐行观察粒子结构体；生成物位于 `build/debug/graphics/particles_demo`。

在 `src/graphics/particles.c` 的 `particles_step` 内设断点检查速度、位置和寿命。在窗口中用 Space 暂停，N 前进一步，R 重置。调试器断点暂停整个进程；窗口暂停只停止模拟，仍处理输入。任务 `C: 测试粒子核心` 无需窗口或 SDL。完整说明见 [粒子教程](particles.md)。

## 康威生命游戏调试

选择 `C: 康威生命游戏` 按 F5，启动任务先更新包含 SDL 参数的编译数据库，再运行 `make life-debug`，程序位于 `build/debug/graphics/life_demo`。在 `life_step` 中观察邻居数量、两张数组与 current 索引；Space 暂停演化，N 推进一代，暂停时左键编辑。任务 `C: 测试生命游戏核心` 不依赖 SDL。规则及操作见 [生命游戏教程](life.md)。

## BMP 导出调试

选择 `C: BMP 图片导出` 按 F5，任务先生成普通编译数据库，再构建 `build/debug/graphics/bmp_demo`，无需 SDL。可在 `gfx_bmp_write` 中检查文件头、BGR 字节和行填充。已有三个窗口调试入口均支持按 S 保存；导出路径位于 `exports/`，详细说明见 [BMP 教程](bmp_export.md)。

## 固定块内存池调试

选择 `C: 固定块内存池` 按 F5，沿用普通编译数据库与 `make debug`，程序位于 `build/debug/advanced/06_memory_pool`。在头文件的分配和归还函数中设断点，观察 free_head、in_use、allocated 数组与块地址；通过实例跟踪 LIFO 地址复用。任务 `C: 测试内存池` 运行专门断言。原理和生命周期约束见 [内存池教程](memory_pool.md)。

## 迷宫生成与寻路调试

选择 `C: 迷宫生成与寻路` 按 F5，启动任务更新图形编译数据库并构建 `build/debug/graphics/maze_demo`。在 generate_step、前沿取出和 mark_path 处设断点，观察 depth、head/tail、discovered 和 parent。Space 暂停，N 单步，生成完成后 1/2 重启不同搜索。任务 `C: 测试迷宫核心` 无需 SDL，完整说明见 [迷宫教程](maze.md)。

## 曼德勃罗分形调试

选择 `C: 曼德勃罗分形` 按 F5，自动更新图形编译数据库并构建 `build/debug/graphics/mandelbrot_demo`。在 escape_unchecked 中观察旧实部、虚部和新实部，在 mandelbrot_zoom 中比较缩放前后鼠标锚点。图像只在 dirty 标记为真时重新计算。任务 `C: 测试曼德勃罗核心` 无需 SDL；数值与操作说明见 [分形教程](mandelbrot.md)。

## 贝塞尔曲线编辑器调试

选择 `C: 贝塞尔曲线编辑器` 按 F5，自动更新图形编译数据库并构建 `build/debug/graphics/bezier_demo`。在 `evaluate_unchecked` 的逐层插值循环设置断点，观察 `work` 如何从四点减少为一点；在 `handle_event` 中观察窗口坐标、`offset` 和控制点坐标。断点期间窗口暂停响应是正常现象，继续执行后恢复。任务 `C: 测试贝塞尔曲线核心` 无需 SDL；操作与原理见 [曲线教程](bezier.md)。

## 波纹模拟调试

选择 `C: 波纹模拟` 按 F5，自动更新图形编译数据库并构建 `build/debug/graphics/ripple_demo`。Space 暂停后，在 `ripple_step` 的 `laplacian` 或 `next` 处设置条件断点（例如 `x == 80 && y == 60`），按 N 单步，观察中心与邻居高度、旧层覆写和 `current` 交换。任务 `C: 测试波纹核心` 无需 SDL；数值原理与使用方法见 [波纹教程](ripple.md)。

## 落沙模拟调试

选择 `C: 落沙模拟` 按 F5，自动生成图形编译数据库并构建 `build/debug/graphics/sand_demo`。Space 暂停后按 N 单步，在 `try_move` 中观察 `source`、`target` 和 `moved`；可设置 `x == 60` 等条件断点，减少逐格停顿。`sand_step` 原地交换物质，不采用生命游戏的同步双缓冲规则。任务 `C: 测试落沙核心` 无需 SDL；规则与操作见 [落沙教程](sand.md)。

## 教学 CPU 调试

先运行 `make compile-commands` 更新多文件编译记录，再选择 `C: 教学 CPU` 按 F5。配置通过现有调试任务构建 `build/debug/cpu/demo`，使用 `-O0 -g3` 保留便于观察的变量；传入 `--trace` 同时记录客体指令状态到 `logs/cpu/trace.log`。

在 `src/cpu/cpu.c` 的 `cpu_step` 设置断点，观察 PC、原始字节、立即数与寄存器。PC 表示模拟机器的字节地址，编辑器行号表示宿主解释器位置，二者不是同一个地址空间。取消 `args` 中的 `--trace` 即可只看两个案例结果。

## 字符 RNN 调试

运行 `make compile-commands` 后选择 `C: ml/10_char_rnn.c`，按 F5 构建并启动调试版。默认不写日志，`args` 加入 `--log` 后双输出至 `logs/ml/10_char_rnn.log`。

先在 main 的 `train` 调用处设置断点，跳过数值差分循环；前向观察 `forward->h[t]` 与概率，反向观察 `future` 沿时间传递及 W_h 梯度累加。每条独立序列重置初始状态，参数只在全部反向计算与裁剪完成后更新。详细步骤见 [字符 RNN 教程](char_rnn.md)。

## GRU 记忆比特调试

运行 `make compile-commands` 后选择 `C: ml/11_gru_memory.c`，按 F5 启动。默认不写文件，`args` 可设为 `["--log"]`，同时追加到 `logs/ml/11_gru_memory.log`。先在 main 的 `train` 调用处断住，再进入前向和反向函数，观察 `z/r/n`、`future` 和共享参数梯度。

模型只在终点计算损失，每条序列重置隐藏状态；GRU 的 z 表示保留旧状态比例。完整公式和对照实验见 [GRU 记忆比特教程](gru_memory.md)。

## 小型 Transformer 调试

运行 `make compile-commands` 后选择 `C: ml/12_tiny_transformer.c`，按 F5 启动独立调试版。在 main 的 `train` 调用处设置断点可跳过数值差分循环，再观察 `network_forward` 的Q/K/V、注意力和LayerNorm缓存；在反向观察残差分支及Q/K/V导数的累加。

默认不写文件，`args` 设为 `["--log"]` 后追加至 `logs/ml/12_tiny_transformer.log`。教程与矩阵读法见 [小型 Transformer 教程](tiny_transformer.md)。

## VAE 图形生成调试

运行 `make compile-commands` 后选择 `C: ml/13_vae_shapes.c`，按 F5 启动并导出BMP。默认 `args` 为 `["--export"]`，可加 `--log` 双输出到 `logs/ml/13_vae_shapes.log`。先在 main 的 `train` 调用处断住，再观察前向缓存的mu/logvar/sigma/epsilon/z，以及反向的dmu/dlv/draw。

梯度差分时必须保持同一epsilon；零epsilon表示解码分布均值，图形先验生成则跳过编码器。详细公式与图片布局见 [VAE 教程](vae.md)。
