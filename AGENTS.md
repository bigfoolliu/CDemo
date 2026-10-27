# 仓库贡献指南

## 1. 项目结构与模块组织

本项目学习 C 基础语法和现代特性，以 C23（ISO/IEC 9899:2024）为目标。`src/` 根目录存放 C23 示例，`src/basics/` 存放基础示例，各自包含 `main` 并按编号命名。`src/modules/` 演示多文件链接，只有入口文件包含 `main`。`docs/00_文档阅读顺序.md` 是文档阅读路线索引，专题文档用两位数前缀标示建议顺序；`README.md` 是项目目录与运行指南。`Makefile` 管理构建，`build/` 存放不纳入版本控制的生成物。目前测试断言直接写在示例中。

## 2. 文档组织与写作规范

`docs/` 中的 Markdown 文档和 SVG 配图使用“顺序编号_中文主题”命名，例如 `32_线性回归分析.md`、`32_线性回归.svg`；仅保留 C、CPU、BMP、CSV、CNN 等必要标准缩写或专有名称。`README.md`、`AGENTS.md` 等工具约定名称保持不变。新增、重命名或移动文档时，同步更新 `00_文档阅读顺序.md`、所有 Markdown 链接及仓库内路径引用。

每篇文档使用一个简洁的 H1 主标题，不给主标题编号；正文按层级编号：H2 使用 `1.`、`2.`，H3 使用 `1.1`、`1.2`，H4 使用 `1.1.1`。编号在每篇文档内重新开始，层级连续，不跳级；课程阶段号等有语义的编号保留在标题文字中。章节应围绕学习目标、前置知识、原理、手算示例、源码对应、边界与练习组织，按主题取舍。

专业术语首次出现时使用“中文名称（English term，缩写）”，例如“主成分分析（principal component analysis，PCA）”；后续可用中文名称或已定义缩写。标题、图注、表格和正文遵循同一规则。优先使用简洁中文解释，不为常见普通词汇堆叠英文。

## 3. 构建、测试与开发命令

`src/graphics/` 的纯 C 绘制核心与 SDL2 窗口分离，教程为 `docs/10_二维图形引擎.md`。`make graphics` 启动可选窗口；`make test-graphics` 和 `make sanitize-graphics` 无需 SDL。默认测试不打开窗口，普通构建不引入 SDL 依赖。窗口使用 `make graphics-debug` 调试，`make compile-commands-graphics` 生成对应代码提示。新增绘图接口需说明坐标边界并验证裁切、退化图形和数值失败。

落沙核心为 `src/graphics/sand.h/c`，窗口为 `sand_demo.c`，教程为 `docs/19_落沙模拟.md`。`make sand` 启动、`make sand-debug` 调试；`make test-sand`、`make sanitize-sand` 无需 SDL。外圈固定墙不可编辑；自底向上、左右交替更新，标记移动目标及交换回源格的水，空出的空气可继续填入。无编辑时保持沙、水及墙格数，保证粒子每步最多移动一次；`make test-sand-window` 验证笔刷路径、持续绘制、失焦取消、时间及截图。

波纹核心为 `src/graphics/ripple.h/c`，窗口为 `ripple_demo.c`，教程为 `docs/18_波纹模拟.md`。`make ripple` 启动、`make ripple-debug` 调试；`make test-ripple`、`make sanitize-ripple` 无需 SDL。两层高度必须保持有限数和零边界，读取当前层、覆盖旧层后交换；保持固定步长和稳定系数，说明阻尼、限幅与反射边界。`make test-ripple-window` 验证输入、时间、暂停、截图和主循环。

贝塞尔曲线核心为 `src/graphics/bezier.h/c`，窗口为 `bezier_demo.c`，教程为 `docs/16_贝塞尔曲线编辑器.md`。`make bezier` 启动，`make bezier-debug` 调试；`make test-bezier`、`make sanitize-bezier` 无需 SDL。控制点须为画布内有限数，参数 t 为 [0,1]；失败不修改输出。保留拖动抓取偏移，处理松键、失焦与离开窗口；`make test-bezier-window` 验证坐标映射、输入、截图和主循环。

曼德勃罗核心为 `src/graphics/mandelbrot.h/c`，窗口为 `mandelbrot_demo.c`，教程为 `docs/21_曼德勃罗分形.md`。`make mandelbrot` 启动，`make mandelbrot-debug` 调试；`make test-mandelbrot`、`make sanitize-mandelbrot` 无需 SDL。区分最后一次迭代逃逸与达到上限未逃逸，有限迭代不能证明集合成员；视图须验证有限数和范围，修改失败保留状态。窗口只在视图变化时重算，`make test-mandelbrot-window` 验证输入、缓存标记、截图和主循环。

迷宫核心为 `src/graphics/maze.h/c`，入口为 `maze_demo.c`，教程为 `docs/14_迷宫生成与寻路.md`。`make maze` 启动、`make maze-debug` 调试，`make test-maze`、`make sanitize-maze` 无需 SDL。DFS 回溯生成可重复的连通无环地图，BFS/DFS 在入队或入栈时标记，前驱链用于路径回溯；需区分无权图 BFS 最短路保证与树中路径唯一。默认测试验证核心，`make test-maze-window` 验证控制、时间、截图和有界循环。

寻路学习核心为 `src/graphics/pathfinding.h/c`，窗口为 `pathfinding_demo.c`，教程为 `docs/15_寻路算法.md`。`make pathfinding` 启动，`make pathfinding-debug` 调试；`make test-pathfinding` 和 `make sanitize-pathfinding` 只编译纯 C 核心，不依赖 SDL。核心实现 BFS、Dijkstra 和 A*，与地图绘制、SDL 输入分离；网格零代价代表墙，正代价表示进入格子的移动成本。Dijkstra/A* 当前用线性数组选择最小优先级节点，需在教程说明其 `O(V²)` 复杂度。`make test-pathfinding-window` 验证算法切换、鼠标编辑、暂停、截图与主循环。

排序学习核心为 `src/graphics/sorting.h/c`，窗口为 `sorting_demo.c`，教程为 `docs/12_排序算法.md`。`make sorting` 启动，`make sorting-debug` 调试；`make test-sorting`、`make sanitize-sorting` 只验证纯 C 核心，`make test-sorting-window` 验证图例、控件、计时和截图。核心提供冒泡、插入、选择和快速排序；快速排序使用末元素枢轴、Lomuto 分区和固定容量显式区间栈，须说明其不稳定性及有序输入下的最坏复杂度。比较键值而保留原始次序字段；不同算法须使用同一确定性输入便于比较，计数定义、稳定性和复杂度需在教程说明。

BMP 编码位于 `src/graphics/bmp.h/c`，截图路径由 `bmp_capture.h/c` 管理，教程为 `docs/11_BMP图片导出.md`。`make bmp` 无需 SDL 导出示例，窗口按 S 截图。图片保存在 `exports/`，必须独占创建并保留已有文件；清理构建产物不得删除图片。`make test-bmp`、`make sanitize-bmp` 验证小端头、BGR、底部行优先和四字节填充，测试使用临时目录。编码与文件写入使用 C 标准 API；errno 错误码约定和目录创建使用 POSIX 平台接口。

康威生命游戏位于 `src/graphics/life.h/c` 和 `life_demo.c`，教程为 `docs/13_康威生命游戏.md`。`make life` 启动，`make life-debug` 调试；`make test-life`、`make sanitize-life` 无需 SDL，核心接入默认测试和检查。使用固定死亡边界与双缓冲同步更新，不得原地覆盖当前代；随机预设应可重现。`make test-life-window` 验证输入、鼠标映射与时间控制，窗口复用共享 SDL 模块。

粒子喷泉的纯 C 核心为 `src/graphics/particles.h/c`，窗口入口为 `particles_demo.c`，教程为 `docs/17_粒子喷泉.md`。`make particles` 启动，`make particles-debug` 构建调试版本；`make test-particles` 和 `make sanitize-particles` 无需 SDL。固定粒子池和随机种子保持可重复，更新使用 1/60 秒固定步长。默认测试只验证核心；`make test-particles-window` 通过 SDL dummy 驱动验证控制。图形演示共用 `graphics_sdl.h/c` 的窗口资源与显示接口。

`src/stdlib/` 是常用 ISO C 标准库专区，指南为 `docs/03_C常用标准库.md`。`make stdlib` 构建并运行本专区，示例同时接入默认构建、测试、检查与调试。示例应无需交互、保留用户文件，并解释 API 返回值、参数长度、所有权和失败行为；临时文件使用 `tmpfile`，持久日志仍须遵守 `logs/` 约束。

人工与 AI 贪吃蛇共用 `src/games/snake_env.h` 规则和 `snake_terminal.h` 终端模块。纯 C DQN 位于 `snake_ai.c` 与 `snake_ai_network.h`，教程为 `docs/62_深度强化学习贪吃蛇.md`。`make snake-ai` 只构建；长训练不加入默认测试。`make test-snake-ai` 运行数值及集成测试，`make sanitize-snake-ai` 验证内存与未定义行为。训练产物位于独立的 `sessions/snake-ai/`，不得覆盖旧实验或人工得分。

`src/games/` 存放交互游戏，`make snake` 启动使用 POSIX 终端 API 的实时贪吃蛇。游戏不加入自动运行的示例列表；通过 `--test` 测试规则并接入 `make test`。`make test-snake-terminal` 使用 Python 3 伪终端验证输入、自动重开、得分持久化及终端恢复，测试使用临时历史文件。保留用户已有 `snake_scores.log`。对应说明位于 `docs/08_终端贪吃蛇.md`。

固定块内存池实现为 `src/common/memory_pool.h`，演示为 `src/advanced/06_memory_pool.c`，教程为 `docs/06_固定块内存池.md`。`make memory-pool` 运行，`make test-memory-pool`、`make sanitize-memory-pool` 验证；示例接入默认构建、测试、检查和调试。保留 max_align_t 对齐、溢出前检查、失败不改状态与所有权约束；不要对未知外来指针做减法或大小比较。必须说明校验释放为 O(capacity)，且陈旧别名不能靠占用位完全检测。

深入示例放在 `src/advanced/`，对应讲解位于 `docs/04_C语言进阶.md`。新增示例需同步学习目录，并说明接口契约、所有权、不变量和失败行为。线程示例使用 POSIX API，须与 ISO C 特性明确区分。

需要 C 编译器和 Make，无第三方依赖。在仓库根目录运行：

- `make`：编译全部示例。
- `make run`：按文件名顺序运行示例。
- `make test`：运行示例及其中的行为断言。
- `make lint`：执行严格警告与语法检查。
- `make clean`：删除示例可执行文件。
- `make advanced`：构建并运行深入示例。
- `make ml`：运行 `src/ml/` 中的机器学习案例，数学库通过 `-lm` 链接。
- `make sanitize`：用地址与未定义行为消毒器验证深入示例，生成物位于 `build/sanitize/`。

用 `make CC=clang` 指定编译器；切换工具链或编译参数前执行 `make clean`。构建优先使用 `-std=c23`，不支持此名称时使用 `-std=c2x`。接受标准选项不代表完整实现全部特性。

## 4. 代码风格与命名规范

使用四个空格缩进，禁止 Tab，采用 UTF-8 编码并保留文件末尾换行。函数与变量使用 `snake_case`，宏使用 `UPPER_SNAKE_CASE`。代码面向学习阅读，生成时须遵守以下排版约束：

- 函数的左花括号另起一行；`if/else`、`for/while/do`、`switch` 的左花括号跟随条件，使用 `} else {`、`} while (...);`。结构体、联合体和枚举的左花括号跟随声明，类型定义的结束名称和分号按 C 语法放在右花括号之后。
- 控制流代码块必须使用花括号，即使内部只有 `return` 或 `continue`；禁止将条件、花括号和执行语句压缩到一行。
- 每行只写一条执行语句，每条变量声明只声明一个变量；`for` 头部的三个子句属于正常语法，不需机械拆开。
- 只在明显的逻辑阶段之间留一行，例如输入检查与主体处理之间、主体处理与资源清理之间；函数之间留一行，禁止连续空行。连续声明、紧密相关的计算及简单循环之间不机械插入空行。
- 每行通常不超过 100 列。长条件在逻辑运算符处换行；参数列表能放进一行时保持一行，超长时换行并允许同一续行容纳多个短参数。长初始化列表分行排列。字符串字面量可用相邻字符串分行，不能为换行改变内容。
- 复杂或嵌套三元表达式优先改为清晰分支，避免在同一行堆叠赋值与副作用；展开逻辑时必须保留短路求值和执行顺序。
- 结构体成员、枚举项逐行排列；简单坐标、短数组和短初始化保持紧凑，如 `(Point){1, 0}`、`int keys[] = {1, 2, 3}`、`{0}`。复杂嵌套初始化按层次展开，但内部简单坐标可保持一行。不为强制换行添加尾逗号；已有尾逗号可能阻止工具压缩初始化，整理时按需处理。
- 中文函数注释使用多行形式，`@brief`、每个 `@param` 和适用的 `@return` 分别成行；说明参数范围、所有权及失败行为。算法说明放在相关代码之前，避免将多个参数说明挤在一行。

例如，输入检查单独分段，紧密相关的准备与处理保持连贯：

```c
/**
 * @brief 将坐标向右移动一格。
 * @param point 可写坐标，不转移所有权；x 须小于 INT_MAX，避免加法溢出。
 * @return 修改成功为 true；空指针时返回 false，不修改任何状态。
 */
static bool move_point(Point *point)
{
    if (point == nullptr) {
        return false;
    }

    Point offset = {1, 0};
    point->x += offset.x;
    point->y += offset.y;
    return true;
}
```

新增代码遵守以上规范；修改旧代码时整理相关函数，避免无关文件的批量排版。`.clang-format` 提供对应的手动格式化规则，但单变量声明、逻辑分段、强制补花括号及注释内容仍须生成代码时主动遵守，不能完全依赖工具。工具不会判断空行的逻辑意义，也不会自动消除所有人为展开的初始化；整理时仍需检查是否过于松散。通过编译器严格警告检查代码。VS Code 配置与使用说明位于 `docs/02_VSCode开发配置.md`；`make compile-commands` 更新代码提示数据库，`make debug` 生成独立的 `build/debug/` 调试程序。

每个示例围绕一个主题，多文件教学示例需共同编译。中文注释应解释特性用途、行为边界及与旧版 C 的区别。直接展示新语法，避免用兼容宏隐藏不支持的情况。明确区分 C23 标准特性、编译器扩展和未来标准草案；涉及标准库时单独验证平台支持。新增示例后同步更新 README 和学习手册，明确已实现内容与待补充主题。

`docs/` 中独立的 `$$` 行间公式使用 `\tag{编号}`，每篇文档从 `1` 顺序编号；行内 `$...$` 不编号。增删公式后运行 `python3 scripts/number_equations.py` 更新，再用 `--check` 验证，确保实时预览和阅读视图一致。

## 5. 日志路径约束

后续开发中，凡涉及写入日志文件，路径必须位于仓库根目录的 `logs/` 目录内，可按功能或运行批次建立子目录，例如 `logs/snake-ai/<运行编号>/train.log`。默认路径和可配置的日志路径均须遵守此约束，不得将新增日志写入仓库根目录、`build/` 或 `sessions/` 等其他目录。写入前确保目标目录存在，创建或写入失败时明确报错，并保留已有日志。已有功能后续修改日志逻辑时也须遵守此约束；模型和其他非日志产物仍按各自约定存放。

## 6. 测试指南

机器学习案例配套 `docs/30_机器学习案例总览.md`，使用固定数据保持可重复，验证损失下降、预测及算法边界。明确区分实现验证与真实数据评估，不将训练集结果当作泛化能力证明。

`src/common/logger.h` 提供单线程双输出日志记录器；线性回归默认追加写入 `logs/linear_regression.log`。每行包含本地毫秒时间戳、调用源码文件名和行号，多行消息逐行加前缀。`make test-logger` 验证双输出一致、调用位置、多行、追加行为及失败路径，保留用户已有日志。

使用标准库 `assert` 检查示例行为，无第三方测试框架或覆盖率门槛。行为变更和缺陷修复应配套验证，覆盖有意义的结果及边界条件，避免未定义行为。验证时不要使用 `-DNDEBUG` 禁用断言。提交前运行 `make test` 和 `make lint`，报告所用编译器及缺失特性。

## 7. 提交与拉取请求规范

目前没有提交历史可供参考。提交标题应简短、明确，以动作描述变更；可采用 Conventional Commits 规范，例如 `feat: 添加解析器入口` 或 `fix: 处理空输入`。每次提交应围绕一个明确的目的。

拉取请求应说明问题、概述解决方案，并列出已完成的验证。关联相关问题单；涉及用户可见变更时，附上截图或示例输出。明确说明新增依赖、配置变更及后续待办事项。

## 8. 协作语言

后续对话、工作进度说明和交付说明均使用中文。代码标识符、命令及工具专有名称保留原有形式。
