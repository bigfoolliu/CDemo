# 仓库贡献指南

## 项目结构与模块组织

本项目学习 C 基础语法和现代特性，以 C23（ISO/IEC 9899:2024）为目标。`src/` 根目录存放 C23 示例，`src/basics/` 存放基础示例，各自包含 `main` 并按编号命名。`src/modules/` 演示多文件链接，只有入口文件包含 `main`。`docs/c_basics.md` 是中文学习手册，`README.md` 是目录与运行指南。`Makefile` 管理构建，`build/` 存放不纳入版本控制的生成物。目前测试断言直接写在示例中。

## 构建、测试与开发命令

`src/graphics/` 的纯 C 绘制核心与 SDL2 窗口分离，教程为 `docs/graphics_2d.md`。`make graphics` 启动可选窗口；`make test-graphics` 和 `make sanitize-graphics` 无需 SDL。默认测试不打开窗口，普通构建不引入 SDL 依赖。窗口使用 `make graphics-debug` 调试，`make compile-commands-graphics` 生成对应代码提示。新增绘图接口需说明坐标边界并验证裁切、退化图形和数值失败。

BMP 编码位于 `src/graphics/bmp.h/c`，截图路径由 `bmp_capture.h/c` 管理，教程为 `docs/bmp_export.md`。`make bmp` 无需 SDL 导出示例，窗口按 S 截图。图片保存在 `exports/`，必须独占创建并保留已有文件；清理构建产物不得删除图片。`make test-bmp`、`make sanitize-bmp` 验证小端头、BGR、底部行优先和四字节填充，测试使用临时目录。编码与文件写入使用 C 标准 API；errno 错误码约定和目录创建使用 POSIX 平台接口。

康威生命游戏位于 `src/graphics/life.h/c` 和 `life_demo.c`，教程为 `docs/life.md`。`make life` 启动，`make life-debug` 调试；`make test-life`、`make sanitize-life` 无需 SDL，核心接入默认测试和检查。使用固定死亡边界与双缓冲同步更新，不得原地覆盖当前代；随机预设应可重现。`make test-life-window` 验证输入、鼠标映射与时间控制，窗口复用共享 SDL 模块。

粒子喷泉的纯 C 核心为 `src/graphics/particles.h/c`，窗口入口为 `particles_demo.c`，教程为 `docs/particles.md`。`make particles` 启动，`make particles-debug` 构建调试版本；`make test-particles` 和 `make sanitize-particles` 无需 SDL。固定粒子池和随机种子保持可重复，更新使用 1/60 秒固定步长。默认测试只验证核心；`make test-particles-window` 通过 SDL dummy 驱动验证控制。图形演示共用 `graphics_sdl.h/c` 的窗口资源与显示接口。

`src/stdlib/` 是常用 ISO C 标准库专区，指南为 `docs/c_standard_library.md`。`make stdlib` 构建并运行本专区，示例同时接入默认构建、测试、检查与调试。示例应无需交互、保留用户文件，并解释 API 返回值、参数长度、所有权和失败行为；临时文件使用 `tmpfile`，持久日志仍须遵守 `logs/` 约束。

人工与 AI 贪吃蛇共用 `src/games/snake_env.h` 规则和 `snake_terminal.h` 终端模块。纯 C DQN 位于 `snake_ai.c` 与 `snake_ai_network.h`，教程为 `docs/snake_ai.md`。`make snake-ai` 只构建；长训练不加入默认测试。`make test-snake-ai` 运行数值及集成测试，`make sanitize-snake-ai` 验证内存与未定义行为。训练产物位于独立的 `sessions/snake-ai/`，不得覆盖旧实验或人工得分。

`src/games/` 存放交互游戏，`make snake` 启动使用 POSIX 终端 API 的实时贪吃蛇。游戏不加入自动运行的示例列表；通过 `--test` 测试规则并接入 `make test`。`make test-snake-terminal` 使用 Python 3 伪终端验证输入、自动重开、得分持久化及终端恢复，测试使用临时历史文件。保留用户已有 `snake_scores.log`。对应说明位于 `docs/snake.md`。

深入示例放在 `src/advanced/`，对应讲解位于 `docs/c_advanced.md`。新增示例需同步学习目录，并说明接口契约、所有权、不变量和失败行为。线程示例使用 POSIX API，须与 ISO C 特性明确区分。

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

## 代码风格与命名规范

使用四个空格缩进、UTF-8 编码、文件末尾换行，函数的左花括号另起一行。函数与变量使用 `snake_case`，宏使用 `UPPER_SNAKE_CASE`。`.clang-format` 提供手动格式化规则，通过编译器严格警告检查代码。VS Code 配置与使用说明位于 `docs/vscode.md`；`make compile-commands` 更新代码提示数据库，`make debug` 生成独立的 `build/debug/` 调试程序。

每个示例围绕一个主题，多文件教学示例需共同编译。中文注释应解释特性用途、行为边界及与旧版 C 的区别。直接展示新语法，避免用兼容宏隐藏不支持的情况。明确区分 C23 标准特性、编译器扩展和未来标准草案；涉及标准库时单独验证平台支持。新增示例后同步更新 README 和学习手册，明确已实现内容与待补充主题。

## 日志路径约束

后续开发中，凡涉及写入日志文件，路径必须位于仓库根目录的 `logs/` 目录内，可按功能或运行批次建立子目录，例如 `logs/snake-ai/<运行编号>/train.log`。默认路径和可配置的日志路径均须遵守此约束，不得将新增日志写入仓库根目录、`build/` 或 `sessions/` 等其他目录。写入前确保目标目录存在，创建或写入失败时明确报错，并保留已有日志。已有功能后续修改日志逻辑时也须遵守此约束；模型和其他非日志产物仍按各自约定存放。

## 测试指南

机器学习案例配套 `docs/machine_learning.md`，使用固定数据保持可重复，验证损失下降、预测及算法边界。明确区分实现验证与真实数据评估，不将训练集结果当作泛化能力证明。

`src/common/logger.h` 提供单线程双输出日志记录器；线性回归默认追加写入 `logs/linear_regression.log`。每行包含本地毫秒时间戳、调用源码文件名和行号，多行消息逐行加前缀。`make test-logger` 验证双输出一致、调用位置、多行、追加行为及失败路径，保留用户已有日志。

使用标准库 `assert` 检查示例行为，无第三方测试框架或覆盖率门槛。行为变更和缺陷修复应配套验证，覆盖有意义的结果及边界条件，避免未定义行为。验证时不要使用 `-DNDEBUG` 禁用断言。提交前运行 `make test` 和 `make lint`，报告所用编译器及缺失特性。

## 提交与拉取请求规范

目前没有提交历史可供参考。提交标题应简短、明确，以动作描述变更；可采用 Conventional Commits 规范，例如 `feat: 添加解析器入口` 或 `fix: 处理空输入`。每次提交应围绕一个明确的目的。

拉取请求应说明问题、概述解决方案，并列出已完成的验证。关联相关问题单；涉及用户可见变更时，附上截图或示例输出。明确说明新增依赖、配置变更及后续待办事项。

## 协作语言

后续对话、工作进度说明和交付说明均使用中文。代码标识符、命令及工具专有名称保留原有形式。
