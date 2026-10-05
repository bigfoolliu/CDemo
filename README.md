# 现代 C 语言学习示例

本项目通过短小、可运行的程序学习 C 语言，以最新正式标准 **C23（ISO/IEC 9899:2024）** 为目标。C2y 是下一版标准的工作名称，相关实验特性应单独讨论。

## 快速开始

需要 C 编译器和 Make，无第三方依赖。在仓库根目录运行：

```sh
make           # 编译所有示例
make run       # 按编号运行示例
make test      # 运行示例及其中的行为断言
make lint      # 严格警告与语法检查
make clean     # 删除示例可执行文件
make advanced  # 只构建并运行深入示例
make sanitize  # 对深入示例执行地址与未定义行为检查
make ml        # 构建并运行机器学习案例
```

`src/*.c` 和 `src/basics/*.c` 都有独立的 `main`；`src/modules/` 演示多个源码链接成一个程序。构建结果放在 `build/`。可用 `make CC=clang` 或 `make CC=gcc` 指定编译器；切换编译器或参数前先执行 `make clean`。

Makefile 优先选择 `-std=c23`；不支持此名称时使用 `-std=c2x`。接受标准选项并不代表完整实现 C23。示例使用真正的新语法，缺少支持时会编译失败，以便明确识别工具链限制。macOS 上名为 `gcc` 的命令也可能实际是 Apple Clang。

## 基础语法学习目录

先阅读 [C 语言基础学习手册](docs/c_basics.md)，再按下表学习。手册包含概念说明、易错点、练习及尚未提供独立示例的深入主题。

| 示例路径（相对于 `src/`） | 内容 |
| --- | --- |
| `basics/01_types.c` | 类型、声明、初始化、常量、sizeof |
| `basics/02_operators.c` | 算术、逻辑、位运算、转换、短路 |
| `basics/03_control_flow.c` | 分支、循环、break、continue |
| `basics/04_functions.c` | 函数原型、按值传参、递归、回调 |
| `basics/05_arrays_strings.c` | 数组、多维数组、字符串及缓冲区 |
| `basics/06_pointers.c` | 地址、解引用、指针算术、const |
| `basics/07_aggregate_types.c` | struct、union、enum、typedef |
| `basics/08_scope_storage.c` | 作用域、生命周期与 static |
| `basics/09_dynamic_memory.c` | 分配、扩容、释放及失败处理 |
| `basics/10_preprocessor.c` | 宏、条件编译与 inline 函数 |
| `basics/11_io_errors.c` | 文本解析、文件读写、错误处理 |
| `basics/12_generic_variadic.c` | 泛型选择与可变参数 |
| `modules/` | 头文件、翻译单元、接口与链接 |

例如，单独编译运行第一个基础示例：

```sh
make build/basics/01_types
./build/basics/01_types
```

## 深入示例

阅读 [深入 C 语言学习说明](docs/c_advanced.md)，了解接口契约、边界测试与扩展练习。

| 文件（位于 `src/advanced/`） | 核心问题 |
| --- | --- |
| `01_vector.c` | 动态数组、所有权、扩容及分配失败注入 |
| `02_parser.c` | 严格整数解析、极值与溢出前检查 |
| `03_binary_protocol.c` | 字节序、截断检测、零拷贝视图生命周期 |
| `04_callbacks.c` | 回调与上下文、策略复用、提前终止 |
| `05_atomics.c` | 双线程原子计数与 release/acquire 发布 |

线程示例采用 POSIX API，需要 macOS/Linux 等支持环境；其他深入示例使用 ISO C 接口。消毒器命令要求编译器支持 `-fsanitize=address,undefined`，生成物放在 `build/sanitize/`。

## 机器学习入门案例

`src/ml/` 提供四个纯 C 案例，使用内置数据，无第三方依赖；数学函数通过 `-lm` 链接。运行 `make ml`，配合 [中文机器学习说明](docs/machine_learning.md) 阅读算法步骤和练习。

| 文件 | 学习目标 |
| --- | --- |
| `01_linear_regression.c` | 梯度下降拟合直线、均方误差与预测 |
| `02_logistic_regression.c` | 二分类、sigmoid、交叉熵及数值稳定性 |
| `03_knn.c` | 距离排序、邻居投票与平局处理 |
| `04_kmeans.c` | 无监督聚类、中心更新与空簇处理 |

这些案例验证合成数据上的算法行为；真实模型评估需要另外划分训练、验证和测试数据。

线性回归的详细日志同时输出到屏幕和当前目录的 `linear_regression.log`，多次运行会追加。用 `less linear_regression.log` 分页查看，或运行 `./build/ml/01_linear_regression build/training.log` 指定日志文件。

## C23 新特性示例

| 文件 | 学习内容 | 建议实验 |
| --- | --- | --- |
| `src/01_bool_nullptr.c` | `bool`、`true`、`false`、`nullptr`、`static_assert` | 改变布尔状态，观察断言失败 |
| `src/02_literals.c` | 二进制字面量与数字分隔符 | 改变二进制位并计算十进制结果 |
| `src/03_bitint.c` | `_BitInt(N)` 与无符号整数回绕 | 修改位宽及初始值，预测输出 |

阅读顺序：先看注释，再预测输出，最后编译运行。测试依赖 `assert`，验证时不要设置 `-DNDEBUG`。

## 深度强化学习自动玩贪吃蛇

纯 C 手写的 `11 → 64 → 64 → 3` 神经网络使用 DQN、经验回放和 Adam，训练时复用人工游戏规则。详见 [实现讲解与运行步骤](docs/snake_ai.md)。

```sh
make snake-ai
./build/games/snake_ai train --seconds 180 --seed 42
# 将实际输出的运行目录替换到模型路径中：
./build/games/snake_ai eval --model sessions/snake-ai/run-XXXXXX/best.model
./build/games/snake_ai play --model sessions/snake-ai/run-XXXXXX/best.model
```

实验日志、CSV 指标和模型放在独立的 `sessions/snake-ai/` 目录；AI 得分与人工历史分开。训练追求分数提升，不保证每局满分。`make test-snake-ai` 验证数值计算与终端自动游戏。

## 终端小游戏

运行 `make snake` 玩实时贪吃蛇：方向键或 WASD 转向，无需回车，蛇持续自动移动；撞墙或撞到自身后一秒自动重开。界面显示当前分数、历史最高及最近五局，历史追加保存至 `snake_scores.log`。Q 或 Ctrl+C 退出。源码位于 `src/games/snake.c`，详见 [游戏说明](docs/snake.md)。适用于 macOS/Linux 交互终端，普通 `make run` 不启动游戏；`make test` 验证规则，`make test-snake-terminal` 执行伪终端集成测试（需要 Python 3）。

## 后续学习路线

1. 完成基础示例及手册练习，深入理解对象生命周期与未定义行为。
2. C23 语言特性：`constexpr`、`typeof`、`auto` 类型推导、标准属性、`#embed`。
3. C23 标准库：`<stdckdint.h>` 检查整数溢出、`<stdbit.h>` 位操作。标准库支持需要单独验证。
4. 综合练习：文件处理、错误处理和多文件组织，并使用消毒器检查内存问题。

## 参考资料

- [ISO C23 标准信息](https://www.iso.org/standard/82075.html)
- [WG14 标准委员会与公开文档](https://www.open-std.org/jtc1/sc22/wg14/)
- [Clang C 语言特性支持状态](https://clang.llvm.org/c_status.html)
