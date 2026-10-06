# C 常用标准库学习专区

按“用途 → 函数声明 → 正常用法 → 错误及边界 → 小练习”学习，每次掌握一个主题。先知道该查哪个头文件，再理解 API 的契约。标准库接口还包括类型和宏，不需要一次背下全部名字。

## 运行与调试

```sh
make stdlib                         # 构建并运行本专区全部示例
make build/stdlib/02_string         # 只构建字符串示例
./build/stdlib/02_string            # 运行单个示例
make debug compile-commands        # 调试构建与代码提示
make test                          # 包含本专区的行为断言
make lint                          # 严格警告与语法检查
```

源码在 `src/stdlib/`，每个文件有独立 `main`，使用 ISO C 接口，无第三方依赖。数学示例自动链接 `-lm`。文件示例使用 `tmpfile`，关闭后自动删除，不覆盖用户文件；无需交互输入，也不新增持久日志。断言用于验证内部结果，资源失败则明确报错，运行时不要使用 `-DNDEBUG`。

VS Code 选择 `C: stdlib/对应文件.c`，设置断点后按 F5。用 F10 观察返回值、数组、指针和 `errno`；默认不在程序入口暂停，短程序没有断点时会直接结束。函数注释注明参数长度、输出修改和资源所有权。

## 头文件导航

下面按用途列出 C23 标准头文件。**本阶段只实现后面的十个常用主题**，其余作为导航，不表示本机完整支持。

| 用途 | 头文件 | 学习安排 |
| --- | --- | --- |
| 输入输出 | `<stdio.h>` | 文件与格式化 |
| 通用工具 | `<stdlib.h>` | 内存、数值转换、排序搜索、退出 |
| 字符串与字节 | `<string.h>` | 长度、比较、查找、复制 |
| 字符分类 | `<ctype.h>` | 分类与大小写转换 |
| 类型与范围 | `<stddef.h>`、`<stdint.h>`、`<inttypes.h>`、`<limits.h>`、`<float.h>` | 类型、格式和边界 |
| 错误与断言 | `<errno.h>`、`<assert.h>` | 转换错误与内部验证 |
| 数学 | `<math.h>`、`<tgmath.h>`、`<complex.h>`、`<fenv.h>` | 本阶段仅常用实数数学 |
| 时间与可变参数 | `<time.h>`、`<stdarg.h>` | 时间、格式化与参数列表 |
| 本地化与文本编码 | `<locale.h>`、`<uchar.h>`、`<wchar.h>`、`<wctype.h>` | 后续主题 |
| 并发 | `<stdatomic.h>`、`<threads.h>` | 后续主题 |
| 信号与非局部跳转 | `<signal.h>`、`<setjmp.h>` | 后续主题 |
| C23 位操作与受检算术 | `<stdbit.h>`、`<stdckdint.h>` | 后续单独验证支持 |
| 历史兼容头文件 | `<stdbool.h>`、`<stdalign.h>`、`<stdnoreturn.h>`、`<iso646.h>` | 了解旧代码用途，不混淆兼容宏与 C23 关键字 |

项目中的 `unistd.h`、`pthread.h`、`termios.h` 和 `sys/stat.h` 属于 POSIX 接口，不属于 ISO C 标准库。本专区不混入终端控制、目录创建或网络 API。

## 示例与常用 API 速查

下表列出本次实际演示的主要 API，不是各头文件的完整接口清单。

| 示例 | 主要 API / 类型 / 宏 | 观察重点 |
| --- | --- | --- |
| [01_stdio.c](../src/stdlib/01_stdio.c) | `snprintf`、`tmpfile`、`fputs`、`fgets`、`fread`、`fwrite`、`fseek`、`fgetc`、`feof`、`ferror`、`clearerr`、`fclose` | 截断、读写计数、EOF、更新流定位 |
| [02_string.c](../src/stdlib/02_string.c) | `strlen`、`strcmp`、`strncmp`、`strchr`、`strstr`、`memcpy`、`memmove`、`memset`、`memcmp`、`memchr` | 终止符、字节长度、重叠复制 |
| [03_memory.c](../src/stdlib/03_memory.c) | `malloc`、`calloc`、`realloc`、`free` | 初始化、失败、所有权 |
| [04_conversion.c](../src/stdlib/04_conversion.c) | `strtol`、`strtod`、`errno`、`ERANGE`、`INT_MAX`、`LONG_MAX` | 完整消费、范围、输出保留 |
| [05_ctype.c](../src/stdlib/05_ctype.c) | `isdigit`、`isalpha`、`isalnum`、`isspace`、`isxdigit`、`ispunct`、`tolower`、`toupper` | 参数域、真值、大小写 |
| [06_types_limits.c](../src/stdlib/06_types_limits.c) | `size_t`、`ptrdiff_t`、`offsetof`、`uint_least32_t`、`PRIuLEAST32`、`SIZE_MAX`、`CHAR_BIT`、`DBL_MAX`、`DBL_EPSILON` | 可移植类型、格式、范围 |
| [07_sort_search.c](../src/stdlib/07_sort_search.c) | `qsort`、`bsearch` | 比较器、极值、重复值、未命中 |
| [08_math.c](../src/stdlib/08_math.c) | `sqrt`、`hypot`、`exp`、`log`、`pow`、`fabs`、`floor`、`ceil`、`round`、`isfinite`、`isnan` | 定义域、误差、NaN 与无穷 |
| [09_time.c](../src/stdlib/09_time.c) | `time`、`localtime`、`strftime`、`difftime`、`timespec_get`、`clock` | 时区、库管理存储、CPU 时间 |
| [10_stdarg.c](../src/stdlib/10_stdarg.c) | `va_list`、`va_start`、`va_copy`、`va_end`、`vsnprintf` | 参数提升、列表消费、类型匹配 |

## 十个主题的学习方法

### 1. 输入输出：先看返回值

`snprintf(buffer, capacity, format, ...)` 的容量包含终止符；返回非负的所需字符数，不含终止符。先检查负值，再比较是否大于等于容量，判断截断。`fgets` 不保证一次读取完整一行，本例使用能容纳整行的缓冲区。`fread/fwrite` 返回完成的**元素数**，不是总字节数。

`fgetc` 的结果存入 `int`，读取失败后通过 `feof/ferror` 区分结束和错误。更新流切换读写需遵守刷新或定位规则，本例使用 `fseek`。`fopen` 基础流程可结合 `src/basics/11_io_errors.c` 阅读。练习：缩小行缓冲区，并加入最后一行没有换行符的文本。

### 2. 字符串与字节：明确长度单位

`strlen` 不含终止符；`strcmp/strncmp/memcmp` 只保证结果的正、零、负。`strchr/strstr` 返回输入中的借用指针或空指针。`memcpy` 不允许重叠，重叠区域用 `memmove`；两者都不自动补终止符。`memset` 填充字节，不是任意类型的赋值。UTF-8 的 `strlen` 统计字节，不是字符数。练习：删除首字符，判断移动长度是否包含终止符。

### 3. 内存：跟踪所有权

`malloc` 不初始化内容，`calloc` 清零字节，不能推广为浮点数或指针的语义零。先检查分配大小乘法溢出，用临时指针接收 `realloc`：失败时原分配仍有效，成功后不得再使用原指针，新增部分需初始化。不要将零大小传给 C23 的 `realloc`。本例不通过耗尽内存模拟失败，已有动态数组示例提供分配失败注入。练习：缩小数组并确认保留数据，每条退出路径都释放资源。

### 4. 数值转换：检查完整输入

调用 `strtol(text, &end, base)` 前清零 `errno`，检查是否读到数字、`end` 是否指向文本末尾及 `ERANGE`，转成 `int` 前再检查整数范围。本例允许前导空白和符号、拒绝尾随空白，这是示例接口的选择。`strtod` 可能接受 NaN 或无穷，本例额外检查 `isfinite`。浮点转换受 locale 影响，示例保持默认 C locale。练习：加入 `INT_MIN`、尾随空白和不同进制验证，不使用 `atoi` 判断错误。

### 5. 字符分类：参数必须合法

`is*` 的真值是非零，不一定为 1。字符串中的 `char` 先转为 `unsigned char`；来自 `fgetc` 的值先判断 EOF，不能把 EOF 先转为字节。字节分类和大小写转换不等于 Unicode 文本处理。练习：统计数字、空白和字母，并加入空文本及高位字节输入。

### 6. 类型与范围：格式匹配类型

`size_t` 用 `%zu`，`ptrdiff_t` 用 `%td`，宽度整数使用 `<inttypes.h>` 格式宏。数组指针相减必须位于同一数组范围内；`offsetof` 不意味着结构没有填充。`DBL_EPSILON` 描述 1 附近的浮点间距，不是所有数量级的通用误差阈值。练习：改变结构成员顺序并观察偏移，解释为什么不硬编码 int 位数。

### 7. 排序搜索：比较规则一致

比较器不要直接返回 `a - b`，极值可能导致有符号溢出。`bsearch` 要求数组按同一比较规则排序，未命中返回空指针；重复值不保证找到哪个位置，`qsort` 不保证稳定性。练习：排序带分数和编号的记录，同分时按编号排序。

### 8. 数学：先校验定义域

本例先拒绝负数和非有限值，再调用 `sqrt`。真实数学错误可能通过 `math_errhandling` 指定的 errno 或浮点异常机制报告，后续再学习。近似计算使用与问题尺度相符的误差容限，本例固定的小数值使用绝对误差。练习：比较负数的 `floor/ceil/round`，解释机器学习中检查 `isfinite` 的原因。

### 9. 时间：明确测量对象

`time` 提供日历时间，`localtime` 返回库管理的存储，本例立即复制。`strftime` 失败或空间不足时返回 0。`clock / CLOCKS_PER_SEC` 是进程 CPU 时间，不是等待时长；日历时间也不保证单调。本例不依赖固定时区、日期或执行速度。练习：添加 `gmtime`，立即复制结果，再比较 UTC 和本地时间。

### 10. 可变参数：调用者负责类型契约

`float` 传入 `...` 时提升为 `double`，小整数按整数提升规则处理。格式必须匹配实际类型。读取参数会推进 `va_list`，复用需要 `va_copy`，每次初始化配套 `va_end`。示例先计算所需长度，容量不足时保留输出。练习：扩展不同类型的格式，再阅读项目日志记录器的动态格式化。

## 查阅方式与覆盖边界

每学一个 API，记录用途、参数、返回值、所有权、失败方式和标准版本六项。本机可尝试 `man 3 strtol` 或 `man 3 memcpy`；系统手册可能包含 POSIX 或平台扩展，不能将其中全部接口视为 ISO C。

[WG14 C23 工作草案 N3096](https://www.open-std.org/jtc1/sc22/wg14/www/docs/n3096.pdf) 第 7 章和附录 B 可用于核对接口契约；它是公开草案，不是正式出版文本。本专区不声称穷尽全部 API。新增 C23 库接口、线程、宽字符、本地化、信号和非局部跳转留待后续独立示例。
