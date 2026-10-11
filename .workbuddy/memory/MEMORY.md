# 项目长期记忆（CDemo）

## 代码格式化规范

- 格式化工具 = 项目根 `.clang-format`（LLVM 派生：4 空格、函数左花括号换行、ColumnLimit 100、ReflowComments false、SortIncludes Never）。
- 参考基准：`src/games/snake_ai.c`（已验证 clang-format 合规）。其风格 = 结构体/枚举成员逐行、紧凑简单初始化、4 空格缩进、强制花括号、单变量声明。
- 命令行：`clang-format -i --style=file <files>`；校验用 `clang-format --style=file <f> | diff -q <f> -`（本机 clang-format 无 `--check`）。

## clang-format 在本项目的已知坑

- **C23 数字分隔符**（`0b1010'0101` / `1'000'000`）会被误插空格破坏 → 含此类字面量的文件必须手工格式化，禁用 clang-format。
- **长字符串字面量拆分**（`BreakStringLiterals` 默认 true）在数组/结构体初始化器里会触发 `-Wstring-concatenation` 且被 `-Werror` 升级为编译错误 → 用 `// clang-format off/on` 包裹该字面量，保留单个长字符串。

## 目录约定

- `src/games/` 不纳入常规格式化/自动修改（用户按需在 games 内单独维护）。
- `tests/` 为独立测试目录，不属于 src 常规格式化范围。
- 日志写入须落在仓库根 `logs/` 下（见 AGENTS.md 约束）。
