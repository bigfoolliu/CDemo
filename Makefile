# 现代 C 学习项目的构建入口（在仓库根目录运行 make）。
# 常用命令：make / make run / make test / make lint / make debug。
# 图形窗口需显式运行，例如 make sand；默认构建与测试无需 SDL2。
# 指定工具链：make CC=clang；切换编译器或参数前先 make clean。
# 本文件直接将源码编译并链接为可执行文件，不生成中间 .o 文件。
#
# 阅读 Make 规则：
#   目标: 普通依赖 | 仅顺序依赖
# 普通依赖不存在或比目标新时重建；右侧的顺序依赖只要求先完成，
# 不因其时间戳触发重建。| check-sdl 避免伪目标导致每次重新编译。
# .PHONY 声明命令型目标，即使存在同名文件也执行对应任务。
# 命令行必须以 Tab 开头；注释放在规则外，避免混入 Shell 命令。
#
# 变量与自动变量：
#   = 延迟展开；:= 定义时立即展开；+= 追加内容。
#   $(wildcard ...) 搜索文件；$(patsubst src/%.c,build/%,...) 转换路径。
#   $@ 是当前目标，$< 是第一个依赖，$(dir $@) 是目标所在目录。
#   配方里的 $$ 留给 Shell，例如 $$program 或 $$(basename ...)。
#   @ 隐藏命令本身的回显，程序输出仍显示；set -e 使循环遇错立即退出。
#   反斜杠续行将多行配方合并为一次 Shell 执行，保留循环中的变量。
#
# 编译命令约定：CPPFLAGS 放 -I/-D 等预处理选项，CFLAGS 放编译选项；
# LDFLAGS 放链接器选项，LDLIBS 放 -lm/-pthread 等所需库或链接参数。
# 本文件未定义 CPPFLAGS/LDFLAGS/LDLIBS 的默认值，可从命令行传入。
# 库参数放在源码后；-o $@ 指定输出可执行文件，mkdir -p 可重复创建目录。
# Make 不跟踪变量值变化，因此切换参数时需要 clean 后重新构建。
# 消毒器参数：address 检查越界/释放后使用等内存错误，undefined 检查
# 有符号溢出、非法移位等未定义行为；它们用于辅助验证，不证明代码无缺陷。
# -----------------------------------------------------------------------------

# 默认使用 cc，覆盖 Make 内置编译器默认值；命令行 CC=clang 仍具有优先级。
CC = cc
# 无参数 make 执行 all，不受本文件中图形规则出现顺序影响。
.DEFAULT_GOAL := all
# 优先使用正式标准名称，旧编译器可能只接受 c2x。
# 用标准输入的小程序探测 -std=c23 是否可接受，只检查语法、不生成文件；
# 失败回退 c2x。此探测不保证编译器完整实现 C23，标准库支持仍需验证。
CSTD := $(shell printf 'int main(void) { return 0; }\n' | $(CC) -std=c23 -x c -fsyntax-only - >/dev/null 2>&1 && echo c23 || echo c2x)
# 普通构建：-std 选择标准，-Wall/-Wextra 启用常见及附加警告，
# -Wpedantic 检查严格标准符合性，-Werror 将警告视为错误，-g 保留调试信息。
# 没有显式优化等级；不要用 -DNDEBUG 禁用示例中的断言。
CFLAGS = -std=$(CSTD) -Wall -Wextra -Wpedantic -Werror -g
# 自动收集各独立 main 示例；不包含 modules、games 和 graphics，
# 这些目录需要多文件链接、特殊依赖或交互入口，分别由专用规则管理。
SOURCES := $(wildcard src/*.c src/basics/*.c src/advanced/*.c src/ml/*.c src/stdlib/*.c)
# 标准库专区可执行文件清单，例如 src/stdlib/01_stdio.c -> build/stdlib/01_stdio。
STDLIB_PROGRAMS := $(patsubst src/%.c,build/%,$(wildcard src/stdlib/*.c))
# 机器学习案例清单，用于批量运行及单独追加数学库 -lm。
ML_PROGRAMS := $(patsubst src/%.c,build/%,$(wildcard src/ml/*.c))
# 默认可自动运行的示例清单，保留 src 下的相对目录与编号。
PROGRAMS := $(patsubst src/%.c,build/%,$(SOURCES))
# 模块示例的全部翻译单元：一次链接成一个程序，而不是每个文件各自运行。
MODULE_SOURCES := $(wildcard src/modules/*.c)
# 把多文件模块程序加入可运行清单，补充前面的单文件路径映射。
PROGRAMS += build/modules_demo build/cpu/demo
# 人工贪吃蛇可执行文件；默认只构建，启动由 snake 目标显式触发。
GAME_PROGRAM := build/games/snake
# DQN 贪吃蛇可执行文件；默认不进行交互或长时间训练。
AI_PROGRAM := build/games/snake_ai
# 游戏公共头文件依赖清单；修改规则、终端或网络头文件时触发相关重建。
AI_HEADERS := $(wildcard src/games/*.h)
# 调试构建关闭优化（-O0）并增加调试信息（-g3），便于断点与变量观察。
DEBUG_CFLAGS = -std=$(CSTD) -Wall -Wextra -Wpedantic -Werror -O0 -g3
# 独立调试产物放在 build/debug/，避免与普通构建覆盖；
# 不含可选图形窗口，图形窗口分别通过 *-debug 构建。
DEBUG_PROGRAMS := $(patsubst src/%.c,build/debug/%,$(SOURCES)) build/debug/games/snake build/debug/games/snake_ai build/debug/modules_demo build/debug/tests/snake_ai_test
# 纯 C 软件绘图源码，无 SDL 依赖；被各模拟和核心测试共同复用。
GRAPHICS_CORE := src/graphics/graphics.c
# 软件画布与几何接口头文件；头文件只参与依赖判断，不直接作为源码链接。
GRAPHICS_HEADER := src/graphics/graphics.h
# 纯 C BMP 编码实现，负责文件头、颜色通道及像素布局。
BMP_CORE := src/graphics/bmp.c
# 截图路径与独占创建实现，负责 exports/ 下的安全命名与目录准备。
BMP_CAPTURE := src/graphics/bmp_capture.c
# BMP 相关接口依赖，接口修改后重建使用它们的程序。
BMP_HEADERS := src/graphics/bmp.h src/graphics/bmp_capture.h
# BMP 编码与截图管理一起链接，供离线示例及窗口截图使用。
BMP_SOURCES := $(BMP_CORE) $(BMP_CAPTURE)
# 共享 SDL 窗口、纹理呈现和资源清理实现，只用于可选窗口。
GRAPHICS_SDL := src/graphics/graphics_sdl.c
# 所有窗口公共接口依赖：画布、SDL 适配和 BMP 截图。
GRAPHICS_WINDOW_HEADERS := $(GRAPHICS_HEADER) src/graphics/graphics_sdl.h $(BMP_HEADERS)
# 曼德勃罗分形的纯 C 核心与共享绘图源码，核心测试只链接这些文件。
MANDELBROT_CORE := src/graphics/mandelbrot.c $(GRAPHICS_CORE)
# 曼德勃罗分形核心接口与画布依赖，修改头文件后重建相关目标。
MANDELBROT_HEADERS := src/graphics/mandelbrot.h $(GRAPHICS_HEADER)
# 曼德勃罗分形窗口入口 + 纯 C 核心 + 共享 SDL/BMP 模块，用于普通及调试窗口。
MANDELBROT_SOURCES := src/graphics/mandelbrot_demo.c $(MANDELBROT_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 贝塞尔曲线的纯 C 核心与共享绘图源码，核心测试只链接这些文件。
BEZIER_CORE := src/graphics/bezier.c $(GRAPHICS_CORE)
# 贝塞尔曲线核心接口与画布依赖，修改头文件后重建相关目标。
BEZIER_HEADERS := src/graphics/bezier.h $(GRAPHICS_HEADER)
# 贝塞尔曲线窗口入口 + 纯 C 核心 + 共享 SDL/BMP 模块，用于普通及调试窗口。
BEZIER_SOURCES := src/graphics/bezier_demo.c $(BEZIER_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 波纹模拟的纯 C 核心与共享绘图源码，核心测试只链接这些文件。
RIPPLE_CORE := src/graphics/ripple.c $(GRAPHICS_CORE)
# 波纹模拟核心接口与画布依赖，修改头文件后重建相关目标。
RIPPLE_HEADERS := src/graphics/ripple.h $(GRAPHICS_HEADER)
# 波纹模拟窗口入口 + 纯 C 核心 + 共享 SDL/BMP 模块，用于普通及调试窗口。
RIPPLE_SOURCES := src/graphics/ripple_demo.c $(RIPPLE_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 落沙模拟的纯 C 核心与共享绘图源码，核心测试只链接这些文件。
SAND_CORE := src/graphics/sand.c $(GRAPHICS_CORE)
# 落沙模拟核心接口与画布依赖，修改头文件后重建相关目标。
SAND_HEADERS := src/graphics/sand.h $(GRAPHICS_HEADER)
# 落沙模拟窗口入口 + 纯 C 核心 + 共享 SDL/BMP 模块，用于普通及调试窗口。
SAND_SOURCES := src/graphics/sand_demo.c $(SAND_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 迷宫生成与寻路的纯 C 核心与共享绘图源码，核心测试只链接这些文件。
MAZE_CORE := src/graphics/maze.c $(GRAPHICS_CORE)
# 迷宫生成与寻路核心接口与画布依赖，修改头文件后重建相关目标。
MAZE_HEADERS := src/graphics/maze.h $(GRAPHICS_HEADER)
# 迷宫生成与寻路窗口入口 + 纯 C 核心 + 共享 SDL/BMP 模块，用于普通及调试窗口。
MAZE_SOURCES := src/graphics/maze_demo.c $(MAZE_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 康威生命游戏的纯 C 核心与共享绘图源码，核心测试只链接这些文件。
LIFE_CORE := src/graphics/life.c $(GRAPHICS_CORE)
# 康威生命游戏核心接口与画布依赖，修改头文件后重建相关目标。
LIFE_HEADERS := src/graphics/life.h $(GRAPHICS_HEADER)
# 康威生命游戏窗口入口 + 纯 C 核心 + 共享 SDL/BMP 模块，用于普通及调试窗口。
LIFE_SOURCES := src/graphics/life_demo.c $(LIFE_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 粒子喷泉的纯 C 核心与共享绘图源码，核心测试只链接这些文件。
PARTICLES_CORE := src/graphics/particles.c $(GRAPHICS_CORE)
# 粒子喷泉核心接口与画布依赖，修改头文件后重建相关目标。
PARTICLES_HEADERS := src/graphics/particles.h $(GRAPHICS_HEADER)
# 粒子喷泉窗口入口 + 纯 C 核心 + 共享 SDL/BMP 模块，用于普通及调试窗口。
PARTICLES_SOURCES := src/graphics/particles_demo.c $(PARTICLES_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 基础二维演示入口 + 纯 C 绘图 + SDL 呈现 + BMP 导出。
GRAPHICS_SOURCES := src/graphics/demo.c $(GRAPHICS_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES)
# 延迟执行 pkg-config --cflags sdl2，取得 SDL 头文件路径与必要宏；
# 使用 = 而非 :=，让不展开本变量的普通目标无需查询 SDL。
SDL_CFLAGS = $(shell pkg-config --cflags sdl2)
# 延迟取得 SDL 链接目录与库名；check-sdl 先给出缺依赖时的明确错误。
SDL_LIBS = $(shell pkg-config --libs sdl2)

# 图形窗口是可选依赖；普通构建和无窗口测试不查询或链接 SDL。
# --- 二维图形：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: graphics graphics-build graphics-debug check-sdl test-graphics sanitize-graphics compile-commands-graphics
.PHONY: test-graphics-window
# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-graphics-window: build/tests/graphics_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/graphics_window_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/graphics_window_test: tests/graphics_window_test.c tests/bmp_window_helpers.h $(GRAPHICS_SOURCES) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/graphics_window_test.c $(GRAPHICS_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@
# 检查 pkg-config 和 SDL2 是否存在；命令失败时向标准错误输出安装提示并终止。
# @ 隐藏检查命令；该目标只检查环境，不安装软件。
check-sdl:
	@command -v pkg-config >/dev/null 2>&1 && pkg-config --exists sdl2 || { echo "需要 SDL2 与 pkg-config；macOS 可使用 brew install sdl2 pkg-config。" >&2; exit 1; }

# 先完成二维图形构建，再启动程序；交互窗口不会被默认 run 自动启动。
graphics: graphics-build
	./build/graphics/demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
graphics-build: check-sdl build/graphics/demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
graphics-debug: check-sdl build/debug/graphics/demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/demo: $(GRAPHICS_SOURCES) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(GRAPHICS_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/demo: $(GRAPHICS_SOURCES) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(GRAPHICS_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/graphics_test: tests/graphics_test.c $(GRAPHICS_CORE) $(GRAPHICS_HEADER) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/graphics_test.c $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-graphics: build/tests/graphics_test
	./build/tests/graphics_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-graphics:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/graphics_test.c $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/graphics_test
	./build/sanitize/graphics_test

# 生成包含 SDL 窗口与窗口测试的 build/compile_commands.json，供 clangd 使用。
# Python 脚本解析 Make 预演命令，本目标不启动窗口或训练。
compile-commands-graphics: check-sdl
	python3 scripts/generate_compile_commands.py --graphics


# --- 粒子喷泉：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: particles particles-build particles-debug test-particles sanitize-particles test-particles-window
# 先完成粒子喷泉构建，再启动程序；交互窗口不会被默认 run 自动启动。
particles: particles-build
	./build/graphics/particles_demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
particles-build: check-sdl build/graphics/particles_demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
particles-debug: check-sdl build/debug/graphics/particles_demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/particles_demo: $(PARTICLES_SOURCES) $(PARTICLES_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(PARTICLES_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/particles_demo: $(PARTICLES_SOURCES) $(PARTICLES_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(PARTICLES_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/particles_test: tests/particles_test.c $(PARTICLES_CORE) $(PARTICLES_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/particles_test.c $(PARTICLES_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-particles: build/tests/particles_test
	./build/tests/particles_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-particles:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/particles_test.c $(PARTICLES_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/particles_test
	./build/sanitize/particles_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/particles_window_test: tests/particles_window_test.c tests/bmp_window_helpers.h $(PARTICLES_SOURCES) $(PARTICLES_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/particles_window_test.c $(PARTICLES_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-particles-window: build/tests/particles_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/particles_window_test

# --- 康威生命游戏：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: life life-build life-debug test-life sanitize-life test-life-window
# 先完成康威生命游戏构建，再启动程序；交互窗口不会被默认 run 自动启动。
life: life-build
	./build/graphics/life_demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
life-build: check-sdl build/graphics/life_demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
life-debug: check-sdl build/debug/graphics/life_demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/life_demo: $(LIFE_SOURCES) $(LIFE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(LIFE_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/life_demo: $(LIFE_SOURCES) $(LIFE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(LIFE_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/life_test: tests/life_test.c $(LIFE_CORE) $(LIFE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/life_test.c $(LIFE_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-life: build/tests/life_test
	./build/tests/life_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-life:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/life_test.c $(LIFE_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/life_test
	./build/sanitize/life_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/life_window_test: tests/life_window_test.c tests/bmp_window_helpers.h $(LIFE_SOURCES) $(LIFE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/life_window_test.c $(LIFE_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-life-window: build/tests/life_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/life_window_test


# --- 迷宫生成与寻路：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: maze maze-build maze-debug test-maze sanitize-maze test-maze-window
# 先完成迷宫生成与寻路构建，再启动程序；交互窗口不会被默认 run 自动启动。
maze: maze-build
	./build/graphics/maze_demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
maze-build: check-sdl build/graphics/maze_demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
maze-debug: check-sdl build/debug/graphics/maze_demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/maze_demo: $(MAZE_SOURCES) $(MAZE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(MAZE_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/maze_demo: $(MAZE_SOURCES) $(MAZE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(MAZE_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/maze_test: tests/maze_test.c $(MAZE_CORE) $(MAZE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/maze_test.c $(MAZE_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-maze: build/tests/maze_test
	./build/tests/maze_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-maze:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/maze_test.c $(MAZE_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/maze_test
	./build/sanitize/maze_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/maze_window_test: tests/maze_window_test.c tests/bmp_window_helpers.h $(MAZE_SOURCES) $(MAZE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/maze_window_test.c $(MAZE_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-maze-window: build/tests/maze_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/maze_window_test


# --- 曼德勃罗分形：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: mandelbrot mandelbrot-build mandelbrot-debug test-mandelbrot sanitize-mandelbrot test-mandelbrot-window
# 先完成曼德勃罗分形构建，再启动程序；交互窗口不会被默认 run 自动启动。
mandelbrot: mandelbrot-build
	./build/graphics/mandelbrot_demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
mandelbrot-build: check-sdl build/graphics/mandelbrot_demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
mandelbrot-debug: check-sdl build/debug/graphics/mandelbrot_demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/mandelbrot_demo: $(MANDELBROT_SOURCES) $(MANDELBROT_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(MANDELBROT_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/mandelbrot_demo: $(MANDELBROT_SOURCES) $(MANDELBROT_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(MANDELBROT_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/mandelbrot_test: tests/mandelbrot_test.c $(MANDELBROT_CORE) $(MANDELBROT_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/mandelbrot_test.c $(MANDELBROT_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-mandelbrot: build/tests/mandelbrot_test
	./build/tests/mandelbrot_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-mandelbrot:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/mandelbrot_test.c $(MANDELBROT_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/mandelbrot_test
	./build/sanitize/mandelbrot_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/mandelbrot_window_test: tests/mandelbrot_window_test.c tests/bmp_window_helpers.h $(MANDELBROT_SOURCES) $(MANDELBROT_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/mandelbrot_window_test.c $(MANDELBROT_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-mandelbrot-window: build/tests/mandelbrot_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/mandelbrot_window_test


# --- 贝塞尔曲线：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: bezier bezier-build bezier-debug test-bezier sanitize-bezier test-bezier-window
# 先完成贝塞尔曲线构建，再启动程序；交互窗口不会被默认 run 自动启动。
bezier: bezier-build
	./build/graphics/bezier_demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
bezier-build: check-sdl build/graphics/bezier_demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
bezier-debug: check-sdl build/debug/graphics/bezier_demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/bezier_demo: $(BEZIER_SOURCES) $(BEZIER_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(BEZIER_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/bezier_demo: $(BEZIER_SOURCES) $(BEZIER_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(BEZIER_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/bezier_test: tests/bezier_test.c $(BEZIER_CORE) $(BEZIER_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/bezier_test.c $(BEZIER_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-bezier: build/tests/bezier_test
	./build/tests/bezier_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-bezier:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/bezier_test.c $(BEZIER_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/bezier_test
	./build/sanitize/bezier_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/bezier_window_test: tests/bezier_window_test.c tests/bmp_window_helpers.h $(BEZIER_SOURCES) $(BEZIER_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/bezier_window_test.c $(BEZIER_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-bezier-window: build/tests/bezier_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/bezier_window_test


# --- 波纹模拟：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: ripple ripple-build ripple-debug test-ripple sanitize-ripple test-ripple-window
# 先完成波纹模拟构建，再启动程序；交互窗口不会被默认 run 自动启动。
ripple: ripple-build
	./build/graphics/ripple_demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
ripple-build: check-sdl build/graphics/ripple_demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
ripple-debug: check-sdl build/debug/graphics/ripple_demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/ripple_demo: $(RIPPLE_SOURCES) $(RIPPLE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(RIPPLE_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/ripple_demo: $(RIPPLE_SOURCES) $(RIPPLE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(RIPPLE_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/ripple_test: tests/ripple_test.c $(RIPPLE_CORE) $(RIPPLE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/ripple_test.c $(RIPPLE_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-ripple: build/tests/ripple_test
	./build/tests/ripple_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-ripple:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/ripple_test.c $(RIPPLE_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ripple_test
	./build/sanitize/ripple_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/ripple_window_test: tests/ripple_window_test.c tests/bmp_window_helpers.h $(RIPPLE_SOURCES) $(RIPPLE_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/ripple_window_test.c $(RIPPLE_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-ripple-window: build/tests/ripple_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/ripple_window_test


# --- 落沙模拟：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: sand sand-build sand-debug test-sand sanitize-sand test-sand-window
# 先完成落沙模拟构建，再启动程序；交互窗口不会被默认 run 自动启动。
sand: sand-build
	./build/graphics/sand_demo

# 仅确保普通可执行文件已构建，不执行；窗口目标同时要求 check-sdl 环境检查。
sand-build: check-sdl build/graphics/sand_demo
# 仅构建独立调试可执行文件，供 VS Code 启动；窗口目标先检查 SDL 环境。
sand-debug: check-sdl build/debug/graphics/sand_demo

# 构建窗口程序并链接核心、共享呈现与截图模块；此目标不运行程序。
build/graphics/sand_demo: $(SAND_SOURCES) $(SAND_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(SAND_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建独立窗口调试版本，使用 DEBUG_CFLAGS；此目标不启动窗口。
build/debug/graphics/sand_demo: $(SAND_SOURCES) $(SAND_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(SAND_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/sand_test: tests/sand_test.c $(SAND_CORE) $(SAND_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/sand_test.c $(SAND_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-sand: build/tests/sand_test
	./build/tests/sand_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-sand:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/sand_test.c $(SAND_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/sand_test
	./build/sanitize/sand_test

# 编译 SDL 窗口集成测试：测试入口包含演示入口，命令不再单独编译 *_demo.c。
# 演示源码仍列为依赖，确保修改入口会重建；共享截图测试助手也参与依赖判断。
build/tests/sand_window_test: tests/sand_window_test.c tests/bmp_window_helpers.h $(SAND_SOURCES) $(SAND_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/sand_window_test.c $(SAND_CORE) $(GRAPHICS_SDL) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

# 使用 SDL dummy 视频驱动运行真实事件/主循环测试，无实际窗口；仍需安装并链接 SDL2。
# SDL_VIDEODRIVER 只作用于这条命令，截图测试在临时目录执行。
test-sand-window: build/tests/sand_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/sand_window_test


# --- BMP 图片导出：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: bmp bmp-build bmp-debug test-bmp sanitize-bmp
# 先构建再运行BMP 图片导出示例，无 SDL 依赖。
bmp: bmp-build
	./build/graphics/bmp_demo

# 仅构建离线 BMP 导出程序，不执行也不产生图片。
bmp-build: build/graphics/bmp_demo
# 仅构建离线 BMP 示例的调试版本，无 SDL 依赖。
bmp-debug: build/debug/graphics/bmp_demo

# 构建离线 BMP 示例，只链接纯 C 编码与绘图模块，不需要 SDL。
build/graphics/bmp_demo: src/graphics/bmp_demo.c $(BMP_SOURCES) $(BMP_HEADERS) $(GRAPHICS_CORE) $(GRAPHICS_HEADER) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) src/graphics/bmp_demo.c $(BMP_SOURCES) $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 构建离线 BMP 调试版本，使用 DEBUG_CFLAGS；不执行也不需要 SDL。
build/debug/graphics/bmp_demo: src/graphics/bmp_demo.c $(BMP_SOURCES) $(BMP_HEADERS) $(GRAPHICS_CORE) $(GRAPHICS_HEADER) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) src/graphics/bmp_demo.c $(BMP_SOURCES) $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/bmp_test: tests/bmp_test.c $(BMP_SOURCES) $(BMP_HEADERS) $(GRAPHICS_CORE) $(GRAPHICS_HEADER) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/bmp_test.c $(BMP_SOURCES) $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-bmp: build/tests/bmp_test
	./build/tests/bmp_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-bmp:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/bmp_test.c $(BMP_SOURCES) $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/bmp_test
	./build/sanitize/bmp_test


# --- 固定块内存池：启动、仅构建、调试、核心测试与可选窗口验证 ---
.PHONY: memory-pool test-memory-pool sanitize-memory-pool
# 先构建再运行固定块内存池示例，无 SDL 依赖。
memory-pool: build/advanced/06_memory_pool
	./build/advanced/06_memory_pool

# 给普通和调试内存池示例同时补充头文件依赖，实际编译复用模式规则。
build/advanced/06_memory_pool build/debug/advanced/06_memory_pool: src/common/memory_pool.h

# 构建无窗口的核心测试程序，显式链接所需实现；源码/头文件或 Makefile 修改时重建。
build/tests/memory_pool_test: tests/memory_pool_test.c src/common/memory_pool.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/memory_pool_test.c $(LDFLAGS) $(LDLIBS) -o $@

# 先构建对应测试，再执行断言验证；本目标不打开窗口。
test-memory-pool: build/tests/memory_pool_test
	./build/tests/memory_pool_test

# 单独重新编译并运行核心测试：-O1 配合地址与未定义行为消毒器，
# -fno-omit-frame-pointer 保留栈帧便于错误定位；产物不覆盖普通测试程序。
sanitize-memory-pool:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/memory_pool_test.c $(LDFLAGS) $(LDLIBS) -o build/sanitize/memory_pool_test
	./build/sanitize/memory_pool_test

# --- 图像滤镜实验室：纯 C 核心、无需窗口，显式导出 BMP ---
FILTER_CORE := src/vision/image_filters.c
FILTER_HEADERS := src/vision/image_filters.h $(GRAPHICS_HEADER)
FILTER_SOURCES := src/vision/filters_demo.c $(FILTER_CORE) $(GRAPHICS_CORE) $(BMP_SOURCES)
# 普通运行只报告响应，不写图片；filters 是显式导出入口。
PROGRAMS += build/vision/filters_demo
DEBUG_PROGRAMS += build/debug/vision/filters_demo
.PHONY: filters filters-debug test-filters sanitize-filters
filters: build/vision/filters_demo
	./build/vision/filters_demo --export

filters-debug: build/debug/vision/filters_demo

build/vision/filters_demo: $(FILTER_SOURCES) $(FILTER_HEADERS) $(BMP_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(FILTER_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o $@

build/debug/vision/filters_demo: $(FILTER_SOURCES) $(FILTER_HEADERS) $(BMP_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(FILTER_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o $@

build/tests/image_filters_test: tests/image_filters_test.c $(FILTER_CORE) $(GRAPHICS_CORE) $(FILTER_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/image_filters_test.c $(FILTER_CORE) $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

test-filters: build/tests/image_filters_test
	./build/tests/image_filters_test

sanitize-filters:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/image_filters_test.c $(FILTER_CORE) $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/image_filters_test
	./build/sanitize/image_filters_test

# --- 哈希表与词频统计：单文件入口，公共实现为头文件 ---
WORD_FREQUENCY_HEADER := src/common/word_map.h
.PHONY: word-frequency test-word-frequency sanitize-word-frequency
word-frequency: build/advanced/07_word_frequency
	./build/advanced/07_word_frequency

build/advanced/07_word_frequency build/debug/advanced/07_word_frequency: $(WORD_FREQUENCY_HEADER)

build/tests/word_frequency_test: tests/word_frequency_test.c src/advanced/07_word_frequency.c $(WORD_FREQUENCY_HEADER) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

test-word-frequency: build/tests/word_frequency_test
	./build/tests/word_frequency_test

sanitize-word-frequency:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/word_frequency_test.c $(LDFLAGS) $(LDLIBS) -o build/sanitize/word_frequency_test
	./build/sanitize/word_frequency_test

# --- 普通调试构建与编辑器代码提示 ---
.PHONY: debug compile-commands
.PHONY: stdlib
# 只构建并运行常用标准库专区；循环中的 $$program 交给 Shell 展开。
stdlib: $(STDLIB_PROGRAMS)
	@set -e; for program in $(STDLIB_PROGRAMS); do "$$program"; done
# 批量构建非窗口调试程序，只构建不执行；VS Code 的普通调试任务使用此目标。
DEBUG_PROGRAMS += build/debug/cpu/demo

debug: $(DEBUG_PROGRAMS)

# 从实际构建命令生成数据库，避免编辑器和 Make 使用不同的参数。
# 生成普通示例的编译数据库，使编辑器与 Make 使用一致参数。
# Python 3 为生成工具依赖；已显式生成的窗口记录由脚本保留。
compile-commands:
	python3 scripts/generate_compile_commands.py

# 目标专用变量：只给原子/线程示例追加 POSIX -pthread；ISO C 原子操作本身不是 POSIX 线程。
build/debug/advanced/05_atomics: DEBUG_CFLAGS += -pthread
# 目标专用链接参数：这些程序使用数学函数，追加 -lm；不让其他独立示例全局依赖数学库。
build/stdlib/08_math build/debug/stdlib/08_math: LDLIBS += -lm
# 目标专用链接参数：这些程序使用数学函数，追加 -lm；不让其他独立示例全局依赖数学库。
$(patsubst src/%.c,build/debug/%,$(wildcard src/ml/*.c)) build/debug/games/snake_ai build/debug/tests/snake_ai_test: LDLIBS += -lm

# 调试模式规则：% 保留相对目录，例如 basics/01_types。
# $< 只取入口 .c；其余是重建依赖，统一使用 DEBUG_CFLAGS 输出到 build/debug/。
build/debug/%: src/%.c $(AI_HEADERS) src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

# 多文件模块的调试版本，与普通版本使用相同源码和独立输出路径。
build/debug/modules_demo: $(MODULE_SOURCES) src/modules/arithmetic.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(MODULE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

# AI 测试的无优化调试版本；实现文件是重建依赖，不重复传给编译器。
build/debug/tests/snake_ai_test: tests/snake_ai_test.c src/games/snake_ai.c $(AI_HEADERS) src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

# --- 默认构建、运行、验证及清理入口（均为伪目标）---
.PHONY: all run test lint clean advanced sanitize ml snake test-snake-terminal
# 默认入口：构建普通示例、多文件模块、人工蛇及 AI 程序，不执行它们。
# SDL 窗口和独立测试程序不属于 all，按需由专用目标构建。
all: $(PROGRAMS) $(GAME_PROGRAM) $(AI_PROGRAM)

.PHONY: snake-ai
# 仅构建 AI 程序；训练、评估和自动游戏需要另行传递运行参数。
snake-ai: $(AI_PROGRAM)

# 构建 AI 数值测试，测试翻译单元包含 AI 实现，因此命令只传 $<，避免重复定义。
# -O2 加快测试，-lm 链接数学函数；这里仅编译，运行由测试目标触发。
build/tests/snake_ai_test: tests/snake_ai_test.c src/games/snake_ai.c $(AI_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $< $(LDFLAGS) $(LDLIBS) -lm -o $@

.PHONY: test-snake-ai sanitize-snake-ai
# 先构建 AI 与 C 测试，再验证网络数值及 Python 集成流程。
# 独立集成测试不并入默认 test，避免默认测试承担更多实验流程。
test-snake-ai: $(AI_PROGRAM) build/tests/snake_ai_test
	./build/tests/snake_ai_test
	python3 tests/snake_ai_integration_test.py

# AI 测试的地址/未定义行为检查；只运行 C 数值测试，不进行长训练或 Python 集成测试。
sanitize-snake-ai:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/snake_ai_test.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/snake_ai_test
	./build/sanitize/snake_ai_test

# 为通用 build/% 规则补充头文件依赖；这条规则没有配方，编译命令由模式规则提供。
$(GAME_PROGRAM): src/games/snake_env.h src/games/snake_terminal.h

# AI 单文件入口通过头文件包含实现，使用 -O2 降低训练计算开销。
# Makefile 或公共头文件变化也触发重建；日志模块遵守 logs/ 路径约束。
$(AI_PROGRAM): src/games/snake_ai.c $(AI_HEADERS) src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $< $(LDFLAGS) $(LDLIBS) -lm -o $@

# 构建并启动实时终端贪吃蛇，需要可交互的 POSIX 终端。
snake: $(GAME_PROGRAM)
	./$(GAME_PROGRAM)

# Python 3 伪终端测试输入、自动重开、得分持久化及终端恢复；使用临时历史文件。
test-snake-terminal: $(GAME_PROGRAM)
	python3 tests/snake_terminal_test.py

# POSIX 线程 API 与 ISO C 原子操作分开：仅线程示例需要 -pthread。
# 目标专用变量：只给原子/线程示例追加 POSIX -pthread；ISO C 原子操作本身不是 POSIX 线程。
build/advanced/05_atomics: CFLAGS += -pthread
# 目标专用链接参数：这些程序使用数学函数，追加 -lm；不让其他独立示例全局依赖数学库。
$(ML_PROGRAMS): LDLIBS += -lm
# 补充线性回归的日志头文件依赖；编译配方仍由通用 build/% 规则提供。
build/ml/01_linear_regression: src/common/logger.h

# 构建并运行机器学习案例；线性回归会追加写入 logs/ 中的持久日志。
ml: $(ML_PROGRAMS)
	@set -e; for program in $(ML_PROGRAMS); do "$$program"; done

# 阶段 0 独立入口；该示例也由 SOURCES 自动接入默认运行、检查及调试。
.PHONY: numeric-foundations test-numeric-foundations sanitize-numeric-foundations
numeric-foundations: build/ml/05_numeric_foundations
	./build/ml/05_numeric_foundations

# 示例内部断言验证手算结果、梯度与矩阵失败路径，无需外部框架。
test-numeric-foundations: numeric-foundations

# 用独立产物验证动态矩阵的内存安全及未定义行为。
sanitize-numeric-foundations:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer src/ml/05_numeric_foundations.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/05_numeric_foundations
	./build/sanitize/ml/05_numeric_foundations

# 阶段 1 独立入口；固定规模的 XOR 短训练也接入默认测试与调试。
.PHONY: xor-mlp test-xor-mlp sanitize-xor-mlp
xor-mlp: build/ml/06_xor_mlp
	./build/ml/06_xor_mlp

# 案例内部检查全部参数梯度、稳定损失与四组真值表。
test-xor-mlp: xor-mlp

# 独立消毒器产物，不覆盖普通或调试程序。
sanitize-xor-mlp:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer src/ml/06_xor_mlp.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/06_xor_mlp
	./build/sanitize/ml/06_xor_mlp

# 阶段 2 使用纯 C 绘图/BMP 多文件链接，无 SDL；默认运行不生成图片。
MLP_2D_SOURCES := src/ml/07_mlp_2d.c $(GRAPHICS_CORE) $(BMP_SOURCES)
MLP_2D_HEADERS := $(GRAPHICS_HEADER) $(BMP_HEADERS)
.PHONY: mlp-2d test-mlp-2d sanitize-mlp-2d
# 显式演示入口导出初始、训练中途及最佳参数的决策边界。
mlp-2d: build/ml/07_mlp_2d
	./build/ml/07_mlp_2d --export

# 覆盖单文件规则以链接软件画布；-O2 缩短小型训练，数学库由 ML_PROGRAMS 提供。
build/ml/07_mlp_2d: $(MLP_2D_SOURCES) $(MLP_2D_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $(MLP_2D_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

# 保留无优化独立调试程序，让前向缓存与梯度易于观察。
build/debug/ml/07_mlp_2d: $(MLP_2D_SOURCES) $(MLP_2D_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(MLP_2D_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

# 安静模式仍完成数值检查和固定短训练，不创建图片或日志。
test-mlp-2d: build/ml/07_mlp_2d
	./build/ml/07_mlp_2d --test

sanitize-mlp-2d:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer $(MLP_2D_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/07_mlp_2d
	./build/sanitize/ml/07_mlp_2d --test

# 阶段 3：纯 C CNN 与软件画布/BMP；默认测试无窗口且不产生图片。
CNN_SHAPES_SOURCES := src/ml/08_cnn_shapes.c $(GRAPHICS_CORE) $(BMP_SOURCES)
CNN_SHAPES_HEADERS := $(GRAPHICS_HEADER) $(BMP_HEADERS)
.PHONY: cnn-shapes test-cnn-shapes sanitize-cnn-shapes
cnn-shapes: build/ml/08_cnn_shapes
	./build/ml/08_cnn_shapes --export

build/ml/08_cnn_shapes: $(CNN_SHAPES_SOURCES) $(CNN_SHAPES_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $(CNN_SHAPES_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/08_cnn_shapes: $(CNN_SHAPES_SOURCES) $(CNN_SHAPES_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(CNN_SHAPES_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test-cnn-shapes: build/ml/08_cnn_shapes
	./build/ml/08_cnn_shapes --test

sanitize-cnn-shapes:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer $(CNN_SHAPES_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/08_cnn_shapes
	./build/sanitize/ml/08_cnn_shapes --test

# 阶段 4：纯 C 去噪自编码器，默认不导出；显式入口展示带噪/重建/目标。
AUTOENCODER_SOURCES := src/ml/09_denoising_autoencoder.c $(GRAPHICS_CORE) $(BMP_SOURCES)
AUTOENCODER_HEADERS := $(GRAPHICS_HEADER) $(BMP_HEADERS)
.PHONY: autoencoder test-autoencoder sanitize-autoencoder
autoencoder: build/ml/09_denoising_autoencoder
	./build/ml/09_denoising_autoencoder --export

build/ml/09_denoising_autoencoder: $(AUTOENCODER_SOURCES) $(AUTOENCODER_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $(AUTOENCODER_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/09_denoising_autoencoder: $(AUTOENCODER_SOURCES) $(AUTOENCODER_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(AUTOENCODER_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test-autoencoder: build/ml/09_denoising_autoencoder
	./build/ml/09_denoising_autoencoder --test

sanitize-autoencoder:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer $(AUTOENCODER_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/09_denoising_autoencoder
	./build/sanitize/ml/09_denoising_autoencoder --test

# 阶段 5：字符 RNN，无窗口/外部语料；显式 --log 才追加训练日志。
.PHONY: char-rnn test-char-rnn sanitize-char-rnn
char-rnn: build/ml/10_char_rnn
	./build/ml/10_char_rnn

build/ml/10_char_rnn: src/ml/10_char_rnn.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $< $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/10_char_rnn: src/ml/10_char_rnn.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

test-char-rnn: build/ml/10_char_rnn
	./build/ml/10_char_rnn --test

sanitize-char-rnn:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer src/ml/10_char_rnn.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/10_char_rnn
	./build/sanitize/ml/10_char_rnn --test

# 阶段 6：同宽 RNN/GRU 记忆比特，默认无文件输出。
.PHONY: gru-memory test-gru-memory sanitize-gru-memory
gru-memory: build/ml/11_gru_memory
	./build/ml/11_gru_memory

build/ml/11_gru_memory: src/ml/11_gru_memory.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $< $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/11_gru_memory: src/ml/11_gru_memory.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

test-gru-memory: build/ml/11_gru_memory
	./build/ml/11_gru_memory --test

sanitize-gru-memory:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer src/ml/11_gru_memory.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/11_gru_memory
	./build/sanitize/ml/11_gru_memory --test

# 阶段 7：纯 C 单层字符 Transformer，默认不创建日志或图片。
.PHONY: tiny-transformer test-tiny-transformer sanitize-tiny-transformer
tiny-transformer: build/ml/12_tiny_transformer
	./build/ml/12_tiny_transformer

build/ml/12_tiny_transformer: src/ml/12_tiny_transformer.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $< $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/12_tiny_transformer: src/ml/12_tiny_transformer.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

test-tiny-transformer: build/ml/12_tiny_transformer
	./build/ml/12_tiny_transformer --test

sanitize-tiny-transformer:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer src/ml/12_tiny_transformer.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/12_tiny_transformer
	./build/sanitize/ml/12_tiny_transformer --test

# VAE：二维潜在空间和图形生成；显式入口导出BMP，默认测试不写文件。
VAE_SOURCES := src/ml/13_vae_shapes.c $(GRAPHICS_CORE) $(BMP_SOURCES)
VAE_HEADERS := $(GRAPHICS_HEADER) $(BMP_HEADERS) src/common/logger.h
.PHONY: vae test-vae sanitize-vae
vae: build/ml/13_vae_shapes
	./build/ml/13_vae_shapes --export

build/ml/13_vae_shapes: $(VAE_SOURCES) $(VAE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $(VAE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/13_vae_shapes: $(VAE_SOURCES) $(VAE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(VAE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test-vae: build/ml/13_vae_shapes
	./build/ml/13_vae_shapes --test

sanitize-vae:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer $(VAE_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/13_vae_shapes
	./build/sanitize/ml/13_vae_shapes --test

# GAN：八簇二维点分布与对抗训练；显式入口导出BMP，默认测试不写文件。
GAN_SOURCES := src/ml/14_gan_2d.c $(GRAPHICS_CORE) $(BMP_SOURCES)
GAN_HEADERS := $(GRAPHICS_HEADER) $(BMP_HEADERS) src/common/logger.h
.PHONY: gan test-gan sanitize-gan
gan: build/ml/14_gan_2d
	./build/ml/14_gan_2d --export

build/ml/14_gan_2d: $(GAN_SOURCES) $(GAN_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $(GAN_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/14_gan_2d: $(GAN_SOURCES) $(GAN_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(GAN_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

test-gan: build/ml/14_gan_2d
	./build/ml/14_gan_2d --test

sanitize-gan:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer $(GAN_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/14_gan_2d
	./build/sanitize/ml/14_gan_2d --test

# 阶段 9.1：多项式朴素贝叶斯，复用词频哈希表；固定小数据可快速完成。
NAIVE_BAYES_HEADER := src/common/word_map.h
.PHONY: naive-bayes test-naive-bayes sanitize-naive-bayes
naive-bayes: build/ml/15_naive_bayes
	./build/ml/15_naive_bayes

build/ml/15_naive_bayes build/debug/ml/15_naive_bayes: $(NAIVE_BAYES_HEADER)

build/tests/naive_bayes_test: tests/naive_bayes_test.c src/ml/15_naive_bayes.c $(NAIVE_BAYES_HEADER) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -lm -o $@

test-naive-bayes: build/tests/naive_bayes_test
	./build/tests/naive_bayes_test

sanitize-naive-bayes:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/naive_bayes_test.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/naive_bayes_test
	./build/sanitize/ml/naive_bayes_test

# 阶段 9.2：二维决策树；仅显式入口导出BMP，默认运行与测试不写图片。
DECISION_TREE_SOURCES := src/ml/16_decision_tree.c $(GRAPHICS_CORE) $(BMP_SOURCES)
DECISION_TREE_HEADERS := $(GRAPHICS_HEADER) $(BMP_HEADERS)
.PHONY: decision-tree test-decision-tree sanitize-decision-tree
decision-tree: build/ml/16_decision_tree
	./build/ml/16_decision_tree --export

build/ml/16_decision_tree: $(DECISION_TREE_SOURCES) $(DECISION_TREE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DECISION_TREE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/ml/16_decision_tree: $(DECISION_TREE_SOURCES) $(DECISION_TREE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(DECISION_TREE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/tests/decision_tree_test: tests/decision_tree_test.c $(DECISION_TREE_SOURCES) $(DECISION_TREE_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/decision_tree_test.c $(GRAPHICS_CORE) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o $@

test-decision-tree: build/tests/decision_tree_test
	./build/tests/decision_tree_test

sanitize-decision-tree:
	mkdir -p build/sanitize/ml
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/decision_tree_test.c $(GRAPHICS_CORE) $(BMP_SOURCES) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/ml/decision_tree_test
	./build/sanitize/ml/decision_tree_test

# 教学 CPU 多文件链接；核心无需第三方依赖，追踪由宿主日志记录器处理。
CPU_SOURCES := src/cpu/demo.c src/cpu/cpu.c
CPU_HEADERS := src/cpu/cpu.h src/cpu/programs.h src/common/logger.h
.PHONY: cpu test-cpu sanitize-cpu
cpu: build/cpu/demo
	./build/cpu/demo

build/cpu/demo: $(CPU_SOURCES) $(CPU_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(CPU_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/cpu/demo: $(CPU_SOURCES) $(CPU_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(CPU_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/tests/cpu_test: tests/cpu_test.c src/cpu/cpu.c $(CPU_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/cpu_test.c src/cpu/cpu.c $(LDFLAGS) $(LDLIBS) -o $@

test-cpu: build/tests/cpu_test
	./build/tests/cpu_test

# 消毒器只运行核心测试，不产生追踪日志。
sanitize-cpu:
	@mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsanitize=address,undefined -fno-omit-frame-pointer tests/cpu_test.c src/cpu/cpu.c $(LDFLAGS) $(LDLIBS) -o build/sanitize/cpu_test
	./build/sanitize/cpu_test

# 只构建并运行深入示例，自动纳入 src/advanced/ 新增的独立入口。
advanced: $(patsubst src/%.c,build/%,$(wildcard src/advanced/*.c))
	@set -e; for program in $(patsubst src/%.c,build/%,$(wildcard src/advanced/*.c)); do "$$program"; done

# 使用独立目录，避免普通构建与消毒器构建混用。
# 深入示例的批量消毒器验证，产物独立放在 build/sanitize/。
# 此循环统一带 -pthread，以涵盖线程示例；不表示所有示例都使用线程。
# $$(basename ...) 交给 Shell 提取文件名，编译或运行失败即退出。
sanitize:
	@set -e; for source in $(wildcard src/advanced/*.c); do \
		name=$$(basename "$$source" .c); mkdir -p build/sanitize; \
		$(CC) $(CPPFLAGS) $(CFLAGS) -pthread -fsanitize=address,undefined -fno-omit-frame-pointer "$$source" $(LDFLAGS) $(LDLIBS) -o "build/sanitize/$$name"; \
		"build/sanitize/$$name"; \
	done

# 普通单文件模式规则：如 src/basics/01_types.c -> build/basics/01_types。
# 目标比源码或 Makefile 旧时重建；特殊示例可在其他规则补充依赖与链接参数。
build/%: src/%.c Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

# 多文件模块示例：所有 .c 一起链接，共享头文件只用于判断是否需要重建。
build/modules_demo: $(MODULE_SOURCES) src/modules/arithmetic.h Makefile
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MODULE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

# 先完成默认构建，再按 PROGRAMS 顺序运行非交互示例（通常按路径和编号排序）。
# 不启动人工蛇、AI 或 SDL 窗口；set -e 保证任一示例失败便停止循环。
run: all
	@set -e; for program in $(PROGRAMS); do echo "运行 $$program"; "$$program"; done

# 日志记录器为头文件实现，编译单个测试入口；测试验证双输出、前缀与失败路径。
build/tests/logger_test: tests/logger_test.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

.PHONY: test-logger
# 运行日志记录器测试，覆盖双输出一致、时间/源码位置、多行、追加和错误处理。
test-logger: build/tests/logger_test
	./build/tests/logger_test

# 默认验证入口：先运行普通示例，再执行蛇规则、AI 数值、日志与各纯 C 核心测试。
# 断言必须保持启用；不打开窗口，也不包含长训练、伪终端及 SDL 窗口集成测试。
test: build/tests/decision_tree_test build/tests/naive_bayes_test build/tests/word_frequency_test build/tests/image_filters_test build/tests/cpu_test run $(GAME_PROGRAM) build/tests/snake_ai_test build/tests/logger_test build/tests/graphics_test build/tests/particles_test build/tests/life_test build/tests/bmp_test build/tests/memory_pool_test build/tests/maze_test build/tests/mandelbrot_test build/tests/bezier_test build/tests/ripple_test build/tests/sand_test
	./build/tests/decision_tree_test
	./build/tests/naive_bayes_test
	$(GAME_PROGRAM) --test
	./build/tests/snake_ai_test
	./build/tests/logger_test
	./build/tests/graphics_test
	./build/tests/particles_test
	./build/tests/life_test
	./build/tests/bmp_test
	./build/tests/memory_pool_test
	./build/tests/maze_test
	./build/tests/mandelbrot_test
	./build/tests/bezier_test
	./build/tests/ripple_test
	./build/tests/sand_test
	./build/tests/cpu_test
	./build/tests/image_filters_test
	./build/tests/word_frequency_test

# 以严格警告执行 -fsyntax-only，不链接、不生成可执行文件，也不验证运行行为。
# 覆盖普通示例、纯 C 图形核心及核心测试，无需 SDL；窗口编译由专用目标验证。
lint:
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(DECISION_TREE_SOURCES) tests/decision_tree_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/naive_bayes_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/word_frequency_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(FILTER_SOURCES) tests/image_filters_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(CPU_SOURCES) tests/cpu_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(SAND_CORE) tests/sand_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(RIPPLE_CORE) tests/ripple_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(BEZIER_CORE) tests/bezier_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(MANDELBROT_CORE) tests/mandelbrot_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(MAZE_CORE) tests/maze_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/memory_pool_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(BMP_SOURCES) src/graphics/bmp_demo.c tests/bmp_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(LIFE_CORE) tests/life_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(PARTICLES_CORE) tests/particles_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(GRAPHICS_CORE) tests/graphics_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only src/games/snake.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only src/games/snake_ai.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/snake_ai_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/logger_test.c
	@set -e; for source in $(SOURCES); do $(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only "$$source"; done
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(MODULE_SOURCES)

# 仅删除规则列出的构建产物与编译数据库；rm -f 允许文件不存在。
# 不删除 logs/、exports/、sessions/ 或用户得分文件，也不递归删除整个 build/。
# 自定义预览文件及这里未列出的生成物会保留。
clean:
	rm -f build/tests/decision_tree_test build/sanitize/ml/decision_tree_test
	rm -f build/tests/naive_bayes_test build/sanitize/ml/naive_bayes_test
	rm -f build/tests/word_frequency_test build/sanitize/word_frequency_test
	rm -f build/tests/image_filters_test build/sanitize/image_filters_test
	rm -f build/sanitize/ml/14_gan_2d
	rm -f build/sanitize/ml/13_vae_shapes
	rm -f build/sanitize/ml/12_tiny_transformer
	rm -f build/sanitize/ml/11_gru_memory
	rm -f build/sanitize/ml/10_char_rnn
	rm -f build/tests/cpu_test build/sanitize/cpu_test
	rm -f build/sanitize/ml/09_denoising_autoencoder
	rm -f build/sanitize/ml/08_cnn_shapes
	rm -f build/sanitize/ml/07_mlp_2d
	rm -f build/sanitize/ml/06_xor_mlp
	rm -f build/sanitize/ml/05_numeric_foundations
	rm -f build/graphics/mandelbrot_demo build/debug/graphics/mandelbrot_demo build/tests/mandelbrot_test build/tests/mandelbrot_window_test build/sanitize/mandelbrot_test
	rm -f build/graphics/bezier_demo build/debug/graphics/bezier_demo build/tests/bezier_test build/tests/bezier_window_test build/sanitize/bezier_test
	rm -f build/graphics/ripple_demo build/debug/graphics/ripple_demo build/tests/ripple_test build/tests/ripple_window_test build/sanitize/ripple_test
	rm -f build/graphics/sand_demo build/debug/graphics/sand_demo build/tests/sand_test build/tests/sand_window_test build/sanitize/sand_test
	rm -f build/graphics/maze_demo build/debug/graphics/maze_demo build/tests/maze_test build/tests/maze_window_test build/sanitize/maze_test
	rm -f build/tests/memory_pool_test build/sanitize/memory_pool_test
	rm -f build/graphics/bmp_demo build/debug/graphics/bmp_demo build/tests/bmp_test build/sanitize/bmp_test
	rm -f build/graphics/life_demo build/debug/graphics/life_demo build/tests/life_test build/tests/life_window_test build/sanitize/life_test
	rm -f build/graphics/particles_demo build/debug/graphics/particles_demo build/tests/particles_test build/tests/particles_window_test build/sanitize/particles_test
	rm -f build/graphics/demo build/debug/graphics/demo build/tests/graphics_test build/tests/graphics_window_test build/sanitize/graphics_test
	rm -f $(DEBUG_PROGRAMS) build/compile_commands.json
	rm -f $(PROGRAMS) $(GAME_PROGRAM) $(AI_PROGRAM)
	rm -f build/tests/snake_ai_test build/sanitize/snake_ai_test build/tests/logger_test
	rm -f $(patsubst src/advanced/%.c,build/sanitize/%,$(wildcard src/advanced/*.c))
