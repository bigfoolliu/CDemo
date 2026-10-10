CC = cc
.DEFAULT_GOAL := all
# 优先使用正式标准名称，旧编译器可能只接受 c2x。
CSTD := $(shell printf 'int main(void) { return 0; }\n' | $(CC) -std=c23 -x c -fsyntax-only - >/dev/null 2>&1 && echo c23 || echo c2x)
CFLAGS = -std=$(CSTD) -Wall -Wextra -Wpedantic -Werror -g
SOURCES := $(wildcard src/*.c src/basics/*.c src/advanced/*.c src/ml/*.c src/stdlib/*.c)
STDLIB_PROGRAMS := $(patsubst src/%.c,build/%,$(wildcard src/stdlib/*.c))
ML_PROGRAMS := $(patsubst src/%.c,build/%,$(wildcard src/ml/*.c))
PROGRAMS := $(patsubst src/%.c,build/%,$(SOURCES))
MODULE_SOURCES := $(wildcard src/modules/*.c)
PROGRAMS += build/modules_demo
GAME_PROGRAM := build/games/snake
AI_PROGRAM := build/games/snake_ai
AI_HEADERS := $(wildcard src/games/*.h)
DEBUG_CFLAGS = -std=$(CSTD) -Wall -Wextra -Wpedantic -Werror -O0 -g3
DEBUG_PROGRAMS := $(patsubst src/%.c,build/debug/%,$(SOURCES)) build/debug/games/snake build/debug/games/snake_ai build/debug/modules_demo build/debug/tests/snake_ai_test
GRAPHICS_CORE := src/graphics/graphics.c
GRAPHICS_HEADER := src/graphics/graphics.h
GRAPHICS_SDL := src/graphics/graphics_sdl.c
GRAPHICS_WINDOW_HEADERS := $(GRAPHICS_HEADER) src/graphics/graphics_sdl.h
PARTICLES_CORE := src/graphics/particles.c $(GRAPHICS_CORE)
PARTICLES_HEADERS := src/graphics/particles.h $(GRAPHICS_HEADER)
PARTICLES_SOURCES := src/graphics/particles_demo.c $(PARTICLES_CORE) $(GRAPHICS_SDL)
GRAPHICS_SOURCES := src/graphics/demo.c $(GRAPHICS_CORE) $(GRAPHICS_SDL)
SDL_CFLAGS = $(shell pkg-config --cflags sdl2)
SDL_LIBS = $(shell pkg-config --libs sdl2)

# 图形窗口是可选依赖；普通构建和无窗口测试不查询或链接 SDL。
.PHONY: graphics graphics-build graphics-debug check-sdl test-graphics sanitize-graphics compile-commands-graphics
.PHONY: test-graphics-window
test-graphics-window: build/tests/graphics_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/graphics_window_test

build/tests/graphics_window_test: tests/graphics_window_test.c $(GRAPHICS_SOURCES) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/graphics_window_test.c $(GRAPHICS_CORE) $(GRAPHICS_SDL) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@
check-sdl:
	@command -v pkg-config >/dev/null 2>&1 && pkg-config --exists sdl2 || { echo "需要 SDL2 与 pkg-config；macOS 可使用 brew install sdl2 pkg-config。" >&2; exit 1; }

graphics: graphics-build
	./build/graphics/demo

graphics-build: check-sdl build/graphics/demo
graphics-debug: check-sdl build/debug/graphics/demo

build/graphics/demo: $(GRAPHICS_SOURCES) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(GRAPHICS_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

build/debug/graphics/demo: $(GRAPHICS_SOURCES) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(GRAPHICS_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

build/tests/graphics_test: tests/graphics_test.c $(GRAPHICS_CORE) $(GRAPHICS_HEADER) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/graphics_test.c $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

test-graphics: build/tests/graphics_test
	./build/tests/graphics_test

sanitize-graphics:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/graphics_test.c $(GRAPHICS_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/graphics_test
	./build/sanitize/graphics_test

compile-commands-graphics: check-sdl
	python3 scripts/generate_compile_commands.py --graphics


.PHONY: particles particles-build particles-debug test-particles sanitize-particles test-particles-window
particles: particles-build
	./build/graphics/particles_demo

particles-build: check-sdl build/graphics/particles_demo
particles-debug: check-sdl build/debug/graphics/particles_demo

build/graphics/particles_demo: $(PARTICLES_SOURCES) $(PARTICLES_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) $(PARTICLES_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

build/debug/graphics/particles_demo: $(PARTICLES_SOURCES) $(PARTICLES_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(SDL_CFLAGS) $(PARTICLES_SOURCES) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

build/tests/particles_test: tests/particles_test.c $(PARTICLES_CORE) $(PARTICLES_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) tests/particles_test.c $(PARTICLES_CORE) $(LDFLAGS) $(LDLIBS) -lm -o $@

test-particles: build/tests/particles_test
	./build/tests/particles_test

sanitize-particles:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/particles_test.c $(PARTICLES_CORE) $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/particles_test
	./build/sanitize/particles_test

build/tests/particles_window_test: tests/particles_window_test.c $(PARTICLES_SOURCES) $(PARTICLES_HEADERS) $(GRAPHICS_WINDOW_HEADERS) Makefile | check-sdl
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SDL_CFLAGS) tests/particles_window_test.c $(PARTICLES_CORE) $(GRAPHICS_SDL) $(LDFLAGS) $(LDLIBS) $(SDL_LIBS) -lm -o $@

test-particles-window: build/tests/particles_window_test
	SDL_VIDEODRIVER=dummy ./build/tests/particles_window_test

.PHONY: debug compile-commands
.PHONY: stdlib
stdlib: $(STDLIB_PROGRAMS)
	@set -e; for program in $(STDLIB_PROGRAMS); do "$$program"; done
debug: $(DEBUG_PROGRAMS)

# 从实际构建命令生成数据库，避免编辑器和 Make 使用不同的参数。
compile-commands:
	python3 scripts/generate_compile_commands.py

build/debug/advanced/05_atomics: DEBUG_CFLAGS += -pthread
build/stdlib/08_math build/debug/stdlib/08_math: LDLIBS += -lm
$(patsubst src/%.c,build/debug/%,$(wildcard src/ml/*.c)) build/debug/games/snake_ai build/debug/tests/snake_ai_test: LDLIBS += -lm

build/debug/%: src/%.c $(AI_HEADERS) src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/debug/modules_demo: $(MODULE_SOURCES) src/modules/arithmetic.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $(MODULE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

build/debug/tests/snake_ai_test: tests/snake_ai_test.c src/games/snake_ai.c $(AI_HEADERS) src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(DEBUG_CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

.PHONY: all run test lint clean advanced sanitize ml snake test-snake-terminal
all: $(PROGRAMS) $(GAME_PROGRAM) $(AI_PROGRAM)

.PHONY: snake-ai
snake-ai: $(AI_PROGRAM)

build/tests/snake_ai_test: tests/snake_ai_test.c src/games/snake_ai.c $(AI_HEADERS) Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $< $(LDFLAGS) $(LDLIBS) -lm -o $@

.PHONY: test-snake-ai sanitize-snake-ai
test-snake-ai: $(AI_PROGRAM) build/tests/snake_ai_test
	./build/tests/snake_ai_test
	python3 tests/snake_ai_integration_test.py

sanitize-snake-ai:
	mkdir -p build/sanitize
	$(CC) $(CPPFLAGS) $(CFLAGS) -O1 -fsanitize=address,undefined -fno-omit-frame-pointer tests/snake_ai_test.c $(LDFLAGS) $(LDLIBS) -lm -o build/sanitize/snake_ai_test
	./build/sanitize/snake_ai_test

$(GAME_PROGRAM): src/games/snake_env.h src/games/snake_terminal.h

$(AI_PROGRAM): src/games/snake_ai.c $(AI_HEADERS) src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -O2 $< $(LDFLAGS) $(LDLIBS) -lm -o $@

snake: $(GAME_PROGRAM)
	./$(GAME_PROGRAM)

test-snake-terminal: $(GAME_PROGRAM)
	python3 tests/snake_terminal_test.py

# POSIX 线程 API 与 ISO C 原子操作分开：仅线程示例需要 -pthread。
build/advanced/05_atomics: CFLAGS += -pthread
$(ML_PROGRAMS): LDLIBS += -lm
build/ml/01_linear_regression: src/common/logger.h

ml: $(ML_PROGRAMS)
	@set -e; for program in $(ML_PROGRAMS); do "$$program"; done

advanced: $(patsubst src/%.c,build/%,$(wildcard src/advanced/*.c))
	@set -e; for program in $(patsubst src/%.c,build/%,$(wildcard src/advanced/*.c)); do "$$program"; done

# 使用独立目录，避免普通构建与消毒器构建混用。
sanitize:
	@set -e; for source in $(wildcard src/advanced/*.c); do \
		name=$$(basename "$$source" .c); mkdir -p build/sanitize; \
		$(CC) $(CPPFLAGS) $(CFLAGS) -pthread -fsanitize=address,undefined -fno-omit-frame-pointer "$$source" $(LDFLAGS) $(LDLIBS) -o "build/sanitize/$$name"; \
		"build/sanitize/$$name"; \
	done

build/%: src/%.c Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

build/modules_demo: $(MODULE_SOURCES) src/modules/arithmetic.h Makefile
	mkdir -p build
	$(CC) $(CPPFLAGS) $(CFLAGS) $(MODULE_SOURCES) $(LDFLAGS) $(LDLIBS) -o $@

run: all
	@set -e; for program in $(PROGRAMS); do echo "运行 $$program"; "$$program"; done

build/tests/logger_test: tests/logger_test.c src/common/logger.h Makefile
	mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

.PHONY: test-logger
test-logger: build/tests/logger_test
	./build/tests/logger_test

test: run $(GAME_PROGRAM) build/tests/snake_ai_test build/tests/logger_test build/tests/graphics_test build/tests/particles_test
	$(GAME_PROGRAM) --test
	./build/tests/snake_ai_test
	./build/tests/logger_test
	./build/tests/graphics_test
	./build/tests/particles_test

lint:
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(PARTICLES_CORE) tests/particles_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(GRAPHICS_CORE) tests/graphics_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only src/games/snake.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only src/games/snake_ai.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/snake_ai_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/logger_test.c
	@set -e; for source in $(SOURCES); do $(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only "$$source"; done
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(MODULE_SOURCES)

clean:
	rm -f build/graphics/particles_demo build/debug/graphics/particles_demo build/tests/particles_test build/tests/particles_window_test build/sanitize/particles_test
	rm -f build/graphics/demo build/debug/graphics/demo build/tests/graphics_test build/tests/graphics_window_test build/sanitize/graphics_test
	rm -f $(DEBUG_PROGRAMS) build/compile_commands.json
	rm -f $(PROGRAMS) $(GAME_PROGRAM) $(AI_PROGRAM)
	rm -f build/tests/snake_ai_test build/sanitize/snake_ai_test build/tests/logger_test
	rm -f $(patsubst src/advanced/%.c,build/sanitize/%,$(wildcard src/advanced/*.c))
