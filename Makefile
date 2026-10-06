CC = cc
# 优先使用正式标准名称，旧编译器可能只接受 c2x。
CSTD := $(shell printf 'int main(void) { return 0; }\n' | $(CC) -std=c23 -x c -fsyntax-only - >/dev/null 2>&1 && echo c23 || echo c2x)
CFLAGS = -std=$(CSTD) -Wall -Wextra -Wpedantic -Werror -g
SOURCES := $(wildcard src/*.c src/basics/*.c src/advanced/*.c src/ml/*.c)
ML_PROGRAMS := $(patsubst src/%.c,build/%,$(wildcard src/ml/*.c))
PROGRAMS := $(patsubst src/%.c,build/%,$(SOURCES))
MODULE_SOURCES := $(wildcard src/modules/*.c)
PROGRAMS += build/modules_demo
GAME_PROGRAM := build/games/snake
AI_PROGRAM := build/games/snake_ai
AI_HEADERS := $(wildcard src/games/*.h)

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

test: run $(GAME_PROGRAM) build/tests/snake_ai_test build/tests/logger_test
	$(GAME_PROGRAM) --test
	./build/tests/snake_ai_test
	./build/tests/logger_test

lint:
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only src/games/snake.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only src/games/snake_ai.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/snake_ai_test.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only tests/logger_test.c
	@set -e; for source in $(SOURCES); do $(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only "$$source"; done
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(MODULE_SOURCES)

clean:
	rm -f $(PROGRAMS) $(GAME_PROGRAM) $(AI_PROGRAM)
	rm -f build/tests/snake_ai_test build/sanitize/snake_ai_test build/tests/logger_test
	rm -f $(patsubst src/advanced/%.c,build/sanitize/%,$(wildcard src/advanced/*.c))
