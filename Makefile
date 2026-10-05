CC = cc
# 优先使用正式标准名称，旧编译器可能只接受 c2x。
CSTD := $(shell printf 'int main(void) { return 0; }\n' | $(CC) -std=c23 -x c -fsyntax-only - >/dev/null 2>&1 && echo c23 || echo c2x)
CFLAGS = -std=$(CSTD) -Wall -Wextra -Wpedantic -Werror -g
SOURCES := $(wildcard src/*.c src/basics/*.c src/advanced/*.c)
PROGRAMS := $(patsubst src/%.c,build/%,$(SOURCES))
MODULE_SOURCES := $(wildcard src/modules/*.c)
PROGRAMS += build/modules_demo

.PHONY: all run test lint clean advanced sanitize
all: $(PROGRAMS)

# POSIX 线程 API 与 ISO C 原子操作分开：仅线程示例需要 -pthread。
build/advanced/05_atomics: CFLAGS += -pthread

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

test: run

lint:
	@set -e; for source in $(SOURCES); do $(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only "$$source"; done
	$(CC) $(CPPFLAGS) $(CFLAGS) -fsyntax-only $(MODULE_SOURCES)

clean:
	rm -f $(PROGRAMS)
	rm -f $(patsubst src/advanced/%.c,build/sanitize/%,$(wildcard src/advanced/*.c))
