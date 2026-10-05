CC = cc
# 优先使用正式标准名称，旧编译器可能只接受 c2x。
CSTD := $(shell printf 'int main(void) { return 0; }\n' | $(CC) -std=c23 -x c -fsyntax-only - >/dev/null 2>&1 && echo c23 || echo c2x)
CFLAGS = -std=$(CSTD) -Wall -Wextra -Wpedantic -Werror -g
SOURCES := $(wildcard src/*.c src/basics/*.c)
PROGRAMS := $(patsubst src/%.c,build/%,$(SOURCES))
MODULE_SOURCES := $(wildcard src/modules/*.c)
PROGRAMS += build/modules_demo

.PHONY: all run test lint clean
all: $(PROGRAMS)

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
