/** @file @brief 内存池边界、生命周期、对齐和失败安全断言。 */
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* 只在这个测试翻译单元替换分配调用，不改变生产接口，也不真的耗尽内存。 */
static bool fail_malloc, fail_calloc;
/** @brief 注入 malloc 失败。 @param size 系统分配字节数。 @return 内存或 nullptr。 */
static void *test_malloc(size_t size)
{
    return fail_malloc ? nullptr : malloc(size);
}
/**
 * @brief 注入 calloc 失败，验证第二次分配失败时的清理。
 * @param count 元素数。
 * @param size 每个元素字节数。
 * @return 内存或 nullptr。
 */
static void *test_calloc(size_t count, size_t size)
{
    return fail_calloc ? nullptr : calloc(count, size);
}
#define malloc test_malloc
#define calloc test_calloc
#include "../src/common/memory_pool.h"
#undef malloc
#undef calloc

/** @brief 参数、舍入溢出、乘法溢出、失败注入与重复初始化都不破坏状态。 */
static void test_initialization(void)
{
    MemoryPool pool = {0};
    assert(!memory_pool_init(nullptr, 8, 2));
    assert(!memory_pool_init(&pool, 0, 2) && !memory_pool_init(&pool, 8, 0));
    assert(!memory_pool_init(&pool, SIZE_MAX, 1));
    assert(!memory_pool_init(&pool, 8, SIZE_MAX));
    fail_malloc = true;
    assert(!memory_pool_init(&pool, 8, 2));
    fail_malloc = false;
    fail_calloc = true;
    assert(!memory_pool_init(&pool, 8, 2));
    fail_calloc = false;
    assert(pool.storage == nullptr && pool.allocated == nullptr && pool.capacity == 0);
    assert(memory_pool_init(&pool, 1, 2));
    assert(pool.stride >= sizeof(size_t) && pool.stride % alignof(max_align_t) == 0);
    unsigned char *storage = pool.storage;
    assert(!memory_pool_init(&pool, 16, 3) && pool.storage == storage && pool.capacity == 2);
    memory_pool_destroy(&pool);
    memory_pool_destroy(&pool);
    assert(memory_pool_available(&pool) == 0 && memory_pool_alloc(&pool, 1) == nullptr);
    assert(!memory_pool_release(&pool, nullptr));
    assert(memory_pool_init(&pool, 8, 2));
    memory_pool_destroy(&pool);
    memory_pool_destroy(nullptr);
}

/** @brief 校验外来、其他池、内部偏移和重复释放，并验证 LIFO 复用。 */
static void test_release_and_reuse(void)
{
    MemoryPool a = {0}, b = {0};
    assert(memory_pool_init(&a, 32, 3) && memory_pool_init(&b, 32, 1));
    void *p[3];
    for (size_t i = 0; i < 3; ++i) {
        p[i] = memory_pool_alloc(&a, 32);
        assert(p[i] != nullptr);
        memset(p[i], (int)i + 1, 32);
        for (size_t j = 0; j < i; ++j) { assert(p[i] != p[j]); }
    }
    assert(memory_pool_alloc(&a, 32) == nullptr && a.in_use == 3);
    assert(memory_pool_alloc(&a, 0) == nullptr && memory_pool_alloc(&a, 33) == nullptr);
    void *foreign = memory_pool_alloc(&b, 8);
    int local = 0;
    assert(!memory_pool_release(&a, foreign) && !memory_pool_release(&a, &local));
    assert(!memory_pool_release(&a, (unsigned char *)p[0] + 1));
    assert(a.in_use == 3 && a.free_head == SIZE_MAX);
    assert(memory_pool_release(&a, nullptr) && a.in_use == 3);
    assert(memory_pool_release(&a, p[0]) && memory_pool_release(&a, p[2]));
    size_t head = a.free_head;
    assert(!memory_pool_release(&a, p[0]) && a.free_head == head && a.in_use == 1);
    assert(memory_pool_alloc(&a, 4) == p[2]);
    assert(memory_pool_alloc(&a, 4) == p[0]);
    /* 归还其他块不能破坏仍被占用块的载荷。 */
    for (size_t i = 0; i < 32; ++i) { assert(((unsigned char *)p[1])[i] == 2); }
    for (size_t i = 0; i < 3; ++i) { assert(memory_pool_release(&a, p[i])); }
    assert(memory_pool_available(&a) == 3);
    memory_pool_destroy(&a);
    /* destroy 也允许尚有对象未归还，但这些地址随后都失效，不再读取或比较。 */
    memory_pool_destroy(&b);
}

/** @brief 所有块都对齐；同一地址支持先后承载不同类型的已初始化对象。 */
static void test_alignment_and_types(void)
{
    typedef struct { long double value; void *pointer; } AlignedObject;
    MemoryPool pool = {0};
    assert(memory_pool_init(&pool, sizeof(AlignedObject), 7));
    AlignedObject *objects[7];
    for (size_t i = 0; i < 7; ++i) {
        objects[i] = memory_pool_alloc(&pool, sizeof *objects[i]);
        assert(objects[i] != nullptr);
        assert((uintptr_t)objects[i] % alignof(max_align_t) == 0);
        *objects[i] = (AlignedObject){.value = (long double)i + 0.5L, .pointer = &pool};
    }
    for (size_t i = 0; i < 7; ++i) {
        assert(objects[i]->value == (long double)i + 0.5L && objects[i]->pointer == &pool);
        assert(memory_pool_release(&pool, objects[i]));
    }
    int *integer = memory_pool_alloc(&pool, sizeof *integer);
    assert(integer != nullptr);
    *integer = 42;
    assert(*integer == 42 && memory_pool_release(&pool, integer));
    memory_pool_destroy(&pool);
}

/** @brief 多轮占满/交错归还检验空闲链表没有环、重复节点或丢失块。 */
static void test_cycles(void)
{
    MemoryPool pool = {0};
    assert(memory_pool_init(&pool, 17, 17));
    for (int cycle = 0; cycle < 100; ++cycle) {
        void *blocks[17];
        for (size_t i = 0; i < 17; ++i) {
            blocks[i] = memory_pool_alloc(&pool, 17);
            assert(blocks[i] != nullptr);
            for (size_t j = 0; j < i; ++j) { assert(blocks[i] != blocks[j]); }
        }
        for (size_t i = 0; i < 17; ++i) {
            /* 5 与 17 互质，序列遍历所有索引，且不是简单的顺序归还。 */
            assert(memory_pool_release(&pool, blocks[(i * 5) % 17]));
        }
        assert(memory_pool_available(&pool) == 17);
    }
    memory_pool_destroy(&pool);
}

/** @brief 执行全部内存池测试。 @return 成功返回零。 */
int main(void)
{
    test_initialization(); test_release_and_reuse(); test_alignment_and_types(); test_cycles();
    puts("内存池测试通过：对齐、耗尽、复用、非法释放、算术溢出与分配失败安全。");
    return 0;
}
