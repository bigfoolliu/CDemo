/** @file @brief 固定块内存池演示：小对象分配、池耗尽、释放和原地址复用。 */
#include "../common/memory_pool.h"
#include <assert.h>
#include <stdio.h>

/** 示例对象；只包含数值字段，归还前无需清理外部资源。 */
typedef struct {
    unsigned int id;
    double x, y;
} PointObject;

/**
 * @brief 按阶段展示内存池状态并验证复用行为。
 * @return 成功返回 EXIT_SUCCESS；系统分配失败返回 EXIT_FAILURE。
 */
int main(void)
{
    MemoryPool pool = {0}; /* 拥有资源的池不可直接浅复制。 */
    if (!memory_pool_init(&pool, sizeof(PointObject), 3)) {
        fputs("内存池初始化失败。\n", stderr);
        return EXIT_FAILURE;
    }
    printf("初始化：块载荷=%zu 字节，实际间距=%zu 字节，对齐=%zu，容量=%zu\n",
           pool.block_size, pool.stride, alignof(max_align_t), pool.capacity);
    PointObject *objects[3];
    for (size_t i = 0; i < 3; ++i) {
        objects[i] = memory_pool_alloc(&pool, sizeof *objects[i]);
        assert(objects[i] != nullptr);
        /* 池不会清零；以正确类型写入对象后才能读取字段。 */
        *objects[i] = (PointObject){.id = (unsigned int)i + 1, .x = (double)i * 10, .y = 20};
        printf("分配对象 %u：地址=%p，剩余块=%zu\n", objects[i]->id,
               (void *)objects[i], memory_pool_available(&pool));
    }
    void *extra = memory_pool_alloc(&pool, sizeof(PointObject));
    assert(extra == nullptr);
    puts("第 4 次分配：池已耗尽，返回 nullptr，已有对象保持有效。");
    assert(objects[0]->id == 1 && objects[2]->x == 20);

    void *original = objects[1]; /* 只用于比较池内存地址，不再读取已归还对象。 */
    bool released = memory_pool_release(&pool, objects[1]);
    assert(released);
    objects[1] = nullptr; /* 让本地变量表达“不再拥有对象”，其他别名仍须自行管理。 */
    printf("归还对象 2：剩余块=%zu\n", memory_pool_available(&pool));
    released = memory_pool_release(&pool, original);
    assert(!released);
    puts("重复归还：被拒绝，空闲链表不变。");

    PointObject *replacement = memory_pool_alloc(&pool, sizeof *replacement);
    assert((void *)replacement == original);
    *replacement = (PointObject){.id = 99, .x = 1.5, .y = 2.5};
    printf("重新分配：复用原地址=%p，新对象 id=%u\n", (void *)replacement, replacement->id);
    released = memory_pool_release(&pool, objects[0]); assert(released);
    released = memory_pool_release(&pool, objects[2]); assert(released);
    released = memory_pool_release(&pool, replacement); assert(released);
    assert(memory_pool_available(&pool) == 3);
    memory_pool_destroy(&pool);
    memory_pool_destroy(&pool);
    puts("所有对象已归还；池已销毁，重复销毁安全。");
    return EXIT_SUCCESS;
}
