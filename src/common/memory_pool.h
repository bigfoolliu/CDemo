/** @file
 * @brief 单线程固定块内存池：一次分配存储，空闲链表复用槽位。
 * 头文件实现便于各独立示例使用，不依赖 POSIX 或第三方库。
 */
#ifndef CDEMO_MEMORY_POOL_H
#define CDEMO_MEMORY_POOL_H
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 * 池拥有 storage 和 allocated，禁止浅复制已初始化的池。
 * 使用前必须零初始化；调用者不得修改内部字段。
 * 空闲链表用槽位索引表示，SIZE_MAX 表示链表结束。
 */
typedef struct {
    unsigned char *storage; /* malloc 获取的无声明类型存储，承载用户对象。 */
    unsigned char *allocated; /* 每块一个状态字节：0 空闲，1 已分配，与载荷分离。 */
    size_t block_size; /* 用户每次可请求的最大字节数，不包括对齐填充。 */
    size_t stride; /* 两个块起始地址间距，至少能容纳一个 size_t 链接。 */
    size_t capacity; /* 固定块数量，初始化后不扩容。 */
    size_t in_use;
    size_t free_head;
} MemoryPool;

/**
 * @brief 在每个空闲槽开头写入下一槽的索引，构造空闲链表。
 * @param pool 已分配存储的可读写池，仅供初始化内部调用。
 * @note memcpy 写对象表示，不将字节地址强转成 size_t * 解引用。
 */
static inline void memory_pool_link_free_blocks(MemoryPool *pool)
{
    for (size_t i = 0; i < pool->capacity; ++i) {
        size_t next = i + 1 == pool->capacity ? SIZE_MAX : i + 1;
        memcpy(pool->storage + i * pool->stride, &next, sizeof next);
    }
    pool->free_head = 0;
}

/**
 * @brief 初始化固定容量内存池，失败时保留原状态。
 * @param pool 有效且零初始化或已销毁的池；已有活动池不能再次初始化。
 * @param block_size 单次请求的最大字节数，必须大于零。
 * @param capacity 块数量，必须大于零。
 * @return 初始化成功为 true；参数、算术溢出或系统分配失败为 false。
 * @note 返回的块满足 alignof(max_align_t)，不支持扩展的过度对齐类型。
 */
static inline bool memory_pool_init(MemoryPool *pool, size_t block_size, size_t capacity)
{
    if (pool == nullptr || pool->storage != nullptr || pool->allocated != nullptr ||
        block_size == 0 || capacity == 0) {
        return false;
    }
    /* 每块既要装用户载荷，也要在空闲时装链表索引。
     * 使用 malloc 存储而不是声明为 char[] 的缓冲，允许用户通过合适类型
     * 的指针写入对象；不能只保证地址对齐，却忽略 C 的有效类型规则。
     */
    size_t minimum = block_size < sizeof(size_t) ? sizeof(size_t) : block_size;
    size_t alignment = alignof(max_align_t);
    size_t remainder = minimum % alignment;
    size_t padding = remainder == 0 ? 0 : alignment - remainder;
    if (minimum > SIZE_MAX - padding) { return false; }
    size_t stride = minimum + padding;
    /* 先验证，再计算乘法；不要等溢出后才检查结果。 */
    if (capacity > SIZE_MAX / stride) { return false; }
    unsigned char *storage = malloc(capacity * stride);
    if (storage == nullptr) { return false; }
    unsigned char *allocated = calloc(capacity, sizeof *allocated);
    if (allocated == nullptr) { free(storage); return false; }
    /* 两次分配都成功后才提交新状态，避免部分初始化交给调用者清理。 */
    *pool = (MemoryPool){.storage = storage, .allocated = allocated,
        .block_size = block_size, .stride = stride, .capacity = capacity};
    memory_pool_link_free_blocks(pool);
    return true;
}

/**
 * @brief 弹出一个空闲块，O(1)，不调用 malloc/calloc/realloc。
 * @param pool 有效且已初始化的可读写池。
 * @param bytes 请求字节数，必须在 1..block_size 内。
 * @return 块起始地址；无空闲块或参数非法返回 nullptr，池不变。
 * @note 内容未初始化，不承诺清零。对象由调用者初始化，不得访问超出请求的区域。
 * 返回地址只能用 memory_pool_release 归还，不能直接 free/realloc。
 */
static inline void *memory_pool_alloc(MemoryPool *pool, size_t bytes)
{
    if (pool == nullptr || pool->storage == nullptr || bytes == 0 ||
        bytes > pool->block_size || pool->free_head == SIZE_MAX) {
        return nullptr;
    }
    size_t index = pool->free_head;
    unsigned char *block = pool->storage + index * pool->stride;
    size_t next;
    /* 空闲状态时，开头保存 size_t 对象表示；分配后不再读取它，
     * 用户可以覆盖整块载荷。元数据状态放在另一个数组中，不占用用户空间。
     */
    memcpy(&next, block, sizeof next);
    pool->free_head = next;
    pool->allocated[index] = 1;
    ++pool->in_use;
    return block;
}

/**
 * @brief 校验并归还一块内存，成功后该块成为空闲链表头。
 * @param pool 有效且已初始化的可读写池。
 * @param pointer 本池当前已分配块的原始起始地址；nullptr 视为成功的无操作。
 * @return 成功为 true；外来地址、块内偏移或尚未重分配的重复释放为 false，池不变。
 * @note 验证采用逐块地址相等比较，复杂度 O(capacity)，不对无关指针减法/排序。
 * 归还后不能再读写对象。无法识别地址重用后的旧别名，调用者须管理生命周期。
 */
static inline bool memory_pool_release(MemoryPool *pool, void *pointer)
{
    if (pool == nullptr || pool->storage == nullptr) { return false; }
    if (pointer == nullptr) { return true; }
    for (size_t index = 0; index < pool->capacity; ++index) {
        unsigned char *block = pool->storage + index * pool->stride;
        if ((void *)block != pointer) { continue; }
        if (pool->allocated[index] == 0) { return false; }
        /* 将原来的链表头写入刚释放的块，再让此块成为新头，形成 LIFO 复用。
         * 此时用户不再拥有这块载荷，空闲链表可以覆盖对象原先的字节。
         */
        memcpy(block, &pool->free_head, sizeof pool->free_head);
        pool->free_head = index;
        pool->allocated[index] = 0;
        --pool->in_use;
        return true;
    }
    return false;
}

/**
 * @brief 返回剩余可分配块数。
 * @param pool 有效的只读池，或 nullptr；零初始化池也可查询。
 * @return 未占用块数；nullptr 返回零。
 */
static inline size_t memory_pool_available(const MemoryPool *pool)
{
    return pool == nullptr ? 0 : pool->capacity - pool->in_use;
}

/**
 * @brief 释放池的两块系统内存并恢复零状态，可重复销毁。
 * @param pool 有效的可读写池，或 nullptr。
 * @note 所有已发出的地址同时失效，包括尚未归还的对象；池不调用对象清理回调。
 * 若对象含文件或独立堆内存，调用者必须先自行释放这些资源。
 */
static inline void memory_pool_destroy(MemoryPool *pool)
{
    if (pool == nullptr) { return; }
    free(pool->storage);
    free(pool->allocated);
    *pool = (MemoryPool){0};
}
#endif
