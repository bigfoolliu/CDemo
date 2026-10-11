/** @file
 * @brief 动态数组的所有权与失败契约。
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief realloc 兼容的扩容回调，供容器注入分配策略。
 * @note 第一个参数为旧内存地址或 nullptr，第二个参数为请求字节数。
 *       成功返回可用 free 释放的新地址；失败返回 nullptr 并保留旧内存。
 */
typedef void *(*Resize)(void *, size_t);
typedef struct {
    int *data;
    size_t length;
    size_t capacity;
    Resize resize;
} Vector;

/**
 * @brief 为动态数组预留至少指定容量。
 * @param vector 满足容器不变量的可读写对象。
 * @param capacity 所需元素容量，不是字节数。
 * @return 容量足够或扩容成功返回 true；失败返回 false 且容器不变。
 * @note 不变量：length <= capacity；只有 [0, length) 内的元素可读取。
 *       resize 必须兼容 realloc，返回的内存必须能用 free 释放。
 *       容器拥有 data；不要浅复制拥有内存的 Vector。
 */
static bool reserve(Vector *vector, size_t capacity)
{
    if (capacity <= vector->capacity) {
        return true;
    }
    if (capacity > SIZE_MAX / sizeof(*vector->data)) {
        return false;
    }
    int *grown = vector->resize(vector->data, capacity * sizeof(*grown));
    if (grown == nullptr) {
        return false; /* 失败时地址、长度、容量及内容全部保持原样。 */
    }
    vector->data = grown;
    vector->capacity = capacity;
    return true;
}

/**
 * @brief 在动态数组末尾追加元素，必要时扩容。
 * @param vector 已初始化的可读写容器。
 * @param value 待追加的整数。
 * @return 追加成功返回 true；容量溢出或分配失败返回 false。
 */
static bool push(Vector *vector, int value)
{
    if (vector->length == vector->capacity) {
        size_t maximum = SIZE_MAX / sizeof(*vector->data);
        if (vector->capacity == maximum) {
            return false;
        }
        size_t next = vector->capacity == 0
                          ? 4
                          : (vector->capacity > maximum / 2 ? maximum : vector->capacity * 2);
        if (!reserve(vector, next)) {
            return false;
        }
    }
    vector->data[vector->length++] = value;
    return true;
}

/**
 * @brief 释放容器拥有的内存并恢复空状态。
 * @param vector 待销毁的有效对象；允许重复销毁空容器。
 */
static void destroy(Vector *vector)
{
    free(vector->data);
    *vector = (Vector){.resize = realloc};
}

/**
 * @brief 注入内存分配失败以验证容器失败路径。
 * @param pointer 模拟旧内存地址；本函数不读取或释放它。
 * @param bytes 模拟请求字节数；本函数忽略它。
 * @return 始终返回 nullptr。
 */
static void *fail_resize(void *pointer, size_t bytes)
{
    (void)pointer;
    (void)bytes;
    return nullptr; /* 注入分配失败，无需真的耗尽内存。 */
}

/**
 * @brief 运行并验证动态数组的所有权与失败契约。
 * @return 正常完成返回 0；资源操作失败可能返回非零，断言失败会终止程序。
 */
int main(void)
{
    Vector vector = {.resize = realloc};
    for (int i = 0; i < 100; ++i) {
        if (!push(&vector, i)) {
            destroy(&vector);
            return EXIT_FAILURE;
        }
    }
    assert(vector.length == 100 && vector.capacity >= 100);
    for (size_t i = 0; i < vector.length; ++i) {
        assert(vector.data[i] == (int)i);
    }
    int *original = vector.data;
    size_t capacity = vector.capacity;
    vector.resize = fail_resize;
    bool success = reserve(&vector, capacity + 1);
    assert(!success && vector.data == original && vector.capacity == capacity);
    assert(vector.length == 100 && vector.data[99] == 99);
    success = reserve(&vector, SIZE_MAX);
    assert(!success);
    destroy(&vector);
    destroy(&vector); /* 销毁后的空状态允许再次销毁。 */
    assert(vector.data == nullptr && vector.length == 0);
    puts("动态数组：几何扩容、所有权、溢出检查、失败不改状态验证通过");
    return EXIT_SUCCESS;
}
