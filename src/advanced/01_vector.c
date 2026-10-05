#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef void *(*Resize)(void *, size_t);
typedef struct {
    int *data;
    size_t length;
    size_t capacity;
    Resize resize;
} Vector;

/* 不变量：length <= capacity；只有 [0, length) 内的元素可读取。
 * resize 必须兼容 realloc，返回的内存必须能用 free 释放。
 * 容器拥有 data；不要浅复制拥有内存的 Vector。 */
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

static bool push(Vector *vector, int value)
{
    if (vector->length == vector->capacity) {
        size_t maximum = SIZE_MAX / sizeof(*vector->data);
        if (vector->capacity == maximum) {
            return false;
        }
        size_t next = vector->capacity == 0 ? 4 :
            (vector->capacity > maximum / 2 ? maximum : vector->capacity * 2);
        if (!reserve(vector, next)) {
            return false;
        }
    }
    vector->data[vector->length++] = value;
    return true;
}

static void destroy(Vector *vector)
{
    free(vector->data);
    *vector = (Vector){.resize = realloc};
}

static void *fail_resize(void *pointer, size_t bytes)
{
    (void)pointer;
    (void)bytes;
    return nullptr; /* 注入分配失败，无需真的耗尽内存。 */
}

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
