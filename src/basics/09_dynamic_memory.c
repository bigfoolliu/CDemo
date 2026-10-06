/** @file
 * @brief 动态内存分配与释放。
 */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief 运行并验证动态内存分配与释放。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    size_t count = 3;
    if (count > SIZE_MAX / sizeof(int)) {
        return EXIT_FAILURE;
    }
    int *values = malloc(count * sizeof(*values)); /* 分配内容尚未初始化。 */
    if (values == nullptr) {
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < count; ++i) {
        values[i] = (int)i + 1;
    }
    /* 本例固定扩为 4 个元素，大小不会溢出；通用代码仍需检查乘法。 */
    int *grown = realloc(values, 4 * sizeof(*values));
    if (grown == nullptr) {
        free(values); /* realloc 失败时原内存仍有效。 */
        return EXIT_FAILURE;
    }
    values = grown; /* 成功后旧指针不可继续使用。 */
    values[3] = 4;
    assert(values[0] == 1 && values[2] == 3 && values[3] == 4);
    free(values);
    values = nullptr; /* 其他别名不会因此自动变为空指针。 */
    int *zeros = calloc(count, sizeof(*zeros));
    if (zeros == nullptr) {
        return EXIT_FAILURE;
    }
    assert(zeros[0] == 0 && zeros[2] == 0);
    free(zeros);
    puts("动态内存：malloc、realloc、calloc、free 验证通过");
    return EXIT_SUCCESS;
}
