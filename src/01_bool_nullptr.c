/** @file
 * @brief C23 布尔关键字与空指针常量。
 */
#include <assert.h>
#include <stdio.h>

/* C23 中 bool、true、false 是关键字，无需包含 stdbool.h。
 * nullptr 表示空指针，不是整数 0 或字符 '\0'。 */
static_assert(sizeof(bool) > 0, "bool 必须占用存储空间");

/**
 * @brief 运行并验证C23 布尔关键字与空指针常量。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    bool ready = true;
    int *value = nullptr;

    assert(ready != false);
    assert(value == nullptr);
    printf("bool 与 nullptr：ready=%d，指针为空=%d\n", ready, value == nullptr);
    return 0;
}
