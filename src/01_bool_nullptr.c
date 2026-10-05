#include <assert.h>
#include <stdio.h>

/* C23 中 bool、true、false 是关键字，无需包含 stdbool.h。
 * nullptr 表示空指针，不是整数 0 或字符 '\0'。 */
static_assert(sizeof(bool) > 0, "bool 必须占用存储空间");

int main(void)
{
    bool ready = true;
    int *value = nullptr;

    assert(ready != false);
    assert(value == nullptr);
    printf("bool 与 nullptr：ready=%d，指针为空=%d\n", ready, value == nullptr);
    return 0;
}
