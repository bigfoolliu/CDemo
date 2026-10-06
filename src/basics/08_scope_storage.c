/** @file
 * @brief 作用域、链接与存储期。
 */
#include <assert.h>
#include <stdio.h>

static int file_counter; /* 文件作用域，静态存储期，内部链接；默认初始化为 0。 */

/**
 * @brief 递增文件计数器和静态局部调用计数。
 * @return 递增后的调用次数；本示例仅执行少量调用以避免溢出。
 */
static int next_count(void)
{
    static int calls; /* 块作用域，但生命周期贯穿整个程序。 */
    ++file_counter;
    return ++calls;
}

/**
 * @brief 运行并验证作用域、链接与存储期。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    int value = 10; /* 自动存储期；离开所在块后生命周期结束。 */
    {
        int value = 20; /* 内层名称遮蔽外层；实际代码应谨慎使用。 */
        assert(value == 20);
    }
    assert(value == 10);
    assert(next_count() == 1 && next_count() == 2 && file_counter == 2);
    puts("作用域与存储期：局部对象、静态局部对象、文件静态对象验证通过");
    return 0;
}
