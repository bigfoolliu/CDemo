/** @file
 * @brief C23 位精确整数。
 */
#include <assert.h>
#include <stdio.h>

/**
 * @brief 运行并验证C23 位精确整数。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    /* unsigned _BitInt(5) 的值域为 0..31。
     * 位精确描述的是值的位宽，sizeof 仍以字节为单位。 */
    unsigned _BitInt(5) counter = 31;
    counter = (unsigned _BitInt(5))(counter + 1);

    assert(counter == 0);
    /* printf 没有通用的 _BitInt 占位符，先转换类型。 */
    printf("位精确整数：5 位无符号计数器回绕后=%u\n", (unsigned int)counter);
    return 0;
}
