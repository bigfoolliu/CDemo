/** @file
 * @brief C23 数字字面量。
 */
#include <assert.h>
#include <stdio.h>

/**
 * @brief 运行并验证C23 数字字面量。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    /* C23 标准化二进制字面量，并允许用单引号分隔数字。 */
    unsigned int flags = 0b1010'0101;
    unsigned int population = 1'000'000;
    unsigned int mask = 0xFF'00;

    assert(flags == 165u);
    assert(population == 1000000u);
    assert(mask == 65280u);
    printf("数字字面量：flags=%u，population=%u，mask=0x%X\n",
           flags, population, mask);
    return 0;
}
