#include <assert.h>
#include <stdio.h>

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
