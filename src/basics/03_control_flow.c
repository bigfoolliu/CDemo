#include <assert.h>
#include <stdio.h>

int main(void)
{
    int sum = 0;
    for (int i = 0; i < 6; ++i) {
        if (i == 2) {
            continue; /* 跳过本轮剩余语句。 */
        } else if (i == 5) {
            break;    /* 结束最内层循环。 */
        }
        sum += i;
    }
    assert(sum == 8);
    int remaining = 2;
    while (remaining > 0) {
        --remaining;
    }
    do {
        ++remaining; /* do/while 至少执行一次。 */
    } while (remaining < 1);
    switch (remaining) {
    case 1:
        sum += 1;
        break; /* 没有 break 时可能继续执行后续 case。 */
    default:
        sum = 0;
        break;
    }
    assert(sum == 9);
    puts("控制流：if、for、while、do/while、switch 验证通过");
    return 0;
}
