/** @file @brief 运行阶段 10.2 正则化案例的有限测试入口。 */
#define main regularization_demo_main
#include "../src/ml/21_regularization.c"
#undef main
#include <assert.h>

/** @brief 梯度、正则目标、偏置与可重复数据行为均通过时退出零。 */
int main(void)
{
    char *arguments[] = {"21_regularization", "--test", nullptr};
    int result = regularization_demo_main(2, arguments);
    assert(result == EXIT_SUCCESS);
    return result;
}
