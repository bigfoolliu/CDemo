/** @file @brief 运行阶段 11.1 ResNet 教学案例的确定性测试。 */
#define main resnet_demo_main
#include "../src/ml/22_resnet.c"
#undef main
#include <assert.h>

/** @brief 验证残差恒等路径、CNN 与 ResNet 的梯度及输入边界。 */
int main(void)
{
    char *arguments[] = {"22_resnet", "--test", nullptr};
    int result = resnet_demo_main(2, arguments);
    assert(result == EXIT_SUCCESS);
    return result;
}
