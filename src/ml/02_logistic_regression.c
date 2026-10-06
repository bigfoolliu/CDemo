/** @file
 * @brief 稳定的二分类逻辑回归。
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief 稳定计算逻辑回归的 Sigmoid 激活。
 * @param z 线性层输出。
 * @return 0 到 1 之间的激活值；极端输入可能舍入到端点。
 */
static double sigmoid(double z)
{
    /* 分支避免 exp 在大正数/负数输入时溢出。 */
    if (z >= 0) {
        return 1 / (1 + exp(-z));
    }
    double exponential = exp(z);
    return exponential / (1 + exponential);
}

/**
 * @brief 稳定计算二分类平均交叉熵。
 * @param weight 线性层权重。
 * @param bias 线性层偏置。
 * @param x 包含 count 个有限输入的只读数组。
 * @param y 包含 count 个 0/1 标签的只读数组。
 * @param count 样本数，须大于零。
 * @return 全部样本交叉熵的平均值。
 */
static double loss(double weight, double bias, const double *x,
                   const double *y, size_t count)
{
    double total = 0;
    for (size_t i = 0; i < count; ++i) {
        double z = weight * x[i] + bias;
        /* 二元交叉熵的稳定形式：softplus(z) - y*z。 */
        total += fmax(z, 0) - y[i] * z + log1p(exp(-fabs(z)));
    }
    return total / (double)count;
}

/**
 * @brief 运行并验证稳定的二分类逻辑回归。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    const double x[] = {-3, -2, -1, 1, 2, 3};
    const double y[] = {0, 0, 0, 1, 1, 1};
    const size_t count = sizeof(x) / sizeof(x[0]);
    double weight = 0;
    double bias = 0;
    double initial_loss = loss(weight, bias, x, y, count);
    for (int epoch = 0; epoch < 1000; ++epoch) {
        double weight_gradient = 0;
        double bias_gradient = 0;
        for (size_t i = 0; i < count; ++i) {
            double error = sigmoid(weight * x[i] + bias) - y[i];
            weight_gradient += error * x[i];
            bias_gradient += error;
        }
        weight -= 0.1 * weight_gradient / (double)count;
        bias -= 0.1 * bias_gradient / (double)count;
    }
    double final_loss = loss(weight, bias, x, y, count);
    assert(isfinite(final_loss) && final_loss < initial_loss && final_loss < 0.05);
    assert(sigmoid(weight * -0.5 + bias) < 0.5);
    assert(sigmoid(weight * 0.5 + bias) > 0.5);
    assert(sigmoid(-1000) == 0 && sigmoid(1000) == 1);
    printf("逻辑回归：交叉熵 %.4f → %.4f，x=-0.5 概率=%.4f，x=0.5 概率=%.4f\n",
           initial_loss, final_loss, sigmoid(weight * -0.5 + bias),
           sigmoid(weight * 0.5 + bias));
    return 0;
}
