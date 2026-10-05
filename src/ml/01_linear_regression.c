#include <assert.h>
#include <math.h>
#include <stdio.h>

typedef struct { double weight; double bias; } Model;

static double predict(Model model, double x)
{
    return model.weight * x + model.bias;
}

static double mse(Model model, const double *x, const double *y, size_t count)
{
    double loss = 0;
    for (size_t i = 0; i < count; ++i) {
        double error = predict(model, x[i]) - y[i];
        loss += error * error;
    }
    return loss / (double)count;
}

int main(void)
{
    /* 用中心对称输入避免尺度过大；真实规律 y = 2x + 1。 */
    const double x[] = {-2, -1, 0, 1, 2};
    const double y[] = {-3, -1, 1, 3, 5};
    const size_t count = sizeof(x) / sizeof(x[0]);
    Model model = {0};
    double initial_loss = mse(model, x, y, count);
    for (int epoch = 0; epoch < 500; ++epoch) {
        double weight_gradient = 0;
        double bias_gradient = 0;
        for (size_t i = 0; i < count; ++i) {
            double error = predict(model, x[i]) - y[i];
            weight_gradient += 2 * error * x[i];
            bias_gradient += 2 * error;
        }
        /* 梯度下降：参数沿损失函数梯度的反方向更新。 */
        model.weight -= 0.05 * weight_gradient / (double)count;
        model.bias -= 0.05 * bias_gradient / (double)count;
    }
    double final_loss = mse(model, x, y, count);
    assert(isfinite(final_loss) && final_loss < initial_loss && final_loss < 1e-10);
    /* 不在训练集中的输入用于检查是否学到直线关系。 */
    assert(fabs(predict(model, 3) - 7) < 1e-5);
    assert(fabs(predict(model, -3) + 5) < 1e-5);
    printf("线性回归：w=%.4f，b=%.4f，MSE %.4f → %.8f，x=3 预测=%.4f\n",
           model.weight, model.bias, initial_loss, final_loss, predict(model, 3));
    return 0;
}
