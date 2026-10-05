#include <assert.h>
#include <math.h>
#include <stdio.h>

static double sigmoid(double z)
{
    /* 分支避免 exp 在大正数/负数输入时溢出。 */
    if (z >= 0) {
        return 1 / (1 + exp(-z));
    }
    double exponential = exp(z);
    return exponential / (1 + exponential);
}

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
