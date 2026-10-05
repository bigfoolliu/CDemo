#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../common/logger.h"

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

int main(int argc, char *argv[])
{
    if (argc > 2) {
        fprintf(stderr, "用法：%s [日志文件路径]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *log_path = argc == 2 ? argv[1] : "linear_regression.log";
    Logger logger;
    if (!logger_open(&logger, log_path, stdout)) {
        fprintf(stderr, "无法打开日志文件：%s\n", log_path);
        perror("日志记录器初始化失败");
        return EXIT_FAILURE;
    }

    /* 用中心对称输入避免尺度过大；真实规律 y = 2x + 1。 */
    const double x[] = {-2, -1, 0, 1, 2};
    const double y[] = {-3, -1, 1, 3, 5};
    const size_t count = sizeof(x) / sizeof(x[0]);
    const int epochs = 500;
    const double learning_rate = 0.05;
    Model model = {0};
    double initial_loss = mse(model, x, y, count);

    logger_write(&logger, "=== 线性回归：批量梯度下降运行过程 ===\n");
    logger_write(&logger, "模型：预测值 = w * x + b；训练数据的真实关系：y = 2 * x + 1\n");
    logger_write(&logger, "损失：MSE = 所有样本误差平方之和 / 样本数\n");
    logger_write(&logger, "配置：样本数=%zu，学习率=%.2f，训练轮数=%d\n", count, learning_rate, epochs);
    for (size_t i = 0; i < count; ++i) {
        logger_write(&logger, "  训练样本 %zu：x=% .1f，y=% .1f\n", i + 1, x[i], y[i]);
    }
    logger_write(&logger, "初始状态：w=%.10f，b=%.10f，MSE=%.10e\n", model.weight, model.bias, initial_loss);
    logger_write(&logger, "每轮使用同一组更新前参数计算全部样本梯度，再统一更新参数。\n");
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double loss_before = mse(model, x, y, count);
        logger_write(&logger, "\n[第 %d/%d 轮] 更新前：w=%.10f，b=%.10f，MSE=%.10e\n",
               epoch + 1, epochs, model.weight, model.bias, loss_before);
        double weight_gradient = 0;
        double bias_gradient = 0;
        for (size_t i = 0; i < count; ++i) {
            double prediction = predict(model, x[i]);
            double error = prediction - y[i];
            double weight_contribution = 2 * error * x[i];
            double bias_contribution = 2 * error;
            weight_gradient += weight_contribution;
            bias_gradient += bias_contribution;
            logger_write(&logger, "  样本 %zu：x=% .1f，目标=% .1f，预测=% .10f，误差=% .10e，误差平方=%.10e\n",
                   i + 1, x[i], y[i], prediction, error, error * error);
            logger_write(&logger, "    梯度贡献：2 * 误差 * x=% .10e，2 * 误差=% .10e\n",
                   weight_contribution, bias_contribution);
        }
        logger_write(&logger, "  梯度求和：w=% .10e，b=% .10e\n", weight_gradient, bias_gradient);
        weight_gradient /= (double)count;
        bias_gradient /= (double)count;
        logger_write(&logger, "  平均梯度：dMSE/dw=% .10e，dMSE/db=% .10e\n",
               weight_gradient, bias_gradient);
        /* 梯度下降：参数沿损失函数梯度的反方向更新。 */
        double weight_step = learning_rate * weight_gradient;
        double bias_step = learning_rate * bias_gradient;
        logger_write(&logger, "  参数更新：w = %.10f - (% .10e)，b = %.10f - (% .10e)\n",
               model.weight, weight_step, model.bias, bias_step);
        model.weight -= weight_step;
        model.bias -= bias_step;
        double loss_after = mse(model, x, y, count);
        logger_write(&logger, "  更新结果：w=%.10f，b=%.10f，MSE=%.10e，损失减少=% .10e\n",
               model.weight, model.bias, loss_after, loss_before - loss_after);
    }
    double final_loss = mse(model, x, y, count);
    logger_write(&logger, "\n=== 训练结束与结果验证 ===\n");
    logger_write(&logger, "最终模型：预测值 = %.10f * x + %.10f\n", model.weight, model.bias);
    logger_write(&logger, "损失变化：%.10e → %.10e\n", initial_loss, final_loss);
    assert(isfinite(final_loss) && final_loss < initial_loss && final_loss < 1e-10);
    logger_write(&logger, "验证通过：最终损失为有限值、低于初始损失且小于 1e-10。\n");
    /* 不在训练集中的输入用于检查是否学到直线关系。 */
    logger_write(&logger, "新输入 x=3：预测=%.10f，目标=7，绝对误差=%.10e\n",
           predict(model, 3), fabs(predict(model, 3) - 7));
    assert(fabs(predict(model, 3) - 7) < 1e-5);
    logger_write(&logger, "新输入 x=-3：预测=%.10f，目标=-5，绝对误差=%.10e\n",
           predict(model, -3), fabs(predict(model, -3) + 5));
    assert(fabs(predict(model, -3) + 5) < 1e-5);
    logger_write(&logger, "验证通过：两个新输入的预测绝对误差均小于 1e-5。\n");
    logger_write(&logger, "线性回归：w=%.4f，b=%.4f，MSE %.4f → %.8f，x=3 预测=%.4f\n",
           model.weight, model.bias, initial_loss, final_loss, predict(model, 3));
    logger_write(&logger, "完整训练日志已追加至：%s\n", log_path);
    if (!logger_close(&logger)) {
        fputs("日志写入、刷新或关闭失败，请检查终端和日志文件。\n", stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
