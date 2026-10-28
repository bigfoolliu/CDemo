/**
 * @file
 * @brief M05：用二类和三类分布实验熵、交叉熵、KL 及损失梯度。
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    MAX_CLASSES = 3
};

#define PI 3.14159265358979323846

typedef enum {
    LOSS_INVALID,
    LOSS_FINITE,
    LOSS_INFINITE
} LossStatus;

typedef struct {
    double entropy;
    double cross_entropy;
    double kl;
    double logit_gradient[MAX_CLASSES];
    bool gradient_valid;
} DistributionMetrics;

/**
 * @brief 验证二类或三类概率分布，允许零概率但不允许负数或非有限值。
 * @param values 长度 count 的只读概率数组，不转移所有权。
 * @param count 类别数，必须为 2 或 3。
 * @return 分量和在 1e-12 内接近 1 时为 true。
 */
static bool valid_distribution(const double *values, size_t count)
{
    if (values == nullptr || count < 2 || count > MAX_CLASSES) {
        return false;
    }
    double sum = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i]) || values[i] < 0 || values[i] > 1) {
            return false;
        }
        sum += values[i];
    }
    return isfinite(sum) && fabs(sum - 1) <= 1e-12;
}

/**
 * @brief 计算熵、交叉熵、KL 和有限 softmax logit 的交叉熵梯度。
 * @param target 长度 count 的只读目标分布 q。
 * @param predicted 长度 count 的只读预测分布 p。
 * @param count 类别数，2 或 3。
 * @param result 可写结果；非法输入时不修改。
 * @return 有限结果为 LOSS_FINITE；目标正质量落在零预测概率上为 LOSS_INFINITE。
 */
static LossStatus distribution_metrics(const double *target, const double *predicted,
                                       size_t count, DistributionMetrics *result)
{
    if (result == nullptr || !valid_distribution(target, count) ||
        !valid_distribution(predicted, count)) {
        return LOSS_INVALID;
    }
    DistributionMetrics next = {0};
    next.gradient_valid = true;
    bool impossible = false;
    for (size_t i = 0; i < count; ++i) {
        double q = target[i];
        double p = predicted[i];
        if (q > 0) {
            next.entropy -= q * log(q);
            if (p == 0) {
                impossible = true;
            } else {
                next.cross_entropy -= q * log(p);
            }
        }
        if (p == 0) {
            next.gradient_valid = false;
        }
        next.logit_gradient[i] = p - q;
    }
    if (impossible) {
        next.cross_entropy = INFINITY;
        next.kl = INFINITY;
        next.gradient_valid = false;
        *result = next;
        return LOSS_INFINITE;
    }
    next.kl = next.cross_entropy - next.entropy;
    if (!isfinite(next.entropy) || !isfinite(next.cross_entropy) || !isfinite(next.kl)) {
        return LOSS_INVALID;
    }
    *result = next;
    return LOSS_FINITE;
}

/**
 * @brief 计算固定方差高斯单样本负对数似然及均值梯度。
 * @param observation 有限观测值。
 * @param mean 有限预测均值。
 * @param variance 有限且严格正的已知方差。
 * @param loss 可写损失；失败时不修改。
 * @param gradient 可写对 mean 的导数；失败时不修改。
 * @return 结果有限时为 true。
 */
static bool gaussian_loss(double observation, double mean, double variance,
                          double *loss, double *gradient)
{
    if (loss == nullptr || gradient == nullptr || !isfinite(observation) ||
        !isfinite(mean) || !isfinite(variance) || variance <= 0) {
        return false;
    }
    double residual = mean - observation;
    double next_loss = 0.5 * log(2 * PI * variance) + residual * residual / (2 * variance);
    double next_gradient = residual / variance;
    if (!isfinite(next_loss) || !isfinite(next_gradient)) {
        return false;
    }
    *loss = next_loss;
    *gradient = next_gradient;
    return true;
}

/** @brief 检查手算、零概率、无效分布与解析梯度的中心差分。 */
static void test_losses(void)
{
    const double binary_target[] = {0, 1};
    const double binary_correct[] = {0.01, 0.99};
    const double binary_wrong[] = {0.99, 0.01};
    const double binary_uncertain[] = {0.5, 0.5};
    DistributionMetrics result = {0};
    assert(distribution_metrics(binary_target, binary_correct, 2, &result) == LOSS_FINITE);
    assert(fabs(result.cross_entropy + log(0.99)) < 1e-12);
    assert(fabs(result.logit_gradient[1] + 0.01) < 1e-12);
    assert(distribution_metrics(binary_target, binary_wrong, 2, &result) == LOSS_FINITE);
    assert(fabs(result.cross_entropy + log(0.01)) < 1e-12);
    assert(distribution_metrics(binary_target, binary_uncertain, 2, &result) == LOSS_FINITE);
    assert(fabs(result.cross_entropy - log(2)) < 1e-12);

    const double ternary_target[] = {0, 1, 0};
    const double ternary_correct[] = {0.05, 0.9, 0.05};
    const double ternary_wrong[] = {0.9, 0.05, 0.05};
    const double ternary_uncertain[] = {1.0 / 3, 1.0 / 3, 1.0 / 3};
    assert(distribution_metrics(ternary_target, ternary_correct, 3, &result) == LOSS_FINITE);
    assert(fabs(result.cross_entropy + log(0.9)) < 1e-12);
    assert(fabs(result.logit_gradient[1] + 0.1) < 1e-12);
    assert(distribution_metrics(ternary_target, ternary_wrong, 3, &result) == LOSS_FINITE);
    assert(fabs(result.cross_entropy + log(0.05)) < 1e-12);
    assert(distribution_metrics(ternary_target, ternary_uncertain, 3, &result) == LOSS_FINITE);
    assert(fabs(result.cross_entropy - log(3)) < 1e-12);

    const double soft_target[] = {0.5, 0.5};
    const double soft_prediction[] = {0.75, 0.25};
    assert(distribution_metrics(soft_target, soft_prediction, 2, &result) == LOSS_FINITE);
    assert(fabs(result.entropy - log(2)) < 1e-12);
    assert(fabs(result.kl - 0.5 * log(4.0 / 3.0)) < 1e-12);
    assert(distribution_metrics(soft_target, soft_target, 2, &result) == LOSS_FINITE);
    assert(fabs(result.kl) < 1e-12);

    DistributionMetrics sentinel = {.entropy = 99};
    const double invalid[] = {0.8, 0.8};
    const double nonfinite[] = {NAN, 1};
    assert(distribution_metrics(soft_target, invalid, 2, &sentinel) == LOSS_INVALID);
    assert(distribution_metrics(soft_target, nonfinite, 2, &sentinel) == LOSS_INVALID);
    assert(distribution_metrics(soft_target, soft_target, 1, &sentinel) == LOSS_INVALID);
    assert(distribution_metrics(nullptr, soft_target, 2, &sentinel) == LOSS_INVALID);
    assert(sentinel.entropy == 99);
    const double zero[] = {1, 0};
    assert(distribution_metrics(binary_target, zero, 2, &result) == LOSS_INFINITE);
    assert(isinf(result.cross_entropy) && isinf(result.kl) && !result.gradient_valid);
    assert(distribution_metrics(zero, zero, 2, &result) == LOSS_FINITE);
    assert(result.cross_entropy == 0 && result.kl == 0 && !result.gradient_valid);

    double step = 1e-5;
    double logit = log(3);
    double high = 1 / (1 + exp(-(logit + step)));
    double low = 1 / (1 + exp(-(logit - step)));
    double numeric = (-log(high) + log(low)) / (2 * step);
    const double positive_prediction[] = {0.25, 0.75};
    assert(distribution_metrics(binary_target, positive_prediction, 2, &result) == LOSS_FINITE);
    assert(fabs(numeric - result.logit_gradient[1]) < 1e-9);
    double high_weight = exp(log(2) + step);
    double low_weight = exp(log(2) - step);
    double ternary_high = -log(high_weight / (2 + high_weight));
    double ternary_low = -log(low_weight / (2 + low_weight));
    const double middle_prediction[] = {0.25, 0.5, 0.25};
    assert(distribution_metrics(ternary_target, middle_prediction, 3, &result) == LOSS_FINITE);
    assert(fabs((ternary_high - ternary_low) / (2 * step) - result.logit_gradient[1]) < 1e-9);
    double loss = 99;
    double gradient = 99;
    assert(gaussian_loss(2, 1, 1, &loss, &gradient));
    assert(fabs(loss - (0.5 * log(2 * PI) + 0.5)) < 1e-12 && gradient == -1);
    assert(!gaussian_loss(2, 1, 0, &loss, &gradient));
    assert(!gaussian_loss(NAN, 1, 1, &loss, &gradient));
    assert(gradient == -1);
}

/**
 * @brief 对固定的二类/三类场景输出可重复结果。
 * @param csv 为 true 时使用 field,value 供两语言比较。
 * @return 成功为 EXIT_SUCCESS，意外计算失败为 EXIT_FAILURE。
 */
static int print_demo(bool csv)
{
    const double binary_target[] = {0, 1};
    const double binary_cases[3][2] = {{0.01, 0.99}, {0.99, 0.01}, {0.5, 0.5}};
    const double ternary_target[] = {0, 1, 0};
    const double ternary_cases[3][3] = {{0.05, 0.9, 0.05}, {0.9, 0.05, 0.05},
                                         {1.0 / 3, 1.0 / 3, 1.0 / 3}};
    const char *names[] = {"correct", "wrong", "uncertain"};
    if (csv) {
        puts("field,value");
    }
    for (size_t i = 0; i < 3; ++i) {
        DistributionMetrics binary = {0};
        DistributionMetrics ternary = {0};
        if (distribution_metrics(binary_target, binary_cases[i], 2, &binary) != LOSS_FINITE ||
            distribution_metrics(ternary_target, ternary_cases[i], 3, &ternary) != LOSS_FINITE) {
            return EXIT_FAILURE;
        }
        if (csv) {
            printf("binary_%s_loss,%.17g\nbinary_%s_logit_gradient,%.17g\n",
                   names[i], binary.cross_entropy, names[i], binary.logit_gradient[1]);
            printf("ternary_%s_loss,%.17g\nternary_%s_target_logit_gradient,%.17g\n",
                   names[i], ternary.cross_entropy, names[i], ternary.logit_gradient[1]);
        } else {
            printf("%s：二类损失 %.6f、目标 logit 梯度 %.6f；"
                   "三类损失 %.6f、目标 logit 梯度 %.6f\n",
                   names[i], binary.cross_entropy, binary.logit_gradient[1],
                   ternary.cross_entropy, ternary.logit_gradient[1]);
        }
    }
    const double soft_target[] = {0.5, 0.5};
    const double soft_prediction[] = {0.75, 0.25};
    DistributionMetrics soft = {0};
    DistributionMetrics equal = {0};
    double gaussian = 0;
    double gaussian_gradient = 0;
    if (distribution_metrics(soft_target, soft_prediction, 2, &soft) != LOSS_FINITE ||
        distribution_metrics(soft_target, soft_target, 2, &equal) != LOSS_FINITE ||
        !gaussian_loss(2, 1, 1, &gaussian, &gaussian_gradient)) {
        return EXIT_FAILURE;
    }
    if (csv) {
        printf("soft_entropy,%.17g\nsoft_cross_entropy,%.17g\nsoft_kl,%.17g\n",
               soft.entropy, soft.cross_entropy, soft.kl);
        printf("equal_kl,%.17g\ngaussian_nll,%.17g\ngaussian_mean_gradient,%.17g\n",
               equal.kl, gaussian, gaussian_gradient);
    } else {
        printf("软目标：熵 %.6f、交叉熵 %.6f、KL %.6f；相同分布 KL %.6f\n",
               soft.entropy, soft.cross_entropy, soft.kl, equal.kl);
        printf("高斯：负对数似然 %.6f，均值梯度 %.6f\n", gaussian, gaussian_gradient);
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行信息论与损失教学案例。
 * @param argc 参数数目，可传 --test 或 --csv。
 * @param argv 参数数组。
 * @return 成功为 EXIT_SUCCESS，参数或计算失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_losses();
        puts("信息论与损失测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1) {
        return print_demo(false);
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        return print_demo(true);
    }
    fputs("用法：34_information_losses [--test | --csv]\n", stderr);
    return EXIT_FAILURE;
}
