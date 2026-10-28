/**
 * @file
 * @brief M06：Softmax 稳定性、差分步长与近奇异矩阵扰动。
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    CLASS_COUNT = 3,
    STEP_COUNT = 8
};

static const double STEPS[STEP_COUNT] = {1e-1, 1e-2, 1e-4, 1e-6,
                                         1e-8, 1e-10, 1e-12, 1e-14};

typedef struct {
    double base_x[2];
    double changed_x[2];
    double condition_infinity;
    double relative_amplification;
} PerturbationResult;

/**
 * @brief 直接对三个分数取指数，再归一化；只用于失败对照。
 * @param logits 长度为 3 的有限输入数组。
 * @param probabilities 长度为 3 的可写输出，失败时不修改。
 * @return 分母与概率均有限时为 true，溢出时为 false。
 */
static bool naive_softmax(const double logits[CLASS_COUNT],
                          double probabilities[CLASS_COUNT])
{
    if (logits == nullptr || probabilities == nullptr) {
        return false;
    }
    double values[CLASS_COUNT] = {0};
    double sum = 0;
    for (size_t i = 0; i < CLASS_COUNT; ++i) {
        if (!isfinite(logits[i])) {
            return false;
        }
        values[i] = exp(logits[i]);
        sum += values[i];
    }
    if (!isfinite(sum) || sum <= 0) {
        return false;
    }
    for (size_t i = 0; i < CLASS_COUNT; ++i) {
        values[i] /= sum;
        if (!isfinite(values[i])) {
            return false;
        }
    }
    memcpy(probabilities, values, sizeof(values));
    return true;
}

/**
 * @brief 减去最大分数后计算稳定 Softmax 与 log-sum-exp。
 * @param logits 长度为 3 的有限输入数组。
 * @param probabilities 长度为 3 的可写输出，失败时不修改。
 * @param logsumexp 可写的对数指数和，失败时不修改。
 * @return 全部结果有限且概率和接近 1 时为 true。
 */
static bool stable_softmax(const double logits[CLASS_COUNT],
                           double probabilities[CLASS_COUNT], double *logsumexp)
{
    if (logits == nullptr || probabilities == nullptr || logsumexp == nullptr) {
        return false;
    }
    double maximum = logits[0];
    for (size_t i = 0; i < CLASS_COUNT; ++i) {
        if (!isfinite(logits[i])) {
            return false;
        }
        if (logits[i] > maximum) {
            maximum = logits[i];
        }
    }
    double values[CLASS_COUNT] = {0};
    double sum = 0;
    for (size_t i = 0; i < CLASS_COUNT; ++i) {
        values[i] = exp(logits[i] - maximum);
        sum += values[i];
    }
    double next_logsumexp = maximum + log(sum);
    if (!isfinite(sum) || sum <= 0 || !isfinite(next_logsumexp)) {
        return false;
    }
    for (size_t i = 0; i < CLASS_COUNT; ++i) {
        values[i] /= sum;
        if (!isfinite(values[i])) {
            return false;
        }
    }
    memcpy(probabilities, values, sizeof(values));
    *logsumexp = next_logsumexp;
    return true;
}

/**
 * @brief 用中心差分近似 x^3 在 x=1 的导数。
 * @param step 有限正步长。
 * @param derivative 可写结果，失败时不修改。
 * @return 两侧函数值与导数都有限时为 true。
 */
static bool cubic_difference(double step, double *derivative)
{
    if (derivative == nullptr || !isfinite(step) || step <= 0) {
        return false;
    }
    double high = 1 + step;
    double low = 1 - step;
    double high_value = high * high * high;
    double low_value = low * low * low;
    double result = (high_value - low_value) / (2 * step);
    if (!isfinite(result)) {
        return false;
    }
    *derivative = result;
    return true;
}

/**
 * @brief 求解近奇异二元系统，并测量右端微扰的相对放大。
 * @param delta 有限正数；1+delta 必须可区别于 1。
 * @param perturbation 有限正数；必须改变第二个右端分量。
 * @param output 可写结果，失败时不修改。
 * @return 分母和所有计算结果有限时为 true。
 */
static bool near_singular_system(double delta, double perturbation,
                                 PerturbationResult *output)
{
    if (output == nullptr || !isfinite(delta) || !isfinite(perturbation) ||
        delta <= 0 || perturbation <= 0) {
        return false;
    }
    double a22 = 1 + delta;
    double determinant = a22 - 1;
    double b1 = 2;
    double b2 = 2 + determinant;
    double changed_b2 = b2 + perturbation;
    if (determinant <= 0 || changed_b2 == b2 || !isfinite(changed_b2)) {
        return false;
    }
    PerturbationResult next = {0};
    next.base_x[1] = (b2 - b1) / determinant;
    next.base_x[0] = b1 - next.base_x[1];
    next.changed_x[1] = (changed_b2 - b1) / determinant;
    next.changed_x[0] = b1 - next.changed_x[1];
    next.condition_infinity = (2 + determinant) * (2 + determinant) / determinant;
    double input_relative = fabs(changed_b2 - b2) / fmax(fabs(b1), fabs(b2));
    double output_relative = fabs(next.changed_x[1] - next.base_x[1]);
    next.relative_amplification = output_relative / input_relative;
    if (!isfinite(next.base_x[0]) || !isfinite(next.base_x[1]) ||
        !isfinite(next.changed_x[0]) || !isfinite(next.changed_x[1]) ||
        !isfinite(next.condition_infinity) || !isfinite(next.relative_amplification)) {
        return false;
    }
    *output = next;
    return true;
}

/** @brief 验证极端分数、手算导数、步长两端误差与矩阵边界。 */
static void test_stability(void)
{
    const double small[] = {0, 1, 2};
    const double large[] = {1000, 1001, 1002};
    double stable[CLASS_COUNT] = {0};
    double naive[CLASS_COUNT] = {0};
    double logsumexp = 0;
    assert(naive_softmax(small, naive));
    assert(!naive_softmax(large, naive));
    assert(stable_softmax(large, stable, &logsumexp));
    for (size_t i = 0; i < CLASS_COUNT; ++i) {
        assert(fabs(stable[i] - naive[i]) < 1e-12);
    }
    assert(fabs(stable[0] + stable[1] + stable[2] - 1) < 1e-12);
    assert(fabs(logsumexp - (1002 + log(1 + exp(-1) + exp(-2)))) < 1e-12);
    double sentinel[CLASS_COUNT] = {99, 99, 99};
    const double invalid[] = {0, NAN, 2};
    double saved = 99;
    assert(!stable_softmax(invalid, sentinel, &saved));
    assert(sentinel[0] == 99 && saved == 99);

    double errors[STEP_COUNT] = {0};
    for (size_t i = 0; i < STEP_COUNT; ++i) {
        double derivative = 0;
        assert(cubic_difference(STEPS[i], &derivative));
        errors[i] = fabs(derivative - 3);
    }
    assert(fabs(errors[0] - 0.01) < 1e-12);
    assert(errors[3] < 1e-8);
    assert(errors[7] > errors[3]);
    saved = 99;
    assert(!cubic_difference(0, &saved) && saved == 99);

    PerturbationResult result = {0};
    assert(near_singular_system(1e-8, 1e-10, &result));
    assert(fabs(result.base_x[0] - 1) < 1e-12);
    assert(fabs(result.base_x[1] - 1) < 1e-12);
    assert(fabs(result.changed_x[0] - 0.99) < 1e-6);
    assert(fabs(result.changed_x[1] - 1.01) < 1e-6);
    assert(result.relative_amplification > 1e8);
    assert(result.relative_amplification <= result.condition_infinity * 1.000001);
    PerturbationResult untouched = {.condition_infinity = 99};
    assert(!near_singular_system(1e-20, 1e-10, &untouched));
    assert(!near_singular_system(1e-8, 1e-20, &untouched));
    assert(untouched.condition_infinity == 99);
}

/**
 * @brief 打印稳定概率、步长误差表和扰动放大结果。
 * @param csv 为 true 时打印 field,value 供跨语言比较。
 * @return 成功为 EXIT_SUCCESS，数值失败为 EXIT_FAILURE。
 */
static int print_demo(bool csv)
{
    const double large[] = {1000, 1001, 1002};
    double probabilities[CLASS_COUNT] = {0};
    double logsumexp = 0;
    PerturbationResult system = {0};
    if (!stable_softmax(large, probabilities, &logsumexp) ||
        !near_singular_system(1e-8, 1e-10, &system)) {
        return EXIT_FAILURE;
    }
    if (csv) {
        puts("field,value");
        for (size_t i = 0; i < CLASS_COUNT; ++i) {
            printf("softmax_%zu,%.17g\n", i, probabilities[i]);
        }
        printf("logsumexp,%.17g\n", logsumexp);
    } else {
        printf("大分数稳定 Softmax：(%.8f, %.8f, %.8f)，log-sum-exp=%.8f\n",
               probabilities[0], probabilities[1], probabilities[2], logsumexp);
        puts("中心差分 f(x)=x^3 在 x=1 的解析导数为 3：");
    }
    for (size_t i = 0; i < STEP_COUNT; ++i) {
        double derivative = 0;
        if (!cubic_difference(STEPS[i], &derivative)) {
            return EXIT_FAILURE;
        }
        if (csv) {
            printf("error_%zu,%.17g\n", i, fabs(derivative - 3));
        } else {
            printf("h=%.0e，差分=%.12g，绝对误差=%.6g\n",
                   STEPS[i], derivative, fabs(derivative - 3));
        }
    }
    if (csv) {
        printf("changed_x0,%.17g\nchanged_x1,%.17g\ncondition_infinity,%.17g\n",
               system.changed_x[0], system.changed_x[1], system.condition_infinity);
        printf("relative_amplification,%.17g\n", system.relative_amplification);
    } else {
        printf("近奇异系统：x=(%.8f, %.8f) → (%.8f, %.8f)，"
               "条件数≈%.3g，实测相对放大≈%.3g\n",
               system.base_x[0], system.base_x[1], system.changed_x[0],
               system.changed_x[1], system.condition_infinity, system.relative_amplification);
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行 M06 数值稳定性实验。
 * @param argc 参数数目，可传 --test 或 --csv。
 * @param argv 参数数组。
 * @return 成功为 EXIT_SUCCESS，错误为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_stability();
        puts("数值稳定性测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1) {
        return print_demo(false);
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        return print_demo(true);
    }
    fputs("用法：35_numerical_stability [--test | --csv]\n", stderr);
    return EXIT_FAILURE;
}
