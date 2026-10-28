/**
 * @file
 * @brief M10：用多项式拟合观察偏差—方差，并用成对自助法估计测试误差差值。
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    MAX_DEGREE = 5,
    MAX_SAMPLE_COUNT = 16,
    DEGREE_COUNT = 3,
    REPEAT_COUNT = 40,
    PROBE_COUNT = 9,
    BOOTSTRAP_COUNT = 1000
};

static const size_t DEGREES[DEGREE_COUNT] = {1, 3, 5};
static const double PROBE_X[PROBE_COUNT] = {-1.0, -0.75, -0.5, -0.25, 0.0,
                                             0.25, 0.5, 0.75, 1.0};

typedef struct {
    double x;
    double y;
} Sample;

typedef struct {
    size_t degree;
    double coefficients[MAX_DEGREE + 1];
} Polynomial;

typedef struct {
    double mean_difference;
    double lower;
    double upper;
} BootstrapInterval;

typedef struct {
    double bias_squared[DEGREE_COUNT];
    double variance[DEGREE_COUNT];
    double expected_mse[DEGREE_COUNT];
    double test_mse[DEGREE_COUNT];
    double shifted_mse[DEGREE_COUNT];
    BootstrapInterval interval;
} Experiment;

static int compare_doubles(const void *left, const void *right);

/**
 * @brief 生成可复现的 32 位伪随机数。
 * @param state 调用方独占维护的非零状态；函数会更新它。
 * @return 下一个伪随机数；本生成器仅用于教学，不用于安全场景。
 */
static uint32_t random_next(uint32_t *state)
{
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

/**
 * @brief 将伪随机数映射到半开区间 [0, 1)。
 * @param state 可写随机状态。
 * @return 均匀映射值。
 */
static double random_unit(uint32_t *state)
{
    return (double)random_next(state) / 4294967296.0;
}

/**
 * @brief 计算合成数据的无噪声目标函数。
 * @param x 有限输入坐标。
 * @return 函数值 $0.75+1.2x-0.8x^2$。
 */
static double target_function(double x)
{
    return 0.75 + 1.2 * x - 0.8 * x * x;
}

/**
 * @brief 生成输入区间内带均匀噪声的固定大小数据集。
 * @param state 可写且非零的伪随机状态。
 * @param count 需要生成的样本数，范围为 1 到 MAX_SAMPLE_COUNT。
 * @param lower_x 输入区间下界，须小于上界。
 * @param upper_x 输入区间上界，须大于下界。
 * @param noise_amplitude 噪声半宽，须为有限非负数。
 * @param output 至少可写 count 个样本；失败时保持不变。
 * @return 全部样本成功生成时为 true。
 */
static bool dataset_generate(uint32_t *state, size_t count, double lower_x, double upper_x,
                             double noise_amplitude, Sample output[MAX_SAMPLE_COUNT])
{
    if (state == nullptr || *state == 0 || count == 0 || count > MAX_SAMPLE_COUNT ||
        !isfinite(lower_x) || !isfinite(upper_x) || lower_x >= upper_x ||
        !isfinite(noise_amplitude) || noise_amplitude < 0 || output == nullptr) {
        return false;
    }

    Sample next[MAX_SAMPLE_COUNT] = {{0}};
    for (size_t index = 0; index < count; ++index) {
        double x = lower_x + (upper_x - lower_x) * random_unit(state);
        double noise = noise_amplitude * (2.0 * random_unit(state) - 1.0);
        next[index] = (Sample){x, target_function(x) + noise};
        if (!isfinite(next[index].x) || !isfinite(next[index].y)) {
            return false;
        }
    }
    memcpy(output, next, count * sizeof(*output));
    return true;
}

/**
 * @brief 用部分主元消元拟合低阶多项式最小二乘解。
 * @param samples 至少 degree+1 个有限训练样本；只读，不含验证/测试数据。
 * @param count 样本数，范围为 degree+1 到 MAX_SAMPLE_COUNT。
 * @param degree 多项式最高次数，范围为 0 到 MAX_DEGREE。
 * @param output 可写模型；输入无效、矩阵秩亏或解非有限时保持不变。
 * @return 求解成功时为 true；正规方程会平方设计矩阵的条件数。
 */
static bool polynomial_fit(const Sample *samples, size_t count, size_t degree,
                           Polynomial *output)
{
    if (samples == nullptr || output == nullptr || degree > MAX_DEGREE ||
        count < degree + 1 || count > MAX_SAMPLE_COUNT) {
        return false;
    }

    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(samples[index].x) || !isfinite(samples[index].y)) {
            return false;
        }
    }

    size_t parameter_count = degree + 1;
    double augmented[MAX_DEGREE + 1][MAX_DEGREE + 2] = {{0}};
    for (size_t row = 0; row < parameter_count; ++row) {
        for (size_t sample = 0; sample < count; ++sample) {
            double power = 1.0;
            for (size_t exponent = 0; exponent < row; ++exponent) {
                power *= samples[sample].x;
            }
            augmented[row][parameter_count] += power * samples[sample].y;
            for (size_t column = 0; column < parameter_count; ++column) {
                augmented[row][column] += power;
                power *= samples[sample].x;
            }
        }
    }

    for (size_t column = 0; column < parameter_count; ++column) {
        size_t pivot = column;
        for (size_t row = column + 1; row < parameter_count; ++row) {
            if (fabs(augmented[row][column]) > fabs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (!isfinite(augmented[pivot][column]) || fabs(augmented[pivot][column]) < 1e-12) {
            return false;
        }
        if (pivot != column) {
            for (size_t entry = column; entry <= parameter_count; ++entry) {
                double temporary = augmented[column][entry];
                augmented[column][entry] = augmented[pivot][entry];
                augmented[pivot][entry] = temporary;
            }
        }

        for (size_t row = column + 1; row < parameter_count; ++row) {
            double factor = augmented[row][column] / augmented[column][column];
            for (size_t entry = column; entry <= parameter_count; ++entry) {
                augmented[row][entry] -= factor * augmented[column][entry];
            }
        }
    }

    Polynomial next = {0};
    next.degree = degree;
    for (size_t reverse = parameter_count; reverse > 0; --reverse) {
        size_t row = reverse - 1;
        double value = augmented[row][parameter_count];
        for (size_t column = row + 1; column < parameter_count; ++column) {
            value -= augmented[row][column] * next.coefficients[column];
        }
        next.coefficients[row] = value / augmented[row][row];
        if (!isfinite(next.coefficients[row])) {
            return false;
        }
    }
    *output = next;
    return true;
}

/**
 * @brief 计算多项式对一个有限坐标的预测。
 * @param model 有效且系数有限的模型。
 * @param x 有限输入坐标。
 * @param output 可写预测结果；失败时保持不变。
 * @return 预测有限时为 true。
 */
static bool polynomial_predict(const Polynomial *model, double x, double *output)
{
    if (model == nullptr || output == nullptr || model->degree > MAX_DEGREE || !isfinite(x)) {
        return false;
    }

    double prediction = 0.0;
    for (size_t reverse = model->degree + 1; reverse > 0; --reverse) {
        size_t coefficient = reverse - 1;
        if (!isfinite(model->coefficients[coefficient])) {
            return false;
        }
        prediction = prediction * x + model->coefficients[coefficient];
    }
    if (!isfinite(prediction)) {
        return false;
    }
    *output = prediction;
    return true;
}

/**
 * @brief 计算模型在固定样本上的均方误差。
 * @param model 已拟合的多项式。
 * @param samples 只用于评估的样本数组；不得传给拟合函数。
 * @param count 正数样本数。
 * @param output 可写有限误差结果；失败时保持不变。
 * @return 评估成功时为 true。
 */
static bool mean_squared_error(const Polynomial *model, const Sample *samples, size_t count,
                               double *output)
{
    if (model == nullptr || samples == nullptr || count == 0 || count > MAX_SAMPLE_COUNT ||
        output == nullptr) {
        return false;
    }

    double sum = 0.0;
    for (size_t index = 0; index < count; ++index) {
        double prediction = 0.0;
        if (!isfinite(samples[index].y) ||
            !polynomial_predict(model, samples[index].x, &prediction)) {
            return false;
        }
        double error = prediction - samples[index].y;
        sum += error * error;
    }
    double result = sum / (double)count;
    if (!isfinite(result)) {
        return false;
    }
    *output = result;
    return true;
}

/**
 * @brief 以一组共同索引计算两模型的成对平均损失差。
 * @param loss_a 模型 A 在原测试样本上的有限逐样本损失。
 * @param loss_b 模型 B 在相同样本上的有限逐样本损失。
 * @param test_count 测试样本数，范围为 1 到 MAX_SAMPLE_COUNT。
 * @param indices 本次重采样索引，取值均须小于 test_count。
 * @param resample_count 本次重采样样本数，须大于 0。
 * @param output 可写结果 $mean(loss_A-loss_B)$；失败时保持不变。
 * @return 索引与数值均有效时为 true。
 */
static bool paired_resampled_difference(const double *loss_a, const double *loss_b,
                                        size_t test_count, const size_t *indices,
                                        size_t resample_count, double *output)
{
    if (loss_a == nullptr || loss_b == nullptr || indices == nullptr || output == nullptr ||
        test_count == 0 || test_count > MAX_SAMPLE_COUNT || resample_count == 0 ||
        resample_count > MAX_SAMPLE_COUNT) {
        return false;
    }

    double sum = 0.0;
    for (size_t sample = 0; sample < resample_count; ++sample) {
        size_t index = indices[sample];
        if (index >= test_count || !isfinite(loss_a[index]) || !isfinite(loss_b[index])) {
            return false;
        }
        sum += loss_a[index] - loss_b[index];
    }
    double result = sum / (double)resample_count;
    if (!isfinite(result)) {
        return false;
    }
    *output = result;
    return true;
}

/**
 * @brief 对固定测试集的逐样本损失差执行百分位成对自助法。
 * @param loss_a 模型 A 的逐样本损失。
 * @param loss_b 模型 B 在同一测试样本上的逐样本损失。
 * @param test_count 固定测试集样本数，范围为 1 到 MAX_SAMPLE_COUNT。
 * @param replicate_count 重采样次数，范围为 1 到 BOOTSTRAP_COUNT。
 * @param seed 可复现且非零的随机种子。
 * @param output 可写均值与 95% 百分位区间；失败时保持不变。
 * @return 所有重采样均有效时为 true；区间条件于已拟合模型。
 */
static bool paired_bootstrap(const double *loss_a, const double *loss_b, size_t test_count,
                             size_t replicate_count, uint32_t seed, BootstrapInterval *output)
{
    if (loss_a == nullptr || loss_b == nullptr || output == nullptr || test_count == 0 ||
        test_count > MAX_SAMPLE_COUNT || replicate_count == 0 ||
        replicate_count > BOOTSTRAP_COUNT || seed == 0) {
        return false;
    }

    for (size_t index = 0; index < test_count; ++index) {
        if (!isfinite(loss_a[index]) || !isfinite(loss_b[index])) {
            return false;
        }
    }

    double estimates[BOOTSTRAP_COUNT] = {0};
    size_t indices[MAX_SAMPLE_COUNT] = {0};
    uint32_t state = seed;
    for (size_t replicate = 0; replicate < replicate_count; ++replicate) {
        for (size_t sample = 0; sample < test_count; ++sample) {
            indices[sample] = random_next(&state) % (uint32_t)test_count;
        }
        if (!paired_resampled_difference(loss_a, loss_b, test_count, indices, test_count,
                                         &estimates[replicate])) {
            return false;
        }
    }

    qsort(estimates, replicate_count, sizeof(*estimates), compare_doubles);
    BootstrapInterval next = {0};
    for (size_t index = 0; index < test_count; ++index) {
        next.mean_difference += loss_a[index] - loss_b[index];
    }
    next.mean_difference /= (double)test_count;
    next.lower = estimates[(size_t)(0.025 * (double)(replicate_count - 1))];
    next.upper = estimates[(size_t)(0.975 * (double)(replicate_count - 1))];
    if (!isfinite(next.mean_difference) || !isfinite(next.lower) || !isfinite(next.upper)) {
        return false;
    }
    *output = next;
    return true;
}

/**
 * @brief 为自助法重采样均值按升序排序。
 * @param left 指向第一个有限 double 的指针。
 * @param right 指向第二个有限 double 的指针。
 * @return 比较结果符合 qsort 约定。
 */
static int compare_doubles(const void *left, const void *right)
{
    double a = *(const double *)left;
    double b = *(const double *)right;
    return (a > b) - (a < b);
}

/**
 * @brief 重复拟合不同次数多项式，估计固定探针点上的偏差平方和方差。
 * @param output 可写完整实验结果；任何生成或拟合失败时保持不变。
 * @return 全部拟合、评估与自助法计算成功时为 true。
 */
static bool experiment_run(Experiment *output)
{
    if (output == nullptr) {
        return false;
    }

    double prediction_sum[DEGREE_COUNT][PROBE_COUNT] = {{0}};
    double prediction_square_sum[DEGREE_COUNT][PROBE_COUNT] = {{0}};
    uint32_t repeated_state = UINT32_C(0x1234abcd);
    for (size_t repeat = 0; repeat < REPEAT_COUNT; ++repeat) {
        Sample training[MAX_SAMPLE_COUNT] = {{0}};
        if (!dataset_generate(&repeated_state, 12, -1.0, 1.0, 0.2, training)) {
            return false;
        }
        for (size_t kind = 0; kind < DEGREE_COUNT; ++kind) {
            Polynomial model = {0};
            if (!polynomial_fit(training, 12, DEGREES[kind], &model)) {
                return false;
            }
            for (size_t point = 0; point < PROBE_COUNT; ++point) {
                double prediction = 0.0;
                if (!polynomial_predict(&model, PROBE_X[point], &prediction)) {
                    return false;
                }
                prediction_sum[kind][point] += prediction;
                prediction_square_sum[kind][point] += prediction * prediction;
            }
        }
    }

    Experiment next = {0};
    for (size_t kind = 0; kind < DEGREE_COUNT; ++kind) {
        for (size_t point = 0; point < PROBE_COUNT; ++point) {
            double mean = prediction_sum[kind][point] / (double)REPEAT_COUNT;
            double second_moment = prediction_square_sum[kind][point] / (double)REPEAT_COUNT;
            double bias = mean - target_function(PROBE_X[point]);
            double variance = second_moment - mean * mean;
            if (variance < 0.0 && variance > -1e-12) {
                variance = 0.0;
            }
            if (!isfinite(bias) || !isfinite(variance) || variance < 0.0) {
                return false;
            }
            next.bias_squared[kind] += bias * bias / (double)PROBE_COUNT;
            next.variance[kind] += variance / (double)PROBE_COUNT;
        }
        next.expected_mse[kind] = next.bias_squared[kind] + next.variance[kind];
    }

    Sample training[MAX_SAMPLE_COUNT] = {{0}};
    Sample test[MAX_SAMPLE_COUNT] = {{0}};
    Sample shifted[MAX_SAMPLE_COUNT] = {{0}};
    uint32_t training_state = UINT32_C(0x10203041);
    uint32_t test_state = UINT32_C(0x50607081);
    uint32_t shifted_state = UINT32_C(0x90abcdef);
    if (!dataset_generate(&training_state, 12, -1.0, 1.0, 0.2, training) ||
        !dataset_generate(&test_state, 12, -1.0, 1.0, 0.2, test) ||
        !dataset_generate(&shifted_state, 12, 1.0, 1.5, 0.2, shifted)) {
        return false;
    }

    double loss_a[MAX_SAMPLE_COUNT] = {0};
    double loss_b[MAX_SAMPLE_COUNT] = {0};
    for (size_t kind = 0; kind < DEGREE_COUNT; ++kind) {
        Polynomial model = {0};
        if (!polynomial_fit(training, 12, DEGREES[kind], &model) ||
            !mean_squared_error(&model, test, 12, &next.test_mse[kind]) ||
            !mean_squared_error(&model, shifted, 12, &next.shifted_mse[kind])) {
            return false;
        }
        if (kind == 0 || kind == DEGREE_COUNT - 1) {
            double *loss = kind == 0 ? loss_a : loss_b;
            for (size_t index = 0; index < 12; ++index) {
                double prediction = 0.0;
                if (!polynomial_predict(&model, test[index].x, &prediction)) {
                    return false;
                }
                double error = prediction - test[index].y;
                loss[index] = error * error;
            }
        }
    }
    if (!paired_bootstrap(loss_a, loss_b, 12, BOOTSTRAP_COUNT, UINT32_C(20261028),
                         &next.interval)) {
        return false;
    }
    *output = next;
    return true;
}

/**
 * @brief 将教学结果按稳定字段输出，供两种语言逐项核对。
 * @param experiment 已完成的实验结果。
 * @return 无。
 */
static void print_csv(const Experiment *experiment)
{
    puts("field,value");
    for (size_t kind = 0; kind < DEGREE_COUNT; ++kind) {
        printf("degree%zu_bias_squared,%.17g\n", DEGREES[kind],
               experiment->bias_squared[kind]);
        printf("degree%zu_variance,%.17g\n", DEGREES[kind], experiment->variance[kind]);
        printf("degree%zu_expected_mse,%.17g\n", DEGREES[kind],
               experiment->expected_mse[kind]);
        printf("degree%zu_test_mse,%.17g\n", DEGREES[kind], experiment->test_mse[kind]);
        printf("degree%zu_shifted_mse,%.17g\n", DEGREES[kind],
               experiment->shifted_mse[kind]);
    }
    printf("paired_bootstrap_mean,%.17g\n", experiment->interval.mean_difference);
    printf("paired_bootstrap_lower,%.17g\n", experiment->interval.lower);
    printf("paired_bootstrap_upper,%.17g\n", experiment->interval.upper);
}

/**
 * @brief 验证手算、失败边界、成对索引与可复现实验。
 * @return 所有断言通过时返回 0。
 */
static int run_tests(void)
{
    Sample exact[] = {{-1.0, 0.75 - 1.2 - 0.8}, {0.0, 0.75},
                      {1.0, 0.75 + 1.2 - 0.8}, {2.0, 0.75 + 2.4 - 3.2}};
    Polynomial model = {0};
    assert(polynomial_fit(exact, 4, 2, &model));
    assert(fabs(model.coefficients[0] - 0.75) < 1e-10);
    assert(fabs(model.coefficients[1] - 1.2) < 1e-10);
    assert(fabs(model.coefficients[2] + 0.8) < 1e-10);
    double prediction = -1.0;
    assert(polynomial_predict(&model, 0.5, &prediction));
    assert(fabs(prediction - 1.15) < 1e-10);

    Polynomial unchanged = {.degree = 1, .coefficients = {9.0, 8.0}};
    Sample singular[] = {{1.0, 1.0}, {1.0, 2.0}, {1.0, 3.0}};
    assert(!polynomial_fit(singular, 3, 2, &unchanged));
    assert(unchanged.degree == 1 && unchanged.coefficients[0] == 9.0);
    assert(!polynomial_fit(exact, 2, 2, &unchanged));
    assert(!polynomial_fit(exact, 4, MAX_DEGREE + 1, &unchanged));
    Sample non_finite[] = {{0.0, INFINITY}};
    assert(!polynomial_fit(non_finite, 1, 0, &unchanged));
    assert(!polynomial_predict(&model, NAN, &prediction));

    double loss_a[] = {4.0, 1.0, 10.0};
    double loss_b[] = {1.0, 1.0, 2.0};
    size_t hand_indices[] = {0, 0, 2};
    double hand_mean = 0.0;
    assert(paired_resampled_difference(loss_a, loss_b, 3, hand_indices, 3, &hand_mean));
    assert(fabs(hand_mean - 14.0 / 3.0) < 1e-12);
    BootstrapInterval interval = {.mean_difference = 7.0, .lower = 8.0, .upper = 9.0};
    assert(!paired_bootstrap(loss_a, loss_b, 3, 0, 1, &interval));
    assert(interval.mean_difference == 7.0 && interval.lower == 8.0 && interval.upper == 9.0);
    assert(!paired_resampled_difference(loss_a, loss_b, 3, hand_indices, 3, nullptr));
    size_t bad_indices[] = {0, 3, 2};
    assert(!paired_resampled_difference(loss_a, loss_b, 3, bad_indices, 3, &hand_mean));

    Experiment first = {0};
    Experiment second = {0};
    assert(experiment_run(&first));
    assert(experiment_run(&second));
    for (size_t kind = 0; kind < DEGREE_COUNT; ++kind) {
        assert(first.bias_squared[kind] == second.bias_squared[kind]);
        assert(first.variance[kind] == second.variance[kind]);
        assert(first.test_mse[kind] == second.test_mse[kind]);
        assert(first.shifted_mse[kind] == second.shifted_mse[kind]);
        assert(isfinite(first.bias_squared[kind]) && first.bias_squared[kind] >= 0.0);
        assert(isfinite(first.variance[kind]) && first.variance[kind] >= 0.0);
        assert(isfinite(first.test_mse[kind]) && first.test_mse[kind] >= 0.0);
        assert(isfinite(first.shifted_mse[kind]) && first.shifted_mse[kind] >= 0.0);
    }
    assert(first.interval.mean_difference == second.interval.mean_difference);
    assert(first.interval.lower == second.interval.lower);
    assert(first.interval.upper == second.interval.upper);
    assert(first.interval.lower <= first.interval.upper);
    return 0;
}

/**
 * @brief 运行泛化、偏差方差与成对自助法演示。
 * @param argc 参数数量；只接受无参数、--csv 或 --test。
 * @param argv 命令行参数数组。
 * @return 成功时返回 EXIT_SUCCESS，参数或实验失败时返回 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--csv") != 0 &&
                     strcmp(argv[1], "--test") != 0)) {
        fprintf(stderr, "用法：%s [--csv|--test]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        return run_tests();
    }

    Experiment experiment = {0};
    if (!experiment_run(&experiment)) {
        fprintf(stderr, "泛化实验失败：数据、拟合或统计量无效。\n");
        return EXIT_FAILURE;
    }
    if (argc == 2) {
        print_csv(&experiment);
        return EXIT_SUCCESS;
    }

    printf("多项式模型：每次使用 12 条训练样本，重复拟合 %d 次；次数预先固定为 1、3、5。\n",
           REPEAT_COUNT);
    puts("次数  平均偏差平方  平均方差  偏差平方+方差  固定测试 MSE  偏移测试 MSE");
    for (size_t kind = 0; kind < DEGREE_COUNT; ++kind) {
        printf("%2zu    %.6f      %.6f    %.6f        %.6f       %.6f\n", DEGREES[kind],
               experiment.bias_squared[kind], experiment.variance[kind],
               experiment.expected_mse[kind], experiment.test_mse[kind],
               experiment.shifted_mse[kind]);
    }
    puts("测试集只用于固定模型的最终评估；偏移测试把 x 从训练区间 [-1,1] 移到 [1,1.5]。");
    printf("成对比较（次数 1 - 次数 5 的逐样本平方损失）：均值=%.6f，"
           "95%% 百分位自助区间=[%.6f, %.6f]\n",
           experiment.interval.mean_difference, experiment.interval.lower,
           experiment.interval.upper);
    puts("区间以已拟合模型为条件，只反映有限测试样本不确定性；重复训练波动另由偏差方差实验观察。");
    puts("小样本合成实验用于学习计算流程，不证明真实任务上的模型优劣。");
    return EXIT_SUCCESS;
}
