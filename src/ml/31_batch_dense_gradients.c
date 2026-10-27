/**
 * @file
 * @brief M02：固定二维全连接层的批量前向、反向与一次参数更新。
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FEATURES = 2,
    OUTPUTS = 2,
    MAX_BATCH = 3
};

typedef struct {
    double weights[FEATURES][OUTPUTS]; /* 行是输入特征，列是输出神经元。 */
    double bias[OUTPUTS];
} DenseModel;

typedef struct {
    size_t count;
    double inputs[MAX_BATCH][FEATURES];
    double targets[MAX_BATCH][OUTPUTS];
} DenseBatch;

typedef struct {
    double predictions[MAX_BATCH][OUTPUTS];
    double errors[MAX_BATCH][OUTPUTS];
    double loss;
} DenseForward;

typedef struct {
    double inputs[MAX_BATCH][FEATURES];
    double weights[FEATURES][OUTPUTS];
    double bias[OUTPUTS];
} DenseGradients;

static const DenseModel INITIAL_MODEL = {
    .weights = {{1.0, -1.0}, {2.0, 0.5}},
    .bias = {0.5, -0.5}
};

static const DenseBatch INITIAL_BATCH = {
    .count = 2,
    .inputs = {{1.0, 2.0}, {-1.0, 1.0}},
    .targets = {{2.0, 0.0}, {0.0, 1.0}}
};

/**
 * @brief 验证活动样本及模型参数均有限。
 * @param model 只读模型，不转移所有权。
 * @param batch 只读批次，count 须在 1..MAX_BATCH。
 * @return 输入满足形状和有限数约束时为 true。
 */
static bool dense_valid(const DenseModel *model, const DenseBatch *batch)
{
    if (model == nullptr || batch == nullptr || batch->count == 0 || batch->count > MAX_BATCH) {
        return false;
    }
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            if (!isfinite(model->weights[feature][output])) {
                return false;
            }
        }
    }
    for (size_t output = 0; output < OUTPUTS; ++output) {
        if (!isfinite(model->bias[output])) {
            return false;
        }
    }
    for (size_t sample = 0; sample < batch->count; ++sample) {
        for (size_t feature = 0; feature < FEATURES; ++feature) {
            if (!isfinite(batch->inputs[sample][feature])) {
                return false;
            }
        }
        for (size_t output = 0; output < OUTPUTS; ++output) {
            if (!isfinite(batch->targets[sample][output])) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief 计算 Z=XW+b 及全部元素的平均半平方误差。
 * @param model 有效只读旧参数，不能与 result 重叠。
 * @param batch 有效只读批次，count 在 1..MAX_BATCH，不能与 result 重叠。
 * @param result 可写前向缓存；成功时完全覆盖，失败时保持原值。
 * @return 输入及全部中间结果有限时为 true。
 */
static bool dense_forward(const DenseModel *model, const DenseBatch *batch, DenseForward *result)
{
    if (result == nullptr || !dense_valid(model, batch)) {
        return false;
    }

    DenseForward candidate = {0};
    for (size_t sample = 0; sample < batch->count; ++sample) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            double prediction = model->bias[output];
            for (size_t feature = 0; feature < FEATURES; ++feature) {
                prediction += batch->inputs[sample][feature] * model->weights[feature][output];
            }
            double error = prediction - batch->targets[sample][output];
            double squared = error * error;
            if (!isfinite(prediction) || !isfinite(error) || !isfinite(squared)) {
                return false;
            }
            candidate.predictions[sample][output] = prediction;
            candidate.errors[sample][output] = error;
            candidate.loss += squared;
            if (!isfinite(candidate.loss)) {
                return false;
            }
        }
    }
    candidate.loss /= 2.0 * (double)batch->count * (double)OUTPUTS;
    *result = candidate;
    return true;
}

/**
 * @brief 根据同一批次和旧参数的缓存计算 dX、dW 与 db。
 * @param model 有效只读旧参数，须与 forward 生成时使用的参数相同。
 * @param batch 有效只读批次，须与 forward 生成时使用的批次相同。
 * @param forward 成功的 dense_forward 结果，不与 gradients 重叠。
 * @param gradients 可写结果；成功时完全覆盖，失败时保持原值。
 * @return 输入及梯度有限时为 true；调用者须保证缓存未过期。
 */
static bool dense_backward(const DenseModel *model, const DenseBatch *batch,
                           const DenseForward *forward, DenseGradients *gradients)
{
    if (gradients == nullptr || forward == nullptr || !dense_valid(model, batch) ||
        !isfinite(forward->loss)) {
        return false;
    }

    DenseGradients candidate = {0};
    double denominator = (double)batch->count * (double)OUTPUTS;
    for (size_t sample = 0; sample < batch->count; ++sample) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            if (!isfinite(forward->errors[sample][output])) {
                return false;
            }
            double delta = forward->errors[sample][output] / denominator;
            candidate.bias[output] += delta;
            for (size_t feature = 0; feature < FEATURES; ++feature) {
                candidate.weights[feature][output] += batch->inputs[sample][feature] * delta;
                candidate.inputs[sample][feature] += model->weights[feature][output] * delta;
            }
        }
    }
    for (size_t sample = 0; sample < batch->count; ++sample) {
        for (size_t feature = 0; feature < FEATURES; ++feature) {
            if (!isfinite(candidate.inputs[sample][feature])) {
                return false;
            }
        }
    }
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            if (!isfinite(candidate.weights[feature][output])) {
                return false;
            }
        }
    }
    for (size_t output = 0; output < OUTPUTS; ++output) {
        if (!isfinite(candidate.bias[output])) {
            return false;
        }
    }
    *gradients = candidate;
    return true;
}

/**
 * @brief 使用同一版本参数下的梯度执行一次梯度下降。
 * @param model 可写参数，失败时保持原值。
 * @param gradients 有限的权重与偏置梯度，不转移所有权。
 * @param rate 有限正学习率，更新结果须有限。
 * @return 更新完成时为 true，否则为 false。
 */
static bool dense_update(DenseModel *model, const DenseGradients *gradients, double rate)
{
    if (model == nullptr || gradients == nullptr || !isfinite(rate) || rate <= 0) {
        return false;
    }
    DenseModel candidate = *model;
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            if (!isfinite(gradients->weights[feature][output])) {
                return false;
            }
            candidate.weights[feature][output] -= rate * gradients->weights[feature][output];
            if (!isfinite(candidate.weights[feature][output])) {
                return false;
            }
        }
    }
    for (size_t output = 0; output < OUTPUTS; ++output) {
        if (!isfinite(gradients->bias[output])) {
            return false;
        }
        candidate.bias[output] -= rate * gradients->bias[output];
        if (!isfinite(candidate.bias[output])) {
            return false;
        }
    }
    *model = candidate;
    return true;
}

/**
 * @brief 计算测试候选模型和批次的损失。
 * @param model 有效只读参数。
 * @param batch 有效只读批次。
 * @return 有限损失；测试夹具保证调用成功。
 */
static double loss_for(const DenseModel *model, const DenseBatch *batch)
{
    DenseForward forward = {0};
    assert(dense_forward(model, batch, &forward));
    return forward.loss;
}

/**
 * @brief 比较解析梯度和中心差分，允许随数值大小变化的小误差。
 * @param analytic 有限解析梯度。
 * @param numerical 有限中心差分梯度。
 */
static void assert_gradient(double analytic, double numerical)
{
    assert(isfinite(analytic) && isfinite(numerical));
    assert(fabs(analytic - numerical) <= 1e-8 + 1e-6 * fmax(fabs(analytic), fabs(numerical)));
}

/** @brief 核对手算、全部十个梯度、均值口径及失败边界。 */
static void test_dense(void)
{
    DenseForward forward = {0};
    DenseGradients gradients = {0};
    assert(dense_forward(&INITIAL_MODEL, &INITIAL_BATCH, &forward));
    assert(dense_backward(&INITIAL_MODEL, &INITIAL_BATCH, &forward, &gradients));
    assert(forward.predictions[0][0] == 5.5 && forward.predictions[0][1] == -0.5);
    assert(forward.predictions[1][0] == 1.5 && forward.predictions[1][1] == 1.0);
    assert(forward.loss == 1.84375);
    assert(gradients.inputs[0][0] == 1.0 && gradients.inputs[0][1] == 1.6875);
    assert(gradients.inputs[1][0] == 0.375 && gradients.inputs[1][1] == 0.75);
    assert(gradients.weights[0][0] == 0.5 && gradients.weights[0][1] == -0.125);
    assert(gradients.weights[1][0] == 2.125 && gradients.weights[1][1] == -0.25);
    assert(gradients.bias[0] == 1.25 && gradients.bias[1] == -0.125);

    const double step = 1e-5;
    for (size_t sample = 0; sample < INITIAL_BATCH.count; ++sample) {
        for (size_t feature = 0; feature < FEATURES; ++feature) {
            DenseBatch plus = INITIAL_BATCH;
            DenseBatch minus = INITIAL_BATCH;
            plus.inputs[sample][feature] += step;
            minus.inputs[sample][feature] -= step;
            double numerical = (loss_for(&INITIAL_MODEL, &plus) -
                                loss_for(&INITIAL_MODEL, &minus)) / (2.0 * step);
            assert_gradient(gradients.inputs[sample][feature], numerical);
        }
    }
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            DenseModel plus = INITIAL_MODEL;
            DenseModel minus = INITIAL_MODEL;
            plus.weights[feature][output] += step;
            minus.weights[feature][output] -= step;
            double numerical = (loss_for(&plus, &INITIAL_BATCH) -
                                loss_for(&minus, &INITIAL_BATCH)) / (2.0 * step);
            assert_gradient(gradients.weights[feature][output], numerical);
        }
    }
    for (size_t output = 0; output < OUTPUTS; ++output) {
        DenseModel plus = INITIAL_MODEL;
        DenseModel minus = INITIAL_MODEL;
        plus.bias[output] += step;
        minus.bias[output] -= step;
        double numerical = (loss_for(&plus, &INITIAL_BATCH) -
                            loss_for(&minus, &INITIAL_BATCH)) / (2.0 * step);
        assert_gradient(gradients.bias[output], numerical);
    }

    DenseBatch one = INITIAL_BATCH;
    one.count = 1;
    DenseForward one_forward = {0};
    DenseGradients one_gradients = {0};
    assert(dense_forward(&INITIAL_MODEL, &one, &one_forward));
    assert(dense_backward(&INITIAL_MODEL, &one, &one_forward, &one_gradients));
    DenseBatch duplicated = one;
    duplicated.count = 2;
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        duplicated.inputs[1][feature] = duplicated.inputs[0][feature];
    }
    for (size_t output = 0; output < OUTPUTS; ++output) {
        duplicated.targets[1][output] = duplicated.targets[0][output];
    }
    DenseForward duplicate_forward = {0};
    DenseGradients duplicate_gradients = {0};
    assert(dense_forward(&INITIAL_MODEL, &duplicated, &duplicate_forward));
    assert(dense_backward(&INITIAL_MODEL, &duplicated, &duplicate_forward, &duplicate_gradients));
    assert(duplicate_forward.loss == one_forward.loss);
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            assert(duplicate_gradients.weights[feature][output] ==
                   one_gradients.weights[feature][output]);
        }
        assert(duplicate_gradients.inputs[0][feature] ==
               one_gradients.inputs[0][feature] / 2.0);
        assert(duplicate_gradients.inputs[1][feature] ==
               one_gradients.inputs[0][feature] / 2.0);
    }
    for (size_t output = 0; output < OUTPUTS; ++output) {
        assert(duplicate_gradients.bias[output] == one_gradients.bias[output]);
    }

    DenseModel updated = INITIAL_MODEL;
    assert(dense_update(&updated, &gradients, 0.1));
    assert(updated.weights[0][0] == 0.95);
    assert(updated.weights[0][1] == -0.9875);
    assert(updated.weights[1][0] == 1.7875);
    assert(updated.weights[1][1] == 0.525);
    assert(updated.bias[0] == 0.375 && updated.bias[1] == -0.4875);
    assert(fabs(loss_for(&updated, &INITIAL_BATCH) - 1.25767578125) < 1e-12);

    DenseForward sentinel = {.loss = 99.0};
    DenseBatch invalid = INITIAL_BATCH;
    invalid.count = 0;
    assert(!dense_forward(&INITIAL_MODEL, &invalid, &sentinel));
    invalid.count = MAX_BATCH + 1;
    assert(!dense_forward(&INITIAL_MODEL, &invalid, &sentinel));
    invalid = INITIAL_BATCH;
    invalid.inputs[0][0] = NAN;
    assert(!dense_forward(&INITIAL_MODEL, &invalid, &sentinel));
    invalid = INITIAL_BATCH;
    invalid.inputs[0][0] = 1e308;
    assert(!dense_forward(&INITIAL_MODEL, &invalid, &sentinel));
    assert(!dense_forward(nullptr, &INITIAL_BATCH, &sentinel));
    assert(!dense_forward(&INITIAL_MODEL, &INITIAL_BATCH, nullptr));
    assert(sentinel.loss == 99.0);

    DenseGradients gradient_sentinel = {.bias = {99.0, 98.0}};
    DenseForward bad_forward = forward;
    bad_forward.errors[0][0] = NAN;
    assert(!dense_backward(&INITIAL_MODEL, &INITIAL_BATCH, &bad_forward,
                           &gradient_sentinel));
    assert(gradient_sentinel.bias[0] == 99.0 && gradient_sentinel.bias[1] == 98.0);
    DenseModel model_sentinel = INITIAL_MODEL;
    assert(!dense_update(&model_sentinel, &gradients, 0.0));
    assert(!dense_update(&model_sentinel, &gradients, NAN));
    DenseGradients invalid_gradient = gradients;
    invalid_gradient.weights[0][0] = INFINITY;
    assert(!dense_update(&model_sentinel, &invalid_gradient, 0.1));
    assert(model_sentinel.weights[0][0] == INITIAL_MODEL.weights[0][0]);
}

/**
 * @brief 打印固定夹具的前向、梯度和一次更新，供手算与 Python 对照。
 * @param csv 为 true 时打印 field,value CSV；否则打印可读等式。
 * @return 数值有效为 EXIT_SUCCESS，否则为 EXIT_FAILURE。
 */
static int print_example(bool csv)
{
    DenseForward forward = {0};
    DenseGradients gradients = {0};
    DenseModel updated = INITIAL_MODEL;
    DenseForward after = {0};
    if (!dense_forward(&INITIAL_MODEL, &INITIAL_BATCH, &forward) ||
        !dense_backward(&INITIAL_MODEL, &INITIAL_BATCH, &forward, &gradients) ||
        !dense_update(&updated, &gradients, 0.1) ||
        !dense_forward(&updated, &INITIAL_BATCH, &after)) {
        fputs("固定示例计算失败。\n", stderr);
        return EXIT_FAILURE;
    }

    if (csv) {
        puts("field,value");
    }
#define PRINT_VALUE(label, value) \
    printf(csv ? "%s,%.17g\n" : "%-14s = %.10f\n", label, value)
    for (size_t sample = 0; sample < INITIAL_BATCH.count; ++sample) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            char label[32];
            snprintf(label, sizeof(label), "z_%zu_%zu", sample, output);
            PRINT_VALUE(label, forward.predictions[sample][output]);
        }
    }
    PRINT_VALUE("loss", forward.loss);
    for (size_t sample = 0; sample < INITIAL_BATCH.count; ++sample) {
        for (size_t feature = 0; feature < FEATURES; ++feature) {
            char label[32];
            snprintf(label, sizeof(label), "dx_%zu_%zu", sample, feature);
            PRINT_VALUE(label, gradients.inputs[sample][feature]);
        }
    }
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        for (size_t output = 0; output < OUTPUTS; ++output) {
            char label[32];
            snprintf(label, sizeof(label), "dw_%zu_%zu", feature, output);
            PRINT_VALUE(label, gradients.weights[feature][output]);
            snprintf(label, sizeof(label), "w_next_%zu_%zu", feature, output);
            PRINT_VALUE(label, updated.weights[feature][output]);
        }
    }
    for (size_t output = 0; output < OUTPUTS; ++output) {
        char label[32];
        snprintf(label, sizeof(label), "db_%zu", output);
        PRINT_VALUE(label, gradients.bias[output]);
        snprintf(label, sizeof(label), "b_next_%zu", output);
        PRINT_VALUE(label, updated.bias[output]);
    }
    PRINT_VALUE("loss_next", after.loss);
#undef PRINT_VALUE
    return EXIT_SUCCESS;
}

/**
 * @brief 运行固定演示或短测试。
 * @param argc 参数个数；只接受零或一个开关。
 * @param argv 参数数组；--csv 为机器对照，--test 为断言测试。
 * @return 成功为 EXIT_SUCCESS，非法参数或计算失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_dense();
        puts("批量全连接层求导测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1) {
        return print_example(false);
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        return print_example(true);
    }
    fputs("用法：31_batch_dense_gradients [--csv | --test]\n", stderr);
    return EXIT_FAILURE;
}
