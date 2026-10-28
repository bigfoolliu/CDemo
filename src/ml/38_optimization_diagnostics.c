/**
 * @file
 * @brief M09：二维二次目标上的 SGD、动量、Adam 与小型 MLP 诊断。
 */
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { PARAMETER_COUNT = 9, SAMPLE_COUNT = 3 };

typedef enum { SGD, MOMENTUM, ADAM } OptimizerKind;

typedef struct {
    double velocity[2];
    double first[2];
    double second[2];
    size_t step;
} OptimizerState;

typedef struct {
    double parameters[PARAMETER_COUNT];
} TinyNetwork;

typedef struct {
    double loss;
    double activation_minimum;
    double activation_maximum;
    double activation_mean;
} MlpMetrics;

static const double MLP_INPUTS[SAMPLE_COUNT][2] = {{-1, 0}, {0, 1}, {1, 0}};
static const double MLP_TARGETS[SAMPLE_COUNT] = {0, 1, 0};
static const TinyNetwork MLP_INITIAL = {{0.2, -0.3, 0.1, 0.25, 0.1, -0.2, 0.05, 0.3, 0}};

/**
 * @brief 计算各向异性二维二次目标及梯度。
 * @param point 两个有限参数。
 * @param loss 可写有限损失输出。
 * @param gradient 可写长度为 2 的梯度。
 * @return 结果有限时为 true，失败时不修改输出。
 */
static bool quadratic_evaluate(const double point[2], double *loss, double gradient[2])
{
    if (point == nullptr || loss == nullptr || gradient == nullptr ||
        !isfinite(point[0]) || !isfinite(point[1])) {
        return false;
    }
    double next_loss = 0.5 * (point[0] * point[0] + 9 * point[1] * point[1]);
    double next_gradient[2] = {point[0], 9 * point[1]};
    if (!isfinite(next_loss) || !isfinite(next_gradient[0]) || !isfinite(next_gradient[1])) {
        return false;
    }
    *loss = next_loss;
    memcpy(gradient, next_gradient, sizeof(next_gradient));
    return true;
}

/**
 * @brief 对二维参数执行一步 SGD、经典动量或 Adam，失败时状态不变。
 * @param point 可写长度为 2 的参数；成功后原位更新。
 * @param gradient 长度为 2 的有限梯度。
 * @param kind 优化器种类。
 * @param rate 有限正学习率。
 * @param state 优化器状态；SGD 可传空指针。
 * @return 更新有限时为 true；参数、状态或输入无效时为 false。
 */
static bool optimizer_step(double point[2], const double gradient[2], OptimizerKind kind,
                           double rate, OptimizerState *state)
{
    if (point == nullptr || gradient == nullptr || !isfinite(rate) || rate <= 0 ||
        kind < SGD || kind > ADAM || (kind != SGD && state == nullptr)) {
        return false;
    }
    OptimizerState next_state = {0};
    if (state != nullptr) {
        next_state = *state;
    }
    double next_point[2] = {0};
    if (kind == ADAM) {
        ++next_state.step;
    }
    for (size_t i = 0; i < 2; ++i) {
        if (!isfinite(point[i]) || !isfinite(gradient[i])) {
            return false;
        }
        double direction = gradient[i];
        if (kind == MOMENTUM) {
            next_state.velocity[i] = 0.5 * next_state.velocity[i] + gradient[i];
            direction = next_state.velocity[i];
        } else if (kind == ADAM) {
            next_state.first[i] = 0.9 * next_state.first[i] + 0.1 * gradient[i];
            next_state.second[i] = 0.999 * next_state.second[i] +
                                   0.001 * gradient[i] * gradient[i];
            double first_corrected = next_state.first[i] /
                                     (1 - pow(0.9, (double)next_state.step));
            double second_corrected = next_state.second[i] /
                                      (1 - pow(0.999, (double)next_state.step));
            direction = first_corrected / (sqrt(second_corrected) + 1e-8);
        }
        next_point[i] = point[i] - rate * direction;
        if (!isfinite(next_point[i])) {
            return false;
        }
    }
    memcpy(point, next_point, sizeof(next_point));
    if (state != nullptr) {
        *state = next_state;
    }
    return true;
}

/**
 * @brief 计算三样本两隐藏单元 tanh 网络的均方半损失和诊断量。
 * @param network 有效只读九参数网络。
 * @param metrics 可写结果；失败时保持原值。
 * @param gradients 可写九参数平均梯度，允许为空以仅评估。
 * @return 前向、反向结果均有限时为 true。
 */
static bool mlp_evaluate(const TinyNetwork *network, MlpMetrics *metrics, double *gradients)
{
    if (network == nullptr || metrics == nullptr) {
        return false;
    }
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
        if (!isfinite(network->parameters[i])) {
            return false;
        }
    }
    MlpMetrics next_metrics = {.activation_minimum = DBL_MAX,
                               .activation_maximum = -DBL_MAX};
    double next_gradients[PARAMETER_COUNT] = {0};
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        double hidden[2] = {0};
        for (size_t unit = 0; unit < 2; ++unit) {
            size_t offset = unit * 2;
            double z = network->parameters[offset] * MLP_INPUTS[sample][0] +
                       network->parameters[offset + 1] * MLP_INPUTS[sample][1] +
                       network->parameters[4 + unit];
            hidden[unit] = tanh(z);
            next_metrics.activation_minimum = fmin(next_metrics.activation_minimum, hidden[unit]);
            next_metrics.activation_maximum = fmax(next_metrics.activation_maximum, hidden[unit]);
            next_metrics.activation_mean += hidden[unit];
        }
        double prediction = network->parameters[6] * hidden[0] +
                            network->parameters[7] * hidden[1] + network->parameters[8];
        double error = prediction - MLP_TARGETS[sample];
        next_metrics.loss += 0.5 * error * error / SAMPLE_COUNT;
        if (gradients != nullptr) {
            double delta = error / SAMPLE_COUNT;
            next_gradients[6] += delta * hidden[0];
            next_gradients[7] += delta * hidden[1];
            next_gradients[8] += delta;
            for (size_t unit = 0; unit < 2; ++unit) {
                size_t offset = unit * 2;
                double hidden_delta = delta * network->parameters[6 + unit] *
                                      (1 - hidden[unit] * hidden[unit]);
                next_gradients[offset] += hidden_delta * MLP_INPUTS[sample][0];
                next_gradients[offset + 1] += hidden_delta * MLP_INPUTS[sample][1];
                next_gradients[4 + unit] += hidden_delta;
            }
        }
    }
    next_metrics.activation_mean /= 2 * SAMPLE_COUNT;
    if (!isfinite(next_metrics.loss) || !isfinite(next_metrics.activation_mean)) {
        return false;
    }
    if (gradients != nullptr) {
        for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
            if (!isfinite(next_gradients[i])) {
                return false;
            }
        }
        memcpy(gradients, next_gradients, sizeof(next_gradients));
    }
    *metrics = next_metrics;
    return true;
}

/**
 * @brief 执行一次全批量 MLP 的 SGD 更新并返回梯度范数与更新范数。
 * @param network 可写九参数网络，失败时保持不变。
 * @param rate 有限正学习率。
 * @param gradient_norm 可写梯度二范数输出。
 * @param update_norm 可写参数改变量二范数输出。
 * @return 更新成功且诊断量有限时为 true。
 */
static bool mlp_train_step(TinyNetwork *network, double rate,
                           double *gradient_norm, double *update_norm)
{
    if (network == nullptr || gradient_norm == nullptr || update_norm == nullptr ||
        !isfinite(rate) || rate <= 0) {
        return false;
    }
    TinyNetwork next = *network;
    MlpMetrics metrics = {0};
    double gradients[PARAMETER_COUNT] = {0};
    if (!mlp_evaluate(&next, &metrics, gradients)) {
        return false;
    }
    double grad_squared = 0;
    double change_squared = 0;
    for (size_t i = 0; i < PARAMETER_COUNT; ++i) {
        grad_squared += gradients[i] * gradients[i];
        next.parameters[i] -= rate * gradients[i];
        double change = next.parameters[i] - network->parameters[i];
        change_squared += change * change;
        if (!isfinite(next.parameters[i]) || !isfinite(grad_squared) ||
            !isfinite(change_squared)) {
            return false;
        }
    }
    double next_gradient_norm = sqrt(grad_squared);
    double next_update_norm = sqrt(change_squared);
    if (!isfinite(next_gradient_norm) || !isfinite(next_update_norm)) {
        return false;
    }
    *network = next;
    *gradient_norm = next_gradient_norm;
    *update_norm = next_update_norm;
    return true;
}

/**
 * @brief 检查手算更新、MLP 梯度、数值边界、权重衰减差异及失败回滚。
 */
static void test_optimization(void)
{
    double point[2] = {1, 1};
    double loss = 0;
    double gradient[2] = {0};
    OptimizerState state = {0};
    assert(quadratic_evaluate(point, &loss, gradient));
    assert(loss == 5 && gradient[0] == 1 && gradient[1] == 9);

    double invalid_point[2] = {NAN, 1};
    double preserved_loss = 13;
    double preserved_gradient[2] = {17, 19};
    assert(!quadratic_evaluate(invalid_point, &preserved_loss, preserved_gradient));
    assert(preserved_loss == 13 && preserved_gradient[0] == 17 &&
           preserved_gradient[1] == 19);
    assert(!quadratic_evaluate(nullptr, &preserved_loss, preserved_gradient));

    assert(optimizer_step(point, gradient, MOMENTUM, 0.1, &state));
    assert(fabs(point[0] - 0.9) < 1e-12 && fabs(point[1] - 0.1) < 1e-12);
    assert(quadratic_evaluate(point, &loss, gradient));
    assert(fabs(gradient[0] - 0.9) < 1e-12 && fabs(gradient[1] - 0.9) < 1e-12);
    assert(optimizer_step(point, gradient, MOMENTUM, 0.1, &state));
    assert(fabs(point[0] - 0.76) < 1e-12 && fabs(point[1] + 0.44) < 1e-12);

    double sgd_point[2] = {1, 1};
    assert(quadratic_evaluate(sgd_point, &loss, gradient));
    assert(optimizer_step(sgd_point, gradient, SGD, 0.1, nullptr));
    assert(fabs(sgd_point[0] - 0.9) < 1e-12 && fabs(sgd_point[1] - 0.1) < 1e-12);
    assert(quadratic_evaluate(sgd_point, &loss, gradient));
    assert(optimizer_step(sgd_point, gradient, SGD, 0.1, nullptr));
    assert(fabs(sgd_point[0] - 0.81) < 1e-12 && fabs(sgd_point[1] - 0.01) < 1e-12);

    double adam_point[2] = {1, 1};
    OptimizerState adam_state = {0};
    assert(quadratic_evaluate(adam_point, &loss, gradient));
    assert(optimizer_step(adam_point, gradient, ADAM, 0.1, &adam_state));
    assert(fabs(adam_point[0] - 0.9) < 1e-8 && fabs(adam_point[1] - 0.9) < 1e-8);
    assert(quadratic_evaluate(adam_point, &loss, gradient));
    assert(optimizer_step(adam_point, gradient, ADAM, 0.1, &adam_state));
    assert(fabs(adam_point[0] - 0.80041223) < 1e-8);
    assert(fabs(adam_point[1] - 0.80041223) < 1e-8);

    const double epsilon = 1e-5;
    double mlp_gradients[PARAMETER_COUNT] = {0};
    TinyNetwork network = MLP_INITIAL;
    MlpMetrics metrics = {0};
    assert(mlp_evaluate(&network, &metrics, mlp_gradients));
    TinyNetwork invalid_network = MLP_INITIAL;
    invalid_network.parameters[0] = NAN;
    MlpMetrics preserved_metrics = {.loss = 13};
    double preserved_mlp_gradients[PARAMETER_COUNT] = {17};
    assert(!mlp_evaluate(&invalid_network, &preserved_metrics, preserved_mlp_gradients));
    assert(preserved_metrics.loss == 13 && preserved_mlp_gradients[0] == 17);
    for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        TinyNetwork plus = network;
        TinyNetwork minus = network;
        plus.parameters[parameter] += epsilon;
        minus.parameters[parameter] -= epsilon;
        MlpMetrics plus_metrics = {0};
        MlpMetrics minus_metrics = {0};
        assert(mlp_evaluate(&plus, &plus_metrics, nullptr));
        assert(mlp_evaluate(&minus, &minus_metrics, nullptr));
        double numerical = (plus_metrics.loss - minus_metrics.loss) / (2 * epsilon);
        assert(fabs(numerical - mlp_gradients[parameter]) < 1e-8);
    }
    double old_loss = metrics.loss;
    double grad_norm = 0;
    double update_norm = 0;
    for (size_t step = 0; step < 40; ++step) {
        assert(mlp_train_step(&network, 0.1, &grad_norm, &update_norm));
    }
    assert(mlp_evaluate(&network, &metrics, nullptr));
    assert(metrics.loss < old_loss);
    assert(grad_norm > 0 && update_norm > 0);
    assert(metrics.activation_minimum <= metrics.activation_maximum);

    double unstable[2] = {1, 1};
    double previous = 0;
    assert(quadratic_evaluate(unstable, &previous, gradient));
    assert(optimizer_step(unstable, gradient, SGD, 1, nullptr));
    assert(quadratic_evaluate(unstable, &loss, gradient));
    assert(unstable[0] == 0 && unstable[1] == -8 && loss > previous);
    for (size_t step = 0; step < 2; ++step) {
        double previous_y = unstable[1];
        previous = loss;
        assert(optimizer_step(unstable, gradient, SGD, 1, nullptr));
        assert(quadratic_evaluate(unstable, &loss, gradient));
        assert(unstable[1] * previous_y < 0 && loss > previous);
    }
    double saved_point[2] = {unstable[0], unstable[1]};
    OptimizerState saved_state = {0};
    assert(!optimizer_step(unstable, gradient, MOMENTUM, DBL_MAX, &saved_state));
    assert(unstable[0] == saved_point[0] && unstable[1] == saved_point[1]);
    assert(saved_state.step == 0);
    assert(!optimizer_step(unstable, gradient, SGD, NAN, nullptr));

    double coupled[2] = {2, 0};
    double decoupled[2] = {2, 0};
    OptimizerState coupled_state = {0};
    OptimizerState decoupled_state = {0};
    for (size_t step = 0; step < 2; ++step) {
        double coupled_gradient[2] = {0.5 + 0.1 * coupled[0], 0};
        double task_gradient[2] = {0.5, 0};
        double old_decoupled_parameter = decoupled[0];
        assert(optimizer_step(coupled, coupled_gradient, ADAM, 0.01, &coupled_state));
        assert(optimizer_step(decoupled, task_gradient, ADAM, 0.01, &decoupled_state));
        decoupled[0] -= 0.01 * 0.1 * old_decoupled_parameter;
    }
    assert(fabs(coupled[0] - decoupled[0]) > 1e-7);
}

/**
 * @brief 打印相同初值预算下的优化轨迹及 MLP 训练诊断。
 * @param csv 为 true 时输出 field,value 便于 C/Python 对照。
 * @return 成功为 EXIT_SUCCESS，否则为 EXIT_FAILURE。
 */
static int print_demo(bool csv)
{
    const OptimizerKind kinds[] = {SGD, MOMENTUM, ADAM};
    const char *names[] = {"SGD", "Momentum", "Adam"};
    const char *csv_names[] = {"sgd", "momentum", "adam"};
    if (csv) {
        puts("field,value");
    }
    for (size_t optimizer = 0; optimizer < 3; ++optimizer) {
        double point[2] = {1, 1};
        OptimizerState state = {0};
        for (size_t step = 0; step < 6; ++step) {
            double loss = 0;
            double gradient[2] = {0};
            if (!quadratic_evaluate(point, &loss, gradient)) {
                return EXIT_FAILURE;
            }
            if (csv) {
                printf("%s_%zu,%.17g\n", csv_names[optimizer], step, loss);
            } else if (step == 0 || step == 1 || step == 5) {
                printf("%-8s 步=%zu，点=(%+.5f,%+.5f)，损失=%.6f\n",
                       names[optimizer], step, point[0], point[1], loss);
            }
            if (!optimizer_step(point, gradient, kinds[optimizer], 0.1,
                                optimizer == 0 ? nullptr : &state)) {
                return EXIT_FAILURE;
            }
        }
    }
    TinyNetwork network = MLP_INITIAL;
    for (size_t step = 0; step <= 40; ++step) {
        MlpMetrics metrics = {0};
        double gradients[PARAMETER_COUNT] = {0};
        if (!mlp_evaluate(&network, &metrics, gradients)) {
            return EXIT_FAILURE;
        }
        double gradient_norm = 0;
        double update_norm = 0;
        if (step < 40 && !mlp_train_step(&network, 0.1, &gradient_norm, &update_norm)) {
            return EXIT_FAILURE;
        }
        if (step == 40) {
            double gradient_squared = 0;
            for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
                gradient_squared += gradients[parameter] * gradients[parameter];
            }
            gradient_norm = sqrt(gradient_squared);
        }
        if (csv) {
            printf("mlp_loss_%zu,%.17g\n", step, metrics.loss);
            printf("mlp_activation_mean_%zu,%.17g\n", step, metrics.activation_mean);
            printf("mlp_activation_min_%zu,%.17g\n", step, metrics.activation_minimum);
            printf("mlp_activation_max_%zu,%.17g\n", step, metrics.activation_maximum);
            printf("mlp_gradient_norm_%zu,%.17g\n", step, gradient_norm);
            printf("mlp_update_norm_%zu,%.17g\n", step, update_norm);
        } else if (step == 0 || step == 1 || step == 2 || step == 10 || step == 20 || step == 40) {
            printf("MLP 步=%zu，损失=%.6f，激活均值=%.4f，范围=[%.4f,%.4f]，"
                   "梯度范数=%.5f，更新范数=%.5f\n",
                   step, metrics.loss, metrics.activation_mean, metrics.activation_minimum,
                   metrics.activation_maximum, gradient_norm, update_norm);
        }
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行 M09 优化器与训练诊断案例。
 * @param argc 参数数量，可使用 --test 或 --csv。
 * @param argv 命令行参数。
 * @return 成功为 EXIT_SUCCESS，参数或数值失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_optimization();
        puts("优化与诊断测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "--csv") == 0)) {
        return print_demo(argc == 2);
    }
    fputs("用法：38_optimization_diagnostics [--test | --csv]\n", stderr);
    return EXIT_FAILURE;
}
