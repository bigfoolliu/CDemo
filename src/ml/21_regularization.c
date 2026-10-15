/**
 * @file
 * @brief 阶段 10.2：用小样本二维分类观察 L2 正则与过拟合。
 * 所有模型、数据和梯度均使用固定容量数组；仅依赖 C 标准库与数学库。
 */
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FEATURE_COUNT = 2,
    HIDDEN_COUNT = 24,
    TRAIN_COUNT = 24,
    VALIDATION_COUNT = 192,
    TEST_COUNT = 192,
    DATASET_CAPACITY = TRAIN_COUNT > VALIDATION_COUNT ? TRAIN_COUNT : VALIDATION_COUNT,
    PARAMETER_COUNT = HIDDEN_COUNT * FEATURE_COUNT + HIDDEN_COUNT + HIDDEN_COUNT + 1,
    EPOCH_COUNT = 7000,
    TRIAL_COUNT = 3,
    LAMBDA_COUNT = 6
};

static const double LAMBDA_VALUES[LAMBDA_COUNT] = {0.0, 0.00003, 0.0001, 0.001, 0.003, 0.01};

enum {
    W1_OFFSET = 0,
    B1_OFFSET = W1_OFFSET + HIDDEN_COUNT * FEATURE_COUNT,
    W2_OFFSET = B1_OFFSET + HIDDEN_COUNT,
    B2_OFFSET = W2_OFFSET + HIDDEN_COUNT
};

typedef struct {
    double features[FEATURE_COUNT];
    double target;
} Sample;

typedef struct {
    size_t count;
    Sample samples[DATASET_CAPACITY];
} Dataset;

typedef struct {
    double parameters[PARAMETER_COUNT];
} Network;

typedef struct {
    double hidden_linear[HIDDEN_COUNT];
    double hidden[HIDDEN_COUNT];
    double logit;
    double probability;
} Forward;

typedef struct {
    double loss;
    double accuracy;
} Metrics;

/**
 * @brief 生成固定种子的 [0,1) 伪随机数。
 * @param state 可写随机状态，由当前调用独占维护。
 * @return 更新状态映射到 [0,1) 的实数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 生成小样本圆形分类数据，并可只对训练标签加入翻转噪声。
 * @param dataset 有效可写数据集，成功后整体替换。
 * @param count 偶数样本数，不能超过 Dataset 的固定容量。
 * @param seed 固定数据种子；训练、验证和测试使用不同种子。
 * @param flip_probability 标签翻转概率，范围 [0,1]；验证和测试传 0。
 * @return 参数有效为 true；失败时不修改 dataset。
 */
static bool dataset_generate(Dataset *dataset, size_t count, uint32_t seed,
                             double flip_probability)
{
    if (dataset == nullptr || count < 2 || count > DATASET_CAPACITY || count % 2 != 0 ||
        !isfinite(flip_probability) || flip_probability < 0.0 || flip_probability > 1.0) {
        return false;
    }
    Dataset generated = {0};
    generated.count = count;
    for (size_t i = 0; i < count; ++i) {
        double target = (double)(i % 2);
        double angle = 2.0 * acos(-1.0) * random_unit(&seed);
        double radius = target == 0.0 ? 0.36 * sqrt(random_unit(&seed))
                                      : 0.62 + 0.30 * random_unit(&seed);
        double noise_x = 0.08 * (2.0 * random_unit(&seed) - 1.0);
        double noise_y = 0.08 * (2.0 * random_unit(&seed) - 1.0);
        generated.samples[i] = (Sample){
            .features = {radius * cos(angle) + noise_x, radius * sin(angle) + noise_y},
            .target = target};
        if (random_unit(&seed) < flip_probability) {
            generated.samples[i].target = 1.0 - target;
        }
    }
    *dataset = generated;
    return true;
}

/**
 * @brief 使用双曲正切计算一个样本的两层全连接前向结果。
 * @param network 参数只读网络。
 * @param sample 特征有限、标签为 0 或 1 的只读样本。
 * @param output 有效输出地址，成功后写入所有中间激活。
 * @return 输入、参数及结果有效时为 true，否则不修改 output。
 */
static bool forward(const Network *network, const Sample *sample, Forward *output)
{
    if (network == nullptr || sample == nullptr || output == nullptr) {
        return false;
    }
    for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        if (!isfinite(network->parameters[parameter])) {
            return false;
        }
    }
    Forward result = {0};
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(sample->features[feature])) {
            return false;
        }
    }
    if (!isfinite(sample->target) || (sample->target != 0.0 && sample->target != 1.0)) {
        return false;
    }
    for (size_t unit = 0; unit < HIDDEN_COUNT; ++unit) {
        double value = network->parameters[B1_OFFSET + unit];
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            value += network->parameters[W1_OFFSET + unit * FEATURE_COUNT + feature] *
                     sample->features[feature];
        }
        result.hidden_linear[unit] = value;
        result.hidden[unit] = tanh(value);
        result.logit += network->parameters[W2_OFFSET + unit] * result.hidden[unit];
    }
    result.logit += network->parameters[B2_OFFSET];
    if (!isfinite(result.logit)) {
        return false;
    }
    if (result.logit >= 0.0) {
        result.probability = 1.0 / (1.0 + exp(-result.logit));
    } else {
        double exponential = exp(result.logit);
        result.probability = exponential / (1.0 + exponential);
    }
    *output = result;
    return true;
}

/**
 * @brief 计算数值稳定的二元交叉熵，输入为 logit 和二元标签。
 * @param logit 有限实数模型输出。
 * @param target 标签 0 或 1。
 * @return log(1+exp(logit))-target*logit 的稳定计算结果。
 */
static double binary_cross_entropy(double logit, double target)
{
    double softplus = fmax(logit, 0.0) + log1p(exp(-fabs(logit)));
    return softplus - target * logit;
}

/**
 * @brief 求平均数据损失和 L2 权重惩罚，并计算完整解析梯度。
 * @param network 待评估的只读网络。
 * @param dataset 非空只读数据集，特征有限且标签为 0/1。
 * @param lambda 有限非负 L2 系数；目标额外加 lambda/2 * sum(W²)。
 * @param gradient 可选的参数梯度输出；失败时保持原样。
 * @param data_loss 可选的数据交叉熵输出，不含正则项；失败时保持原样。
 * @param objective 可选的总目标输出；失败时保持原样。
 * @return 参数、数据及中间结果均有限时为 true，否则不修改输出。
 */
static bool loss_and_gradient(const Network *network, const Dataset *dataset, double lambda,
                              double gradient[PARAMETER_COUNT], double *data_loss,
                              double *objective)
{
    if (network == nullptr || dataset == nullptr || dataset->count == 0 ||
        dataset->count > DATASET_CAPACITY || !isfinite(lambda) || lambda < 0.0) {
        return false;
    }
    for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        if (!isfinite(network->parameters[parameter])) {
            return false;
        }
    }
    double accumulated[PARAMETER_COUNT] = {0};
    double total_loss = 0.0;
    for (size_t i = 0; i < dataset->count; ++i) {
        const Sample *sample = &dataset->samples[i];
        Forward pass;
        if (!forward(network, sample, &pass)) {
            return false;
        }
        total_loss += binary_cross_entropy(pass.logit, sample->target);
        double output_delta = pass.probability - sample->target;
        accumulated[B2_OFFSET] += output_delta;
        for (size_t unit = 0; unit < HIDDEN_COUNT; ++unit) {
            accumulated[W2_OFFSET + unit] += output_delta * pass.hidden[unit];
            double hidden_delta = output_delta * network->parameters[W2_OFFSET + unit] *
                                  (1.0 - pass.hidden[unit] * pass.hidden[unit]);
            accumulated[B1_OFFSET + unit] += hidden_delta;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                accumulated[W1_OFFSET + unit * FEATURE_COUNT + feature] +=
                    hidden_delta * sample->features[feature];
            }
        }
    }
    double inverse_count = 1.0 / (double)dataset->count;
    double mean_loss = total_loss * inverse_count;
    double penalty = 0.0;
    for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        accumulated[parameter] *= inverse_count;
    }
    for (size_t parameter = W1_OFFSET; parameter < B1_OFFSET; ++parameter) {
        double weight = network->parameters[parameter];
        penalty += weight * weight;
        accumulated[parameter] += lambda * weight;
    }
    for (size_t parameter = W2_OFFSET; parameter < B2_OFFSET; ++parameter) {
        double weight = network->parameters[parameter];
        penalty += weight * weight;
        accumulated[parameter] += lambda * weight;
    }
    double total_objective = mean_loss + 0.5 * lambda * penalty;
    if (!isfinite(mean_loss) || !isfinite(total_objective)) {
        return false;
    }
    for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        if (!isfinite(accumulated[parameter])) {
            return false;
        }
    }
    if (gradient != nullptr) {
        memcpy(gradient, accumulated, sizeof accumulated);
    }
    if (data_loss != nullptr) {
        *data_loss = mean_loss;
    }
    if (objective != nullptr) {
        *objective = total_objective;
    }
    return true;
}

/**
 * @brief 以 Xavier 范围初始化网络，固定初始种子便于 lambda 公平比较。
 * @param network 有效可写网络，成功后全部参数被覆盖。
 * @param seed 固定伪随机种子。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    double limit = sqrt(6.0 / (double)(FEATURE_COUNT + HIDDEN_COUNT));
    for (size_t parameter = W1_OFFSET; parameter < B1_OFFSET; ++parameter) {
        network->parameters[parameter] = limit * (2.0 * random_unit(&seed) - 1.0);
    }
    limit = sqrt(6.0 / (double)(HIDDEN_COUNT + 1));
    for (size_t parameter = W2_OFFSET; parameter < B2_OFFSET; ++parameter) {
        network->parameters[parameter] = limit * (2.0 * random_unit(&seed) - 1.0);
    }
}

/**
 * @brief 对固定训练集执行全批量 SGD；更新前先基于同一组参数计算梯度。
 * @param network 已初始化的可写网络。
 * @param dataset 有效非空训练集。
 * @param lambda 有限非负 L2 系数。
 * @param epochs 正训练轮数。
 * @param learning_rate 有限正学习率。
 * @return 全部更新保持有限时为 true；数值或参数错误时为 false。
 */
static bool train(Network *network, const Dataset *dataset, double lambda, size_t epochs,
                  double learning_rate)
{
    if (network == nullptr || dataset == nullptr || dataset->count == 0 || epochs == 0 ||
        !isfinite(lambda) || lambda < 0.0 || !isfinite(learning_rate) || learning_rate <= 0.0) {
        return false;
    }
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double gradient[PARAMETER_COUNT];
        if (!loss_and_gradient(network, dataset, lambda, gradient, nullptr, nullptr)) {
            return false;
        }
        for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
            double updated = network->parameters[parameter] - learning_rate * gradient[parameter];
            if (!isfinite(updated)) {
                return false;
            }
            network->parameters[parameter] = updated;
        }
    }
    return true;
}

/**
 * @brief 在数据集上计算平均 BCE 和阈值 0.5 的分类准确率。
 * @param network 只读已训练网络。
 * @param dataset 非空只读数据集。
 * @param output 有效可写指标，成功后整体替换。
 * @return 所有样本均成功计算时为 true，否则不修改 output。
 */
static bool evaluate(const Network *network, const Dataset *dataset, Metrics *output)
{
    if (network == nullptr || dataset == nullptr || dataset->count == 0 ||
        dataset->count > DATASET_CAPACITY || output == nullptr) {
        return false;
    }
    Metrics result = {0};
    size_t correct = 0;
    for (size_t i = 0; i < dataset->count; ++i) {
        Forward pass;
        if (!forward(network, &dataset->samples[i], &pass)) {
            return false;
        }
        result.loss += binary_cross_entropy(pass.logit, dataset->samples[i].target);
        if ((pass.probability >= 0.5) == (dataset->samples[i].target == 1.0)) {
            ++correct;
        }
    }
    result.loss /= (double)dataset->count;
    result.accuracy = (double)correct / (double)dataset->count;
    if (!isfinite(result.loss) || !isfinite(result.accuracy)) {
        return false;
    }
    *output = result;
    return true;
}

/**
 * @brief 计算权重的 L2 范数平方，排除两层偏置。
 * @param network 只读网络。
 * @return 所有权重平方和。
 */
static double weight_norm_squared(const Network *network)
{
    double norm = 0.0;
    for (size_t parameter = W1_OFFSET; parameter < B1_OFFSET; ++parameter) {
        norm += network->parameters[parameter] * network->parameters[parameter];
    }
    for (size_t parameter = W2_OFFSET; parameter < B2_OFFSET; ++parameter) {
        norm += network->parameters[parameter] * network->parameters[parameter];
    }
    return norm;
}

/**
 * @brief 比较单种子下不同 L2 系数的训练/验证损失并只评估选中模型一次。
 * @param trial 当前实验编号，仅用于说明输出。
 * @param seed 用于本次开发、验证和独立测试数据的种子基值。
 * @param test_metrics 可写最终测试指标，成功后整体替换。
 * @param selected_lambda 可写验证集选出的正则系数。
 * @return 完成验证集选参和一次测试评估时为 true。
 */
static bool run_trial(size_t trial, uint32_t seed, Metrics *test_metrics,
                      double *selected_lambda)
{
    Dataset training;
    Dataset validation;
    Dataset test_data;
    if (test_metrics == nullptr || selected_lambda == nullptr ||
        !dataset_generate(&training, TRAIN_COUNT, seed, 0.20) ||
        !dataset_generate(&validation, VALIDATION_COUNT, seed + 1000, 0.0) ||
        !dataset_generate(&test_data, TEST_COUNT, seed + 2000, 0.0)) {
        return false;
    }
    printf("\n种子 %u（实验 %zu）：训练 %d 条含噪样本，"
           "验证/测试各 %d 条干净样本\n",
           seed, trial + 1, TRAIN_COUNT, VALIDATION_COUNT);
    double best_validation_loss = INFINITY;
    Network best_network = {0};
    double best_lambda = 0.0;
    for (size_t candidate = 0; candidate < LAMBDA_COUNT; ++candidate) {
        Network network;
        network_initialize(&network, seed + UINT32_C(5000));
        if (!train(&network, &training, LAMBDA_VALUES[candidate], EPOCH_COUNT, 0.12)) {
            return false;
        }
        Metrics training_metrics;
        Metrics validation_metrics;
        double objective;
        if (!evaluate(&network, &training, &training_metrics) ||
            !evaluate(&network, &validation, &validation_metrics) ||
            !loss_and_gradient(&network, &training, LAMBDA_VALUES[candidate], nullptr, nullptr,
                               &objective)) {
            return false;
        }
        printf("lambda=%8.5f  train_BCE=%.4f  train_objective=%.4f  "
               "val_BCE=%.4f  val_accuracy=%.1f%%  ||W||²=%.3f\n",
               LAMBDA_VALUES[candidate], training_metrics.loss, objective,
               validation_metrics.loss, 100.0 * validation_metrics.accuracy,
               weight_norm_squared(&network));
        if (validation_metrics.loss < best_validation_loss) {
            best_validation_loss = validation_metrics.loss;
            best_network = network;
            best_lambda = LAMBDA_VALUES[candidate];
        }
    }
    if (!evaluate(&best_network, &test_data, test_metrics)) {
        return false;
    }
    *selected_lambda = best_lambda;
    printf("验证集选择 lambda=%.4f；封存测试集 BCE=%.4f，准确率=%.1f%%\n",
           best_lambda, test_metrics->loss, 100.0 * test_metrics->accuracy);
    return true;
}

/**
 * @brief 用中心差分检查全部网络参数，并验证惩罚不作用于偏置。
 * @return 所有误差小于阈值时为 true。
 */
static bool test_gradients(void)
{
    Dataset dataset;
    if (!dataset_generate(&dataset, 8, UINT32_C(71), 0.0)) {
        return false;
    }
    Network network;
    network_initialize(&network, UINT32_C(17));
    double analytic[PARAMETER_COUNT];
    double objective;
    if (!loss_and_gradient(&network, &dataset, 0.07, analytic, nullptr, &objective)) {
        return false;
    }
    for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        double original = network.parameters[parameter];
        network.parameters[parameter] = original + 1e-5;
        double plus;
        if (!loss_and_gradient(&network, &dataset, 0.07, nullptr, nullptr, &plus)) {
            return false;
        }
        network.parameters[parameter] = original - 1e-5;
        double minus;
        if (!loss_and_gradient(&network, &dataset, 0.07, nullptr, nullptr, &minus)) {
            return false;
        }
        network.parameters[parameter] = original;
        double numeric = (plus - minus) / 2e-5;
        double scale = fmax(1.0, fmax(fabs(numeric), fabs(analytic[parameter])));
        if (fabs(numeric - analytic[parameter]) / scale > 2e-6) {
            fprintf(stderr, "参数 %zu 梯度不匹配：解析 %.9f，数值 %.9f\n",
                    parameter, analytic[parameter], numeric);
            return false;
        }
    }
    double data_loss;
    if (!loss_and_gradient(&network, &dataset, 0.0, nullptr, &data_loss, &objective) ||
        fabs(data_loss - objective) > 1e-12) {
        return false;
    }
    double regularized[PARAMETER_COUNT];
    double plain[PARAMETER_COUNT];
    if (!loss_and_gradient(&network, &dataset, 0.0, plain, nullptr, nullptr) ||
        !loss_and_gradient(&network, &dataset, 0.07, regularized, nullptr, nullptr)) {
        return false;
    }
    for (size_t parameter = B1_OFFSET; parameter < W2_OFFSET; ++parameter) {
        if (fabs(regularized[parameter] - plain[parameter]) > 1e-12) {
            return false;
        }
    }
    if (fabs(regularized[B2_OFFSET] - plain[B2_OFFSET]) > 1e-12) {
        return false;
    }
    return true;
}

/**
 * @brief 运行无文件副作用的正则化行为与梯度测试。
 * @return 所有断言成立时为 EXIT_SUCCESS。
 */
static int run_tests(void)
{
    if (!test_gradients()) {
        fputs("正则化梯度或偏置排除测试失败\n", stderr);
        return EXIT_FAILURE;
    }
    Dataset first;
    Dataset second;
    if (!dataset_generate(&first, TRAIN_COUNT, UINT32_C(123), 0.20) ||
        !dataset_generate(&second, TRAIN_COUNT, UINT32_C(123), 0.20) ||
        first.count != second.count) {
        fputs("固定种子数据不可复现\n", stderr);
        return EXIT_FAILURE;
    }
    for (size_t i = 0; i < first.count; ++i) {
        if (first.samples[i].features[0] != second.samples[i].features[0] ||
            first.samples[i].features[1] != second.samples[i].features[1] ||
            first.samples[i].target != second.samples[i].target) {
            fputs("固定种子数据不可复现\n", stderr);
            return EXIT_FAILURE;
        }
    }
    if (dataset_generate(&first, TRAIN_COUNT - 1, UINT32_C(123), 0.20) ||
        first.count != TRAIN_COUNT) {
        fputs("非法样本数未被安全拒绝\n", stderr);
        return EXIT_FAILURE;
    }
    Dataset invalid = second;
    invalid.samples[0].features[0] = NAN;
    Network network;
    network_initialize(&network, UINT32_C(17));
    double untouched[PARAMETER_COUNT];
    for (size_t parameter = 0; parameter < PARAMETER_COUNT; ++parameter) {
        untouched[parameter] = -7.0;
    }
    double untouched_loss = -9.0;
    if (loss_and_gradient(&network, &invalid, 0.1, untouched, &untouched_loss, nullptr) ||
        untouched[0] != -7.0 || untouched_loss != -9.0 ||
        loss_and_gradient(&network, &second, -0.1, nullptr, nullptr, nullptr)) {
        fputs("非法输入未被安全拒绝，或失败时修改了输出\n", stderr);
        return EXIT_FAILURE;
    }
    puts("正则化测试通过：全部参数梯度、L2 偏置排除、"
         "固定种子和有限输入边界。");
    return EXIT_SUCCESS;
}

/**
 * @brief 运行多随机种子的正则强度对照实验，或执行无训练单元测试。
 * @param argc 命令行参数数量，只允许无参数或 --test。
 * @param argv 参数数组，借用且不修改。
 * @return 成功为 EXIT_SUCCESS；参数、训练或评估失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        return run_tests();
    }
    if (argc != 1) {
        fputs("用法：21_regularization [--test]\n", stderr);
        return EXIT_FAILURE;
    }
    puts("阶段 10.2：L2 正则对照；数据损失为平均 BCE，"
         "目标另加 lambda/2 * sum(W²)。");
    puts("只用验证 BCE 选择 lambda；训练仅 24 条，标签翻转概率为 20%；"
         "验证与测试标签保持干净。");
    double lambda_sum = 0.0;
    double test_loss_sum = 0.0;
    double test_accuracy_sum = 0.0;
    for (size_t trial = 0; trial < TRIAL_COUNT; ++trial) {
        Metrics metrics;
        double selected_lambda;
        if (!run_trial(trial, UINT32_C(42) + (uint32_t)(trial * 101), &metrics,
                       &selected_lambda)) {
            fputs("正则化训练或评估出现无效数值\n", stderr);
            return EXIT_FAILURE;
        }
        lambda_sum += selected_lambda;
        test_loss_sum += metrics.loss;
        test_accuracy_sum += metrics.accuracy;
    }
    printf("\n三个种子的汇总：选中 lambda 均值=%.4f，测试 BCE 均值=%.4f，"
           "测试准确率均值=%.1f%%\n",
           lambda_sum / TRIAL_COUNT, test_loss_sum / TRIAL_COUNT,
           100.0 * test_accuracy_sum / TRIAL_COUNT);
    puts("测试集只用于最终评估；正则可能降低过拟合风险，"
         "但不保证每次提高测试分数。");
    return !ferror(stdout) ? EXIT_SUCCESS : EXIT_FAILURE;
}
