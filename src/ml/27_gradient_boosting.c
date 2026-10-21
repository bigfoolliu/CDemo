/**
 * @file
 * @brief 用平方误差和回归树桩实现可逐轮观察的梯度提升回归。
 */
#include <float.h>
#include <math.h>
#include <stdio.h>

enum {
    SAMPLE_COUNT = 12,
    MAX_ROUND_COUNT = 6
};

static const double LEARNING_RATE = 0.5;
static const size_t MAX_TRAINING_ROUNDS = 4;

typedef enum {
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_TEST
} DataSplit;

typedef struct {
    double feature;
    double target;
    DataSplit split;
} Sample;

typedef struct {
    double threshold;
    double left_value;
    double right_value;
    bool has_split;
} RegressionStump;

typedef struct {
    double base_value;
    size_t round_count;
    RegressionStump stumps[MAX_ROUND_COUNT];
} BoostedModel;

typedef struct {
    double mean_squared_error;
    size_t count;
} Metrics;

/**
 * @brief 构造目标随特征分段变化的固定数据集。
 * @param samples 至少 SAMPLE_COUNT 个可写位置；成功时整体写入固定样本。
 * @return 输出指针有效时为 true。
 */
static bool dataset_generate(Sample samples[SAMPLE_COUNT])
{
    if (samples == nullptr) {
        return false;
    }

    static const double targets[SAMPLE_COUNT] = {1.0, 1.0, 2.0, 2.0, 4.0, 4.0,
                                                  5.0, 5.0, 5.0, 6.0, 7.0, 7.0};
    Sample generated[SAMPLE_COUNT] = {{0}};
    for (size_t index = 0; index < SAMPLE_COUNT; ++index) {
        generated[index].feature = (double)index;
        generated[index].target = targets[index];
        if (index < 8) {
            generated[index].split = SPLIT_TRAIN;
        } else if (index < 10) {
            generated[index].split = SPLIT_VALIDATION;
        } else {
            generated[index].split = SPLIT_TEST;
        }
    }
    for (size_t index = 0; index < SAMPLE_COUNT; ++index) {
        samples[index] = generated[index];
    }
    return true;
}

/**
 * @brief 为残差寻找使平方误差最小的单阈值切分。
 * @param features 有限的一维特征数组。
 * @param residuals 与特征一一对应的有限残差数组。
 * @param count 样本数，范围为 1..SAMPLE_COUNT。
 * @param output 有效可写树桩；成功后整体替换，失败时不修改。
 * @return 输入有效且树桩统计有限时为 true。
 */
static bool stump_fit(const double *features, const double *residuals, size_t count,
                      RegressionStump *output)
{
    if (features == nullptr || residuals == nullptr || output == nullptr || count == 0 ||
        count > SAMPLE_COUNT) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(features[index]) || !isfinite(residuals[index])) {
            return false;
        }
    }

    RegressionStump best = {0};
    double total_residual = 0.0;
    for (size_t index = 0; index < count; ++index) {
        total_residual += residuals[index];
    }
    best.left_value = total_residual / (double)count;
    best.right_value = best.left_value;
    if (!isfinite(best.left_value)) {
        return false;
    }

    double unique_features[SAMPLE_COUNT] = {0};
    size_t unique_count = 0;
    for (size_t index = 0; index < count; ++index) {
        size_t insertion = unique_count;
        while (insertion > 0 && unique_features[insertion - 1] > features[index]) {
            unique_features[insertion] = unique_features[insertion - 1];
            --insertion;
        }
        if ((insertion == 0 || unique_features[insertion - 1] != features[index]) &&
            (insertion == unique_count || unique_features[insertion] != features[index])) {
            unique_features[insertion] = features[index];
            ++unique_count;
        }
    }

    double best_error = DBL_MAX;
    for (size_t boundary = 0; boundary + 1 < unique_count; ++boundary) {
        double threshold = unique_features[boundary] * 0.5 +
                           unique_features[boundary + 1] * 0.5;
        if (threshold >= unique_features[boundary + 1]) {
            threshold = unique_features[boundary];
        }

        double left_sum = 0.0;
        double right_sum = 0.0;
        size_t left_count = 0;
        size_t right_count = 0;
        for (size_t index = 0; index < count; ++index) {
            if (features[index] <= threshold) {
                left_sum += residuals[index];
                ++left_count;
            } else {
                right_sum += residuals[index];
                ++right_count;
            }
        }
        if (left_count == 0 || right_count == 0) {
            continue;
        }

        double left_value = left_sum / (double)left_count;
        double right_value = right_sum / (double)right_count;
        double error = 0.0;
        for (size_t index = 0; index < count; ++index) {
            double value = features[index] <= threshold ? left_value : right_value;
            double difference = residuals[index] - value;
            error += difference * difference;
        }
        if (!isfinite(error)) {
            return false;
        }
        if (error < best_error) {
            best_error = error;
            best.threshold = threshold;
            best.left_value = left_value;
            best.right_value = right_value;
            best.has_split = true;
        }
    }

    *output = best;
    return true;
}

/**
 * @brief 计算树桩对单个特征的叶节点输出。
 * @param stump 有限的只读回归树桩。
 * @param feature 有限的一维特征。
 * @param output 有效可写预测值；成功时写入一个有限数。
 * @return 输入有效时为 true，否则不修改输出。
 */
static bool stump_predict(const RegressionStump *stump, double feature, double *output)
{
    if (stump == nullptr || output == nullptr || !isfinite(feature) ||
        !isfinite(stump->left_value) || !isfinite(stump->right_value) ||
        (stump->has_split && !isfinite(stump->threshold))) {
        return false;
    }
    if (stump->has_split && feature > stump->threshold) {
        *output = stump->right_value;
    } else {
        *output = stump->left_value;
    }
    return true;
}

/**
 * @brief 将基线均值与各轮树桩输出相加，得到集成模型预测。
 * @param model 已拟合的只读提升模型。
 * @param feature 有限的一维特征。
 * @param output 有效可写预测值；成功时写入有限结果。
 * @return 模型结构与中间计算有效时为 true。
 */
static bool model_predict(const BoostedModel *model, double feature, double *output)
{
    if (model == nullptr || output == nullptr || !isfinite(feature) ||
        !isfinite(model->base_value) || model->round_count > MAX_ROUND_COUNT) {
        return false;
    }

    double prediction = model->base_value;
    for (size_t round = 0; round < model->round_count; ++round) {
        double contribution = 0.0;
        if (!stump_predict(&model->stumps[round], feature, &contribution)) {
            return false;
        }
        prediction += contribution;
    }
    if (!isfinite(prediction)) {
        return false;
    }
    *output = prediction;
    return true;
}

/**
 * @brief 计算平方误差的梯度提升，并按验证误差保留最佳轮次。
 * @param samples 固定数据集只读视图。
 * @param count 样本数，必须为 SAMPLE_COUNT。
 * @param learning_rate 有限正数步长，通常不大于 1。
 * @param max_rounds 要尝试的轮数，不超过 MAX_ROUND_COUNT。
 * @param output 有效可写模型；成功时保存验证误差最佳检查点。
 * @param best_round 有效可写轮次结果；零表示仅使用初始均值。
 * @return 训练集和验证集非空且数值有限时为 true；失败时不修改输出参数。
 */
static bool model_train(const Sample *samples, size_t count, double learning_rate,
                        size_t max_rounds, BoostedModel *output, size_t *best_round)
{
    if (samples == nullptr || output == nullptr || best_round == nullptr || count != SAMPLE_COUNT ||
        !isfinite(learning_rate) || learning_rate <= 0.0 || learning_rate > 1.0 ||
        max_rounds > MAX_ROUND_COUNT) {
        return false;
    }

    size_t training_count = 0;
    double target_sum = 0.0;
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(samples[index].feature) || !isfinite(samples[index].target) ||
            samples[index].split < SPLIT_TRAIN || samples[index].split > SPLIT_TEST) {
            return false;
        }
        if (samples[index].split == SPLIT_TRAIN) {
            target_sum += samples[index].target;
            ++training_count;
        }
    }
    if (training_count == 0) {
        return false;
    }

    BoostedModel current = {0};
    current.base_value = target_sum / (double)training_count;
    if (!isfinite(current.base_value)) {
        return false;
    }
    Metrics validation = {0};
    double best_validation_error = DBL_MAX;
    BoostedModel best_model = current;
    size_t selected_round = 0;

    double validation_error_sum = 0.0;
    size_t validation_count = 0;
    for (size_t index = 0; index < count; ++index) {
        if (samples[index].split == SPLIT_VALIDATION) {
            double difference = samples[index].target - current.base_value;
            validation_error_sum += difference * difference;
            ++validation_count;
        }
    }
    if (validation_count == 0) {
        return false;
    }
    best_validation_error = validation_error_sum / (double)validation_count;
    if (!isfinite(best_validation_error)) {
        return false;
    }

    for (size_t round = 0; round < max_rounds; ++round) {
        double features[SAMPLE_COUNT] = {0};
        double residuals[SAMPLE_COUNT] = {0};
        size_t residual_count = 0;
        for (size_t index = 0; index < count; ++index) {
            if (samples[index].split != SPLIT_TRAIN) {
                continue;
            }
            double prediction = 0.0;
            if (!model_predict(&current, samples[index].feature, &prediction)) {
                return false;
            }
            features[residual_count] = samples[index].feature;
            residuals[residual_count] = samples[index].target - prediction;
            ++residual_count;
        }

        RegressionStump stump = {0};
        if (!stump_fit(features, residuals, residual_count, &stump)) {
            return false;
        }
        /* 叶子均值拟合当前残差，乘学习率后作为本轮较小的修正量。 */
        double left_update = learning_rate * stump.left_value;
        double right_update = learning_rate * stump.right_value;
        if (!isfinite(left_update) || !isfinite(right_update)) {
            return false;
        }
        stump.left_value = left_update;
        stump.right_value = right_update;
        current.stumps[current.round_count] = stump;
        ++current.round_count;

        validation_error_sum = 0.0;
        validation_count = 0;
        for (size_t index = 0; index < count; ++index) {
            if (samples[index].split != SPLIT_VALIDATION) {
                continue;
            }
            double prediction = 0.0;
            if (!model_predict(&current, samples[index].feature, &prediction)) {
                return false;
            }
            double difference = samples[index].target - prediction;
            validation_error_sum += difference * difference;
            ++validation_count;
        }
        validation.mean_squared_error = validation_error_sum / (double)validation_count;
        validation.count = validation_count;
        if (!isfinite(validation.mean_squared_error)) {
            return false;
        }
        /* 验证误差只用于选轮次，不参与残差和树桩阈值的拟合。 */
        if (validation.mean_squared_error < best_validation_error) {
            best_validation_error = validation.mean_squared_error;
            best_model = current;
            selected_round = current.round_count;
        }
    }

    *output = best_model;
    *best_round = selected_round;
    return true;
}

/**
 * @brief 汇总指定数据划分的均方误差。
 * @param samples 固定数据集只读视图。
 * @param count 样本数，必须为 SAMPLE_COUNT。
 * @param split 要评估的训练、验证或测试划分。
 * @param model 用于预测的只读提升模型。
 * @param output 有效可写指标；成功时整体替换。
 * @return 划分非空且预测有限时为 true。
 */
static bool evaluate(const Sample *samples, size_t count, DataSplit split,
                     const BoostedModel *model, Metrics *output)
{
    if (samples == nullptr || model == nullptr || output == nullptr || count != SAMPLE_COUNT ||
        split < SPLIT_TRAIN || split > SPLIT_TEST) {
        return false;
    }

    Metrics metrics = {0};
    for (size_t index = 0; index < count; ++index) {
        if (samples[index].split != split) {
            continue;
        }
        double prediction = 0.0;
        if (!model_predict(model, samples[index].feature, &prediction)) {
            return false;
        }
        double difference = samples[index].target - prediction;
        metrics.mean_squared_error += difference * difference;
        ++metrics.count;
    }
    if (metrics.count == 0) {
        return false;
    }
    metrics.mean_squared_error /= (double)metrics.count;
    if (!isfinite(metrics.mean_squared_error)) {
        return false;
    }
    *output = metrics;
    return true;
}

/**
 * @brief 训练树桩提升回归器并显示验证选轮次及独立测试误差。
 * @return 训练或数据评估失败时返回非零值。
 */
int main(void)
{
    Sample samples[SAMPLE_COUNT] = {{0}};
    BoostedModel model = {0};
    size_t best_round = 0;
    if (!dataset_generate(samples) ||
        !model_train(samples, SAMPLE_COUNT, LEARNING_RATE, MAX_TRAINING_ROUNDS, &model,
                     &best_round)) {
        return 1;
    }

    BoostedModel baseline = {.base_value = model.base_value};
    Metrics baseline_test = {0};
    Metrics training = {0};
    Metrics validation = {0};
    Metrics test = {0};
    if (!evaluate(samples, SAMPLE_COUNT, SPLIT_TEST, &baseline, &baseline_test) ||
        !evaluate(samples, SAMPLE_COUNT, SPLIT_TRAIN, &model, &training) ||
        !evaluate(samples, SAMPLE_COUNT, SPLIT_VALIDATION, &model, &validation) ||
        !evaluate(samples, SAMPLE_COUNT, SPLIT_TEST, &model, &test)) {
        return 1;
    }

    printf("平方误差梯度提升回归：一维数据、树桩弱学习器、学习率 %.2f\n",
           LEARNING_RATE);
    printf("尝试 %zu 轮，根据验证集选择第 %zu 轮检查点（实际树桩数 %zu）\n",
           MAX_TRAINING_ROUNDS, best_round, model.round_count);
    printf("训练 MSE %.4f，验证 MSE %.4f\n", training.mean_squared_error,
           validation.mean_squared_error);
    printf("独立测试 MSE：均值基线 %.4f，提升模型 %.4f（%zu 个样本）\n",
           baseline_test.mean_squared_error, test.mean_squared_error, test.count);
    for (size_t index = 0; index < SAMPLE_COUNT; ++index) {
        double prediction = 0.0;
        if (samples[index].split == SPLIT_TEST &&
            model_predict(&model, samples[index].feature, &prediction)) {
            printf("测试 x=%.1f，真实值=%.1f，预测=%.4f\n", samples[index].feature,
                   samples[index].target, prediction);
        }
    }
    return 0;
}
