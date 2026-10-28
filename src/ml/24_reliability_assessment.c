/**
 * @file
 * @brief 用类别不平衡数据学习分类指标、阈值选择和概率校准。
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    FEATURE_COUNT = 2,
    NEGATIVE_SAMPLE_COUNT = 360,
    POSITIVE_SAMPLE_COUNT = 40,
    MAX_SAMPLE_COUNT = NEGATIVE_SAMPLE_COUNT + POSITIVE_SAMPLE_COUNT,
    SPLIT_COUNT = 3,
    RELIABILITY_BIN_COUNT = 10,
    TRAIN_EPOCH_COUNT = 1200,
    CALIBRATION_EPOCH_COUNT = 2000
};

static const double TRAINING_RATE = 0.1;
static const double CALIBRATION_RATE = 0.03;

typedef enum {
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_TEST
} DataSplit;

typedef struct {
    double features[FEATURE_COUNT];
    int label;
    DataSplit split;
} Sample;

typedef struct {
    Sample samples[MAX_SAMPLE_COUNT];
    size_t count;
    size_t split_counts[SPLIT_COUNT];
    size_t class_counts[SPLIT_COUNT][2];
} Dataset;

typedef struct {
    double mean[FEATURE_COUNT];
    double scale[FEATURE_COUNT];
} Standardizer;

typedef struct {
    double weights[FEATURE_COUNT];
    double bias;
} LogisticModel;

typedef struct {
    double slope;
    double intercept;
} CalibrationMap;

typedef struct {
    int label;
    double probability;
    double logit;
} Prediction;

typedef struct {
    size_t count;
    size_t true_negative;
    size_t false_positive;
    size_t false_negative;
    size_t true_positive;
    double accuracy;
    double precision;
    double recall;
    double f1;
    double balanced_accuracy;
    double brier_score;
    double cross_entropy;
    double expected_calibration_error;
} Metrics;

typedef struct {
    size_t count;
    double probability_sum;
    size_t positive_count;
} ReliabilityBin;

/**
 * @brief 产生固定种子的 [0,1) 伪随机数。
 * @param state 可写随机状态，由调用者独占维护。
 * @return 更新状态映射到 [0,1) 的实数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 打乱同一类别的样本，避免划分依赖生成顺序。
 * @param samples 至少 count 个可写样本；内容会被原地重排。
 * @param count 样本数量。
 * @param state 有效可写的伪随机状态。
 */
static void shuffle_samples(Sample *samples, size_t count, uint32_t *state)
{
    for (size_t index = count; index > 1; --index) {
        size_t selected = (size_t)(random_unit(state) * (double)index);
        Sample temporary = samples[index - 1];
        samples[index - 1] = samples[selected];
        samples[selected] = temporary;
    }
}

/**
 * @brief 生成固定种子的少数类二分类样本，并按类别分层切分。
 * @param output 有效可写数据集；成功后整体替换，失败时不修改。
 * @param seed 固定随机种子，用于可重复教学实验。
 * @return 特征、类别比例和三种划分有效时为 true。
 */
static bool dataset_generate(Dataset *output, uint32_t seed)
{
    if (output == nullptr) {
        return false;
    }

    Sample negatives[NEGATIVE_SAMPLE_COUNT] = {0};
    Sample positives[POSITIVE_SAMPLE_COUNT] = {0};
    for (size_t index = 0; index < NEGATIVE_SAMPLE_COUNT; ++index) {
        negatives[index].features[0] = 2.0 * random_unit(&seed) - 1.0;
        negatives[index].features[1] = 2.0 * random_unit(&seed) - 1.0;
        negatives[index].label = 0;
    }
    for (size_t index = 0; index < POSITIVE_SAMPLE_COUNT; ++index) {
        positives[index].features[0] = 0.75 + 2.4 * random_unit(&seed) - 1.2;
        positives[index].features[1] = 0.55 + 2.4 * random_unit(&seed) - 1.2;
        positives[index].label = 1;
    }

    shuffle_samples(negatives, NEGATIVE_SAMPLE_COUNT, &seed);
    shuffle_samples(positives, POSITIVE_SAMPLE_COUNT, &seed);

    Dataset generated = {0};
    generated.count = MAX_SAMPLE_COUNT;
    size_t negative_train_count = NEGATIVE_SAMPLE_COUNT * 60 / 100;
    size_t negative_validation_count = NEGATIVE_SAMPLE_COUNT * 20 / 100;
    size_t positive_train_count = POSITIVE_SAMPLE_COUNT * 60 / 100;
    size_t positive_validation_count = POSITIVE_SAMPLE_COUNT * 20 / 100;

    for (size_t index = 0; index < NEGATIVE_SAMPLE_COUNT; ++index) {
        Sample sample = negatives[index];
        if (index < negative_train_count) {
            sample.split = SPLIT_TRAIN;
        } else if (index < negative_train_count + negative_validation_count) {
            sample.split = SPLIT_VALIDATION;
        } else {
            sample.split = SPLIT_TEST;
        }
        generated.samples[index] = sample;
    }
    for (size_t index = 0; index < POSITIVE_SAMPLE_COUNT; ++index) {
        Sample sample = positives[index];
        if (index < positive_train_count) {
            sample.split = SPLIT_TRAIN;
        } else if (index < positive_train_count + positive_validation_count) {
            sample.split = SPLIT_VALIDATION;
        } else {
            sample.split = SPLIT_TEST;
        }
        generated.samples[NEGATIVE_SAMPLE_COUNT + index] = sample;
    }

    for (size_t index = 0; index < generated.count; ++index) {
        const Sample *sample = &generated.samples[index];
        ++generated.split_counts[sample->split];
        ++generated.class_counts[sample->split][sample->label];
    }

    *output = generated;
    return true;
}

/**
 * @brief 检查样本、标签和分层划分计数是否一致。
 * @param dataset 待检查的只读数据集。
 * @return 三种划分均非空且每类在训练集中均存在时为 true。
 */
static bool dataset_is_valid(const Dataset *dataset)
{
    if (dataset == nullptr || dataset->count != MAX_SAMPLE_COUNT) {
        return false;
    }

    size_t split_counts[SPLIT_COUNT] = {0};
    size_t class_counts[SPLIT_COUNT][2] = {{0}};
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if ((sample->label != 0 && sample->label != 1) || sample->split < SPLIT_TRAIN ||
            sample->split >= SPLIT_COUNT) {
            return false;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            if (!isfinite(sample->features[feature])) {
                return false;
            }
        }
        ++split_counts[sample->split];
        ++class_counts[sample->split][sample->label];
    }

    for (size_t split = 0; split < SPLIT_COUNT; ++split) {
        if (split_counts[split] != dataset->split_counts[split] || split_counts[split] == 0 ||
            class_counts[split][0] != dataset->class_counts[split][0] ||
            class_counts[split][1] != dataset->class_counts[split][1]) {
            return false;
        }
    }
    return class_counts[SPLIT_TRAIN][0] > 0 && class_counts[SPLIT_TRAIN][1] > 0;
}

/**
 * @brief 仅用训练集拟合特征均值和总体标准差。
 * @param dataset 有效的分层数据集。
 * @param output 有效可写标准化参数；失败时保持原值。
 * @return 所有训练特征均有限且方差非零时为 true。
 */
static bool standardizer_fit(const Dataset *dataset, Standardizer *output)
{
    if (!dataset_is_valid(dataset) || output == nullptr) {
        return false;
    }

    Standardizer fitted = {0};
    size_t training_count = dataset->split_counts[SPLIT_TRAIN];
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != SPLIT_TRAIN) {
            continue;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            fitted.mean[feature] += sample->features[feature];
        }
    }
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        fitted.mean[feature] /= (double)training_count;
    }
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != SPLIT_TRAIN) {
            continue;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double difference = sample->features[feature] - fitted.mean[feature];
            fitted.scale[feature] += difference * difference;
        }
    }
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        fitted.scale[feature] = sqrt(fitted.scale[feature] / (double)training_count);
        if (!isfinite(fitted.mean[feature]) || !isfinite(fitted.scale[feature]) ||
            fitted.scale[feature] <= DBL_EPSILON) {
            return false;
        }
    }

    *output = fitted;
    return true;
}

/**
 * @brief 按训练统计量标准化一个样本。
 * @param standardizer 只读训练集均值和标准差。
 * @param sample 含有限原始特征的只读样本。
 * @param output 至少 FEATURE_COUNT 个可写元素；失败时保持原值。
 * @return 变换成功且结果有限时为 true。
 */
static bool standardize_sample(const Standardizer *standardizer, const Sample *sample,
                               double output[FEATURE_COUNT])
{
    if (standardizer == nullptr || sample == nullptr || output == nullptr) {
        return false;
    }

    double transformed[FEATURE_COUNT];
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(sample->features[feature]) || !isfinite(standardizer->mean[feature]) ||
            !isfinite(standardizer->scale[feature]) || standardizer->scale[feature] <= 0.0) {
            return false;
        }
        transformed[feature] =
            (sample->features[feature] - standardizer->mean[feature]) /
            standardizer->scale[feature];
        if (!isfinite(transformed[feature])) {
            return false;
        }
    }
    memcpy(output, transformed, sizeof(transformed));
    return true;
}

/**
 * @brief 稳定地计算 sigmoid 概率。
 * @param value 有限的 logit。
 * @return [0,1] 区间内的概率。
 */
static double sigmoid(double value)
{
    if (value >= 0.0) {
        return 1.0 / (1.0 + exp(-value));
    }
    double exponential = exp(value);
    return exponential / (1.0 + exponential);
}

/**
 * @brief 计算 logit 形式的稳定二元交叉熵。
 * @param logit sigmoid 之前的有限分数。
 * @param label 真实二分类标签 0 或 1。
 * @return 有限非负损失。
 */
static double binary_cross_entropy(double logit, int label)
{
    return fmax(logit, 0.0) - (double)label * logit + log1p(exp(-fabs(logit)));
}

/**
 * @brief 计算模型对单个样本的 logit 和正类概率。
 * @param model 只读逻辑回归模型。
 * @param standardizer 只读训练集标准化参数。
 * @param sample 待预测样本。
 * @param output 有效可写预测；失败时保持原值。
 * @return 预测成功且结果有限时为 true。
 */
static bool model_predict(const LogisticModel *model, const Standardizer *standardizer,
                          const Sample *sample, Prediction *output)
{
    if (model == nullptr || standardizer == nullptr || sample == nullptr || output == nullptr) {
        return false;
    }

    double features[FEATURE_COUNT];
    if (!standardize_sample(standardizer, sample, features) || !isfinite(model->bias)) {
        return false;
    }
    double logit = model->bias;
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(model->weights[feature])) {
            return false;
        }
        logit += model->weights[feature] * features[feature];
    }
    if (!isfinite(logit)) {
        return false;
    }

    Prediction prediction = {sample->label, sigmoid(logit), logit};
    if (!isfinite(prediction.probability)) {
        return false;
    }
    *output = prediction;
    return true;
}

/**
 * @brief 使用普通或类别均衡的二元交叉熵训练逻辑回归。
 * @param dataset 有效的训练、验证和测试数据集。
 * @param standardizer 只从训练集拟合的标准化参数。
 * @param balance_classes true 时按训练类频率设置权重，false 时两类权重均为 1。
 * @param output 有效可写模型；失败时保持原值。
 * @return 固定训练轮次完成且参数有限时为 true。
 */
static bool model_train(const Dataset *dataset, const Standardizer *standardizer,
                        bool balance_classes, LogisticModel *output)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || output == nullptr) {
        return false;
    }

    size_t negative_count = dataset->class_counts[SPLIT_TRAIN][0];
    size_t positive_count = dataset->class_counts[SPLIT_TRAIN][1];
    size_t training_count = dataset->split_counts[SPLIT_TRAIN];
    double class_weights[2] = {1.0, 1.0};
    if (balance_classes) {
        class_weights[0] = (double)training_count / (2.0 * (double)negative_count);
        class_weights[1] = (double)training_count / (2.0 * (double)positive_count);
    }
    LogisticModel fitted = {0};

    for (size_t epoch = 0; epoch < TRAIN_EPOCH_COUNT; ++epoch) {
        double weight_gradients[FEATURE_COUNT] = {0};
        double bias_gradient = 0.0;
        for (size_t index = 0; index < dataset->count; ++index) {
            const Sample *sample = &dataset->samples[index];
            if (sample->split != SPLIT_TRAIN) {
                continue;
            }
            Prediction prediction = {0};
            if (!model_predict(&fitted, standardizer, sample, &prediction)) {
                return false;
            }
            double error = class_weights[sample->label] *
                           (prediction.probability - (double)sample->label);
            double features[FEATURE_COUNT];
            if (!standardize_sample(standardizer, sample, features)) {
                return false;
            }
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                weight_gradients[feature] += error * features[feature];
            }
            bias_gradient += error;
        }

        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double gradient = weight_gradients[feature] / (double)training_count;
            fitted.weights[feature] -= TRAINING_RATE * gradient;
        }
        fitted.bias -= TRAINING_RATE * bias_gradient / (double)training_count;
        if (!isfinite(fitted.weights[0]) || !isfinite(fitted.weights[1]) ||
            !isfinite(fitted.bias)) {
            return false;
        }
    }

    *output = fitted;
    return true;
}

/**
 * @brief 收集指定划分的模型预测，供阈值选择、校准或最终评估使用。
 * @param model 只读逻辑回归模型。
 * @param standardizer 只读训练集标准化参数。
 * @param dataset 完整只读数据集。
 * @param split 需要收集的非空划分。
 * @param output 至少该划分样本数个可写预测。
 * @param output_count 有效可写数量输出；失败时保持原值。
 * @return 预测全部成功且数量匹配时为 true。
 */
static bool collect_predictions(const LogisticModel *model, const Standardizer *standardizer,
                                const Dataset *dataset, DataSplit split,
                                Prediction output[MAX_SAMPLE_COUNT], size_t *output_count)
{
    if (model == nullptr || standardizer == nullptr || !dataset_is_valid(dataset) ||
        split < SPLIT_TRAIN || split >= SPLIT_COUNT || output == nullptr ||
        output_count == nullptr) {
        return false;
    }

    Prediction collected[MAX_SAMPLE_COUNT] = {{0}};
    size_t count = 0;
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != split) {
            continue;
        }
        if (!model_predict(model, standardizer, sample, &collected[count])) {
            return false;
        }
        ++count;
    }
    if (count != dataset->split_counts[split]) {
        return false;
    }

    memcpy(output, collected, count * sizeof(*output));
    *output_count = count;
    return true;
}

/**
 * @brief 计算阈值下的混淆矩阵和分类指标。
 * @param predictions 含标签与概率的非空预测数组。
 * @param count 预测数量。
 * @param threshold 范围 [0,1] 的正类判定阈值。
 * @param output 有效可写指标；失败时保持原值。
 * @return 所有概率有效且指标可计算时为 true。
 */
static bool classification_metrics(const Prediction *predictions, size_t count, double threshold,
                                   Metrics *output)
{
    if (predictions == nullptr || count == 0 || count > MAX_SAMPLE_COUNT || output == nullptr ||
        !isfinite(threshold) || threshold < 0.0 || threshold > 1.0) {
        return false;
    }

    Metrics metrics = {0};
    metrics.count = count;
    for (size_t index = 0; index < count; ++index) {
        const Prediction *prediction = &predictions[index];
        if ((prediction->label != 0 && prediction->label != 1) ||
            !isfinite(prediction->probability) || prediction->probability < 0.0 ||
            prediction->probability > 1.0) {
            return false;
        }
        int predicted_label = prediction->probability >= threshold ? 1 : 0;
        if (prediction->label == 0 && predicted_label == 0) {
            ++metrics.true_negative;
        } else if (prediction->label == 0) {
            ++metrics.false_positive;
        } else if (predicted_label == 0) {
            ++metrics.false_negative;
        } else {
            ++metrics.true_positive;
        }
    }

    size_t actual_negative = metrics.true_negative + metrics.false_positive;
    size_t actual_positive = metrics.false_negative + metrics.true_positive;
    size_t predicted_positive = metrics.false_positive + metrics.true_positive;
    metrics.accuracy = (double)(metrics.true_negative + metrics.true_positive) / (double)count;
    metrics.precision = predicted_positive == 0
                            ? 0.0
                            : (double)metrics.true_positive / (double)predicted_positive;
    metrics.recall = actual_positive == 0
                         ? 0.0
                         : (double)metrics.true_positive / (double)actual_positive;
    metrics.f1 = metrics.precision + metrics.recall == 0.0
                     ? 0.0
                     : 2.0 * metrics.precision * metrics.recall /
                           (metrics.precision + metrics.recall);
    double negative_recall = actual_negative == 0
                                 ? 0.0
                                 : (double)metrics.true_negative / (double)actual_negative;
    metrics.balanced_accuracy = 0.5 * (metrics.recall + negative_recall);
    *output = metrics;
    return true;
}

/**
 * @brief 依据验证集 F1 选择分类阈值，平局时优先平衡准确率和较高阈值。
 * @param predictions 非空的验证集预测；测试集预测不得传入。
 * @param count 验证预测数量。
 * @param threshold 有效可写阈值输出；失败时保持原值。
 * @return 至少一个候选阈值可评估时为 true。
 */
static bool select_threshold(const Prediction *predictions, size_t count, double *threshold)
{
    if (predictions == nullptr || count == 0 || threshold == nullptr) {
        return false;
    }

    double best_threshold = 0.5;
    Metrics best_metrics = {0};
    bool has_best = false;
    for (size_t step = 5; step <= 95; ++step) {
        double candidate = (double)step / 100.0;
        Metrics metrics = {0};
        if (!classification_metrics(predictions, count, candidate, &metrics)) {
            return false;
        }
        bool better_f1 = !has_best || metrics.f1 > best_metrics.f1 + 1e-12;
        bool tied_f1 = has_best && fabs(metrics.f1 - best_metrics.f1) <= 1e-12;
        bool better_balance = tied_f1 &&
                              metrics.balanced_accuracy > best_metrics.balanced_accuracy + 1e-12;
        bool tied_balance = tied_f1 &&
                            fabs(metrics.balanced_accuracy - best_metrics.balanced_accuracy) <=
                                1e-12;
        if (better_f1 || better_balance || (tied_balance && candidate > best_threshold)) {
            best_threshold = candidate;
            best_metrics = metrics;
            has_best = true;
        }
    }

    if (!has_best) {
        return false;
    }
    *threshold = best_threshold;
    return true;
}

/**
 * @brief 用验证集 logits 拟合 sigmoid(a*z+c) 概率校准映射。
 * @param predictions 含真实标签和原始 logit 的验证集预测。
 * @param count 验证预测数量，必须同时包含正负类。
 * @param output 有效可写校准参数；失败时保持原值。
 * @return 梯度下降完成且斜率、截距有限时为 true。
 */
static bool calibrator_fit(const Prediction *predictions, size_t count, CalibrationMap *output)
{
    if (predictions == nullptr || count < 2 || output == nullptr || count > MAX_SAMPLE_COUNT) {
        return false;
    }

    size_t classes[2] = {0};
    for (size_t index = 0; index < count; ++index) {
        if ((predictions[index].label != 0 && predictions[index].label != 1) ||
            !isfinite(predictions[index].logit)) {
            return false;
        }
        ++classes[predictions[index].label];
    }
    if (classes[0] == 0 || classes[1] == 0) {
        return false;
    }

    CalibrationMap fitted = {1.0, 0.0};
    for (size_t epoch = 0; epoch < CALIBRATION_EPOCH_COUNT; ++epoch) {
        double slope_gradient = 0.0;
        double intercept_gradient = 0.0;
        for (size_t index = 0; index < count; ++index) {
            const Prediction *prediction = &predictions[index];
            double calibrated_logit =
                fitted.slope * prediction->logit + fitted.intercept;
            double error = sigmoid(calibrated_logit) - (double)prediction->label;
            slope_gradient += error * prediction->logit;
            intercept_gradient += error;
        }
        fitted.slope -= CALIBRATION_RATE * slope_gradient / (double)count;
        fitted.intercept -= CALIBRATION_RATE * intercept_gradient / (double)count;
        if (!isfinite(fitted.slope) || !isfinite(fitted.intercept)) {
            return false;
        }
    }

    *output = fitted;
    return true;
}

/**
 * @brief 用概率和 logit 计算 Brier、交叉熵和可靠性分箱误差。
 * @param predictions 概率与 logit 相互对应的非空预测数组。
 * @param count 预测数量。
 * @param output 有效可写指标；失败时保持原值。
 * @param bins 可选的 RELIABILITY_BIN_COUNT 个可靠性分箱输出。
 * @return 输入合法且指标有限时为 true。
 */
static bool probability_metrics(const Prediction *predictions, size_t count, Metrics *output,
                               ReliabilityBin bins[RELIABILITY_BIN_COUNT])
{
    if (predictions == nullptr || count == 0 || count > MAX_SAMPLE_COUNT || output == nullptr) {
        return false;
    }

    Metrics metrics = {0};
    ReliabilityBin local_bins[RELIABILITY_BIN_COUNT] = {{0}};
    metrics.count = count;
    for (size_t index = 0; index < count; ++index) {
        const Prediction *prediction = &predictions[index];
        if ((prediction->label != 0 && prediction->label != 1) ||
            !isfinite(prediction->probability) || prediction->probability < 0.0 ||
            prediction->probability > 1.0 || !isfinite(prediction->logit)) {
            return false;
        }
        double difference = prediction->probability - (double)prediction->label;
        metrics.brier_score += difference * difference;
        metrics.cross_entropy += binary_cross_entropy(prediction->logit, prediction->label);
        size_t bin_index = (size_t)(prediction->probability * RELIABILITY_BIN_COUNT);
        if (bin_index >= RELIABILITY_BIN_COUNT) {
            bin_index = RELIABILITY_BIN_COUNT - 1;
        }
        ++local_bins[bin_index].count;
        local_bins[bin_index].probability_sum += prediction->probability;
        local_bins[bin_index].positive_count += (size_t)prediction->label;
    }

    metrics.brier_score /= (double)count;
    metrics.cross_entropy /= (double)count;
    for (size_t index = 0; index < RELIABILITY_BIN_COUNT; ++index) {
        if (local_bins[index].count == 0) {
            continue;
        }
        double mean_probability =
            local_bins[index].probability_sum / (double)local_bins[index].count;
        double observed_rate =
            (double)local_bins[index].positive_count / (double)local_bins[index].count;
        metrics.expected_calibration_error +=
            (double)local_bins[index].count / (double)count *
            fabs(mean_probability - observed_rate);
    }
    if (!isfinite(metrics.brier_score) || !isfinite(metrics.cross_entropy) ||
        !isfinite(metrics.expected_calibration_error)) {
        return false;
    }

    *output = metrics;
    if (bins != nullptr) {
        memcpy(bins, local_bins, sizeof(local_bins));
    }
    return true;
}

/**
 * @brief 应用校准仿射映射，原位更新概率和用于交叉熵的 logit。
 * @param map 已由验证集拟合的只读校准映射。
 * @param predictions 可写预测数组；成功后概率和 logit 均被更新。
 * @param count 预测数量。
 * @return 映射参数、输入和输出均有限时为 true。
 */
static bool calibrator_apply(const CalibrationMap *map, Prediction *predictions, size_t count)
{
    if (map == nullptr || predictions == nullptr || count == 0 || count > MAX_SAMPLE_COUNT ||
        !isfinite(map->slope) || !isfinite(map->intercept)) {
        return false;
    }

    Prediction transformed[MAX_SAMPLE_COUNT];
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(predictions[index].logit)) {
            return false;
        }
        transformed[index] = predictions[index];
        transformed[index].logit =
            map->slope * predictions[index].logit + map->intercept;
        transformed[index].probability = sigmoid(transformed[index].logit);
        if (!isfinite(transformed[index].logit) ||
            !isfinite(transformed[index].probability)) {
            return false;
        }
    }
    memcpy(predictions, transformed, count * sizeof(*predictions));
    return true;
}

/**
 * @brief 计算“始终预测负类”的多数类基线混淆矩阵。
 * @param predictions 测试集预测，仅使用其中真实标签。
 * @param count 非零测试样本数。
 * @param output 有效可写分类指标。
 * @return 标签有效且指标成功生成时为 true。
 */
static bool majority_baseline_metrics(const Prediction *predictions, size_t count,
                                      Metrics *output)
{
    if (predictions == nullptr || count == 0 || count > MAX_SAMPLE_COUNT || output == nullptr) {
        return false;
    }
    Metrics metrics = {0};
    metrics.count = count;
    for (size_t index = 0; index < count; ++index) {
        if (predictions[index].label == 0) {
            ++metrics.true_negative;
        } else if (predictions[index].label == 1) {
            ++metrics.false_negative;
        } else {
            return false;
        }
    }
    size_t negative_count = metrics.true_negative;
    metrics.accuracy = (double)negative_count / (double)count;
    metrics.precision = 0.0;
    metrics.recall = 0.0;
    metrics.f1 = 0.0;
    double positive_recall = 0.0;
    double negative_recall = negative_count == 0 ? 0.0 : 1.0;
    metrics.balanced_accuracy = 0.5 * (positive_recall + negative_recall);
    *output = metrics;
    return true;
}

/**
 * @brief 显示混淆矩阵和常用不平衡分类指标。
 * @param title 简短报告标题。
 * @param metrics 已计算的分类指标。
 */
static void print_classification_metrics(const char *title, const Metrics *metrics)
{
    printf("%s：准确率=%.3f 精确率=%.3f 召回率=%.3f F1=%.3f 平衡准确率=%.3f\n",
           title, metrics->accuracy, metrics->precision, metrics->recall, metrics->f1,
           metrics->balanced_accuracy);
    printf("  混淆矩阵：TN=%zu FP=%zu FN=%zu TP=%zu（n=%zu）\n", metrics->true_negative,
           metrics->false_positive, metrics->false_negative, metrics->true_positive,
           metrics->count);
}

/**
 * @brief 显示概率质量指标、分箱表和 ASCII 可靠性图。
 * @param title 报告标题，用于区分校准前后。
 * @param metrics Brier、交叉熵和 ECE 指标。
 * @param bins 十个概率区间的计数和正例数。
 */
static void print_probability_metrics(const char *title, const Metrics *metrics,
                                      const ReliabilityBin bins[RELIABILITY_BIN_COUNT])
{
    printf("%s：Brier=%.4f BCE=%.4f ECE=%.4f\n", title, metrics->brier_score,
           metrics->cross_entropy, metrics->expected_calibration_error);
    printf("  概率区间       样本数  平均预测概率  实际正例率\n");
    for (size_t index = 0; index < RELIABILITY_BIN_COUNT; ++index) {
        const ReliabilityBin *bin = &bins[index];
        double lower = (double)index / (double)RELIABILITY_BIN_COUNT;
        double upper = (double)(index + 1) / (double)RELIABILITY_BIN_COUNT;
        if (bin->count == 0) {
            printf("  [%.1f, %.1f)       0          -            -\n", lower, upper);
            continue;
        }
        double average_probability = bin->probability_sum / (double)bin->count;
        double observed_rate = (double)bin->positive_count / (double)bin->count;
        printf("  [%.1f, %.1f)      %2zu       %.3f         %.3f\n", lower, upper,
               bin->count, average_probability, observed_rate);
    }

    char chart[11][51];
    memset(chart, ' ', sizeof(chart));
    for (size_t point = 0; point <= 10; ++point) {
        size_t coordinate = point * 5;
        chart[10 - point][coordinate] = '.';
    }
    for (size_t index = 0; index < RELIABILITY_BIN_COUNT; ++index) {
        const ReliabilityBin *bin = &bins[index];
        if (bin->count == 0) {
            continue;
        }
        double mean_probability = bin->probability_sum / (double)bin->count;
        double observed_rate = (double)bin->positive_count / (double)bin->count;
        size_t column = (size_t)lround(mean_probability * 50.0);
        size_t row = (size_t)lround((1.0 - observed_rate) * 10.0);
        if (column > 50) {
            column = 50;
        }
        if (row > 10) {
            row = 10;
        }
        chart[row][column] = chart[row][column] == '.' ? '*' : 'o';
    }
    printf("  可靠性图（横轴=平均预测概率，纵轴=实际正例率；"
           ". 为理想线，o 为观测）\n");
    for (size_t row = 0; row <= 10; ++row) {
        printf("  %.1f |", 1.0 - (double)row / 10.0);
        for (size_t column = 0; column <= 50; ++column) {
            putchar(chart[row][column]);
        }
        putchar('\n');
    }
    printf("      +---------------------------------------------------\n");
    printf("       0.0                 0.5                      1.0\n");
}

/**
 * @brief 运行不平衡指标、验证集阈值选择和验证集 Platt 校准教学流程。
 * @return 全部计算成功返回 EXIT_SUCCESS，否则返回 EXIT_FAILURE。
 */
int main(void)
{
    Dataset dataset = {0};
    Standardizer standardizer = {0};
    LogisticModel model = {0};
    LogisticModel unweighted_model = {0};
    Prediction validation_predictions[MAX_SAMPLE_COUNT] = {{0}};
    Prediction test_predictions[MAX_SAMPLE_COUNT] = {{0}};
    Prediction unweighted_test[MAX_SAMPLE_COUNT] = {{0}};
    size_t validation_count = 0;
    size_t test_count = 0;
    size_t unweighted_test_count = 0;
    double raw_threshold = 0.5;
    double calibrated_threshold = 0.5;
    CalibrationMap calibration = {0};
    Metrics validation_raw_classification = {0};
    Metrics validation_calibrated_classification = {0};
    Metrics validation_raw_probability = {0};
    Metrics validation_calibrated_probability = {0};

    if (!dataset_generate(&dataset, UINT32_C(20261021)) || !dataset_is_valid(&dataset) ||
        !standardizer_fit(&dataset, &standardizer) ||
        !model_train(&dataset, &standardizer, false, &unweighted_model) ||
        !model_train(&dataset, &standardizer, true, &model) ||
        !collect_predictions(&model, &standardizer, &dataset, SPLIT_VALIDATION,
                             validation_predictions, &validation_count)) {
        fprintf(stderr, "可靠性评估失败：输入、模型或验证预测无效。\n");
        return 1;
    }

    if (!select_threshold(validation_predictions, validation_count, &raw_threshold) ||
        !classification_metrics(validation_predictions, validation_count, raw_threshold,
                                &validation_raw_classification) ||
        !probability_metrics(validation_predictions, validation_count,
                             &validation_raw_probability, nullptr) ||
        !calibrator_fit(validation_predictions, validation_count, &calibration) ||
        !calibrator_apply(&calibration, validation_predictions, validation_count) ||
        !select_threshold(validation_predictions, validation_count, &calibrated_threshold) ||
        !classification_metrics(validation_predictions, validation_count,
                                calibrated_threshold, &validation_calibrated_classification) ||
        !probability_metrics(validation_predictions, validation_count,
                             &validation_calibrated_probability, nullptr) ||
        !collect_predictions(&model, &standardizer, &dataset, SPLIT_TEST, test_predictions,
                             &test_count) ||
        !collect_predictions(&unweighted_model, &standardizer, &dataset, SPLIT_TEST,
                             unweighted_test, &unweighted_test_count)) {
        fprintf(stderr, "可靠性评估失败：验证选择、校准或测试预测无效。\n");
        return 1;
    }

    Prediction calibrated_test[MAX_SAMPLE_COUNT];
    memcpy(calibrated_test, test_predictions, test_count * sizeof(*test_predictions));
    if (!calibrator_apply(&calibration, calibrated_test, test_count)) {
        fprintf(stderr, "可靠性评估失败：测试集概率校准无效。\n");
        return 1;
    }

    Metrics baseline = {0};
    Metrics unweighted_classification = {0};
    Metrics weighted_default_classification = {0};
    Metrics raw_classification = {0};
    Metrics calibrated_classification = {0};
    Metrics raw_probability = {0};
    Metrics calibrated_probability = {0};
    ReliabilityBin raw_bins[RELIABILITY_BIN_COUNT] = {{0}};
    ReliabilityBin calibrated_bins[RELIABILITY_BIN_COUNT] = {{0}};
    if (!majority_baseline_metrics(test_predictions, test_count, &baseline) ||
        unweighted_test_count != test_count ||
        !classification_metrics(unweighted_test, unweighted_test_count, 0.5,
                                &unweighted_classification) ||
        !classification_metrics(test_predictions, test_count, 0.5,
                                &weighted_default_classification) ||
        !classification_metrics(test_predictions, test_count, raw_threshold,
                                &raw_classification) ||
        !classification_metrics(calibrated_test, test_count, calibrated_threshold,
                                &calibrated_classification) ||
        !probability_metrics(test_predictions, test_count, &raw_probability, raw_bins) ||
        !probability_metrics(calibrated_test, test_count, &calibrated_probability,
                             calibrated_bins)) {
        fprintf(stderr, "可靠性评估失败：无法计算最终指标。\n");
        return 1;
    }

    printf("类别不平衡可靠性评估\n");
    printf("划分：训练 %zu（负类 %zu，正类 %zu），"
           "验证 %zu（负类 %zu，正类 %zu），"
           "测试 %zu（负类 %zu，正类 %zu）\n",
           dataset.split_counts[SPLIT_TRAIN], dataset.class_counts[SPLIT_TRAIN][0],
           dataset.class_counts[SPLIT_TRAIN][1], dataset.split_counts[SPLIT_VALIDATION],
           dataset.class_counts[SPLIT_VALIDATION][0],
           dataset.class_counts[SPLIT_VALIDATION][1], dataset.split_counts[SPLIT_TEST],
           dataset.class_counts[SPLIT_TEST][0], dataset.class_counts[SPLIT_TEST][1]);
    printf("类别加权：负类权重=%.3f，正类权重=%.3f；训练轮数=%d\n",
           (double)dataset.split_counts[SPLIT_TRAIN] /
               (2.0 * (double)dataset.class_counts[SPLIT_TRAIN][0]),
           (double)dataset.split_counts[SPLIT_TRAIN] /
               (2.0 * (double)dataset.class_counts[SPLIT_TRAIN][1]),
           TRAIN_EPOCH_COUNT);
    printf("验证集选择阈值：原始概率 %.2f，校准后概率 %.2f；校准映射："
           "sigmoid(%.4f × logit %+.4f)\n",
           raw_threshold, calibrated_threshold, calibration.slope, calibration.intercept);
    printf("验证集：阈值前后 F1=%.3f/%.3f；校准前后 BCE=%.4f/%.4f\n",
           validation_raw_classification.f1, validation_calibrated_classification.f1,
           validation_raw_probability.cross_entropy,
           validation_calibrated_probability.cross_entropy);
    print_classification_metrics("多数类基线（始终预测负类）", &baseline);
    print_classification_metrics("普通损失模型（固定阈值 0.50）",
                                 &unweighted_classification);
    print_classification_metrics("类别加权模型（固定阈值 0.50）",
                                 &weighted_default_classification);
    print_classification_metrics("加权模型（验证集阈值）", &raw_classification);
    print_probability_metrics("测试集校准前概率", &raw_probability, raw_bins);
    print_classification_metrics("校准模型（验证集阈值）", &calibrated_classification);
    print_probability_metrics("测试集校准后概率", &calibrated_probability, calibrated_bins);
    for (size_t index = 0; index < test_count; ++index) {
        if (test_predictions[index].label == 1) {
            printf("首个测试正例：logit=%.4f，原始概率=%.4f，校准概率=%.4f\n",
                   test_predictions[index].logit, test_predictions[index].probability,
                   calibrated_test[index].probability);
            break;
        }
    }
    printf("说明：训练集拟合标准化与模型；"
           "验证集选择阈值并拟合校准映射；"
           "测试集只用于最终报告；上述指标是点估计，不含置信区间。"
           "合成结果不代表真实数据表现。\n");
    return 0;
}
