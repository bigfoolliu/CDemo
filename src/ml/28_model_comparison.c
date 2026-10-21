/**
 * @file
 * @brief 在固定 CSV 数据上公平比较基线、逻辑回归、K 近邻和决策树桩。
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "study_dataset.h"

enum {
    RUN_COUNT = 5,
    LOGISTIC_CONFIG_COUNT = 3,
    KNN_CONFIG_COUNT = 3,
    LOGISTIC_EPOCH_COUNT = 500
};

static const uint32_t RUN_SEEDS[RUN_COUNT] = {11, 29, 47, 71, 101};
static const double LOGISTIC_LEARNING_RATES[LOGISTIC_CONFIG_COUNT] = {0.03, 0.10, 0.30};
static const size_t KNN_NEIGHBOR_COUNTS[KNN_CONFIG_COUNT] = {1, 3, 5};

typedef enum {
    MODEL_MAJORITY_BASELINE,
    MODEL_LOGISTIC_REGRESSION,
    MODEL_KNN,
    MODEL_DECISION_STUMP,
    MODEL_COUNT
} ModelKind;

typedef struct {
    size_t count;
    size_t true_negative;
    size_t false_positive;
    size_t false_negative;
    size_t true_positive;
    double accuracy;
    double balanced_accuracy;
    double precision;
    double recall;
    double f1;
} Metrics;

typedef struct {
    double weights[FEATURE_COUNT];
    double bias;
} LogisticModel;

typedef struct {
    size_t neighbor_count;
} KnnModel;

typedef struct {
    bool has_split;
    size_t feature;
    double threshold;
    int left_label;
    int right_label;
} DecisionStump;

typedef union {
    int majority_label;
    LogisticModel logistic;
    KnnModel knn;
    DecisionStump stump;
} ModelParameters;

typedef struct {
    ModelKind kind;
    ModelParameters parameters;
    double config_value;
    Metrics validation;
    Metrics test;
} SelectedModel;

typedef struct {
    uint32_t seed;
    Standardizer standardizer;
    SelectedModel models[MODEL_COUNT];
} RunResult;

typedef struct {
    double squared_distance;
    size_t sample_index;
    int label;
} Neighbor;

/**
 * @brief 推进用于划分的确定性 xorshift 伪随机数生成器。
 * @param state 调用方持有的可写非零生成器状态；函数会更新该状态。
 * @return 下一个可重复的 32 位伪随机数。
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
 * @brief 将开发样本分层打乱为共用的训练集和验证集。
 * @param source 有效 CSV 数据；失败时不修改原始划分。
 * @param seed 控制本轮划分的固定非零种子。
 * @param output 可写完整划分；无法划分时保持不变。
 * @return 两类均可按 3:1 划分且测试行未改变时为 true。
 */
static bool repeated_stratified_split(const Dataset *source, uint32_t seed, Dataset *output)
{
    if (!dataset_is_valid(source) || output == nullptr) {
        return false;
    }

    Dataset split = *source;
    size_t class_indices[2][MAX_SAMPLE_COUNT] = {{0}};
    size_t class_counts[2] = {0};
    for (size_t index = 0; index < source->count; ++index) {
        const Sample *sample = &source->samples[index];
        if (sample->split == SPLIT_TEST) {
            continue;
        }
        class_indices[sample->label][class_counts[sample->label]] = index;
        ++class_counts[sample->label];
    }

    for (size_t label = 0; label < 2; ++label) {
        if (class_counts[label] < 4 || class_counts[label] % 4 != 0) {
            return false;
        }
    }

    uint32_t random_state = seed == 0 ? UINT32_C(1) : seed;
    for (size_t label = 0; label < 2; ++label) {
        for (size_t remaining = class_counts[label]; remaining > 1; --remaining) {
            size_t selected = (size_t)(random_next(&random_state) % (uint32_t)remaining);
            size_t temporary = class_indices[label][remaining - 1];
            class_indices[label][remaining - 1] = class_indices[label][selected];
            class_indices[label][selected] = temporary;
        }

        size_t training_count = class_counts[label] * 3 / 4;
        for (size_t position = 0; position < class_counts[label]; ++position) {
            size_t index = class_indices[label][position];
            split.samples[index].split = position < training_count ? SPLIT_TRAIN : SPLIT_VALIDATION;
        }
    }

    memset(split.split_counts, 0, sizeof(split.split_counts));
    for (size_t index = 0; index < split.count; ++index) {
        ++split.split_counts[split.samples[index].split];
    }
    if (!dataset_is_valid(&split)) {
        return false;
    }

    *output = split;
    return true;
}

/**
 * @brief 将一个二分类预测计入混淆矩阵。
 * @param metrics 已清零的可写计数；标签有效时才更新。
 * @param actual 真实类别，只能为 0 或 1。
 * @param predicted 预测类别，只能为 0 或 1。
 * @return 两个类别值均有效时为 true。
 */
static bool metrics_add(Metrics *metrics, int actual, int predicted)
{
    if (metrics == nullptr || (actual != 0 && actual != 1) || (predicted != 0 && predicted != 1)) {
        return false;
    }

    if (actual == 0 && predicted == 0) {
        ++metrics->true_negative;
    } else if (actual == 0) {
        ++metrics->false_positive;
    } else if (predicted == 0) {
        ++metrics->false_negative;
    } else {
        ++metrics->true_positive;
    }
    ++metrics->count;
    return true;
}

/**
 * @brief 根据混淆矩阵计数计算统一的分类指标。
 * @param counts 二分类任务的非空混淆矩阵计数。
 * @param output 可写指标；失败时保持不变。
 * @return 所有计算结果均有限并可提交时为 true。
 */
static bool metrics_finish(const Metrics *counts, Metrics *output)
{
    if (counts == nullptr || output == nullptr || counts->count == 0) {
        return false;
    }

    Metrics completed = *counts;
    size_t negative_count = counts->true_negative + counts->false_positive;
    size_t positive_count = counts->true_positive + counts->false_negative;
    if (negative_count == 0 || positive_count == 0) {
        return false;
    }

    completed.accuracy =
        (double)(counts->true_negative + counts->true_positive) / (double)counts->count;
    double negative_recall = (double)counts->true_negative / (double)negative_count;
    completed.recall = (double)counts->true_positive / (double)positive_count;
    completed.balanced_accuracy = (negative_recall + completed.recall) / 2.0;
    size_t predicted_positive_count = counts->true_positive + counts->false_positive;
    completed.precision = predicted_positive_count == 0
                              ? 0.0
                              : (double)counts->true_positive / (double)predicted_positive_count;
    double precision_recall_sum = completed.precision + completed.recall;
    completed.f1 = precision_recall_sum == 0.0
                       ? 0.0
                       : 2.0 * completed.precision * completed.recall / precision_recall_sum;

    if (!isfinite(completed.accuracy) || !isfinite(completed.balanced_accuracy) ||
        !isfinite(completed.precision) || !isfinite(completed.recall) || !isfinite(completed.f1)) {
        return false;
    }

    *output = completed;
    return true;
}

/**
 * @brief 稳定计算一个标准化双特征样本的逻辑回归概率。
 * @param model 只读的已训练权重和偏置。
 * @param features 两个有限的标准化特征值。
 * @return [0, 1] 范围内的概率；中间值无效时返回 NAN。
 */
static double logistic_probability(const LogisticModel *model, const double features[FEATURE_COUNT])
{
    if (model == nullptr || features == nullptr) {
        return NAN;
    }

    double logit = model->bias;
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        logit += model->weights[feature] * features[feature];
    }
    if (!isfinite(logit)) {
        return NAN;
    }
    if (logit >= 0.0) {
        return 1.0 / (1.0 + exp(-logit));
    }

    double exponential = exp(logit);
    return exponential / (1.0 + exponential);
}

/**
 * @brief 只用当前训练样本通过全批量梯度下降拟合逻辑回归。
 * @param dataset 完整重复划分；更新参数时不会读取验证或测试样本。
 * @param standardizer 仅由本轮训练样本拟合的统计量。
 * @param learning_rate 从固定候选集合中选择的有限正步长。
 * @param output 可写拟合模型；失败时保留原值。
 * @return 固定轮数训练完成且参数均有限时为 true。
 */
static bool logistic_fit(const Dataset *dataset, const Standardizer *standardizer,
                         double learning_rate, LogisticModel *output)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || output == nullptr ||
        !isfinite(learning_rate) || learning_rate <= 0.0) {
        return false;
    }

    LogisticModel model = {0};
    double training_count = (double)dataset->split_counts[SPLIT_TRAIN];
    for (size_t epoch = 0; epoch < LOGISTIC_EPOCH_COUNT; ++epoch) {
        double weight_gradients[FEATURE_COUNT] = {0};
        double bias_gradient = 0.0;
        for (size_t index = 0; index < dataset->count; ++index) {
            const Sample *sample = &dataset->samples[index];
            if (sample->split != SPLIT_TRAIN) {
                continue;
            }

            double features[FEATURE_COUNT];
            if (!standardize_sample(standardizer, sample, features)) {
                return false;
            }
            double probability = logistic_probability(&model, features);
            if (!isfinite(probability)) {
                return false;
            }

            double error = probability - (double)sample->label;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                weight_gradients[feature] += error * features[feature];
            }
            bias_gradient += error;
        }

        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            model.weights[feature] -= learning_rate * weight_gradients[feature] / training_count;
        }
        model.bias -= learning_rate * bias_gradient / training_count;
    }

    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(model.weights[feature])) {
            return false;
        }
    }
    if (!isfinite(model.bias)) {
        return false;
    }

    *output = model;
    return true;
}

/**
 * @brief 先按距离、再按稳定 CSV 行索引排列邻居记录。
 * @param left qsort 传入的第一个邻居记录。
 * @param right qsort 传入的第二个邻居记录。
 * @return 按确定顺序比较后返回负数、零或正数。
 */
static int compare_neighbors(const void *left, const void *right)
{
    const Neighbor *first = left;
    const Neighbor *second = right;
    if (first->squared_distance < second->squared_distance) {
        return -1;
    }
    if (first->squared_distance > second->squared_distance) {
        return 1;
    }
    if (first->sample_index < second->sample_index) {
        return -1;
    }
    if (first->sample_index > second->sample_index) {
        return 1;
    }
    return 0;
}

/**
 * @brief 使用与其他模型相同的标准化特征，通过欧氏距离执行 K 近邻预测。
 * @param dataset 完整重复划分；仅训练样本参与投票。
 * @param standardizer 只由训练样本拟合的均值和尺度。
 * @param sample 待分类样本；忽略其标签。
 * @param neighbor_count 不超过训练样本数的奇数正整数。
 * @param output 可写类别预测；失败时保持不变。
 * @return 距离均有限且训练邻居数量足够时为 true。
 */
static bool knn_predict(const Dataset *dataset, const Standardizer *standardizer,
                        const Sample *sample, size_t neighbor_count, int *output)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || sample == nullptr ||
        output == nullptr || neighbor_count == 0 || neighbor_count % 2 == 0 ||
        neighbor_count > dataset->split_counts[SPLIT_TRAIN]) {
        return false;
    }

    double query[FEATURE_COUNT];
    if (!standardize_sample(standardizer, sample, query)) {
        return false;
    }

    Neighbor neighbors[MAX_SAMPLE_COUNT];
    size_t neighbor_total = 0;
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *candidate = &dataset->samples[index];
        if (candidate->split != SPLIT_TRAIN) {
            continue;
        }

        double features[FEATURE_COUNT];
        if (!standardize_sample(standardizer, candidate, features)) {
            return false;
        }
        double squared_distance = 0.0;
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double difference = query[feature] - features[feature];
            squared_distance += difference * difference;
        }
        if (!isfinite(squared_distance)) {
            return false;
        }
        neighbors[neighbor_total] = (Neighbor){squared_distance, index, candidate->label};
        ++neighbor_total;
    }

    if (neighbor_total != dataset->split_counts[SPLIT_TRAIN]) {
        return false;
    }
    qsort(neighbors, neighbor_total, sizeof(neighbors[0]), compare_neighbors);

    size_t positive_votes = 0;
    for (size_t index = 0; index < neighbor_count; ++index) {
        positive_votes += (size_t)neighbors[index].label;
    }
    *output = positive_votes * 2 > neighbor_count ? 1 : 0;
    return true;
}

/**
 * @brief 比较有限特征值，以便排序树桩阈值候选点。
 * @param left qsort 传入的第一个 double 指针。
 * @param right qsort 传入的第二个 double 指针。
 * @return 按数值顺序比较后返回负数、零或正数。
 */
static int compare_doubles(const void *left, const void *right)
{
    double first = *(const double *)left;
    double second = *(const double *)right;
    if (first < second) {
        return -1;
    }
    if (first > second) {
        return 1;
    }
    return 0;
}

/**
 * @brief 计算候选单层树切分的加权基尼不纯度。
 * @param dataset 完整划分；叶节点计数只来自训练样本。
 * @param standardizer 仅由训练样本拟合的特征变换参数。
 * @param feature 范围为 [0, FEATURE_COUNT) 的特征索引。
 * @param threshold 标准化特征阈值；等于阈值时归入左叶。
 * @param output 可写有限不纯度；失败时保持不变。
 * @return 两个叶节点均有样本且分数有限时为 true。
 */
static bool stump_split_impurity(const Dataset *dataset, const Standardizer *standardizer,
                                 size_t feature, double threshold, double *output)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || feature >= FEATURE_COUNT ||
        !isfinite(threshold) || output == nullptr) {
        return false;
    }

    size_t class_counts[2][2] = {{0}};
    size_t leaf_counts[2] = {0};
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != SPLIT_TRAIN) {
            continue;
        }

        double features[FEATURE_COUNT];
        if (!standardize_sample(standardizer, sample, features)) {
            return false;
        }
        size_t leaf = features[feature] <= threshold ? 0 : 1;
        ++class_counts[leaf][sample->label];
        ++leaf_counts[leaf];
    }

    if (leaf_counts[0] == 0 || leaf_counts[1] == 0) {
        return false;
    }

    double total = (double)(leaf_counts[0] + leaf_counts[1]);
    double weighted_impurity = 0.0;
    for (size_t leaf = 0; leaf < 2; ++leaf) {
        double leaf_total = (double)leaf_counts[leaf];
        double negative_fraction = (double)class_counts[leaf][0] / leaf_total;
        double positive_fraction = (double)class_counts[leaf][1] / leaf_total;
        double gini =
            1.0 - negative_fraction * negative_fraction - positive_fraction * positive_fraction;
        weighted_impurity += leaf_total / total * gini;
    }

    if (!isfinite(weighted_impurity)) {
        return false;
    }
    *output = weighted_impurity;
    return true;
}

/**
 * @brief 通过最小化训练集加权基尼不纯度拟合单次切分的 CART 分类器。
 * @param dataset 完整划分；只有训练样本决定树结构。
 * @param standardizer 由相同训练样本拟合的标准化统计量。
 * @param output 可写树桩；失败时保留原值。
 * @return 成功拟合有限切分或多数类常量叶时为 true。
 */
static bool stump_fit(const Dataset *dataset, const Standardizer *standardizer,
                      DecisionStump *output)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || output == nullptr) {
        return false;
    }

    DecisionStump stump = {0};
    size_t training_classes[2] = {0};
    for (size_t index = 0; index < dataset->count; ++index) {
        if (dataset->samples[index].split == SPLIT_TRAIN) {
            ++training_classes[dataset->samples[index].label];
        }
    }

    double best_impurity = INFINITY;
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        double values[MAX_SAMPLE_COUNT];
        size_t value_count = 0;
        for (size_t index = 0; index < dataset->count; ++index) {
            const Sample *sample = &dataset->samples[index];
            if (sample->split != SPLIT_TRAIN) {
                continue;
            }
            double features[FEATURE_COUNT];
            if (!standardize_sample(standardizer, sample, features)) {
                return false;
            }
            values[value_count] = features[feature];
            ++value_count;
        }

        qsort(values, value_count, sizeof(values[0]), compare_doubles);
        for (size_t index = 1; index < value_count; ++index) {
            if (values[index - 1] == values[index]) {
                continue;
            }
            double threshold = values[index - 1] + (values[index] - values[index - 1]) / 2.0;
            double impurity = 0.0;
            if (!stump_split_impurity(dataset, standardizer, feature, threshold, &impurity)) {
                return false;
            }
            if (impurity < best_impurity) {
                best_impurity = impurity;
                stump.has_split = true;
                stump.feature = feature;
                stump.threshold = threshold;
            }
        }
    }

    if (stump.has_split) {
        size_t leaf_classes[2][2] = {{0}};
        for (size_t index = 0; index < dataset->count; ++index) {
            const Sample *sample = &dataset->samples[index];
            if (sample->split != SPLIT_TRAIN) {
                continue;
            }
            double features[FEATURE_COUNT];
            if (!standardize_sample(standardizer, sample, features)) {
                return false;
            }
            size_t leaf = features[stump.feature] <= stump.threshold ? 0 : 1;
            ++leaf_classes[leaf][sample->label];
        }
        stump.left_label = leaf_classes[0][1] > leaf_classes[0][0] ? 1 : 0;
        stump.right_label = leaf_classes[1][1] > leaf_classes[1][0] ? 1 : 0;
    } else {
        int majority_label = training_classes[1] > training_classes[0] ? 1 : 0;
        stump.left_label = majority_label;
        stump.right_label = majority_label;
    }

    *output = stump;
    return true;
}

/**
 * @brief 使用已拟合的决策树桩预测一个类别。
 * @param stump 只读的已拟合树桩。
 * @param standardizer 仅由训练样本拟合的特征变换参数。
 * @param sample 待分类样本；忽略其标签。
 * @param output 可写类别预测；失败时保持不变。
 * @return 所选标准化特征有限时为 true。
 */
static bool stump_predict(const DecisionStump *stump, const Standardizer *standardizer,
                          const Sample *sample, int *output)
{
    if (stump == nullptr || standardizer == nullptr || sample == nullptr || output == nullptr) {
        return false;
    }
    if (!stump->has_split) {
        *output = stump->left_label;
        return true;
    }
    if (stump->feature >= FEATURE_COUNT || !isfinite(stump->threshold)) {
        return false;
    }

    double features[FEATURE_COUNT];
    if (!standardize_sample(standardizer, sample, features)) {
        return false;
    }
    *output = features[stump->feature] <= stump->threshold ? stump->left_label : stump->right_label;
    return true;
}

/**
 * @brief 根据所选模型保存的参数生成类别预测。
 * @param model 只读已拟合模型及选中的配置。
 * @param dataset 完整重复划分；KNN 只读取训练样本。
 * @param standardizer 仅由训练样本拟合的特征变换参数。
 * @param sample 待分类样本；忽略其标签。
 * @param output 可写预测类别；失败时保持不变。
 * @return 模型对应的预测成功时为 true。
 */
static bool model_predict(const SelectedModel *model, const Dataset *dataset,
                          const Standardizer *standardizer, const Sample *sample, int *output)
{
    if (model == nullptr || dataset == nullptr || standardizer == nullptr || sample == nullptr ||
        output == nullptr) {
        return false;
    }

    switch (model->kind) {
    case MODEL_MAJORITY_BASELINE:
        *output = model->parameters.majority_label;
        return true;
    case MODEL_LOGISTIC_REGRESSION: {
        double features[FEATURE_COUNT];
        if (!standardize_sample(standardizer, sample, features)) {
            return false;
        }
        double probability = logistic_probability(&model->parameters.logistic, features);
        if (!isfinite(probability)) {
            return false;
        }
        *output = probability >= 0.5 ? 1 : 0;
        return true;
    }
    case MODEL_KNN:
        return knn_predict(dataset, standardizer, sample, model->parameters.knn.neighbor_count,
                           output);
    case MODEL_DECISION_STUMP:
        return stump_predict(&model->parameters.stump, standardizer, sample, output);
    case MODEL_COUNT:
        return false;
    }
    return false;
}

/**
 * @brief 在指定划分上使用统一分类指标评估一个模型。
 * @param model 只读的已选模型。
 * @param dataset 完整重复划分。
 * @param standardizer 只由训练样本拟合的统计量。
 * @param split 非空验证集或测试集。
 * @param output 可写指标；失败时保持不变。
 * @return 每行均成功预测且划分含两个类别时为 true。
 */
static bool evaluate_model(const SelectedModel *model, const Dataset *dataset,
                           const Standardizer *standardizer, DataSplit split, Metrics *output)
{
    if (model == nullptr || !dataset_is_valid(dataset) || standardizer == nullptr ||
        split < SPLIT_TRAIN || split >= SPLIT_COUNT || output == nullptr ||
        dataset->split_counts[split] == 0) {
        return false;
    }

    Metrics counts = {0};
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != split) {
            continue;
        }
        int prediction = 0;
        if (!model_predict(model, dataset, standardizer, sample, &prediction) ||
            !metrics_add(&counts, sample->label, prediction)) {
            return false;
        }
    }
    if (counts.count != dataset->split_counts[split]) {
        return false;
    }
    return metrics_finish(&counts, output);
}

/**
 * @brief 统计训练标签并拟合确定性的多数类基线。
 * @param dataset 完整重复划分。
 * @param output 可写基线模型；失败时保持不变。
 * @return 训练集含两个类别时为 true；票数相同时预测类别 0。
 */
static bool majority_fit(const Dataset *dataset, SelectedModel *output)
{
    if (!dataset_is_valid(dataset) || output == nullptr) {
        return false;
    }

    size_t class_counts[2] = {0};
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split == SPLIT_TRAIN) {
            ++class_counts[sample->label];
        }
    }

    SelectedModel model = {0};
    model.kind = MODEL_MAJORITY_BASELINE;
    model.parameters.majority_label = class_counts[1] > class_counts[0] ? 1 : 0;
    *output = model;
    return true;
}

/**
 * @brief 按验证集平衡准确率选择最佳配置；并列时保留靠前配置。
 * @param dataset 当前种子下各模型共用的开发集与测试集划分。
 * @param standardizer 本轮只由训练样本拟合的预处理参数。
 * @param kind 要拟合的模型类别。
 * @param output 可写选中模型；失败时保持不变。
 * @return 所有候选配置均能在验证样本上评估时为 true。
 */
static bool select_model(const Dataset *dataset, const Standardizer *standardizer, ModelKind kind,
                         SelectedModel *output)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || output == nullptr ||
        kind < MODEL_MAJORITY_BASELINE || kind >= MODEL_COUNT) {
        return false;
    }

    if (kind == MODEL_MAJORITY_BASELINE) {
        SelectedModel baseline = {0};
        if (!majority_fit(dataset, &baseline) ||
            !evaluate_model(&baseline, dataset, standardizer, SPLIT_VALIDATION,
                            &baseline.validation)) {
            return false;
        }
        *output = baseline;
        return true;
    }

    size_t candidate_count = 1;
    if (kind == MODEL_LOGISTIC_REGRESSION) {
        candidate_count = LOGISTIC_CONFIG_COUNT;
    } else if (kind == MODEL_KNN) {
        candidate_count = KNN_CONFIG_COUNT;
    }
    bool has_best = false;
    SelectedModel best = {0};
    for (size_t candidate = 0; candidate < candidate_count; ++candidate) {
        SelectedModel current = {0};
        current.kind = kind;
        if (kind == MODEL_LOGISTIC_REGRESSION) {
            current.config_value = LOGISTIC_LEARNING_RATES[candidate];
            if (!logistic_fit(dataset, standardizer, current.config_value,
                              &current.parameters.logistic)) {
                return false;
            }
        } else if (kind == MODEL_KNN) {
            current.parameters.knn.neighbor_count = KNN_NEIGHBOR_COUNTS[candidate];
            current.config_value = (double)current.parameters.knn.neighbor_count;
        } else if (!stump_fit(dataset, standardizer, &current.parameters.stump)) {
            return false;
        }

        if (!evaluate_model(&current, dataset, standardizer, SPLIT_VALIDATION,
                            &current.validation)) {
            return false;
        }
        if (!has_best ||
            current.validation.balanced_accuracy > best.validation.balanced_accuracy + 1e-12) {
            best = current;
            has_best = true;
        }
    }

    if (!has_best) {
        return false;
    }
    *output = best;
    return true;
}

/**
 * @brief 执行共用划分、训练集预处理、验证选参与测试评估的完整流程。
 * @param source 有效的原始 CSV 数据集。
 * @param seed 用于开发集分层划分的固定非零种子。
 * @param output 可写单轮结果；失败时保持不变。
 * @return 四种模型均完成验证选参与测试评估时为 true。
 */
static bool run_single_comparison(const Dataset *source, uint32_t seed, RunResult *output)
{
    if (!dataset_is_valid(source) || output == nullptr) {
        return false;
    }

    Dataset split = {0};
    if (!repeated_stratified_split(source, seed, &split)) {
        return false;
    }
    Standardizer standardizer = {0};
    if (!standardizer_fit(&split, &standardizer)) {
        return false;
    }

    RunResult result = {0};
    result.seed = seed;
    result.standardizer = standardizer;
    for (size_t kind = 0; kind < MODEL_COUNT; ++kind) {
        if (!select_model(&split, &standardizer, (ModelKind)kind, &result.models[kind]) ||
            !evaluate_model(&result.models[kind], &split, &standardizer, SPLIT_TEST,
                            &result.models[kind].test)) {
            return false;
        }
    }

    *output = result;
    return true;
}

/**
 * @brief 返回比较模型的可读名称。
 * @param kind 有效的模型枚举值。
 * @return 静态模型名称；枚举值无效时返回“未知模型”。
 */
static const char *model_name(ModelKind kind)
{
    switch (kind) {
    case MODEL_MAJORITY_BASELINE:
        return "多数类基线";
    case MODEL_LOGISTIC_REGRESSION:
        return "逻辑回归";
    case MODEL_KNN:
        return "K 近邻";
    case MODEL_DECISION_STUMP:
        return "决策树桩";
    case MODEL_COUNT:
        return "未知模型";
    }
    return "未知模型";
}

/**
 * @brief 输出验证集选中的配置和本轮测试指标。
 * @param model 含验证分数和测试分数的模型结果。
 * @return 模型类别有效且所有结果均可输出时为 true。
 */
static bool print_run_model(const SelectedModel *model)
{
    if (model == nullptr || model->kind < MODEL_MAJORITY_BASELINE || model->kind >= MODEL_COUNT) {
        return false;
    }

    printf("  %-12s 配置=", model_name(model->kind));
    if (model->kind == MODEL_MAJORITY_BASELINE) {
        printf("多数类=%d", model->parameters.majority_label);
    } else if (model->kind == MODEL_LOGISTIC_REGRESSION) {
        printf("学习率=%.2f", model->config_value);
    } else if (model->kind == MODEL_KNN) {
        printf("k=%zu", model->parameters.knn.neighbor_count);
    } else if (model->parameters.stump.has_split) {
        printf("特征%zu阈值=%+.3f", model->parameters.stump.feature,
               model->parameters.stump.threshold);
    } else {
        printf("常数叶=%d", model->parameters.stump.left_label);
    }
    printf(" 验证平衡准确率=%.3f 测试准确率=%.3f 平衡准确率=%.3f"
           " 精确率=%.3f 召回率=%.3f F1=%.3f\n",
           model->validation.balanced_accuracy, model->test.accuracy, model->test.balanced_accuracy,
           model->test.precision, model->test.recall, model->test.f1);
    return true;
}

/**
 * @brief 计算多轮测试指标的均值与总体标准差。
 * @param results 已完成的多轮结果记录。
 * @param kind 要汇总测试指标的模型类别。
 * @param metric_index 指标编号：0 准确率、1 平衡准确率、2 精确率、3 召回率、4 F1。
 * @param mean 可写的有限均值输出；失败时保持不变。
 * @param deviation 可写的有限总体标准差输出。
 * @return 所需数据均存在且计算结果有限时为 true。
 */
static bool summarize_test_metric(const RunResult results[RUN_COUNT], ModelKind kind,
                                  size_t metric_index, double *mean, double *deviation)
{
    if (results == nullptr || kind < MODEL_MAJORITY_BASELINE || kind >= MODEL_COUNT ||
        metric_index > 4 || mean == nullptr || deviation == nullptr) {
        return false;
    }

    double values[RUN_COUNT];
    double total = 0.0;
    for (size_t run = 0; run < RUN_COUNT; ++run) {
        const Metrics *metrics = &results[run].models[kind].test;
        if (metric_index == 0) {
            values[run] = metrics->accuracy;
        } else if (metric_index == 1) {
            values[run] = metrics->balanced_accuracy;
        } else if (metric_index == 2) {
            values[run] = metrics->precision;
        } else if (metric_index == 3) {
            values[run] = metrics->recall;
        } else {
            values[run] = metrics->f1;
        }
        if (!isfinite(values[run])) {
            return false;
        }
        total += values[run];
    }

    double average = total / (double)RUN_COUNT;
    double squared_deviation_sum = 0.0;
    for (size_t run = 0; run < RUN_COUNT; ++run) {
        double difference = values[run] - average;
        squared_deviation_sum += difference * difference;
    }
    double standard_deviation = sqrt(squared_deviation_sum / (double)RUN_COUNT);
    if (!isfinite(average) || !isfinite(standard_deviation)) {
        return false;
    }

    *mean = average;
    *deviation = standard_deviation;
    return true;
}

/**
 * @brief 使用五个固定种子执行公平比较，并输出每轮记录和汇总。
 * @param argc 命令行参数数量；可选传入 CSV 路径。
 * @param argv 参数数组；argv[1] 可替换默认 CSV 路径。
 * @return 比较完整成功时返回 EXIT_SUCCESS，否则返回 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc > 2) {
        fprintf(stderr, "用法：%s [CSV路径]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *path = argc == 2 ? argv[1] : DEFAULT_CSV_PATH;
    Dataset source = {0};
    size_t error_line = 0;
    if (!dataset_load_csv(path, &source, &error_line)) {
        if (error_line > 0) {
            fprintf(stderr, "CSV 格式错误：%s（第 %zu 行）\n", path, error_line);
        } else {
            perror(path);
        }
        return EXIT_FAILURE;
    }
    if (source.split_counts[SPLIT_TEST] == 0) {
        fprintf(stderr, "数据集缺少独立测试划分\n");
        return EXIT_FAILURE;
    }

    RunResult results[RUN_COUNT] = {{0}};
    printf("数据集：%s（%zu 行；原始 train=%zu、validation=%zu、test=%zu）\n", path, source.count,
           source.split_counts[SPLIT_TRAIN], source.split_counts[SPLIT_VALIDATION],
           source.split_counts[SPLIT_TEST]);
    printf("每轮合并 train/validation 作为开发集，再按类别 3:1 分为训练/验证；"
           "固定 test 不参与标准化或选参。\n");

    for (size_t run = 0; run < RUN_COUNT; ++run) {
        if (!run_single_comparison(&source, RUN_SEEDS[run], &results[run])) {
            fprintf(stderr, "种子 %u 的比较流程失败\n", RUN_SEEDS[run]);
            return EXIT_FAILURE;
        }

        printf("\n重复实验 %zu/%d，seed=%u，标准化均值=(%.3f, %.3f)，"
               "标准差=(%.3f, %.3f)\n",
               run + 1, RUN_COUNT, results[run].seed, results[run].standardizer.mean[0],
               results[run].standardizer.mean[1], results[run].standardizer.scale[0],
               results[run].standardizer.scale[1]);
        for (size_t kind = 0; kind < MODEL_COUNT; ++kind) {
            if (!print_run_model(&results[run].models[kind])) {
                fprintf(stderr, "无法输出模型结果\n");
                return EXIT_FAILURE;
            }
        }
    }

    printf("\n测试集结果：五轮均值 ± 总体标准差（每轮 test=%zu 条）\n",
           source.split_counts[SPLIT_TEST]);
    printf("  %-12s 准确率       平衡准确率   精确率       召回率       F1\n", "模型");
    for (size_t kind = 0; kind < MODEL_COUNT; ++kind) {
        double accuracy_mean = 0.0;
        double accuracy_deviation = 0.0;
        double balanced_mean = 0.0;
        double balanced_deviation = 0.0;
        double precision_mean = 0.0;
        double precision_deviation = 0.0;
        double recall_mean = 0.0;
        double recall_deviation = 0.0;
        double f1_mean = 0.0;
        double f1_deviation = 0.0;
        if (!summarize_test_metric(results, (ModelKind)kind, 0, &accuracy_mean,
                                   &accuracy_deviation) ||
            !summarize_test_metric(results, (ModelKind)kind, 1, &balanced_mean,
                                   &balanced_deviation) ||
            !summarize_test_metric(results, (ModelKind)kind, 2, &precision_mean,
                                   &precision_deviation) ||
            !summarize_test_metric(results, (ModelKind)kind, 3, &recall_mean, &recall_deviation) ||
            !summarize_test_metric(results, (ModelKind)kind, 4, &f1_mean, &f1_deviation)) {
            fprintf(stderr, "无法汇总测试指标\n");
            return EXIT_FAILURE;
        }
        printf("  %-12s %.3f ± %.3f  %.3f ± %.3f  %.3f ± %.3f"
               "  %.3f ± %.3f  %.3f ± %.3f\n",
               model_name((ModelKind)kind), accuracy_mean, accuracy_deviation, balanced_mean,
               balanced_deviation, precision_mean, precision_deviation, recall_mean,
               recall_deviation, f1_mean, f1_deviation);
    }

    puts("\n测试集每轮都只用于最终评估；小样本结果仅用于学习流程，不代表现实泛化能力。");
    return EXIT_SUCCESS;
}
