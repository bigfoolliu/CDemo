/**
 * @file
 * @brief 分层 K 折交叉验证：折内标准化、逻辑回归正则选参与最终测试。
 */
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

enum {
    FEATURE_COUNT = 2,
    DEVELOPMENT_COUNT = 120,
    TEST_COUNT = 48,
    FOLD_COUNT = 4,
    EPOCH_COUNT = 600,
    REGULARIZATION_COUNT = 4
};

static const double REGULARIZATION_CANDIDATES[REGULARIZATION_COUNT] = {
    0.0001, 0.01, 0.1, 1.0};

typedef struct {
    double features[FEATURE_COUNT];
    int label;
} Sample;

typedef struct {
    Sample samples[DEVELOPMENT_COUNT];
    size_t count;
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
    double log_loss;
    size_t errors;
    size_t count;
} Metrics;

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
 * @brief 生成固定、平衡且特征尺度不同的二维二分类数据。
 * @param dataset 有效可写数据集；成功后整体替换。
 * @param count 偶数样本数，范围 2..DEVELOPMENT_COUNT。
 * @param seed 固定种子；开发集与最终测试集使用不同种子。
 * @return 参数有效且数据有限为 true，否则不修改数据集。
 */
static bool dataset_generate(Dataset *dataset, size_t count, uint32_t seed)
{
    if (dataset == nullptr || count < 2 || count > DEVELOPMENT_COUNT || count % 2 != 0) {
        return false;
    }
    Dataset generated = {0};
    generated.count = count;
    for (size_t i = 0; i < count; ++i) {
        int label = i % 2 == 0 ? -1 : 1;
        double latent = (double)label * 0.65 + 1.1 * (2.0 * random_unit(&seed) - 1.0);
        double small_noise = 2.0 * random_unit(&seed) - 1.0;
        generated.samples[i] = (Sample){{12.0 * latent + 3.0 * small_noise,
                                        0.03 * latent + 0.012 * small_noise}, label};
    }
    *dataset = generated;
    return true;
}

/**
 * @brief 打乱指定范围内的下标，供可重复的分层折分使用。
 * @param indices 至少 count 个可写下标；内容会被重排。
 * @param count 下标数量。
 * @param state 有效可写伪随机状态。
 */
static void shuffle_indices(size_t *indices, size_t count, uint32_t *state)
{
    for (size_t i = count; i > 1; --i) {
        size_t selected = (size_t)(random_unit(state) * (double)i);
        size_t temporary = indices[i - 1];
        indices[i - 1] = indices[selected];
        indices[selected] = temporary;
    }
}

/**
 * @brief 按类别分别打乱并轮流分配样本，使每折尽量保持类别比例。
 * @param dataset 标签为 -1/+1 的有效只读开发集。
 * @param folds 可写折编号数组，长度至少 dataset->count。
 * @param fold_count 折数，范围 2..dataset->count。
 * @param seed 固定折分种子。
 * @return 每个样本恰好分配到一个有效折为 true，否则为 false。
 */
static bool stratified_folds(const Dataset *dataset, size_t *folds, size_t fold_count,
                             uint32_t seed)
{
    if (dataset == nullptr || folds == nullptr || fold_count < 2 ||
        fold_count > dataset->count || dataset->count > DEVELOPMENT_COUNT) {
        return false;
    }
    size_t class_indices[2][DEVELOPMENT_COUNT] = {{0}};
    size_t class_counts[2] = {0};
    for (size_t i = 0; i < dataset->count; ++i) {
        int label = dataset->samples[i].label;
        if (label != -1 && label != 1) {
            return false;
        }
        size_t class_index = label > 0 ? 1 : 0;
        class_indices[class_index][class_counts[class_index]++] = i;
    }
    if (class_counts[0] == 0 || class_counts[1] == 0) {
        return false;
    }
    for (size_t class_index = 0; class_index < 2; ++class_index) {
        shuffle_indices(class_indices[class_index], class_counts[class_index], &seed);
        for (size_t i = 0; i < class_counts[class_index]; ++i) {
            folds[class_indices[class_index][i]] = i % fold_count;
        }
    }
    return true;
}

/**
 * @brief 仅从给定训练下标拟合每特征均值和标准差。
 * @param dataset 完整只读开发集。
 * @param indices 训练子集下标，不含当前验证折。
 * @param count 训练子集样本数，必须大于零。
 * @param output 有效可写标准化参数；成功后整体替换。
 * @return 所有统计量有限且尺度非零为 true，否则不修改输出。
 */
static bool standardizer_fit(const Dataset *dataset, const size_t *indices, size_t count,
                             Standardizer *output)
{
    if (dataset == nullptr || indices == nullptr || output == nullptr || count == 0) {
        return false;
    }
    Standardizer fitted = {0};
    for (size_t i = 0; i < count; ++i) {
        size_t index = indices[i];
        if (index >= dataset->count) {
            return false;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double value = dataset->samples[index].features[feature];
            if (!isfinite(value)) {
                return false;
            }
            fitted.mean[feature] += value;
        }
    }
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        fitted.mean[feature] /= (double)count;
    }
    for (size_t i = 0; i < count; ++i) {
        size_t index = indices[i];
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double difference = dataset->samples[index].features[feature] - fitted.mean[feature];
            fitted.scale[feature] += difference * difference;
        }
    }
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        fitted.scale[feature] = sqrt(fitted.scale[feature] / (double)count);
        if (!isfinite(fitted.scale[feature]) || fitted.scale[feature] <= DBL_EPSILON) {
            return false;
        }
    }
    *output = fitted;
    return true;
}

/**
 * @brief 把一个样本用已拟合训练子集的统计量标准化。
 * @param standardizer 只读标准化参数。
 * @param sample 只读输入样本。
 * @param output 两个有效可写标准化特征；成功后整体替换。
 * @return 输入和结果均有限为 true，否则不修改输出。
 */
static bool standardize_sample(const Standardizer *standardizer, const Sample *sample,
                               double output[FEATURE_COUNT])
{
    if (standardizer == nullptr || sample == nullptr || output == nullptr) {
        return false;
    }
    double standardized[FEATURE_COUNT];
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(sample->features[feature]) ||
            !isfinite(standardizer->mean[feature]) ||
            !isfinite(standardizer->scale[feature]) || standardizer->scale[feature] <= 0.0) {
            return false;
        }
        standardized[feature] =
            (sample->features[feature] - standardizer->mean[feature]) /
            standardizer->scale[feature];
    }
    output[0] = standardized[0];
    output[1] = standardized[1];
    return true;
}

/**
 * @brief 对有限线性输出稳定计算 sigmoid 概率。
 * @param value 有限实数决策分数。
 * @return [0,1] 范围的正类概率。
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
 * @brief 用逻辑回归平均交叉熵加 L2 权重项拟合标准化训练子集。
 * @param model 可写模型；成功后整体替换。
 * @param dataset 完整只读开发集。
 * @param indices 训练子集下标。
 * @param count 训练子集大小，至少为 2。
 * @param standardizer 仅由此训练子集拟合的标准化参数。
 * @param regularization 非负有限 L2 系数。
 * @return 优化完成且参数有限为 true；失败时模型不变。
 */
static bool logistic_fit(LogisticModel *model, const Dataset *dataset, const size_t *indices,
                         size_t count, const Standardizer *standardizer,
                         double regularization)
{
    if (model == nullptr || dataset == nullptr || indices == nullptr || standardizer == nullptr ||
        count < 2 || !isfinite(regularization) || regularization < 0.0) {
        return false;
    }
    LogisticModel fitted = {0};
    for (size_t epoch = 0; epoch < EPOCH_COUNT; ++epoch) {
        double gradient[FEATURE_COUNT] = {0};
        double bias_gradient = 0.0;
        for (size_t i = 0; i < count; ++i) {
            if (indices[i] >= dataset->count) {
                return false;
            }
            const Sample *sample = &dataset->samples[indices[i]];
            double features[FEATURE_COUNT];
            if ((sample->label != -1 && sample->label != 1) ||
                !standardize_sample(standardizer, sample, features)) {
                return false;
            }
            double score = fitted.bias;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                score += fitted.weights[feature] * features[feature];
            }
            double target = sample->label > 0 ? 1.0 : 0.0;
            double error = sigmoid(score) - target;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                gradient[feature] += error * features[feature];
            }
            bias_gradient += error;
        }
        double learning_rate = 0.2 / (1.0 + 0.003 * (double)epoch);
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            gradient[feature] = gradient[feature] / (double)count +
                                regularization * fitted.weights[feature];
            fitted.weights[feature] -= learning_rate * gradient[feature];
        }
        fitted.bias -= learning_rate * bias_gradient / (double)count;
    }
    if (!isfinite(fitted.weights[0]) || !isfinite(fitted.weights[1]) ||
        !isfinite(fitted.bias)) {
        return false;
    }
    *model = fitted;
    return true;
}

/**
 * @brief 在一个指定子集上计算平均交叉熵和分类错误数。
 * @param model 已拟合只读模型。
 * @param dataset 完整只读开发集或最终测试集。
 * @param indices 评估子集下标。
 * @param count 评估样本数，大于零。
 * @param standardizer 用对应训练子集拟合的只读标准化参数。
 * @param output 有效可写指标；失败时不修改。
 * @return 评估成功且损失有限为 true，否则为 false。
 */
static bool evaluate_indices(const LogisticModel *model, const Dataset *dataset,
                             const size_t *indices, size_t count,
                             const Standardizer *standardizer, Metrics *output)
{
    if (model == nullptr || dataset == nullptr || indices == nullptr ||
        standardizer == nullptr || output == nullptr || count == 0) {
        return false;
    }
    Metrics metrics = {.count = count};
    for (size_t i = 0; i < count; ++i) {
        if (indices[i] >= dataset->count) {
            return false;
        }
        const Sample *sample = &dataset->samples[indices[i]];
        double features[FEATURE_COUNT];
        if ((sample->label != -1 && sample->label != 1) ||
            !standardize_sample(standardizer, sample, features)) {
            return false;
        }
        double score = model->bias;
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            score += model->weights[feature] * features[feature];
        }
        double target = sample->label > 0 ? 1.0 : 0.0;
        double probability = sigmoid(score);
        double clipped = fmin(1.0 - 1e-15, fmax(1e-15, probability));
        metrics.log_loss -= target * log(clipped) + (1.0 - target) * log1p(-clipped);
        metrics.errors += (score >= 0.0 ? 1 : -1) != sample->label;
    }
    metrics.log_loss /= (double)count;
    if (!isfinite(metrics.log_loss)) {
        return false;
    }
    *output = metrics;
    return true;
}

/**
 * @brief 汇总全部 K 折，为每个配置计算一次平均验证交叉熵。
 * @param dataset 只读训练开发集；不得传入最终测试集。
 * @param folds 每个开发样本对应的折编号。
 * @param fold_count 折数，至少为 2。
 * @param regularization 待比较的非负 L2 系数。
 * @param fold_losses 可写数组，长度至少 fold_count；成功后写入各折损失。
 * @param mean_loss 有效可写平均损失输出。
 * @return 每折拟合和验证均成功为 true，否则不保证数组内容。
 */
static bool cross_validate(const Dataset *dataset, const size_t *folds, size_t fold_count,
                           double regularization, double *fold_losses, double *mean_loss)
{
    if (dataset == nullptr || folds == nullptr || fold_losses == nullptr || mean_loss == nullptr ||
        dataset->count < fold_count || fold_count < 2) {
        return false;
    }
    size_t coverage[DEVELOPMENT_COUNT] = {0};
    double total_loss = 0.0;
    for (size_t fold = 0; fold < fold_count; ++fold) {
        size_t training_indices[DEVELOPMENT_COUNT];
        size_t validation_indices[DEVELOPMENT_COUNT];
        size_t training_count = 0;
        size_t validation_count = 0;
        for (size_t i = 0; i < dataset->count; ++i) {
            if (folds[i] >= fold_count) {
                return false;
            }
            if (folds[i] == fold) {
                validation_indices[validation_count++] = i;
                ++coverage[i];
            } else {
                training_indices[training_count++] = i;
            }
        }
        Standardizer standardizer;
        LogisticModel model;
        Metrics metrics;
        if (training_count == 0 || validation_count == 0 ||
            !standardizer_fit(dataset, training_indices, training_count, &standardizer) ||
            !logistic_fit(&model, dataset, training_indices, training_count, &standardizer,
                          regularization) ||
            !evaluate_indices(&model, dataset, validation_indices, validation_count,
                              &standardizer, &metrics)) {
            return false;
        }
        fold_losses[fold] = metrics.log_loss;
        total_loss += metrics.log_loss;
    }
    for (size_t i = 0; i < dataset->count; ++i) {
        if (coverage[i] != 1) {
            return false;
        }
    }
    *mean_loss = total_loss / (double)fold_count;
    return isfinite(*mean_loss);
}

/**
 * @brief 对训练开发集计算正则系数交叉验证分数并只在选参后评估最终测试集。
 * @return 所有分折、重训和最终评估成功为 0，否则为 1。
 */
int main(void)
{
    Dataset development = {0};
    Dataset test = {0};
    if (!dataset_generate(&development, DEVELOPMENT_COUNT, UINT32_C(20261015)) ||
        !dataset_generate(&test, TEST_COUNT, UINT32_C(20261016))) {
        fputs("生成固定数据失败\n", stderr);
        return 1;
    }
    size_t folds[DEVELOPMENT_COUNT];
    if (!stratified_folds(&development, folds, FOLD_COUNT, UINT32_C(4567))) {
        fputs("分层 K 折失败\n", stderr);
        return 1;
    }
    puts("4 折分层交叉验证：只使用训练开发集选正则强度；最终测试集保持封存。");
    double best_loss = INFINITY;
    double selected_regularization = REGULARIZATION_CANDIDATES[0];
    for (size_t candidate = 0; candidate < REGULARIZATION_COUNT; ++candidate) {
        double fold_losses[FOLD_COUNT] = {0};
        double mean_loss = 0.0;
        if (!cross_validate(&development, folds, FOLD_COUNT,
                            REGULARIZATION_CANDIDATES[candidate], fold_losses, &mean_loss)) {
            fputs("交叉验证拟合或评估失败\n", stderr);
            return 1;
        }
        printf("lambda=%7.4f：各折 log-loss=[%.4f, %.4f, %.4f, %.4f]，平均 %.4f\n",
               REGULARIZATION_CANDIDATES[candidate], fold_losses[0], fold_losses[1],
               fold_losses[2], fold_losses[3], mean_loss);
        if (mean_loss < best_loss) {
            best_loss = mean_loss;
            selected_regularization = REGULARIZATION_CANDIDATES[candidate];
        }
    }

    size_t all_indices[DEVELOPMENT_COUNT];
    size_t test_indices[TEST_COUNT];
    for (size_t i = 0; i < development.count; ++i) {
        all_indices[i] = i;
    }
    for (size_t i = 0; i < test.count; ++i) {
        test_indices[i] = i;
    }
    Standardizer final_standardizer;
    LogisticModel final_model;
    Metrics final_metrics;
    if (!standardizer_fit(&development, all_indices, development.count, &final_standardizer) ||
        !logistic_fit(&final_model, &development, all_indices, development.count,
                      &final_standardizer, selected_regularization) ||
        !evaluate_indices(&final_model, &test, test_indices, test.count, &final_standardizer,
                          &final_metrics)) {
        fputs("选参后重训或最终测试评估失败\n", stderr);
        return 1;
    }
    printf("选择 lambda=%.4f；在全部 %zu 条开发样本上重训。\n",
           selected_regularization, development.count);
    printf("最终测试集（仅评估一次）：log-loss=%.4f，错误=%zu/%zu，准确率=%.1f%%\n",
           final_metrics.log_loss, final_metrics.errors, final_metrics.count,
           100.0 * (double)(final_metrics.count - final_metrics.errors) /
               (double)final_metrics.count);
    printf("最终标准化参数（全开发集重训时拟合）：均值=[%.3f, %.5f]，标准差=[%.3f, %.5f]\n",
           final_standardizer.mean[0], final_standardizer.mean[1],
           final_standardizer.scale[0], final_standardizer.scale[1]);
    return !ferror(stdout) ? 0 : 1;
}
