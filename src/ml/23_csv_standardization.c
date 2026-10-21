/**
 * @file
 * @brief 从 CSV 读取小型二分类数据，训练集标准化并独立评估测试集。
 */
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "study_dataset.h"

enum {
    EPOCH_LIMIT = 1500,
    EARLY_STOP_PATIENCE = 80
};

static const double LEARNING_RATE = 0.2;

typedef struct {
    double weights[FEATURE_COUNT];
    double bias;
} LogisticModel;

typedef struct {
    double cross_entropy;
    size_t count;
    size_t correct;
    size_t true_negative;
    size_t false_positive;
    size_t false_negative;
    size_t true_positive;
} Metrics;

typedef struct {
    size_t epochs_run;
    size_t best_epoch;
    double best_validation_loss;
} TrainingSummary;

/**
 * @brief 稳定计算 sigmoid，避免对大正数直接计算指数。
 * @param value 有限线性分数。
 * @return [0,1] 范围内的正类概率。
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
 * @brief 以模型和训练标准化参数计算一个样本的概率与变换后特征。
 * @param model 只读逻辑回归模型。
 * @param standardizer 只读训练集标准化参数。
 * @param sample 待预测的只读样本。
 * @param probability 有效可写概率输出；失败时保持不变。
 * @param standardized 可选的 FEATURE_COUNT 个变换特征输出。
 * @return 中间值与输出均有限时为 true。
 */
static bool model_predict(const LogisticModel *model, const Standardizer *standardizer,
                          const Sample *sample, double *probability,
                          double standardized[FEATURE_COUNT])
{
    if (model == nullptr || standardizer == nullptr || sample == nullptr || probability == nullptr) {
        return false;
    }

    double transformed[FEATURE_COUNT];
    if (!standardize_sample(standardizer, sample, transformed) || !isfinite(model->bias)) {
        return false;
    }

    double logit = model->bias;
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(model->weights[feature])) {
            return false;
        }
        logit += model->weights[feature] * transformed[feature];
    }
    if (!isfinite(logit)) {
        return false;
    }

    double predicted_probability = sigmoid(logit);
    if (!isfinite(predicted_probability)) {
        return false;
    }

    *probability = predicted_probability;
    if (standardized != nullptr) {
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            standardized[feature] = transformed[feature];
        }
    }
    return true;
}

/**
 * @brief 稳定计算二元交叉熵对应的单样本 logit 损失。
 * @param logit sigmoid 之前的有限实数分数。
 * @param label 真实标签 0 或 1。
 * @return 有限的非负交叉熵。
 */
static double binary_cross_entropy(double logit, int label)
{
    return fmax(logit, 0.0) - (double)label * logit + log1p(exp(-fabs(logit)));
}

/**
 * @brief 计算指定划分的交叉熵、准确率和二分类混淆矩阵。
 * @param model 只读已训练模型。
 * @param standardizer 只读训练集拟合的标准化参数。
 * @param dataset 完整只读数据集。
 * @param split 要评估的非空划分。
 * @param output 有效可写指标；失败时保持不变。
 * @return 划分中每个样本都成功预测且指标有限时为 true。
 */
static bool evaluate_split(const LogisticModel *model, const Standardizer *standardizer,
                           const Dataset *dataset, DataSplit split, Metrics *output)
{
    if (model == nullptr || standardizer == nullptr || !dataset_is_valid(dataset) ||
        split < SPLIT_TRAIN || split >= SPLIT_COUNT || output == nullptr ||
        dataset->split_counts[split] == 0) {
        return false;
    }

    Metrics metrics = {0};
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != split) {
            continue;
        }

        double probability = 0.0;
        double standardized[FEATURE_COUNT];
        if (!model_predict(model, standardizer, sample, &probability, standardized)) {
            return false;
        }

        double logit = model->bias;
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            logit += model->weights[feature] * standardized[feature];
        }
        metrics.cross_entropy += binary_cross_entropy(logit, sample->label);
        int prediction = probability >= 0.5 ? 1 : 0;
        if (prediction == sample->label) {
            ++metrics.correct;
        }
        if (sample->label == 0 && prediction == 0) {
            ++metrics.true_negative;
        } else if (sample->label == 0) {
            ++metrics.false_positive;
        } else if (prediction == 0) {
            ++metrics.false_negative;
        } else {
            ++metrics.true_positive;
        }
        ++metrics.count;
    }

    if (metrics.count != dataset->split_counts[split]) {
        return false;
    }
    metrics.cross_entropy /= (double)metrics.count;
    if (!isfinite(metrics.cross_entropy)) {
        return false;
    }

    *output = metrics;
    return true;
}

/**
 * @brief 用全批量梯度下降训练，并按验证交叉熵保留最佳轮次。
 * @param dataset 含非空训练、验证和测试集的只读数据。
 * @param standardizer 仅由训练集拟合的标准化参数。
 * @param output 有效可写模型；失败时保持不变。
 * @param summary 有效可写训练摘要；失败时保持不变。
 * @return 找到有限的最佳模型并完成训练时为 true。
 */
static bool model_train(const Dataset *dataset, const Standardizer *standardizer,
                        LogisticModel *output, TrainingSummary *summary)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || output == nullptr ||
        summary == nullptr || !isfinite(LEARNING_RATE) || LEARNING_RATE <= 0.0) {
        return false;
    }

    LogisticModel current = {0};
    LogisticModel best_model = current;
    Metrics validation_metrics = {0};
    if (!evaluate_split(&current, standardizer, dataset, SPLIT_VALIDATION, &validation_metrics)) {
        return false;
    }

    double best_validation_loss = validation_metrics.cross_entropy;
    size_t best_epoch = 0;
    size_t stale_epochs = 0;
    size_t epochs_run = 0;
    size_t training_count = dataset->split_counts[SPLIT_TRAIN];

    for (size_t epoch = 1; epoch <= EPOCH_LIMIT; ++epoch) {
        double weight_gradients[FEATURE_COUNT] = {0};
        double bias_gradient = 0.0;
        for (size_t index = 0; index < dataset->count; ++index) {
            const Sample *sample = &dataset->samples[index];
            if (sample->split != SPLIT_TRAIN) {
                continue;
            }

            double probability = 0.0;
            double standardized[FEATURE_COUNT];
            if (!model_predict(&current, standardizer, sample, &probability, standardized)) {
                return false;
            }

            double error = probability - (double)sample->label;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                weight_gradients[feature] += error * standardized[feature];
            }
            bias_gradient += error;
        }

        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            current.weights[feature] -=
                LEARNING_RATE * weight_gradients[feature] / (double)training_count;
        }
        current.bias -= LEARNING_RATE * bias_gradient / (double)training_count;
        epochs_run = epoch;

        if (!evaluate_split(&current, standardizer, dataset, SPLIT_VALIDATION,
                            &validation_metrics)) {
            return false;
        }
        if (validation_metrics.cross_entropy < best_validation_loss - 1e-12) {
            best_validation_loss = validation_metrics.cross_entropy;
            best_model = current;
            best_epoch = epoch;
            stale_epochs = 0;
        } else {
            ++stale_epochs;
            if (stale_epochs >= EARLY_STOP_PATIENCE) {
                break;
            }
        }
    }

    if (!isfinite(best_validation_loss)) {
        return false;
    }

    *output = best_model;
    *summary = (TrainingSummary){epochs_run, best_epoch, best_validation_loss};
    return true;
}

/**
 * @brief 输出一个划分中的全部模型预测，供观察原始与标准化特征。
 * @param model 只读模型。
 * @param standardizer 只读训练集标准化参数。
 * @param dataset 只读数据集。
 * @param split 要显示的非空划分。
 * @return 所有样本成功预测时为 true。
 */
static bool print_predictions(const LogisticModel *model, const Standardizer *standardizer,
                              const Dataset *dataset, DataSplit split)
{
    if (model == nullptr || standardizer == nullptr || !dataset_is_valid(dataset) ||
        split < SPLIT_TRAIN || split >= SPLIT_COUNT) {
        return false;
    }

    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != split) {
            continue;
        }

        double probability = 0.0;
        double standardized[FEATURE_COUNT];
        if (!model_predict(model, standardizer, sample, &probability, standardized)) {
            return false;
        }
        int prediction = probability >= 0.5 ? 1 : 0;
        printf("  原始=(%5.1f, %6.1f) 标准化=(%+.6f, %+.6f) 标签=%d 预测=%d 概率=%.6f\n",
               sample->features[0], sample->features[1], standardized[0], standardized[1],
               sample->label, prediction, probability);
    }
    return true;
}

/**
 * @brief 运行 CSV 读取、训练集标准化、验证集选轮次和独立测试集评估。
 * @param argc 命令行参数数量；可选提供一个 CSV 路径。
 * @param argv 参数数组；argv[1] 替换默认数据路径。
 * @return 成功返回 EXIT_SUCCESS；输入、数值或评估失败返回 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc > 2) {
        fprintf(stderr, "用法：%s [CSV路径]\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *path = argc == 2 ? argv[1] : DEFAULT_CSV_PATH;
    Dataset dataset = {0};
    size_t error_line = 0;
    if (!dataset_load_csv(path, &dataset, &error_line)) {
        if (error_line > 0) {
            fprintf(stderr, "CSV 格式错误：%s（第 %zu 行）\n", path, error_line);
        } else {
            perror(path);
        }
        return EXIT_FAILURE;
    }

    Standardizer standardizer = {0};
    if (!standardizer_fit(&dataset, &standardizer)) {
        fprintf(stderr, "无法从训练集拟合标准化参数\n");
        return EXIT_FAILURE;
    }

    LogisticModel model = {0};
    TrainingSummary summary = {0};
    if (!model_train(&dataset, &standardizer, &model, &summary)) {
        fprintf(stderr, "模型训练或验证集评估失败\n");
        return EXIT_FAILURE;
    }

    Metrics training_metrics = {0};
    Metrics validation_metrics = {0};
    Metrics test_metrics = {0};
    if (!evaluate_split(&model, &standardizer, &dataset, SPLIT_TRAIN, &training_metrics) ||
        !evaluate_split(&model, &standardizer, &dataset, SPLIT_VALIDATION,
                        &validation_metrics) ||
        !evaluate_split(&model, &standardizer, &dataset, SPLIT_TEST, &test_metrics)) {
        fprintf(stderr, "最终指标计算失败\n");
        return EXIT_FAILURE;
    }

    printf("CSV 数据：%s（%zu 行：训练 %zu，验证 %zu，测试 %zu）\n", path,
           dataset.count, dataset.split_counts[SPLIT_TRAIN],
           dataset.split_counts[SPLIT_VALIDATION], dataset.split_counts[SPLIT_TEST]);
    printf("训练集标准化参数（总体标准差）：\n");
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        printf("  特征 %zu：均值=%10.8f，标准差=%10.8f\n", feature,
               standardizer.mean[feature], standardizer.scale[feature]);
    }
    printf("验证集选择最佳轮次：%zu（实际训练 %zu 轮，验证 BCE=%.4f）\n",
           summary.best_epoch, summary.epochs_run, summary.best_validation_loss);
    printf("模型参数：w=(%+.8f, %+.8f)，b=%+.8f\n", model.weights[0],
           model.weights[1], model.bias);
    printf("训练集 BCE=%.4f，验证集 BCE=%.4f\n", training_metrics.cross_entropy,
           validation_metrics.cross_entropy);
    printf("独立测试集：BCE=%.4f，准确率=%zu/%zu（%.1f%%）\n",
           test_metrics.cross_entropy, test_metrics.correct, test_metrics.count,
           100.0 * (double)test_metrics.correct / (double)test_metrics.count);
    printf("混淆矩阵（行=真实标签，列=预测标签）：\n");
    printf("              预测 0  预测 1\n");
    printf("  真实 0      %6zu  %6zu\n", test_metrics.true_negative,
           test_metrics.false_positive);
    printf("  真实 1      %6zu  %6zu\n", test_metrics.false_negative,
           test_metrics.true_positive);
    puts("独立测试样本预测：");
    if (!print_predictions(&model, &standardizer, &dataset, SPLIT_TEST)) {
        fprintf(stderr, "测试样本输出失败\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
