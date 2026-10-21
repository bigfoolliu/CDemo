/**
 * @file
 * @brief 用纯 C 实现带训练、验证和测试划分的三分类 Softmax 回归。
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

enum {
    FEATURE_COUNT = 2,
    CLASS_COUNT = 3,
    POINTS_PER_CLASS = 12,
    SAMPLE_COUNT = CLASS_COUNT * POINTS_PER_CLASS,
    TRAIN_PER_CLASS = 8,
    VALIDATION_PER_CLASS = 2,
    TEST_PER_CLASS = POINTS_PER_CLASS - TRAIN_PER_CLASS - VALIDATION_PER_CLASS,
    EPOCH_COUNT = 400
};

static const double LEARNING_RATE = 0.12;

typedef enum {
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_TEST,
    SPLIT_COUNT
} DataSplit;

typedef struct {
    double features[FEATURE_COUNT];
    size_t label;
    DataSplit split;
} Sample;

typedef struct {
    Sample samples[SAMPLE_COUNT];
    size_t count;
} Dataset;

typedef struct {
    double weights[CLASS_COUNT][FEATURE_COUNT];
    double biases[CLASS_COUNT];
} SoftmaxModel;

typedef struct {
    double probabilities[CLASS_COUNT];
    size_t predicted_class;
    double loss;
} Prediction;

typedef struct {
    size_t correct;
    size_t count;
    double accuracy;
    double cross_entropy;
} Metrics;

/**
 * @brief 生成三个线性可分的二维类别及固定的数据划分。
 * @param output 有效可写数据集；成功时整体替换，失败时不修改。
 * @return 数据集大小及每类训练、验证、测试样本数有效时为 true。
 */
static bool dataset_generate(Dataset *output)
{
    if (output == nullptr) {
        return false;
    }

    static const double centers[CLASS_COUNT][FEATURE_COUNT] = {
        {-2.0, -1.2}, {2.0, -1.2}, {0.0, 2.2}};
    static const double offsets[POINTS_PER_CLASS][FEATURE_COUNT] = {
        {-0.55, -0.20}, {-0.35, 0.35}, {-0.15, -0.45}, {0.05, 0.10},
        {0.20, -0.25}, {0.35, 0.40}, {0.50, -0.05}, {0.60, 0.20},
        {-0.45, 0.15}, {-0.25, -0.10}, {0.15, 0.30}, {0.45, -0.35}};
    Dataset generated = {0};
    generated.count = SAMPLE_COUNT;

    for (size_t class_index = 0; class_index < CLASS_COUNT; ++class_index) {
        for (size_t point = 0; point < POINTS_PER_CLASS; ++point) {
            size_t index = class_index * POINTS_PER_CLASS + point;
            Sample *sample = &generated.samples[index];
            sample->label = class_index;
            if (point < TRAIN_PER_CLASS) {
                sample->split = SPLIT_TRAIN;
            } else if (point < TRAIN_PER_CLASS + VALIDATION_PER_CLASS) {
                sample->split = SPLIT_VALIDATION;
            } else {
                sample->split = SPLIT_TEST;
            }
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                sample->features[feature] = centers[class_index][feature] + offsets[point][feature];
            }
        }
    }

    *output = generated;
    return true;
}

/**
 * @brief 计算稳定的 Softmax 概率，先减去最大 logit 避免指数溢出。
 * @param logits 三个有限类别分数的只读数组。
 * @param probabilities 至少 CLASS_COUNT 个可写位置；成功时写入概率。
 * @return 输入有限且输出指针有效时为 true，失败时不修改输出。
 */
static bool softmax(const double logits[CLASS_COUNT], double probabilities[CLASS_COUNT])
{
    if (logits == nullptr || probabilities == nullptr) {
        return false;
    }

    /* 先找到最大分数，避免直接计算 exp(很大的正数)。 */
    double maximum = -DBL_MAX;
    for (size_t class_index = 0; class_index < CLASS_COUNT; ++class_index) {
        if (!isfinite(logits[class_index])) {
            return false;
        }
        if (logits[class_index] > maximum) {
            maximum = logits[class_index];
        }
    }

    double values[CLASS_COUNT] = {0};
    double denominator = 0.0;
    /* 使用减最大值后的指数，再归一化为概率。 */
    for (size_t class_index = 0; class_index < CLASS_COUNT; ++class_index) {
        values[class_index] = exp(logits[class_index] - maximum);
        denominator += values[class_index];
    }
    if (!isfinite(denominator) || denominator <= 0.0) {
        return false;
    }
    for (size_t class_index = 0; class_index < CLASS_COUNT; ++class_index) {
        values[class_index] /= denominator;
    }

    memcpy(probabilities, values, sizeof(values));
    return true;
}

/**
 * @brief 对单个样本执行线性打分、Softmax 和交叉熵计算。
 * @param model 有限的只读模型参数。
 * @param sample 特征有限且类别编号在有效范围内的样本。
 * @param output 有效可写预测结果；成功时整体替换。
 * @return 参数和中间数值有效时为 true，否则不修改预测结果。
 */
static bool model_predict(const SoftmaxModel *model, const Sample *sample, Prediction *output)
{
    if (model == nullptr || sample == nullptr || output == nullptr ||
        sample->label >= CLASS_COUNT) {
        return false;
    }

    double logits[CLASS_COUNT] = {0};
    for (size_t class_index = 0; class_index < CLASS_COUNT; ++class_index) {
        if (!isfinite(model->biases[class_index])) {
            return false;
        }
        logits[class_index] = model->biases[class_index];
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            if (!isfinite(model->weights[class_index][feature]) ||
                !isfinite(sample->features[feature])) {
                return false;
            }
            logits[class_index] += model->weights[class_index][feature] * sample->features[feature];
        }
    }

    Prediction prediction = {0};
    if (!softmax(logits, prediction.probabilities)) {
        return false;
    }
    prediction.predicted_class = 0;
    for (size_t class_index = 1; class_index < CLASS_COUNT; ++class_index) {
        if (prediction.probabilities[class_index] >
            prediction.probabilities[prediction.predicted_class]) {
            prediction.predicted_class = class_index;
        }
    }
    prediction.loss = -log(fmax(prediction.probabilities[sample->label], DBL_MIN));
    if (!isfinite(prediction.loss)) {
        return false;
    }

    *output = prediction;
    return true;
}

/**
 * @brief 用训练集平均梯度执行一次多分类交叉熵下降。
 * @param dataset 含有限训练样本的只读数据集。
 * @param learning_rate 有限正学习率。
 * @param model 可读写模型；成功后参数整体更新。
 * @return 至少一个训练样本且更新有限时为 true。
 */
static bool model_train_epoch(const Dataset *dataset, double learning_rate, SoftmaxModel *model)
{
    if (dataset == nullptr || model == nullptr || !isfinite(learning_rate) ||
        learning_rate <= 0.0 || dataset->count == 0 || dataset->count > SAMPLE_COUNT) {
        return false;
    }

    double weight_gradients[CLASS_COUNT][FEATURE_COUNT] = {{0}};
    double bias_gradients[CLASS_COUNT] = {0};
    size_t training_count = 0;
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != SPLIT_TRAIN) {
            continue;
        }
        if (sample->label >= CLASS_COUNT) {
            return false;
        }
        Prediction prediction = {0};
        if (!model_predict(model, sample, &prediction)) {
            return false;
        }
        /* Softmax + 交叉熵对 logit 的导数是概率减 one-hot 标签。 */
        for (size_t class_index = 0; class_index < CLASS_COUNT; ++class_index) {
            double error = prediction.probabilities[class_index] -
                           (class_index == sample->label ? 1.0 : 0.0);
            bias_gradients[class_index] += error;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                weight_gradients[class_index][feature] += error * sample->features[feature];
            }
        }
        ++training_count;
    }
    if (training_count == 0) {
        return false;
    }

    /* 梯度先全部算完，再更新临时副本，保证失败时模型保持原状。 */
    SoftmaxModel updated = *model;
    for (size_t class_index = 0; class_index < CLASS_COUNT; ++class_index) {
        updated.biases[class_index] -= learning_rate * bias_gradients[class_index] /
                                       (double)training_count;
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            updated.weights[class_index][feature] -= learning_rate *
                weight_gradients[class_index][feature] / (double)training_count;
        }
        if (!isfinite(updated.biases[class_index])) {
            return false;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            if (!isfinite(updated.weights[class_index][feature])) {
                return false;
            }
        }
    }

    *model = updated;
    return true;
}

/**
 * @brief 汇总指定划分的准确率和平均交叉熵。
 * @param dataset 含有效样本划分的只读数据集。
 * @param split 训练、验证或测试划分。
 * @param model 用于预测的只读模型。
 * @param output 有效可写指标；成功时整体替换。
 * @return 划分非空且预测有效时为 true。
 */
static bool evaluate(const Dataset *dataset, DataSplit split, const SoftmaxModel *model,
                     Metrics *output)
{
    if (dataset == nullptr || model == nullptr || output == nullptr || split < SPLIT_TRAIN ||
        split >= SPLIT_COUNT || dataset->count == 0 || dataset->count > SAMPLE_COUNT) {
        return false;
    }

    Metrics metrics = {0};
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != split) {
            continue;
        }
        Prediction prediction = {0};
        if (!model_predict(model, sample, &prediction)) {
            return false;
        }
        metrics.correct += prediction.predicted_class == sample->label;
        metrics.cross_entropy += prediction.loss;
        ++metrics.count;
    }
    if (metrics.count == 0) {
        return false;
    }
    metrics.accuracy = (double)metrics.correct / (double)metrics.count;
    metrics.cross_entropy /= (double)metrics.count;
    *output = metrics;
    return true;
}

/**
 * @brief 运行短训练并展示训练、验证和测试指标。
 * @return 数据生成、训练或评估失败时返回非零值。
 */
int main(void)
{
    Dataset dataset = {0};
    SoftmaxModel model = {0};
    if (!dataset_generate(&dataset)) {
        return 1;
    }

    Metrics initial_validation = {0};
    if (!evaluate(&dataset, SPLIT_VALIDATION, &model, &initial_validation)) {
        return 1;
    }
    for (size_t epoch = 0; epoch < EPOCH_COUNT; ++epoch) {
        if (!model_train_epoch(&dataset, LEARNING_RATE, &model)) {
            return 1;
        }
    }

    Metrics training = {0};
    Metrics validation = {0};
    Metrics test = {0};
    if (!evaluate(&dataset, SPLIT_TRAIN, &model, &training) ||
        !evaluate(&dataset, SPLIT_VALIDATION, &model, &validation) ||
        !evaluate(&dataset, SPLIT_TEST, &model, &test)) {
        return 1;
    }

    printf("三分类 Softmax 回归（%d 个固定样本，每类训练/验证/测试=%d/%d/%d）\n",
           SAMPLE_COUNT, TRAIN_PER_CLASS, VALIDATION_PER_CLASS, TEST_PER_CLASS);
    printf("学习率=%.2f，训练轮数=%d，验证交叉熵 %.4f -> %.4f\n", LEARNING_RATE,
           EPOCH_COUNT, initial_validation.cross_entropy, validation.cross_entropy);
    printf("训练：准确率 %.1f%%，交叉熵 %.4f（%zu/%zu）\n", training.accuracy * 100.0,
           training.cross_entropy, training.correct, training.count);
    printf("验证：准确率 %.1f%%，交叉熵 %.4f（%zu/%zu）\n", validation.accuracy * 100.0,
           validation.cross_entropy, validation.correct, validation.count);
    printf("测试：准确率 %.1f%%，交叉熵 %.4f（%zu/%zu）\n", test.accuracy * 100.0,
           test.cross_entropy, test.correct, test.count);
    return 0;
}
