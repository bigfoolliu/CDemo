/** @file @brief 验证 Softmax 概率、手算梯度和三分类数据流程。 */
#define main softmax_regression_demo_main
#include "../src/ml/25_softmax_regression.c"
#undef main
#include <assert.h>

/**
 * @brief 用比例为 1:2:3 的指数检查 Softmax 归一化与极端分数稳定性。
 */
static void test_softmax_probabilities_and_stability(void)
{
    double logits[CLASS_COUNT] = {0.0, log(2.0), log(3.0)};
    double probabilities[CLASS_COUNT] = {0};
    assert(softmax(logits, probabilities));
    assert(fabs(probabilities[0] - 1.0 / 6.0) < 1e-12);
    assert(fabs(probabilities[1] - 2.0 / 6.0) < 1e-12);
    assert(fabs(probabilities[2] - 3.0 / 6.0) < 1e-12);
    assert(fabs(probabilities[0] + probabilities[1] + probabilities[2] - 1.0) < 1e-12);

    double extreme_logits[CLASS_COUNT] = {1000.0, 999.0, 998.0};
    assert(softmax(extreme_logits, probabilities));
    assert(isfinite(probabilities[0]));
    assert(fabs(probabilities[0] - exp(0.0) / (exp(0.0) + exp(-1.0) + exp(-2.0))) < 1e-12);

    double invalid_logits[CLASS_COUNT] = {0.0, NAN, 1.0};
    double unchanged[CLASS_COUNT] = {7.0, 8.0, 9.0};
    assert(!softmax(invalid_logits, unchanged));
    assert(unchanged[0] == 7.0 && unchanged[1] == 8.0 && unchanged[2] == 9.0);
}

/**
 * @brief 用零 logits 的单样本手算交叉熵梯度并检查参数更新方向。
 */
static void test_one_sample_gradient_update(void)
{
    Dataset dataset = {0};
    dataset.count = 1;
    dataset.samples[0] = (Sample){{1.0, -1.0}, 0, SPLIT_TRAIN};
    SoftmaxModel model = {0};
    const double learning_rate = 0.1;
    assert(model_train_epoch(&dataset, learning_rate, &model));

    assert(fabs(model.weights[0][0] - learning_rate * 2.0 / 3.0) < 1e-12);
    assert(fabs(model.weights[0][1] + learning_rate * 2.0 / 3.0) < 1e-12);
    assert(fabs(model.biases[0] - learning_rate * 2.0 / 3.0) < 1e-12);
    for (size_t class_index = 1; class_index < CLASS_COUNT; ++class_index) {
        assert(fabs(model.weights[class_index][0] + learning_rate / 3.0) < 1e-12);
        assert(fabs(model.weights[class_index][1] - learning_rate / 3.0) < 1e-12);
        assert(fabs(model.biases[class_index] + learning_rate / 3.0) < 1e-12);
    }
}

/**
 * @brief 检查固定数据划分、训练效果和无效参数的失败保留。
 */
static void test_training_pipeline_and_failures(void)
{
    Dataset dataset = {0};
    assert(dataset_generate(&dataset));
    assert(dataset.count == SAMPLE_COUNT);
    size_t split_counts[SPLIT_COUNT] = {0};
    for (size_t index = 0; index < dataset.count; ++index) {
        ++split_counts[dataset.samples[index].split];
    }
    assert(split_counts[SPLIT_TRAIN] == 24);
    assert(split_counts[SPLIT_VALIDATION] == 6);
    assert(split_counts[SPLIT_TEST] == 6);

    SoftmaxModel model = {0};
    Metrics initial = {0};
    assert(evaluate(&dataset, SPLIT_VALIDATION, &model, &initial));
    for (size_t epoch = 0; epoch < EPOCH_COUNT; ++epoch) {
        assert(model_train_epoch(&dataset, LEARNING_RATE, &model));
    }
    Metrics training = {0};
    Metrics validation = {0};
    Metrics test = {0};
    assert(evaluate(&dataset, SPLIT_TRAIN, &model, &training));
    assert(evaluate(&dataset, SPLIT_VALIDATION, &model, &validation));
    assert(evaluate(&dataset, SPLIT_TEST, &model, &test));
    assert(validation.cross_entropy < initial.cross_entropy);
    assert(training.accuracy >= 0.95);
    assert(test.accuracy >= 0.9);

    SoftmaxModel unchanged = {.biases = {3.0, 4.0, 5.0}};
    assert(!model_train_epoch(&dataset, -1.0, &unchanged));
    assert(unchanged.biases[0] == 3.0 && unchanged.biases[2] == 5.0);
    Metrics unchanged_metrics = {.count = 77, .accuracy = 0.25};
    assert(!evaluate(&dataset, (DataSplit)-1, &model, &unchanged_metrics));
    assert(unchanged_metrics.count == 77 && unchanged_metrics.accuracy == 0.25);
}

/**
 * @brief 运行 Softmax 回归的确定性行为断言。
 * @return 所有断言通过时返回 0。
 */
int main(void)
{
    test_softmax_probabilities_and_stability();
    test_one_sample_gradient_update();
    test_training_pipeline_and_failures();
    puts("Softmax 回归测试通过");
    return 0;
}
