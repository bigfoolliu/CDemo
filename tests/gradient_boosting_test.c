/** @file @brief 验证回归树桩切分、残差拟合和验证集轮次选择。 */
#define main gradient_boosting_demo_main
#include "../src/ml/27_gradient_boosting.c"
#undef main
#include <assert.h>
#include <string.h>

/**
 * @brief 用手算残差验证最佳切分阈值与两片叶子的均值。
 */
static void test_stump_fit_by_hand(void)
{
    const double features[] = {0.0, 1.0, 2.0, 3.0};
    const double residuals[] = {0.0, 0.0, 2.0, 2.0};
    RegressionStump stump = {0};
    assert(stump_fit(features, residuals, 4, &stump));
    assert(stump.has_split);
    assert(stump.threshold == 1.5);
    assert(stump.left_value == 0.0);
    assert(stump.right_value == 2.0);

    double prediction = -1.0;
    assert(stump_predict(&stump, 1.5, &prediction));
    assert(prediction == 0.0);
    assert(stump_predict(&stump, 2.0, &prediction));
    assert(prediction == 2.0);
}

/**
 * @brief 检查特征无可切分值时回退到常数叶子且无效输入保留结果。
 */
static void test_stump_without_split_and_failure(void)
{
    const double features[] = {4.0, 4.0, 4.0};
    const double residuals[] = {1.0, 2.0, 3.0};
    RegressionStump stump = {0};
    assert(stump_fit(features, residuals, 3, &stump));
    assert(!stump.has_split);
    assert(stump.left_value == 2.0 && stump.right_value == 2.0);

    RegressionStump unchanged = {.threshold = 17.0, .left_value = 8.0};
    const double invalid_residuals[] = {1.0, NAN, 3.0};
    assert(!stump_fit(features, invalid_residuals, 3, &unchanged));
    assert(unchanged.threshold == 17.0 && unchanged.left_value == 8.0);
}

/**
 * @brief 检查训练只使用训练/验证划分，并且提升模型改善固定测试基线。
 */
static void test_training_selection_and_test_isolation(void)
{
    Sample samples[SAMPLE_COUNT] = {{0}};
    assert(dataset_generate(samples));
    BoostedModel model = {0};
    size_t best_round = 99;
    assert(model_train(samples, SAMPLE_COUNT, LEARNING_RATE, MAX_TRAINING_ROUNDS,
                       &model, &best_round));
    assert(best_round == model.round_count);
    assert(model.round_count > 0 && model.round_count <= MAX_TRAINING_ROUNDS);

    BoostedModel baseline = {.base_value = model.base_value};
    Metrics baseline_test = {0};
    Metrics boosted_test = {0};
    assert(evaluate(samples, SAMPLE_COUNT, SPLIT_TEST, &baseline, &baseline_test));
    assert(evaluate(samples, SAMPLE_COUNT, SPLIT_TEST, &model, &boosted_test));
    assert(boosted_test.mean_squared_error < baseline_test.mean_squared_error);

    Sample changed_test[SAMPLE_COUNT];
    memcpy(changed_test, samples, sizeof(changed_test));
    for (size_t index = 0; index < SAMPLE_COUNT; ++index) {
        if (changed_test[index].split == SPLIT_TEST) {
            changed_test[index].feature += 100.0;
            changed_test[index].target -= 50.0;
        }
    }
    BoostedModel isolated_model = {0};
    size_t isolated_round = 99;
    assert(model_train(changed_test, SAMPLE_COUNT, LEARNING_RATE, MAX_TRAINING_ROUNDS,
                       &isolated_model, &isolated_round));
    assert(isolated_round == best_round);
    assert(isolated_model.base_value == model.base_value);
    assert(isolated_model.round_count == model.round_count);
    for (size_t round = 0; round < model.round_count; ++round) {
        assert(isolated_model.stumps[round].threshold == model.stumps[round].threshold);
        assert(isolated_model.stumps[round].left_value == model.stumps[round].left_value);
        assert(isolated_model.stumps[round].right_value == model.stumps[round].right_value);
    }

    BoostedModel unchanged = {.base_value = 123.0};
    size_t unchanged_round = 77;
    assert(!model_train(samples, SAMPLE_COUNT, 0.0, MAX_TRAINING_ROUNDS,
                        &unchanged, &unchanged_round));
    assert(unchanged.base_value == 123.0 && unchanged_round == 77);
}

/**
 * @brief 运行梯度提升回归的确定性行为断言。
 * @return 所有断言通过时返回 0。
 */
int main(void)
{
    test_stump_fit_by_hand();
    test_stump_without_split_and_failure();
    test_training_selection_and_test_isolation();
    puts("梯度提升测试通过");
    return 0;
}
