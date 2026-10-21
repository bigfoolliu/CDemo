/** @file @brief 验证类别不平衡指标、阈值选择和概率校准。 */
#define main reliability_assessment_demo_main
#include "../src/ml/24_reliability_assessment.c"
#undef main
#include <assert.h>

/**
 * @brief 检查固定种子生成、分层比例、训练和验证拟合。
 */
static void test_reproducible_pipeline_and_split_isolation(void)
{
    Dataset dataset = {0};
    Dataset repeated = {0};
    assert(dataset_generate(&dataset, UINT32_C(20261021)));
    assert(dataset_generate(&repeated, UINT32_C(20261021)));
    assert(dataset_is_valid(&dataset));
    assert(dataset.count == 400);
    assert(dataset.split_counts[SPLIT_TRAIN] == 240);
    assert(dataset.split_counts[SPLIT_VALIDATION] == 80);
    assert(dataset.split_counts[SPLIT_TEST] == 80);
    assert(dataset.class_counts[SPLIT_TRAIN][0] == 216);
    assert(dataset.class_counts[SPLIT_TRAIN][1] == 24);
    assert(dataset.class_counts[SPLIT_VALIDATION][0] == 72);
    assert(dataset.class_counts[SPLIT_VALIDATION][1] == 8);
    assert(dataset.class_counts[SPLIT_TEST][0] == 72);
    assert(dataset.class_counts[SPLIT_TEST][1] == 8);
    for (size_t index = 0; index < dataset.count; ++index) {
        assert(dataset.samples[index].label == repeated.samples[index].label);
        assert(dataset.samples[index].split == repeated.samples[index].split);
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            assert(dataset.samples[index].features[feature] ==
                   repeated.samples[index].features[feature]);
        }
    }

    Standardizer standardizer = {0};
    LogisticModel model = {0};
    Prediction validation[MAX_SAMPLE_COUNT] = {{0}};
    size_t validation_count = 0;
    double threshold = 0.0;
    CalibrationMap calibration = {0};
    assert(standardizer_fit(&dataset, &standardizer));
    assert(model_train(&dataset, &standardizer, true, &model));
    assert(collect_predictions(&model, &standardizer, &dataset, SPLIT_VALIDATION,
                               validation, &validation_count));
    assert(validation_count == 80);
    Prediction unchanged_predictions[MAX_SAMPLE_COUNT] = {{0}};
    unchanged_predictions[0].logit = 17.0;
    size_t unchanged_count = 99;
    assert(!collect_predictions(&model, &standardizer, &dataset, (DataSplit)-1,
                                unchanged_predictions, &unchanged_count));
    assert(unchanged_predictions[0].logit == 17.0);
    assert(unchanged_count == 99);
    assert(select_threshold(validation, validation_count, &threshold));
    assert(threshold >= 0.05 && threshold <= 0.95);
    assert(calibrator_fit(validation, validation_count, &calibration));
    assert(isfinite(calibration.slope));
    assert(isfinite(calibration.intercept));

    Prediction test[MAX_SAMPLE_COUNT] = {{0}};
    Prediction calibrated_test[MAX_SAMPLE_COUNT] = {{0}};
    size_t test_count = 0;
    Metrics raw_probability = {0};
    Metrics calibrated_probability = {0};
    assert(collect_predictions(&model, &standardizer, &dataset, SPLIT_TEST, test, &test_count));
    assert(test_count == 80);
    assert(probability_metrics(test, test_count, &raw_probability, nullptr));
    memcpy(calibrated_test, test, test_count * sizeof(*test));
    assert(calibrator_apply(&calibration, calibrated_test, test_count));
    assert(probability_metrics(calibrated_test, test_count, &calibrated_probability, nullptr));
    assert(calibrated_probability.brier_score < raw_probability.brier_score);
    assert(calibrated_probability.cross_entropy < raw_probability.cross_entropy);
    assert(calibrated_probability.expected_calibration_error <
           raw_probability.expected_calibration_error);

    Dataset altered_test = dataset;
    altered_test.class_counts[SPLIT_TEST][0] = 0;
    altered_test.class_counts[SPLIT_TEST][1] = 0;
    for (size_t index = 0; index < altered_test.count; ++index) {
        Sample *sample = &altered_test.samples[index];
        if (sample->split != SPLIT_TEST) {
            continue;
        }
        sample->features[0] += 1000.0;
        sample->features[1] -= 1000.0;
        sample->label = 1 - sample->label;
        ++altered_test.class_counts[SPLIT_TEST][sample->label];
    }
    Standardizer altered_standardizer = {0};
    LogisticModel altered_model = {0};
    assert(dataset_is_valid(&altered_test));
    assert(standardizer_fit(&altered_test, &altered_standardizer));
    assert(model_train(&altered_test, &altered_standardizer, true, &altered_model));
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        assert(standardizer.mean[feature] == altered_standardizer.mean[feature]);
        assert(standardizer.scale[feature] == altered_standardizer.scale[feature]);
        assert(model.weights[feature] == altered_model.weights[feature]);
    }
    assert(model.bias == altered_model.bias);
}

/**
 * @brief 用手算小样本检查混淆矩阵、分母为零和 Brier 分数。
 */
static void test_metrics_and_zero_denominators(void)
{
    Prediction predictions[] = {
        {1, 0.9, 2.1972245773362196},
        {1, 0.8, 1.3862943611198906},
        {0, 0.4, -0.4054651081081643},
        {0, 0.2, -1.3862943611198906}};
    Metrics classification = {0};
    Metrics probability = {0};
    ReliabilityBin bins[RELIABILITY_BIN_COUNT] = {{0}};
    assert(classification_metrics(predictions, 4, 0.5, &classification));
    assert(classification.true_negative == 2);
    assert(classification.false_positive == 0);
    assert(classification.false_negative == 0);
    assert(classification.true_positive == 2);
    assert(classification.accuracy == 1.0);
    assert(classification.precision == 1.0);
    assert(classification.recall == 1.0);
    assert(classification.f1 == 1.0);
    assert(classification.balanced_accuracy == 1.0);
    assert(probability_metrics(predictions, 4, &probability, bins));
    assert(fabs(probability.brier_score - 0.0625) < 1e-12);
    assert(fabs(probability.expected_calibration_error - 0.225) < 1e-12);

    Prediction all_negative[] = {{0, 0.1, -2.1972245773362196},
                                 {0, 0.2, -1.3862943611198906}};
    assert(classification_metrics(all_negative, 2, 0.5, &classification));
    assert(classification.precision == 0.0);
    assert(classification.recall == 0.0);
    assert(classification.f1 == 0.0);
    assert(classification.balanced_accuracy == 0.5);

    Metrics unchanged = {.count = 77, .true_positive = 9};
    assert(!classification_metrics(predictions, 4, NAN, &unchanged));
    assert(unchanged.count == 77);
    assert(unchanged.true_positive == 9);
}

/**
 * @brief 检查空类别验证集不能拟合校准映射且保留旧参数。
 */
static void test_calibration_requires_both_classes(void)
{
    Prediction only_negative[] = {{0, 0.2, -1.3862943611198906},
                                  {0, 0.3, -0.8472978603872037}};
    CalibrationMap unchanged = {.slope = 2.0, .intercept = -3.0};
    assert(!calibrator_fit(only_negative, 2, &unchanged));
    assert(unchanged.slope == 2.0);
    assert(unchanged.intercept == -3.0);
}

/**
 * @brief 运行可靠性评估相关的确定性断言。
 * @return 所有断言通过时返回 0。
 */
int main(void)
{
    test_reproducible_pipeline_and_split_isolation();
    test_metrics_and_zero_denominators();
    test_calibration_requires_both_classes();
    puts("可靠性评估测试通过");
    return 0;
}
