/** @file @brief 验证公平比较中的分层划分、指标、模型选择与测试集隔离。 */
#define main model_comparison_demo_main
#include "../src/ml/28_model_comparison.c"
#undef main

#include <assert.h>
#include <string.h>

/**
 * @brief 用手算验证准确率、平衡准确率、精确率、召回率和 F1。
 */
static void test_metrics_by_hand(void)
{
    Metrics counts = {0};
    assert(metrics_add(&counts, 0, 0));
    assert(metrics_add(&counts, 0, 1));
    assert(metrics_add(&counts, 1, 1));
    assert(metrics_add(&counts, 1, 1));

    Metrics metrics = {0};
    assert(metrics_finish(&counts, &metrics));
    assert(metrics.true_negative == 1);
    assert(metrics.false_positive == 1);
    assert(metrics.false_negative == 0);
    assert(metrics.true_positive == 2);
    assert(fabs(metrics.accuracy - 0.75) < 1e-12);
    assert(fabs(metrics.balanced_accuracy - 0.75) < 1e-12);
    assert(fabs(metrics.precision - 2.0 / 3.0) < 1e-12);
    assert(fabs(metrics.recall - 1.0) < 1e-12);
    assert(fabs(metrics.f1 - 0.8) < 1e-12);

    Metrics unchanged = {.count = 17, .accuracy = 42.0};
    Metrics missing_class = {.count = 2, .true_negative = 2};
    assert(!metrics_finish(&missing_class, &unchanged));
    assert(unchanged.count == 17 && unchanged.accuracy == 42.0);
}

/**
 * @brief 验证固定种子生成可复现的分层划分，供所有模型共用。
 */
static void test_repeated_split_is_reproducible_and_shared(void)
{
    Dataset source = {0};
    assert(dataset_load_csv(DEFAULT_CSV_PATH, &source, nullptr));
    Dataset first = {0};
    Dataset repeated = {0};
    Dataset other_seed = {0};
    assert(repeated_stratified_split(&source, 11, &first));
    assert(repeated_stratified_split(&source, 11, &repeated));
    assert(repeated_stratified_split(&source, 29, &other_seed));
    assert(first.split_counts[SPLIT_TRAIN] == 24);
    assert(first.split_counts[SPLIT_VALIDATION] == 8);
    assert(first.split_counts[SPLIT_TEST] == 8);

    size_t training_classes[2] = {0};
    size_t validation_classes[2] = {0};
    size_t changed_development_memberships = 0;
    for (size_t index = 0; index < source.count; ++index) {
        assert(first.samples[index].features[0] == repeated.samples[index].features[0]);
        assert(first.samples[index].features[1] == repeated.samples[index].features[1]);
        assert(first.samples[index].label == repeated.samples[index].label);
        assert(first.samples[index].split == repeated.samples[index].split);
        if (first.samples[index].split == SPLIT_TRAIN) {
            ++training_classes[first.samples[index].label];
        } else if (first.samples[index].split == SPLIT_VALIDATION) {
            ++validation_classes[first.samples[index].label];
        } else {
            assert(first.samples[index].split == source.samples[index].split);
        }
        if (first.samples[index].split != SPLIT_TEST &&
            first.samples[index].split != other_seed.samples[index].split) {
            ++changed_development_memberships;
        }
    }
    assert(training_classes[0] == 12 && training_classes[1] == 12);
    assert(validation_classes[0] == 4 && validation_classes[1] == 4);
    assert(changed_development_memberships > 0);
    assert(source.split_counts[SPLIT_TRAIN] == 24);
    assert(source.split_counts[SPLIT_VALIDATION] == 8);
}

/**
 * @brief 验证每轮标准化拟合会忽略验证集和测试集的极端值。
 */
static void test_standardizer_uses_training_rows_only(void)
{
    Dataset source = {0};
    assert(dataset_load_csv(DEFAULT_CSV_PATH, &source, nullptr));
    Dataset split = {0};
    assert(repeated_stratified_split(&source, 47, &split));
    Standardizer original = {0};
    assert(standardizer_fit(&split, &original));

    Dataset changed_held_out = split;
    for (size_t index = 0; index < changed_held_out.count; ++index) {
        Sample *sample = &changed_held_out.samples[index];
        if (sample->split != SPLIT_TRAIN) {
            sample->features[0] += 10000.0;
            sample->features[1] -= 20000.0;
        }
    }
    Standardizer unchanged = {0};
    assert(standardizer_fit(&changed_held_out, &unchanged));
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        assert(original.mean[feature] == unchanged.mean[feature]);
        assert(original.scale[feature] == unchanged.scale[feature]);
    }
}

/**
 * @brief 验证手工构造的树桩能找到单特征的完美切分。
 */
static void test_stump_split_by_hand(void)
{
    const char *csv = "study_hours,practice_questions,passed,split\n"
                      "-2,0,0,train\n"
                      "-1,0,0,train\n"
                      "1,1,1,train\n"
                      "2,1,1,train\n"
                      "-0.5,0,0,validation\n"
                      "0.5,1,1,validation\n"
                      "-0.25,0,0,test\n"
                      "0.25,1,1,test\n";
    FILE *stream = tmpfile();
    assert(stream != nullptr);
    assert(fputs(csv, stream) >= 0);
    rewind(stream);

    Dataset dataset = {0};
    assert(dataset_load_stream(stream, &dataset, nullptr));
    assert(fclose(stream) == 0);
    Standardizer standardizer = {0};
    assert(standardizer_fit(&dataset, &standardizer));

    DecisionStump stump = {0};
    assert(stump_fit(&dataset, &standardizer, &stump));
    assert(stump.has_split);
    assert(stump.feature == 0);
    assert(stump.left_label == 0 && stump.right_label == 1);
    double impurity = 1.0;
    assert(
        stump_split_impurity(&dataset, &standardizer, stump.feature, stump.threshold, &impurity));
    assert(impurity == 0.0);

    int prediction = -1;
    assert(stump_predict(&stump, &standardizer, &dataset.samples[6], &prediction));
    assert(prediction == 0);
    assert(stump_predict(&stump, &standardizer, &dataset.samples[7], &prediction));
    assert(prediction == 1);
}

/**
 * @brief 验证共同测试下标的手算差值和固定模型自助区间。
 */
static void test_paired_bootstrap(void)
{
    double differences[] = {3.0, 0.0, 8.0};
    size_t indices[] = {0, 0, 2};
    double resampled_mean = 0.0;
    assert(resampled_mean_difference(differences, 3, indices, 3, &resampled_mean));
    assert(resampled_mean == 14.0 / 3.0);
    size_t invalid_indices[] = {0, 3, 2};
    assert(!resampled_mean_difference(differences, 3, invalid_indices, 3, &resampled_mean));

    Dataset source = {0};
    assert(dataset_load_csv(DEFAULT_CSV_PATH, &source, nullptr));
    RunResult run = {0};
    assert(run_single_comparison(&source, RUN_SEEDS[0], &run));
    double paired_losses[MAX_SAMPLE_COUNT] = {0};
    size_t paired_count = 0;
    assert(collect_paired_test_losses(&source, &run, MODEL_LOGISTIC_REGRESSION, MODEL_KNN,
                                      paired_losses, &paired_count));
    assert(paired_count == source.split_counts[SPLIT_TEST]);
    BootstrapInterval first = {0};
    BootstrapInterval second = {0};
    assert(paired_bootstrap(paired_losses, paired_count, UINT32_C(0x28b007), &first));
    assert(paired_bootstrap(paired_losses, paired_count, UINT32_C(0x28b007), &second));
    assert(memcmp(&first, &second, sizeof(first)) == 0);
    assert(first.lower <= first.upper);
    assert(isfinite(first.mean_difference));
    assert(!paired_bootstrap(paired_losses, 0, UINT32_C(0x28b007), &second));
}

/**
 * @brief 验证重复结果有限，且测试数据变化不会影响选参。
 */
static void test_comparison_and_test_isolation(void)
{
    Dataset source = {0};
    assert(dataset_load_csv(DEFAULT_CSV_PATH, &source, nullptr));
    RunResult original = {0};
    assert(run_single_comparison(&source, 71, &original));

    for (size_t kind = 0; kind < MODEL_COUNT; ++kind) {
        const SelectedModel *model = &original.models[kind];
        assert(model->validation.count == 8);
        assert(model->test.count == 8);
        assert(isfinite(model->validation.balanced_accuracy));
        assert(isfinite(model->test.accuracy));
        assert(isfinite(model->test.balanced_accuracy));
        assert(isfinite(model->test.f1));
    }
    assert(original.models[MODEL_LOGISTIC_REGRESSION].config_value == 0.03 ||
           original.models[MODEL_LOGISTIC_REGRESSION].config_value == 0.10 ||
           original.models[MODEL_LOGISTIC_REGRESSION].config_value == 0.30);
    assert(original.models[MODEL_KNN].parameters.knn.neighbor_count == 1 ||
           original.models[MODEL_KNN].parameters.knn.neighbor_count == 3 ||
           original.models[MODEL_KNN].parameters.knn.neighbor_count == 5);

    Dataset changed_test = source;
    for (size_t index = 0; index < changed_test.count; ++index) {
        if (changed_test.samples[index].split == SPLIT_TEST) {
            changed_test.samples[index].label = 1 - changed_test.samples[index].label;
        }
    }
    RunResult after_test_change = {0};
    assert(run_single_comparison(&changed_test, 71, &after_test_change));
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        assert(original.standardizer.mean[feature] == after_test_change.standardizer.mean[feature]);
        assert(original.standardizer.scale[feature] ==
               after_test_change.standardizer.scale[feature]);
    }
    for (size_t kind = 0; kind < MODEL_COUNT; ++kind) {
        const SelectedModel *before = &original.models[kind];
        const SelectedModel *after = &after_test_change.models[kind];
        assert(before->kind == after->kind);
        assert(before->config_value == after->config_value);
        assert(before->validation.balanced_accuracy == after->validation.balanced_accuracy);
        if (kind == MODEL_LOGISTIC_REGRESSION) {
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                assert(before->parameters.logistic.weights[feature] ==
                       after->parameters.logistic.weights[feature]);
            }
            assert(before->parameters.logistic.bias == after->parameters.logistic.bias);
        } else if (kind == MODEL_MAJORITY_BASELINE) {
            assert(before->parameters.majority_label == after->parameters.majority_label);
        } else if (kind == MODEL_KNN) {
            assert(before->parameters.knn.neighbor_count == after->parameters.knn.neighbor_count);
        } else {
            assert(before->parameters.stump.has_split == after->parameters.stump.has_split);
            assert(before->parameters.stump.feature == after->parameters.stump.feature);
            assert(before->parameters.stump.threshold == after->parameters.stump.threshold);
            assert(before->parameters.stump.left_label == after->parameters.stump.left_label);
            assert(before->parameters.stump.right_label == after->parameters.stump.right_label);
        }
    }

    RunResult runs[RUN_COUNT] = {{0}};
    for (size_t run = 0; run < RUN_COUNT; ++run) {
        assert(run_single_comparison(&source, RUN_SEEDS[run], &runs[run]));
    }
    double mean = 0.0;
    double deviation = 0.0;
    assert(summarize_test_metric(runs, MODEL_KNN, 0, &mean, &deviation));
    assert(isfinite(mean) && mean >= 0.0 && mean <= 1.0);
    assert(isfinite(deviation) && deviation >= 0.0 && deviation <= 0.5);
}

/**
 * @brief 执行公平比较的划分、指标、树桩与数据隔离断言。
 * @return 所有断言通过时返回 0。
 */
int main(void)
{
    test_metrics_by_hand();
    test_repeated_split_is_reproducible_and_shared();
    test_standardizer_uses_training_rows_only();
    test_stump_split_by_hand();
    test_paired_bootstrap();
    test_comparison_and_test_isolation();
    puts("公平模型比较测试通过");
    return EXIT_SUCCESS;
}
