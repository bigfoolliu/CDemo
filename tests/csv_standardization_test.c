/** @file @brief 验证 CSV 解析、仅训练集标准化和独立测试评估。 */
#define main csv_standardization_demo_main
#include "../src/ml/23_csv_standardization.c"
#undef main
#include <assert.h>

/**
 * @brief 将内存中的 CSV 文本写入临时流，避免测试触碰用户文件。
 * @param contents 以空字符结尾且包含 CSV 行的文本。
 * @return 写入并回卷成功的临时流；失败时断言终止测试。
 */
static FILE *make_csv_stream(const char *contents)
{
    FILE *stream = tmpfile();
    assert(stream != nullptr);
    assert(fputs(contents, stream) >= 0);
    rewind(stream);
    return stream;
}

/**
 * @brief 用人为极端的验证/测试值证明标准化参数仅来自训练行。
 */
static void test_training_only_standardization(void)
{
    const char *contents =
        "study_hours,practice_questions,passed,split\n"
        "2,10,0,train\n"
        "4,14,1,train\n"
        "100,1000,1,validation\n"
        "200,2000,0,test\n";
    FILE *stream = make_csv_stream(contents);
    Dataset dataset = {0};
    size_t error_line = 99;
    assert(dataset_load_stream(stream, &dataset, &error_line));
    assert(error_line == 0);
    assert(dataset.count == 4);
    assert(dataset.split_counts[SPLIT_TRAIN] == 2);
    assert(dataset.split_counts[SPLIT_VALIDATION] == 1);
    assert(dataset.split_counts[SPLIT_TEST] == 1);
    assert(fclose(stream) == 0);

    Standardizer standardizer = {0};
    assert(standardizer_fit(&dataset, &standardizer));
    assert(fabs(standardizer.mean[0] - 3.0) < 1e-12);
    assert(fabs(standardizer.scale[0] - 1.0) < 1e-12);
    assert(fabs(standardizer.mean[1] - 12.0) < 1e-12);
    assert(fabs(standardizer.scale[1] - 2.0) < 1e-12);

    double transformed[FEATURE_COUNT] = {0};
    assert(standardize_sample(&standardizer, &dataset.samples[2], transformed));
    assert(fabs(transformed[0] - 97.0) < 1e-12);
    assert(fabs(transformed[1] - 494.0) < 1e-12);
}

/**
 * @brief 验证畸形 CSV 和零方差特征失败时不覆盖已有输出。
 */
static void test_invalid_input_preserves_outputs(void)
{
    FILE *malformed = make_csv_stream(
        "study_hours,practice_questions,passed,split\n"
        "nan,10,0,train\n");
    Dataset unchanged_dataset = {0};
    unchanged_dataset.count = 77;
    unchanged_dataset.samples[0].label = 9;
    size_t error_line = 0;
    assert(!dataset_load_stream(malformed, &unchanged_dataset, &error_line));
    assert(error_line == 2);
    assert(unchanged_dataset.count == 77);
    assert(unchanged_dataset.samples[0].label == 9);
    assert(fclose(malformed) == 0);

    FILE *constant_feature = make_csv_stream(
        "study_hours,practice_questions,passed,split\n"
        "4,10,0,train\n"
        "4,20,1,train\n"
        "4,30,0,validation\n"
        "5,40,1,test\n");
    Dataset dataset = {0};
    assert(dataset_load_stream(constant_feature, &dataset, nullptr));
    assert(fclose(constant_feature) == 0);

    Standardizer unchanged_standardizer = {{31.0, 32.0}, {41.0, 42.0}};
    assert(!standardizer_fit(&dataset, &unchanged_standardizer));
    assert(unchanged_standardizer.mean[0] == 31.0);
    assert(unchanged_standardizer.mean[1] == 32.0);
    assert(unchanged_standardizer.scale[0] == 41.0);
    assert(unchanged_standardizer.scale[1] == 42.0);
}

/**
 * @brief 验证仓库 CSV 可训练，并且最终测试统计只覆盖测试行。
 */
static void test_repository_dataset_evaluation(void)
{
    Dataset dataset = {0};
    assert(dataset_load_csv(DEFAULT_CSV_PATH, &dataset, nullptr));
    assert(dataset.count == 40);
    assert(dataset.split_counts[SPLIT_TRAIN] == 24);
    assert(dataset.split_counts[SPLIT_VALIDATION] == 8);
    assert(dataset.split_counts[SPLIT_TEST] == 8);

    Standardizer standardizer = {0};
    assert(standardizer_fit(&dataset, &standardizer));
    assert(fabs(standardizer.mean[0] - 5.275) < 1e-12);
    assert(fabs(standardizer.mean[1] - 59.4583333333333) < 1e-10);

    LogisticModel model = {0};
    TrainingSummary summary = {0};
    assert(model_train(&dataset, &standardizer, &model, &summary));
    assert(summary.best_epoch > 0 && summary.epochs_run >= summary.best_epoch);

    Metrics test_metrics = {0};
    assert(evaluate_split(&model, &standardizer, &dataset, SPLIT_TEST, &test_metrics));
    assert(test_metrics.count == 8);
    assert(test_metrics.correct >= 7);
    assert(isfinite(test_metrics.cross_entropy));
    assert(test_metrics.true_negative + test_metrics.false_positive == 4);
    assert(test_metrics.false_negative + test_metrics.true_positive == 4);
}

/**
 * @brief 运行 CSV、标准化失败安全和独立评估断言。
 * @return 全部通过返回 EXIT_SUCCESS；断言失败会终止程序。
 */
int main(void)
{
    test_training_only_standardization();
    test_invalid_input_preserves_outputs();
    test_repository_dataset_evaluation();
    puts("CSV 标准化测试通过");
    return EXIT_SUCCESS;
}
