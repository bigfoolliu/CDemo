/**
 * @file
 * @brief 朴素贝叶斯的手算概率、分词、先验、未登录词和失败保留测试。
 */
#define main naive_bayes_demo_main
#include "../src/ml/15_naive_bayes.c"
#undef main
#include <assert.h>

/**
 * @brief 比较小型手算语料的词频、平滑概率和重复词贡献。
 */
static void test_hand_calculation(void)
{
    const Document train[] = {{"HAM ham!", 0}, {"spam", 1}};
    NaiveBayes model = {0};
    assert(nb_train(&model, train, 2));
    assert(model.vocabulary.size == 2 && model.tokens[0] == 2 && model.tokens[1] == 1);
    uint64_t count = 0;
    assert(word_map_get(&model.words[0], "ham", 3, &count) && count == 2);

    Prediction result = {0};
    assert(nb_predict(&model, "HaM", &result));
    assert(result.label == 0);
    assert(fabs(result.log_scores[0] - log(0.5 * 3.0 / 4.0)) < 1e-12);
    assert(fabs(result.log_scores[1] - log(0.5 * 1.0 / 3.0)) < 1e-12);
    assert(nb_predict(&model, "ham ham", &result));
    assert(fabs(result.log_scores[0] - (log(0.5) + 2.0 * log(3.0 / 4.0))) < 1e-12);
    nb_destroy(&model);
}

/**
 * @brief 验证空输入和未登录词只剩先验，标点与数字执行相同分词规则。
 */
static void test_prior_and_unknown_words(void)
{
    const Document train[] = {{"hello", 0}, {"HELLO", 0}, {"free", 1}};
    NaiveBayes model = {0};
    assert(nb_train(&model, train, 3));
    Prediction result = {0};
    assert(nb_predict(&model, "", &result));
    assert(result.label == 0);
    assert(fabs(result.log_scores[0] - log(2.0 / 3.0)) < 1e-12);
    assert(nb_predict(&model, "xyz!?", &result));
    assert(fabs(result.log_scores[0] - log(2.0 / 3.0)) < 1e-12);
    assert(fabs(result.log_scores[1] - log(1.0 / 3.0)) < 1e-12);
    assert(nb_predict(&model, "FREE123free", &result));
    assert(result.label == 1);
    uint64_t count = 0;
    assert(!word_map_get(&model.vocabulary, "xyz", 3, &count));
    nb_destroy(&model);
}

/**
 * @brief 验证无效训练不覆盖旧模型，预测失败也不改写调用者输出。
 */
static void test_failure_keeps_state(void)
{
    const Document train[] = {{"good", 0}, {"bad", 1}};
    const Document invalid[] = {{"other", 0}, {nullptr, 1}};
    const Document one_class[] = {{"only", 0}};
    const Document no_words[] = {{"! 123", 0}, {"?", 1}};
    const Document invalid_label[] = {{"other", 0}, {"wrong", 2}};
    NaiveBayes model = {0};
    assert(nb_train(&model, train, 2));
    assert(!nb_train(&model, invalid, 2));
    assert(!nb_train(&model, one_class, 1));
    assert(!nb_train(&model, no_words, 2));
    assert(!nb_train(&model, invalid_label, 2));
    assert(!nb_train(&model, nullptr, 0));
    assert(model.vocabulary.size == 2 && model.documents[0] == 1 && model.documents[1] == 1);

    Prediction result = {.label = 42, .log_scores = {11.0, 22.0}};
    assert(!nb_predict(&model, nullptr, &result));
    assert(result.label == 42 && result.log_scores[0] == 11.0 && result.log_scores[1] == 22.0);
    assert(nb_predict(&model, "bad", &result) && result.label == 1);
    nb_destroy(&model);
    assert(!nb_predict(&model, "bad", &result));
}

/**
 * @brief 运行概率和错误路径断言。
 * @return 全部通过为 0；断言失败会终止程序。
 */
int main(void)
{
    test_hand_calculation();
    test_prior_and_unknown_words();
    test_failure_keeps_state();
    puts("朴素贝叶斯测试通过");
    return 0;
}
