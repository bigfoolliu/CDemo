/**
 * @file
 * @brief 多项式朴素贝叶斯文本分类：词频、平滑、对数概率与独立测试。
 */
#include "../common/word_map.h"
#include <math.h>
#include <stdio.h>

enum {
    CLASS_COUNT = 2
};

typedef struct {
    const char *text;
    int label;
} Document;

typedef struct {
    WordMap words[CLASS_COUNT];
    WordMap vocabulary;
    uint64_t documents[CLASS_COUNT];
    uint64_t tokens[CLASS_COUNT];
} NaiveBayes;

typedef struct {
    NaiveBayes *model;
    int label;
} TrainContext;

typedef struct {
    const NaiveBayes *model;
    double scores[CLASS_COUNT];
} ScoreContext;

typedef struct {
    int label;
    double log_scores[CLASS_COUNT];
} Prediction;

/**
 * @brief 释放模型拥有的词表和类别计数；可用于零初始化对象。
 * @param model 可写模型；为空时无操作，销毁后可再次训练。
 */
static void nb_destroy(NaiveBayes *model)
{
    if (model == nullptr) {
        return;
    }
    for (int label = 0; label < CLASS_COUNT; ++label) {
        word_map_destroy(&model->words[label]);
    }
    word_map_destroy(&model->vocabulary);
    *model = (NaiveBayes){0};
}

/**
 * @brief 扫描 NUL 结尾文本，将 ASCII 字母单词转成小写后交给回调。
 * @param text 只读文本；其他字节都是分隔符，不执行 Unicode 分词。
 * @param visit 每个单词调用一次；回调不得保存临时单词指针。
 * @param context 传给 visit 的借用状态。
 * @return 全部处理成功为 true；参数、分配或回调失败为 false。
 */
static bool scan_words(const char *text, bool (*visit)(const char *, size_t, void *),
                       void *context)
{
    if (text == nullptr || visit == nullptr) {
        return false;
    }
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor != '\0') {
        while (*cursor != '\0' && !(*cursor >= 'A' && *cursor <= 'Z') &&
               !(*cursor >= 'a' && *cursor <= 'z')) {
            ++cursor;
        }
        const unsigned char *start = cursor;
        while ((*cursor >= 'A' && *cursor <= 'Z') || (*cursor >= 'a' && *cursor <= 'z')) {
            ++cursor;
        }
        size_t length = (size_t)(cursor - start);
        if (length == 0) {
            continue;
        }
        if (length == SIZE_MAX) {
            return false;
        }
        char *word = malloc(length + 1);
        if (word == nullptr) {
            return false;
        }
        for (size_t i = 0; i < length; ++i) {
            unsigned char byte = start[i];
            word[i] = (char)(byte >= 'A' && byte <= 'Z' ? byte - 'A' + 'a' : byte);
        }
        word[length] = '\0';
        bool ok = visit(word, length, context);
        free(word);
        if (!ok) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 将一个训练词加入所属类别和全局词表。
 * @param word 借用的小写 ASCII 单词，至少 length 字节。
 * @param length 单词字节数，大于零。
 * @param context 可写 TrainContext；失败时临时模型由调用者丢弃。
 * @return 计数成功为 true；溢出或分配失败为 false。
 */
static bool train_word(const char *word, size_t length, void *context)
{
    TrainContext *training = context;
    NaiveBayes *model = training->model;
    int label = training->label;
    if (model->tokens[label] == UINT64_MAX ||
        !word_map_increment(&model->words[label], word, length) ||
        !word_map_increment(&model->vocabulary, word, length)) {
        return false;
    }
    ++model->tokens[label];
    return true;
}

/**
 * @brief 从带标签短文本重新训练二分类模型，成功时替换旧模型。
 * @param model 已零初始化或已训练的可写模型，拥有内部词表。
 * @param examples 至少 count 条只读文档；标签只能为 0 或 1，文本非空指针。
 * @param count 文档数，大于零，两个类别都至少出现一次且总词表非空。
 * @return 成功为 true；无效数据或内存失败为 false，旧模型保持不变。
 */
static bool nb_train(NaiveBayes *model, const Document *examples, size_t count)
{
    if (model == nullptr || examples == nullptr || count == 0) {
        return false;
    }
    NaiveBayes next = {0};
    for (int label = 0; label < CLASS_COUNT; ++label) {
        if (!word_map_init(&next.words[label], nullptr, nullptr)) {
            nb_destroy(&next);
            return false;
        }
    }
    if (!word_map_init(&next.vocabulary, nullptr, nullptr)) {
        nb_destroy(&next);
        return false;
    }

    for (size_t i = 0; i < count; ++i) {
        int label = examples[i].label;
        if (label < 0 || label >= CLASS_COUNT || examples[i].text == nullptr ||
            next.documents[label] == UINT64_MAX) {
            nb_destroy(&next);
            return false;
        }
        TrainContext context = {&next, label};
        if (!scan_words(examples[i].text, train_word, &context)) {
            nb_destroy(&next);
            return false;
        }
        ++next.documents[label];
    }
    if (next.documents[0] == 0 || next.documents[1] == 0 || next.vocabulary.size == 0) {
        nb_destroy(&next);
        return false;
    }

    nb_destroy(model);
    *model = next;
    return true;
}

/**
 * @brief 把已知词的两类对数条件概率加到当前分数，未登录词忽略。
 * @param word 借用的小写 ASCII 单词，至少 length 字节。
 * @param length 单词字节数，大于零。
 * @param context 可写 ScoreContext，模型为只读且已训练。
 * @return 分数保持有限时为 true；数值失败为 false。
 */
static bool score_word(const char *word, size_t length, void *context)
{
    ScoreContext *scoring = context;
    uint64_t presence = 0;
    if (!word_map_get(&scoring->model->vocabulary, word, length, &presence)) {
        return true;
    }
    for (int label = 0; label < CLASS_COUNT; ++label) {
        uint64_t frequency = 0;
        (void)word_map_get(&scoring->model->words[label], word, length, &frequency);
        double numerator = (double)frequency + 1.0;
        double denominator = (double)scoring->model->tokens[label] +
                             (double)scoring->model->vocabulary.size;
        scoring->scores[label] += log(numerator / denominator);
        if (!isfinite(scoring->scores[label])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 使用多项式朴素贝叶斯预测文本类别及未归一化对数分数。
 * @param model 已训练的只读模型，两个类别都有训练文档。
 * @param text NUL 结尾只读文本；空文本及仅含未登录词的文本按先验分类。
 * @param output 可写结果；分数平局时类别为 0，失败时不修改。
 * @return 预测成功为 true；参数、内存或数值失败为 false。
 */
static bool nb_predict(const NaiveBayes *model, const char *text, Prediction *output)
{
    if (model == nullptr || text == nullptr || output == nullptr ||
        model->documents[0] == 0 || model->documents[1] == 0 || model->vocabulary.size == 0) {
        return false;
    }
    double total = (double)model->documents[0] + (double)model->documents[1];
    ScoreContext scoring = {.model = model};
    for (int label = 0; label < CLASS_COUNT; ++label) {
        scoring.scores[label] = log((double)model->documents[label] / total);
    }
    if (!scan_words(text, score_word, &scoring)) {
        return false;
    }
    Prediction result = {.label = scoring.scores[1] > scoring.scores[0] ? 1 : 0,
                         .log_scores = {scoring.scores[0], scoring.scores[1]}};
    *output = result;
    return true;
}

/**
 * @brief 训练固定英文短句并在独立句子上评估，展示基线与每条预测。
 * @return 成功且固定测试全部分类正确为 0；训练、预测或输出失败为 1。
 */
int main(void)
{
    static const Document train[] = {
        {"team meeting tomorrow", 0}, {"project update meeting", 0},
        {"family dinner tonight", 0}, {"please review project notes", 0},
        {"lunch with family today", 0}, {"meeting notes for team", 0},
        {"win cash prize", 1}, {"claim your free prize", 1},
        {"free cash offer", 1}, {"win bonus cash now", 1},
        {"claim bonus offer", 1}, {"your prize is free", 1}
    };
    static const Document test[] = {
        {"review team project", 0}, {"family lunch tonight", 0},
        {"meeting schedule today", 0}, {"claim free cash", 1},
        {"win bonus prize", 1}, {"exclusive prize offer", 1}
    };
    NaiveBayes model = {0};
    if (!nb_train(&model, train, sizeof(train) / sizeof(train[0]))) {
        fputs("朴素贝叶斯训练失败\n", stderr);
        return 1;
    }

    size_t correct = 0;
    size_t baseline_correct = 0;
    printf("训练文档: 类别0=%llu, 类别1=%llu, 词表=%zu\n",
           (unsigned long long)model.documents[0], (unsigned long long)model.documents[1],
           model.vocabulary.size);
    puts("测试句子 | 实际 | 预测 | log P(0,文本) | log P(1,文本)");
    for (size_t i = 0; i < sizeof(test) / sizeof(test[0]); ++i) {
        Prediction prediction;
        if (!nb_predict(&model, test[i].text, &prediction)) {
            fputs("朴素贝叶斯预测失败\n", stderr);
            nb_destroy(&model);
            return 1;
        }
        correct += prediction.label == test[i].label;
        baseline_correct += test[i].label == 0;
        printf("%s | %d | %d | %.3f | %.3f\n", test[i].text, test[i].label,
               prediction.label, prediction.log_scores[0], prediction.log_scores[1]);
    }
    printf("独立测试: %zu/%zu；始终猜类别0的基线: %zu/%zu\n", correct,
           sizeof(test) / sizeof(test[0]), baseline_correct, sizeof(test) / sizeof(test[0]));
    nb_destroy(&model);
    return correct == sizeof(test) / sizeof(test[0]) && !ferror(stdout) ? 0 : 1;
}
