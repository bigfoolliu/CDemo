/**
 * @file
 * @brief 固定小词表的一元/二元语言模型、留出评估与有界采样。
 * 语料先按原始文档划分；BOS 只提供上下文，EOS 参与训练和计分。
 * 词表外 token 会映射到 UNK。
 */
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOCAB_SIZE 12
#define MAX_VOCAB_SIZE 16
#define MAX_DOCUMENTS 8
#define MAX_SEQUENCE_LENGTH 16
#define MAX_WORD_LENGTH 48
#define MAX_GENERATED_TOKENS 12
#define BOS_TOKEN 1u
#define EOS_TOKEN 2u
#define ALPHA 1.0

enum {
    TOKEN_UNK,
    TOKEN_BOS,
    TOKEN_EOS,
    TOKEN_PERIOD,
    TOKEN_C,
    TOKEN_I,
    TOKEN_LANGUAGE,
    TOKEN_LEARN,
    TOKEN_LIKE,
    TOKEN_PYTHON,
    TOKEN_STUDY,
    TOKEN_YOU
};

typedef enum {
    MODEL_UNIGRAM,
    MODEL_BIGRAM
} ModelKind;

typedef struct {
    size_t length;
    unsigned token[MAX_SEQUENCE_LENGTH];
} Document;

typedef struct {
    size_t document_count;
    Document documents[MAX_DOCUMENTS];
} Dataset;

typedef struct {
    ModelKind kind;
    size_t vocabulary_size;
    size_t unigram_count[MAX_VOCAB_SIZE];
    size_t bigram_count[MAX_VOCAB_SIZE][MAX_VOCAB_SIZE];
    size_t context_count[MAX_VOCAB_SIZE];
    size_t token_count;
} Model;

typedef struct {
    double nll;
    double perplexity;
    size_t token_count;
} Metrics;

static const char *const vocabulary[VOCAB_SIZE] = {
    "<unk>", "<bos>", "<eos>", ".", "c", "i",
    "language", "learn", "like", "python", "study", "you"
};

/* 每条字符串是一份原始文档；先划分文档，再编码成 token 序列。 */
static const char *const training_text[] = {
    "i like c .",
    "i like c .",
    "you like c .",
    "i learn c .",
    "you learn python .",
    "i like language ."
};

static const char *const testing_text[] = {
    "you like language .",
    "you study rust ."
};

/**
 * @brief 将 ASCII 单词映射到固定词表，未收录词映射到 UNK。
 * @param word 以零结尾的单词；大小写按 ASCII 忽略。
 * @return 词表编号；输入不在词表时返回 TOKEN_UNK。
 */
static unsigned vocabulary_lookup(const char *word)
{
    char normalized[MAX_WORD_LENGTH + 1];
    size_t length = strlen(word);
    if (length > MAX_WORD_LENGTH) {
        return TOKEN_UNK;
    }

    for (size_t i = 0; i < length; ++i) {
        unsigned char byte = (unsigned char)word[i];
        normalized[i] = byte >= 'A' && byte <= 'Z' ? (char)(byte + ('a' - 'A')) : (char)byte;
    }
    normalized[length] = '\0';

    for (size_t token = 3; token < VOCAB_SIZE; ++token) {
        if (strcmp(normalized, vocabulary[token]) == 0) {
            return (unsigned)token;
        }
    }
    return TOKEN_UNK;
}

/**
 * @brief 将空白分隔文本编码为 BOS、词项和 EOS。
 * @param output 有效可写文档；失败时原内容保持不变。
 * @param text 非空、以零结尾的 UTF-8/ASCII 字节串；仅 ASCII 空白用于分词。
 * @return 编码成功为 true；空指针或超出固定容量时返回 false。
 */
static bool document_encode(Document *output, const char *text)
{
    if (output == nullptr || text == nullptr) {
        return false;
    }

    Document encoded = {.length = 1, .token = {BOS_TOKEN}};
    const unsigned char *cursor = (const unsigned char *)text;
    while (*cursor != '\0') {
        while (*cursor == ' ' || *cursor == '\t' || *cursor == '\n' || *cursor == '\r') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }

        char word[MAX_WORD_LENGTH + 1];
        size_t length = 0;
        bool too_long = false;
        while (*cursor != '\0' && *cursor != ' ' && *cursor != '\t' &&
               *cursor != '\n' && *cursor != '\r') {
            if (length < MAX_WORD_LENGTH) {
                word[length] = (char)*cursor;
            } else {
                too_long = true;
            }
            ++length;
            ++cursor;
        }
        unsigned token = TOKEN_UNK;
        if (!too_long) {
            word[length] = '\0';
            token = vocabulary_lookup(word);
        }
        if (encoded.length + 1 >= MAX_SEQUENCE_LENGTH) {
            return false;
        }
        encoded.token[encoded.length++] = token;
    }

    if (encoded.length >= MAX_SEQUENCE_LENGTH) {
        return false;
    }
    encoded.token[encoded.length++] = EOS_TOKEN;
    *output = encoded;
    return true;
}

/**
 * @brief 按原始文档数组构造固定容量数据集。
 * @param output 有效可写数据集；失败时不修改。
 * @param texts 文档字符串数组，不转移所有权。
 * @param count 文档数量，须不超过 MAX_DOCUMENTS。
 * @return 所有文档均编码成功时为 true。
 */
static bool dataset_encode(Dataset *output, const char *const *texts, size_t count)
{
    if (output == nullptr || texts == nullptr || count == 0 || count > MAX_DOCUMENTS) {
        return false;
    }

    Dataset encoded = {.document_count = count};
    for (size_t i = 0; i < count; ++i) {
        if (!document_encode(&encoded.documents[i], texts[i])) {
            return false;
        }
    }
    *output = encoded;
    return true;
}

/**
 * @brief 在训练文档上统计一元 token 次数和二元上下文转移次数。
 * @param model 有效可写模型；统计成功后覆盖原内容。
 * @param dataset 只包含训练文档的数据集，序列必须以 BOS 开始、EOS 结束。
 * @param kind 要建立的一元或二元模型。
 * @param vocabulary_size 固定词表大小，范围为 3..MAX_VOCAB_SIZE。
 * @return 所有 token 合法且计数不溢出时为 true；失败时模型不变。
 */
static bool model_train(Model *model, const Dataset *dataset, ModelKind kind,
                        size_t vocabulary_size)
{
    if (model == nullptr || dataset == nullptr || dataset->document_count == 0 ||
        dataset->document_count > MAX_DOCUMENTS || vocabulary_size < 3 ||
        vocabulary_size > MAX_VOCAB_SIZE || (kind != MODEL_UNIGRAM && kind != MODEL_BIGRAM)) {
        return false;
    }

    Model trained = {.kind = kind, .vocabulary_size = vocabulary_size};
    for (size_t document_index = 0; document_index < dataset->document_count; ++document_index) {
        const Document *document = &dataset->documents[document_index];
        if (document->length < 2 || document->length > MAX_SEQUENCE_LENGTH ||
            document->token[0] != BOS_TOKEN ||
            document->token[document->length - 1] != EOS_TOKEN) {
            return false;
        }
        for (size_t position = 1; position < document->length; ++position) {
            unsigned history = document->token[position - 1];
            unsigned target = document->token[position];
            if (history >= vocabulary_size || target >= vocabulary_size || target == BOS_TOKEN ||
                trained.unigram_count[target] == SIZE_MAX || trained.token_count == SIZE_MAX ||
                trained.context_count[history] == SIZE_MAX ||
                trained.bigram_count[history][target] == SIZE_MAX) {
                return false;
            }
            ++trained.unigram_count[target];
            ++trained.token_count;
            ++trained.context_count[history];
            ++trained.bigram_count[history][target];
        }
    }
    *model = trained;
    return true;
}

/**
 * @brief 計算指定上下文的平滑下一 token 分布。
 * @param model 有效只读模型。
 * @param history 二元模型的上下文编号；一元模型忽略此值。
 * @param probabilities 至少 vocabulary_size 个可写 double，不与模型重叠。
 * @return 分布有效时返回 true；BOS 的输出概率固定为 0。
 */
static bool model_probabilities(const Model *model, unsigned history, double *probabilities)
{
    if (model == nullptr || probabilities == nullptr || model->vocabulary_size < 3 ||
        model->vocabulary_size > MAX_VOCAB_SIZE || history >= model->vocabulary_size ||
        (model->kind != MODEL_UNIGRAM && model->kind != MODEL_BIGRAM)) {
        return false;
    }

    size_t outcome_count = model->vocabulary_size - 1;
    double observed = model->kind == MODEL_UNIGRAM
                         ? (double)model->token_count
                         : (double)model->context_count[history];
    double denominator = observed + ALPHA * (double)outcome_count;
    if (!(denominator > 0.0) || !isfinite(denominator)) {
        return false;
    }

    for (size_t token = 0; token < model->vocabulary_size; ++token) {
        if (token == BOS_TOKEN) {
            probabilities[token] = 0.0;
            continue;
        }
        size_t count = model->kind == MODEL_UNIGRAM
                           ? model->unigram_count[token]
                           : model->bigram_count[history][token];
        probabilities[token] = ((double)count + ALPHA) / denominator;
        if (!(probabilities[token] > 0.0) || !isfinite(probabilities[token])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 在留出文档上计算 token 平均 NLL 与困惑度。
 * @param model 有效只读模型。
 * @param dataset 只用于评估的数据集，所有目标 token 均须在模型词表内。
 * @param metrics 有效可写结果；失败时不修改。
 * @return 至少一个预测目标且计算有限时为 true。
 */
static bool model_evaluate(const Model *model, const Dataset *dataset, Metrics *metrics)
{
    if (model == nullptr || dataset == nullptr || metrics == nullptr ||
        dataset->document_count == 0 || dataset->document_count > MAX_DOCUMENTS) {
        return false;
    }

    Metrics result = {0};
    double probabilities[MAX_VOCAB_SIZE];
    for (size_t document_index = 0; document_index < dataset->document_count; ++document_index) {
        const Document *document = &dataset->documents[document_index];
        if (document->length < 2 || document->length > MAX_SEQUENCE_LENGTH ||
            document->token[0] != BOS_TOKEN ||
            document->token[document->length - 1] != EOS_TOKEN) {
            return false;
        }
        for (size_t position = 1; position < document->length; ++position) {
            unsigned history = document->token[position - 1];
            unsigned target = document->token[position];
            if (target == BOS_TOKEN || target >= model->vocabulary_size ||
                !model_probabilities(model, history, probabilities) ||
                !(probabilities[target] > 0.0)) {
                return false;
            }
            result.nll -= log(probabilities[target]);
            ++result.token_count;
        }
    }

    if (result.token_count == 0 || !isfinite(result.nll)) {
        return false;
    }
    result.nll /= (double)result.token_count;
    result.perplexity = exp(result.nll);
    if (!isfinite(result.perplexity)) {
        return false;
    }
    *metrics = result;
    return true;
}

/**
 * @brief 用固定 LCG 伪随机数从模型生成有长度上限的 token 序列。
 * @param model 有效只读模型。
 * @param seed 可复现随机种子；不用于安全用途。
 * @param output 至少 capacity 个可写 token 编号。
 * @param capacity 最大输出长度，不能超过 MAX_GENERATED_TOKENS。
 * @param output_count 成功时写入生成长度；EOS 也计入长度。
 * @return 成功时为 true；遇到非法参数或概率分布错误时为 false。
 */
static bool model_sample(const Model *model, uint32_t seed, unsigned *output, size_t capacity,
                         size_t *output_count)
{
    if (model == nullptr || output == nullptr || output_count == nullptr || capacity == 0 ||
        capacity > MAX_GENERATED_TOKENS) {
        return false;
    }

    unsigned generated[MAX_GENERATED_TOKENS];
    size_t generated_count = 0;
    unsigned history = BOS_TOKEN;
    double probabilities[MAX_VOCAB_SIZE];
    while (generated_count < capacity) {
        if (!model_probabilities(model, history, probabilities)) {
            return false;
        }
        seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
        double choice = (double)(seed >> 8) / 16777216.0;
        double cumulative = 0.0;
        unsigned selected = EOS_TOKEN;
        for (size_t token = 0; token < model->vocabulary_size; ++token) {
            cumulative += probabilities[token];
            if (token != BOS_TOKEN && choice < cumulative) {
                selected = (unsigned)token;
                break;
            }
        }
        generated[generated_count++] = selected;
        if (selected == EOS_TOKEN) {
            break;
        }
        history = selected;
    }

    memcpy(output, generated, generated_count * sizeof(generated[0]));
    *output_count = generated_count;
    return true;
}

/**
 * @brief 构造手算夹具：唯一训练句子为 a b，词表含 UNK/BOS/EOS/a/b。
 * @param unigram 有效可写一元模型。
 * @param bigram 有效可写二元模型。
 * @return 两个模型都训练成功时为 true。
 */
static bool build_hand_models(Model *unigram, Model *bigram)
{
    Dataset dataset = {.document_count = 1};
    Document *document = &dataset.documents[0];
    document->length = 4;
    document->token[0] = BOS_TOKEN;
    document->token[1] = 3;
    document->token[2] = 4;
    document->token[3] = EOS_TOKEN;
    return model_train(unigram, &dataset, MODEL_UNIGRAM, 5) &&
           model_train(bigram, &dataset, MODEL_BIGRAM, 5);
}

/**
 * @brief 输出小型语料的计数、条件概率、句子概率和 PPL 手算过程。
 * @param stream 可写输出流，由调用方持有。
 * @param csv 为 true 时输出便于 C/Python 逐字段比较的数字行。
 * @return 输出成功且两个手算模型均建立时为 true。
 */
static bool run_hand(FILE *stream, bool csv)
{
    Model unigram;
    Model bigram;
    if (!build_hand_models(&unigram, &bigram)) {
        return false;
    }
    double uni_probability[MAX_VOCAB_SIZE];
    double bos_probability[MAX_VOCAB_SIZE];
    double a_probability[MAX_VOCAB_SIZE];
    double b_probability[MAX_VOCAB_SIZE];
    if (!model_probabilities(&unigram, BOS_TOKEN, uni_probability) ||
        !model_probabilities(&bigram, BOS_TOKEN, bos_probability) ||
        !model_probabilities(&bigram, 3, a_probability) ||
        !model_probabilities(&bigram, 4, b_probability)) {
        return false;
    }

    double uni_sentence = uni_probability[3] * uni_probability[4] * uni_probability[EOS_TOKEN];
    double bi_sentence = bos_probability[3] * a_probability[4] * b_probability[EOS_TOKEN];
    double uni_nll = -log(uni_sentence) / 3.0;
    double bi_nll = -log(bi_sentence) / 3.0;
    if (csv) {
        fprintf(stream, "hand,unigram,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                uni_probability[3], uni_probability[4], uni_probability[EOS_TOKEN],
                uni_probability[TOKEN_UNK], uni_sentence, uni_nll, exp(uni_nll));
        fprintf(stream, "hand,bigram,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",
                bos_probability[3], a_probability[4], b_probability[EOS_TOKEN],
                bi_sentence, bi_nll, exp(bi_nll));
    } else {
        fprintf(stream, "手算语料：BOS a b EOS；可预测词表为 UNK、EOS、a、b，"
                        "V_next=4，alpha=1\n");
        fprintf(stream, "一元计数：a=1、b=1、EOS=1、UNK=0、N=3；"
                        "平滑概率分别为 2/7、2/7、2/7、1/7\n");
        fprintf(stream, "一元句子概率：P(a b EOS)=(2/7)^3=%.9f，"
                        "token NLL=%.9f，PPL=%.9f\n",
                uni_sentence, uni_nll, exp(uni_nll));
        fprintf(stream, "二元转移：BOS→a=2/5，a→b=2/5，b→EOS=2/5\n");
        fprintf(stream, "二元句子概率：P(a b EOS|BOS)=(2/5)^3=%.9f，"
                        "token NLL=%.9f，PPL=%.9f\n",
                bi_sentence, bi_nll, exp(bi_nll));
    }
    return !ferror(stream);
}

/**
 * @brief 验证计数、特殊 token、平滑概率、评估、采样和失败边界。
 * @return 所有断言通过时为 true。
 */
static bool run_tests(void)
{
    Dataset training;
    Dataset testing;
    assert(dataset_encode(&training, training_text,
                          sizeof(training_text) / sizeof(training_text[0])));
    assert(dataset_encode(&testing, testing_text, sizeof(testing_text) / sizeof(testing_text[0])));
    assert(training.document_count == 6 && testing.document_count == 2);
    assert(training.documents[0].token[0] == BOS_TOKEN);
    assert(training.documents[0].token[training.documents[0].length - 1] == EOS_TOKEN);
    assert(testing.documents[1].token[3] == TOKEN_UNK);

    Model unigram;
    Model bigram;
    assert(model_train(&unigram, &training, MODEL_UNIGRAM, VOCAB_SIZE));
    assert(model_train(&bigram, &training, MODEL_BIGRAM, VOCAB_SIZE));
    Model unchanged_model = bigram;
    assert(!model_train(&bigram, &training, MODEL_BIGRAM, 2));
    assert(memcmp(&bigram, &unchanged_model, sizeof(bigram)) == 0);
    assert(unigram.unigram_count[BOS_TOKEN] == 0);
    assert(unigram.unigram_count[EOS_TOKEN] == training.document_count);

    double probabilities[MAX_VOCAB_SIZE];
    for (unsigned history = 0; history < VOCAB_SIZE; ++history) {
        assert(model_probabilities(&bigram, history, probabilities));
        double sum = 0.0;
        for (size_t token = 0; token < VOCAB_SIZE; ++token) {
            if (token == BOS_TOKEN) {
                assert(probabilities[token] == 0.0);
            } else {
                assert(probabilities[token] > 0.0);
                sum += probabilities[token];
            }
        }
        assert(fabs(sum - 1.0) < 1e-12);
    }
    assert(model_probabilities(&unigram, BOS_TOKEN, probabilities));
    double unigram_sum = 0.0;
    for (size_t token = 0; token < VOCAB_SIZE; ++token) {
        unigram_sum += probabilities[token];
    }
    assert(fabs(unigram_sum - 1.0) < 1e-12);
    assert(model_probabilities(&bigram, TOKEN_STUDY, probabilities));
    assert(probabilities[TOKEN_UNK] > 0.0);

    Metrics uni_metrics;
    Metrics bi_metrics;
    assert(model_evaluate(&unigram, &testing, &uni_metrics));
    assert(model_evaluate(&bigram, &testing, &bi_metrics));
    assert(uni_metrics.token_count == 10 && bi_metrics.token_count == 10);
    assert(isfinite(uni_metrics.nll) && uni_metrics.nll > 0.0);
    assert(isfinite(bi_metrics.nll) && bi_metrics.nll > 0.0);
    assert(uni_metrics.perplexity > 1.0 && bi_metrics.perplexity > 1.0);

    Model hand_unigram;
    Model hand_bigram;
    assert(build_hand_models(&hand_unigram, &hand_bigram));
    assert(model_probabilities(&hand_unigram, BOS_TOKEN, probabilities));
    assert(fabs(probabilities[3] - 2.0 / 7.0) < 1e-12);
    assert(fabs(probabilities[TOKEN_UNK] - 1.0 / 7.0) < 1e-12);
    assert(model_probabilities(&hand_bigram, BOS_TOKEN, probabilities));
    assert(fabs(probabilities[3] - 2.0 / 5.0) < 1e-12);
    assert(model_probabilities(&hand_bigram, 4, probabilities));
    assert(fabs(probabilities[EOS_TOKEN] - 2.0 / 5.0) < 1e-12);

    unsigned sample_a[MAX_GENERATED_TOKENS];
    unsigned sample_b[MAX_GENERATED_TOKENS];
    size_t sample_a_count = 0;
    size_t sample_b_count = 0;
    assert(model_sample(&bigram, 12345, sample_a, MAX_GENERATED_TOKENS, &sample_a_count));
    assert(model_sample(&bigram, 12345, sample_b, MAX_GENERATED_TOKENS, &sample_b_count));
    assert(sample_a_count == sample_b_count && sample_a_count <= MAX_GENERATED_TOKENS);
    assert(memcmp(sample_a, sample_b, sample_a_count * sizeof(sample_a[0])) == 0);

    Document unchanged = {.length = 1, .token = {77}};
    Document snapshot = unchanged;
    assert(!document_encode(&unchanged, nullptr));
    assert(memcmp(&unchanged, &snapshot, sizeof(unchanged)) == 0);
    char long_document[MAX_SEQUENCE_LENGTH * 2 + 1];
    for (size_t i = 0; i < MAX_SEQUENCE_LENGTH; ++i) {
        long_document[i * 2] = 'x';
        long_document[i * 2 + 1] = ' ';
    }
    long_document[sizeof(long_document) - 1] = '\0';
    assert(!document_encode(&unchanged, long_document));
    assert(memcmp(&unchanged, &snapshot, sizeof(unchanged)) == 0);
    assert(!model_probabilities(nullptr, BOS_TOKEN, probabilities));
    assert(!model_sample(&bigram, 1, sample_a, 0, &sample_a_count));
    return true;
}

/**
 * @brief 把采样 token 输出成便于阅读的一行文本。
 * @param label 当前模型的显示名称。
 * @param model 已训练的一元或二元模型，只读借用。
 * @param seed 固定伪随机种子，用于复现实验。
 */
static void print_sample(const char *label, const Model *model, uint32_t seed)
{
    unsigned tokens[MAX_GENERATED_TOKENS];
    size_t count = 0;
    printf("%s: ", label);
    if (!model_sample(model, seed, tokens, MAX_GENERATED_TOKENS, &count)) {
        puts("采样失败");
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        if (tokens[i] == EOS_TOKEN) {
            break;
        }
        const char *token_text = tokens[i] < VOCAB_SIZE ? vocabulary[tokens[i]] : "<invalid>";
        printf("%s%s", i == 0 ? "" : " ", token_text);
    }
    if (count == MAX_GENERATED_TOKENS && tokens[count - 1] != EOS_TOKEN) {
        printf(" …[达到长度上限]");
    }
    putchar('\n');
}

/**
 * @brief 运行固定语料的留出评估和一元/二元采样演示。
 * @return 数据编码、训练和评估都成功时为 true。
 */
static bool run_demo(void)
{
    Dataset training;
    Dataset testing;
    Model unigram;
    Model bigram;
    Metrics uni_metrics;
    Metrics bi_metrics;
    if (!dataset_encode(&training, training_text,
                        sizeof(training_text) / sizeof(training_text[0])) ||
        !dataset_encode(&testing, testing_text, sizeof(testing_text) / sizeof(testing_text[0])) ||
        !model_train(&unigram, &training, MODEL_UNIGRAM, VOCAB_SIZE) ||
        !model_train(&bigram, &training, MODEL_BIGRAM, VOCAB_SIZE) ||
        !model_evaluate(&unigram, &testing, &uni_metrics) ||
        !model_evaluate(&bigram, &testing, &bi_metrics)) {
        return false;
    }

    printf("按原始文档拆分：训练 %zu 篇，测试 %zu 篇；"
           "词表 %u 个，UNK 包含 OOV\n",
           training.document_count, testing.document_count, VOCAB_SIZE);
    printf("评分单位：词项加 EOS；BOS 仅作上下文。加性平滑 alpha=%.1f\n", ALPHA);
    printf("模型    测试 token 数    token 平均 NLL    困惑度 PPL\n");
    printf("一元    %12zu    %14.6f    %10.6f\n", uni_metrics.token_count,
           uni_metrics.nll, uni_metrics.perplexity);
    printf("二元    %12zu    %14.6f    %10.6f\n", bi_metrics.token_count,
           bi_metrics.nll, bi_metrics.perplexity);
    print_sample("一元生成", &unigram, 2026);
    print_sample("二元生成", &bigram, 2026);
    return true;
}

/**
 * @brief 解析教学演示、手算、CSV 对照和测试命令。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数字符串数组。
 * @return EXIT_SUCCESS 表示成功，其余值表示参数或运行失败。
 */
int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        if (!run_tests()) {
            return EXIT_FAILURE;
        }
        puts("ngram_language_model: all assertions passed");
        return EXIT_SUCCESS;
    }
    if (argc == 2 && strcmp(argv[1], "--hand") == 0) {
        return run_hand(stdout, false) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        Dataset training;
        Dataset testing;
        Model unigram;
        Model bigram;
        Metrics uni_metrics;
        Metrics bi_metrics;
        if (!dataset_encode(&training, training_text,
                            sizeof(training_text) / sizeof(training_text[0])) ||
            !dataset_encode(&testing, testing_text,
                            sizeof(testing_text) / sizeof(testing_text[0])) ||
            !model_train(&unigram, &training, MODEL_UNIGRAM, VOCAB_SIZE) ||
            !model_train(&bigram, &training, MODEL_BIGRAM, VOCAB_SIZE) ||
            !model_evaluate(&unigram, &testing, &uni_metrics) ||
            !model_evaluate(&bigram, &testing, &bi_metrics)) {
            return EXIT_FAILURE;
        }
        printf("test,unigram,%zu,%.12g,%.12g\n", uni_metrics.token_count,
               uni_metrics.nll, uni_metrics.perplexity);
        printf("test,bigram,%zu,%.12g,%.12g\n", bi_metrics.token_count,
               bi_metrics.nll, bi_metrics.perplexity);
        return run_hand(stdout, true) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc != 1) {
        fprintf(stderr, "用法：%s [--test|--hand|--csv]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (!run_demo()) {
        fputs("语言模型演示失败：请检查固定词表、语料和数值结果。\n",
              stderr);
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
