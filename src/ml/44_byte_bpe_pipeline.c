/**
 * @file
 * @brief UTF-8 字节编码、简化 BPE 训练和文档级 next-token 数据管线。
 *
 * 词表只从训练文档学习；验证集和测试集仅应用已有合并规则。
 * 每篇原文分别生成窗口，BOS/EOS/PAD 是不可参与 BPE 的控制 token。
 */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE_VOCAB_SIZE 256u
#define BOS_ID 256u
#define EOS_ID 257u
#define PAD_ID 258u
#define MERGE_ID_START 259u
#define MAX_MERGES 8u
#define MAX_VOCAB_SIZE (MERGE_ID_START + MAX_MERGES)
#define MAX_DOC_BYTES 64u
#define MAX_DOC_SEQUENCE (MAX_DOC_BYTES + 2u)
#define MAX_DOCS_PER_SPLIT 8u
#define MAX_WINDOWS 128u
#define CONTEXT_LENGTH 4u
#define WINDOW_STRIDE 2u
#define BATCH_SIZE 2u
#define MIN_PAIR_FREQUENCY 2u

typedef uint16_t TokenId;

typedef struct {
    size_t length;
    unsigned char bytes[MAX_DOC_BYTES];
} TokenInfo;

typedef struct {
    TokenId left;
    TokenId right;
    size_t frequency;
} Merge;

typedef struct {
    size_t merge_count;
    TokenInfo tokens[MAX_VOCAB_SIZE];
    Merge merges[MAX_MERGES];
} BPEModel;

typedef struct {
    size_t length;
    TokenId ids[MAX_DOC_SEQUENCE];
} TokenSequence;

typedef struct {
    size_t id;
    const char *text;
} RawDocument;

typedef enum {
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_TEST
} DataSplit;

typedef struct {
    DataSplit split;
    RawDocument source;
    size_t byte_length;
    TokenSequence sequence;
} EncodedDocument;

typedef struct {
    DataSplit split;
    size_t document_id;
    size_t start;
    size_t length;
    TokenId inputs[CONTEXT_LENGTH];
    TokenId targets[CONTEXT_LENGTH];
} TokenWindow;

typedef struct {
    size_t count;
    TokenWindow values[MAX_WINDOWS];
} WindowSet;

typedef struct {
    DataSplit split;
    size_t count;
    size_t document_ids[BATCH_SIZE];
    TokenId inputs[BATCH_SIZE][CONTEXT_LENGTH];
    TokenId targets[BATCH_SIZE][CONTEXT_LENGTH];
    bool mask[BATCH_SIZE][CONTEXT_LENGTH];
} TokenBatch;

typedef struct {
    size_t count;
    TokenBatch values[(MAX_WINDOWS + BATCH_SIZE - 1u) / BATCH_SIZE];
} BatchSet;

typedef struct {
    DataSplit split;
    const RawDocument *documents;
    size_t document_count;
    EncodedDocument encoded[MAX_DOCS_PER_SPLIT];
    WindowSet windows;
    BatchSet batches;
} SplitPipeline;

typedef struct {
    size_t split_count;
    SplitPipeline splits[3];
} Pipeline;

static const unsigned char MODEL_MAGIC[4] = {'B', 'P', 'E', '1'};
static const uint16_t MODEL_VERSION = 1u;

static const RawDocument training_documents[] = {
    {101u, "ababab"},
    {102u, "ababac"},
    {103u, "你好 ab"}
};

static const RawDocument validation_documents[] = {
    {201u, "abab"},
    {202u, "中文"}
};

static const RawDocument test_documents[] = {
    {301u, "ababa!"},
    {302u, "学AI"}
};

typedef struct {
    DataSplit split;
    const RawDocument *documents;
    size_t count;
} SplitSource;

static const SplitSource split_sources[] = {
    {SPLIT_TRAIN, training_documents, sizeof(training_documents) / sizeof(training_documents[0])},
    {SPLIT_VALIDATION, validation_documents,
     sizeof(validation_documents) / sizeof(validation_documents[0])},
    {SPLIT_TEST, test_documents, sizeof(test_documents) / sizeof(test_documents[0])}
};

/**
 * @brief 将数据划分枚举转换为稳定输出名称。
 * @param split 有效的数据划分枚举。
 * @return 对应名称；枚举无效时返回 "invalid"。
 */
static const char *split_name(DataSplit split)
{
    switch (split) {
        case SPLIT_TRAIN:
            return "train";
        case SPLIT_VALIDATION:
            return "validation";
        case SPLIT_TEST:
            return "test";
        default:
            return "invalid";
    }
}

/**
 * @brief 初始化 256 个单字节 token 和三个独立控制符。
 * @param model 可写模型；成功后旧内容被覆盖。
 * @return 无。
 */
static void model_init(BPEModel *model)
{
    memset(model, 0, sizeof(*model));
    for (size_t byte = 0; byte < BASE_VOCAB_SIZE; ++byte) {
        model->tokens[byte].length = 1;
        model->tokens[byte].bytes[0] = (unsigned char)byte;
    }
}

/**
 * @brief 判断编号是否为模型中可展开的普通文本 token。
 * @param model 已初始化模型。
 * @param token 待检查 token 编号。
 * @return 字节 token 或已学习合并 token 返回 true；
 *         控制符及越界编号返回 false。
 */
static bool is_text_token(const BPEModel *model, TokenId token)
{
    size_t vocabulary_size = MERGE_ID_START + model->merge_count;
    return token < BASE_VOCAB_SIZE ||
           (token >= MERGE_ID_START && token < vocabulary_size);
}

/**
 * @brief 将原始字节逐个转换成基础 byte token。
 * @param output 有效可写序列；失败时不修改。
 * @param bytes 字节数组；长度为零时可为空指针。
 * @param length 字节数，不能超过 MAX_DOC_BYTES。
 * @return 成功返回 true；输入无效或超出容量返回 false。
 */
static bool sequence_from_bytes(TokenSequence *output, const unsigned char *bytes, size_t length)
{
    if (output == nullptr || (bytes == nullptr && length != 0) || length > MAX_DOC_BYTES) {
        return false;
    }

    TokenSequence sequence = {.length = length};
    for (size_t index = 0; index < length; ++index) {
        sequence.ids[index] = bytes[index];
    }
    *output = sequence;
    return true;
}

/**
 * @brief 对序列执行一条从左向右、位置不重叠的合并规则。
 * @param model 合并 token 已写入其中的模型。
 * @param output 可写目标序列；失败时不修改。
 * @param input 只读输入序列。
 * @param merge 当前规则的左右 token 编号。
 * @param result 合并 token 编号。
 * @return 成功返回 true；序列或 token 非法时返回 false。
 */
static bool sequence_apply_merge(const BPEModel *model, TokenSequence *output,
                                 const TokenSequence *input, Merge merge, TokenId result)
{
    if (model == nullptr || output == nullptr || input == nullptr ||
        input->length > MAX_DOC_SEQUENCE || !is_text_token(model, merge.left) ||
        !is_text_token(model, merge.right) || !is_text_token(model, result)) {
        return false;
    }

    TokenSequence merged = {0};
    for (size_t index = 0; index < input->length;) {
        if (!is_text_token(model, input->ids[index])) {
            return false;
        }
        if (index + 1 < input->length && input->ids[index] == merge.left &&
            input->ids[index + 1] == merge.right) {
            if (merged.length >= MAX_DOC_SEQUENCE) {
                return false;
            }
            merged.ids[merged.length++] = result;
            index += 2;
        } else {
            if (merged.length >= MAX_DOC_SEQUENCE) {
                return false;
            }
            merged.ids[merged.length++] = input->ids[index++];
        }
    }
    *output = merged;
    return true;
}

/**
 * @brief 在模型词表末尾追加左右 token 的字节拼接结果。
 * @param model 可写模型；追加成功后 merge_count 增加一。
 * @param left 左 token 编号，必须是已有文本 token。
 * @param right 右 token 编号，必须是已有文本 token。
 * @param frequency 训练语料中该相邻对的出现次数，必须大于零。
 * @return 成功返回 true；容量、引用或 token 字节长度无效时返回 false。
 */
static bool model_add_merge(BPEModel *model, TokenId left, TokenId right, size_t frequency)
{
    if (model == nullptr || model->merge_count >= MAX_MERGES || frequency == 0 ||
        !is_text_token(model, left) || !is_text_token(model, right)) {
        return false;
    }

    const TokenInfo *left_info = &model->tokens[left];
    const TokenInfo *right_info = &model->tokens[right];
    if (left_info->length > MAX_DOC_BYTES - right_info->length) {
        return false;
    }

    TokenId result = (TokenId)(MERGE_ID_START + model->merge_count);
    TokenInfo merged = {.length = left_info->length + right_info->length};
    memcpy(merged.bytes, left_info->bytes, left_info->length);
    memcpy(merged.bytes + left_info->length, right_info->bytes, right_info->length);
    model->tokens[result] = merged;
    model->merges[model->merge_count] = (Merge){left, right, frequency};
    ++model->merge_count;
    return true;
}

/**
 * @brief 对任意受限字节串应用模型中的全部 BPE 合并规则。
 * @param model 已训练模型；其规则按学习顺序执行。
 * @param output 可写输出序列；失败时保持原内容不变。
 * @param bytes 输入字节；零长度时可为空指针。
 * @param length 输入字节数，范围为 0..MAX_DOC_BYTES。
 * @return 编码成功返回 true；输入、模型或容量非法时返回 false。
 */
static bool bpe_encode(const BPEModel *model, TokenSequence *output,
                       const unsigned char *bytes, size_t length)
{
    if (model == nullptr || output == nullptr || model->merge_count > MAX_MERGES) {
        return false;
    }

    TokenSequence encoded = {0};
    if (!sequence_from_bytes(&encoded, bytes, length)) {
        return false;
    }
    for (size_t rank = 0; rank < model->merge_count; ++rank) {
        TokenSequence merged = {0};
        TokenId result = (TokenId)(MERGE_ID_START + rank);
        if (!sequence_apply_merge(model, &merged, &encoded, model->merges[rank], result)) {
            return false;
        }
        encoded = merged;
    }
    *output = encoded;
    return true;
}

/**
 * @brief 将文本 token 展开回原始字节，不验证 UTF-8 合法性。
 * @param model 编码时使用的同一词表。
 * @param sequence 只包含普通字节或 BPE token 的序列。
 * @param output 可写字节缓冲区；展开结果为空时可为空指针。
 * @param capacity 输出缓冲区容量。
 * @param output_length 可写长度结果；失败时保持原值不变。
 * @return 解码成功返回 true；控制符、非法 ID 或输出容量不足返回 false。
 */
static bool bpe_decode(const BPEModel *model, const TokenSequence *sequence,
                       unsigned char *output, size_t capacity, size_t *output_length)
{
    if (model == nullptr || sequence == nullptr || output_length == nullptr ||
        sequence->length > MAX_DOC_SEQUENCE) {
        return false;
    }

    size_t required = 0;
    for (size_t index = 0; index < sequence->length; ++index) {
        TokenId token = sequence->ids[index];
        if (!is_text_token(model, token) ||
            model->tokens[token].length > MAX_DOC_BYTES - required) {
            return false;
        }
        required += model->tokens[token].length;
    }
    if (required > capacity || (required != 0 && output == nullptr)) {
        return false;
    }

    size_t cursor = 0;
    for (size_t index = 0; index < sequence->length; ++index) {
        const TokenInfo *token = &model->tokens[sequence->ids[index]];
        memcpy(output + cursor, token->bytes, token->length);
        cursor += token->length;
    }
    *output_length = required;
    return true;
}

/**
 * @brief 在独立的训练文档上学习最高频相邻 token 对。
 * @param output 可写目标模型；失败时保持原内容不变。
 * @param documents 唯一参与词表拟合的原始文档数组，不转移所有权。
 * @param document_count 文档数，范围为 1..MAX_DOCS_PER_SPLIT。
 * @param max_merges 最多新增的规则数，不超过 MAX_MERGES。
 * @param min_frequency 允许合并的最低频次，必须大于零。
 * @return 成功返回 true；重复 ID、空文本、越界或计数溢出返回 false。
 */
static bool bpe_train(BPEModel *output, const RawDocument *documents, size_t document_count,
                      size_t max_merges, size_t min_frequency)
{
    if (output == nullptr || documents == nullptr || document_count == 0 ||
        document_count > MAX_DOCS_PER_SPLIT || max_merges > MAX_MERGES || min_frequency == 0) {
        return false;
    }

    BPEModel trained = {0};
    model_init(&trained);
    TokenSequence sequences[MAX_DOCS_PER_SPLIT] = {0};
    for (size_t document = 0; document < document_count; ++document) {
        if (documents[document].text == nullptr) {
            return false;
        }
        for (size_t earlier = 0; earlier < document; ++earlier) {
            if (documents[earlier].id == documents[document].id) {
                return false;
            }
        }
        size_t length = strlen(documents[document].text);
        if (!sequence_from_bytes(&sequences[document],
                                 (const unsigned char *)documents[document].text, length)) {
            return false;
        }
    }

    for (size_t round = 0; round < max_merges; ++round) {
        size_t counts[MAX_VOCAB_SIZE][MAX_VOCAB_SIZE] = {{0}};
        size_t vocabulary_size = MERGE_ID_START + trained.merge_count;
        for (size_t document = 0; document < document_count; ++document) {
            const TokenSequence *sequence = &sequences[document];
            for (size_t position = 0; position + 1 < sequence->length; ++position) {
                TokenId left = sequence->ids[position];
                TokenId right = sequence->ids[position + 1];
                if (left >= vocabulary_size || right >= vocabulary_size ||
                    counts[left][right] == SIZE_MAX) {
                    return false;
                }
                ++counts[left][right];
            }
        }

        bool found = false;
        TokenId best_left = 0;
        TokenId best_right = 0;
        size_t best_count = 0;
        for (size_t left = 0; left < vocabulary_size; ++left) {
            for (size_t right = 0; right < vocabulary_size; ++right) {
                if (counts[left][right] > best_count) {
                    best_left = (TokenId)left;
                    best_right = (TokenId)right;
                    best_count = counts[left][right];
                    found = true;
                }
            }
        }
        if (!found || best_count < min_frequency) {
            break;
        }
        if (!model_add_merge(&trained, best_left, best_right, best_count)) {
            return false;
        }

        TokenId result = (TokenId)(MERGE_ID_START + trained.merge_count - 1u);
        for (size_t document = 0; document < document_count; ++document) {
            TokenSequence merged = {0};
            if (!sequence_apply_merge(&trained, &merged, &sequences[document],
                                     trained.merges[trained.merge_count - 1u], result)) {
                return false;
            }
            sequences[document] = merged;
        }
    }

    *output = trained;
    return true;
}

/**
 * @brief 将一篇原文编码并添加 BOS/EOS，保留文档来源元数据。
 * @param output 可写文档结果；失败时不修改。
 * @param model 仅由训练文本拟合的模型。
 * @param source 原始文档及其唯一 ID。
 * @param split 当前文档所属划分。
 * @return 成功返回 true；文本为空指针、超长或编码失败返回 false。
 */
static bool document_encode(EncodedDocument *output, const BPEModel *model,
                            RawDocument source, DataSplit split)
{
    if (output == nullptr || model == nullptr || source.text == nullptr ||
        (split != SPLIT_TRAIN && split != SPLIT_VALIDATION && split != SPLIT_TEST)) {
        return false;
    }

    size_t byte_length = strlen(source.text);
    TokenSequence body = {0};
    if (!bpe_encode(model, &body, (const unsigned char *)source.text, byte_length) ||
        body.length + 2u > MAX_DOC_SEQUENCE) {
        return false;
    }
    EncodedDocument encoded = {
        .split = split,
        .source = source,
        .byte_length = byte_length,
        .sequence = {.length = body.length + 2u}
    };
    encoded.sequence.ids[0] = BOS_ID;
    memcpy(encoded.sequence.ids + 1, body.ids, body.length * sizeof(body.ids[0]));
    encoded.sequence.ids[encoded.sequence.length - 1u] = EOS_ID;
    *output = encoded;
    return true;
}

/**
 * @brief 只从单篇文档的序列中生成重叠的 next-token 窗口。
 * @param windows 可写窗口集合；追加成功后 count 更新。
 * @param document 已编码的单篇文档。
 * @return 成功返回 true；容量或序列无效返回 false。
 */
static bool windows_append_document(WindowSet *windows, const EncodedDocument *document)
{
    if (windows == nullptr || document == nullptr || document->sequence.length < 2 ||
        document->sequence.length > MAX_DOC_SEQUENCE) {
        return false;
    }

    size_t final_input_start = document->sequence.length - 1u;
    for (size_t start = 0; start < final_input_start; start += WINDOW_STRIDE) {
        if (windows->count >= MAX_WINDOWS) {
            return false;
        }
        size_t available = final_input_start - start;
        size_t length = available < CONTEXT_LENGTH ? available : CONTEXT_LENGTH;
        TokenWindow window = {
            .split = document->split,
            .document_id = document->source.id,
            .start = start,
            .length = length
        };
        for (size_t offset = 0; offset < length; ++offset) {
            window.inputs[offset] = document->sequence.ids[start + offset];
            window.targets[offset] = document->sequence.ids[start + offset + 1u];
        }
        windows->values[windows->count++] = window;
    }
    return true;
}

/**
 * @brief 将同一划分的窗口按行组成定长批次并用 PAD 填充。
 * @param output 可写批次集合；失败时保持原内容不变。
 * @param windows 来源窗口；所有窗口必须属于同一数据划分。
 * @return 成功返回 true；窗口划分混杂或容量非法返回 false。
 */
static bool batches_build(BatchSet *output, const WindowSet *windows)
{
    if (output == nullptr || windows == nullptr || windows->count > MAX_WINDOWS) {
        return false;
    }
    if (windows->count > 0) {
        DataSplit split = windows->values[0].split;
        for (size_t index = 0; index < windows->count; ++index) {
            if (windows->values[index].split != split) {
                return false;
            }
        }
    }

    BatchSet batches = {0};
    for (size_t start = 0; start < windows->count; start += BATCH_SIZE) {
        TokenBatch batch = {
            .split = windows->values[start].split,
            .count = windows->count - start < BATCH_SIZE ? windows->count - start : BATCH_SIZE
        };
        for (size_t row = 0; row < BATCH_SIZE; ++row) {
            for (size_t column = 0; column < CONTEXT_LENGTH; ++column) {
                batch.inputs[row][column] = PAD_ID;
                batch.targets[row][column] = PAD_ID;
            }
        }
        for (size_t row = 0; row < batch.count; ++row) {
            const TokenWindow *window = &windows->values[start + row];
            batch.document_ids[row] = window->document_id;
            for (size_t column = 0; column < window->length; ++column) {
                batch.inputs[row][column] = window->inputs[column];
                batch.targets[row][column] = window->targets[column];
                batch.mask[row][column] = true;
            }
        }
        batches.values[batches.count++] = batch;
    }
    *output = batches;
    return true;
}

/**
 * @brief 分别对训练、验证和测试原文编码，再在各自边界内生成窗口。
 * @param output 可写完整管线；失败时不修改。
 * @param model 只从训练文档学习的 BPE 模型。
 * @return 成功返回 true；任何文档、窗口或批次构造失败时返回 false。
 */
static bool pipeline_build(Pipeline *output, const BPEModel *model)
{
    if (output == nullptr || model == nullptr) {
        return false;
    }

    Pipeline pipeline = {.split_count = sizeof(split_sources) / sizeof(split_sources[0])};
    for (size_t split_index = 0; split_index < pipeline.split_count; ++split_index) {
        const SplitSource *source = &split_sources[split_index];
        SplitPipeline *split = &pipeline.splits[split_index];
        split->split = source->split;
        split->documents = source->documents;
        split->document_count = source->count;
        if (split->document_count > MAX_DOCS_PER_SPLIT) {
            return false;
        }
        for (size_t document = 0; document < split->document_count; ++document) {
            if (!document_encode(&split->encoded[document], model,
                                 split->documents[document], split->split) ||
                !windows_append_document(&split->windows, &split->encoded[document])) {
                return false;
            }
        }
        if (!batches_build(&split->batches, &split->windows)) {
            return false;
        }
    }
    *output = pipeline;
    return true;
}

/**
 * @brief 以小端序写出 16 位无符号整数。
 * @param stream 可写二进制流。
 * @param value 待写数值。
 * @return 写出两个字节返回 true，否则返回 false。
 */
static bool write_u16(FILE *stream, uint16_t value)
{
    unsigned char bytes[2] = {(unsigned char)value, (unsigned char)(value >> 8)};
    return fwrite(bytes, sizeof(bytes), 1, stream) == 1;
}

/**
 * @brief 以小端序写出 32 位无符号整数。
 * @param stream 可写二进制流。
 * @param value 待写数值。
 * @return 写出四个字节返回 true，否则返回 false。
 */
static bool write_u32(FILE *stream, uint32_t value)
{
    unsigned char bytes[4] = {
        (unsigned char)value,
        (unsigned char)(value >> 8),
        (unsigned char)(value >> 16),
        (unsigned char)(value >> 24)
    };
    return fwrite(bytes, sizeof(bytes), 1, stream) == 1;
}

/**
 * @brief 从二进制流读取小端序 16 位无符号整数。
 * @param stream 可读二进制流。
 * @param output 可写结果；读取失败时不修改。
 * @return 成功读取两个字节返回 true。
 */
static bool read_u16(FILE *stream, uint16_t *output)
{
    unsigned char bytes[2];
    if (output == nullptr || fread(bytes, sizeof(bytes), 1, stream) != 1) {
        return false;
    }
    *output = (uint16_t)((uint16_t)bytes[0] | (uint16_t)((uint16_t)bytes[1] << 8));
    return true;
}

/**
 * @brief 从二进制流读取小端序 32 位无符号整数。
 * @param stream 可读二进制流。
 * @param output 可写结果；读取失败时不修改。
 * @return 成功读取四个字节返回 true。
 */
static bool read_u32(FILE *stream, uint32_t *output)
{
    unsigned char bytes[4];
    if (output == nullptr || fread(bytes, sizeof(bytes), 1, stream) != 1) {
        return false;
    }
    *output = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
              ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return true;
}

/**
 * @brief 将模型魔数、版本号、合并数和合并规则写到流。
 * @param stream 可写二进制流，调用者负责打开和关闭。
 * @param model 已训练模型；token 字节由基础字节与规则推导。
 * @return 完整写入返回 true；非法模型或 I/O 错误返回 false。
 */
static bool model_save_stream(FILE *stream, const BPEModel *model)
{
    if (stream == nullptr || model == nullptr || model->merge_count > MAX_MERGES ||
        fwrite(MODEL_MAGIC, sizeof(MODEL_MAGIC), 1, stream) != 1 ||
        !write_u16(stream, MODEL_VERSION) || !write_u16(stream, (uint16_t)model->merge_count)) {
        return false;
    }

    for (size_t rank = 0; rank < model->merge_count; ++rank) {
        const Merge *merge = &model->merges[rank];
        if (merge->frequency == 0 || merge->frequency > UINT32_MAX ||
            !write_u16(stream, merge->left) || !write_u16(stream, merge->right) ||
            !write_u32(stream, (uint32_t)merge->frequency)) {
            return false;
        }
    }
    return !ferror(stream);
}

/**
 * @brief 读取并校验版本化模型，验证成功后才替换调用者模型。
 * @param stream 可读二进制流，读取后须恰好到达文件末尾。
 * @param output 可写模型；任何校验失败时原内容保持不变。
 * @return 读取和校验成功返回 true；
 *         截断、版本错误、非法引用或尾随字节返回 false。
 */
static bool model_load_stream(FILE *stream, BPEModel *output)
{
    if (stream == nullptr || output == nullptr) {
        return false;
    }

    BPEModel loaded = {0};
    model_init(&loaded);
    unsigned char magic[sizeof(MODEL_MAGIC)];
    uint16_t version = 0;
    uint16_t merge_count = 0;
    if (fread(magic, sizeof(magic), 1, stream) != 1 ||
        memcmp(magic, MODEL_MAGIC, sizeof(magic)) != 0 ||
        !read_u16(stream, &version) || version != MODEL_VERSION ||
        !read_u16(stream, &merge_count) || merge_count > MAX_MERGES) {
        return false;
    }

    for (size_t rank = 0; rank < merge_count; ++rank) {
        uint16_t left = 0;
        uint16_t right = 0;
        uint32_t frequency = 0;
        if (!read_u16(stream, &left) || !read_u16(stream, &right) ||
            !read_u32(stream, &frequency) ||
            !model_add_merge(&loaded, left, right, frequency)) {
            return false;
        }
    }
    if (fgetc(stream) != EOF || ferror(stream)) {
        return false;
    }
    *output = loaded;
    return true;
}

/**
 * @brief 以独占新建方式保存模型，避免覆盖已有实验文件。
 * @param path 新文件路径；调用者须确保父目录存在。
 * @param model 要保存的模型。
 * @return 保存并关闭成功返回 true；路径已存在或 I/O 失败返回 false。
 */
static bool model_save_new_file(const char *path, const BPEModel *model)
{
    if (path == nullptr || model == nullptr) {
        return false;
    }
    FILE *stream = fopen(path, "wbx");
    if (stream == nullptr) {
        perror("无法独占创建模型文件");
        return false;
    }
    bool saved = model_save_stream(stream, model);
    if (fclose(stream) != 0) {
        saved = false;
    }
    return saved;
}

/**
 * @brief 从指定文件加载完整版本化模型。
 * @param path 已存在的模型文件路径。
 * @param output 可写模型；加载失败时不修改。
 * @return 加载及关闭成功返回 true；文件或格式错误返回 false。
 */
static bool model_load_file(const char *path, BPEModel *output)
{
    if (path == nullptr || output == nullptr) {
        return false;
    }
    FILE *stream = fopen(path, "rb");
    if (stream == nullptr) {
        perror("无法打开模型文件");
        return false;
    }
    BPEModel loaded_model = {0};
    bool loaded = model_load_stream(stream, &loaded_model);
    int close_result = fclose(stream);
    if (!loaded || close_result != 0) {
        return false;
    }
    *output = loaded_model;
    return true;
}

/**
 * @brief 打印 token ID 序列的分号分隔形式。
 * @param ids token 数组。
 * @param count token 数量。
 * @return 无。
 */
static void print_ids(const TokenId *ids, size_t count)
{
    for (size_t index = 0; index < count; ++index) {
        printf("%s%u", index == 0 ? "" : ";", (unsigned)ids[index]);
    }
}

/**
 * @brief 打印原始字节的小写十六进制形式。
 * @param bytes 原始字节数组。
 * @param count 字节数量。
 * @return 无。
 */
static void print_hex(const unsigned char *bytes, size_t count)
{
    for (size_t index = 0; index < count; ++index) {
        printf("%02x", (unsigned)bytes[index]);
    }
}

/**
 * @brief 输出模型、文档、窗口和批次的稳定 CSV 记录。
 * @param model 已训练或已加载模型。
 * @param pipeline 按来源文档独立生成的数据管线。
 * @return 无。
 */
static void print_csv(const BPEModel *model, const Pipeline *pipeline)
{
    for (size_t rank = 0; rank < model->merge_count; ++rank) {
        const Merge *merge = &model->merges[rank];
        printf("merge,%zu,%u,%u,%zu,%u\n", rank, (unsigned)merge->left,
               (unsigned)merge->right, merge->frequency,
               (unsigned)(MERGE_ID_START + rank));
    }
    for (size_t split_index = 0; split_index < pipeline->split_count; ++split_index) {
        const SplitPipeline *split = &pipeline->splits[split_index];
        for (size_t document = 0; document < split->document_count; ++document) {
            const EncodedDocument *encoded = &split->encoded[document];
            printf("document,%s,%zu,%zu,", split_name(encoded->split), encoded->source.id,
                   encoded->byte_length);
            print_ids(encoded->sequence.ids + 1, encoded->sequence.length - 2u);
            putchar(',');
            print_hex((const unsigned char *)encoded->source.text, encoded->byte_length);
            putchar(',');
            print_ids(encoded->sequence.ids, encoded->sequence.length);
            putchar('\n');
        }
    }
    for (size_t split_index = 0; split_index < pipeline->split_count; ++split_index) {
        const WindowSet *windows = &pipeline->splits[split_index].windows;
        for (size_t index = 0; index < windows->count; ++index) {
            const TokenWindow *window = &windows->values[index];
            printf("window,%s,%zu,%zu,%zu,", split_name(window->split),
                   window->document_id, window->start, window->length);
            print_ids(window->inputs, window->length);
            putchar(',');
            print_ids(window->targets, window->length);
            putchar(',');
            for (size_t position = 0; position < window->length; ++position) {
                printf("%s1", position == 0 ? "" : ";");
            }
            putchar('\n');
        }
    }
    for (size_t split_index = 0; split_index < pipeline->split_count; ++split_index) {
        const BatchSet *batches = &pipeline->splits[split_index].batches;
        for (size_t index = 0; index < batches->count; ++index) {
            const TokenBatch *batch = &batches->values[index];
            printf("batch,%s,%zu,%zu,", split_name(batch->split), index, batch->count);
            for (size_t row = 0; row < batch->count; ++row) {
                printf("%s%zu", row == 0 ? "" : ";", batch->document_ids[row]);
            }
            putchar(',');
            for (size_t row = 0; row < batch->count; ++row) {
                if (row != 0) {
                    putchar('|');
                }
                print_ids(batch->inputs[row], CONTEXT_LENGTH);
            }
            putchar(',');
            for (size_t row = 0; row < batch->count; ++row) {
                if (row != 0) {
                    putchar('|');
                }
                print_ids(batch->targets[row], CONTEXT_LENGTH);
            }
            putchar(',');
            for (size_t row = 0; row < batch->count; ++row) {
                if (row != 0) {
                    putchar('|');
                }
                for (size_t column = 0; column < CONTEXT_LENGTH; ++column) {
                    printf("%s%d", column == 0 ? "" : ";", batch->mask[row][column] ? 1 : 0);
                }
            }
            putchar('\n');
        }
    }
}

/**
 * @brief 校验 UTF-8 字节往返、BPE 次序、数据边界和版本化持久化。
 * @return 全部断言通过返回 0。
 */
static int run_tests(void)
{
    BPEModel model = {0};
    assert(bpe_train(&model, training_documents,
                     sizeof(training_documents) / sizeof(training_documents[0]),
                     MAX_MERGES, MIN_PAIR_FREQUENCY));
    assert(model.merge_count == 2);
    assert(model.merges[0].left == 'a' && model.merges[0].right == 'b');
    assert(model.merges[0].frequency == 6);
    assert(model.merges[1].left == MERGE_ID_START);
    assert(model.merges[1].right == MERGE_ID_START);
    assert(model.merges[1].frequency == 3);

    const unsigned char chinese[] = "中文学习";
    TokenSequence encoded = {0};
    assert(bpe_encode(&model, &encoded, chinese, sizeof(chinese) - 1u));
    unsigned char decoded[MAX_DOC_BYTES] = {0};
    size_t decoded_length = 0;
    assert(bpe_decode(&model, &encoded, decoded, sizeof(decoded), &decoded_length));
    assert(decoded_length == sizeof(chinese) - 1u);
    assert(memcmp(decoded, chinese, decoded_length) == 0);

    const unsigned char arbitrary_bytes[] = {0u, 255u, 128u};
    assert(bpe_encode(&model, &encoded, arbitrary_bytes, sizeof(arbitrary_bytes)));
    assert(bpe_decode(&model, &encoded, decoded, sizeof(decoded), &decoded_length));
    assert(decoded_length == sizeof(arbitrary_bytes));
    assert(memcmp(decoded, arbitrary_bytes, decoded_length) == 0);

    unsigned char maximum_bytes[MAX_DOC_BYTES];
    for (size_t index = 0; index < MAX_DOC_BYTES; ++index) {
        maximum_bytes[index] = (unsigned char)index;
    }
    assert(bpe_encode(&model, &encoded, maximum_bytes, sizeof(maximum_bytes)));
    assert(bpe_decode(&model, &encoded, decoded, sizeof(decoded), &decoded_length));
    assert(decoded_length == sizeof(maximum_bytes));
    assert(memcmp(decoded, maximum_bytes, decoded_length) == 0);

    TokenSequence empty = {.length = 99};
    assert(bpe_encode(&model, &empty, nullptr, 0));
    assert(empty.length == 0);
    decoded_length = 99;
    assert(bpe_decode(&model, &empty, nullptr, 0, &decoded_length));
    assert(decoded_length == 0);
    const RawDocument empty_source = {998u, ""};
    EncodedDocument empty_document = {0};
    assert(document_encode(&empty_document, &model, empty_source, SPLIT_TRAIN));
    assert(empty_document.sequence.length == 2);
    assert(empty_document.sequence.ids[0] == BOS_ID);
    assert(empty_document.sequence.ids[1] == EOS_ID);

    unsigned char oversized[MAX_DOC_BYTES + 1u] = {0};
    TokenSequence unchanged = {.length = 77};
    assert(!bpe_encode(&model, &unchanged, oversized, sizeof(oversized)));
    assert(unchanged.length == 77);
    decoded_length = 77;
    assert(!bpe_decode(&model, &encoded, decoded, 1, &decoded_length));
    assert(decoded_length == 77);
    TokenSequence control = {.length = 1, .ids = {BOS_ID}};
    assert(!bpe_decode(&model, &control, decoded, sizeof(decoded), &decoded_length));
    assert(decoded_length == 77);

    const RawDocument tie_document[] = {{1u, "aba"}};
    BPEModel tie_model = {0};
    assert(bpe_train(&tie_model, tie_document, 1, 1, 1));
    assert(tie_model.merges[0].left == 'a' && tie_model.merges[0].right == 'b');

    RawDocument train_plus_validation[5] = {
        training_documents[0], training_documents[1], training_documents[2],
        validation_documents[0], validation_documents[1]
    };
    BPEModel leaked_fit = {0};
    assert(bpe_train(&leaked_fit, train_plus_validation, 5, 1, 1));
    assert(leaked_fit.merges[0].frequency == 8);
    assert(model.merges[0].frequency == 6);

    BPEModel byte_model = {0};
    model_init(&byte_model);
    EncodedDocument long_document = {0};
    const RawDocument overlap_source[] = {{999u, "abcdef"}};
    assert(document_encode(&long_document, &byte_model, overlap_source[0], SPLIT_TRAIN));
    WindowSet overlap_windows = {0};
    assert(windows_append_document(&overlap_windows, &long_document));
    assert(overlap_windows.count >= 2);
    assert(overlap_windows.values[0].inputs[2] == overlap_windows.values[1].inputs[0]);
    assert(overlap_windows.values[0].split == overlap_windows.values[1].split);
    assert(overlap_windows.values[0].document_id == overlap_windows.values[1].document_id);

    Pipeline pipeline = {0};
    assert(pipeline_build(&pipeline, &model));
    for (size_t split_index = 0; split_index < pipeline.split_count; ++split_index) {
        const SplitPipeline *split = &pipeline.splits[split_index];
        for (size_t window_index = 0; window_index < split->windows.count; ++window_index) {
            const TokenWindow *window = &split->windows.values[window_index];
            assert(window->split == split->split);
            bool source_found = false;
            for (size_t document = 0; document < split->document_count; ++document) {
                const EncodedDocument *encoded_document = &split->encoded[document];
                if (encoded_document->source.id == window->document_id) {
                    source_found = true;
                    assert(window->start + window->length < encoded_document->sequence.length);
                }
            }
            assert(source_found);
        }
        for (size_t batch_index = 0; batch_index < split->batches.count; ++batch_index) {
            const TokenBatch *batch = &split->batches.values[batch_index];
            assert(batch->split == split->split);
            for (size_t row = 0; row < batch->count; ++row) {
                for (size_t column = 0; column < CONTEXT_LENGTH; ++column) {
                    if (!batch->mask[row][column]) {
                        assert(batch->inputs[row][column] == PAD_ID);
                        assert(batch->targets[row][column] == PAD_ID);
                    }
                }
            }
        }
        for (size_t other_split = split_index + 1u;
             other_split < pipeline.split_count; ++other_split) {
            const SplitPipeline *other = &pipeline.splits[other_split];
            for (size_t document = 0; document < split->document_count; ++document) {
                for (size_t other_document = 0;
                     other_document < other->document_count; ++other_document) {
                    assert(split->encoded[document].source.id !=
                           other->encoded[other_document].source.id);
                }
            }
        }
    }

    FILE *stream = tmpfile();
    assert(stream != nullptr);
    assert(model_save_stream(stream, &model));
    rewind(stream);
    BPEModel loaded = {0};
    assert(model_load_stream(stream, &loaded));
    assert(loaded.merge_count == model.merge_count);
    assert(loaded.merges[0].left == model.merges[0].left);
    assert(loaded.merges[1].right == model.merges[1].right);
    assert(loaded.tokens[MERGE_ID_START + 1u].length ==
           model.tokens[MERGE_ID_START + 1u].length);
    assert(fclose(stream) == 0);

    stream = tmpfile();
    assert(stream != nullptr);
    assert(model_save_stream(stream, &model));
    assert(fseek(stream, 4, SEEK_SET) == 0);
    assert(write_u16(stream, 99u));
    rewind(stream);
    BPEModel unchanged_model = {0};
    model_init(&unchanged_model);
    assert(!bpe_train(&unchanged_model, training_documents, 0, 1, 1));
    assert(unchanged_model.merge_count == 0);
    assert(!bpe_train(&unchanged_model, training_documents,
                      sizeof(training_documents) / sizeof(training_documents[0]),
                      MAX_MERGES + 1u, 1));
    assert(unchanged_model.merge_count == 0);
    assert(!model_load_stream(stream, &unchanged_model));
    assert(unchanged_model.merge_count == 0);
    assert(fclose(stream) == 0);
    return 0;
}

/**
 * @brief 打印可手工复算的两轮合并与并列规则。
 * @return 无。
 */
static void print_hand_example(void)
{
    const RawDocument example[] = {{1u, "ababab"}};
    BPEModel model = {0};
    const RawDocument tie[] = {{2u, "aba"}};
    BPEModel tie_model = {0};
    bool trained = bpe_train(&model, example, 1, 2, 1);
    bool tie_trained = bpe_train(&tie_model, tie, 1, 1, 1);
    assert(trained && tie_trained);
    puts("文本 ababab：");
    puts("第 0 轮：ab 出现 3 次、ba 出现 2 次，先合并 (a,b)。");
    puts("第 1 轮：新 token ab 的相邻对 (ab,ab) 出现 2 次，再合并为 abab。");
    printf("计算得到规则：(%u,%u,%zu)；(%u,%u,%zu)。\n",
           (unsigned)model.merges[0].left, (unsigned)model.merges[0].right,
           model.merges[0].frequency, (unsigned)model.merges[1].left,
           (unsigned)model.merges[1].right, model.merges[1].frequency);
    puts("并列例子 aba：ab 与 ba 各出现 1 次；token ID 字典序选择 (97,98)。");
    printf("计算得到并列规则：(%u,%u,%zu)。\n",
           (unsigned)tie_model.merges[0].left, (unsigned)tie_model.merges[0].right,
           tie_model.merges[0].frequency);
}

/**
 * @brief 生成默认演示的规则、UTF-8 往返和三划分管线摘要。
 * @param model 已训练或已加载模型。
 * @param pipeline 已分别构造的训练/验证/测试管线。
 * @return 无。
 */
static void print_demo(const BPEModel *model, const Pipeline *pipeline)
{
    printf("L02 UTF-8 字节与简化 BPE\n"
           "训练得到 %zu 条合并规则：\n", model->merge_count);
    for (size_t rank = 0; rank < model->merge_count; ++rank) {
        const Merge *merge = &model->merges[rank];
        printf("  rank=%zu pair=(%u,%u) frequency=%zu result=%u\n", rank,
               (unsigned)merge->left, (unsigned)merge->right, merge->frequency,
               (unsigned)(MERGE_ID_START + rank));
    }

    const unsigned char chinese[] = "你好";
    TokenSequence encoded = {0};
    unsigned char decoded[MAX_DOC_BYTES] = {0};
    size_t decoded_length = 0;
    bool roundtrip = bpe_encode(model, &encoded, chinese, sizeof(chinese) - 1u) &&
                     bpe_decode(model, &encoded, decoded, sizeof(decoded), &decoded_length);
    assert(roundtrip && decoded_length == sizeof(chinese) - 1u);
    printf("中文字节：");
    print_hex(chinese, sizeof(chinese) - 1u);
    printf(" -> token IDs ");
    print_ids(encoded.ids, encoded.length);
    printf(" -> ");
    print_hex(decoded, decoded_length);
    putchar('\n');

    for (size_t split_index = 0; split_index < pipeline->split_count; ++split_index) {
        const SplitPipeline *split = &pipeline->splits[split_index];
        printf("%s: documents=%zu windows=%zu batches=%zu\n", split_name(split->split),
               split->document_count, split->windows.count, split->batches.count);
    }
    puts("BOS/EOS/PAD 是独立控制符；字节词表覆盖全部 256 种字节，没有 UNK。");
}

/**
 * @brief 训练或加载模型，按命令行选项运行演示、序列化或断言。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数数组。
 * @return 成功返回 EXIT_SUCCESS；参数、文件或管线错误返回 EXIT_FAILURE。
 */
int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        int result = run_tests();
        puts("L02 C assertions passed");
        return result;
    }
    if (argc == 2 && strcmp(argv[1], "--hand") == 0) {
        print_hand_example();
        return EXIT_SUCCESS;
    }

    bool csv = argc == 2 && strcmp(argv[1], "--csv") == 0;
    bool save = argc == 3 && strcmp(argv[1], "--save") == 0;
    bool load = argc == 3 && strcmp(argv[1], "--load") == 0;
    bool load_csv = argc == 4 && strcmp(argv[1], "--load") == 0 &&
                    strcmp(argv[3], "--csv") == 0;
    if (!(argc == 1 || csv || save || load || load_csv)) {
        fprintf(stderr,
                "用法：%s [--hand | --csv | --save 路径 | --load 路径 [--csv]]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    BPEModel model = {0};
    bool ready = false;
    if (load || load_csv) {
        ready = model_load_file(argv[2], &model);
    } else {
        ready = bpe_train(&model, training_documents,
                          sizeof(training_documents) / sizeof(training_documents[0]),
                          MAX_MERGES, MIN_PAIR_FREQUENCY);
    }
    if (!ready) {
        fputs("无法建立或加载 BPE 模型。\n", stderr);
        return EXIT_FAILURE;
    }
    if (save && !model_save_new_file(argv[2], &model)) {
        fputs("模型未能保存；已有文件不会被覆盖。\n", stderr);
        return EXIT_FAILURE;
    }

    Pipeline pipeline = {0};
    if (!pipeline_build(&pipeline, &model)) {
        fputs("文本数据管线构造失败。\n", stderr);
        return EXIT_FAILURE;
    }
    if (csv || load_csv) {
        print_csv(&model, &pipeline);
    } else {
        print_demo(&model, &pipeline);
        if (save) {
            printf("已独占创建版本化模型文件：%s\n", argv[2]);
        }
    }
    return EXIT_SUCCESS;
}
