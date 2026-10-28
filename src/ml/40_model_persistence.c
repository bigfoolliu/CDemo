/**
 * @file
 * @brief 将训练集标准化参数和逻辑回归权重保存为可独立加载的模型文件。
 */
#include <errno.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "study_dataset.h"

enum {
    MODEL_HEADER_BYTES = 24,
    MODEL_PAYLOAD_BYTES = 56,
    MODEL_FILE_BYTES = MODEL_HEADER_BYTES + MODEL_PAYLOAD_BYTES + 4,
    MODEL_PARAMETER_COUNT = 7,
    MODEL_FORMAT_VERSION = 1,
    TRAINING_EPOCHS = 1200
};

static const uint8_t MODEL_MAGIC[8] = {'C', 'D', 'M', 'L', 'P', 'R', 'S', 'T'};
static const double LEARNING_RATE = 0.15;

typedef struct {
    double weights[FEATURE_COUNT];
    double bias;
} LogisticModel;

typedef struct {
    Standardizer standardizer;
    LogisticModel model;
} PersistedModel;

typedef struct {
    double probability;
    double logit;
    double standardized[FEATURE_COUNT];
    int label;
} Prediction;

/**
 * @brief 检查当前平台能否把 double 按 IEEE 754 binary64 写入文件。
 * @return double 满足 binary64 基数、有效位数和指数范围时为 true。
 */
static bool wire_format_supported(void)
{
    return sizeof(double) == sizeof(uint64_t) && FLT_RADIX == 2 && DBL_MANT_DIG == 53 &&
           DBL_MAX_EXP == 1024;
}

/**
 * @brief 将 32 位无符号整数编码为小端序。
 * @param bytes 至少四字节的输出缓冲区。
 * @param value 要编码的整数。
 */
static void write_u32_le(uint8_t *bytes, uint32_t value)
{
    for (size_t index = 0; index < 4; ++index) {
        bytes[index] = (uint8_t)(value >> (index * 8));
    }
}

/**
 * @brief 从小端序字节读取 32 位无符号整数。
 * @param bytes 至少四字节的输入缓冲区。
 * @return 解码出的整数。
 */
static uint32_t read_u32_le(const uint8_t *bytes)
{
    uint32_t value = 0;
    for (size_t index = 0; index < 4; ++index) {
        value |= (uint32_t)bytes[index] << (index * 8);
    }
    return value;
}

/**
 * @brief 将 64 位无符号整数编码为小端序。
 * @param bytes 至少八字节的输出缓冲区。
 * @param value 要编码的整数。
 */
static void write_u64_le(uint8_t *bytes, uint64_t value)
{
    for (size_t index = 0; index < 8; ++index) {
        bytes[index] = (uint8_t)(value >> (index * 8));
    }
}

/**
 * @brief 从小端序字节读取 64 位无符号整数。
 * @param bytes 至少八字节的输入缓冲区。
 * @return 解码出的整数。
 */
static uint64_t read_u64_le(const uint8_t *bytes)
{
    uint64_t value = 0;
    for (size_t index = 0; index < 8; ++index) {
        value |= (uint64_t)bytes[index] << (index * 8);
    }
    return value;
}

/**
 * @brief 计算 CRC-32/ISO-HDLC，用于发现模型字节的意外损坏。
 * @param bytes 要校验的字节序列。
 * @param length 字节数。
 * @return 32 位校验值；不提供密码学防篡改保证。
 */
static uint32_t crc32_bytes(const uint8_t *bytes, size_t length)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0; index < length; ++index) {
        crc ^= bytes[index];
        for (size_t bit = 0; bit < 8; ++bit) {
            uint32_t mask = (uint32_t)-(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

/**
 * @brief 覆盖文件尾部的 CRC 字段，使测试能构造结构有效但内容非法的文件。
 * @param bytes 完整模型字节数组。
 */
static void refresh_checksum(uint8_t bytes[MODEL_FILE_BYTES])
{
    write_u32_le(bytes + MODEL_FILE_BYTES - 4,
                 crc32_bytes(bytes, MODEL_FILE_BYTES - 4));
}

/**
 * @brief 验证推理所需的均值、尺度、权重和偏置均处于有效范围。
 * @param state 待验证的完整推理状态。
 * @return 参数有限且标准差大于机器精度阈值时为 true。
 */
static bool persisted_model_is_valid(const PersistedModel *state)
{
    if (state == nullptr) {
        return false;
    }

    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(state->standardizer.mean[feature]) ||
            !isfinite(state->standardizer.scale[feature]) ||
            state->standardizer.scale[feature] <= DBL_EPSILON ||
            !isfinite(state->model.weights[feature])) {
            return false;
        }
    }
    return isfinite(state->model.bias);
}

/**
 * @brief 按固定字段顺序把推理状态编码成版本化小端序二进制文件。
 * @param state 均值、标准差、权重和偏置均有效的推理状态。
 * @param output 至少 MODEL_FILE_BYTES 字节的输出缓冲区；失败时保持不变。
 * @return 编码成功时为 true。
 */
static bool model_encode(const PersistedModel *state, uint8_t *output, size_t capacity)
{
    if (!wire_format_supported() || !persisted_model_is_valid(state) || output == nullptr ||
        capacity < MODEL_FILE_BYTES) {
        return false;
    }

    uint8_t encoded[MODEL_FILE_BYTES] = {0};
    memcpy(encoded, MODEL_MAGIC, sizeof(MODEL_MAGIC));
    write_u32_le(encoded + 8, MODEL_FORMAT_VERSION);
    write_u32_le(encoded + 12, FEATURE_COUNT);
    write_u32_le(encoded + 16, MODEL_PARAMETER_COUNT);
    write_u32_le(encoded + 20, MODEL_PAYLOAD_BYTES);

    size_t offset = MODEL_HEADER_BYTES;
    const double parameters[MODEL_PARAMETER_COUNT] = {
        state->standardizer.mean[0], state->standardizer.mean[1],
        state->standardizer.scale[0], state->standardizer.scale[1],
        state->model.weights[0], state->model.weights[1], state->model.bias
    };
    for (size_t index = 0; index < MODEL_PARAMETER_COUNT; ++index) {
        uint64_t bits = 0;
        memcpy(&bits, &parameters[index], sizeof(bits));
        write_u64_le(encoded + offset, bits);
        offset += sizeof(bits);
    }

    refresh_checksum(encoded);
    memcpy(output, encoded, sizeof(encoded));
    return true;
}

/**
 * @brief 严格解析模型文件，并在全部字段验证成功后一次性替换输出状态。
 * @param bytes 输入模型字节。
 * @param length 必须恰好等于 MODEL_FILE_BYTES。
 * @param output 有效可写状态；任何失败都保持原值。
 * @return 魔数、版本、维度、长度、校验和及参数均有效时为 true。
 */
static bool model_decode(const uint8_t *bytes, size_t length, PersistedModel *output)
{
    if (!wire_format_supported() || bytes == nullptr || output == nullptr ||
        length != MODEL_FILE_BYTES || memcmp(bytes, MODEL_MAGIC, sizeof(MODEL_MAGIC)) != 0 ||
        read_u32_le(bytes + 8) != MODEL_FORMAT_VERSION ||
        read_u32_le(bytes + 12) != FEATURE_COUNT ||
        read_u32_le(bytes + 16) != MODEL_PARAMETER_COUNT ||
        read_u32_le(bytes + 20) != MODEL_PAYLOAD_BYTES ||
        read_u32_le(bytes + MODEL_FILE_BYTES - 4) !=
            crc32_bytes(bytes, MODEL_FILE_BYTES - 4)) {
        return false;
    }

    double parameters[MODEL_PARAMETER_COUNT];
    size_t offset = MODEL_HEADER_BYTES;
    for (size_t index = 0; index < MODEL_PARAMETER_COUNT; ++index) {
        uint64_t bits = read_u64_le(bytes + offset);
        memcpy(&parameters[index], &bits, sizeof(bits));
        if (!isfinite(parameters[index])) {
            return false;
        }
        offset += sizeof(bits);
    }

    PersistedModel loaded = {0};
    loaded.standardizer.mean[0] = parameters[0];
    loaded.standardizer.mean[1] = parameters[1];
    loaded.standardizer.scale[0] = parameters[2];
    loaded.standardizer.scale[1] = parameters[3];
    loaded.model.weights[0] = parameters[4];
    loaded.model.weights[1] = parameters[5];
    loaded.model.bias = parameters[6];
    if (!persisted_model_is_valid(&loaded)) {
        return false;
    }

    *output = loaded;
    return true;
}

/**
 * @brief 独占创建新模型文件，写失败时移除本次创建的不完整文件。
 * @param path 目标文件路径；父目录必须已存在。
 * @param state 通过完整参数验证的推理状态。
 * @return 文件创建、写入、刷新和关闭均成功时为 true；已有路径绝不覆盖。
 */
static bool model_save_exclusive(const char *path, const PersistedModel *state)
{
    if (path == nullptr || path[0] == '\0') {
        return false;
    }

    uint8_t encoded[MODEL_FILE_BYTES];
    if (!model_encode(state, encoded, sizeof(encoded))) {
        return false;
    }

    FILE *stream = fopen(path, "wbx");
    if (stream == nullptr) {
        return false;
    }

    bool written = fwrite(encoded, 1, sizeof(encoded), stream) == sizeof(encoded) &&
                   fflush(stream) == 0;
    int close_result = fclose(stream);
    if (!written || close_result != 0) {
        remove(path);
        return false;
    }
    return true;
}

/**
 * @brief 加载并验证固定长度模型文件，避免把部分读取结果写入调用方状态。
 * @param path 模型文件路径。
 * @param output 有效可写推理状态；失败时保持原值。
 * @return 文件长度、版本、校验和及全部参数均有效时为 true。
 */
static bool model_load(const char *path, PersistedModel *output)
{
    if (path == nullptr || path[0] == '\0' || output == nullptr) {
        return false;
    }

    FILE *stream = fopen(path, "rb");
    if (stream == nullptr) {
        return false;
    }

    uint8_t bytes[MODEL_FILE_BYTES];
    size_t count = fread(bytes, 1, sizeof(bytes), stream);
    int extra = fgetc(stream);
    bool read_ok = count == sizeof(bytes) && extra == EOF && !ferror(stream);
    int close_result = fclose(stream);
    if (!read_ok || close_result != 0) {
        return false;
    }

    PersistedModel loaded = {0};
    if (!model_decode(bytes, sizeof(bytes), &loaded)) {
        return false;
    }
    *output = loaded;
    return true;
}

/**
 * @brief 稳定计算 sigmoid，避免大负数直接导致指数溢出。
 * @param value 有限 logit。
 * @return [0,1] 内的正类概率。
 */
static double sigmoid(double value)
{
    if (value >= 0.0) {
        return 1.0 / (1.0 + exp(-value));
    }

    double exponential = exp(value);
    return exponential / (1.0 + exponential);
}

/**
 * @brief 计算稳定二元交叉熵及其对 logit 的解析导数。
 * @param logit 有限线性分数。
 * @param label 真实二分类标签 0 或 1。
 * @param loss_output 有效可写损失；失败时保持不变。
 * @param gradient_output 有效可写 logit 梯度；失败时保持不变。
 * @return 损失和梯度均有限时为 true。
 */
static bool binary_loss_gradient(double logit, int label, double *loss_output,
                                 double *gradient_output)
{
    if (!isfinite(logit) || (label != 0 && label != 1) || loss_output == nullptr ||
        gradient_output == nullptr) {
        return false;
    }

    double loss = fmax(logit, 0.0) - (double)label * logit +
                  log1p(exp(-fabs(logit)));
    double gradient = sigmoid(logit) - (double)label;
    if (!isfinite(loss) || !isfinite(gradient)) {
        return false;
    }

    *loss_output = loss;
    *gradient_output = gradient;
    return true;
}

/**
 * @brief 先使用训练时保存的标准化参数，再计算逻辑回归预测。
 * @param state 完整且有效的推理状态。
 * @param features 两个有限原始特征值。
 * @param output 有效可写预测；失败时保持不变。
 * @return 中间值和预测均有限时为 true。
 */
static bool model_predict(const PersistedModel *state, const double features[FEATURE_COUNT],
                          Prediction *output)
{
    if (!persisted_model_is_valid(state) || features == nullptr || output == nullptr) {
        return false;
    }

    Sample sample = {.features = {features[0], features[1]}};
    Prediction prediction = {0};
    if (!standardize_sample(&state->standardizer, &sample, prediction.standardized)) {
        return false;
    }
    prediction.logit = state->model.bias;
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        prediction.logit += state->model.weights[feature] * prediction.standardized[feature];
    }
    if (!isfinite(prediction.logit)) {
        return false;
    }

    prediction.probability = sigmoid(prediction.logit);
    if (!isfinite(prediction.probability)) {
        return false;
    }
    prediction.label = prediction.probability >= 0.5 ? 1 : 0;
    *output = prediction;
    return true;
}

/**
 * @brief 用训练划分拟合固定轮数的全批量逻辑回归。
 * @param dataset 含有效训练集的只读 CSV 数据。
 * @param standardizer 只由训练集计算的均值和标准差。
 * @param output 有效可写模型；失败时保持不变。
 * @return 全部更新有限且完成训练时为 true。
 */
static bool model_train(const Dataset *dataset, const Standardizer *standardizer,
                        LogisticModel *output)
{
    if (!dataset_is_valid(dataset) || standardizer == nullptr || output == nullptr ||
        !isfinite(LEARNING_RATE) || LEARNING_RATE <= 0.0) {
        return false;
    }

    LogisticModel trained = {0};
    size_t training_count = dataset->split_counts[SPLIT_TRAIN];
    for (size_t epoch = 0; epoch < TRAINING_EPOCHS; ++epoch) {
        double weight_gradients[FEATURE_COUNT] = {0};
        double bias_gradient = 0.0;
        double training_loss = 0.0;
        for (size_t index = 0; index < dataset->count; ++index) {
            const Sample *sample = &dataset->samples[index];
            if (sample->split != SPLIT_TRAIN) {
                continue;
            }

            double normalized[FEATURE_COUNT];
            if (!standardize_sample(standardizer, sample, normalized)) {
                return false;
            }
            double logit = trained.bias;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                logit += trained.weights[feature] * normalized[feature];
            }
            if (!isfinite(logit)) {
                return false;
            }

            double sample_loss = 0.0;
            double error = 0.0;
            if (!binary_loss_gradient(logit, sample->label, &sample_loss, &error)) {
                return false;
            }
            training_loss += sample_loss;
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                weight_gradients[feature] += error * normalized[feature];
            }
            bias_gradient += error;
        }
        if (!isfinite(training_loss)) {
            return false;
        }

        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            trained.weights[feature] -=
                LEARNING_RATE * weight_gradients[feature] / (double)training_count;
        }
        trained.bias -= LEARNING_RATE * bias_gradient / (double)training_count;
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            if (!isfinite(trained.weights[feature])) {
                return false;
            }
        }
        if (!isfinite(trained.bias)) {
            return false;
        }
    }

    *output = trained;
    return true;
}

/**
 * @brief 从仓库 CSV 拟合训练统计量和逻辑回归参数。
 * @param path CSV 文件路径。
 * @param output 完整可推理模型状态；失败时保持不变。
 * @return CSV、标准化和训练均成功时为 true。
 */
static bool train_from_csv(const char *path, PersistedModel *output)
{
    if (path == nullptr || output == nullptr) {
        return false;
    }

    Dataset dataset = {0};
    Standardizer standardizer = {0};
    LogisticModel model = {0};
    if (!dataset_load_csv(path, &dataset, nullptr) ||
        !standardizer_fit(&dataset, &standardizer) ||
        !model_train(&dataset, &standardizer, &model)) {
        return false;
    }

    PersistedModel trained = {standardizer, model};
    if (!persisted_model_is_valid(&trained)) {
        return false;
    }
    *output = trained;
    return true;
}

/**
 * @brief 严格解析一个有限数值特征。
 * @param text 以空字符结尾的命令行字段。
 * @param output 有效可写数值；失败时保持不变。
 * @return 整个字符串都是有限数值时为 true。
 */
static bool parse_finite_number(const char *text, double *output)
{
    if (text == nullptr || output == nullptr || text[0] == '\0') {
        return false;
    }
    errno = 0;
    char *end = nullptr;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !isfinite(value)) {
        return false;
    }
    *output = value;
    return true;
}

/**
 * @brief 打印固定格式的预测，方便人工阅读和 C/Python 对照。
 * @param features 原始输入特征。
 * @param prediction 已成功计算的预测结果。
 */
static void print_prediction(const double features[FEATURE_COUNT], const Prediction *prediction)
{
    printf("features=%.17g,%.17g standardized=%.17g,%.17g logit=%.17g "
           "probability=%.17g class=%d\n",
           features[0], features[1], prediction->standardized[0],
           prediction->standardized[1], prediction->logit, prediction->probability,
           prediction->label);
}

/**
 * @brief 运行内存级格式测试，检查损坏数据不会改写已有状态。
 */
static void test_codec(void)
{
    PersistedModel original = {
        .standardizer = {.mean = {2.0, 10.0}, .scale = {2.0, 4.0}},
        .model = {.weights = {0.5, -0.25}, .bias = -0.1}
    };
    uint8_t encoded[MODEL_FILE_BYTES] = {0};
    assert(model_encode(&original, encoded, sizeof(encoded)));

    PersistedModel decoded = {0};
    assert(model_decode(encoded, sizeof(encoded), &decoded));
    assert(decoded.standardizer.mean[0] == 2.0);
    assert(decoded.standardizer.scale[1] == 4.0);
    assert(decoded.model.weights[1] == -0.25);
    assert(decoded.model.bias == -0.1);

    PersistedModel preserved = {
        .standardizer = {.mean = {91.0, 92.0}, .scale = {3.0, 4.0}},
        .model = {.weights = {5.0, 6.0}, .bias = 7.0}
    };
    uint8_t invalid[MODEL_FILE_BYTES];
    memcpy(invalid, encoded, sizeof(invalid));
    invalid[8] = MODEL_FORMAT_VERSION + 1;
    refresh_checksum(invalid);
    assert(!model_decode(invalid, sizeof(invalid), &preserved));
    assert(preserved.standardizer.mean[0] == 91.0);
    assert(preserved.model.bias == 7.0);

    memcpy(invalid, encoded, sizeof(invalid));
    write_u32_le(invalid + 12, FEATURE_COUNT + 1);
    refresh_checksum(invalid);
    assert(!model_decode(invalid, sizeof(invalid), &preserved));
    assert(preserved.standardizer.mean[0] == 91.0);

    assert(!model_decode(encoded, sizeof(encoded) - 1, &preserved));
    assert(preserved.model.bias == 7.0);

    memcpy(invalid, encoded, sizeof(invalid));
    write_u64_le(invalid + MODEL_HEADER_BYTES, UINT64_C(0x7ff0000000000000));
    refresh_checksum(invalid);
    assert(!model_decode(invalid, sizeof(invalid), &preserved));
    assert(preserved.standardizer.mean[0] == 91.0);

    memcpy(invalid, encoded, sizeof(invalid));
    invalid[MODEL_HEADER_BYTES + 3] ^= 1u;
    assert(!model_decode(invalid, sizeof(invalid), &preserved));
    assert(preserved.model.bias == 7.0);

    double invalid_features[FEATURE_COUNT] = {NAN, 4.0};
    Prediction prediction = {.probability = 0.123};
    assert(!model_predict(&original, invalid_features, &prediction));
    assert(prediction.probability == 0.123);
}

/**
 * @brief 用中心差分检查 BCE 对 logit 的解析梯度。
 */
static void test_binary_loss_gradient(void)
{
    const double logit = 0.4;
    const double step = 1e-6;
    for (int label = 0; label <= 1; ++label) {
        double loss = 0.0;
        double analytic_gradient = 0.0;
        assert(binary_loss_gradient(logit, label, &loss, &analytic_gradient));
        assert(loss >= 0.0 && isfinite(loss));

        double loss_before = 0.0;
        double ignored_gradient = 0.0;
        double loss_after = 0.0;
        assert(binary_loss_gradient(logit - step, label, &loss_before, &ignored_gradient));
        assert(binary_loss_gradient(logit + step, label, &loss_after, &ignored_gradient));
        double numeric_gradient = (loss_after - loss_before) / (2.0 * step);
        assert(fabs(numeric_gradient - analytic_gradient) < 1e-8);
    }

    double preserved_loss = 12.0;
    double preserved_gradient = 34.0;
    assert(!binary_loss_gradient(NAN, 1, &preserved_loss, &preserved_gradient));
    assert(preserved_loss == 12.0 && preserved_gradient == 34.0);
    assert(!binary_loss_gradient(0.0, 2, &preserved_loss, &preserved_gradient));
    assert(preserved_loss == 12.0 && preserved_gradient == 34.0);
}

/**
 * @brief 运行编码、版本、维度、截断、校验和与失败原子性断言。
 * @return 全部断言通过时返回 EXIT_SUCCESS。
 */
static int run_tests(void)
{
    assert(wire_format_supported());
    test_codec();
    test_binary_loss_gradient();

    PersistedModel fixture = {0};
    assert(model_load("tests/fixtures/ml/model_v1.bin", &fixture));
    const double features[FEATURE_COUNT] = {4.0, 14.0};
    Prediction prediction = {0};
    assert(model_predict(&fixture, features, &prediction));
    assert(fabs(prediction.standardized[0] - 1.0) < 1e-14);
    assert(fabs(prediction.standardized[1] - 1.0) < 1e-14);
    assert(fabs(prediction.probability - 0.5374298453437496) < 1e-14);
    assert(prediction.label == 1);
    puts("模型持久化 C 断言通过");
    return EXIT_SUCCESS;
}

/**
 * @brief 演示训练状态序列化后再加载，以及独立推理命令的用法。
 * @return 数据读取、拟合、编码、解码和预测均成功时返回 EXIT_SUCCESS。
 */
static int run_demo(void)
{
    PersistedModel trained = {0};
    if (!train_from_csv(DEFAULT_CSV_PATH, &trained)) {
        fprintf(stderr, "无法从 %s 训练模型\n", DEFAULT_CSV_PATH);
        return EXIT_FAILURE;
    }

    uint8_t bytes[MODEL_FILE_BYTES];
    PersistedModel loaded = {0};
    if (!model_encode(&trained, bytes, sizeof(bytes)) ||
        !model_decode(bytes, sizeof(bytes), &loaded)) {
        fputs("模型内存序列化演示失败\n", stderr);
        return EXIT_FAILURE;
    }

    const double features[FEATURE_COUNT] = {5.2, 64.0};
    Prediction prediction = {0};
    if (!model_predict(&loaded, features, &prediction)) {
        fputs("模型推理失败\n", stderr);
        return EXIT_FAILURE;
    }

    puts("训练 -> 保存推理状态 -> 加载 -> 对新样本预测：");
    printf("train_mean=%.6f,%.6f train_scale=%.6f,%.6f weights=%.6f,%.6f bias=%.6f\n",
           loaded.standardizer.mean[0], loaded.standardizer.mean[1],
           loaded.standardizer.scale[0], loaded.standardizer.scale[1],
           loaded.model.weights[0], loaded.model.weights[1], loaded.model.bias);
    print_prediction(features, &prediction);
    puts("持久文件用法：make model-persistence-train MODEL_PATH=sessions/ml-models/study-v1.bin");
    puts("独立进程推理：make model-persistence-predict MODEL_PATH=... FEATURE_0=5.2 FEATURE_1=64");
    return EXIT_SUCCESS;
}

/**
 * @brief 提供演示、独占训练保存和只加载预测三个进程入口。
 * @param argc 命令行参数个数。
 * @param argv 命令、模型路径和两个原始特征。
 * @return 成功返回 EXIT_SUCCESS；任何解析、读写或推理错误返回失败。
 */
int main(int argc, char *argv[])
{
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "demo") == 0)) {
        return run_demo();
    }
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        return run_tests();
    }
    if (argc == 3 && strcmp(argv[1], "train") == 0) {
        PersistedModel trained = {0};
        if (!train_from_csv(DEFAULT_CSV_PATH, &trained)) {
            fprintf(stderr, "无法从 %s 训练模型\n", DEFAULT_CSV_PATH);
            return EXIT_FAILURE;
        }
        if (!model_save_exclusive(argv[2], &trained)) {
            fprintf(stderr, "无法独占保存模型 %s（文件已存在或路径不可写）\n", argv[2]);
            return EXIT_FAILURE;
        }
        printf("已保存版本 %u 模型：%s\n", MODEL_FORMAT_VERSION, argv[2]);
        return EXIT_SUCCESS;
    }
    if (argc == 5 && strcmp(argv[1], "predict") == 0) {
        PersistedModel loaded = {0};
        double features[FEATURE_COUNT];
        if (!model_load(argv[2], &loaded) ||
            !parse_finite_number(argv[3], &features[0]) ||
            !parse_finite_number(argv[4], &features[1])) {
            fprintf(stderr, "模型文件或推理特征无效：%s\n", argv[2]);
            return EXIT_FAILURE;
        }
        Prediction prediction = {0};
        if (!model_predict(&loaded, features, &prediction)) {
            fputs("推理失败：输入或模型参数导致非有限中间值\n", stderr);
            return EXIT_FAILURE;
        }
        print_prediction(features, &prediction);
        return EXIT_SUCCESS;
    }

    fprintf(stderr,
            "用法：%s [demo | train MODEL_PATH | predict MODEL_PATH FEATURE_0 FEATURE_1 | --test]\n",
            argv[0]);
    return EXIT_FAILURE;
}
