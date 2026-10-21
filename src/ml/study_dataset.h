/**
 * @file
 * @brief CSV 学习案例共用的数据划分、读取和训练集标准化工具。
 */
#ifndef STUDY_DATASET_H
#define STUDY_DATASET_H

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FEATURE_COUNT = 2,
    MAX_SAMPLE_COUNT = 64,
    CSV_LINE_CAPACITY = 256
};

static const char *const DEFAULT_CSV_PATH = "data/ml/study_outcomes.csv";

typedef enum {
    SPLIT_TRAIN,
    SPLIT_VALIDATION,
    SPLIT_TEST,
    SPLIT_COUNT
} DataSplit;

typedef enum {
    CSV_READ_ERROR = -1,
    CSV_READ_EOF = 0,
    CSV_READ_RECORD = 1
} CsvReadStatus;

typedef struct {
    double features[FEATURE_COUNT];
    int label;
    DataSplit split;
} Sample;

typedef struct {
    Sample samples[MAX_SAMPLE_COUNT];
    size_t count;
    size_t split_counts[SPLIT_COUNT];
} Dataset;

typedef struct {
    double mean[FEATURE_COUNT];
    double scale[FEATURE_COUNT];
} Standardizer;

/**
 * @brief 读取一条有限长度 CSV 记录，并移除行尾换行符。
 * @param stream 已打开的只读输入流。
 * @param line 至少 CSV_LINE_CAPACITY 字节的可写缓冲区。
 * @return 记录、文件结束或读取/长度错误。
 */
static CsvReadStatus csv_read_record(FILE *stream, char line[CSV_LINE_CAPACITY])
{
    if (stream == nullptr || line == nullptr) {
        return CSV_READ_ERROR;
    }

    if (fgets(line, CSV_LINE_CAPACITY, stream) == nullptr) {
        return ferror(stream) ? CSV_READ_ERROR : CSV_READ_EOF;
    }

    if (strchr(line, '\n') == nullptr) {
        int next_character = fgetc(stream);
        if (next_character != EOF || ferror(stream)) {
            return CSV_READ_ERROR;
        }
    }

    line[strcspn(line, "\r\n")] = '\0';
    return CSV_READ_RECORD;
}

/**
 * @brief 将一个 CSV 字段严格解析为有限实数。
 * @param field 以空字符结尾的字段文本。
 * @param output 有效可写结果；失败时保持不变。
 * @return 完整字段是有限且可表示的实数时为 true。
 */
static bool parse_finite_double(const char *field, double *output)
{
    if (field == nullptr || output == nullptr || field[0] == '\0') {
        return false;
    }

    errno = 0;
    char *end = nullptr;
    double value = strtod(field, &end);
    if (end == field || *end != '\0' || errno == ERANGE || !isfinite(value)) {
        return false;
    }

    *output = value;
    return true;
}

/**
 * @brief 将 CSV 文本解析为一个样本，拒绝缺列、多列和非法标签。
 * @param line 可写的四字段文本，内容会被逗号分隔符修改。
 * @param output 有效可写样本；失败时保持不变。
 * @return 两个特征、0/1 标签和已知数据划分均有效时为 true。
 */
static bool parse_sample_record(char *line, Sample *output)
{
    if (line == nullptr || output == nullptr) {
        return false;
    }

    char *fields[4] = {nullptr};
    char *field = line;
    for (size_t index = 0; index < 4; ++index) {
        char *comma = strchr(field, ',');
        if (index < 3) {
            if (comma == nullptr) {
                return false;
            }
            *comma = '\0';
            fields[index] = field;
            field = comma + 1;
        } else {
            if (comma != nullptr) {
                return false;
            }
            fields[index] = field;
        }
    }

    Sample parsed = {0};
    if (!parse_finite_double(fields[0], &parsed.features[0]) ||
        !parse_finite_double(fields[1], &parsed.features[1])) {
        return false;
    }

    errno = 0;
    char *label_end = nullptr;
    long label = strtol(fields[2], &label_end, 10);
    if (label_end == fields[2] || *label_end != '\0' || errno == ERANGE ||
        (label != 0 && label != 1)) {
        return false;
    }
    parsed.label = (int)label;

    if (strcmp(fields[3], "train") == 0) {
        parsed.split = SPLIT_TRAIN;
    } else if (strcmp(fields[3], "validation") == 0) {
        parsed.split = SPLIT_VALIDATION;
    } else if (strcmp(fields[3], "test") == 0) {
        parsed.split = SPLIT_TEST;
    } else {
        return false;
    }

    *output = parsed;
    return true;
}

/**
 * @brief 检查数据集包含三种划分，且训练集同时包含两个类别。
 * @param dataset 待检查的只读数据集。
 * @return 样本数、划分计数及训练标签满足模型训练前提时为 true。
 */
static bool dataset_is_valid(const Dataset *dataset)
{
    if (dataset == nullptr || dataset->count == 0 || dataset->count > MAX_SAMPLE_COUNT) {
        return false;
    }

    size_t observed_split_counts[SPLIT_COUNT] = {0};
    size_t training_classes[2] = {0};
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->label != 0 && sample->label != 1) {
            return false;
        }
        if (sample->split < SPLIT_TRAIN || sample->split >= SPLIT_COUNT) {
            return false;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            if (!isfinite(sample->features[feature])) {
                return false;
            }
        }
        ++observed_split_counts[sample->split];
        if (sample->split == SPLIT_TRAIN) {
            ++training_classes[sample->label];
        }
    }

    for (size_t split = 0; split < SPLIT_COUNT; ++split) {
        if (observed_split_counts[split] == 0 ||
            observed_split_counts[split] != dataset->split_counts[split]) {
            return false;
        }
    }

    return training_classes[0] > 0 && training_classes[1] > 0;
}

/**
 * @brief 从流中读取固定列名的 CSV，并在完整验证后替换输出数据集。
 * @param stream 已打开的只读流；格式为两个数值特征、0/1 标签和划分名。
 * @param output 有效可写数据集；任何失败都不修改它。
 * @param error_line 可选错误行号输出；成功时写 0，表头行为第 1 行。
 * @return CSV 完整读取且训练、验证、测试划分均有效时为 true。
 */
static bool dataset_load_stream(FILE *stream, Dataset *output, size_t *error_line)
{
    if (stream == nullptr || output == nullptr) {
        return false;
    }

    Dataset loaded = {0};
    char line[CSV_LINE_CAPACITY];
    size_t line_number = 1;
    CsvReadStatus status = csv_read_record(stream, line);
    if (status != CSV_READ_RECORD ||
        strcmp(line, "study_hours,practice_questions,passed,split") != 0) {
        if (error_line != nullptr) {
            *error_line = line_number;
        }
        return false;
    }

    for (;;) {
        status = csv_read_record(stream, line);
        if (status == CSV_READ_EOF) {
            break;
        }
        ++line_number;
        if (status != CSV_READ_RECORD || loaded.count >= MAX_SAMPLE_COUNT) {
            if (error_line != nullptr) {
                *error_line = line_number;
            }
            return false;
        }

        Sample sample = {0};
        if (!parse_sample_record(line, &sample)) {
            if (error_line != nullptr) {
                *error_line = line_number;
            }
            return false;
        }

        loaded.samples[loaded.count] = sample;
        ++loaded.count;
        ++loaded.split_counts[sample.split];
    }

    if (!dataset_is_valid(&loaded)) {
        if (error_line != nullptr) {
            *error_line = line_number;
        }
        return false;
    }

    *output = loaded;
    if (error_line != nullptr) {
        *error_line = 0;
    }
    return true;
}

/**
 * @brief 独占读取 CSV 文件；关闭成功后才提交解析结果。
 * @param path CSV 文件路径。
 * @param output 有效可写数据集；失败时保持不变。
 * @param error_line 可选错误行号输出；文件打开或关闭失败时写 0。
 * @return 文件可读、格式有效且关闭成功时为 true。
 */
static bool dataset_load_csv(const char *path, Dataset *output, size_t *error_line)
{
    if (path == nullptr || output == nullptr) {
        return false;
    }

    FILE *stream = fopen(path, "r");
    if (stream == nullptr) {
        if (error_line != nullptr) {
            *error_line = 0;
        }
        return false;
    }

    Dataset loaded = {0};
    bool parsed = dataset_load_stream(stream, &loaded, error_line);
    int close_result = fclose(stream);
    if (!parsed || close_result != 0) {
        return false;
    }

    *output = loaded;
    return true;
}

/**
 * @brief 只用训练划分拟合每个特征的均值和总体标准差。
 * @param dataset 含有有效训练、验证和测试划分的数据集。
 * @param output 有效可写标准化参数；失败时保持不变。
 * @return 训练统计量有限且每个特征方差非零时为 true。
 */
static bool standardizer_fit(const Dataset *dataset, Standardizer *output)
{
    if (!dataset_is_valid(dataset) || output == nullptr) {
        return false;
    }

    Standardizer fitted = {0};
    size_t training_count = dataset->split_counts[SPLIT_TRAIN];
    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != SPLIT_TRAIN) {
            continue;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            fitted.mean[feature] += sample->features[feature];
        }
    }

    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        fitted.mean[feature] /= (double)training_count;
    }

    for (size_t index = 0; index < dataset->count; ++index) {
        const Sample *sample = &dataset->samples[index];
        if (sample->split != SPLIT_TRAIN) {
            continue;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double difference = sample->features[feature] - fitted.mean[feature];
            fitted.scale[feature] += difference * difference;
        }
    }

    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        fitted.scale[feature] = sqrt(fitted.scale[feature] / (double)training_count);
        if (!isfinite(fitted.mean[feature]) || !isfinite(fitted.scale[feature]) ||
            fitted.scale[feature] <= DBL_EPSILON) {
            return false;
        }
    }

    *output = fitted;
    return true;
}

/**
 * @brief 使用训练集统计量变换单个样本，不改动原始 CSV 数据。
 * @param standardizer 只读训练集均值和尺度。
 * @param sample 含有限原始特征的只读样本。
 * @param output 至少 FEATURE_COUNT 个可写数值；失败时保持不变。
 * @return 变换结果均为有限数时为 true。
 */
static bool standardize_sample(const Standardizer *standardizer, const Sample *sample,
                               double output[FEATURE_COUNT])
{
    if (standardizer == nullptr || sample == nullptr || output == nullptr) {
        return false;
    }

    double transformed[FEATURE_COUNT];
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(sample->features[feature]) || !isfinite(standardizer->mean[feature]) ||
            !isfinite(standardizer->scale[feature]) || standardizer->scale[feature] <= 0.0) {
            return false;
        }
        transformed[feature] = (sample->features[feature] - standardizer->mean[feature]) /
                               standardizer->scale[feature];
        if (!isfinite(transformed[feature])) {
            return false;
        }
    }

    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        output[feature] = transformed[feature];
    }
    return true;
}

#endif
