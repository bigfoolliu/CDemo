/**
 * @file
 * @brief 阶段 6：纯 C GRU 记忆比特；与同宽 RNN 比较完整 BPTT。
 * 输入为起始比特、无关噪声、结束标记；只在结束处计算二分类损失。
 */
#include "../common/logger.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUTS 5
#define HIDDEN 8
#define MAX_GAP 32
#define MAX_STEPS (MAX_GAP + 2)
#define CAPACITY 128
#define EPOCHS 300

enum {
    WX = 0,
    WH = HIDDEN * INPUTS,
    BIAS = WH + HIDDEN * HIDDEN,
    BLOCK = BIAS + HIDDEN,
    MAX_PARAMETERS = 3 * BLOCK + HIDDEN + 1
};

typedef enum {
    MODEL_RNN,
    MODEL_GRU
} ModelKind;

typedef struct {
    size_t steps;
    unsigned tokens[MAX_STEPS]; /* 0/1 起始比特，2/3 无关噪声，4 结束。 */
    unsigned label;
} Sequence;

typedef struct {
    size_t count;
    Sequence sequences[CAPACITY];
} Dataset;

typedef struct {
    ModelKind kind;
    double parameters[MAX_PARAMETERS]; /* 只访问 parameter_count(kind) 个元素。 */
} Network;

typedef struct {
    double h[MAX_STEPS + 1][HIDDEN];
    double z[MAX_STEPS][HIDDEN];
    double r[MAX_STEPS][HIDDEN];
    double n[MAX_STEPS][HIDDEN];
    double probability;
    double loss;
} Forward;

typedef struct {
    double loss;
    double accuracy;
} Metrics;

typedef struct {
    double first[MAX_PARAMETERS];
    double second[MAX_PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

typedef struct {
    Logger *logger;
    bool failed;
} Output;

/**
 * @brief 返回活动参数数目，尾部未使用元素不参与优化或裁剪。
 * @param kind 有效模型类型。
 * @return RNN 为121，GRU 为345。
 */
static size_t parameter_count(ModelKind kind)
{
    assert(kind == MODEL_RNN || kind == MODEL_GRU);
    return (kind == MODEL_GRU ? 3 * BLOCK : BLOCK) + HIDDEN + 1;
}

/**
 * @brief 报告实验指标，可选追加日志；累计而不丢弃输出错误。
 * @param output 有效可写上下文，借用 logger。
 * @param format 有效格式字符串及匹配的后续参数。
 */
static void report(Output *output, const char *format, ...)
{
    char message[512];
    va_list arguments;
    va_start(arguments, format);
    int length = vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    if (length < 0 || (size_t)length >= sizeof(message)) {
        output->failed = true;
        return;
    }
    if (output->logger != nullptr) {
        logger_write(output->logger, "%s", message);
        output->failed |= output->logger->failed;
    } else if (printf("%s\n", message) < 0) {
        output->failed = true;
    }
}

/**
 * @brief 生成固定容量、标签平衡的独立序列；每次调用按传入种子重建。
 * @param data 有效可写数据集，非法参数时保持不变。
 * @param count 2..CAPACITY 内的偶数序列数。
 * @param gap 0..MAX_GAP 个噪声时间步。
 * @param seed 噪声伪随机种子，不用于安全用途。
 * @return 参数合法时为 true。
 */
static bool dataset_generate(Dataset *data, size_t count, size_t gap, uint32_t seed)
{
    if (count < 2 || count > CAPACITY || count % 2 != 0 || gap > MAX_GAP) {
        return false;
    }
    *data = (Dataset){.count = count};
    for (size_t s = 0; s < count; ++s) {
        Sequence *sequence = &data->sequences[s];
        sequence->label = (unsigned)(s % 2);
        sequence->steps = gap + 2;
        sequence->tokens[0] = sequence->label;
        for (size_t t = 1; t <= gap; ++t) {
            seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
            sequence->tokens[t] = 2 + (seed >> 31);
        }
        sequence->tokens[gap + 1] = 4;
    }
    return true;
}

/**
 * @brief 验证比特任务的长度、字符与监督协议。
 * @param sequence 有效只读序列。
 * @return 合法时为 true，不修改输入。
 */
static bool sequence_valid(const Sequence *sequence)
{
    if (sequence->steps < 2 || sequence->steps > MAX_STEPS || sequence->label > 1 ||
        sequence->tokens[0] != sequence->label || sequence->tokens[sequence->steps - 1] != 4) {
        return false;
    }
    for (size_t t = 1; t + 1 < sequence->steps; ++t) {
        if (sequence->tokens[t] != 2 && sequence->tokens[t] != 3) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 数值稳定的 sigmoid，避免计算极大的正指数。
 * @param value 有限 logit。
 * @return 0..1 范围内的概率。
 */
static double sigmoid(double value)
{
    if (value >= 0) {
        return 1 / (1 + exp(-value));
    }
    double e = exp(value);
    return e / (1 + e);
}

/**
 * @brief 初始化同宽模型；权重使用固定尺度随机值，偏置为零。
 * @param network 有效可写网络，覆盖旧状态。
 * @param kind RNN 或 GRU。
 * @param seed 初始化种子；相同种子不表示两种结构逐参数等价。
 */
static void network_initialize(Network *network, ModelKind kind, uint32_t seed)
{
    *network = (Network){.kind = kind};
    size_t blocks = kind == MODEL_GRU ? 3 : 1;
    for (size_t i = 0; i < parameter_count(kind); ++i) {
        if ((i < blocks * BLOCK && i % BLOCK >= BIAS) || i + 1 == parameter_count(kind)) {
            continue;
        }
        seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
        network->parameters[i] = 0.5 * (2 * (double)(seed >> 8) / 16777216.0 - 1);
    }
}

/**
 * @brief 展开完整序列，GRU 使用 reset-before 形式，z 表示保留旧状态比例。
 * @param network 有效只读有限参数网络。
 * @param sequence 合法只读序列，2..MAX_STEPS 步。
 * @param initial HIDDEN 个只读有限初始值，独立序列使用零。
 * @param forward 有效可写前向缓存，不与其他输入重叠。
 */
static void network_forward(const Network *network, const Sequence *sequence, const double *initial,
                            Forward *forward)
{
    assert(sequence_valid(sequence));
    *forward = (Forward){0};
    memcpy(forward->h[0], initial, HIDDEN * sizeof(double));
    const double *p = network->parameters;
    for (size_t t = 0; t < sequence->steps; ++t) {
        unsigned token = sequence->tokens[t];
        const double *previous = forward->h[t];
        if (network->kind == MODEL_GRU) {
            /* 门必须先完整计算，候选状态的第 i 行会使用所有 r[j]。 */
            for (size_t g = 0; g < 2; ++g) {
                double *gate = g == 0 ? forward->z[t] : forward->r[t];
                for (size_t i = 0; i < HIDDEN; ++i) {
                    size_t base = g * BLOCK;
                    double value = p[base + BIAS + i] + p[base + WX + i * INPUTS + token];
                    for (size_t j = 0; j < HIDDEN; ++j) {
                        value += p[base + WH + i * HIDDEN + j] * previous[j];
                    }
                    gate[i] = sigmoid(value);
                }
            }
        }
        size_t base = network->kind == MODEL_GRU ? 2 * BLOCK : 0;
        for (size_t i = 0; i < HIDDEN; ++i) {
            double value = p[base + BIAS + i] + p[base + WX + i * INPUTS + token];
            for (size_t j = 0; j < HIDDEN; ++j) {
                double reset = network->kind == MODEL_GRU ? forward->r[t][j] : 1;
                value += p[base + WH + i * HIDDEN + j] * reset * previous[j];
            }
            double candidate = tanh(value);
            forward->n[t][i] = candidate;
            double retain = network->kind == MODEL_GRU ? forward->z[t][i] : 0;
            forward->h[t + 1][i] = retain * previous[i] + (1 - retain) * candidate;
        }
    }
    size_t output = parameter_count(network->kind) - HIDDEN - 1;
    double logit = p[output + HIDDEN];
    for (size_t i = 0; i < HIDDEN; ++i) {
        logit += p[output + i] * forward->h[sequence->steps][i];
    }
    forward->probability = sigmoid(logit);
    forward->loss = fmax(logit, 0) - (double)sequence->label * logit + log1p(exp(-fabs(logit)));
    assert(isfinite(forward->loss));
}

/**
 * @brief 最终二分类损失沿时间回传，累加所有门和候选参数的贡献。
 * @param network 有效只读前向参数，完成反向前不可更新。
 * @param sequence 与缓存对应的合法序列。
 * @param forward 借用只读前向缓存。
 * @param gradient MAX_PARAMETERS 个可写值，先清零，只使用活动参数。
 * @param initial_gradient HIDDEN 个可写值，返回对 h[0] 的梯度。
 */
static void network_backward(const Network *network, const Sequence *sequence,
                             const Forward *forward, double *gradient, double *initial_gradient)
{
    memset(gradient, 0, MAX_PARAMETERS * sizeof(double));
    const double *p = network->parameters;
    size_t output = parameter_count(network->kind) - HIDDEN - 1;
    double dz_output = forward->probability - (double)sequence->label;
    double future[HIDDEN];
    gradient[output + HIDDEN] = dz_output;
    for (size_t i = 0; i < HIDDEN; ++i) {
        gradient[output + i] = dz_output * forward->h[sequence->steps][i];
        future[i] = p[output + i] * dz_output;
    }
    for (size_t reverse = sequence->steps; reverse > 0; --reverse) {
        size_t t = reverse - 1;
        const double *previous = forward->h[t];
        double dh[HIDDEN];
        double dr[HIDDEN] = {0};
        double dz[HIDDEN] = {0};
        memcpy(dh, future, sizeof(dh));
        memset(future, 0, sizeof(future));
        size_t base = network->kind == MODEL_GRU ? 2 * BLOCK : 0;
        for (size_t i = 0; i < HIDDEN; ++i) {
            double retain = network->kind == MODEL_GRU ? forward->z[t][i] : 0;
            double n = forward->n[t][i];
            double da = dh[i] * (1 - retain) * (1 - n * n);
            gradient[base + BIAS + i] += da;
            gradient[base + WX + i * INPUTS + sequence->tokens[t]] += da;
            if (network->kind == MODEL_GRU) {
                dz[i] = dh[i] * (previous[i] - n) * retain * (1 - retain);
                future[i] += dh[i] * retain; /* 状态混合的直接保留路径。 */
            }
            for (size_t j = 0; j < HIDDEN; ++j) {
                double reset = network->kind == MODEL_GRU ? forward->r[t][j] : 1;
                gradient[base + WH + i * HIDDEN + j] += da * reset * previous[j];
                future[j] += p[base + WH + i * HIDDEN + j] * da * reset;
                dr[j] += p[base + WH + i * HIDDEN + j] * da * previous[j];
            }
        }
        if (network->kind == MODEL_GRU) {
            for (size_t g = 0; g < 2; ++g) {
                for (size_t i = 0; i < HIDDEN; ++i) {
                    double reset = forward->r[t][i];
                    double da = g == 0 ? dz[i] : dr[i] * reset * (1 - reset);
                    size_t gate = g * BLOCK;
                    gradient[gate + BIAS + i] += da;
                    gradient[gate + WX + i * INPUTS + sequence->tokens[t]] += da;
                    for (size_t j = 0; j < HIDDEN; ++j) {
                        gradient[gate + WH + i * HIDDEN + j] += da * previous[j];
                        future[j] += p[gate + WH + i * HIDDEN + j] * da;
                    }
                }
            }
        }
    }
    memcpy(initial_gradient, future, sizeof(future));
}

/**
 * @brief 按序列平均终点损失，每条序列从零状态开始。
 * @param network 有效只读网络。
 * @param data 非空合法只读数据集。
 * @return 平均 BCE 与终点分类准确率，阈值为0.5。
 */
static Metrics evaluate(const Network *network, const Dataset *data)
{
    assert(data->count > 0 && data->count <= CAPACITY);
    Metrics result = {0};
    const double zero[HIDDEN] = {0};
    for (size_t s = 0; s < data->count; ++s) {
        Forward forward;
        network_forward(network, &data->sequences[s], zero, &forward);
        result.loss += forward.loss;
        result.accuracy += (forward.probability >= 0.5) == data->sequences[s].label;
    }
    result.loss /= (double)data->count;
    result.accuracy /= (double)data->count;
    return result;
}

/**
 * @brief 对全部参数梯度做一个全局 L2 范数裁剪，保留梯度方向。
 * @param gradient count 个可写值；非有限值导致失败且不修改。
 * @param count 有效活动参数长度，不超过 MAX_PARAMETERS。
 * @param limit 有限正阈值。
 * @param norm 有效可写指针，成功时保存裁剪前范数。
 * @return 合法且范数可表示时为 true；失败不修改梯度和 norm。
 */
static bool clip_gradient(double *gradient, size_t count, double limit, double *norm)
{
    assert(count > 0 && count <= MAX_PARAMETERS);
    if (!isfinite(limit) || limit <= 0) {
        return false;
    }
    double value = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(gradient[i])) {
            return false;
        }
        value = hypot(value, gradient[i]);
    }
    if (!isfinite(value)) {
        return false;
    }
    if (value > limit) {
        double scale = limit / value;
        for (size_t i = 0; i < count; ++i) {
            gradient[i] *= scale;
        }
    }
    *norm = value;
    return true;
}

/**
 * @brief 使用裁剪后的梯度更新 Adam；先检查候选结果再统一提交。
 * @param network 有效可写网络。
 * @param adam 有效可写优化器，初始两个 beta 幂为 1，其余为零。
 * @param gradient parameter_count(network->kind) 个只读有限梯度。
 * @return 固定学习率 0.01 的更新有限时为 true，失败不修改参数或优化器。
 */
static bool adam_update(Network *network, Adam *adam, const double *gradient)
{
    Network next = *network;
    Adam state = *adam;
    state.beta1_power *= 0.9;
    state.beta2_power *= 0.999;
    for (size_t i = 0; i < parameter_count(network->kind); ++i) {
        double g = gradient[i];
        state.first[i] = 0.9 * state.first[i] + 0.1 * g;
        state.second[i] = 0.999 * state.second[i] + 0.001 * g * g;
        double first = state.first[i] / (1 - state.beta1_power);
        double second = state.second[i] / (1 - state.beta2_power);
        next.parameters[i] -= 0.01 * first / (sqrt(second) + 1e-8);
        if (!isfinite(next.parameters[i]) || !isfinite(state.first[i]) ||
            !isfinite(state.second[i])) {
            return false;
        }
    }
    *network = next;
    *adam = state;
    return true;
}

/**
 * @brief 相同数据预算训练两种模型，使用验证集选择检查点。
 * @param network 有效可写模型，成功返回最佳验证损失对应的参数。
 * @param gap 0..MAX_GAP 的噪声长度。
 * @param valid 只读独立验证集，固定且不参与梯度。
 * @param output 可选输出上下文，空时省略训练报告。
 * @return 所有更新有限时为 true，失败保留最近一次成功更新。
 */
static bool train(Network *network, size_t gap, const Dataset *valid, Output *output)
{
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    Network best = *network;
    double best_loss = evaluate(network, valid).loss;
    size_t best_epoch = 0;
    size_t count = parameter_count(network->kind);
    const double zero[HIDDEN] = {0};
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        Dataset batch;
        /* 每轮新噪声；两模型每轮使用相同种子和32个样本，不读取测试集。 */
        assert(dataset_generate(&batch, 32, gap, (uint32_t)(4000 + epoch)));
        double gradient[MAX_PARAMETERS] = {0};
        for (size_t s = 0; s < batch.count; ++s) {
            Forward forward;
            double single[MAX_PARAMETERS];
            double dh0[HIDDEN];
            network_forward(network, &batch.sequences[s], zero, &forward);
            network_backward(network, &batch.sequences[s], &forward, single, dh0);
            for (size_t i = 0; i < count; ++i) {
                gradient[i] += single[i] / (double)batch.count;
            }
        }
        double norm = 0;
        if (!clip_gradient(gradient, count, 1, &norm) || !adam_update(network, &adam, gradient)) {
            return false;
        }
        Metrics metrics = evaluate(network, valid);
        if (metrics.loss < best_loss) {
            best_loss = metrics.loss;
            best = *network;
            best_epoch = epoch + 1;
        }
        if (output != nullptr && (epoch == 0 || epoch + 1 == EPOCHS)) {
            report(output, "%s gap=%zu epoch=%zu train_BCE=%.5f val_BCE=%.5f grad_norm=%.5f",
                   network->kind == MODEL_GRU ? "GRU" : "RNN", gap, epoch + 1,
                   evaluate(network, &batch).loss, metrics.loss, norm);
        }
    }
    *network = best;
    if (output != nullptr) {
        report(output, "验证检查点：epoch=%zu BCE=%.5f。", best_epoch, best_loss);
    }
    return true;
}

/**
 * @brief 显示两条序列在起始、间隔中点和结束处的状态与门均值。
 * @param network 有效只读已训练模型。
 * @param data 至少两条合法序列的只读数据集。
 * @param output 有效可写输出上下文。
 */
static void inspect_memory(const Network *network, const Dataset *data, Output *output)
{
    const double zero[HIDDEN] = {0};
    for (size_t s = 0; s < 2; ++s) {
        const Sequence *sequence = &data->sequences[s];
        Forward forward;
        network_forward(network, sequence, zero, &forward);
        report(output, "观察 bit=%u，结束处 P(bit=1)=%.5f", sequence->label, forward.probability);
        const size_t positions[] = {0, sequence->steps / 2, sequence->steps - 1};
        for (size_t k = 0; k < sizeof(positions) / sizeof(positions[0]); ++k) {
            size_t t = positions[k];
            double z = 0;
            double r = 0;
            for (size_t i = 0; i < HIDDEN; ++i) {
                z += forward.z[t][i] / HIDDEN;
                r += forward.r[t][i] / HIDDEN;
            }
            if (network->kind == MODEL_GRU) {
                report(output, "t=%zu token=%u h[0]=%+.4f mean_z=%.4f mean_r=%.4f", t,
                       sequence->tokens[t], forward.h[t + 1][0], z, r);
            } else {
                report(output, "t=%zu token=%u h[0]=%+.4f", t, sequence->tokens[t],
                       forward.h[t + 1][0]);
            }
        }
    }
}

/**
 * @brief 三步时间展开验证所有活动参数与初始状态导数；包含全部门。
 * @param kind RNN 或 GRU，两种实现分别验证。
 */
static void test_gradients(ModelKind kind)
{
    Network network;
    network_initialize(&network, kind, 42);
    Dataset data;
    assert(dataset_generate(&data, 2, 1, 2025));
    double initial[HIDDEN];
    for (size_t i = 0; i < HIDDEN; ++i) {
        initial[i] = 0.03 * (double)(i + 1);
    }
    const Sequence *sequence = &data.sequences[1];
    Forward forward;
    double gradient[MAX_PARAMETERS];
    double dh0[HIDDEN];
    network_forward(&network, sequence, initial, &forward);
    network_backward(&network, sequence, &forward, gradient, dh0);
    size_t count = parameter_count(kind);
    const double epsilon = 1e-5;
    for (size_t i = 0; i < count + HIDDEN; ++i) {
        double *value = i < count ? &network.parameters[i] : &initial[i - count];
        double analytical = i < count ? gradient[i] : dh0[i - count];
        double saved = *value;
        *value = saved + epsilon;
        network_forward(&network, sequence, initial, &forward);
        double plus = forward.loss;
        *value = saved - epsilon;
        network_forward(&network, sequence, initial, &forward);
        double minus = forward.loss;
        *value = saved;
        double numerical = (plus - minus) / (2 * epsilon);
        assert(fabs(numerical - analytical) < 1e-7 + 1e-4 * fabs(numerical));
    }
}

/**
 * @brief 验证可手算的门、状态保留、输入边界和失败不部分修改。
 */
static void test_boundaries(void)
{
    Dataset data;
    assert(dataset_generate(&data, 2, 0, 2025));
    assert(!dataset_generate(&data, 1, 0, 2025) && data.count == 2);
    assert(!dataset_generate(&data, CAPACITY + 2, 0, 2025));
    assert(!dataset_generate(&data, 2, MAX_GAP + 1, 2025));
    assert(sequence_valid(&data.sequences[0]));
    Sequence invalid = data.sequences[0];
    invalid.steps = MAX_STEPS + 1;
    assert(!sequence_valid(&invalid));
    invalid = data.sequences[0];
    invalid.tokens[1] = 5;
    assert(!sequence_valid(&invalid));
    invalid = data.sequences[0];
    invalid.label = 2;
    assert(!sequence_valid(&invalid));

    Network network = {.kind = MODEL_GRU};
    double initial[HIDDEN] = {0.8};
    Forward forward;
    network_forward(&network, &data.sequences[0], initial, &forward);
    /* 零参数时 z=r=0.5，n=0，每步 h 减半，两步后0.2。 */
    assert(forward.z[0][0] == 0.5 && forward.r[0][0] == 0.5);
    assert(fabs(forward.h[1][0] - 0.4) < 1e-12);
    assert(fabs(forward.h[2][0] - 0.2) < 1e-12);
    assert(fabs(forward.loss - log(2)) < 1e-12);
    network.parameters[BIAS] = 1000;
    network_forward(&network, &data.sequences[0], initial, &forward);
    assert(forward.h[2][0] == initial[0]); /* z≈1 保留旧状态，不保证任意网络都能长期记忆。 */
    network.parameters[BIAS] = -1000;
    network_forward(&network, &data.sequences[0], initial, &forward);
    assert(forward.h[2][0] == 0);
    network.parameters[3 * BLOCK + HIDDEN] = 1000;
    network_forward(&network, &data.sequences[0], initial, &forward);
    assert(isfinite(forward.loss) && forward.loss > 999);
    const double zero[HIDDEN] = {0};
    assert(dataset_generate(&data, 2, MAX_GAP, 2026));
    network_forward(&network, &data.sequences[1], zero, &forward);
    Forward repeated;
    network_forward(&network, &data.sequences[1], zero, &repeated);
    assert(memcmp(forward.h, repeated.h, sizeof(forward.h)) == 0);
    double longest_gradient[MAX_PARAMETERS];
    double initial_gradient[HIDDEN];
    network_backward(&network, &data.sequences[1], &forward, longest_gradient, initial_gradient);
    for (size_t i = 0; i < MAX_PARAMETERS; ++i) {
        assert(isfinite(longest_gradient[i]));
    }

    double gradient[MAX_PARAMETERS] = {3, 4};
    double norm = -1;
    assert(clip_gradient(gradient, MAX_PARAMETERS, 1, &norm) && fabs(norm - 5) < 1e-12);
    assert(fabs(gradient[0] - 0.6) < 1e-12 && fabs(gradient[1] - 0.8) < 1e-12);
    double saved = gradient[0];
    gradient[MAX_PARAMETERS - 1] = NAN;
    assert(!clip_gradient(gradient, MAX_PARAMETERS, 1, &norm) && gradient[0] == saved);
    assert(!clip_gradient(gradient, MAX_PARAMETERS, 0, &norm));
    memset(gradient, 0, sizeof(gradient));
    assert(clip_gradient(gradient, MAX_PARAMETERS, 1, &norm) && norm == 0);
    Network before = network;
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    gradient[MAX_PARAMETERS - 1] = INFINITY;
    assert(!adam_update(&network, &adam, gradient));
    assert(memcmp(network.parameters, before.parameters, sizeof(network.parameters)) == 0);
    assert(adam.beta1_power == 1 && adam.first[0] == 0);
}

/**
 * @brief 对不同间隔和种子进行相同预算比较；不预设长间隔胜者。
 * @param argc 参数数量。
 * @param argv 可选 --test 或 --log，借用而不释放。
 * @return 数值验证、训练和输出成功为 EXIT_SUCCESS，否则失败。
 */
int main(int argc, char **argv)
{
    bool quiet = argc == 2 && strcmp(argv[1], "--test") == 0;
    bool logging = argc == 2 && strcmp(argv[1], "--log") == 0;
    if (argc != 1 && !quiet && !logging) {
        fprintf(stderr, "用法：%s [--test | --log]\n", argv[0]);
        return EXIT_FAILURE;
    }
    Logger logger = {0};
    if (logging && !logger_open(&logger, "logs/ml/11_gru_memory.log", stdout)) {
        fprintf(stderr, "GRU 日志打开失败：%s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    Output output = {.logger = logging ? &logger : nullptr};
    bool success = true;
    test_gradients(MODEL_RNN);
    test_gradients(MODEL_GRU);
    test_boundaries();
    const size_t gaps[] = {2, 8, 24};
    const uint32_t seeds[] = {42, 123};
    if (!quiet) {
        report(&output, "=== 记忆比特：5种输入、8维状态；RNN121/GRU345个参数 ===");
        report(&output,
               "每组300轮×32序列；Adam0.01，裁剪1；训练噪声4000+轮号，验证2026，测试2027。");
        report(&output, "只在结束标记计算损失，独立序列重置状态；多数类基线准确率50%%、BCE=%.5f。",
               log(2));
    }
    for (size_t g = 0; g < sizeof(gaps) / sizeof(gaps[0]); ++g) {
        Dataset valid;
        Dataset testing;
        assert(dataset_generate(&valid, 32, gaps[g], 2026));
        assert(dataset_generate(&testing, 128, gaps[g], 2027));
        for (size_t s = 0; s < sizeof(seeds) / sizeof(seeds[0]); ++s) {
            for (ModelKind kind = MODEL_RNN; kind <= MODEL_GRU; ++kind) {
                Network network;
                network_initialize(&network, kind, seeds[s]);
                if (!quiet) {
                    report(&output, "开始 %s gap=%zu seed=%u", kind == MODEL_GRU ? "GRU" : "RNN",
                           gaps[g], (unsigned)seeds[s]);
                }
                if (!train(&network, gaps[g], &valid, quiet ? nullptr : &output)) {
                    fputs("记忆模型更新失败：非有限数值\n", stderr);
                    success = false;
                    goto cleanup;
                }
                Metrics result = evaluate(&network, &testing);
                assert(isfinite(result.loss) && result.accuracy >= 0 && result.accuracy <= 1);
                if (gaps[g] == 2) {
                    assert(result.loss < 0.1 && result.accuracy >= 0.95);
                }
                report(&output, "%s gap=%zu seed=%u test_BCE=%.5f accuracy=%.2f%%",
                       kind == MODEL_GRU ? "GRU" : "RNN", gaps[g], (unsigned)seeds[s], result.loss,
                       100 * result.accuracy);
                if (!quiet && gaps[g] == 24 && seeds[s] == 42) {
                    inspect_memory(&network, &testing, &output);
                }
            }
        }
    }
    report(&output, "阶段 6 验证通过：RNN/GRU全部参数与初始状态梯度、门控边界和同预算记忆实验。");
cleanup:
    if (logging && !logger_close(&logger)) {
        fputs("GRU 日志写入或关闭失败\n", stderr);
        success = false;
    }
    if (fflush(stdout) != 0) {
        output.failed = true;
    }
    return success && !output.failed ? EXIT_SUCCESS : EXIT_FAILURE;
}
