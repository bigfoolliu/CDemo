/**
 * @file
 * @brief 阶段 5：纯 C 字符 RNN、完整 BPTT 与全局梯度裁剪。
 * 字符表为 a/b/c/空格；时间步共享参数，独立序列的隐藏状态从零开始。
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

/* 字符表大小、隐藏状态宽度、单条序列步数上限和训练轮数。 */
#define VOCAB 4
#define HIDDEN 16
#define MAX_STEPS 48
#define EPOCHS 600

/* 参数按层平铺；以下枚举值是矩阵和偏置块的起始位置。 */
enum {
    WX = 0,
    WH = WX + HIDDEN * VOCAB,
    BH = WH + HIDDEN * HIDDEN,
    WY = BH + HIDDEN,
    BY = WY + VOCAB * HIDDEN,
    PARAMETERS = BY + VOCAB
};

static const char alphabet[] = "abc ";
/* 每个字符串是一条独立序列；分组在构造训练窗口之前确定。 */
static const char *const training[] = {
    "abac acbc babc cbac abac babc ", "acbc cbac abac babc acbc abac ",
    "babc abac cbac acbc babc cbac ", "cbac babc acbc abac cbac acbc ",
    "abac abac acbc acbc cbac babc ", "babc babc cbac cbac abac acbc ",
    "acbc babc abac cbac acbc cbac ", "cbac acbc babc abac cbac abac "};
static const char *const validation[] = {"abac cbac babc acbc abac acbc ",
                                         "acbc abac babc cbac acbc babc "};
static const char *const testing[] = {"babc acbc cbac abac babc abac ",
                                      "cbac abac acbc babc cbac babc "};

/* 一条序列包含 T 个输入和 T 个“预测下一个字符”的目标。 */
typedef struct {
    size_t steps;
    unsigned input[MAX_STEPS];
    unsigned target[MAX_STEPS];
} Sequence;

/* 所有时间步共享同一组循环网络参数。 */
typedef struct {
    double parameters[PARAMETERS];
} Network;

typedef struct {
    /* h[0] 是初始状态，h[t+1] 是读取 input[t] 后的状态。 */
    double h[MAX_STEPS + 1][HIDDEN];
    /* network_forward 返回后，这里保存每步归一化的下一字符概率。 */
    double probability[MAX_STEPS][VOCAB];
    double loss;
} Forward;

/* 按目标字符总数汇总的交叉熵和准确率。 */
typedef struct {
    double loss;
    double accuracy;
} Metrics;

/* Adam 的参数级一、二阶矩，以及用于偏差修正的 beta 幂。 */
typedef struct {
    double first[PARAMETERS];
    double second[PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

/* logger 由 main 持有；这里仅借用它，并记住任意一次输出失败。 */
typedef struct {
    Logger *logger;
    bool failed;
} Output;

/**
 * @brief 输出短报告，可选同时追加项目日志，并累计输出错误。
 * @param output 可写上下文，借用 logger；格式化或写入失败会置 failed。
 * @param format 有效 printf 格式；后续可变参数须匹配，结果须小于 512 字节。
 */
static void report(Output *output, const char *format, ...)
{
    /* 固定缓冲区让报告路径简单可控；超长格式化结果按失败处理。 */
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
        /* logger 会累计文件或终端写入错误，向 main 传播失败状态。 */
        output->failed |= output->logger->failed;
    } else if (printf("%s\n", message) < 0) {
        output->failed = true;
    }
}

/**
 * @brief 将字符串变为输入/下一字符标签，先验证再提交。
 * @param sequence 有效可写序列，失败时不修改。
 * @param text 非空、以零结尾的借用字符串，长度 2..MAX_STEPS+1。
 * @return 仅含 alphabet 中字符且长度合法时为 true。
 */
static bool sequence_encode(Sequence *sequence, const char *text)
{
    if (text == nullptr) {
        return false;
    }
    size_t length = strlen(text);
    if (length < 2 || length > MAX_STEPS + 1) {
        return false;
    }
    Sequence encoded = {.steps = length - 1};
    /* input 保留最后一个字符之前的内容，target 从第二个字符开始对齐。 */
    for (size_t t = 0; t < length; ++t) {
        const char *position = strchr(alphabet, text[t]);
        if (position == nullptr) {
            return false;
        }
        unsigned token = (unsigned)(position - alphabet);
        if (t < encoded.steps) {
            encoded.input[t] = token;
        }
        if (t > 0) {
            encoded.target[t - 1] = token;
        }
    }
    *sequence = encoded;
    return true;
}

/**
 * @brief 固定种子初始化权重，偏置为零，打破隐藏神经元的对称性。
 * @param network 有效可写网络，旧参数覆盖。
 * @param seed 初始化伪随机状态，不用于安全用途。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    /* 权重使用有界伪随机值；跳过偏置区间，因此偏置保持为零。 */
    for (size_t i = 0; i < PARAMETERS; ++i) {
        if ((i >= BH && i < WY) || i >= BY) {
            continue;
        }
        seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
        network->parameters[i] = 0.4 * (2 * (double)(seed >> 8) / 16777216.0 - 1);
    }
}

/**
 * @brief 计算一个时间步的 tanh 状态与输出 logits，返回稳定归一化常数。
 * @param network 有效只读有限参数网络。
 * @param token 输入字符编号，必须小于 VOCAB。
 * @param previous HIDDEN 个只读隐藏值，不与 next 重叠。
 * @param next HIDDEN 个可写隐藏值，不与 previous 或 probability 重叠。
 * @param probability VOCAB 个 logits 输出；与 next 不重叠，随后原地归一化。
 * @return log-sum-exp，供计算稳定交叉熵。
 */
static double forward_step(const Network *network, unsigned token, const double *previous,
                           double *next, double *probability)
{
    assert(token < VOCAB);
    const double *p = network->parameters;
    for (size_t i = 0; i < HIDDEN; ++i) {
        /* one-hot 乘法等价于读取 W_x 的第 token 列。 */
        double z = p[BH + i] + p[WX + i * VOCAB + token];
        for (size_t j = 0; j < HIDDEN; ++j) {
            z += p[WH + i * HIDDEN + j] * previous[j];
        }
        next[i] = tanh(z);
    }
    /* logit 是未归一化分数；先减最大值再指数化可避免 exp 溢出。 */
    double maximum = -INFINITY;
    for (size_t k = 0; k < VOCAB; ++k) {
        double z = p[BY + k];
        for (size_t i = 0; i < HIDDEN; ++i) {
            z += p[WY + k * HIDDEN + i] * next[i];
        }
        probability[k] = z; /* 暂存 logits，调用方随后归一化。 */
        maximum = fmax(maximum, z);
    }
    double sum = 0;
    for (size_t k = 0; k < VOCAB; ++k) {
        sum += exp(probability[k] - maximum);
    }
    double log_sum = maximum + log(sum);
    assert(isfinite(log_sum));
    return log_sum;
}

/**
 * @brief 展开完整序列，保存反向传播需要的状态，返回按字符平均损失。
 * @param network 有效只读有限参数网络。
 * @param sequence 有效编码序列，1..MAX_STEPS 个时间步。
 * @param initial HIDDEN 个只读有限初始状态，独立序列使用全零。
 * @param forward 有效可写缓存，旧内容覆盖；不与输入重叠。
 */
static void network_forward(const Network *network, const Sequence *sequence, const double *initial,
                            Forward *forward)
{
    assert(sequence->steps >= 1 && sequence->steps <= MAX_STEPS);
    *forward = (Forward){0};
    memcpy(forward->h[0], initial, HIDDEN * sizeof(double));
    /* 每条序列从给定初态展开；训练和评估使用全零初态。 */
    for (size_t t = 0; t < sequence->steps; ++t) {
        assert(sequence->target[t] < VOCAB);
        double *probability = forward->probability[t];
        double log_sum = forward_step(network, sequence->input[t], forward->h[t], forward->h[t + 1],
                                      probability);
        /* 此时 probability 仍是 logits；目标损失先用 log-sum-exp 稳定计算。 */
        forward->loss += log_sum - probability[sequence->target[t]];
        for (size_t k = 0; k < VOCAB; ++k) {
            probability[k] = exp(probability[k] - log_sum);
        }
    }
    forward->loss /= (double)sequence->steps;
    assert(isfinite(forward->loss));
}

/**
 * @brief 完整 BPTT：从最后时刻回传并累加所有时间步对共享参数的贡献。
 * @param network 有效只读前向参数，计算结束前不得更新。
 * @param sequence 与 forward 对应的有效序列。
 * @param forward 借用只读前向缓存。
 * @param gradient PARAMETERS 个可写梯度，覆盖后保存平均损失的导数。
 * @param initial_gradient HIDDEN 个可写 h[0] 梯度；不与 gradient 重叠。
 */
static void network_backward(const Network *network, const Sequence *sequence,
                             const Forward *forward, double *gradient, double *initial_gradient)
{
    /* 共享参数需累加各时间步梯度，因此先整体清零。 */
    memset(gradient, 0, PARAMETERS * sizeof(double));
    /* 保存后续时间步传回来的隐藏状态梯度，最后得到对 h[0] 的导数。 */
    double future[HIDDEN] = {0};
    const double *p = network->parameters;
    for (size_t reverse = sequence->steps; reverse > 0; --reverse) {
        size_t t = reverse - 1;
        double dh[HIDDEN];
        memcpy(dh, future, sizeof(dh));
        /* softmax 与交叉熵合并后的 logit 梯度为 (概率 - one-hot) / 步数。 */
        for (size_t k = 0; k < VOCAB; ++k) {
            double dz = (forward->probability[t][k] - (double)(k == sequence->target[t])) /
                        (double)sequence->steps;
            gradient[BY + k] += dz;
            for (size_t i = 0; i < HIDDEN; ++i) {
                gradient[WY + k * HIDDEN + i] += dz * forward->h[t + 1][i];
                dh[i] += p[WY + k * HIDDEN + i] * dz;
            }
        }
        memset(future, 0, sizeof(future));
        for (size_t i = 0; i < HIDDEN; ++i) {
            double h = forward->h[t + 1][i];
            /* tanh'(z)=1-h^2，把输出层和未来时刻的影响传回本时间步。 */
            double da = dh[i] * (1 - h * h);
            gradient[BH + i] += da;
            gradient[WX + i * VOCAB + sequence->input[t]] += da;
            for (size_t j = 0; j < HIDDEN; ++j) {
                gradient[WH + i * HIDDEN + j] += da * forward->h[t][j];
                /* W_h 转置乘 da；future 是下一轮反向循环中的时间梯度。 */
                future[j] += p[WH + i * HIDDEN + j] * da;
            }
        }
    }
    memcpy(initial_gradient, future, sizeof(future));
}

/**
 * @brief 对全部参数梯度做一个全局 L2 范数裁剪，保留梯度方向。
 * @param gradient PARAMETERS 个可写值；非有限值导致失败且不修改。
 * @param limit 有限正阈值。
 * @param norm 有效可写指针，成功时保存裁剪前范数。
 * @return 合法且范数可表示时为 true；失败不修改梯度和 norm。
 */
static bool clip_gradient(double *gradient, double limit, double *norm)
{
    if (!isfinite(limit) || limit <= 0) {
        return false;
    }
    /* 逐项 hypot 计算 L2 范数，避免直接平方时过早溢出。 */
    double value = 0;
    for (size_t i = 0; i < PARAMETERS; ++i) {
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
        for (size_t i = 0; i < PARAMETERS; ++i) {
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
 * @param gradient PARAMETERS 个只读有限梯度。
 * @return 固定学习率 0.01 的更新有限时为 true，失败不修改参数或优化器。
 */
static bool adam_update(Network *network, Adam *adam, const double *gradient)
{
    /* 在副本上计算候选更新；任一参数或矩非有限时，原状态保持不变。 */
    Network next = *network;
    Adam state = *adam;
    state.beta1_power *= 0.9;
    state.beta2_power *= 0.999;
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double g = gradient[i];
        /* m 和 v 分别平滑梯度及其平方，记录一阶、二阶矩估计。 */
        state.first[i] = 0.9 * state.first[i] + 0.1 * g;
        state.second[i] = 0.999 * state.second[i] + 0.001 * g * g;
        /* 除以 1-beta^t 修正初始矩为零带来的偏差。 */
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
 * @brief 返回最高概率字符，平局选择最小编号。
 * @param probability VOCAB 个只读概率。
 * @return 0..VOCAB-1 的字符编号。
 */
static unsigned argmax(const double *probability)
{
    unsigned best = 0;
    for (unsigned k = 1; k < VOCAB; ++k) {
        if (probability[k] > probability[best]) {
            best = k;
        }
    }
    return best;
}

/**
 * @brief teacher forcing 评估，每条独立序列重置状态，按字符加权指标。
 * @param network 有效只读网络。
 * @param texts count 个借用字符串，均可合法编码。
 * @param count 非零序列数。
 * @return 平均交叉熵和下一字符准确率；不用预测字符替换真实输入。
 */
static Metrics evaluate(const Network *network, const char *const *texts, size_t count)
{
    assert(count > 0);
    Metrics metrics = {0};
    size_t tokens = 0;
    size_t correct = 0;
    const double zero[HIDDEN] = {0};
    for (size_t s = 0; s < count; ++s) {
        Sequence sequence;
        assert(sequence_encode(&sequence, texts[s]));
        Forward forward;
        /* 独立序列重置状态；teacher forcing 始终使用真实输入字符。 */
        network_forward(network, &sequence, zero, &forward);
        tokens += sequence.steps;
        metrics.loss += forward.loss * (double)sequence.steps;
        for (size_t t = 0; t < sequence.steps; ++t) {
            correct += argmax(forward.probability[t]) == sequence.target[t];
        }
    }
    metrics.loss /= (double)tokens;
    metrics.accuracy = (double)correct / (double)tokens;
    return metrics;
}

/**
 * @brief 全批量训练：完整序列反向后按字符聚合、裁剪，再更新参数。
 * @param network 有效可写网络，输出验证损失最小的检查点。
 * @param texts count 个合法独立训练字符串；只读借用，不转移所有权。
 * @param count 非零训练序列数。
 * @param valid valid_count 个只读验证字符串；仅选检查点，不用于梯度计算。
 * @param valid_count 非零验证序列数。
 * @param epochs 非零训练轮数。
 * @param output 输出上下文；为空时不报告训练过程。
 * @return 所有更新数值有限时为 true，失败时 network 保留最近更新结果。
 */
static bool train(Network *network, const char *const *texts, size_t count,
                  const char *const *valid, size_t valid_count, size_t epochs, Output *output)
{
    assert(count > 0 && valid_count > 0 && epochs > 0);
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    Network best = *network;
    size_t best_epoch = 0;
    /* 用训练前的验证损失初始化检查点，之后只在验证损失下降时替换。 */
    double best_loss = evaluate(network, valid, valid_count).loss;
    const double zero[HIDDEN] = {0};
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        /* batch 累积所有训练字符的梯度，最后按 token 总数求平均。 */
        double batch[PARAMETERS] = {0};
        size_t tokens = 0;
        for (size_t s = 0; s < count; ++s) {
            Sequence sequence;
            assert(sequence_encode(&sequence, texts[s]));
            Forward forward;
            double gradient[PARAMETERS];
            double initial_gradient[HIDDEN];
            network_forward(network, &sequence, zero, &forward);
            network_backward(network, &sequence, &forward, gradient, initial_gradient);
            /* 将序列平均梯度乘回步数，才能按全语料字符数加权。 */
            tokens += sequence.steps;
            for (size_t i = 0; i < PARAMETERS; ++i) {
                batch[i] += gradient[i] * (double)sequence.steps;
            }
        }
        for (size_t i = 0; i < PARAMETERS; ++i) {
            batch[i] /= (double)tokens;
        }
        double norm = 0;
        /* 先把完整语料梯度裁剪到 L2 范数 1，再进行一次 Adam 更新。 */
        if (!clip_gradient(batch, 1, &norm) || !adam_update(network, &adam, batch)) {
            return false;
        }
        /* 验证集只选检查点，不参与当前轮的反向传播。 */
        Metrics check = evaluate(network, valid, valid_count);
        if (check.loss < best_loss) {
            best_loss = check.loss;
            best = *network;
            best_epoch = epoch + 1;
        }
        if (output != nullptr &&
            (epoch == 0 || (epoch + 1) % 200 == 0 || epoch + 1 == epochs)) {
            report(output, "epoch=%zu train_CE=%.5f validation_CE=%.5f gradient_norm=%.5f",
                   epoch + 1, evaluate(network, texts, count).loss, check.loss, norm);
        }
    }
    *network = best;
    if (output != nullptr) {
        report(output,
               "保留验证损失最小的检查点：epoch=%zu "
               "validation_CE=%.5f。",
               best_epoch, best_loss);
    }
    return true;
}

/**
 * @brief 从种子字符开始贪心生成，保留状态并把预测字符作为下一步输入。
 * @param network 有效只读网络。
 * @param seed 合法字符编号。
 * @param text 至少 count+1 字节的可写数组。
 * @param count 输出字符数；非零时首字符为种子，末尾额外写入 NUL。
 */
static void generate(const Network *network, unsigned seed, char *text, size_t count)
{
    double previous[HIDDEN] = {0};
    unsigned token = seed;
    /* 写出当前字符，再把最高概率预测作为下一输入；全程贪心。 */
    for (size_t t = 0; t < count; ++t) {
        text[t] = alphabet[token];
        double next[HIDDEN];
        double probability[VOCAB];
        double log_sum = forward_step(network, token, previous, next, probability);
        for (size_t k = 0; k < VOCAB; ++k) {
            probability[k] = exp(probability[k] - log_sum);
        }
        token = argmax(probability);
        memcpy(previous, next, sizeof(previous));
    }
    text[count] = '\0';
}

/**
 * @brief 从训练字符转移计数构造加一平滑 bigram 基线，再评估测试文本。
 * @param train_texts train_count 个合法训练字符串。
 * @param train_count 训练序列数，非零。
 * @param test_texts test_count 个合法独立测试字符串。
 * @param test_count 测试序列数，非零。
 * @return 按测试字符平均的交叉熵和准确率。
 */
static Metrics bigram_evaluate(const char *const *train_texts, size_t train_count,
                               const char *const *test_texts, size_t test_count)
{
    /* 该基线只依赖紧邻的前一个字符，无法利用更早的序列上下文。 */
    double counts[VOCAB][VOCAB];
    /* 每种转移先加一，避免训练语料未出现的字符对得到零概率。 */
    for (size_t i = 0; i < VOCAB; ++i) {
        for (size_t j = 0; j < VOCAB; ++j) {
            counts[i][j] = 1;
        }
    }
    for (size_t s = 0; s < train_count; ++s) {
        Sequence sequence;
        assert(sequence_encode(&sequence, train_texts[s]));
        for (size_t t = 0; t < sequence.steps; ++t) {
            ++counts[sequence.input[t]][sequence.target[t]];
        }
    }
    Metrics metrics = {0};
    size_t tokens = 0;
    size_t correct = 0;
    for (size_t s = 0; s < test_count; ++s) {
        Sequence sequence;
        assert(sequence_encode(&sequence, test_texts[s]));
        for (size_t t = 0; t < sequence.steps; ++t) {
            const double *row = counts[sequence.input[t]];
            double sum = 0;
            for (size_t j = 0; j < VOCAB; ++j) {
                sum += row[j];
            }
            metrics.loss -= log(row[sequence.target[t]] / sum);
            correct += argmax(row) == sequence.target[t];
        }
        tokens += sequence.steps;
    }
    metrics.loss /= (double)tokens;
    metrics.accuracy = (double)correct / (double)tokens;
    return metrics;
}

/**
 * @brief 三步序列检查所有共享参数及初始状态的中心差分梯度。
 */
static void test_gradients(void)
{
    Network network;
    network_initialize(&network, 42);
    Sequence sequence;
    assert(sequence_encode(&sequence, "aaba"));
    double initial[HIDDEN];
    for (size_t i = 0; i < HIDDEN; ++i) {
        initial[i] = 0.01 * (double)(i + 1);
    }
    Forward forward;
    network_forward(&network, &sequence, initial, &forward);
    double gradient[PARAMETERS];
    double dh0[HIDDEN];
    network_backward(&network, &sequence, &forward, gradient, dh0);
    const double epsilon = 1e-5;
    /* 中心差分逐一检查 404 个网络参数和 16 个初始隐藏状态分量。 */
    for (size_t i = 0; i < PARAMETERS + HIDDEN; ++i) {
        double *value = i < PARAMETERS ? &network.parameters[i] : &initial[i - PARAMETERS];
        double analytical = i < PARAMETERS ? gradient[i] : dh0[i - PARAMETERS];
        double saved = *value;
        *value = saved + epsilon;
        network_forward(&network, &sequence, initial, &forward);
        double plus = forward.loss;
        *value = saved - epsilon;
        network_forward(&network, &sequence, initial, &forward);
        double minus = forward.loss;
        *value = saved;
        double numerical = (plus - minus) / (2 * epsilon);
        assert(fabs(numerical - analytical) < 1e-7 + 1e-4 * fabs(numerical));
    }
}

/**
 * @brief 验证长度/字符拒绝、状态重置、稳定损失和裁剪失败原子性。
 */
static void test_boundaries(void)
{
    /* 非法字符串必须拒绝，且不能覆盖上一次成功编码的序列。 */
    Sequence sequence;
    assert(sequence_encode(&sequence, "abc"));
    Sequence saved = sequence;
    assert(!sequence_encode(&sequence, nullptr));
    assert(!sequence_encode(&sequence, ""));
    assert(!sequence_encode(&sequence, "a"));
    assert(!sequence_encode(&sequence, "ad"));
    assert(sequence.steps == saved.steps && sequence.input[0] == saved.input[0]);
    char long_text[MAX_STEPS + 3];
    memset(long_text, 'a', sizeof(long_text) - 1);
    long_text[sizeof(long_text) - 1] = '\0';
    assert(!sequence_encode(&sequence, long_text));
    long_text[MAX_STEPS + 1] = '\0';
    assert(sequence_encode(&sequence, long_text) && sequence.steps == MAX_STEPS);

    /* 验证全局范数缩放、零梯度，以及非法输入时失败不改已有梯度。 */
    double gradient[PARAMETERS] = {3, 4};
    double norm = -1;
    assert(clip_gradient(gradient, 1, &norm) && fabs(norm - 5) < 1e-12);
    assert(fabs(gradient[0] - 0.6) < 1e-12 && fabs(gradient[1] - 0.8) < 1e-12);
    double saved_gradient = gradient[0];
    gradient[PARAMETERS - 1] = NAN;
    assert(!clip_gradient(gradient, 1, &norm) && gradient[0] == saved_gradient);
    gradient[PARAMETERS - 1] = 0;
    assert(!clip_gradient(gradient, 0, &norm));
    memset(gradient, 0, sizeof(gradient));
    assert(clip_gradient(gradient, 1, &norm) && norm == 0);

    /* Adam 对非有限梯度拒绝提交，网络和优化器状态都保持原值。 */
    Network network;
    network_initialize(&network, 42);
    Network saved_network = network;
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    gradient[PARAMETERS - 1] = INFINITY;
    assert(!adam_update(&network, &adam, gradient));
    assert(memcmp(network.parameters, saved_network.parameters, sizeof(network.parameters)) == 0);
    assert(adam.beta1_power == 1 && adam.beta2_power == 1 && adam.first[0] == 0);
    /* 长序列损失须稳定；相同序列和初态必须得到相同状态与预测。 */
    const double zero[HIDDEN] = {0};
    Forward longest;
    network_forward(&network, &sequence, zero, &longest);
    assert(isfinite(longest.loss));
    assert(sequence_encode(&sequence, "abca"));
    Forward first;
    Forward second;
    network_forward(&network, &sequence, zero, &first);
    network_forward(&network, &sequence, zero, &second);
    assert(memcmp(first.h, second.h, sizeof(first.h)) == 0);
    assert(first.loss == second.loss);
    /* 小网络用少量非零权重，检查后续状态依赖前一状态。 */
    Network tiny = {0};
    tiny.parameters[WX] = atanh(0.5);
    tiny.parameters[WH] = 0.5;
    tiny.parameters[BY] = log(3);
    Sequence repeated;
    assert(sequence_encode(&repeated, "aaa"));
    network_forward(&tiny, &repeated, zero, &second);
    assert(fabs(second.h[1][0] - 0.5) < 1e-12);
    assert(fabs(second.h[2][0] - tanh(atanh(0.5) + 0.25)) < 1e-12);
    assert(fabs(second.probability[0][0] - 0.5) < 1e-12);
    assert(fabs(second.loss - log(2)) < 1e-12);
    /* 极端 logits 的差值达到 2000，稳定 softmax 仍应给出有限损失。 */
    network.parameters[BY] = 1000;
    network.parameters[BY + 1] = -1000;
    network_forward(&network, &sequence, zero, &first);
    assert(isfinite(first.loss));
    for (size_t t = 0; t < sequence.steps; ++t) {
        double sum = 0;
        for (size_t k = 0; k < VOCAB; ++k) {
            sum += first.probability[t][k];
        }
        assert(fabs(sum - 1) < 1e-12);
    }
}

/**
 * @brief 学习周期序列，再训练微型语料；断言验证后报告独立测试指标。
 * @param argc 参数数量。
 * @param argv 可选 --test（简短验证）或 --log（双输出日志），不转移所有权。
 * @return 验证与输出成功返回 EXIT_SUCCESS，否则失败。
 */
int main(int argc, char **argv)
{
    /* --test 省略训练报告；--log 追加到 logs/ 并同步输出到终端。 */
    bool quiet = argc == 2 && strcmp(argv[1], "--test") == 0;
    bool logging = argc == 2 && strcmp(argv[1], "--log") == 0;
    if (argc != 1 && !quiet && !logging) {
        fprintf(stderr, "用法：%s [--test | --log]\n", argv[0]);
        return EXIT_FAILURE;
    }
    Logger logger = {0};
    if (logging && !logger_open(&logger, "logs/ml/10_char_rnn.log", stdout)) {
        fprintf(stderr, "RNN 日志打开失败：%s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    Output output = {.logger = logging ? &logger : nullptr};

    /* 先运行梯度和边界断言，训练前验证数值实现与失败路径。 */
    test_gradients();
    test_boundaries();

    /* 周期任务隔离长程依赖：训练、验证、测试使用不同的起始相位。 */
    const char *const periodic[] = {"abcabcabcabcabcabcabcabc", "bcabcabcabcabcabcabcabca",
                                    "cabcabcabcabcabcabcabcab"};
    const char *const periodic_validation[] = {"bcabcabcabcabcabcabcabc"};
    const char *const periodic_test[] = {"cabcabcabcabcabcabcab"};
    Network cycle;
    network_initialize(&cycle, 42);
    if (!quiet) {
        report(&output, "=== 字符 RNN：4→16(tanh)→4(softmax)，%d 参数 ===", PARAMETERS);
        report(&output,
               "完整 BPTT；Adam 0.01；全局梯度裁剪1；初始化42；"
               "每条序列 h[0]=0。");
    }
    bool success = train(&cycle, periodic, 3, periodic_validation, 1, 200,
                         quiet ? nullptr : &output);
    if (!success) {
        fputs("周期 RNN 更新失败：出现非有限数值\n", stderr);
        goto cleanup;
    }
    Metrics period = evaluate(&cycle, periodic_test, 1);
    assert(period.loss < 0.05 && period.accuracy == 1);
    /* 缓冲区为 60 个字符加终止符；种子编号 0 对应字符 a。 */
    char generated[61];
    generate(&cycle, 0, generated, 60);
    for (size_t i = 0; i < 60; ++i) {
        assert(generated[i] == alphabet[i % 3]);
    }
    if (!quiet) {
        report(&output, "周期序列测试 CE=%.5f accuracy=%.2f%%；生成：%s", period.loss,
               100 * period.accuracy, generated);
    }
    Network network;
    network_initialize(&network, 42);
    /* 微型语料按完整字符串拆分，验证和测试序列不参与训练梯度。 */
    double initial_loss = evaluate(&network, training, 8).loss;
    if (!quiet) {
        report(&output,
               "微型字符语料：训练/验证/测试=8/2/2 条独立序列，"
               "每条29步；初始 CE=%.5f。",
               initial_loss);
    }
    success = train(&network, training, 8, validation, 2, EPOCHS, quiet ? nullptr : &output);
    if (!success) {
        fputs("字符 RNN 更新失败：出现非有限数值\n", stderr);
        goto cleanup;
    }
    /* bigram 只从训练集计数；与 RNN 在相同独立测试字符上比较。 */
    Metrics final = evaluate(&network, testing, 2);
    Metrics baseline = bigram_evaluate(training, 8, testing, 2);
    assert(evaluate(&network, training, 8).loss < initial_loss * 0.5);
    assert(final.loss < baseline.loss && final.accuracy > baseline.accuracy);
    report(&output,
           "独立测试：RNN CE=%.5f accuracy=%.2f%%；"
           "bigram CE=%.5f accuracy=%.2f%%。",
           final.loss, 100 * final.accuracy, baseline.loss, 100 * baseline.accuracy);
    if (!quiet) {
        generate(&network, 0, generated, 60);
        report(&output, "贪心生成：[%s]；"
                        "生成文本仅供观察，不代表真实语言能力。",
               generated);
    }
    report(&output,
           "阶段 5 验证通过：404个参数及16个初始状态梯度、裁剪、"
           "周期拟合和独立语料评估。");
cleanup:
    if (logging && !logger_close(&logger)) {
        fputs("RNN 日志写入或关闭失败\n", stderr);
        success = false;
    }
    if (fflush(stdout) != 0) {
        output.failed = true;
    }
    return success && !output.failed ? EXIT_SUCCESS : EXIT_FAILURE;
}
