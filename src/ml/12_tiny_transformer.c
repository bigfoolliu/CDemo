/**
 * @file
 * @brief 阶段 7：单层、双头、16维字符 Transformer，手写完整反向传播。
 * pre-LayerNorm、可学习位置、因果注意力、GELU前馈和残差；无自动求导。
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

/* 教学规模：词表、宽度、头数、上下文、前馈宽度、容量与轮数。 */
#define VOCAB 4
#define DIM 16
#define HEADS 2
#define CONTEXT 16
#define FF 32
#define CAPACITY 32
#define EPOCHS 250

/* 每个注意力头按连续特征切片，模型宽度必须能被头数整除。 */
static_assert(DIM % HEADS == 0, "每个注意力头必须获得整数维度");

/* 所有可训练张量平铺在一个数组里；枚举值是每个张量的起始偏移。 */
enum {
    EMB = 0,
    POS = EMB + VOCAB * DIM,
    G1 = POS + CONTEXT * DIM,
    B1 = G1 + DIM,
    QW = B1 + DIM,
    QB = QW + DIM * DIM,
    KW = QB + DIM,
    KB = KW + DIM * DIM,
    VW = KB + DIM,
    VB = VW + DIM * DIM,
    OW = VB + DIM,
    OB = OW + DIM * DIM,
    G2 = OB + DIM,
    B2 = G2 + DIM,
    F1W = B2 + DIM,
    F1B = F1W + FF * DIM,
    F2W = F1B + FF,
    F2B = F2W + DIM * FF,
    G3 = F2B + DIM,
    B3 = G3 + DIM,
    UW = B3 + DIM,
    UB = UW + VOCAB * DIM,
    PARAMETERS = UB + VOCAB
};

static const char alphabet[] = "abc ";
/* 与阶段5相同的完整语料划分，分组之后才构造上下文块。 */
static const char *const training[] = {
    "abac acbc babc cbac abac babc ", "acbc cbac abac babc acbc abac ",
    "babc abac cbac acbc babc cbac ", "cbac babc acbc abac cbac acbc ",
    "abac abac acbc acbc cbac babc ", "babc babc cbac cbac abac acbc ",
    "acbc babc abac cbac acbc cbac ", "cbac acbc babc abac cbac abac "};
static const char *const validation[] = {"abac cbac babc acbc abac acbc ",
                                         "acbc abac babc cbac acbc babc "};
static const char *const testing[] = {"babc acbc cbac abac babc abac ",
                                      "cbac abac acbc babc cbac babc "};

typedef struct {
    /* steps 个连续位置各预测后一字符；上下文不跨块传递。 */
    size_t steps;
    unsigned input[CONTEXT];
    unsigned target[CONTEXT];
} Sequence;

typedef struct {
    /* count 只表示 samples 中有效的前 count 个块。 */
    size_t count;
    Sequence samples[CAPACITY];
} Dataset;

typedef struct {
    /* 以参数偏移访问嵌入、投影、归一化参数和分类器。 */
    double parameters[PARAMETERS];
} Network;

/* 一次前向的缓存，行优先布局为 [位置][特征]，供反向传播复用。
 * x/u 是嵌入与位置相加后的表示及其归一化值；q/k/v 是注意力投影。
 * c/o/y 是注意力上下文、输出投影和第一次残差结果；m/pre/activation/f
 * 是前馈分支，z/last 是第二次残差结果及最终归一化值。
 */
typedef struct {
    double x[CONTEXT * DIM];
    double u[CONTEXT * DIM];
    double q[CONTEXT * DIM];
    double k[CONTEXT * DIM];
    double v[CONTEXT * DIM];
    double c[CONTEXT * DIM];
    double o[CONTEXT * DIM];
    double y[CONTEXT * DIM];
    double m[CONTEXT * DIM];
    double pre[CONTEXT * FF];
    double activation[CONTEXT * FF];
    double f[CONTEXT * DIM];
    double z[CONTEXT * DIM];
    double last[CONTEXT * DIM];
    double hats[3][CONTEXT * DIM];
    /* 每个 LayerNorm 缓存标准化值和各位置的逆标准差。 */
    double inverse[3][CONTEXT];
    /* 注意力按 [头][查询位置][键位置] 存储，未来键位置保持为零。 */
    double attention[HEADS * CONTEXT * CONTEXT];
    /* 分类器一开始写入 logits，softmax 后原地改为概率。 */
    double probability[CONTEXT * VOCAB];
    double loss;
} Forward;

/* 按所有有效位置汇总的平均交叉熵和下一字符准确率。 */
typedef struct {
    double loss;
    double accuracy;
} Metrics;

typedef struct {
    /* 对模型参数的 Adam 一、二阶矩和偏差修正状态。 */
    double first[PARAMETERS];
    double second[PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

typedef struct {
    /* logger 由 main 持有；Output 只借用它并累计输出失败。 */
    Logger *logger;
    bool failed;
} Output;

/**
 * @brief 输出短报告，可选追加日志，并累计输出错误。
 * @param output 有效可写输出状态，借用 logger 并累计失败。
 * @param format 有效 printf 格式；后续参数须匹配，格式化结果小于 512 字节。
 */
static void report(Output *output, const char *format, ...)
{
    /* 有界缓冲区避免报告内容溢出；过长或写入失败都会记录到 failed。 */
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
 * @brief 独立字符串分组内切成最多16步的非重叠预测块，不丢尾块。
 * @param data 有效可写数据集，失败保持不变。
 * @param texts count个有效零结尾字符串，只含a/b/c/空格，长度至少2。
 * @param count 非零字符串数量；生成块数不得超过CAPACITY。
 * @return 字符和容量合法时为true；不转移任何输入所有权。
 */
static bool dataset_encode(Dataset *data, const char *const *texts, size_t count)
{
    if (count == 0) {
        return false;
    }
    /* 先构造候选集；全部字符串和容量合法后才覆盖调用者数据。 */
    Dataset result = {0};
    for (size_t s = 0; s < count; ++s) {
        if (texts[s] == nullptr || strlen(texts[s]) < 2) {
            return false;
        }
        size_t length = strlen(texts[s]);
        for (size_t t = 0; t < length; ++t) {
            if (strchr(alphabet, texts[s][t]) == nullptr) {
                return false;
            }
        }
        /* 分块只在当前字符串内推进，末尾不足 CONTEXT 的块也会保留。 */
        for (size_t start = 0; start < length - 1;) {
            if (result.count == CAPACITY) {
                return false;
            }
            Sequence *sample = &result.samples[result.count++];
            sample->steps = length - 1 - start;
            if (sample->steps > CONTEXT) {
                sample->steps = CONTEXT;
            }
            for (size_t t = 0; t < sample->steps; ++t) {
                sample->input[t] = (unsigned)(strchr(alphabet, texts[s][start + t]) - alphabet);
                sample->target[t] =
                    (unsigned)(strchr(alphabet, texts[s][start + t + 1]) - alphabet);
            }
            start += sample->steps;
        }
    }
    *data = result;
    return true;
}

/**
 * @brief 固定种子随机权重和位置向量；LayerNorm缩放初始为1，偏置为0。
 * @param network 有效可写模型，旧参数覆盖。
 * @param seed 可重复初始化种子，不用于安全用途。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    /* 权重随机初始化，偏置为零；LayerNorm gamma 设为一。 */
    const size_t starts[] = {EMB, POS, QW, KW, VW, OW, F1W, F2W, UW};
    const size_t lengths[] = {VOCAB * DIM, CONTEXT * DIM, DIM * DIM,
                              DIM * DIM,   DIM * DIM,     DIM * DIM,
                              FF * DIM,    DIM * FF,      VOCAB * DIM};
    for (size_t block = 0; block < sizeof(starts) / sizeof(starts[0]); ++block) {
        for (size_t i = 0; i < lengths[block]; ++i) {
            seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
            network->parameters[starts[block] + i] =
                0.3 * (2 * (double)(seed >> 8) / 16777216.0 - 1);
        }
    }
    for (size_t i = 0; i < DIM; ++i) {
        network->parameters[G1 + i] = 1;
        network->parameters[G2 + i] = 1;
        network->parameters[G3 + i] = 1;
    }
}

/**
 * @brief 行优先全连接前向，权重为[outputs][inputs]。
 * @param x rows×inputs个只读值。
 * @param y rows×outputs个可写值，不与x重叠。
 * @param rows 1..CONTEXT个位置。
 * @param inputs 输入宽度。
 * @param outputs 输出宽度。
 * @param w outputs×inputs个只读权重。
 * @param b outputs个只读偏置。
 */
static void dense_forward(const double *x, double *y, size_t rows, size_t inputs, size_t outputs,
                          const double *w, const double *b)
{
    /* 每个位置独立计算 y[t,i] = b[i] + Σ_j w[i,j] x[t,j]。 */
    for (size_t t = 0; t < rows; ++t) {
        for (size_t i = 0; i < outputs; ++i) {
            double value = b[i];
            for (size_t j = 0; j < inputs; ++j) {
                value += w[i * inputs + j] * x[t * inputs + j];
            }
            y[t * outputs + i] = value;
        }
    }
}

/**
 * @brief 全连接反向，覆盖输入导数，累加共享权重与偏置导数。
 * @param x rows×inputs个只读前向值。
 * @param dy rows×outputs个只读上游导数。
 * @param dx rows×inputs个可写导数，不与其他数组重叠。
 * @param rows 有效位置数。
 * @param inputs 输入宽度。
 * @param outputs 输出宽度。
 * @param w outputs×inputs个只读前向权重。
 * @param dw outputs×inputs个可写累计梯度。
 * @param db outputs个可写累计梯度。
 */
static void dense_backward(const double *x, const double *dy, double *dx, size_t rows,
                           size_t inputs, size_t outputs, const double *w, double *dw, double *db)
{
    /* dx 每次覆盖；dw 和 db 使用 +=，由调用者先清零以支持梯度累加。 */
    memset(dx, 0, rows * inputs * sizeof(double));
    for (size_t t = 0; t < rows; ++t) {
        for (size_t i = 0; i < outputs; ++i) {
            double delta = dy[t * outputs + i];
            /* 偏置梯度累加 delta，权重梯度累加 delta×输入。 */
            db[i] += delta;
            for (size_t j = 0; j < inputs; ++j) {
                dw[i * inputs + j] += delta * x[t * inputs + j];
                dx[t * inputs + j] += w[i * inputs + j] * delta;
            }
        }
    }
}

/**
 * @brief 每个位置独立沿特征归一化，采用总体方差和epsilon=1e-5。
 * @param x rows×DIM个只读输入。
 * @param y rows×DIM个可写输出。
 * @param hat rows×DIM个可写标准化值，供反向使用。
 * @param inverse rows个可写逆标准差。
 * @param rows 1..CONTEXT个位置。
 * @param gamma DIM个只读可学习缩放。
 * @param beta DIM个只读可学习平移。
 */
static void norm_forward(const double *x, double *y, double *hat, double *inverse, size_t rows,
                         const double *gamma, const double *beta)
{
    /* 每个位置单独沿 DIM 个特征求均值与总体方差，不混合不同位置。 */
    for (size_t t = 0; t < rows; ++t) {
        double mean = 0;
        for (size_t i = 0; i < DIM; ++i) {
            mean += x[t * DIM + i] / DIM;
        }
        double variance = 0;
        for (size_t i = 0; i < DIM; ++i) {
            double centered = x[t * DIM + i] - mean;
            variance += centered * centered / DIM;
        }
        /* epsilon 防止常量特征行除以零；缓存逆标准差供反向使用。 */
        inverse[t] = 1 / sqrt(variance + 1e-5);
        for (size_t i = 0; i < DIM; ++i) {
            size_t index = t * DIM + i;
            /* 标准化后再做可学习的逐特征缩放和平移。 */
            hat[index] = (x[index] - mean) * inverse[t];
            y[index] = gamma[i] * hat[index] + beta[i];
        }
    }
}

/**
 * @brief LayerNorm反向，包括均值和方差对输入的影响。
 * @param dy rows×DIM个只读上游导数。
 * @param dx rows×DIM个可写输入导数，不与输入重叠。
 * @param hat rows×DIM个只读前向标准化值。
 * @param inverse rows个只读前向逆标准差。
 * @param rows 有效位置数。
 * @param gamma DIM个只读缩放参数。
 * @param dg DIM个可写缩放累计梯度。
 * @param db DIM个可写平移累计梯度。
 */
static void norm_backward(const double *dy, double *dx, const double *hat, const double *inverse,
                          size_t rows, const double *gamma, double *dg, double *db)
{
    /* 扣除均值和方差方向投影；gamma、beta 梯度分别累加到 dg、db。 */
    for (size_t t = 0; t < rows; ++t) {
        double sum = 0;
        double projection = 0;
        for (size_t i = 0; i < DIM; ++i) {
            size_t index = t * DIM + i;
            double value = dy[index] * gamma[i];
            sum += value;
            projection += value * hat[index];
            dg[i] += dy[index] * hat[index];
            db[i] += dy[index];
        }
        for (size_t i = 0; i < DIM; ++i) {
            size_t index = t * DIM + i;
            dx[index] =
                inverse[t] * (dy[index] * gamma[i] - sum / DIM - hat[index] * projection / DIM);
        }
    }
}

/**
 * @brief 因果注意力；只对s<=t的键归一化，未来权重严格为零。
 * @param q rows×DIM个只读查询。
 * @param k rows×DIM个只读键。
 * @param v rows×DIM个只读值。
 * @param c rows×DIM个可写拼接上下文。
 * @param a heads×CONTEXT×CONTEXT个可写权重，先全部清零。
 * @param rows 1..CONTEXT个位置。
 * @param heads 1或HEADS，每头宽度DIM/heads。
 */
static void attention_forward(const double *q, const double *k, const double *v, double *c,
                              double *a, size_t rows, size_t heads)
{
    assert(rows > 0 && rows <= CONTEXT && (heads == 1 || heads == HEADS));
    /* 未来位置先保持零；后面只填写因果掩码允许的键位置。 */
    memset(a, 0, heads * CONTEXT * CONTEXT * sizeof(double));
    memset(c, 0, rows * DIM * sizeof(double));
    size_t width = DIM / heads;
    /* 缩放点积避免分数随每头维度增大而过度放大。 */
    double scale = 1 / sqrt((double)width);
    for (size_t head = 0; head < heads; ++head) {
        for (size_t t = 0; t < rows; ++t) {
            double *row = a + (head * CONTEXT + t) * CONTEXT;
            double maximum = -INFINITY;
            /* 只计算 s<=t 的键，因此当前位置看不到未来 token。 */
            for (size_t s = 0; s <= t; ++s) {
                double score = 0;
                for (size_t i = 0; i < width; ++i) {
                    size_t d = head * width + i;
                    score += q[t * DIM + d] * k[s * DIM + d] * scale;
                }
                row[s] = score;
                maximum = fmax(maximum, score);
            }
            double sum = 0;
            /* 减去该行最大分数后再取指数，稳定计算因果 softmax。 */
            for (size_t s = 0; s <= t; ++s) {
                row[s] = exp(row[s] - maximum);
                sum += row[s];
            }
            for (size_t s = 0; s <= t; ++s) {
                row[s] /= sum;
                /* 每个头只写自己的特征切片，最后拼接成 DIM 维上下文。 */
                for (size_t i = 0; i < width; ++i) {
                    size_t d = head * width + i;
                    c[t * DIM + d] += row[s] * v[s * DIM + d];
                }
            }
        }
    }
}

/**
 * @brief 注意力反向，softmax行内雅可比乘法只遍历未掩码位置。
 * @param q rows×DIM个只读查询。
 * @param k rows×DIM个只读键。
 * @param v rows×DIM个只读值。
 * @param a heads×CONTEXT×CONTEXT个只读前向权重。
 * @param dc rows×DIM个只读上下文导数。
 * @param dq rows×DIM个可写查询导数。
 * @param dk rows×DIM个可写键导数。
 * @param dv rows×DIM个可写值导数；三个导数数组不重叠。
 * @param rows 有效位置数。
 * @param heads 1或HEADS，与前向一致。
 */
static void attention_backward(const double *q, const double *k, const double *v, const double *a,
                               const double *dc, double *dq, double *dk, double *dv, size_t rows,
                               size_t heads)
{
    /* 清零三个输出后，逐查询位置累加共享键和值收到的梯度。 */
    memset(dq, 0, rows * DIM * sizeof(double));
    memset(dk, 0, rows * DIM * sizeof(double));
    memset(dv, 0, rows * DIM * sizeof(double));
    size_t width = DIM / heads;
    double scale = 1 / sqrt((double)width);
    for (size_t head = 0; head < heads; ++head) {
        for (size_t t = 0; t < rows; ++t) {
            const double *row = a + (head * CONTEXT + t) * CONTEXT;
            double da[CONTEXT] = {0};
            double mean = 0;
            /* da 是损失对权重的导数；mean 是 softmax 雅可比的加权平均项。 */
            for (size_t s = 0; s <= t; ++s) {
                for (size_t i = 0; i < width; ++i) {
                    size_t d = head * width + i;
                    da[s] += dc[t * DIM + d] * v[s * DIM + d];
                    dv[s * DIM + d] += row[s] * dc[t * DIM + d];
                }
                mean += row[s] * da[s];
            }
            /* dL/d(q·k_s) = a_s(da_s - Σ_r a_r da_r)/sqrt(width)。 */
            for (size_t s = 0; s <= t; ++s) {
                double ds = row[s] * (da[s] - mean) * scale;
                for (size_t i = 0; i < width; ++i) {
                    size_t d = head * width + i;
                    dq[t * DIM + d] += ds * k[s * DIM + d];
                    dk[s * DIM + d] += ds * q[t * DIM + d];
                }
            }
        }
    }
}

/**
 * @brief 单层decoder前向；pre-LN和最终LN均不跨时间位置归一化。
 * @param network 有效只读有限参数模型。
 * @param sequence 1..CONTEXT步合法字符输入/标签。
 * @param forward 有效可写缓存，覆盖旧值；不得与模型重叠。
 */
static void network_forward(const Network *network, const Sequence *sequence, Forward *forward)
{
    assert(sequence->steps > 0 && sequence->steps <= CONTEXT);
    *forward = (Forward){0};
    size_t rows = sequence->steps;
    const double *p = network->parameters;
    /* token embedding 与可学习位置向量相加，作为第一个残差支路的输入。 */
    for (size_t t = 0; t < rows; ++t) {
        assert(sequence->input[t] < VOCAB && sequence->target[t] < VOCAB);
        for (size_t i = 0; i < DIM; ++i) {
            forward->x[t * DIM + i] =
                p[EMB + sequence->input[t] * DIM + i] + p[POS + t * DIM + i];
        }
    }
    /* pre-LayerNorm 后分别投影 Q、K、V，再做因果多头注意力。 */
    norm_forward(forward->x, forward->u, forward->hats[0], forward->inverse[0], rows, p + G1,
                 p + B1);
    dense_forward(forward->u, forward->q, rows, DIM, DIM, p + QW, p + QB);
    dense_forward(forward->u, forward->k, rows, DIM, DIM, p + KW, p + KB);
    dense_forward(forward->u, forward->v, rows, DIM, DIM, p + VW, p + VB);
    attention_forward(forward->q, forward->k, forward->v, forward->c, forward->attention, rows,
                      HEADS);
    dense_forward(forward->c, forward->o, rows, DIM, DIM, p + OW, p + OB);
    for (size_t i = 0; i < rows * DIM; ++i) {
        forward->y[i] = forward->x[i] + forward->o[i];
    }
    /* 第一次残差后再 pre-LayerNorm，经过扩张、GELU 和降维前馈层。 */
    norm_forward(forward->y, forward->m, forward->hats[1], forward->inverse[1], rows, p + G2,
                 p + B2);
    dense_forward(forward->m, forward->pre, rows, DIM, FF, p + F1W, p + F1B);
    for (size_t i = 0; i < rows * FF; ++i) {
        double value = forward->pre[i];
        /* 精确 GELU：x·Φ(x)，用 erf 计算标准正态分布函数 Φ。 */
        forward->activation[i] = 0.5 * value * (1 + erf(value / sqrt(2)));
    }
    dense_forward(forward->activation, forward->f, rows, FF, DIM, p + F2W, p + F2B);
    for (size_t i = 0; i < rows * DIM; ++i) {
        forward->z[i] = forward->y[i] + forward->f[i];
    }
    /* 最终 LayerNorm 和字符分类器逐位置给出下一字符分布。 */
    norm_forward(forward->z, forward->last, forward->hats[2], forward->inverse[2], rows, p + G3,
                 p + B3);
    dense_forward(forward->last, forward->probability, rows, DIM, VOCAB, p + UW, p + UB);
    /* 对每个位置用稳定 log-sum-exp 算交叉熵，再把 logits 原地转成概率。 */
    for (size_t t = 0; t < rows; ++t) {
        double *row = forward->probability + t * VOCAB;
        double maximum = row[0];
        for (size_t k = 1; k < VOCAB; ++k) {
            maximum = fmax(maximum, row[k]);
        }
        double sum = 0;
        for (size_t k = 0; k < VOCAB; ++k) {
            sum += exp(row[k] - maximum);
        }
        double log_sum = maximum + log(sum);
        forward->loss += log_sum - row[sequence->target[t]];
        for (size_t k = 0; k < VOCAB; ++k) {
            row[k] = exp(row[k] - log_sum);
        }
    }
    forward->loss /= (double)rows;
    assert(isfinite(forward->loss));
}

/**
 * @brief 从平均字符CE反传全部参数，残差和Q/K/V的分支导数需要相加。
 * @param network 有效只读前向模型，反向完成前不更新。
 * @param sequence 与前向缓存对应的合法序列。
 * @param forward 借用只读前向缓存。
 * @param gradient PARAMETERS个可写导数，先清零再累计。
 */
static void network_backward(const Network *network, const Sequence *sequence,
                             const Forward *forward, double *gradient)
{
    /* 所有分支对同一参数的影响相加；先清零，后续反向调用均使用 +=。 */
    memset(gradient, 0, PARAMETERS * sizeof(double));
    size_t rows = sequence->steps;
    const double *p = network->parameters;
    double logits[CONTEXT * VOCAB] = {0};
    /* 交叉熵对每个 logit 的导数为 (概率 - one-hot) / 位置数。 */
    for (size_t t = 0; t < rows; ++t) {
        for (size_t k = 0; k < VOCAB; ++k) {
            logits[t * VOCAB + k] =
                (forward->probability[t * VOCAB + k] - (double)(k == sequence->target[t])) /
                (double)rows;
        }
    }
    double last[CONTEXT * DIM];
    double dz[CONTEXT * DIM];
    double da[CONTEXT * FF];
    double dm[CONTEXT * DIM];
    double dy[CONTEXT * DIM];
    /* softmax + 交叉熵的 logit 导数，再经过分类器与末端归一化。 */
    dense_backward(forward->last, logits, last, rows, DIM, VOCAB, p + UW, gradient + UW,
                   gradient + UB);
    norm_backward(last, dz, forward->hats[2], forward->inverse[2], rows, p + G3, gradient + G3,
                  gradient + B3);
    dense_backward(forward->activation, dz, da, rows, FF, DIM, p + F2W, gradient + F2W,
                   gradient + F2B);
    for (size_t i = 0; i < rows * FF; ++i) {
        double value = forward->pre[i];
        /* 对 exact GELU 求导，再乘到前馈第二层传回的上游梯度上。 */
        double derivative = 0.5 * (1 + erf(value / sqrt(2))) +
                            value * exp(-0.5 * value * value) / sqrt(2 * acos(-1));
        da[i] *= derivative;
    }
    dense_backward(forward->m, da, dm, rows, DIM, FF, p + F1W, gradient + F1W, gradient + F1B);
    norm_backward(dm, dy, forward->hats[1], forward->inverse[1], rows, p + G2, gradient + G2,
                  gradient + B2);
    /* 残差恒等支路的导数直接相加到前馈支路传回的 dy。 */
    for (size_t i = 0; i < rows * DIM; ++i) {
        dy[i] += dz[i];
    }
    double dc[CONTEXT * DIM];
    double dq[CONTEXT * DIM];
    double dk[CONTEXT * DIM];
    double dv[CONTEXT * DIM];
    /* 注意力输出投影反向后，分别回到上下文和注意力参数。 */
    dense_backward(forward->c, dy, dc, rows, DIM, DIM, p + OW, gradient + OW, gradient + OB);
    attention_backward(forward->q, forward->k, forward->v, forward->attention, dc, dq, dk, dv, rows,
                       HEADS);
    double du[CONTEXT * DIM];
    double branch[CONTEXT * DIM];
    /* Q、K、V 是并行分支；它们对 LayerNorm 输入的梯度必须相加。 */
    dense_backward(forward->u, dq, du, rows, DIM, DIM, p + QW, gradient + QW, gradient + QB);
    dense_backward(forward->u, dk, branch, rows, DIM, DIM, p + KW, gradient + KW, gradient + KB);
    for (size_t i = 0; i < rows * DIM; ++i) {
        du[i] += branch[i];
    }
    dense_backward(forward->u, dv, branch, rows, DIM, DIM, p + VW, gradient + VW, gradient + VB);
    for (size_t i = 0; i < rows * DIM; ++i) {
        du[i] += branch[i];
    }
    norm_backward(du, branch, forward->hats[0], forward->inverse[0], rows, p + G1, gradient + G1,
                  gradient + B1);
    /* 重复 token 的 embedding 梯度累加；位置向量按位置分别累加。 */
    for (size_t t = 0; t < rows; ++t) {
        for (size_t i = 0; i < DIM; ++i) {
            double dx = dy[t * DIM + i] + branch[t * DIM + i];
            gradient[EMB + sequence->input[t] * DIM + i] += dx;
            gradient[POS + t * DIM + i] += dx;
        }
    }
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
    /* 用 hypot 逐项累积 L2 范数，避免直接平方导致中间值溢出。 */
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
        /* 所有分量乘同一比例，限制长度但不改变梯度方向。 */
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
 * @return 固定学习率 0.003 的更新有限时为 true，失败不修改参数或优化器。
 */
static bool adam_update(Network *network, Adam *adam, const double *gradient)
{
    /* 在副本上计算；全部有限后才提交模型和优化器状态。 */
    Network next = *network;
    Adam state = *adam;
    state.beta1_power *= 0.9;
    state.beta2_power *= 0.999;
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double g = gradient[i];
        /* 一阶矩平滑梯度，二阶矩平滑梯度平方。 */
        state.first[i] = 0.9 * state.first[i] + 0.1 * g;
        state.second[i] = 0.999 * state.second[i] + 0.001 * g * g;
        /* beta 幂用于修正矩状态从零开始产生的早期偏差。 */
        double first = state.first[i] / (1 - state.beta1_power);
        double second = state.second[i] / (1 - state.beta2_power);
        next.parameters[i] -= 0.003 * first / (sqrt(second) + 1e-8);
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
 * @brief 按字符加权评估，块间不传递状态；评估输入是真实字符。
 * @param network 有效只读网络。
 * @param data 非空合法只读数据集。
 * @return 平均CE和下一字符准确率。
 */
static Metrics evaluate(const Network *network, const Dataset *data)
{
    assert(data->count > 0);
    Metrics result = {0};
    size_t tokens = 0;
    size_t correct = 0;
    /* 每个非重叠块独立前向；以有效字符数加权，避免尾块权重过大。 */
    for (size_t s = 0; s < data->count; ++s) {
        const Sequence *sequence = &data->samples[s];
        Forward forward;
        network_forward(network, sequence, &forward);
        result.loss += forward.loss * (double)sequence->steps;
        tokens += sequence->steps;
        for (size_t t = 0; t < sequence->steps; ++t) {
            correct += argmax(forward.probability + t * VOCAB) == sequence->target[t];
        }
    }
    result.loss /= (double)tokens;
    result.accuracy = (double)correct / (double)tokens;
    return result;
}

/**
 * @brief 全批量训练，按总字符数平均梯度，使用验证CE选择检查点。
 * @param network 有效可写网络，成功返回验证损失最低的参数。
 * @param data 非空只读训练集。
 * @param valid 非空只读验证集，不参与参数梯度计算。
 * @param epochs 非零轮数。
 * @param output 可选报告上下文，空则省略训练过程。
 * @return 更新全部有限时为true，失败保留最近更新。
 */
static bool train(Network *network, const Dataset *data, const Dataset *valid, size_t epochs,
                  Output *output)
{
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    Network best = *network;
    size_t best_epoch = 0;
    /* 初始模型作为 epoch 0 检查点，验证损失只负责选模型。 */
    double best_loss = evaluate(network, valid).loss;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        /* 一个 epoch 对全部训练块做一次全批量梯度更新。 */
        double batch[PARAMETERS] = {0};
        size_t tokens = 0;
        for (size_t s = 0; s < data->count; ++s) {
            const Sequence *sequence = &data->samples[s];
            Forward forward;
            double gradient[PARAMETERS];
            network_forward(network, sequence, &forward);
            network_backward(network, sequence, &forward, gradient);
            /* 单块梯度按该块 steps 平均；乘回 steps 后按字符数累加。 */
            tokens += sequence->steps;
            for (size_t i = 0; i < PARAMETERS; ++i) {
                batch[i] += gradient[i] * (double)sequence->steps;
            }
        }
        for (size_t i = 0; i < PARAMETERS; ++i) {
            batch[i] /= (double)tokens;
        }
        double norm = 0;
        /* 全语料梯度先裁剪到 L2 范数 1，再执行一次 Adam 更新。 */
        if (!clip_gradient(batch, 1, &norm) || !adam_update(network, &adam, batch)) {
            return false;
        }
        /* 不用测试集挑参；只保留验证交叉熵更低的参数快照。 */
        Metrics check = evaluate(network, valid);
        if (check.loss < best_loss) {
            best_loss = check.loss;
            best = *network;
            best_epoch = epoch + 1;
        }
        if (output != nullptr && (epoch == 0 || (epoch + 1) % 50 == 0 || epoch + 1 == epochs)) {
            report(output, "epoch=%zu train_CE=%.5f val_CE=%.5f gradient_norm=%.5f", epoch + 1,
                   evaluate(network, data).loss, check.loss, norm);
        }
    }
    *network = best;
    if (output != nullptr) {
        report(output, "保留验证检查点 epoch=%zu CE=%.5f", best_epoch, best_loss);
    }
    return true;
}

/**
 * @brief 训练计数的加一平滑bigram基线，预测只依赖当前字符。
 * @param data 合法只读训练集。
 * @param test 合法只读独立测试集。
 * @return 按测试字符平均的CE和准确率。
 */
static Metrics bigram_evaluate(const Dataset *data, const Dataset *test)
{
    /* 基线只统计当前字符到下一字符的转移，不利用更早上下文。 */
    double counts[VOCAB][VOCAB];
    /* 加一平滑让训练中未出现的转移仍有非零概率。 */
    for (size_t i = 0; i < VOCAB; ++i) {
        for (size_t j = 0; j < VOCAB; ++j) {
            counts[i][j] = 1;
        }
    }
    for (size_t s = 0; s < data->count; ++s) {
        const Sequence *sequence = &data->samples[s];
        for (size_t t = 0; t < sequence->steps; ++t) {
            ++counts[sequence->input[t]][sequence->target[t]];
        }
    }
    Metrics result = {0};
    size_t tokens = 0;
    size_t correct = 0;
    for (size_t s = 0; s < test->count; ++s) {
        const Sequence *sequence = &test->samples[s];
        for (size_t t = 0; t < sequence->steps; ++t) {
            const double *row = counts[sequence->input[t]];
            double sum = 0;
            for (size_t k = 0; k < VOCAB; ++k) {
                sum += row[k];
            }
            result.loss -= log(row[sequence->target[t]] / sum);
            correct += argmax(row) == sequence->target[t];
        }
        tokens += sequence->steps;
    }
    result.loss /= (double)tokens;
    result.accuracy = (double)correct / (double)tokens;
    return result;
}

/**
 * @brief 贪心生成，每次重算最近16个字符，位置编号从零重新开始。
 * @param network 有效只读已训练模型。
 * @param seed 小于VOCAB的起始字符编号。
 * @param text count+1个可写字节。
 * @param count 非零生成字符数，包括种子。
 */
static void generate(const Network *network, unsigned seed, char *text, size_t count)
{
    assert(count > 0 && seed < VOCAB);
    text[0] = alphabet[seed];
    for (size_t length = 1; length < count; ++length) {
        /* 只保留最近 CONTEXT 个字符；每次前向的位置从零开始。 */
        Sequence sequence = {0};
        sequence.steps = length < CONTEXT ? length : CONTEXT;
        size_t start = length - sequence.steps;
        for (size_t t = 0; t < sequence.steps; ++t) {
            sequence.input[t] = (unsigned)(strchr(alphabet, text[start + t]) - alphabet);
        }
        Forward forward;
        /* 标签只是占位；前向概率有效，但此时 loss 没有评估意义。 */
        network_forward(network, &sequence, &forward);
        text[length] = alphabet[argmax(forward.probability + (sequence.steps - 1) * VOCAB)];
    }
    text[count] = '\0';
}

/**
 * @brief 展示注意力矩阵，每行查询对应各列键，未来列显示为零。
 * @param forward 有效只读前向缓存。
 * @param sequence 对应合法序列。
 * @param output 有效可写报告上下文。
 */
static void show_attention(const Forward *forward, const Sequence *sequence, Output *output)
{
    for (size_t head = 0; head < HEADS; ++head) {
        /* 输出每个查询位置的一整行权重，未来位置应为零。 */
        report(output, "head=%zu，列为输入位置；行内权重和为1。", head);
        for (size_t t = 0; t < sequence->steps; ++t) {
            char row[256] = {0};
            size_t used = 0;
            /* used 保持已写字节数，确保 snprintf 总能收到剩余容量。 */
            for (size_t s = 0; s < sequence->steps; ++s) {
                double value = forward->attention[(head * CONTEXT + t) * CONTEXT + s];
                int written = snprintf(row + used, sizeof(row) - used, " %.3f", value);
                if (written < 0 || (size_t)written >= sizeof(row) - used) {
                    output->failed = true;
                    return;
                }
                used += (size_t)written;
            }
            report(output, "query=%zu '%c':%s", t, alphabet[sequence->input[t]], row);
        }
    }
}

/**
 * @brief 检查三位置展开的全部参数中心差分，包括三个LayerNorm和位置向量。
 */
static void test_gradients(void)
{
    Network network;
    network_initialize(&network, 42);
    Sequence sequence = {.steps = 3, .input = {0, 1, 0}, .target = {1, 0, 2}};
    Forward forward;
    double gradient[PARAMETERS];
    network_forward(&network, &sequence, &forward);
    network_backward(&network, &sequence, &forward, gradient);
    const double epsilon = 1e-5;
    /* 中心差分逐个扰动参数，与手写的整网反向传播结果比较。 */
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double saved = network.parameters[i];
        network.parameters[i] = saved + epsilon;
        network_forward(&network, &sequence, &forward);
        double plus = forward.loss;
        network.parameters[i] = saved - epsilon;
        network_forward(&network, &sequence, &forward);
        double minus = forward.loss;
        network.parameters[i] = saved;
        double numerical = (plus - minus) / (2 * epsilon);
        assert(fabs(numerical - gradient[i]) < 2e-6 + 2e-4 * fabs(numerical));
    }
}

/**
 * @brief 独立验证注意力对Q/K/V的导数和LayerNorm输入/缩放/平移导数。
 */
static void test_primitive_gradients(void)
{
    double q[3 * DIM];
    double k[3 * DIM];
    double v[3 * DIM];
    double upstream[3 * DIM];
    for (size_t i = 0; i < 3 * DIM; ++i) {
        q[i] = 0.02 * ((double)(i % 7) - 3);
        k[i] = 0.03 * ((double)(i % 11) - 5);
        v[i] = 0.1 * ((double)(i % 5) - 2);
        upstream[i] = 0.04 * ((double)(i % 13) - 6);
    }
    double c[3 * DIM];
    double a[HEADS * CONTEXT * CONTEXT];
    double dq[3 * DIM];
    double dk[3 * DIM];
    double dv[3 * DIM];
    attention_forward(q, k, v, c, a, 3, HEADS);
    attention_backward(q, k, v, a, upstream, dq, dk, dv, 3, HEADS);
    double *values[] = {q, k, v};
    const double *derivatives[] = {dq, dk, dv};
    const double epsilon = 1e-5;
    /* 以 Σ(context × upstream) 为标量目标，分别检查 Q、K、V 的导数。 */
    for (size_t block = 0; block < 3; ++block) {
        for (size_t i = 0; i < 3 * DIM; ++i) {
            double saved = values[block][i];
            double loss[2] = {0};
            for (size_t sign = 0; sign < 2; ++sign) {
                values[block][i] = saved + (sign == 0 ? epsilon : -epsilon);
                attention_forward(q, k, v, c, a, 3, HEADS);
                for (size_t j = 0; j < 3 * DIM; ++j) {
                    loss[sign] += c[j] * upstream[j];
                }
            }
            values[block][i] = saved;
            assert(fabs((loss[0] - loss[1]) / (2 * epsilon) - derivatives[block][i]) < 1e-8);
        }
    }
    /* 后两个查询没有上游损失时，对应位置的注意力梯度应为零。 */
    memset(upstream + DIM, 0, 2 * DIM * sizeof(double));
    attention_forward(q, k, v, c, a, 3, HEADS);
    attention_backward(q, k, v, a, upstream, dq, dk, dv, 3, HEADS);
    for (size_t i = DIM; i < 3 * DIM; ++i) {
        assert(dq[i] == 0 && dk[i] == 0 && dv[i] == 0);
    }

    /* 用同一个中心差分目标检查 LayerNorm 的输入、gamma 和 beta 梯度。 */
    double x[DIM];
    double gamma[DIM];
    double beta[DIM];
    for (size_t i = 0; i < DIM; ++i) {
        x[i] = 0.1 * (double)i;
        gamma[i] = 1 + 0.05 * (double)i;
        beta[i] = -0.01 * (double)i;
    }
    double y[DIM];
    double hat[DIM];
    double inverse[1];
    double dx[DIM];
    double dg[DIM] = {0};
    double db[DIM] = {0};
    norm_forward(x, y, hat, inverse, 1, gamma, beta);
    norm_backward(upstream, dx, hat, inverse, 1, gamma, dg, db);
    double *norm_values[] = {x, gamma, beta};
    const double *norm_derivatives[] = {dx, dg, db};
    for (size_t block = 0; block < 3; ++block) {
        for (size_t i = 0; i < DIM; ++i) {
            double saved = norm_values[block][i];
            double loss[2] = {0};
            for (size_t sign = 0; sign < 2; ++sign) {
                norm_values[block][i] = saved + (sign == 0 ? epsilon : -epsilon);
                norm_forward(x, y, hat, inverse, 1, gamma, beta);
                for (size_t j = 0; j < DIM; ++j) {
                    loss[sign] += y[j] * upstream[j];
                }
            }
            norm_values[block][i] = saved;
            assert(fabs((loss[0] - loss[1]) / (2 * epsilon) - norm_derivatives[block][i]) < 1e-8);
        }
    }
}

/**
 * @brief 单头手算、掩码、常量特征归一化、字符及容量边界验证。
 * @param output 可选报告上下文，用于显示单头入门矩阵。
 */
static void test_boundaries(Output *output)
{
    /* Q=K=0 时允许位置等权，逐步验证因果掩码与加权 V 的手算结果。 */
    double q[3 * DIM] = {0};
    double k[3 * DIM] = {0};
    double v[3 * DIM] = {0};
    for (size_t t = 0; t < 3; ++t) {
        for (size_t i = 0; i < DIM; ++i) {
            v[t * DIM + i] = (double)(2 * t + 1);
        }
    }
    double c[3 * DIM];
    double a[CONTEXT * CONTEXT];
    attention_forward(q, k, v, c, a, 3, 1);
    for (size_t t = 0; t < 3; ++t) {
        assert(fabs(c[t * DIM] - (double)(t + 1)) < 1e-12);
        for (size_t s = 0; s < 3; ++s) {
            double expected = s <= t ? 1 / (double)(t + 1) : 0;
            assert(fabs(a[t * CONTEXT + s] - expected) < 1e-12);
        }
    }
    if (output != nullptr) {
        report(output, "单头手算：Q=K=0，V每行=1/3/5，因果加权结果=1/2/3。");
        report(output, "注意力矩阵：[1 0 0]；[0.5 0.5 0]；[1/3 1/3 1/3]。");
    }
    /* 常量特征行的方差为零；epsilon 应让前向与反向仍保持有限。 */
    double x[DIM];
    double y[DIM];
    double hat[DIM];
    double inverse[1];
    double gamma[DIM];
    double beta[DIM] = {0};
    double upstream[DIM];
    double dx[DIM];
    double dg[DIM] = {0};
    double db[DIM] = {0};
    for (size_t i = 0; i < DIM; ++i) {
        x[i] = 5;
        gamma[i] = 1;
        upstream[i] = (double)i;
    }
    norm_forward(x, y, hat, inverse, 1, gamma, beta);
    norm_backward(upstream, dx, hat, inverse, 1, gamma, dg, db);
    for (size_t i = 0; i < DIM; ++i) {
        assert(y[i] == 0 && isfinite(dx[i]));
        assert(fabs(dx[i] - inverse[0] * ((double)i - 7.5)) < 1e-10);
    }

    /* 非法字符或容量失败不应覆盖原数据；测试尾块是否保留。 */
    Dataset data;
    const char *const short_text[] = {"ab"};
    assert(dataset_encode(&data, short_text, 1) && data.count == 1 && data.samples[0].steps == 1);
    const char *const bad[] = {"ad"};
    assert(!dataset_encode(&data, bad, 1) && data.samples[0].target[0] == 1);
    const char *const empty[] = {"a"};
    assert(!dataset_encode(&data, empty, 1));
    assert(!dataset_encode(&data, short_text, 0));
    char long_text[CAPACITY * CONTEXT + 3];
    memset(long_text, 'a', sizeof(long_text) - 1);
    long_text[sizeof(long_text) - 1] = '\0';
    const char *const oversized[] = {long_text};
    assert(!dataset_encode(&data, oversized, 1) && data.count == 1);
    assert(dataset_encode(&data, training, 8) && data.count == 16);
    assert(data.samples[0].steps == 16 && data.samples[1].steps == 13);

    Network network;
    network_initialize(&network, 42);
    Forward one;
    Sequence one_step = {.steps = 1, .input = {0}, .target = {1}};
    network_forward(&network, &one_step, &one);
    double one_gradient[PARAMETERS];
    network_backward(&network, &one_step, &one, one_gradient);
    assert(isfinite(one.loss));
    assert(one.attention[0] == 1 && one.attention[CONTEXT * CONTEXT] == 1);
    /* 改动位置 5 之后的输入，不得影响更早的预测结果。 */
    Sequence sequence = {.steps = CONTEXT};
    for (size_t t = 0; t < CONTEXT; ++t) {
        sequence.input[t] = (unsigned)(t % VOCAB);
        sequence.target[t] = (unsigned)((t + 1) % VOCAB);
    }
    Forward before;
    Forward after;
    network_forward(&network, &sequence, &before);
    for (size_t t = 5; t < CONTEXT; ++t) {
        sequence.input[t] = (sequence.input[t] + 1) % VOCAB;
    }
    network_forward(&network, &sequence, &after);
    for (size_t t = 0; t < 5; ++t) {
        for (size_t k = 0; k < VOCAB; ++k) {
            assert(before.probability[t * VOCAB + k] == after.probability[t * VOCAB + k]);
        }
    }
    Sequence prefix = sequence;
    prefix.steps = 5;
    Forward truncated;
    network_forward(&network, &prefix, &truncated);
    for (size_t i = 0; i < prefix.steps * VOCAB; ++i) {
        assert(before.probability[i] == truncated.probability[i]);
    }
    for (size_t head = 0; head < HEADS; ++head) {
        for (size_t t = 0; t < CONTEXT; ++t) {
            double sum = 0;
            for (size_t s = 0; s < CONTEXT; ++s) {
                double value = after.attention[(head * CONTEXT + t) * CONTEXT + s];
                sum += value;
                if (s > t) {
                    assert(value == 0);
                }
            }
            assert(fabs(sum - 1) < 1e-12);
        }
    }
    double gradient[PARAMETERS];
    network_backward(&network, &sequence, &after, gradient);
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(isfinite(gradient[i]));
    }
    double norm = 0;
    memset(gradient, 0, sizeof(gradient));
    gradient[0] = 3;
    gradient[1] = 4;
    assert(clip_gradient(gradient, 1, &norm) && fabs(norm - 5) < 1e-12);
    double saved = gradient[0];
    gradient[PARAMETERS - 1] = NAN;
    assert(!clip_gradient(gradient, 1, &norm) && gradient[0] == saved);
    Network original = network;
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    assert(!adam_update(&network, &adam, gradient));
    assert(memcmp(network.parameters, original.parameters, sizeof(network.parameters)) == 0);
    assert(adam.beta1_power == 1);
}

/**
 * @brief 先微型拟合，再训练独立字符语料，展示因果注意力和生成结果。
 * @param argc 参数数目。
 * @param argv 可选--test或--log，借用不释放。
 * @return 数值验证、训练与输出全部成功时返回EXIT_SUCCESS。
 */
int main(int argc, char **argv)
{
    /* --test 压低报告量；--log 在 logs/ 中追加日志并同时输出到终端。 */
    bool quiet = argc == 2 && strcmp(argv[1], "--test") == 0;
    bool logging = argc == 2 && strcmp(argv[1], "--log") == 0;
    if (argc != 1 && !quiet && !logging) {
        fprintf(stderr, "用法：%s [--test | --log]\n", argv[0]);
        return EXIT_FAILURE;
    }
    Logger logger = {0};
    if (logging && !logger_open(&logger, "logs/ml/12_tiny_transformer.log", stdout)) {
        fprintf(stderr, "Transformer日志打开失败：%s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    Output output = {.logger = logging ? &logger : nullptr};
    bool success = true;

    /* 先检查整网梯度、注意力/LayerNorm 原语和边界条件，再开始训练。 */
    test_gradients();
    test_primitive_gradients();
    test_boundaries(quiet ? nullptr : &output);
    if (!quiet) {
        report(&output, "=== 单层Transformer：维度16，双头(每头8)，上下文16，"
                        "FF32，%d个参数 ===",
               PARAMETERS);
        report(&output, "pre-LN/学习位置/GELU/残差；初始化42；"
                        "Adam0.003；梯度裁剪1。");
    }
    /* 单条周期串只验证实现能拟合小样本，不作为泛化能力结论。 */
    Network tiny;
    network_initialize(&tiny, 42);
    const char *const mini[] = {"abcabcabc"};
    Dataset small;
    assert(dataset_encode(&small, mini, 1));
    if (!train(&tiny, &small, &small, 150, nullptr)) {
        success = false;
        goto cleanup;
    }
    Metrics fit = evaluate(&tiny, &small);
    assert(fit.loss < 0.02 && fit.accuracy == 1);
    if (!quiet) {
        report(&output,
               "单条周期语料拟合：CE=%.5f accuracy=%.2f%%（仅实现验证）。",
               fit.loss, 100 * fit.accuracy);
    }
    /* 先按完整文本划分 train/validation/test，再分别切上下文块。 */
    Dataset data;
    Dataset valid;
    Dataset test;
    assert(dataset_encode(&data, training, 8));
    assert(dataset_encode(&valid, validation, 2));
    assert(dataset_encode(&test, testing, 2));
    Network network;
    network_initialize(&network, 42);
    double initial_loss = evaluate(&network, &data).loss;
    if (!quiet) {
        report(&output,
               "完整文本8/2/2条→训练/验证/测试16/4/4块；"
               "尾块13步；初始训练CE=%.5f。",
               initial_loss);
    }
    if (!train(&network, &data, &valid, EPOCHS, quiet ? nullptr : &output)) {
        success = false;
        goto cleanup;
    }
    /* 测试集只在检查点选定后评估，并与仅看相邻字符的基线比较。 */
    Metrics result = evaluate(&network, &test);
    Metrics baseline = bigram_evaluate(&data, &test);
    assert(evaluate(&network, &data).loss < initial_loss * 0.6);
    assert(isfinite(result.loss) && result.loss < baseline.loss &&
           result.accuracy > baseline.accuracy);
    report(&output,
           "独立测试：Transformer CE=%.5f accuracy=%.2f%%；bigram CE=%.5f accuracy=%.2f%%。",
           result.loss, 100 * result.accuracy, baseline.loss, 100 * baseline.accuracy);
    if (!quiet) {
        /* 生成和注意力仅供观察；泛化评估以独立测试集指标为准。 */
        char generated[61];
        generate(&network, 0, generated, 60);
        report(&output, "贪心生成：[%s]；生成不是独立泛化评价。", generated);
        const char *const example[] = {"abac acbc "};
        Dataset display;
        assert(dataset_encode(&display, example, 1));
        Forward forward;
        network_forward(&network, &display.samples[0], &forward);
        show_attention(&forward, &display.samples[0], &output);
    }
    report(&output,
           "阶段7验证通过：2644参数梯度、单头手算、LayerNorm、因果掩码、"
           "微型拟合和独立评估。");
cleanup:
    /* 所有早退共用关闭日志和刷新标准输出的清理路径。 */
    if (!success) {
        fputs("Transformer更新失败：非有限数值\n", stderr);
    }
    if (logging && !logger_close(&logger)) {
        fputs("Transformer日志写入或关闭失败\n", stderr);
        success = false;
    }
    if (fflush(stdout) != 0) {
        output.failed = true;
    }
    return success && !output.failed ? EXIT_SUCCESS : EXIT_FAILURE;
}
