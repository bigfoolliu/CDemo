/**
 * @file
 * @brief 纯 C VAE：256→32→(mu2/logvar2)→采样→32→256。
 * Bernoulli重建负对数似然加标准正态KL；二维潜在空间可直接观察。
 */
#include "../common/logger.h"
#include "../graphics/bmp_capture.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>

/* 16×16 图像经 32 维隐藏层压缩到二维潜在空间。 */
#define SIDE 16
#define PIXELS (SIDE * SIDE)
#define WIDTH 32
#define LATENT 2
#define CAPACITY 96
#define EPOCHS 250
#define BATCH_SIZE 8
#define MC_SAMPLES 4

/* 所有权重、偏置连续存储；枚举值给出各参数块的起始偏移。 */
enum {
    EW = 0,
    EB = EW + WIDTH * PIXELS,
    MW = EB + WIDTH,
    MB = MW + LATENT * WIDTH,
    LW = MB + LATENT,
    LB = LW + LATENT * WIDTH,
    DW = LB + LATENT,
    DB = DW + WIDTH * LATENT,
    OW = DB + WIDTH,
    OB = OW + PIXELS * WIDTH,
    PARAMETERS = OB + PIXELS
};

/* 只有前 count 张图有效，每张图按行优先顺序展成 PIXELS 个像素。 */
typedef struct {
    size_t count;
    double images[CAPACITY][PIXELS];
} Dataset;

/* 权重矩阵按 [输出神经元][输入特征] 排列。 */
typedef struct {
    double parameters[PARAMETERS];
} Network;

/* 一次前向所需的分布参数、随机样本、激活和两项损失。 */
typedef struct {
    double encoder[WIDTH];
    /* logvar 是后验方差的对数；sigma 才是标准差。 */
    double mu[LATENT];
    double logvar[LATENT];
    double sigma[LATENT];
    double epsilon[LATENT];
    double z[LATENT];
    double decoder[WIDTH];
    double image[PIXELS];
    double reconstruction;
    double kl;
} Forward;

/* 重建 NLL 与 KL 按每张图统计；MSE 仅用于评估均值重建。 */
typedef struct {
    double reconstruction;
    double kl;
    double mse;
} Metrics;

/* Adam 的逐参数一、二阶矩及用于偏差修正的 beta 幂。 */
typedef struct {
    double first[PARAMETERS];
    double second[PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

/* 报告上下文借用 Logger，累计格式化和写入错误。 */
typedef struct {
    Logger *logger;
    bool failed;
} Output;

/**
 * @brief 输出报告，可选双输出日志，累计输出失败。
 * @param output 有效可写上下文，借用logger。
 * @param format 有效格式字符串及匹配的后续参数。
 */
static void report(Output *output, const char *format, ...)
{
    /* 超出固定缓冲区的消息按失败处理，不写入截断内容。 */
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
 * @brief 返回开区间(0,1)伪随机数，避免Box-Muller对零取对数。
 * @param state 有效可写独立随机状态，不用于安全用途。
 * @return 可重复均匀数。
 */
static double random_unit(uint32_t *state)
{
    /* 高 24 位加半个量化单位，保证结果严格处于 (0,1)。 */
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return ((double)(*state >> 8) + 0.5) / 16777216.0;
}

/**
 * @brief 用Box-Muller生成两个标准正态数，训练和评估使用独立状态。
 * @param state 有效可写随机状态。
 * @param epsilon LATENT个可写值，覆盖旧内容。
 */
static void sample_normal(uint32_t *state, double *epsilon)
{
    /* Box-Muller 用两次独立均匀抽样生成半径和角度。 */
    double radius = sqrt(-2 * log(random_unit(state)));
    double angle = 2 * acos(-1) * random_unit(state);
    epsilon[0] = radius * cos(angle);
    epsilon[1] = radius * sin(angle);
}

/**
 * @brief 生成数量均衡的圆、方、三角形二值图。
 * @param data 有效可写固定容量数据集。
 * @param count 3..CAPACITY，必须为3的倍数。
 * @param seed 本数据分组的独立图形随机种子。
 */
static void dataset_generate(Dataset *data, size_t count, uint32_t seed)
{
    assert(count >= 3 && count <= CAPACITY && count % 3 == 0);
    data->count = count;
    for (size_t s = 0; s < count; ++s) {
        /* 依次轮换圆、方、三角形，三类共享位置与尺寸变化范围。 */
        double cx = 6.75 + 1.5 * random_unit(&seed);
        double cy = 6.75 + 1.5 * random_unit(&seed);
        double radius = 3 + random_unit(&seed);
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                double dx = (double)column - cx;
                double dy = (double)row - cy;
                bool inside = dx * dx + dy * dy <= radius * radius;
                if (s % 3 == 1) {
                    inside = fabs(dx) <= radius && fabs(dy) <= radius;
                } else if (s % 3 == 2) {
                    inside = dy >= -radius && dy <= radius && fabs(dx) <= (dy + radius) * 0.5;
                }
                data->images[s][row * SIDE + column] = (double)inside;
            }
        }
    }
}

/**
 * @brief Xavier权重初始化，偏置为零。
 * @param network 有效可写模型，覆盖旧参数。
 * @param seed 固定初始化种子。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    /* 清零偏置，只对五块权重执行 Xavier 均匀初始化。 */
    *network = (Network){0};
    const size_t weights[] = {EW, MW, LW, DW, OW};
    const size_t inputs[] = {PIXELS, WIDTH, WIDTH, LATENT, WIDTH};
    const size_t outputs[] = {WIDTH, LATENT, LATENT, WIDTH, PIXELS};
    for (size_t layer = 0; layer < 5; ++layer) {
        double scale = sqrt(6.0 / (double)(inputs[layer] + outputs[layer]));
        for (size_t i = 0; i < inputs[layer] * outputs[layer]; ++i) {
            network->parameters[weights[layer] + i] = scale * (2 * random_unit(&seed) - 1);
        }
    }
}

/**
 * @brief 稳定sigmoid。
 * @param value 有限logit。
 * @return 0..1范围的像素概率。
 */
static double sigmoid(double value)
{
    /* 按符号分支，避免负输入的 exp(-value) 溢出。 */
    if (value >= 0) {
        return 1 / (1 + exp(-value));
    }
    double e = exp(value);
    return e / (1 + e);
}

/**
 * @brief 将潜在坐标解码成像素概率，可用于直接生成而不经过编码器。
 * @param network 有效只读有限参数模型。
 * @param z LATENT个有限只读坐标。
 * @param forward 有效可写缓存，更新z/decoder/image；有target时覆盖重建损失。
 * @param target 可选PIXELS个只读二值目标；空时不计算重建损失。
 */
static void decode(const Network *network, const double *z, Forward *forward, const double *target)
{
    const double *p = network->parameters;
    memcpy(forward->z, z, LATENT * sizeof(double));
    /* 二维潜变量先经 tanh 隐藏层，再为每个像素输出一个 Bernoulli 概率。 */
    for (size_t i = 0; i < WIDTH; ++i) {
        double value = p[DB + i];
        for (size_t j = 0; j < LATENT; ++j) {
            value += p[DW + i * LATENT + j] * z[j];
        }
        forward->decoder[i] = tanh(value);
    }
    if (target != nullptr) {
        forward->reconstruction = 0;
    }
    for (size_t i = 0; i < PIXELS; ++i) {
        double value = p[OB + i];
        for (size_t j = 0; j < WIDTH; ++j) {
            value += p[OW + i * WIDTH + j] * forward->decoder[j];
        }
        forward->image[i] = sigmoid(value);
        if (target != nullptr) {
            /* softplus(logit)-target×logit 是数值稳定的像素 BCE。 */
            forward->reconstruction +=
                fmax(value, 0) - target[i] * value + log1p(exp(-fabs(value)));
        }
    }
}

/**
 * @brief 编码分布并重参数化，损失为像素BCE之和加KL，不是平均BCE加KL。
 * @param network 有效只读有限模型。
 * @param input PIXELS个[0,1]内只读像素，输入亦为重建目标。
 * @param epsilon LATENT个只读固定标准正态样本；零表示解码均值。
 * @param forward 有效可写缓存，不与输入重叠。
 */
static void network_forward(const Network *network, const double *input, const double *epsilon,
                            Forward *forward)
{
    *forward = (Forward){0};
    const double *p = network->parameters;
    /* 编码器将展平的图像映射到共享的 tanh 隐藏层。 */
    for (size_t i = 0; i < WIDTH; ++i) {
        double value = p[EB + i];
        for (size_t j = 0; j < PIXELS; ++j) {
            value += p[EW + i * PIXELS + j] * input[j];
        }
        forward->encoder[i] = tanh(value);
    }
    double z[LATENT];
    for (size_t i = 0; i < LATENT; ++i) {
        /* 两个独立输出头分别预测潜变量均值与原始 logvar 分数。 */
        double mu = p[MB + i];
        double raw = p[LB + i];
        for (size_t j = 0; j < WIDTH; ++j) {
            mu += p[MW + i * WIDTH + j] * forward->encoder[j];
            raw += p[LW + i * WIDTH + j] * forward->encoder[j];
        }
        /* 光滑有界logvar，梯度必须包含此tanh的导数，不使用硬裁切。 */
        double logvar = 6 * tanh(raw / 6);
        forward->mu[i] = mu;
        forward->logvar[i] = logvar;
        forward->sigma[i] = exp(0.5 * logvar);
        forward->epsilon[i] = epsilon[i];
        /* 固定 epsilon 后，采样成为可微的 z = mu + sigma×epsilon。 */
        z[i] = mu + forward->sigma[i] * epsilon[i];
        /* 每个潜在维度对标准正态先验的 KL，expm1 在零附近更稳定。 */
        forward->kl += 0.5 * (mu * mu + expm1(logvar) - logvar);
    }
    decode(network, z, forward, input);
    assert(isfinite(forward->reconstruction) && isfinite(forward->kl));
}

/**
 * @brief 完整反向传播，重参数化将重建导数传到均值和方差分支。
 * @param network 与缓存对应的有效只读旧模型。
 * @param input PIXELS个只读前向输入和目标。
 * @param forward 借用只读前向缓存，epsilon须与前向保持一致。
 * @param gradient PARAMETERS个可写梯度，覆盖后保存总损失导数。
 * @param input_gradient PIXELS个可写导数，包含目标随输入变化的BCE项。
 * @param epsilon_gradient LATENT个可写噪声导数，用于重参数化验证。
 */
static void network_backward(const Network *network, const double *input, const Forward *forward,
                             double *gradient, double *input_gradient, double *epsilon_gradient)
{
    memset(gradient, 0, PARAMETERS * sizeof(double));
    const double *p = network->parameters;
    double decoder[WIDTH] = {0};
    /* sigmoid 与 BCE 合并求导：logit 梯度为预测概率减目标值。 */
    for (size_t i = 0; i < PIXELS; ++i) {
        double delta = forward->image[i] - input[i];
        gradient[OB + i] = delta;
        double logit = p[OB + i];
        for (size_t j = 0; j < WIDTH; ++j) {
            gradient[OW + i * WIDTH + j] = delta * forward->decoder[j];
            decoder[j] += p[OW + i * WIDTH + j] * delta;
            logit += p[OW + i * WIDTH + j] * forward->decoder[j];
        }
        /* 输入还充当 BCE 目标，因此其直接导数为 -logit。 */
        input_gradient[i] = -logit;
    }
    double dz[LATENT] = {0};
    /* 先穿过解码器的 tanh，再把重建梯度汇集到潜变量 z。 */
    for (size_t i = 0; i < WIDTH; ++i) {
        double delta = decoder[i] * (1 - forward->decoder[i] * forward->decoder[i]);
        gradient[DB + i] = delta;
        for (size_t j = 0; j < LATENT; ++j) {
            gradient[DW + i * LATENT + j] = delta * forward->z[j];
            dz[j] += p[DW + i * LATENT + j] * delta;
        }
    }
    double encoder[WIDTH] = {0};
    /* z 的梯度经重参数化分给均值与 logvar；两支再加上各自的 KL 梯度。 */
    for (size_t i = 0; i < LATENT; ++i) {
        double dmu = dz[i] + forward->mu[i];
        double dlv =
            dz[i] * 0.5 * forward->sigma[i] * forward->epsilon[i] + 0.5 * expm1(forward->logvar[i]);
        /* logvar = 6*tanh(raw/6)，链式求导还需乘其限幅导数。 */
        double scaled = forward->logvar[i] / 6;
        double draw = dlv * (1 - scaled * scaled);
        gradient[MB + i] = dmu;
        gradient[LB + i] = draw;
        epsilon_gradient[i] = dz[i] * forward->sigma[i];
        for (size_t j = 0; j < WIDTH; ++j) {
            gradient[MW + i * WIDTH + j] = dmu * forward->encoder[j];
            gradient[LW + i * WIDTH + j] = draw * forward->encoder[j];
            encoder[j] += p[MW + i * WIDTH + j] * dmu + p[LW + i * WIDTH + j] * draw;
        }
    }
    /* 两个输出头在共享编码器汇合；输入梯度再叠加编码路径的贡献。 */
    for (size_t i = 0; i < WIDTH; ++i) {
        double delta = encoder[i] * (1 - forward->encoder[i] * forward->encoder[i]);
        gradient[EB + i] = delta;
        for (size_t j = 0; j < PIXELS; ++j) {
            gradient[EW + i * PIXELS + j] = delta * input[j];
            input_gradient[j] += p[EW + i * PIXELS + j] * delta;
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
    /* 所有分量乘相同系数，缩短梯度但不改变方向。 */
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
 * @return 固定学习率 0.003 的更新有限时为 true，失败不修改参数或优化器。
 */
static bool adam_update(Network *network, Adam *adam, const double *gradient)
{
    /* 在副本上计算动量、偏差修正与新参数，任一非有限值都不提交。 */
    Network next = *network;
    Adam state = *adam;
    state.beta1_power *= 0.9;
    state.beta2_power *= 0.999;
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double g = gradient[i];
        state.first[i] = 0.9 * state.first[i] + 0.1 * g;
        state.second[i] = 0.999 * state.second[i] + 0.001 * g * g;
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
 * @brief 像素均方误差；只是直观指标，不参与VAE的ELBO训练目标。
 * @param a PIXELS个有限只读像素。
 * @param b PIXELS个有限只读像素。
 * @return 按像素平均的平方误差。
 */
static double image_mse(const double *a, const double *b)
{
    double sum = 0;
    for (size_t i = 0; i < PIXELS; ++i) {
        double delta = a[i] - b[i];
        sum += delta * delta;
    }
    return sum / PIXELS;
}

/**
 * @brief 固定评估噪声估计负ELBO，同时报告解码均值的确定性MSE。
 * @param network 有效只读模型。
 * @param data 非空只读独立数据集。
 * @return 每图重建NLL、每图KL及均值重建MSE；每次重用同一评估种子。
 */
static Metrics evaluate(const Network *network, const Dataset *data)
{
    assert(data->count > 0);
    Metrics result = {0};
    /* 每次评估重置随机种子，使不同时期的验证损失可以直接比较。 */
    uint32_t seed = 5025;
    const double zero[LATENT] = {0};
    for (size_t s = 0; s < data->count; ++s) {
        Forward forward;
        /* 多次采样估计重建项；解析 KL 对同一图像不依赖采样噪声。 */
        for (size_t mc = 0; mc < MC_SAMPLES; ++mc) {
            double epsilon[LATENT];
            sample_normal(&seed, epsilon);
            network_forward(network, data->images[s], epsilon, &forward);
            result.reconstruction += forward.reconstruction / MC_SAMPLES;
            result.kl += forward.kl / MC_SAMPLES;
        }
        /* 令 epsilon=0 得到解码器在潜变量均值处的输出，只用于 MSE。 */
        network_forward(network, data->images[s], zero, &forward);
        result.mse += image_mse(data->images[s], forward.image);
    }
    result.reconstruction /= (double)data->count;
    result.kl /= (double)data->count;
    result.mse /= (double)data->count;
    return result;
}

/**
 * @brief 小批量随机重参数训练，固定验证epsilon的负ELBO选择检查点。
 * @param network 有效可写模型，成功返回最佳验证参数。
 * @param data 非空只读训练集，图形不被修改。
 * @param valid 非空只读验证集，不参与梯度计算。
 * @param output 可选报告上下文，为空省略过程。
 * @return 更新全部有限时为true，失败保留最近一次更新。
 */
static bool train(Network *network, const Dataset *data, const Dataset *valid, Output *output)
{
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    Network best = *network;
    Metrics baseline = evaluate(network, valid);
    double best_loss = baseline.reconstruction + baseline.kl;
    size_t best_epoch = 0;
    /* 训练随机流跨批次、跨轮次持续前进；验证使用独立固定随机流。 */
    uint32_t seed = 4025;
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        for (size_t start = 0; start < data->count; start += BATCH_SIZE) {
            size_t count = data->count - start;
            if (count > BATCH_SIZE) {
                count = BATCH_SIZE;
            }
            /* 按实际批量大小平均梯度，最后一批不足八张时权重仍正确。 */
            double batch[PARAMETERS] = {0};
            for (size_t s = start; s < start + count; ++s) {
                double epsilon[LATENT];
                sample_normal(&seed, epsilon);
                Forward forward;
                double gradient[PARAMETERS];
                double dx[PIXELS];
                double de[LATENT];
                network_forward(network, data->images[s], epsilon, &forward);
                network_backward(network, data->images[s], &forward, gradient, dx, de);
                for (size_t i = 0; i < PARAMETERS; ++i) {
                    batch[i] += gradient[i] / (double)count;
                }
            }
            double norm = 0;
            if (!clip_gradient(batch, 5, &norm) || !adam_update(network, &adam, batch)) {
                return false;
            }
        }
        /* 用验证集负 ELBO 选择检查点，不用测试集挑选参数。 */
        Metrics metrics = evaluate(network, valid);
        if (metrics.reconstruction + metrics.kl < best_loss) {
            best_loss = metrics.reconstruction + metrics.kl;
            best = *network;
            best_epoch = epoch + 1;
        }
        if (output != nullptr && (epoch == 0 || (epoch + 1) % 50 == 0)) {
            report(output, "epoch=%zu val_NLL=%.4f val_KL=%.4f negative_ELBO=%.4f mean_MSE=%.5f",
                   epoch + 1, metrics.reconstruction, metrics.kl,
                   metrics.reconstruction + metrics.kl, metrics.mse);
        }
    }
    *network = best;
    if (output != nullptr) {
        report(output, "验证检查点epoch=%zu negative_ELBO=%.4f", best_epoch, best_loss);
    }
    return true;
}

/**
 * @brief 固定同一噪声，用中心差分验证参数、输入及噪声导数。
 */
static void test_gradients(void)
{
    Network network;
    network_initialize(&network, 42);
    double input[PIXELS];
    for (size_t i = 0; i < PIXELS; ++i) {
        input[i] = 0.1 + 0.6 * (double)(i % 3) / 2;
    }
    double noise[LATENT] = {0.7, -0.4};
    Forward forward;
    double gradient[PARAMETERS];
    double dx[PIXELS];
    double de[LATENT];
    network_forward(&network, input, noise, &forward);
    network_backward(&network, input, &forward, gradient, dx, de);
    /* 对参数、输入、噪声逐一作中心差分；正负扰动共用同一噪声。 */
    const double epsilon = 1e-5;
    for (size_t block = 0; block < 3; ++block) {
        double *values = network.parameters;
        const double *derivative = gradient;
        size_t count = PARAMETERS;
        if (block == 1) {
            values = input;
            derivative = dx;
            count = PIXELS;
        } else if (block == 2) {
            values = noise;
            derivative = de;
            count = LATENT;
        }
        for (size_t i = 0; i < count; ++i) {
            double saved = values[i];
            values[i] = saved + epsilon;
            network_forward(&network, input, noise, &forward);
            double plus = forward.reconstruction + forward.kl;
            values[i] = saved - epsilon;
            network_forward(&network, input, noise, &forward);
            double minus = forward.reconstruction + forward.kl;
            values[i] = saved;
            double numerical = (plus - minus) / (2 * epsilon);
            assert(fabs(numerical - derivative[i]) < 2e-6 + 2e-4 * fabs(numerical));
        }
    }
}

/**
 * @brief 验证标准正态KL为零、重参数化、稳定边界与失败不部分更新。
 */
static void test_boundaries(void)
{
    /* 零参数时后验等于先验，所有像素概率均为 1/2。 */
    Network network = {0};
    double input[PIXELS] = {0};
    double epsilon[LATENT] = {0.5, -1};
    Forward forward;
    network_forward(&network, input, epsilon, &forward);
    assert(forward.kl == 0 && forward.sigma[0] == 1);
    assert(forward.z[0] == epsilon[0] && forward.z[1] == epsilon[1]);
    assert(fabs(forward.reconstruction - PIXELS * log(2)) < 1e-9);
    network.parameters[MB] = 2;
    network_forward(&network, input, epsilon, &forward);
    assert(fabs(forward.kl - 2) < 1e-12 && forward.z[0] == 2.5);
    /* 极端原始分数用于检查 logvar 限幅和 BCE 的有限性。 */
    network.parameters[LB] = 1000;
    network.parameters[OB] = 1000;
    network_forward(&network, input, epsilon, &forward);
    assert(forward.logvar[0] <= 6 && isfinite(forward.kl));
    assert(forward.image[0] == 1 && isfinite(forward.reconstruction));
    network.parameters[LB] = -1000;
    network_forward(&network, input, epsilon, &forward);
    assert(forward.logvar[0] >= -6 && forward.sigma[0] > 0);
    uint32_t a = 7;
    uint32_t b = 7;
    double first[LATENT];
    double second[LATENT];
    sample_normal(&a, first);
    sample_normal(&b, second);
    assert(first[0] == second[0] && first[1] == second[1] && isfinite(first[0]));
    /* 非有限梯度不能让裁剪或 Adam 留下部分更新。 */
    double gradient[PARAMETERS] = {3, 4};
    double norm = 0;
    assert(clip_gradient(gradient, 1, &norm) && fabs(norm - 5) < 1e-12);
    double saved = gradient[0];
    gradient[PARAMETERS - 1] = NAN;
    assert(!clip_gradient(gradient, 1, &norm) && gradient[0] == saved);
    Network before = network;
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    assert(!adam_update(&network, &adam, gradient));
    assert(memcmp(network.parameters, before.parameters, sizeof(network.parameters)) == 0);
    assert(adam.beta1_power == 1);
}

/**
 * @brief 固定强度灰度绘制，不对每张图片单独拉伸对比度。
 * @param canvas 有效可写画布。
 * @param image PIXELS个只读[0,1]像素。
 * @param left 起始横坐标，完整SIDE×scale矩形须在画布内。
 * @param top 起始纵坐标，同样须容纳完整图片。
 * @param scale 像素放大倍数，实际调用为2或4。
 */
static void draw_tile(GfxCanvas *canvas, const double *image, int left, int top, int scale)
{
    /* 所有图像共用 [0,1] 到灰度的映射，便于直接比较亮度。 */
    for (size_t row = 0; row < SIDE; ++row) {
        for (size_t column = 0; column < SIDE; ++column) {
            uint8_t gray = (uint8_t)lround(255 * image[row * SIDE + column]);
            gfx_rect(canvas, left + (int)column * scale, top + (int)row * scale, scale, scale, true,
                     gfx_rgb(gray, gray, gray));
        }
    }
}

/**
 * @brief 独占创建BMP并报告路径，保留已有图片。
 * @param canvas 有效只读画布。
 * @param name 合法ASCII文件名前缀。
 * @param output 有效可写报告上下文。
 * @return 写入成功为true，失败明确报告errno。
 */
static bool save_canvas(const GfxCanvas *canvas, const char *name, Output *output)
{
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(canvas, name, path)) {
        fprintf(stderr, "VAE图片导出失败：%s\n", strerror(errno));
        return false;
    }
    report(output, "已导出：%s", path);
    return true;
}

/**
 * @brief 导出重建、二维网格、均值插值与标准正态先验样本四种视图。
 * @param network 有效只读已训练模型。
 * @param data 至少3张独立测试图，不参与训练或选择参数。
 * @param output 有效可写报告上下文。
 * @return 四张BMP均保存成功为true，失败停止后续导出，已有结果保留。
 */
static bool export_images(const Network *network, const Dataset *data, Output *output)
{
    static GfxCanvas canvas;
    /* 每行依次为测试原图、epsilon=0 的均值重建、一次后验采样重建。 */
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    const double zero[LATENT] = {0};
    uint32_t seed = 6025;
    for (size_t s = 0; s < 3; ++s) {
        Forward mean;
        Forward sampled;
        double epsilon[LATENT];
        sample_normal(&seed, epsilon);
        network_forward(network, data->images[s], zero, &mean);
        network_forward(network, data->images[s], epsilon, &sampled);
        int top = 5 + (int)s * 80;
        draw_tile(&canvas, data->images[s], 20, top, 4);
        draw_tile(&canvas, mean.image, 120, top, 4);
        draw_tile(&canvas, sampled.image, 220, top, 4);
    }
    if (!save_canvas(&canvas, "vae-reconstruction", output)) {
        return false;
    }
    /* 在二维潜在空间的 [-2,2] 网格上直接解码，观察连续变化。 */
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    for (size_t row = 0; row < 7; ++row) {
        for (size_t column = 0; column < 7; ++column) {
            double z[LATENT] = {-2 + 4 * (double)column / 6, 2 - 4 * (double)row / 6};
            Forward decoded = {0};
            decode(network, z, &decoded, nullptr);
            draw_tile(&canvas, decoded.image, 48 + (int)column * 32, 8 + (int)row * 32, 2);
        }
    }
    if (!save_canvas(&canvas, "vae-latent-grid", output)) {
        return false;
    }
    /* 在两张测试图的后验均值之间线性插值，不重新采样。 */
    Forward from;
    Forward to;
    network_forward(network, data->images[0], zero, &from);
    network_forward(network, data->images[2], zero, &to);
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    for (size_t i = 0; i < 8; ++i) {
        double alpha = (double)i / 7;
        double z[LATENT];
        for (size_t j = 0; j < LATENT; ++j) {
            z[j] = (1 - alpha) * from.mu[j] + alpha * to.mu[j];
        }
        Forward decoded = {0};
        decode(network, z, &decoded, nullptr);
        draw_tile(&canvas, decoded.image, 16 + (int)(i % 4) * 72, 40 + (int)(i / 4) * 80, 4);
    }
    if (!save_canvas(&canvas, "vae-interpolation", output)) {
        return false;
    }
    /* 从标准正态先验取样后直接解码，用于观察生成结果。 */
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    for (size_t i = 0; i < 12; ++i) {
        double z[LATENT];
        sample_normal(&seed, z);
        Forward decoded = {0};
        decode(network, z, &decoded, nullptr);
        draw_tile(&canvas, decoded.image, 16 + (int)(i % 4) * 72, 12 + (int)(i / 4) * 72, 4);
    }
    return save_canvas(&canvas, "vae-prior-samples", output);
}

/**
 * @brief 验证、训练并可选导出图形或双输出日志。
 * @param argc 参数数量。
 * @param argv 可选--test或--export和--log；测试模式不写文件。
 * @return 数值验证、训练及输出成功时为EXIT_SUCCESS。
 */
int main(int argc, char **argv)
{
    bool quiet = false;
    bool images = false;
    bool logging = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--test") == 0 && !quiet) {
            quiet = true;
        } else if (strcmp(argv[i], "--export") == 0 && !images) {
            images = true;
        } else if (strcmp(argv[i], "--log") == 0 && !logging) {
            logging = true;
        } else {
            fprintf(stderr, "用法：%s [--export] [--log] 或 --test\n", argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (quiet && (images || logging)) {
        fputs("--test不能与文件输出选项合用\n", stderr);
        return EXIT_FAILURE;
    }
    /* 日志只在显式指定 --log 时追加写入 logs/。 */
    Logger logger = {0};
    if (logging && !logger_open(&logger, "logs/ml/13_vae_shapes.log", stdout)) {
        fprintf(stderr, "VAE日志打开失败：%s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    Output output = {.logger = logging ? &logger : nullptr};
    bool success = true;
    /* 数值梯度与边界验证先于训练执行。 */
    test_gradients();
    test_boundaries();
    static Dataset data;
    static Dataset valid;
    static Dataset test;
    dataset_generate(&data, 96, 2025);
    dataset_generate(&valid, 48, 2026);
    dataset_generate(&test, 48, 2027);
    Network network;
    network_initialize(&network, 42);
    Metrics initial = evaluate(&network, &valid);
    if (!quiet) {
        report(&output,
               "=== VAE：256→32→mu/logvar各2维→32→256，%d个参数 ===", PARAMETERS);
        report(&output,
               "BCE像素求和+KL(beta=1)；logvar=6*tanh(raw/6)；"
               "Adam0.003，裁剪5，批量8。");
        report(&output,
               "图形2025/2026/2027，初始化42，训练epsilon4025，"
               "评估5025；每图4次评估采样。");
    }
    if (!train(&network, &data, &valid, quiet ? nullptr : &output)) {
        success = false;
        goto cleanup;
    }
    Metrics final = evaluate(&network, &test);
    /* 训练图的逐像素均值作为基线，在独立测试集计算其 MSE。 */
    double mean_image[PIXELS] = {0};
    for (size_t s = 0; s < data.count; ++s) {
        for (size_t i = 0; i < PIXELS; ++i) {
            mean_image[i] += data.images[s][i] / (double)data.count;
        }
    }
    double baseline = 0;
    for (size_t s = 0; s < test.count; ++s) {
        baseline += image_mse(mean_image, test.images[s]) / (double)test.count;
    }
    assert(final.reconstruction + final.kl < initial.reconstruction + initial.kl);
    assert(final.mse < baseline && final.kl >= -1e-9);
    report(&output,
           "独立测试：NLL=%.4f KL=%.4f negative_ELBO=%.4f；"
           "均值重建MSE=%.5f，平均图MSE=%.5f。",
           final.reconstruction, final.kl, final.reconstruction + final.kl, final.mse, baseline);
    if (images && !export_images(&network, &test, &output)) {
        success = false;
    }
    report(&output,
           "VAE验证通过：16900参数、256输入和2噪声导数，"
           "重参数化/KL边界及独立重建评估。");
cleanup:
    if (!success) {
        fputs("VAE训练或导出失败\n", stderr);
    }
    if (logging && !logger_close(&logger)) {
        fputs("VAE日志写入或关闭失败\n", stderr);
        success = false;
    }
    if (fflush(stdout) != 0) {
        output.failed = true;
    }
    return success && !output.failed ? EXIT_SUCCESS : EXIT_FAILURE;
}
