/** @file
 * @brief DQN 网络计算、优化与模型读写。
 */
#ifndef CDEMO_SNAKE_AI_NETWORK_H
#define CDEMO_SNAKE_AI_NETWORK_H

/* 网络数值层：初始化 → 前向预测 → 反向求导 → 参数更新 → 保存/加载。
 * static inline 使这些函数可被多个独立示例包含，无需额外链接实现文件。 */

#include "snake_env.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { HIDDEN = 64, W1 = 0, B1 = 704, W2 = 768, B2 = 4864,
       W3 = 4928, B3 = 5120, PARAMETERS = 5123 };
/* 各层矩阵按输出行连续存放，其后跟偏置。修改维度也需升级模型格式。 */
static_assert(B1 == FEATURES * HIDDEN && W2 == B1 + HIDDEN);
static_assert(B2 == W2 + HIDDEN * HIDDEN && W3 == B2 + HIDDEN);
static_assert(B3 == W3 + ACTIONS * HIDDEN && PARAMETERS == B3 + ACTIONS);
typedef struct { double p[PARAMETERS]; } Network; /* 平坦数组同时保存权重与偏置。 */
/* 前向缓存供反向传播使用，q 是动作价值，不是概率。 */
typedef struct {
    double input[FEATURES];
    double h1[HIDDEN];
    double h2[HIDDEN];
    double q[ACTIONS];
} Forward;
/* Adam 保存梯度及梯度平方的滑动平均，跨批次保留历史。 */
typedef struct {
    double m[PARAMETERS];
    double v[PARAMETERS];
    double beta1_power;
    double beta2_power;
    size_t updates;
} Adam;

/**
 * @brief 通过 Box–Muller 变换生成标准正态样本。
 * @param rng 有效的可读写随机状态地址。
 * @return 近似服从标准正态分布的伪随机样本。
 * @note Box–Muller 将伪随机数转换为标准正态样本；u 严格大于零以避免 log(0)。
 */
static inline double gaussian(uint32_t *rng)
{
    double u = ((double)snake_random(rng) + 1) / 4294967297.0;
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * snake_uniform(rng));
}

/**
 * @brief 初始化网络权重并将偏置清零。
 * @param net 有效的网络输出地址，原参数会被覆盖。
 * @param rng 有效的可读写初始化随机状态。
 * @note 偏置清零；ReLU 层用 He 缩放，线性输出层用较小的方差。
 */
static inline void network_init(Network *net, uint32_t *rng)
{
    *net = (Network){0};
    for (int i = W1; i < B1; ++i) {
        net->p[i] = gaussian(rng) * sqrt(2.0 / FEATURES);
    }
    for (int i = W2; i < B2; ++i) {
        net->p[i] = gaussian(rng) * sqrt(2.0 / HIDDEN);
    }
    for (int i = W3; i < B3; ++i) {
        net->p[i] = gaussian(rng) * sqrt(1.0 / HIDDEN);
    }
}

/**
 * @brief 执行一层全连接计算，并可选应用 ReLU。
 * @param weights 只读行优先矩阵，至少 inputs * outputs 个元素。
 * @param bias 只读偏置数组，至少 outputs 个元素。
 * @param input 只读输入数组，至少 inputs 个元素。
 * @param inputs 输入维度，须为正数。
 * @param outputs 输出维度，须为正数。
 * @param result 可写输出数组，至少 outputs 个元素，不与输入存储重叠。
 * @param relu 是否对输出应用 ReLU 激活。
 * @note 一层全连接：result[row] = Σ weights[row,column] * input[column] + bias[row]。
 */
static inline void dense(const double *weights, const double *bias, const double *input,
                         int inputs, int outputs, double *result, bool relu)
{
    for (int row = 0; row < outputs; ++row) {
        double value = bias[row];
        for (int column = 0; column < inputs; ++column) {
            value += weights[row * inputs + column] * input[column];
        }
        result[row] = relu ? fmax(0, value) : value;
    }
}

/**
 * @brief 执行网络前向传播并保存反向传播缓存。
 * @param net 已初始化的只读网络。
 * @param state 只读特征数组，包含 FEATURES 个元素。
 * @param cache 有效的前向缓存输出地址。
 * @note 两层 ReLU 提取特征，线性层输出三个动作的预计折扣收益。
 */
static inline void network_forward(const Network *net, const double state[FEATURES], Forward *cache)
{
    memcpy(cache->input, state, sizeof(cache->input));
    dense(net->p + W1, net->p + B1, state, FEATURES, HIDDEN, cache->h1, true);
    dense(net->p + W2, net->p + B2, cache->h1, HIDDEN, HIDDEN, cache->h2, true);
    dense(net->p + W3, net->p + B3, cache->h2, HIDDEN, ACTIONS, cache->q, false);
}

/**
 * @brief 选择预测 Q 值最大的动作。
 * @param net 已初始化的只读网络。
 * @param state 只读特征数组，包含 FEATURES 个元素。
 * @return 0..ACTIONS-1 的动作编号；平局取较小编号。
 * @note 选择最大 Q 值；平局取较小编号，不额外屏蔽危险动作。
 */
static inline int greedy_action(const Network *net, const double state[FEATURES])
{
    Forward cache;
    network_forward(net, state, &cache);
    int best = 0;
    for (int action = 1; action < ACTIONS; ++action) {
        if (cache.q[action] > cache.q[best]) {
            best = action;
        }
    }
    return best;
}

/**
 * @brief 计算阈值为一的 Huber 损失。
 * @param error 预测值与目标值之差。
 * @return 非负损失；小误差用平方项，大误差用线性项。
 * @note 小误差用平方项，大误差用线性项，减弱异常 TD 误差的影响。
 */
static inline double huber(double error)
{
    return fabs(error) <= 1 ? 0.5 * error * error : fabs(error) - 0.5;
}

/**
 * @brief 计算单样本损失并累积所选动作的梯度。
 * @param net 与 cache 对应的只读网络参数。
 * @param cache 当前样本的只读前向缓存。
 * @param action 参与反向传播的动作编号，范围为 0..ACTIONS-1。
 * @param target 视为常数的 TD 目标值。
 * @param gradient 可读写梯度累加器，批次开始时须清零。
 * @return 所选动作的 Huber 损失；不修改网络参数。
 * @note 只对所选动作的 Q 值求导。TD 目标是常数，不能沿目标网络反传。
 */
static inline double network_gradient(const Network *net, const Forward *cache,
                                     int action, double target, Network *gradient)
{
    double error = cache->q[action] - target;
    double derivative = fmax(-1, fmin(1, error));
    /* 链式法则从输出层倒推；+= 将多个样本的梯度累积到同一批次。 */
    gradient->p[B3 + action] += derivative;
    double delta2[HIDDEN];
    for (int i = 0; i < HIDDEN; ++i) {
        gradient->p[W3 + action * HIDDEN + i] += derivative * cache->h2[i];
        delta2[i] = cache->h2[i] > 0 ? derivative * net->p[W3 + action * HIDDEN + i] : 0;
        gradient->p[B2 + i] += delta2[i];
        for (int j = 0; j < HIDDEN; ++j) {
            gradient->p[W2 + i * HIDDEN + j] += delta2[i] * cache->h1[j];
        }
    }
    for (int j = 0; j < HIDDEN; ++j) {
        /* 汇总第二隐藏层传来的梯度，再乘第一层 ReLU 的导数。 */
        double delta1 = 0;
        for (int i = 0; i < HIDDEN; ++i) {
            delta1 += delta2[i] * net->p[W2 + i * HIDDEN + j];
        }
        if (cache->h1[j] <= 0) {
            delta1 = 0;
        }
        gradient->p[B1 + j] += delta1;
        for (int i = 0; i < FEATURES; ++i) {
            gradient->p[W1 + j * FEATURES + i] += delta1 * cache->input[i];
        }
    }
    return huber(error);
}

/**
 * @brief 平均并裁剪批次梯度，再执行 Adam 更新。
 * @param net 待更新的网络参数。
 * @param adam 可读写优化器状态，首次使用须清零。
 * @param gradient 批次梯度和，会原地除以 batch。
 * @param batch 批次样本数，须大于零。
 * @return 数值有效并更新成功返回 true；非有限值返回 false，失败不保证回滚。
 * @note batch 必须大于零；先平均梯度并裁剪整体范数，再执行 Adam。
 */
static inline bool adam_update(Network *net, Adam *adam, Network *gradient, size_t batch)
{
    double norm_squared = 0;
    for (int i = 0; i < PARAMETERS; ++i) {
        gradient->p[i] /= (double)batch;
        if (!isfinite(gradient->p[i])) {
            return false;
        }
        norm_squared += gradient->p[i] * gradient->p[i];
    }
    if (!isfinite(norm_squared)) {
        return false;
    }
    double scale = norm_squared > 100 ? 10 / sqrt(norm_squared) : 1;
    if (adam->updates == 0) {
        adam->beta1_power = 1;
        adam->beta2_power = 1;
    }
    adam->beta1_power *= 0.9;
    adam->beta2_power *= 0.999;
    ++adam->updates;
    for (int i = 0; i < PARAMETERS; ++i) {
        double g = gradient->p[i] * scale;
        adam->m[i] = 0.9 * adam->m[i] + 0.1 * g;
        adam->v[i] = 0.999 * adam->v[i] + 0.001 * g * g;
        double m = adam->m[i] / (1 - adam->beta1_power);
        /* 偏差校正抵消滑动平均从零初始化造成的早期偏小；1e-8 防止除零。 */
        double v = adam->v[i] / (1 - adam->beta2_power);
        net->p[i] -= 0.001 * m / (sqrt(v) + 1e-8);
        if (!isfinite(net->p[i])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 将推理参数写入临时文件并替换正式模型。
 * @param net 待保存的只读网络，参数须为有限值。
 * @param path 有效的 NUL 结尾模型路径，父目录须已存在。
 * @return 完整写入并重命名成功返回 true；失败返回 false。
 * @note 模型仅保存推理参数，不是可恢复 Adam / 回放池的训练检查点。
 */
static inline bool network_save(const Network *net, const char *path)
{
    char temporary[1024];
    int length = snprintf(temporary, sizeof(temporary), "%s.tmp", path);
    if (length < 0 || (size_t)length >= sizeof(temporary)) {
        return false;
    }
    FILE *file = fopen(temporary, "wx");
    /* wx 拒绝覆盖已有临时文件；写入完整后再替换正式模型。 */
    if (file == nullptr) {
        return false;
    }
    bool ok = fprintf(file, "CDEMO_SNAKE_DQN 1 11 64 64 3 1 20 10\n") >= 0;
    for (int i = 0; i < PARAMETERS && ok; ++i) {
        ok = isfinite(net->p[i]) && fprintf(file, "%.17g\n", net->p[i]) >= 0;
    }
    if (fclose(file) == EOF) {
        ok = false;
    }
    if (ok) {
        ok = rename(temporary, path) == 0;
    }
    if (!ok) {
        (void)remove(temporary);
    }
    return ok;
}

/**
 * @brief 校验模型文件后一次性更新网络参数。
 * @param net 有效的网络输出地址；校验失败时保持原值。
 * @param path 有效的 NUL 结尾模型文件路径。
 * @return 版本、尺寸、数值及文件结尾均正确时返回 true。
 * @note 先读入候选副本，版本、参数和文件结尾全部校验成功才更新 net。
 */
static inline bool network_load(Network *net, const char *path)
{
    FILE *file = fopen(path, "r");
    if (file == nullptr) {
        return false;
    }
    char header[128];
    Network candidate;
    bool ok = fgets(header, sizeof(header), file) != nullptr &&
        strcmp(header, "CDEMO_SNAKE_DQN 1 11 64 64 3 1 20 10\n") == 0;
    for (int i = 0; i < PARAMETERS && ok; ++i) {
        ok = fscanf(file, "%lf", &candidate.p[i]) == 1 && isfinite(candidate.p[i]);
    }
    if (ok) {
        char extra;
        ok = fscanf(file, " %c", &extra) == EOF && !ferror(file);
    }
    if (fclose(file) == EOF) {
        ok = false;
    }
    if (ok) {
        *net = candidate;
    }
    return ok;
}

#endif
