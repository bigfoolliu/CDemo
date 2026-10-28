/**
 * @file
 * @brief D01：对照 LN、BN、RMSNorm、初始化方差和残差梯度路径。
 * 固定的小批量用于手算与梯度检查；不训练预测模型或依赖外部框架。
 */
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    MAX_ROWS = 4,
    MAX_FEATURES = 4,
    MAX_VALUES = MAX_ROWS * MAX_FEATURES,
    DEMO_ROWS = 2,
    DEMO_FEATURES = 3,
    INIT_WEIGHT_COUNT = 12,
    RESIDUAL_DEPTH = 8
};

typedef struct {
    size_t rows;
    size_t features;
    double normalized[MAX_VALUES];
    double inverse_scale[MAX_ROWS];
} LayerNormCache;

typedef struct {
    size_t features;
    double running_mean[MAX_FEATURES];
    double running_variance[MAX_FEATURES];
    bool initialized;
} BatchNormState;

typedef struct {
    size_t rows;
    size_t features;
    bool training;
    double normalized[MAX_VALUES];
    double inverse_scale[MAX_FEATURES];
} BatchNormCache;

typedef struct {
    size_t rows;
    size_t features;
    double normalized[MAX_VALUES];
    double inverse_scale[MAX_ROWS];
} RmsNormCache;

typedef enum {
    INIT_GLOROT,
    INIT_HE
} InitializerKind;

static const double DEMO_INPUT[DEMO_ROWS * DEMO_FEATURES] = {
    1, 2, 3,
    2, 4, 6
};
static const double DEMO_GAMMA[DEMO_FEATURES] = {1.2, 0.8, -0.5};
static const double DEMO_BETA[DEMO_FEATURES] = {0.1, -0.2, 0.3};
static const double DEMO_UPSTREAM[DEMO_ROWS * DEMO_FEATURES] = {
    1, 0.5, -1,
    -0.5, 1, 2
};

/**
 * @brief 检查小型矩阵的行列数是否在固定缓冲区范围内。
 * @param rows 行数，范围为 1..MAX_ROWS。
 * @param features 特征数，范围为 1..MAX_FEATURES。
 * @return 形状有效时为 true。
 */
static bool shape_is_valid(size_t rows, size_t features)
{
    return rows > 0 && rows <= MAX_ROWS && features > 0 && features <= MAX_FEATURES;
}

/**
 * @brief 检查一段数组是否全为有限数。
 * @param values 待读取数组。
 * @param count 元素数量。
 * @return 所有元素有限时为 true。
 */
static bool values_are_finite(const double *values, size_t count)
{
    if (values == nullptr) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 对每个样本沿特征轴计算层归一化（LayerNorm）。
 * @param input 行优先 rows×features 输入；有限且不与 output 重叠。
 * @param rows 行数，范围为 1..MAX_ROWS。
 * @param features 特征数，范围为 1..MAX_FEATURES。
 * @param gamma 每个特征的有限缩放参数。
 * @param beta 每个特征的有限平移参数。
 * @param epsilon 有限正稳定项。
 * @param output 可写 rows×features 输出；失败时保持不变。
 * @param cache 可写前向缓存；成功时覆盖，失败时保持不变。
 * @return 计算成功且结果有限时为 true。
 */
static bool layer_norm_forward(const double *input, size_t rows, size_t features,
                               const double *gamma, const double *beta, double epsilon,
                               double *output, LayerNormCache *cache)
{
    if (!shape_is_valid(rows, features) || input == nullptr || gamma == nullptr ||
        beta == nullptr || output == nullptr || cache == nullptr ||
        !isfinite(epsilon) || epsilon <= 0 ||
        !values_are_finite(input, rows * features) ||
        !values_are_finite(gamma, features) || !values_are_finite(beta, features)) {
        return false;
    }

    LayerNormCache next_cache = {.rows = rows, .features = features};
    double next_output[MAX_VALUES] = {0};
    for (size_t row = 0; row < rows; ++row) {
        size_t offset = row * features;
        double mean = 0;
        for (size_t feature = 0; feature < features; ++feature) {
            mean += input[offset + feature];
        }
        mean /= (double)features;
        if (!isfinite(mean)) {
            return false;
        }
        double variance = 0;
        for (size_t feature = 0; feature < features; ++feature) {
            double difference = input[offset + feature] - mean;
            variance += difference * difference;
        }
        variance /= (double)features;
        if (!isfinite(variance)) {
            return false;
        }
        next_cache.inverse_scale[row] = 1.0 / sqrt(variance + epsilon);
        for (size_t feature = 0; feature < features; ++feature) {
            size_t index = offset + feature;
            next_cache.normalized[index] =
                (input[index] - mean) * next_cache.inverse_scale[row];
            next_output[index] = gamma[feature] * next_cache.normalized[index] + beta[feature];
        }
    }
    if (!values_are_finite(next_output, rows * features) ||
        !values_are_finite(next_cache.normalized, rows * features) ||
        !values_are_finite(next_cache.inverse_scale, rows)) {
        return false;
    }
    memcpy(output, next_output, rows * features * sizeof(*output));
    *cache = next_cache;
    return true;
}

/**
 * @brief 反向传播层归一化，并累加共享 gamma/beta 的逐特征梯度。
 * @param cache 与上游梯度对应的层归一化缓存。
 * @param gamma 有效的每特征缩放参数。
 * @param upstream rows×features 上游梯度。
 * @param input_gradient 可写 rows×features 输入梯度；失败时保持不变。
 * @param gamma_gradient 可写 features 梯度；失败时保持不变。
 * @param beta_gradient 可写 features 梯度；失败时保持不变。
 * @return 梯度有限时为 true。
 */
static bool layer_norm_backward(const LayerNormCache *cache, const double *gamma,
                               const double *upstream, double *input_gradient,
                               double *gamma_gradient, double *beta_gradient)
{
    if (cache == nullptr || !shape_is_valid(cache->rows, cache->features) || gamma == nullptr ||
        upstream == nullptr || input_gradient == nullptr || gamma_gradient == nullptr ||
        beta_gradient == nullptr || !values_are_finite(gamma, cache->features) ||
        !values_are_finite(upstream, cache->rows * cache->features) ||
        !values_are_finite(cache->normalized, cache->rows * cache->features) ||
        !values_are_finite(cache->inverse_scale, cache->rows)) {
        return false;
    }

    double next_input_gradient[MAX_VALUES] = {0};
    double next_gamma_gradient[MAX_FEATURES] = {0};
    double next_beta_gradient[MAX_FEATURES] = {0};
    for (size_t row = 0; row < cache->rows; ++row) {
        size_t offset = row * cache->features;
        double sum_scaled = 0;
        double sum_scaled_normalized = 0;
        for (size_t feature = 0; feature < cache->features; ++feature) {
            size_t index = offset + feature;
            double scaled = upstream[index] * gamma[feature];
            sum_scaled += scaled;
            sum_scaled_normalized += scaled * cache->normalized[index];
            next_gamma_gradient[feature] += upstream[index] * cache->normalized[index];
            next_beta_gradient[feature] += upstream[index];
        }
        for (size_t feature = 0; feature < cache->features; ++feature) {
            size_t index = offset + feature;
            double scaled = upstream[index] * gamma[feature];
            next_input_gradient[index] = cache->inverse_scale[row] /
                (double)cache->features *
                ((double)cache->features * scaled - sum_scaled -
                 cache->normalized[index] * sum_scaled_normalized);
        }
    }
    if (!values_are_finite(next_input_gradient, cache->rows * cache->features) ||
        !values_are_finite(next_gamma_gradient, cache->features) ||
        !values_are_finite(next_beta_gradient, cache->features)) {
        return false;
    }
    memcpy(input_gradient, next_input_gradient,
           cache->rows * cache->features * sizeof(*input_gradient));
    memcpy(gamma_gradient, next_gamma_gradient, cache->features * sizeof(*gamma_gradient));
    memcpy(beta_gradient, next_beta_gradient, cache->features * sizeof(*beta_gradient));
    return true;
}

/**
 * @brief 初始化批归一化运行统计量，均值为零、方差为一。
 * @param state 可写状态；成功后不持有外部内存。
 * @param features 特征数，范围为 1..MAX_FEATURES。
 * @return 参数有效时为 true；失败时不修改状态。
 */
static bool batch_norm_state_init(BatchNormState *state, size_t features)
{
    if (state == nullptr || features == 0 || features > MAX_FEATURES) {
        return false;
    }
    BatchNormState initialized = {.features = features, .initialized = true};
    for (size_t feature = 0; feature < features; ++feature) {
        initialized.running_variance[feature] = 1;
    }
    *state = initialized;
    return true;
}

/**
 * @brief 执行按批次行轴统计的批归一化（BatchNorm）。
 * @param input 行优先 rows×features 输入；有限且不与 output 重叠。
 * @param rows 行数，训练时为 1..MAX_ROWS。
 * @param features 特征数，须与 state 一致。
 * @param gamma 每个特征的有限缩放参数。
 * @param beta 每个特征的有限平移参数。
 * @param epsilon 有限正稳定项。
 * @param momentum 旧运行统计量占比，范围为 [0,1)。
 * @param training true 时使用当前批次统计并更新运行状态；false 时只读运行状态。
 * @param state 已初始化的运行统计状态；失败时保持不变。
 * @param output 可写 rows×features 输出；失败时保持不变。
 * @param cache 可写反向传播缓存；失败时保持不变。
 * @return 计算成功且结果有限时为 true。
 */
static bool batch_norm_forward(const double *input, size_t rows, size_t features,
                               const double *gamma, const double *beta, double epsilon,
                               double momentum, bool training, BatchNormState *state,
                               double *output, BatchNormCache *cache)
{
    if (!shape_is_valid(rows, features) || input == nullptr || gamma == nullptr ||
        beta == nullptr || state == nullptr || output == nullptr || cache == nullptr ||
        !isfinite(epsilon) || epsilon <= 0 || !isfinite(momentum) ||
        momentum < 0 || momentum >= 1 || !state->initialized ||
        state->features != features || !values_are_finite(input, rows * features) ||
        !values_are_finite(gamma, features) || !values_are_finite(beta, features) ||
        !values_are_finite(state->running_mean, features) ||
        !values_are_finite(state->running_variance, features)) {
        return false;
    }

    BatchNormState next_state = *state;
    BatchNormCache next_cache = {
        .rows = rows,
        .features = features,
        .training = training
    };
    double next_output[MAX_VALUES] = {0};
    for (size_t feature = 0; feature < features; ++feature) {
        double mean = state->running_mean[feature];
        double variance = state->running_variance[feature];
        if (training) {
            mean = 0;
            for (size_t row = 0; row < rows; ++row) {
                mean += input[row * features + feature];
            }
            mean /= (double)rows;
            if (!isfinite(mean)) {
                return false;
            }
            variance = 0;
            for (size_t row = 0; row < rows; ++row) {
                double difference = input[row * features + feature] - mean;
                variance += difference * difference;
            }
            variance /= (double)rows;
            if (!isfinite(variance)) {
                return false;
            }
            next_state.running_mean[feature] = momentum * state->running_mean[feature] +
                                               (1 - momentum) * mean;
            next_state.running_variance[feature] =
                momentum * state->running_variance[feature] + (1 - momentum) * variance;
        }
        if (variance < 0 || !isfinite(variance)) {
            return false;
        }
        next_cache.inverse_scale[feature] = 1.0 / sqrt(variance + epsilon);
        for (size_t row = 0; row < rows; ++row) {
            size_t index = row * features + feature;
            next_cache.normalized[index] =
                (input[index] - mean) * next_cache.inverse_scale[feature];
            next_output[index] = gamma[feature] * next_cache.normalized[index] + beta[feature];
        }
    }
    if (!values_are_finite(next_output, rows * features) ||
        !values_are_finite(next_cache.normalized, rows * features) ||
        !values_are_finite(next_cache.inverse_scale, features) ||
        !values_are_finite(next_state.running_mean, features) ||
        !values_are_finite(next_state.running_variance, features)) {
        return false;
    }
    *state = next_state;
    memcpy(output, next_output, rows * features * sizeof(*output));
    *cache = next_cache;
    return true;
}

/**
 * @brief 反向传播批归一化；训练态对批统计求导，推理态使用固定统计量。
 * @param cache 与上游梯度对应的 BN 缓存。
 * @param gamma 有效的每特征缩放参数。
 * @param upstream rows×features 上游梯度。
 * @param input_gradient 可写 rows×features 输入梯度；失败时保持不变。
 * @param gamma_gradient 可写 features 梯度；失败时保持不变。
 * @param beta_gradient 可写 features 梯度；失败时保持不变。
 * @return 梯度有限时为 true。
 */
static bool batch_norm_backward(const BatchNormCache *cache, const double *gamma,
                               const double *upstream, double *input_gradient,
                               double *gamma_gradient, double *beta_gradient)
{
    if (cache == nullptr || !shape_is_valid(cache->rows, cache->features) || gamma == nullptr ||
        upstream == nullptr || input_gradient == nullptr || gamma_gradient == nullptr ||
        beta_gradient == nullptr || !values_are_finite(gamma, cache->features) ||
        !values_are_finite(upstream, cache->rows * cache->features) ||
        !values_are_finite(cache->normalized, cache->rows * cache->features) ||
        !values_are_finite(cache->inverse_scale, cache->features)) {
        return false;
    }

    double next_input_gradient[MAX_VALUES] = {0};
    double next_gamma_gradient[MAX_FEATURES] = {0};
    double next_beta_gradient[MAX_FEATURES] = {0};
    for (size_t feature = 0; feature < cache->features; ++feature) {
        double sum_scaled = 0;
        double sum_scaled_normalized = 0;
        for (size_t row = 0; row < cache->rows; ++row) {
            size_t index = row * cache->features + feature;
            double scaled = upstream[index] * gamma[feature];
            sum_scaled += scaled;
            sum_scaled_normalized += scaled * cache->normalized[index];
            next_gamma_gradient[feature] += upstream[index] * cache->normalized[index];
            next_beta_gradient[feature] += upstream[index];
        }
        for (size_t row = 0; row < cache->rows; ++row) {
            size_t index = row * cache->features + feature;
            double scaled = upstream[index] * gamma[feature];
            if (cache->training) {
                next_input_gradient[index] = cache->inverse_scale[feature] /
                    (double)cache->rows *
                    ((double)cache->rows * scaled - sum_scaled -
                     cache->normalized[index] * sum_scaled_normalized);
            } else {
                next_input_gradient[index] = cache->inverse_scale[feature] * scaled;
            }
        }
    }
    if (!values_are_finite(next_input_gradient, cache->rows * cache->features) ||
        !values_are_finite(next_gamma_gradient, cache->features) ||
        !values_are_finite(next_beta_gradient, cache->features)) {
        return false;
    }
    memcpy(input_gradient, next_input_gradient,
           cache->rows * cache->features * sizeof(*input_gradient));
    memcpy(gamma_gradient, next_gamma_gradient, cache->features * sizeof(*gamma_gradient));
    memcpy(beta_gradient, next_beta_gradient, cache->features * sizeof(*beta_gradient));
    return true;
}

/**
 * @brief 对每个样本按特征均方根执行 RMSNorm，不减去特征均值。
 * @param input 行优先 rows×features 输入；有限且不与 output 重叠。
 * @param rows 行数，范围为 1..MAX_ROWS。
 * @param features 特征数，范围为 1..MAX_FEATURES。
 * @param gamma 每个特征的有限缩放参数。
 * @param epsilon 有限正稳定项。
 * @param output 可写 rows×features 输出；失败时保持不变。
 * @param cache 可写反向传播缓存；失败时保持不变。
 * @return 计算成功且结果有限时为 true。
 */
static bool rms_norm_forward(const double *input, size_t rows, size_t features,
                             const double *gamma, double epsilon, double *output,
                             RmsNormCache *cache)
{
    if (!shape_is_valid(rows, features) || input == nullptr || gamma == nullptr ||
        output == nullptr || cache == nullptr || !isfinite(epsilon) || epsilon <= 0 ||
        !values_are_finite(input, rows * features) || !values_are_finite(gamma, features)) {
        return false;
    }

    RmsNormCache next_cache = {.rows = rows, .features = features};
    double next_output[MAX_VALUES] = {0};
    for (size_t row = 0; row < rows; ++row) {
        size_t offset = row * features;
        double mean_square = 0;
        for (size_t feature = 0; feature < features; ++feature) {
            double value = input[offset + feature];
            mean_square += value * value;
        }
        mean_square /= (double)features;
        if (!isfinite(mean_square)) {
            return false;
        }
        next_cache.inverse_scale[row] = 1.0 / sqrt(mean_square + epsilon);
        for (size_t feature = 0; feature < features; ++feature) {
            size_t index = offset + feature;
            next_cache.normalized[index] = input[index] * next_cache.inverse_scale[row];
            next_output[index] = gamma[feature] * next_cache.normalized[index];
        }
    }
    if (!values_are_finite(next_output, rows * features) ||
        !values_are_finite(next_cache.normalized, rows * features) ||
        !values_are_finite(next_cache.inverse_scale, rows)) {
        return false;
    }
    memcpy(output, next_output, rows * features * sizeof(*output));
    *cache = next_cache;
    return true;
}

/**
 * @brief 反向传播 RMSNorm，并累加共享缩放参数的梯度。
 * @param cache 与上游梯度对应的 RMSNorm 缓存。
 * @param gamma 有效的每特征缩放参数。
 * @param upstream rows×features 上游梯度。
 * @param input_gradient 可写 rows×features 输入梯度；失败时保持不变。
 * @param gamma_gradient 可写 features 梯度；失败时保持不变。
 * @return 梯度有限时为 true。
 */
static bool rms_norm_backward(const RmsNormCache *cache, const double *gamma,
                             const double *upstream, double *input_gradient,
                             double *gamma_gradient)
{
    if (cache == nullptr || !shape_is_valid(cache->rows, cache->features) || gamma == nullptr ||
        upstream == nullptr || input_gradient == nullptr || gamma_gradient == nullptr ||
        !values_are_finite(gamma, cache->features) ||
        !values_are_finite(upstream, cache->rows * cache->features) ||
        !values_are_finite(cache->normalized, cache->rows * cache->features) ||
        !values_are_finite(cache->inverse_scale, cache->rows)) {
        return false;
    }

    double next_input_gradient[MAX_VALUES] = {0};
    double next_gamma_gradient[MAX_FEATURES] = {0};
    for (size_t row = 0; row < cache->rows; ++row) {
        size_t offset = row * cache->features;
        double normalized_dot = 0;
        for (size_t feature = 0; feature < cache->features; ++feature) {
            size_t index = offset + feature;
            double scaled = upstream[index] * gamma[feature];
            normalized_dot += scaled * cache->normalized[index];
            next_gamma_gradient[feature] += upstream[index] * cache->normalized[index];
        }
        for (size_t feature = 0; feature < cache->features; ++feature) {
            size_t index = offset + feature;
            double scaled = upstream[index] * gamma[feature];
            next_input_gradient[index] = cache->inverse_scale[row] *
                (scaled - cache->normalized[index] * normalized_dot /
                 (double)cache->features);
        }
    }
    if (!values_are_finite(next_input_gradient, cache->rows * cache->features) ||
        !values_are_finite(next_gamma_gradient, cache->features)) {
        return false;
    }
    memcpy(input_gradient, next_input_gradient,
           cache->rows * cache->features * sizeof(*input_gradient));
    memcpy(gamma_gradient, next_gamma_gradient, cache->features * sizeof(*gamma_gradient));
    return true;
}

/**
 * @brief 根据 fan-in/fan-out 和初始化方案计算目标权重方差。
 * @param fan_in 输入连接数，必须大于零。
 * @param fan_out 输出连接数，必须大于零。
 * @param kind Glorot 均匀初始化或 ReLU 使用的 He 均匀初始化。
 * @param variance 可写目标方差；失败时保持不变。
 * @return 参数有效且目标方差有限时为 true。
 */
static bool initializer_variance(size_t fan_in, size_t fan_out, InitializerKind kind,
                                 double *variance)
{
    if (fan_in == 0 || fan_out == 0 || variance == nullptr ||
        (kind != INIT_GLOROT && kind != INIT_HE) ||
        (kind == INIT_GLOROT && fan_in > SIZE_MAX - fan_out)) {
        return false;
    }
    double denominator = kind == INIT_HE ? (double)fan_in : (double)(fan_in + fan_out);
    double result = 2.0 / denominator;
    if (!isfinite(result) || result <= 0) {
        return false;
    }
    *variance = result;
    return true;
}

/**
 * @brief 用固定 LCG 种子产生 Glorot/He 均匀权重并测量样本方差。
 * @param weights 可写 count 个权重。
 * @param count 权重数，必须大于零。
 * @param fan_in 输入连接数，必须大于零。
 * @param fan_out 输出连接数，必须大于零。
 * @param kind 初始化方案。
 * @param seed 可重复种子，不用于安全用途。
 * @param sample_variance 可写有限样本方差输出；失败时保持不变。
 * @return 权重生成成功时为 true。
 */
static bool initialize_weights(double *weights, size_t count, size_t fan_in, size_t fan_out,
                               InitializerKind kind, uint32_t seed, double *sample_variance)
{
    double target_variance = 0;
    if (weights == nullptr || count == 0 || sample_variance == nullptr ||
        !initializer_variance(fan_in, fan_out, kind, &target_variance)) {
        return false;
    }
    double limit = sqrt(3.0 * target_variance);
    double generated[INIT_WEIGHT_COUNT] = {0};
    if (count > INIT_WEIGHT_COUNT || !isfinite(limit)) {
        return false;
    }
    double mean = 0;
    for (size_t i = 0; i < count; ++i) {
        seed = seed * UINT32_C(1664525) + UINT32_C(1013904223);
        double unit = (double)(seed >> 8) / 16777216.0;
        generated[i] = limit * (2.0 * unit - 1.0);
        mean += generated[i];
    }
    mean /= (double)count;
    double variance = 0;
    for (size_t i = 0; i < count; ++i) {
        double difference = generated[i] - mean;
        variance += difference * difference;
    }
    variance /= (double)count;
    if (!isfinite(variance) || !values_are_finite(generated, count)) {
        return false;
    }
    memcpy(weights, generated, count * sizeof(*weights));
    *sample_variance = variance;
    return true;
}

/**
 * @brief 比较重复线性主分支和带恒等捷径的标量梯度传递。
 * @param depth 重复层数，范围为 1..1024。
 * @param branch_derivative 主分支导数，每层相同且必须有限。
 * @param use_residual true 时每层导数为 1+branch_derivative。
 * @param activation 可写输入为 1、输出为重复映射后的标量。
 * @param input_gradient 可写损失对输入的导数，输出上游梯度固定为 1。
 * @return 结果有限时为 true；失败时不修改输出。
 */
static bool gradient_chain(size_t depth, double branch_derivative, bool use_residual,
                           double *activation, double *input_gradient)
{
    if (depth == 0 || depth > 1024 || !isfinite(branch_derivative) ||
        activation == nullptr || input_gradient == nullptr) {
        return false;
    }
    double gain = branch_derivative + (use_residual ? 1.0 : 0.0);
    double value = 1.0;
    double gradient = 1.0;
    for (size_t layer = 0; layer < depth; ++layer) {
        value *= gain;
        gradient *= gain;
        if (!isfinite(value) || !isfinite(gradient)) {
            return false;
        }
    }
    *activation = value;
    *input_gradient = gradient;
    return true;
}

/**
 * @brief 计算矩阵梯度的欧几里得范数。
 * @param values 有限梯度数组。
 * @param count 元素数。
 * @return 梯度二范数。
 */
static double gradient_norm(const double *values, size_t count)
{
    double squared = 0;
    for (size_t i = 0; i < count; ++i) {
        squared += values[i] * values[i];
    }
    return sqrt(squared);
}

/**
 * @brief 把固定演示值写成供 Python 对照的稳定 CSV 键值行。
 * @param name 字段名。
 * @param value 有限演示数值。
 */
static void print_csv_value(const char *name, double value)
{
    printf("%s,%.17g\n", name, value);
}

/**
 * @brief 输出归一化前向/反向、运行统计、初始化与残差的共享夹具。
 */
static void print_csv_fixture(void)
{
    double layer_output[MAX_VALUES] = {0};
    double layer_gradient[MAX_VALUES] = {0};
    double layer_gamma_gradient[MAX_FEATURES] = {0};
    double layer_beta_gradient[MAX_FEATURES] = {0};
    LayerNormCache layer_cache = {0};
    assert(layer_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, layer_output, &layer_cache));
    assert(layer_norm_backward(&layer_cache, DEMO_GAMMA, DEMO_UPSTREAM,
                               layer_gradient, layer_gamma_gradient, layer_beta_gradient));

    BatchNormState batch_state = {0};
    BatchNormCache batch_cache = {0};
    double batch_output[MAX_VALUES] = {0};
    double batch_gradient[MAX_VALUES] = {0};
    double batch_gamma_gradient[MAX_FEATURES] = {0};
    double batch_beta_gradient[MAX_FEATURES] = {0};
    assert(batch_norm_state_init(&batch_state, DEMO_FEATURES));
    assert(batch_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, 0.9, true, &batch_state,
                              batch_output, &batch_cache));
    assert(batch_norm_backward(&batch_cache, DEMO_GAMMA, DEMO_UPSTREAM,
                               batch_gradient, batch_gamma_gradient, batch_beta_gradient));
    double inference_input[DEMO_FEATURES] = {3, 6, 9};
    double inference_output[DEMO_FEATURES] = {0};
    BatchNormCache inference_cache = {0};
    assert(batch_norm_forward(inference_input, 1, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, 0.9, false, &batch_state,
                              inference_output, &inference_cache));

    RmsNormCache rms_cache = {0};
    double rms_output[MAX_VALUES] = {0};
    double rms_gradient[MAX_VALUES] = {0};
    double rms_gamma_gradient[MAX_FEATURES] = {0};
    assert(rms_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                            0.01, rms_output, &rms_cache));
    assert(rms_norm_backward(&rms_cache, DEMO_GAMMA, DEMO_UPSTREAM,
                             rms_gradient, rms_gamma_gradient));

    double he_variance = 0;
    double glorot_variance = 0;
    double he_weights[INIT_WEIGHT_COUNT] = {0};
    double he_sample_variance = 0;
    assert(initializer_variance(4, 3, INIT_HE, &he_variance));
    assert(initializer_variance(4, 3, INIT_GLOROT, &glorot_variance));
    assert(initialize_weights(he_weights, INIT_WEIGHT_COUNT, 4, 3, INIT_HE,
                              UINT32_C(41), &he_sample_variance));

    double plain_activation = 0;
    double plain_gradient = 0;
    double residual_activation = 0;
    double residual_gradient = 0;
    assert(gradient_chain(RESIDUAL_DEPTH, -0.1, false,
                          &plain_activation, &plain_gradient));
    assert(gradient_chain(RESIDUAL_DEPTH, -0.1, true,
                          &residual_activation, &residual_gradient));

    puts("field,value");
    const double *outputs[] = {layer_output, layer_gradient, batch_output, batch_gradient,
                               rms_output, rms_gradient};
    const char *const names[] = {"ln_y", "ln_dx", "bn_y", "bn_dx", "rms_y", "rms_dx"};
    for (size_t block = 0; block < sizeof(outputs) / sizeof(outputs[0]); ++block) {
        for (size_t i = 0; i < DEMO_ROWS * DEMO_FEATURES; ++i) {
            char field[32];
            int length = snprintf(field, sizeof(field), "%s_%zu", names[block], i);
            assert(length > 0 && (size_t)length < sizeof(field));
            print_csv_value(field, outputs[block][i]);
        }
    }
    for (size_t feature = 0; feature < DEMO_FEATURES; ++feature) {
        char field[32];
        assert(snprintf(field, sizeof(field), "ln_dgamma_%zu", feature) > 0);
        print_csv_value(field, layer_gamma_gradient[feature]);
        assert(snprintf(field, sizeof(field), "ln_dbeta_%zu", feature) > 0);
        print_csv_value(field, layer_beta_gradient[feature]);
        assert(snprintf(field, sizeof(field), "bn_dgamma_%zu", feature) > 0);
        print_csv_value(field, batch_gamma_gradient[feature]);
        assert(snprintf(field, sizeof(field), "bn_dbeta_%zu", feature) > 0);
        print_csv_value(field, batch_beta_gradient[feature]);
        assert(snprintf(field, sizeof(field), "rms_dgamma_%zu", feature) > 0);
        print_csv_value(field, rms_gamma_gradient[feature]);
        assert(snprintf(field, sizeof(field), "bn_running_mean_%zu", feature) > 0);
        print_csv_value(field, batch_state.running_mean[feature]);
        assert(snprintf(field, sizeof(field), "bn_running_variance_%zu", feature) > 0);
        print_csv_value(field, batch_state.running_variance[feature]);
        assert(snprintf(field, sizeof(field), "bn_inference_y_%zu", feature) > 0);
        print_csv_value(field, inference_output[feature]);
    }
    print_csv_value("ln_gradient_norm", gradient_norm(layer_gradient, DEMO_ROWS * DEMO_FEATURES));
    print_csv_value("bn_gradient_norm", gradient_norm(batch_gradient, DEMO_ROWS * DEMO_FEATURES));
    print_csv_value("rms_gradient_norm", gradient_norm(rms_gradient, DEMO_ROWS * DEMO_FEATURES));
    print_csv_value("identity_gradient_norm",
                    gradient_norm(DEMO_UPSTREAM, DEMO_ROWS * DEMO_FEATURES));
    print_csv_value("he_target_variance", he_variance);
    print_csv_value("glorot_target_variance", glorot_variance);
    print_csv_value("he_sample_variance", he_sample_variance);
    print_csv_value("plain_activation", plain_activation);
    print_csv_value("plain_gradient", plain_gradient);
    print_csv_value("residual_activation", residual_activation);
    print_csv_value("residual_gradient", residual_gradient);
}

/**
 * @brief 计算固定矩阵与上游梯度的标量内积，作为中心差分目标。
 * @param output 前向输出矩阵。
 * @param upstream 同形状上游梯度。
 * @param count 元素数量。
 * @return 内积值。
 */
static double objective(const double *output, const double *upstream, size_t count)
{
    double result = 0;
    for (size_t i = 0; i < count; ++i) {
        result += output[i] * upstream[i];
    }
    return result;
}

/**
 * @brief 用中心差分验证 LN 输入、gamma 和 beta 的解析梯度。
 */
static void test_layer_norm_gradient(void)
{
    double input[MAX_VALUES];
    double gamma[MAX_FEATURES];
    double beta[MAX_FEATURES];
    memcpy(input, DEMO_INPUT, DEMO_ROWS * DEMO_FEATURES * sizeof(*input));
    memcpy(gamma, DEMO_GAMMA, DEMO_FEATURES * sizeof(*gamma));
    memcpy(beta, DEMO_BETA, DEMO_FEATURES * sizeof(*beta));
    double output[MAX_VALUES] = {0};
    double dx[MAX_VALUES] = {0};
    double dgamma[MAX_FEATURES] = {0};
    double dbeta[MAX_FEATURES] = {0};
    LayerNormCache cache = {0};
    assert(layer_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma, beta,
                              0.01, output, &cache));
    assert(layer_norm_backward(&cache, gamma, DEMO_UPSTREAM, dx, dgamma, dbeta));
    double *parameters[] = {input, gamma, beta};
    const double *derivatives[] = {dx, dgamma, dbeta};
    const size_t lengths[] = {DEMO_ROWS * DEMO_FEATURES, DEMO_FEATURES, DEMO_FEATURES};
    const double step = 1e-6;
    for (size_t group = 0; group < 3; ++group) {
        for (size_t i = 0; i < lengths[group]; ++i) {
            double original = parameters[group][i];
            parameters[group][i] = original + step;
            assert(layer_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma, beta,
                                      0.01, output, &cache));
            double high = objective(output, DEMO_UPSTREAM, DEMO_ROWS * DEMO_FEATURES);
            parameters[group][i] = original - step;
            assert(layer_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma, beta,
                                      0.01, output, &cache));
            double low = objective(output, DEMO_UPSTREAM, DEMO_ROWS * DEMO_FEATURES);
            parameters[group][i] = original;
            double numerical = (high - low) / (2 * step);
            assert(fabs(numerical - derivatives[group][i]) < 2e-7);
        }
    }
}

/**
 * @brief 用中心差分验证训练态 BN 输入、gamma 和 beta 的解析梯度。
 */
static void test_batch_norm_gradient(void)
{
    double input[MAX_VALUES];
    double gamma[MAX_FEATURES];
    double beta[MAX_FEATURES];
    memcpy(input, DEMO_INPUT, DEMO_ROWS * DEMO_FEATURES * sizeof(*input));
    memcpy(gamma, DEMO_GAMMA, DEMO_FEATURES * sizeof(*gamma));
    memcpy(beta, DEMO_BETA, DEMO_FEATURES * sizeof(*beta));
    BatchNormState state = {0};
    BatchNormCache cache = {0};
    double output[MAX_VALUES] = {0};
    double dx[MAX_VALUES] = {0};
    double dgamma[MAX_FEATURES] = {0};
    double dbeta[MAX_FEATURES] = {0};
    assert(batch_norm_state_init(&state, DEMO_FEATURES));
    assert(batch_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma, beta,
                              0.01, 0.9, true, &state, output, &cache));
    assert(batch_norm_backward(&cache, gamma, DEMO_UPSTREAM, dx, dgamma, dbeta));
    double *parameters[] = {input, gamma, beta};
    const double *derivatives[] = {dx, dgamma, dbeta};
    const size_t lengths[] = {DEMO_ROWS * DEMO_FEATURES, DEMO_FEATURES, DEMO_FEATURES};
    const double step = 1e-6;
    for (size_t group = 0; group < 3; ++group) {
        for (size_t i = 0; i < lengths[group]; ++i) {
            double original = parameters[group][i];
            double values[2] = {0};
            for (size_t side = 0; side < 2; ++side) {
                parameters[group][i] = original + (side == 0 ? step : -step);
                BatchNormState local_state = {0};
                BatchNormCache local_cache = {0};
                assert(batch_norm_state_init(&local_state, DEMO_FEATURES));
                assert(batch_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma, beta,
                                          0.01, 0.9, true, &local_state, output, &local_cache));
                values[side] = objective(output, DEMO_UPSTREAM, DEMO_ROWS * DEMO_FEATURES);
            }
            parameters[group][i] = original;
            double numerical = (values[0] - values[1]) / (2 * step);
            assert(fabs(numerical - derivatives[group][i]) < 2e-7);
        }
    }
}

/**
 * @brief 用中心差分验证 RMSNorm 输入和 gamma 的解析梯度。
 */
static void test_rms_norm_gradient(void)
{
    double input[MAX_VALUES];
    double gamma[MAX_FEATURES];
    memcpy(input, DEMO_INPUT, DEMO_ROWS * DEMO_FEATURES * sizeof(*input));
    memcpy(gamma, DEMO_GAMMA, DEMO_FEATURES * sizeof(*gamma));
    double output[MAX_VALUES] = {0};
    double dx[MAX_VALUES] = {0};
    double dgamma[MAX_FEATURES] = {0};
    RmsNormCache cache = {0};
    assert(rms_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma,
                            0.01, output, &cache));
    assert(rms_norm_backward(&cache, gamma, DEMO_UPSTREAM, dx, dgamma));
    double *parameters[] = {input, gamma};
    const double *derivatives[] = {dx, dgamma};
    const size_t lengths[] = {DEMO_ROWS * DEMO_FEATURES, DEMO_FEATURES};
    const double step = 1e-6;
    for (size_t group = 0; group < 2; ++group) {
        for (size_t i = 0; i < lengths[group]; ++i) {
            double original = parameters[group][i];
            parameters[group][i] = original + step;
            assert(rms_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma,
                                    0.01, output, &cache));
            double high = objective(output, DEMO_UPSTREAM, DEMO_ROWS * DEMO_FEATURES);
            parameters[group][i] = original - step;
            assert(rms_norm_forward(input, DEMO_ROWS, DEMO_FEATURES, gamma,
                                    0.01, output, &cache));
            double low = objective(output, DEMO_UPSTREAM, DEMO_ROWS * DEMO_FEATURES);
            parameters[group][i] = original;
            double numerical = (high - low) / (2 * step);
            assert(fabs(numerical - derivatives[group][i]) < 2e-7);
        }
    }
}

/**
 * @brief 检查手算锚点、推理状态、退化输入、失败原子性和初始化边界。
 */
static void test_boundaries(void)
{
    double output[MAX_VALUES];
    LayerNormCache layer_cache = {0};
    assert(layer_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, output, &layer_cache));
    double expected_layer = -1.0 / sqrt(2.0 / 3.0 + 0.01) * DEMO_GAMMA[0] + DEMO_BETA[0];
    assert(fabs(output[0] - expected_layer) < 1e-12);

    BatchNormState state = {0};
    BatchNormCache batch_cache = {0};
    double batch_output[MAX_VALUES] = {0};
    assert(batch_norm_state_init(&state, DEMO_FEATURES));
    assert(batch_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, 0.9, true, &state,
                              batch_output, &batch_cache));
    double expected_batch = -0.5 / sqrt(0.25 + 0.01) * DEMO_GAMMA[0] + DEMO_BETA[0];
    assert(fabs(batch_output[0] - expected_batch) < 1e-12);
    double saved_state = state.running_mean[0];
    double evaluation_sample[DEMO_FEATURES] = {3, 6, 9};
    double evaluation_output[DEMO_FEATURES] = {0};
    BatchNormCache evaluation_cache = {0};
    assert(batch_norm_forward(evaluation_sample, 1, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, 0.9, false, &state,
                              evaluation_output, &evaluation_cache));
    assert(state.running_mean[0] == saved_state);
    assert(!evaluation_cache.training);
    double expected_inference = DEMO_GAMMA[0] *
        (evaluation_sample[0] - state.running_mean[0]) /
        sqrt(state.running_variance[0] + 0.01) + DEMO_BETA[0];
    assert(fabs(evaluation_output[0] - expected_inference) < 1e-12);

    const double singleton[2] = {5, -2};
    BatchNormState singleton_state = {0};
    BatchNormCache singleton_cache = {0};
    double singleton_output[2] = {0};
    double singleton_gradient[2] = {0};
    double singleton_dgamma[2] = {0};
    double singleton_dbeta[2] = {0};
    const double singleton_gamma[2] = {2, -3};
    const double singleton_beta[2] = {0.25, -0.75};
    assert(batch_norm_state_init(&singleton_state, 2));
    assert(batch_norm_forward(singleton, 1, 2, singleton_gamma, singleton_beta,
                              1e-5, 0.9, true, &singleton_state,
                              singleton_output, &singleton_cache));
    assert(singleton_output[0] == singleton_beta[0]);
    assert(singleton_output[1] == singleton_beta[1]);
    assert(batch_norm_backward(&singleton_cache, singleton_gamma, singleton,
                               singleton_gradient, singleton_dgamma, singleton_dbeta));
    assert(singleton_gradient[0] == 0 && singleton_gradient[1] == 0);
    assert(singleton_dgamma[0] == 0 && singleton_dgamma[1] == 0);

    const double constant_input[3] = {4, 4, 4};
    const double unit_gamma[3] = {1, 1, 1};
    const double zero_beta[3] = {0, 0, 0};
    LayerNormCache constant_layer_cache = {0};
    RmsNormCache constant_rms_cache = {0};
    double constant_output[3] = {0};
    assert(layer_norm_forward(constant_input, 1, 3, unit_gamma, zero_beta,
                              1e-5, constant_output, &constant_layer_cache));
    assert(rms_norm_forward(constant_input, 1, 3, unit_gamma,
                            1e-5, constant_output, &constant_rms_cache));
    assert(values_are_finite(constant_output, 3));

    double preserved_output[DEMO_ROWS * DEMO_FEATURES];
    memset(preserved_output, 0x5a, sizeof(preserved_output));
    double old_output[DEMO_ROWS * DEMO_FEATURES];
    memcpy(old_output, preserved_output, sizeof(old_output));
    LayerNormCache old_cache = {.rows = 3, .features = 2};
    LayerNormCache saved_cache = old_cache;
    double invalid_input[MAX_VALUES];
    memcpy(invalid_input, DEMO_INPUT, DEMO_ROWS * DEMO_FEATURES * sizeof(*invalid_input));
    invalid_input[0] = NAN;
    assert(!layer_norm_forward(invalid_input, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                               DEMO_BETA, 0.01, preserved_output, &old_cache));
    assert(memcmp(preserved_output, old_output, sizeof(old_output)) == 0);
    assert(old_cache.rows == saved_cache.rows && old_cache.features == saved_cache.features);
    assert(memcmp(old_cache.normalized, saved_cache.normalized,
                  sizeof(old_cache.normalized)) == 0);
    assert(memcmp(old_cache.inverse_scale, saved_cache.inverse_scale,
                  sizeof(old_cache.inverse_scale)) == 0);
    assert(!layer_norm_forward(DEMO_INPUT, 0, DEMO_FEATURES, DEMO_GAMMA,
                               DEMO_BETA, 0.01, preserved_output, &old_cache));
    assert(!layer_norm_forward(DEMO_INPUT, MAX_ROWS + 1, DEMO_FEATURES, DEMO_GAMMA,
                               DEMO_BETA, 0.01, preserved_output, &old_cache));
    assert(!layer_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                               DEMO_BETA, 0, preserved_output, &old_cache));

    BatchNormState preserved_state = {0};
    assert(batch_norm_state_init(&preserved_state, DEMO_FEATURES));
    preserved_state.running_mean[0] = 0.25;
    BatchNormState old_state = preserved_state;
    BatchNormCache old_batch_cache = {.rows = 3, .features = 2, .training = true};
    BatchNormCache saved_batch_cache = old_batch_cache;
    assert(!batch_norm_forward(invalid_input, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                               DEMO_BETA, 0.01, 0.9, true, &preserved_state,
                               preserved_output, &old_batch_cache));
    assert(preserved_state.features == old_state.features &&
           preserved_state.initialized == old_state.initialized);
    assert(memcmp(preserved_state.running_mean, old_state.running_mean,
                  sizeof(old_state.running_mean)) == 0);
    assert(memcmp(preserved_state.running_variance, old_state.running_variance,
                  sizeof(old_state.running_variance)) == 0);
    assert(memcmp(preserved_output, old_output, sizeof(old_output)) == 0);
    assert(old_batch_cache.rows == saved_batch_cache.rows &&
           old_batch_cache.features == saved_batch_cache.features &&
           old_batch_cache.training == saved_batch_cache.training);
    assert(memcmp(old_batch_cache.normalized, saved_batch_cache.normalized,
                  sizeof(old_batch_cache.normalized)) == 0);
    assert(memcmp(old_batch_cache.inverse_scale, saved_batch_cache.inverse_scale,
                  sizeof(old_batch_cache.inverse_scale)) == 0);

    double variance = -1;
    assert(initializer_variance(4, 3, INIT_HE, &variance));
    assert(fabs(variance - 0.5) < 1e-15);
    assert(!initializer_variance(0, 3, INIT_HE, &variance));
    assert(variance == 0.5);
    double plain_value = 0;
    double plain_gradient = 0;
    double residual_value = 0;
    double residual_gradient = 0;
    assert(gradient_chain(RESIDUAL_DEPTH, -0.1, false, &plain_value, &plain_gradient));
    assert(gradient_chain(RESIDUAL_DEPTH, -0.1, true, &residual_value, &residual_gradient));
    assert(fabs(plain_gradient - pow(-0.1, RESIDUAL_DEPTH)) < 1e-20);
    assert(fabs(residual_gradient - pow(0.9, RESIDUAL_DEPTH)) < 1e-15);
    assert(residual_gradient > plain_gradient * 1e7);
    double step = 1e-6;
    double point = 0.37;
    double high = residual_value * (point + step);
    double low = residual_value * (point - step);
    assert(fabs((high - low) / (2 * step) - residual_gradient) < 1e-10);
    double preserved_value = 7;
    double preserved_gradient = 8;
    assert(!gradient_chain(0, -0.1, true, &preserved_value, &preserved_gradient));
    assert(preserved_value == 7 && preserved_gradient == 8);
}

/**
 * @brief 检查三种归一化的全部可训练输入梯度及残差梯度。
 */
static void run_tests(void)
{
    test_layer_norm_gradient();
    test_batch_norm_gradient();
    test_rms_norm_gradient();
    test_boundaries();
}

/**
 * @brief 以简洁报告展示固定批次的输出、梯度范数和残差路径。
 */
static void print_demo(void)
{
    double layer_output[MAX_VALUES] = {0};
    double layer_gradient[MAX_VALUES] = {0};
    double layer_dgamma[MAX_FEATURES] = {0};
    double layer_dbeta[MAX_FEATURES] = {0};
    LayerNormCache layer_cache = {0};
    assert(layer_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, layer_output, &layer_cache));
    assert(layer_norm_backward(&layer_cache, DEMO_GAMMA, DEMO_UPSTREAM,
                               layer_gradient, layer_dgamma, layer_dbeta));

    BatchNormState batch_state = {0};
    BatchNormCache batch_cache = {0};
    double batch_output[MAX_VALUES] = {0};
    double batch_gradient[MAX_VALUES] = {0};
    double batch_dgamma[MAX_FEATURES] = {0};
    double batch_dbeta[MAX_FEATURES] = {0};
    assert(batch_norm_state_init(&batch_state, DEMO_FEATURES));
    assert(batch_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                              DEMO_BETA, 0.01, 0.9, true, &batch_state,
                              batch_output, &batch_cache));
    assert(batch_norm_backward(&batch_cache, DEMO_GAMMA, DEMO_UPSTREAM,
                               batch_gradient, batch_dgamma, batch_dbeta));

    RmsNormCache rms_cache = {0};
    double rms_output[MAX_VALUES] = {0};
    double rms_gradient[MAX_VALUES] = {0};
    double rms_dgamma[MAX_FEATURES] = {0};
    assert(rms_norm_forward(DEMO_INPUT, DEMO_ROWS, DEMO_FEATURES, DEMO_GAMMA,
                            0.01, rms_output, &rms_cache));
    assert(rms_norm_backward(&rms_cache, DEMO_GAMMA, DEMO_UPSTREAM,
                             rms_gradient, rms_dgamma));

    double plain_activation = 0;
    double plain_gradient = 0;
    double residual_activation = 0;
    double residual_gradient = 0;
    assert(gradient_chain(RESIDUAL_DEPTH, -0.1, false,
                          &plain_activation, &plain_gradient));
    assert(gradient_chain(RESIDUAL_DEPTH, -0.1, true,
                          &residual_activation, &residual_gradient));

    printf("输入 X=[1,2,3; 2,4,6]，gamma=[1.2,0.8,-0.5]，epsilon=0.01\n");
    printf("LayerNorm 第一行输出：[%.6f, %.6f, %.6f]\n",
           layer_output[0], layer_output[1], layer_output[2]);
    printf("BatchNorm 第一行输出：[%.6f, %.6f, %.6f]\n",
           batch_output[0], batch_output[1], batch_output[2]);
    printf("RMSNorm 第一行输出：[%.6f, %.6f, %.6f]\n",
           rms_output[0], rms_output[1], rms_output[2]);
    printf("输入梯度 L2：恒等基线 %.6f，LayerNorm %.6f，BatchNorm %.6f，RMSNorm %.6f\n",
           gradient_norm(DEMO_UPSTREAM, DEMO_ROWS * DEMO_FEATURES),
           gradient_norm(layer_gradient, DEMO_ROWS * DEMO_FEATURES),
           gradient_norm(batch_gradient, DEMO_ROWS * DEMO_FEATURES),
           gradient_norm(rms_gradient, DEMO_ROWS * DEMO_FEATURES));
    printf("8 层主分支导数 -0.1：普通链梯度 %.3e，恒等残差梯度 %.6f\n",
           plain_gradient, residual_gradient);
}

/**
 * @brief 打印数值实验摘要或运行无副作用断言。
 * @param argc 命令行参数数量。
 * @param argv 命令行参数；支持无参数、--test 和 --csv。
 * @return 成功为 EXIT_SUCCESS；参数错误或输出错误时失败。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        run_tests();
        puts("D01 C 数值与边界检查通过。");
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        print_csv_fixture();
        return ferror(stdout) ? 1 : 0;
    }
    if (argc != 1) {
        fprintf(stderr, "用法：%s [--test | --csv]\n", argv[0]);
        return 2;
    }

    run_tests();
    printf("=== 归一化、初始化与残差梯度实验（固定 2×3 输入）===\n");
    print_demo();
    return ferror(stdout) ? 1 : 0;
}
