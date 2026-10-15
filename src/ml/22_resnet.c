/**
 * @file
 * @brief 阶段 11.1：以纯 C 实现同尺寸卷积残差块并比较恒等捷径。
 * 普通 CNN 与 ResNet 共享参数布局、数据、初始化和训练轮数。
 */
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SIDE = 8,
    PIXELS = SIDE * SIDE,
    KERNEL_SIDE = 3,
    KERNEL_SIZE = KERNEL_SIDE * KERNEL_SIDE,
    CHANNELS = 2,
    TENSOR_SIZE = CHANNELS * PIXELS,
    CLASS_COUNT = 3,
    TRAIN_COUNT = 60,
    VALIDATION_COUNT = 60,
    TEST_COUNT = 120,
    DATASET_CAPACITY = TEST_COUNT,
    EPOCH_COUNT = 50,
    TRIAL_COUNT = 3,
    STEM_W = 0,
    STEM_B = STEM_W + CHANNELS * KERNEL_SIZE,
    CONV1_W = STEM_B + CHANNELS,
    CONV1_B = CONV1_W + CHANNELS * CHANNELS * KERNEL_SIZE,
    CONV2_W = CONV1_B + CHANNELS,
    CONV2_B = CONV2_W + CHANNELS * CHANNELS * KERNEL_SIZE,
    HEAD_W = CONV2_B + CHANNELS,
    HEAD_B = HEAD_W + CLASS_COUNT * CHANNELS,
    PARAMETER_COUNT = HEAD_B + CLASS_COUNT
};

typedef struct {
    double pixels[PIXELS];
    size_t target;
} Sample;

typedef struct {
    size_t count;
    Sample samples[DATASET_CAPACITY];
} Dataset;

typedef struct {
    double parameters[PARAMETER_COUNT];
} Network;

typedef struct {
    double stem_pre[TENSOR_SIZE];
    double stem[TENSOR_SIZE];
    double conv1_pre[TENSOR_SIZE];
    double conv1[TENSOR_SIZE];
    double branch[TENSOR_SIZE];
    double block_pre[TENSOR_SIZE];
    double block[TENSOR_SIZE];
    double pooled[CHANNELS];
    double logits[CLASS_COUNT];
    double probabilities[CLASS_COUNT];
} Forward;

typedef struct {
    double loss;
    double accuracy;
} Metrics;

/**
 * @brief 推进局部伪随机数发生器，得到 [0,1) 内的数。
 * @param state 可写随机状态，由调用者独占维护。
 * @return 状态高 24 位映射得到的伪随机数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 生成固定种子的三类 8×8 线条图像，包含平移、亮度与像素噪声。
 * @param dataset 有效可写数据集，成功后覆盖旧内容。
 * @param count 正样本数，须为三的倍数且不超过 DATASET_CAPACITY。
 * @param seed 数据种子；训练、验证、测试数据使用不同种子。
 * @return 参数有效为 true；失败时不修改 dataset。
 */
static bool dataset_generate(Dataset *dataset, size_t count, uint32_t seed)
{
    if (dataset == nullptr || count == 0 || count > DATASET_CAPACITY ||
        count % CLASS_COUNT != 0) {
        return false;
    }
    Dataset generated = {0};
    generated.count = count;
    for (size_t i = 0; i < count; ++i) {
        Sample *sample = &generated.samples[i];
        sample->target = i % CLASS_COUNT;
        double center_x = 2.5 + 3.0 * random_unit(&seed);
        double center_y = 2.5 + 3.0 * random_unit(&seed);
        double brightness = 0.75 + 0.25 * random_unit(&seed);
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                double x = (double)column - center_x;
                double y = (double)row - center_y;
                bool inside = false;
                if (sample->target == 0) {
                    inside = fabs(x) <= 0.75;
                } else if (sample->target == 1) {
                    inside = fabs(y) <= 0.75;
                } else {
                    inside = fabs(x - y) <= 0.85;
                }
                double value = inside ? brightness : 0.05;
                value += 0.08 * (random_unit(&seed) - 0.5);
                sample->pixels[row * SIDE + column] = fmin(1.0, fmax(0.0, value));
            }
        }
    }
    *dataset = generated;
    return true;
}

/**
 * @brief 对多个通道执行步长一、零填充的 3×3 互相关。
 * @param input 输入张量，按 [channel][row][column] 连续存储。
 * @param input_channels 输入通道数，须为正数。
 * @param weights 卷积核，按 [out][in][ky][kx] 存储。
 * @param bias 每个输出通道一个偏置。
 * @param output_channels 输出通道数，须为正数。
 * @param output 可写输出张量，容量 output_channels*PIXELS。
 */
static void convolution_forward(const double *input, size_t input_channels,
                                const double *weights, const double *bias,
                                size_t output_channels, double *output)
{
    for (size_t out = 0; out < output_channels; ++out) {
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                double value = bias[out];
                for (size_t in = 0; in < input_channels; ++in) {
                    for (size_t ky = 0; ky < KERNEL_SIDE; ++ky) {
                        for (size_t kx = 0; kx < KERNEL_SIDE; ++kx) {
                            int input_y = (int)row + (int)ky - 1;
                            int input_x = (int)column + (int)kx - 1;
                            if (input_y < 0 || input_y >= SIDE || input_x < 0 || input_x >= SIDE) {
                                continue;
                            }
                            size_t weight_index =
                                ((out * input_channels + in) * KERNEL_SIDE + ky) * KERNEL_SIDE + kx;
                            size_t input_index = in * PIXELS + (size_t)input_y * SIDE +
                                                 (size_t)input_x;
                            value += weights[weight_index] * input[input_index];
                        }
                    }
                }
                output[out * PIXELS + row * SIDE + column] = value;
            }
        }
    }
}

/**
 * @brief 对卷积执行反向传播，累加权重/偏置梯度并写输入梯度。
 * @param input 与该卷积前向相同的只读输入张量。
 * @param input_channels 输入通道数，须与权重布局一致。
 * @param weights 前向使用的只读卷积核。
 * @param output_channels 输出通道数。
 * @param output_gradient 输出激活的上游梯度。
 * @param input_gradient 有效可写输入梯度，调用前须清零。
 * @param weight_gradient 可写卷积核梯度，累加到已有值。
 * @param bias_gradient 可写偏置梯度，累加到已有值。
 */
static void convolution_backward(const double *input, size_t input_channels,
                                 const double *weights, size_t output_channels,
                                 const double *output_gradient, double *input_gradient,
                                 double *weight_gradient, double *bias_gradient)
{
    for (size_t out = 0; out < output_channels; ++out) {
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                size_t output_index = out * PIXELS + row * SIDE + column;
                double delta = output_gradient[output_index];
                bias_gradient[out] += delta;
                for (size_t in = 0; in < input_channels; ++in) {
                    for (size_t ky = 0; ky < KERNEL_SIDE; ++ky) {
                        for (size_t kx = 0; kx < KERNEL_SIDE; ++kx) {
                            int input_y = (int)row + (int)ky - 1;
                            int input_x = (int)column + (int)kx - 1;
                            if (input_y < 0 || input_y >= SIDE || input_x < 0 || input_x >= SIDE) {
                                continue;
                            }
                            size_t weight_index =
                                ((out * input_channels + in) * KERNEL_SIDE + ky) * KERNEL_SIDE + kx;
                            size_t input_index = in * PIXELS + (size_t)input_y * SIDE +
                                                 (size_t)input_x;
                            weight_gradient[weight_index] += delta * input[input_index];
                            input_gradient[input_index] += delta * weights[weight_index];
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief 合并残差块主分支与恒等捷径，并通过 ReLU。
 * @param skip 恒等捷径输入；普通网络传 nullptr 表示无捷径。
 * @param branch 两层卷积产生的主分支张量。
 * @param preactivation 可写加法结果缓存。
 * @param output 可写 ReLU 输出。
 */
static void residual_merge(const double *skip, const double *branch,
                           double preactivation[TENSOR_SIZE], double output[TENSOR_SIZE])
{
    for (size_t i = 0; i < TENSOR_SIZE; ++i) {
        preactivation[i] = branch[i];
        if (skip != nullptr) {
            preactivation[i] += skip[i];
        }
        output[i] = fmax(0.0, preactivation[i]);
    }
}

/**
 * @brief 反传残差合并处的 ReLU；恒等路径将上游梯度直接加回输入。
 * @param preactivation 前向加法结果，只读。
 * @param output_gradient ReLU 输出的上游梯度，只读。
 * @param branch_gradient 主分支梯度输出。
 * @param skip_gradient 恒等路径梯度输出，调用前须清零。
 */
static void residual_merge_backward(const double preactivation[TENSOR_SIZE],
                                    const double output_gradient[TENSOR_SIZE],
                                    double branch_gradient[TENSOR_SIZE],
                                    double *skip_gradient)
{
    for (size_t i = 0; i < TENSOR_SIZE; ++i) {
        double gradient = preactivation[i] > 0.0 ? output_gradient[i] : 0.0;
        branch_gradient[i] = gradient;
        if (skip_gradient != nullptr) {
            skip_gradient[i] += gradient;
        }
    }
}

/**
 * @brief 对减最大值的三个 logit 计算稳定 softmax。
 * @param logits 三个有限类别分数。
 * @param probabilities 三个可写输出概率。
 */
static void softmax(const double logits[CLASS_COUNT], double probabilities[CLASS_COUNT])
{
    double maximum = logits[0];
    for (size_t c = 1; c < CLASS_COUNT; ++c) {
        maximum = fmax(maximum, logits[c]);
    }
    double total = 0.0;
    for (size_t c = 0; c < CLASS_COUNT; ++c) {
        probabilities[c] = exp(logits[c] - maximum);
        total += probabilities[c];
    }
    for (size_t c = 0; c < CLASS_COUNT; ++c) {
        probabilities[c] /= total;
    }
}

/**
 * @brief 计算一个样本的前向传播缓存。
 * @param network 只读网络，参数须有限。
 * @param sample 含 PIXELS 个有限灰度值和有效类别标签的样本。
 * @param use_residual true 时使用 stem 输出作为恒等捷径。
 * @param output 有效缓存地址，成功后覆盖全部字段。
 * @return 计算成功且结果有限时为 true。
 */
static bool network_forward(const Network *network, const Sample *sample, bool use_residual,
                            Forward *output)
{
    if (network == nullptr || sample == nullptr || output == nullptr ||
        sample->target >= CLASS_COUNT) {
        return false;
    }
    for (size_t p = 0; p < PARAMETER_COUNT; ++p) {
        if (!isfinite(network->parameters[p])) {
            return false;
        }
    }
    for (size_t p = 0; p < PIXELS; ++p) {
        if (!isfinite(sample->pixels[p])) {
            return false;
        }
    }
    Forward result = {0};
    convolution_forward(sample->pixels, 1, &network->parameters[STEM_W],
                        &network->parameters[STEM_B], CHANNELS, result.stem_pre);
    for (size_t i = 0; i < TENSOR_SIZE; ++i) {
        result.stem[i] = fmax(0.0, result.stem_pre[i]);
    }
    convolution_forward(result.stem, CHANNELS, &network->parameters[CONV1_W],
                        &network->parameters[CONV1_B], CHANNELS, result.conv1_pre);
    for (size_t i = 0; i < TENSOR_SIZE; ++i) {
        result.conv1[i] = fmax(0.0, result.conv1_pre[i]);
    }
    convolution_forward(result.conv1, CHANNELS, &network->parameters[CONV2_W],
                        &network->parameters[CONV2_B], CHANNELS, result.branch);
    const double *skip = use_residual ? result.stem : nullptr;
    residual_merge(skip, result.branch, result.block_pre, result.block);
    for (size_t channel = 0; channel < CHANNELS; ++channel) {
        for (size_t pixel = 0; pixel < PIXELS; ++pixel) {
            result.pooled[channel] += result.block[channel * PIXELS + pixel] / (double)PIXELS;
        }
    }
    for (size_t c = 0; c < CLASS_COUNT; ++c) {
        result.logits[c] = network->parameters[HEAD_B + c];
        for (size_t channel = 0; channel < CHANNELS; ++channel) {
            result.logits[c] += network->parameters[HEAD_W + c * CHANNELS + channel] *
                                result.pooled[channel];
        }
    }
    for (size_t c = 0; c < CLASS_COUNT; ++c) {
        if (!isfinite(result.logits[c])) {
            return false;
        }
    }
    softmax(result.logits, result.probabilities);
    *output = result;
    return true;
}

/**
 * @brief 累加单样本交叉熵对网络参数的梯度，可选计算输入梯度。
 * @param network 当前只读网络。
 * @param sample 当前只读样本。
 * @param use_residual 是否启用恒等捷径。
 * @param parameter_gradient 参数梯度累加区，调用前由调用者清零。
 * @param input_gradient 可选 PIXELS 元素输入梯度输出。
 * @param loss 可选的单样本交叉熵输出。
 * @return 前向和反向结果均有限时为 true。
 */
static bool sample_backward(const Network *network, const Sample *sample, bool use_residual,
                           double parameter_gradient[PARAMETER_COUNT],
                           double input_gradient[PIXELS], double *loss)
{
    Forward pass;
    if (!network_forward(network, sample, use_residual, &pass)) {
        return false;
    }
    double output_delta[CLASS_COUNT];
    double block_gradient[TENSOR_SIZE] = {0};
    double branch_gradient[TENSOR_SIZE];
    double skip_gradient[TENSOR_SIZE] = {0};
    double conv1_gradient[TENSOR_SIZE] = {0};
    double stem_gradient[TENSOR_SIZE] = {0};
    double stem_gradient_from_branch[TENSOR_SIZE] = {0};
    double local_input_gradient[PIXELS] = {0};
    for (size_t c = 0; c < CLASS_COUNT; ++c) {
        output_delta[c] = pass.probabilities[c] - (c == sample->target ? 1.0 : 0.0);
        parameter_gradient[HEAD_B + c] += output_delta[c];
        for (size_t channel = 0; channel < CHANNELS; ++channel) {
            parameter_gradient[HEAD_W + c * CHANNELS + channel] +=
                output_delta[c] * pass.pooled[channel];
            double pooled_gradient = output_delta[c] *
                                    network->parameters[HEAD_W + c * CHANNELS + channel];
            for (size_t pixel = 0; pixel < PIXELS; ++pixel) {
                block_gradient[channel * PIXELS + pixel] += pooled_gradient / (double)PIXELS;
            }
        }
    }
    residual_merge_backward(pass.block_pre, block_gradient, branch_gradient,
                            use_residual ? skip_gradient : nullptr);
    convolution_backward(pass.conv1, CHANNELS, &network->parameters[CONV2_W], CHANNELS,
                         branch_gradient, conv1_gradient,
                         &parameter_gradient[CONV2_W], &parameter_gradient[CONV2_B]);
    for (size_t i = 0; i < TENSOR_SIZE; ++i) {
        if (pass.conv1_pre[i] <= 0.0) {
            conv1_gradient[i] = 0.0;
        }
    }
    convolution_backward(pass.stem, CHANNELS, &network->parameters[CONV1_W], CHANNELS,
                         conv1_gradient, stem_gradient_from_branch,
                         &parameter_gradient[CONV1_W], &parameter_gradient[CONV1_B]);
    for (size_t i = 0; i < TENSOR_SIZE; ++i) {
        stem_gradient[i] = stem_gradient_from_branch[i] + skip_gradient[i];
        if (pass.stem_pre[i] <= 0.0) {
            stem_gradient[i] = 0.0;
        }
    }
    convolution_backward(sample->pixels, 1, &network->parameters[STEM_W], CHANNELS,
                         stem_gradient, local_input_gradient,
                         &parameter_gradient[STEM_W], &parameter_gradient[STEM_B]);
    if (input_gradient != nullptr) {
        memcpy(input_gradient, local_input_gradient, sizeof local_input_gradient);
    }
    if (loss != nullptr) {
        double maximum = pass.logits[0];
        for (size_t c = 1; c < CLASS_COUNT; ++c) {
            maximum = fmax(maximum, pass.logits[c]);
        }
        double exponential_sum = 0.0;
        for (size_t c = 0; c < CLASS_COUNT; ++c) {
            exponential_sum += exp(pass.logits[c] - maximum);
        }
        *loss = maximum - pass.logits[sample->target] + log(exponential_sum);
    }
    return true;
}

/**
 * @brief 初始化卷积核与分类权重，并设置小的正卷积偏置。
 * @param network 有效可写网络，旧参数被整体覆盖。
 * @param seed 固定初始化种子。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    for (size_t p = STEM_W; p < STEM_B; ++p) {
        network->parameters[p] = (2.0 * random_unit(&seed) - 1.0) * sqrt(6.0 / 10.0);
    }
    for (size_t p = CONV1_W; p < CONV1_B; ++p) {
        network->parameters[p] = (2.0 * random_unit(&seed) - 1.0) * sqrt(6.0 / 18.0);
    }
    for (size_t p = CONV2_W; p < CONV2_B; ++p) {
        network->parameters[p] = (2.0 * random_unit(&seed) - 1.0) * sqrt(6.0 / 18.0);
    }
    for (size_t p = HEAD_W; p < HEAD_B; ++p) {
        network->parameters[p] = (2.0 * random_unit(&seed) - 1.0) * sqrt(3.0 / CHANNELS);
    }
    for (size_t channel = 0; channel < CHANNELS; ++channel) {
        network->parameters[STEM_B + channel] = 0.03 * (double)(channel + 1);
        network->parameters[CONV1_B + channel] = 0.04 * (double)(channel + 1);
        network->parameters[CONV2_B + channel] = 0.05 * (double)(channel + 1);
    }
}

/**
 * @brief 计算整批平均交叉熵和梯度，所有梯度来自同一参数快照。
 * @param network 当前只读网络。
 * @param dataset 非空只读数据集。
 * @param use_residual 是否启用恒等捷径。
 * @param gradient 可选参数梯度输出；失败时保持原样。
 * @param loss 可选平均数据损失输出；失败时保持原样。
 * @return 样本均有效且梯度有限时为 true。
 */
static bool loss_and_gradient(const Network *network, const Dataset *dataset, bool use_residual,
                              double gradient[PARAMETER_COUNT], double *loss)
{
    if (network == nullptr || dataset == nullptr || dataset->count == 0 ||
        dataset->count > DATASET_CAPACITY) {
        return false;
    }
    double accumulated[PARAMETER_COUNT] = {0};
    double total_loss = 0.0;
    for (size_t i = 0; i < dataset->count; ++i) {
        double sample_loss;
        if (!sample_backward(network, &dataset->samples[i], use_residual, accumulated, nullptr,
                             &sample_loss)) {
            return false;
        }
        total_loss += sample_loss;
    }
    double inverse_count = 1.0 / (double)dataset->count;
    total_loss *= inverse_count;
    for (size_t p = 0; p < PARAMETER_COUNT; ++p) {
        accumulated[p] *= inverse_count;
        if (!isfinite(accumulated[p])) {
            return false;
        }
    }
    if (!isfinite(total_loss)) {
        return false;
    }
    if (gradient != nullptr) {
        memcpy(gradient, accumulated, sizeof accumulated);
    }
    if (loss != nullptr) {
        *loss = total_loss;
    }
    return true;
}

/**
 * @brief 以全批量 SGD 训练固定轮数，每轮只提交一次完整平均梯度。
 * @param network 有效可写网络。
 * @param dataset 非空只读训练数据。
 * @param use_residual 是否启用恒等捷径。
 * @param epochs 正训练轮数。
 * @param learning_rate 有限正学习率。
 * @return 所有梯度与参数均有限时为 true。
 */
static bool train(Network *network, const Dataset *dataset, bool use_residual,
                  size_t epochs, double learning_rate)
{
    if (network == nullptr || dataset == nullptr || dataset->count == 0 || epochs == 0 ||
        !isfinite(learning_rate) || learning_rate <= 0.0) {
        return false;
    }
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double gradient[PARAMETER_COUNT];
        if (!loss_and_gradient(network, dataset, use_residual, gradient, nullptr)) {
            return false;
        }
        for (size_t p = 0; p < PARAMETER_COUNT; ++p) {
            double updated = network->parameters[p] - learning_rate * gradient[p];
            if (!isfinite(updated)) {
                return false;
            }
            network->parameters[p] = updated;
        }
    }
    return true;
}

/**
 * @brief 在数据集上计算平均交叉熵和 top-1 准确率。
 * @param network 只读已初始化网络。
 * @param dataset 非空只读数据集。
 * @param use_residual 是否启用恒等捷径。
 * @param output 有效可写指标，成功后整体替换。
 * @return 全部样本评估成功时为 true，否则不修改 output。
 */
static bool evaluate(const Network *network, const Dataset *dataset, bool use_residual,
                     Metrics *output)
{
    if (network == nullptr || dataset == nullptr || dataset->count == 0 ||
        dataset->count > DATASET_CAPACITY || output == nullptr) {
        return false;
    }
    Metrics result = {0};
    for (size_t i = 0; i < dataset->count; ++i) {
        Forward pass;
        if (!network_forward(network, &dataset->samples[i], use_residual, &pass)) {
            return false;
        }
        double maximum = pass.logits[0];
        size_t prediction = 0;
        for (size_t c = 1; c < CLASS_COUNT; ++c) {
            if (pass.logits[c] > maximum) {
                maximum = pass.logits[c];
                prediction = c;
            }
        }
        double exponential_sum = 0.0;
        for (size_t c = 0; c < CLASS_COUNT; ++c) {
            exponential_sum += exp(pass.logits[c] - maximum);
        }
        result.loss += maximum - pass.logits[dataset->samples[i].target] + log(exponential_sum);
        if (prediction == dataset->samples[i].target) {
            result.accuracy += 1.0;
        }
    }
    result.loss /= (double)dataset->count;
    result.accuracy /= (double)dataset->count;
    if (!isfinite(result.loss) || !isfinite(result.accuracy)) {
        return false;
    }
    *output = result;
    return true;
}

/**
 * @brief 用中心差分验证一个网络模式的全部参数和一个输入像素梯度。
 * @param use_residual true 检查带恒等捷径网络，false 检查普通 CNN。
 * @return 相对误差均在容差内时为 true。
 */
static bool check_gradients(bool use_residual)
{
    Dataset dataset;
    if (!dataset_generate(&dataset, 3, UINT32_C(2026))) {
        return false;
    }
    Network network;
    network_initialize(&network, UINT32_C(91));
    double analytic[PARAMETER_COUNT] = {0};
    double analytic_loss;
    if (!loss_and_gradient(&network, &dataset, use_residual, analytic, &analytic_loss)) {
        fputs("无法计算解析梯度\n", stderr);
        return false;
    }
    const double epsilon = 1e-5;
    for (size_t p = 0; p < PARAMETER_COUNT; ++p) {
        double original = network.parameters[p];
        network.parameters[p] = original + epsilon;
        double plus;
        if (!loss_and_gradient(&network, &dataset, use_residual, nullptr, &plus)) {
            fprintf(stderr, "参数 %zu 正向扰动计算失败\n", p);
            return false;
        }
        network.parameters[p] = original - epsilon;
        double minus;
        if (!loss_and_gradient(&network, &dataset, use_residual, nullptr, &minus)) {
            fprintf(stderr, "参数 %zu 负向扰动计算失败\n", p);
            return false;
        }
        network.parameters[p] = original;
        double numeric = (plus - minus) / (2.0 * epsilon);
        double scale = fmax(1.0, fmax(fabs(numeric), fabs(analytic[p])));
        if (fabs(numeric - analytic[p]) / scale > 2e-5) {
            fprintf(stderr, "模式 %s 参数 %zu 梯度不匹配：解析 %.8f，数值 %.8f\n",
                    use_residual ? "ResNet" : "CNN", p, analytic[p], numeric);
            return false;
        }
    }
    double input_gradient[PIXELS];
    double parameter_gradient[PARAMETER_COUNT] = {0};
    if (!sample_backward(&network, &dataset.samples[0], use_residual, parameter_gradient,
                         input_gradient, nullptr)) {
        fputs("无法计算输入梯度\n", stderr);
        return false;
    }
    double original_pixel = dataset.samples[0].pixels[17];
    dataset.samples[0].pixels[17] = original_pixel + epsilon;
    double plus;
    if (!loss_and_gradient(&network, &dataset, use_residual, nullptr, &plus)) {
        fputs("输入正向扰动计算失败\n", stderr);
        return false;
    }
    dataset.samples[0].pixels[17] = original_pixel - epsilon;
    double minus;
    if (!loss_and_gradient(&network, &dataset, use_residual, nullptr, &minus)) {
        fputs("输入负向扰动计算失败\n", stderr);
        return false;
    }
    double numeric_input = (plus - minus) * (double)dataset.count / (2.0 * epsilon);
    dataset.samples[0].pixels[17] = original_pixel;
    double input_error = fabs(numeric_input - input_gradient[17]) /
                         fmax(1.0, fmax(fabs(numeric_input), fabs(input_gradient[17])));
    if (input_error >= 2e-5) {
        fprintf(stderr, "输入梯度不匹配：解析 %.8f，数值 %.8f\n", input_gradient[17],
                numeric_input);
    }
    return input_error < 2e-5 && isfinite(analytic_loss);
}

/**
 * @brief 手算验证残差相加两条路径的梯度，并检查 ReLU 关闭情形。
 * @return 恒等路径与主路径梯度符合逐元素公式时为 true。
 */
static bool test_identity_gradient(void)
{
    double preactivation[TENSOR_SIZE] = {0};
    double output_gradient[TENSOR_SIZE] = {0};
    double branch_gradient[TENSOR_SIZE] = {0};
    double skip_gradient[TENSOR_SIZE] = {0};
    preactivation[0] = 3.0;
    output_gradient[0] = 2.5;
    preactivation[1] = -1.0;
    output_gradient[1] = 7.0;
    residual_merge_backward(preactivation, output_gradient, branch_gradient, skip_gradient);
    return branch_gradient[0] == 2.5 && skip_gradient[0] == 2.5 &&
           branch_gradient[1] == 0.0 && skip_gradient[1] == 0.0;
}

/**
 * @brief 执行无文件副作用的前向边界、捷径公式与两种网络梯度测试。
 * @return 全部检查通过时为 EXIT_SUCCESS。
 */
static int run_tests(void)
{
    if (!test_identity_gradient()) {
        fputs("恒等路径手算检查失败\n", stderr);
        return EXIT_FAILURE;
    }
    if (!check_gradients(false)) {
        fputs("普通 CNN 梯度检查失败\n", stderr);
        return EXIT_FAILURE;
    }
    if (!check_gradients(true)) {
        fputs("ResNet 梯度检查失败\n", stderr);
        return EXIT_FAILURE;
    }
    Network network;
    network_initialize(&network, UINT32_C(91));
    Dataset invalid;
    if (dataset_generate(&invalid, 2, UINT32_C(1))) {
        fputs("非法数据尺寸未被拒绝\n", stderr);
        return EXIT_FAILURE;
    }
    Dataset data;
    if (!dataset_generate(&data, 3, UINT32_C(7))) {
        return EXIT_FAILURE;
    }
    data.samples[0].pixels[0] = NAN;
    Metrics untouched = {.loss = -1.0, .accuracy = -1.0};
    if (evaluate(&network, &data, true, &untouched) || untouched.loss != -1.0 ||
        untouched.accuracy != -1.0) {
        fputs("非有限输入未被拒绝或失败时修改了输出\n", stderr);
        return EXIT_FAILURE;
    }
    puts("ResNet 测试通过：恒等分支手算、两种网络梯度和失败边界。");
    return EXIT_SUCCESS;
}

/**
 * @brief 用三个固定种子比较相同参数量和训练预算下的普通 CNN 与 ResNet。
 * @param argc 命令行参数数量，只允许无参数或 --test。
 * @param argv 参数数组，借用且不修改。
 * @return 成功为 EXIT_SUCCESS；参数或训练评估失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        return run_tests();
    }
    if (argc != 1) {
        fputs("用法：22_resnet [--test]\n", stderr);
        return EXIT_FAILURE;
    }
    printf("阶段 11.1：8×8 三类线条；CNN 与 ResNet 各 %d 个参数，训练 %d 轮，全批量 SGD。\n",
           PARAMETER_COUNT, EPOCH_COUNT);
    puts("ResNet 块：stem 恒等捷径 + 两层同尺寸 3×3 卷积，合并后 ReLU。");
    puts("两模型共用每个种子的所有数据和初始化；不以单次胜负判断结构优劣。");
    double cnn_accuracy_sum = 0.0;
    double resnet_accuracy_sum = 0.0;
    for (size_t trial = 0; trial < TRIAL_COUNT; ++trial) {
        uint32_t seed = UINT32_C(42) + (uint32_t)(trial * 101);
        Dataset training;
        Dataset validation;
        Dataset test;
        if (!dataset_generate(&training, TRAIN_COUNT, seed) ||
            !dataset_generate(&validation, VALIDATION_COUNT, seed + 1000) ||
            !dataset_generate(&test, TEST_COUNT, seed + 2000)) {
            fputs("无法生成固定实验数据\n", stderr);
            return EXIT_FAILURE;
        }
        Network cnn;
        Network resnet;
        network_initialize(&cnn, seed + 5000);
        resnet = cnn;
        if (!train(&cnn, &training, false, EPOCH_COUNT, 0.08) ||
            !train(&resnet, &training, true, EPOCH_COUNT, 0.08)) {
            fputs("网络训练产生无效梯度或参数\n", stderr);
            return EXIT_FAILURE;
        }
        Metrics cnn_train;
        Metrics cnn_validation;
        Metrics cnn_test;
        Metrics resnet_train;
        Metrics resnet_validation;
        Metrics resnet_test;
        if (!evaluate(&cnn, &training, false, &cnn_train) ||
            !evaluate(&cnn, &validation, false, &cnn_validation) ||
            !evaluate(&cnn, &test, false, &cnn_test) ||
            !evaluate(&resnet, &training, true, &resnet_train) ||
            !evaluate(&resnet, &validation, true, &resnet_validation) ||
            !evaluate(&resnet, &test, true, &resnet_test)) {
            fputs("模型评估失败\n", stderr);
            return EXIT_FAILURE;
        }
        printf("\n种子 %u（实验 %zu）\n", seed, trial + 1);
        printf("CNN    train_loss=%.4f val_loss=%.4f val_acc=%.1f%% "
               "test_loss=%.4f test_acc=%.1f%%\n",
               cnn_train.loss, cnn_validation.loss, 100.0 * cnn_validation.accuracy,
               cnn_test.loss, 100.0 * cnn_test.accuracy);
        printf("ResNet train_loss=%.4f val_loss=%.4f val_acc=%.1f%% "
               "test_loss=%.4f test_acc=%.1f%%\n",
               resnet_train.loss, resnet_validation.loss,
               100.0 * resnet_validation.accuracy, resnet_test.loss,
               100.0 * resnet_test.accuracy);
        cnn_accuracy_sum += cnn_test.accuracy;
        resnet_accuracy_sum += resnet_test.accuracy;
    }
    printf("\n三个种子测试准确率均值：CNN=%.1f%%，ResNet=%.1f%%。"
           "该合成任务不证明任一结构普遍更优。\n",
           100.0 * cnn_accuracy_sum / TRIAL_COUNT,
           100.0 * resnet_accuracy_sum / TRIAL_COUNT);
    return !ferror(stdout) ? EXIT_SUCCESS : EXIT_FAILURE;
}
