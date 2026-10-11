/**
 * @file
 * @brief 阶段 4：纯 C 去噪自编码器，256→32→8→32→256。
 * 带噪图像为输入、干净图像为目标；手写 MSE、反向传播与 Adam。
 */
#include "../graphics/bmp_capture.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIDE 16
#define PIXELS (SIDE * SIDE)
#define WIDTH 32
#define LATENT 8
#define LAYERS 4
#define CAPACITY 96
#define BATCH_SIZE 8
#define EPOCHS 400

enum {
    W1 = 0,
    B1 = W1 + WIDTH * PIXELS,
    W2 = B1 + WIDTH,
    B2 = W2 + LATENT * WIDTH,
    W3 = B2 + LATENT,
    B3 = W3 + WIDTH * LATENT,
    W4 = B3 + WIDTH,
    B4 = W4 + PIXELS * WIDTH,
    PARAMETERS = B4 + PIXELS
};

typedef struct {
    size_t inputs;
    size_t outputs;
    size_t weights;
    size_t biases;
} Layer;

/* 仅描述这个固定网络；行优先权重 W[输出神经元][输入特征]。 */
static const Layer layers[LAYERS] = {{PIXELS, WIDTH, W1, B1},
                                     {WIDTH, LATENT, W2, B2},
                                     {LATENT, WIDTH, W3, B3},
                                     {WIDTH, PIXELS, W4, B4}};

typedef struct {
    double clean[PIXELS];
    double noisy[PIXELS];
} Sample;

typedef struct {
    size_t count;
    Sample samples[CAPACITY];
} Dataset;

typedef struct {
    double parameters[PARAMETERS];
} Network;

typedef struct {
    /* 各行只使用层宽对应的元素；第 0 行为输入，第 2 行为瓶颈表示。 */
    double activations[LAYERS + 1][PIXELS];
} Forward;

typedef struct {
    double first[PARAMETERS];
    double second[PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

/**
 * @brief 产生可重复的 [0,1) 伪随机数。
 * @param state 有效可写状态，不转移所有权；数据、噪声、初始化、打乱分别使用。
 * @return 状态高 24 位映射的实数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 对干净图加入翻转噪声及小幅灰度扰动，目标保持不变。
 * @param sample 有效可写样本，只覆盖 noisy，不改 clean。
 * @param state 独立可写噪声状态，持续推进，不每张图片重新设种子。
 */
static void corrupt_sample(Sample *sample, uint32_t *state)
{
    for (size_t i = 0; i < PIXELS; ++i) {
        double value = sample->clean[i];
        if (random_unit(state) < 0.2) {
            value = 1 - value;
        }
        value += 0.16 * (2 * random_unit(state) - 1);
        sample->noisy[i] = fmin(1, fmax(0, value));
    }
}

/**
 * @brief 生成平衡的圆形、方形、三角形干净图像及初始噪声版本。
 * @param data 有效可写数据集，旧内容覆盖。
 * @param count 3..CAPACITY 内的样本数，须为 3 的倍数。
 * @param shape_seed 图形生成种子，位置与尺寸变化对各类使用相同范围。
 * @param noise_seed 初始噪声种子，独立于图形状态。
 */
static void dataset_generate(Dataset *data, size_t count, uint32_t shape_seed, uint32_t noise_seed)
{
    assert(count >= 3 && count <= CAPACITY && count % 3 == 0);
    data->count = count;
    for (size_t i = 0; i < count; ++i) {
        Sample *sample = &data->samples[i];
        double center_x = 6.75 + 1.5 * random_unit(&shape_seed);
        double center_y = 6.75 + 1.5 * random_unit(&shape_seed);
        double radius = 3 + random_unit(&shape_seed);
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                double dx = (double)column - center_x;
                double dy = (double)row - center_y;
                bool inside = dx * dx + dy * dy <= radius * radius;
                if (i % 3 == 1) {
                    inside = fabs(dx) <= radius && fabs(dy) <= radius;
                } else if (i % 3 == 2) {
                    inside = dy >= -radius && dy <= radius && fabs(dx) <= (dy + radius) * 0.5;
                }
                sample->clean[row * SIDE + column] = (double)inside;
            }
        }
        corrupt_sample(sample, &noise_seed);
    }
}

/**
 * @brief 稳定计算输出层 sigmoid。
 * @param z 有限实数分数。
 * @return [0,1] 内的重建像素。
 */
static double sigmoid(double z)
{
    if (z >= 0) {
        return 1 / (1 + exp(-z));
    }
    double exponential = exp(z);
    return exponential / (1 + exponential);
}

/**
 * @brief Xavier 均匀初始化各层权重，偏置置零。
 * @param network 有效可写网络，旧参数全部覆盖。
 * @param seed 独立初始化种子，不转移所有权。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    for (size_t layer = 0; layer < LAYERS; ++layer) {
        const Layer *spec = &layers[layer];
        double scale = sqrt(6.0 / (double)(spec->inputs + spec->outputs));
        for (size_t i = 0; i < spec->inputs * spec->outputs; ++i) {
            network->parameters[spec->weights + i] = (2 * random_unit(&seed) - 1) * scale;
        }
    }
}

/**
 * @brief 编码为 8 维瓶颈后解码；前三层 tanh，最后一层 sigmoid。
 * @param network 有效只读网络，参数及计算须有限且不溢出。
 * @param input PIXELS 个有限只读灰度元素，只传 noisy，不传干净目标。
 * @param f 有效可写缓存，与输入不重叠，只写各层实际使用的元素。
 */
static void network_forward(const Network *network, const double *input, Forward *f)
{
    memcpy(f->activations[0], input, PIXELS * sizeof(double));
    for (size_t layer = 0; layer < LAYERS; ++layer) {
        const Layer *spec = &layers[layer];
        for (size_t out = 0; out < spec->outputs; ++out) {
            double z = network->parameters[spec->biases + out];
            for (size_t in = 0; in < spec->inputs; ++in) {
                z += network->parameters[spec->weights + out * spec->inputs + in] *
                     f->activations[layer][in];
            }
            if (layer == LAYERS - 1) {
                f->activations[layer + 1][out] = sigmoid(z);
            } else {
                f->activations[layer + 1][out] = tanh(z);
            }
        }
    }
}

/**
 * @brief 计算逐像素平均平方误差，不乘 1/2。
 * @param a PIXELS 个有限只读像素。
 * @param b PIXELS 个有限只读目标像素。
 * @return 均方误差，中间运算须不溢出。
 */
static double image_mse(const double *a, const double *b)
{
    double total = 0;
    for (size_t i = 0; i < PIXELS; ++i) {
        double error = a[i] - b[i];
        total += error * error;
    }
    return total / PIXELS;
}

/**
 * @brief 手写各层反向传播，覆盖参数梯度与带噪输入梯度。
 * @param network 与前向缓存对应的有效只读旧参数。
 * @param target PIXELS 个 [0,1] 范围的只读干净目标。
 * @param f 同一次前向的只读缓存。
 * @param gradients 至少 PARAMETERS 个可写元素，不与其他数组重叠。
 * @param input_gradient 至少 PIXELS 个可写元素，不与其他数组重叠。
 */
static void network_backward(const Network *network, const double *target, const Forward *f,
                             double gradients[PARAMETERS], double input_gradient[PIXELS])
{
    double delta[PIXELS] = {0};
    for (size_t i = 0; i < PIXELS; ++i) {
        double output = f->activations[LAYERS][i];
        /* MSE 的 1/PIXELS 与 sigmoid 导数都不可漏掉。 */
        delta[i] = 2 * (output - target[i]) / PIXELS * output * (1 - output);
    }
    for (size_t remaining = LAYERS; remaining > 0; --remaining) {
        size_t layer = remaining - 1;
        const Layer *spec = &layers[layer];
        double upstream[PIXELS] = {0};
        for (size_t out = 0; out < spec->outputs; ++out) {
            gradients[spec->biases + out] = delta[out];
            for (size_t in = 0; in < spec->inputs; ++in) {
                size_t weight = spec->weights + out * spec->inputs + in;
                gradients[weight] = delta[out] * f->activations[layer][in];
                upstream[in] += delta[out] * network->parameters[weight];
            }
        }
        for (size_t in = 0; in < spec->inputs; ++in) {
            if (layer == 0) {
                input_gradient[in] = upstream[in];
            } else {
                double activation = f->activations[layer][in];
                delta[in] = upstream[in] * (1 - activation * activation);
            }
        }
    }
}

/**
 * @brief 用小批量平均梯度更新 Adam 和参数。
 * @param network 有效可写网络，不与其他数组重叠。
 * @param gradients PARAMETERS 个有限只读梯度。
 * @param adam 有效可写状态，初始矩为零、beta 的幂为 1。
 * @param rate 有限正学习率，计算须不溢出。
 */
static void network_update(Network *network, const double *gradients, Adam *adam, double rate)
{
    adam->beta1_power *= 0.9;
    adam->beta2_power *= 0.999;
    for (size_t p = 0; p < PARAMETERS; ++p) {
        adam->first[p] = 0.9 * adam->first[p] + 0.1 * gradients[p];
        adam->second[p] = 0.999 * adam->second[p] + 0.001 * gradients[p] * gradients[p];
        double first = adam->first[p] / (1 - adam->beta1_power);
        double second = adam->second[p] / (1 - adam->beta2_power);
        network->parameters[p] -= rate * first / (sqrt(second) + 1e-8);
        assert(isfinite(network->parameters[p]));
    }
}

/**
 * @brief 评估当前参数的平均重建 MSE，不更新网络或噪声。
 * @param network 有效只读网络。
 * @param data 非空有效只读集合，只用 noisy 前向、clean 算损失。
 * @return 样本平均 MSE，要求数值有限。
 */
static double evaluate(const Network *network, const Dataset *data)
{
    double total = 0;
    for (size_t i = 0; i < data->count; ++i) {
        Forward f;
        network_forward(network, data->samples[i].noisy, &f);
        total += image_mse(f.activations[LAYERS], data->samples[i].clean);
    }
    double loss = total / (double)data->count;
    assert(isfinite(loss));
    return loss;
}

/**
 * @brief 累加一批样本的平均梯度，所有样本共享同一份旧参数。
 * @param network 有效只读网络。
 * @param data 有效只读集合。
 * @param order 至少 data->count 个合法样本下标的只读数组。
 * @param start 批次起点。
 * @param count 实际批次大小，非零且 start+count 不超过 data->count。
 * @param average PARAMETERS 个可写元素，与输入不重叠，覆盖旧值。
 */
static void batch_gradients(const Network *network, const Dataset *data, const size_t *order,
                            size_t start, size_t count, double *average)
{
    assert(count > 0 && start <= data->count && count <= data->count - start);
    for (size_t p = 0; p < PARAMETERS; ++p) {
        average[p] = 0;
    }
    for (size_t i = start; i < start + count; ++i) {
        assert(order[i] < data->count);
        const Sample *sample = &data->samples[order[i]];
        Forward f;
        network_forward(network, sample->noisy, &f);
        double gradients[PARAMETERS];
        double input_gradient[PIXELS];
        network_backward(network, sample->clean, &f, gradients, input_gradient);
        for (size_t p = 0; p < PARAMETERS; ++p) {
            average[p] += gradients[p] / (double)count;
        }
    }
}

/**
 * @brief 检查全部参数和输入梯度，不使用解析公式重新验证自身。
 */
static void test_gradients(void)
{
    Network network;
    network_initialize(&network, 42);
    Sample sample;
    uint32_t seed = 17;
    for (size_t i = 0; i < PIXELS; ++i) {
        sample.noisy[i] = 0.2 + 0.6 * random_unit(&seed);
        sample.clean[i] = 0.3 + 0.4 * random_unit(&seed);
    }
    Forward f;
    network_forward(&network, sample.noisy, &f);
    double gradients[PARAMETERS];
    double input_gradient[PIXELS];
    network_backward(&network, sample.clean, &f, gradients, input_gradient);
    const double epsilon = 1e-5;
    for (size_t p = 0; p < PARAMETERS; ++p) {
        double original = network.parameters[p];
        Forward plus;
        Forward minus;
        network.parameters[p] = original + epsilon;
        network_forward(&network, sample.noisy, &plus);
        network.parameters[p] = original - epsilon;
        network_forward(&network, sample.noisy, &minus);
        network.parameters[p] = original;
        double numerical = (image_mse(plus.activations[LAYERS], sample.clean) -
                            image_mse(minus.activations[LAYERS], sample.clean)) /
                           (2 * epsilon);
        assert(fabs(numerical - gradients[p]) < 1e-9 + 1e-5 * fabs(numerical));
    }
    for (size_t i = 0; i < PIXELS; ++i) {
        double original = sample.noisy[i];
        Forward plus;
        Forward minus;
        sample.noisy[i] = original + epsilon;
        network_forward(&network, sample.noisy, &plus);
        sample.noisy[i] = original - epsilon;
        network_forward(&network, sample.noisy, &minus);
        sample.noisy[i] = original;
        double numerical = (image_mse(plus.activations[LAYERS], sample.clean) -
                            image_mse(minus.activations[LAYERS], sample.clean)) /
                           (2 * epsilon);
        assert(fabs(numerical - input_gradient[i]) < 1e-9 + 1e-5 * fabs(numerical));
    }
    assert(sigmoid(-1000) == 0 && sigmoid(1000) == 1);
    double zero[PIXELS] = {0};
    double one[PIXELS];
    for (size_t i = 0; i < PIXELS; ++i) {
        one[i] = 1;
    }
    assert(image_mse(zero, zero) == 0 && image_mse(zero, one) == 1);
}

/**
 * @brief 先在三张固定图上拟合，确认网络能重建微型数据，非泛化评估。
 * @param initial 有效只读起始参数，不修改它。
 */
static void test_small_fit(const Network *initial)
{
    static Dataset data;
    dataset_generate(&data, 3, 51, 52);
    Network network = *initial;
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    size_t order[] = {0, 1, 2};
    double before = evaluate(&network, &data);
    for (size_t iteration = 0; iteration < 200; ++iteration) {
        double average[PARAMETERS];
        batch_gradients(&network, &data, order, 0, 3, average);
        network_update(&network, average, &adam, 0.01);
    }
    double after = evaluate(&network, &data);
    assert(after < before * 0.2 && after < 0.03);
}

/**
 * @brief 验证噪声可重复、目标不变和尾批平均。
 * @param initial 有效只读网络，测试不修改它。
 */
static void test_data_and_batch(const Network *initial)
{
    static Dataset first;
    static Dataset repeated;
    dataset_generate(&first, 3, 61, 62);
    dataset_generate(&repeated, 3, 61, 62);
    assert(memcmp(first.samples, repeated.samples, 3 * sizeof(Sample)) == 0);
    uint32_t seed = 63;
    corrupt_sample(&first.samples[0], &seed);
    assert(memcmp(first.samples[0].clean, repeated.samples[0].clean,
                  sizeof(first.samples[0].clean)) == 0);
    assert(memcmp(first.samples[0].noisy, repeated.samples[0].noisy,
                  sizeof(first.samples[0].noisy)) != 0);
    size_t order[] = {0, 1, 2};
    double average[PARAMETERS];
    batch_gradients(initial, &first, order, 2, 1, average);
    Forward f;
    network_forward(initial, first.samples[2].noisy, &f);
    double single[PARAMETERS];
    double input_gradient[PIXELS];
    network_backward(initial, first.samples[2].clean, &f, single, input_gradient);
    for (size_t p = 0; p < PARAMETERS; ++p) {
        assert(average[p] == single[p]);
    }
}

/**
 * @brief 每轮生成新训练噪声、打乱并更新，以固定验证噪声选择最佳参数。
 * @param initial 有效只读起始参数。
 * @param training 有效可写训练集，仅修改 noisy，clean 不变。
 * @param validation 非空只读验证集，噪声始终不变。
 * @param quiet 是否省略每 25 轮的输出。
 * @param best 有效可写最佳参数副本，供推理使用，不保存优化器状态。
 */
static void train(const Network *initial, Dataset *training, const Dataset *validation, bool quiet,
                  Network *best)
{
    Network current = *initial;
    *best = current;
    double best_loss = evaluate(&current, validation);
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    uint32_t noise_seed = 4025;
    uint32_t shuffle_seed = 2028;
    for (size_t epoch = 1; epoch <= EPOCHS; ++epoch) {
        size_t order[CAPACITY];
        for (size_t i = 0; i < training->count; ++i) {
            corrupt_sample(&training->samples[i], &noise_seed);
            order[i] = i;
        }
        for (size_t remaining = training->count; remaining > 1; --remaining) {
            size_t selected = (size_t)(random_unit(&shuffle_seed) * (double)remaining);
            size_t temporary = order[remaining - 1];
            order[remaining - 1] = order[selected];
            order[selected] = temporary;
        }
        for (size_t start = 0; start < training->count; start += BATCH_SIZE) {
            size_t count = training->count - start;
            if (count > BATCH_SIZE) {
                count = BATCH_SIZE;
            }
            double average[PARAMETERS];
            batch_gradients(&current, training, order, start, count, average);
            network_update(&current, average, &adam, 0.003);
        }
        double valid = evaluate(&current, validation);
        if (valid < best_loss) {
            best_loss = valid;
            *best = current;
        }
        if (!quiet && epoch % 25 == 0) {
            printf("epoch=%3zu，当前训练噪声 MSE=%.5f，固定验证噪声 MSE=%.5f\n", epoch,
                   evaluate(&current, training), valid);
        }
    }
}

/**
 * @brief 绘制原始灰度图，不单独归一化，便于比较实际重建强度。
 * @param canvas 有效可写画布。
 * @param image PIXELS 个 [0,1] 内的有限只读像素。
 * @param left 画布内起始 x，须容纳 SIDE*4 像素。
 * @param top 画布内起始 y，须容纳 SIDE*4 像素。
 */
static void draw_tile(GfxCanvas *canvas, const double *image, int left, int top)
{
    for (size_t row = 0; row < SIDE; ++row) {
        for (size_t column = 0; column < SIDE; ++column) {
            uint8_t gray = (uint8_t)lround(255 * image[row * SIDE + column]);
            gfx_rect(canvas, left + (int)column * 4, top + (int)row * 4, 4, 4, true,
                     gfx_rgb(gray, gray, gray));
        }
    }
}

/**
 * @brief 导出三行样本，三列分别为带噪输入、模型重建、干净目标。
 * @param network 有效只读网络，推理输出在 [0,1] 内。
 * @param data 至少三个样本的只读数据。
 * @param name 合法文件名前缀，使用 exports 内独占创建的 BMP。
 * @return 成功为 true；文件失败时明确报错，已有图片保持不变。
 */
static bool export_reconstruction(const Network *network, const Dataset *data, const char *name)
{
    static GfxCanvas canvas;
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    for (size_t i = 0; i < 3; ++i) {
        const Sample *sample = &data->samples[i];
        Forward f;
        network_forward(network, sample->noisy, &f);
        int top = 5 + (int)i * 80;
        draw_tile(&canvas, sample->noisy, 20, top);
        draw_tile(&canvas, f.activations[LAYERS], 120, top);
        draw_tile(&canvas, sample->clean, 220, top);
    }
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(&canvas, name, path)) {
        fprintf(stderr, "自编码器图片导出失败：%s\n", strerror(errno));
        return false;
    }
    printf("已导出：%s\n", path);
    return true;
}

/**
 * @brief 运行去噪案例与数值断言，默认不创建文件。
 * @param argc 命令行参数数目。
 * @param argv 可选 --export 导出 BMP，--test 安静验证；其他参数报错。
 * @return 成功为 EXIT_SUCCESS；参数或导出失败为 EXIT_FAILURE，断言失败终止。
 */
int main(int argc, char **argv)
{
    bool images = false;
    bool quiet = false;
    if (argc == 2 && strcmp(argv[1], "--export") == 0) {
        images = true;
    } else if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        quiet = true;
    } else if (argc != 1) {
        fprintf(stderr, "用法：%s [--export | --test]\n", argv[0]);
        return EXIT_FAILURE;
    }
    Network initial;
    network_initialize(&initial, 42);
    test_gradients();
    test_data_and_batch(&initial);
    test_small_fit(&initial);
    static Dataset training;
    static Dataset validation;
    static Dataset testing;
    dataset_generate(&training, 96, 2025, 3025);
    dataset_generate(&validation, 48, 2026, 3026);
    dataset_generate(&testing, 48, 2027, 3027);
    if (!quiet) {
        puts("=== 去噪自编码器：256→32(tanh)→8(tanh)→32(tanh)→256(sigmoid) ===\n"
             "17224 个参数；训练/验证/测试=96/48/48；Adam 0.003，批量8，400轮。\n"
             "输入为带噪图，目标为干净图；训练噪声每轮更新，评估噪声固定。\n"
             "初始化42，图形2025/2026/2027，初始噪声3025/3026/3027，训练噪声4025，打乱2028。");
        printf("初始训练 MSE=%.5f，三张微型数据拟合检查通过。\n", evaluate(&initial, &training));
    }
    Network best;
    train(&initial, &training, &validation, quiet, &best);
    double test_loss = evaluate(&best, &testing);
    double noisy_loss = 0;
    double mean_image[PIXELS] = {0};
    for (size_t i = 0; i < training.count; ++i) {
        for (size_t pixel = 0; pixel < PIXELS; ++pixel) {
            mean_image[pixel] += training.samples[i].clean[pixel] / (double)training.count;
        }
    }
    double mean_loss = 0;
    for (size_t i = 0; i < testing.count; ++i) {
        noisy_loss += image_mse(testing.samples[i].noisy, testing.samples[i].clean);
        mean_loss += image_mse(mean_image, testing.samples[i].clean);
    }
    noisy_loss /= (double)testing.count;
    mean_loss /= (double)testing.count;
    assert(test_loss < noisy_loss * 0.4 && test_loss < mean_loss && test_loss < 0.08);
    printf("独立测试 MSE：带噪输入 %.5f，训练集平均图 %.5f，模型重建 %.5f\n", noisy_loss, mean_loss,
           test_loss);
    if (!quiet) {
        Forward f;
        network_forward(&best, testing.samples[0].noisy, &f);
        printf("首张测试图的 8 维瓶颈：");
        for (size_t i = 0; i < LATENT; ++i) {
            printf(" %+.3f", f.activations[2][i]);
        }
        putchar('\n');
    }
    if (images && (!export_reconstruction(&initial, &validation, "autoencoder-initial") ||
                   !export_reconstruction(&best, &validation, "autoencoder-best") ||
                   !export_reconstruction(&best, &testing, "autoencoder-test"))) {
        return EXIT_FAILURE;
    }
    puts("阶段 4 验证通过：17224个参数和256个输入梯度、噪声/尾批、微型拟合与独立去噪评估。");
    return EXIT_SUCCESS;
}
