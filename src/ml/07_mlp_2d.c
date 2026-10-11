/**
 * @file
 * @brief 阶段 2：二维圆环分类、ReLU、小批量 SGD/Adam 与决策边界。
 * 计算为纯 C；BMP 复用软件绘图核心，不依赖 SDL 或机器学习框架。
 */
#include "../graphics/bmp_capture.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUTS 2
#define WIDTH 16
#define CAPACITY 256
#define BATCH_SIZE 16
#define EPOCHS 200

enum {
    W1 = 0,
    B1 = W1 + WIDTH * INPUTS,
    W2 = B1 + WIDTH,
    B2 = W2 + WIDTH * WIDTH,
    W3 = B2 + WIDTH,
    B3 = W3 + WIDTH,
    PARAMETERS = B3 + 1
};

typedef struct {
    double x[INPUTS];
    double target;
} Sample;

typedef struct {
    size_t count;
    Sample samples[CAPACITY];
} Dataset;

typedef struct {
    double parameters[PARAMETERS];
} Network;

typedef struct {
    double z1[WIDTH];
    double h1[WIDTH];
    double z2[WIDTH];
    double h2[WIDTH];
    double logit;
    double probability;
} Forward;

typedef struct {
    double first[PARAMETERS];
    double second[PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

typedef struct {
    double loss;
    double accuracy;
} Metrics;

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM
} Optimizer;

/**
 * @brief 产生可重复的 [0,1) 伪随机数，数据、初始化、打乱各自使用独立状态。
 * @param state 有效可写 uint32_t 状态，不转移所有权。
 * @return 取状态高 24 位映射的实数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 生成平衡的内圆盘/外圆环样本，半径平方均匀以获得均匀面积采样。
 * @param data 可写数据集，原内容被覆盖。
 * @param count 偶数样本数，2..CAPACITY。
 * @param seed 本数据集的种子；训练、验证、测试分别生成。
 */
static void dataset_generate(Dataset *data, size_t count, uint32_t seed)
{
    assert(count >= 2 && count <= CAPACITY && count % 2 == 0);
    data->count = count;
    for (size_t i = 0; i < count; ++i) {
        double target = (double)(i % 2);
        double minimum = 0;
        double maximum = 0.55;
        if (target == 0) {
            minimum = 0.75;
            maximum = 1;
        }
        double radius =
            sqrt(minimum * minimum + random_unit(&seed) * (maximum * maximum - minimum * minimum));
        double angle = 6.283185307179586 * random_unit(&seed);
        data->samples[i] = (Sample){{radius * cos(angle), radius * sin(angle)}, target};
    }
}

/**
 * @brief 初始化权重；ReLU 层使用 He 均匀尺度，输出层使用较小尺度。
 * @param network 有效可写网络，旧内容全部覆盖；偏置置零。
 * @param seed 初始化种子，与数据和打乱状态分开。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    for (size_t h = 0; h < WIDTH; ++h) {
        for (size_t i = 0; i < INPUTS; ++i) {
            network->parameters[W1 + h * INPUTS + i] =
                (2 * random_unit(&seed) - 1) * sqrt(6.0 / INPUTS);
        }
        for (size_t i = 0; i < WIDTH; ++i) {
            network->parameters[W2 + h * WIDTH + i] =
                (2 * random_unit(&seed) - 1) * sqrt(6.0 / WIDTH);
        }
        network->parameters[W3 + h] = (2 * random_unit(&seed) - 1) * sqrt(3.0 / WIDTH);
    }
}

/**
 * @brief 稳定 sigmoid。
 * @param z 有限 logit。
 * @return [0,1] 内的预测概率。
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
 * @brief 从 logit 计算交叉熵，避免对概率端点取对数。
 * @param z 有限 logit。
 * @param target 0/1 标签。
 * @return 有限非负损失；要求中间运算不溢出。
 */
static double cross_entropy(double z, double target)
{
    return fmax(z, 0) - target * z + log1p(exp(-fabs(z)));
}

/**
 * @brief 两个 ReLU 隐藏层的前向传播，缓存激活前后数值。
 * @param network 有效只读网络，所有参数及计算须有限且不溢出。
 * @param x 长度为 INPUTS 的只读特征，不转移所有权。
 * @return 前向缓存，用于相同参数版本的反向传播。
 */
static Forward network_forward(const Network *network, const double x[INPUTS])
{
    Forward f = {0};
    for (size_t h = 0; h < WIDTH; ++h) {
        f.z1[h] = network->parameters[B1 + h];
        for (size_t i = 0; i < INPUTS; ++i) {
            f.z1[h] += network->parameters[W1 + h * INPUTS + i] * x[i];
        }
        f.h1[h] = fmax(0, f.z1[h]);
    }
    f.logit = network->parameters[B3];
    for (size_t h = 0; h < WIDTH; ++h) {
        f.z2[h] = network->parameters[B2 + h];
        for (size_t i = 0; i < WIDTH; ++i) {
            f.z2[h] += network->parameters[W2 + h * WIDTH + i] * f.h1[i];
        }
        f.h2[h] = fmax(0, f.z2[h]);
        f.logit += network->parameters[W3 + h] * f.h2[h];
    }
    f.probability = sigmoid(f.logit);
    return f;
}

/**
 * @brief 覆盖单样本的全部梯度，ReLU 在零点采用导数 0。
 * @param network 对应前向缓存的只读旧参数。
 * @param sample 有效只读样本，标签为 0/1。
 * @param f 相同参数版本及样本的只读缓存。
 * @param gradients 容量至少 PARAMETERS 的可写数组，不能与参数重叠。
 */
static void network_backward(const Network *network, const Sample *sample, const Forward *f,
                             double gradients[PARAMETERS])
{
    double delta3 = f->probability - sample->target;
    double delta2[WIDTH] = {0};
    gradients[B3] = delta3;
    for (size_t h = 0; h < WIDTH; ++h) {
        gradients[W3 + h] = delta3 * f->h2[h];
        if (f->z2[h] > 0) {
            delta2[h] = delta3 * network->parameters[W3 + h];
        }
        gradients[B2 + h] = delta2[h];
        for (size_t i = 0; i < WIDTH; ++i) {
            gradients[W2 + h * WIDTH + i] = delta2[h] * f->h1[i];
        }
    }
    for (size_t h = 0; h < WIDTH; ++h) {
        double delta1 = 0;
        if (f->z1[h] > 0) {
            /* 第一层收到来自所有第二层神经元的梯度，不能只取其中一条路径。 */
            for (size_t next = 0; next < WIDTH; ++next) {
                delta1 += delta2[next] * network->parameters[W2 + next * WIDTH + h];
            }
        }
        gradients[B1 + h] = delta1;
        for (size_t i = 0; i < INPUTS; ++i) {
            gradients[W1 + h * INPUTS + i] = delta1 * sample->x[i];
        }
    }
}

/**
 * @brief 计算一个小批量的平均梯度，所有样本使用同一版本参数。
 * @param network 有效只读网络，不更新参数。
 * @param data 有效只读数据，count 在 1..CAPACITY 内。
 * @param order 至少 data->count 个有效样本下标的只读数组。
 * @param start 本批在 order 中的起点。
 * @param count 本批实际大小，必须非零且 start+count 不超过 data->count。
 * @param gradients 至少 PARAMETERS 个可写元素，不能与输入重叠。
 */
static void batch_gradients(const Network *network, const Dataset *data, const size_t *order,
                            size_t start, size_t count, double gradients[PARAMETERS])
{
    assert(count > 0 && start <= data->count && count <= data->count - start);
    for (size_t p = 0; p < PARAMETERS; ++p) {
        gradients[p] = 0;
    }
    for (size_t i = start; i < start + count; ++i) {
        assert(order[i] < data->count);
        const Sample *sample = &data->samples[order[i]];
        Forward f = network_forward(network, sample->x);
        double single[PARAMETERS] = {0};
        network_backward(network, sample, &f, single);
        for (size_t p = 0; p < PARAMETERS; ++p) {
            gradients[p] += single[p] / (double)count;
        }
    }
}

/**
 * @brief Fisher–Yates 打乱下标，不改变原数据集，状态每轮继续推进。
 * @param order 至少 count 个可写下标，本函数先重建 0..count-1。
 * @param count 样本数，1..CAPACITY。
 * @param state 有效可写打乱随机状态，独立于数据生成状态。
 */
static void shuffle_indices(size_t *order, size_t count, uint32_t *state)
{
    assert(count > 0 && count <= CAPACITY);
    for (size_t i = 0; i < count; ++i) {
        order[i] = i;
    }
    for (size_t remaining = count; remaining > 1; --remaining) {
        size_t selected = (size_t)(random_unit(state) * (double)remaining);
        size_t temporary = order[remaining - 1];
        order[remaining - 1] = order[selected];
        order[selected] = temporary;
    }
}

/**
 * @brief 用平均梯度更新参数；Adam 按每个批次更新矩估计及偏差修正。
 * @param network 可写网络，不与 gradients 重叠。
 * @param gradients 至少 PARAMETERS 个有限只读梯度。
 * @param optimizer SGD 或 Adam。
 * @param adam 可写 Adam 状态；初始矩为零、两个 beta_power 为 1，SGD 时不使用。
 * @param rate 有限正学习率；计算须不溢出。
 */
static void network_update(Network *network, const double gradients[PARAMETERS],
                           Optimizer optimizer, Adam *adam, double rate)
{
    if (optimizer == OPTIMIZER_ADAM) {
        adam->beta1_power *= 0.9;
        adam->beta2_power *= 0.999;
    }
    for (size_t p = 0; p < PARAMETERS; ++p) {
        double direction = gradients[p];
        if (optimizer == OPTIMIZER_ADAM) {
            adam->first[p] = 0.9 * adam->first[p] + 0.1 * gradients[p];
            adam->second[p] = 0.999 * adam->second[p] + 0.001 * gradients[p] * gradients[p];
            double first_corrected = adam->first[p] / (1 - adam->beta1_power);
            double second_corrected = adam->second[p] / (1 - adam->beta2_power);
            direction = first_corrected / (sqrt(second_corrected) + 1e-8);
        }
        network->parameters[p] -= rate * direction;
        assert(isfinite(network->parameters[p]));
    }
}

/**
 * @brief 独立评估固定参数的平均损失与准确率，不执行训练或修改状态。
 * @param network 有效只读网络。
 * @param data 非空有效只读数据，count 不超过 CAPACITY。
 * @return 交叉熵与 [0,1] 准确率，概率 >=0.5 判为正类。
 */
static Metrics evaluate(const Network *network, const Dataset *data)
{
    Metrics metrics = {0};
    for (size_t i = 0; i < data->count; ++i) {
        Forward f = network_forward(network, data->samples[i].x);
        metrics.loss += cross_entropy(f.logit, data->samples[i].target);
        metrics.accuracy += (f.probability >= 0.5) == (data->samples[i].target == 1);
    }
    metrics.loss /= (double)data->count;
    metrics.accuracy /= (double)data->count;
    assert(isfinite(metrics.loss));
    return metrics;
}

/**
 * @brief 绘制概率背景与训练点，左上角映射 (-1,+1)，右下角映射 (+1,-1)。
 * @param network 有效只读网络，所有像素预测须有限。
 * @param data 用于叠加的只读数据，不用于更新模型。
 * @param canvas 可写完整画布；蓝色代表外环类 0，橙色代表内盘类 1。
 */
static void draw_boundary(const Network *network, const Dataset *data, GfxCanvas *canvas)
{
    for (int py = 0; py < GFX_HEIGHT; ++py) {
        for (int px = 0; px < GFX_WIDTH; ++px) {
            double x[] = {2.0 * px / (GFX_WIDTH - 1) - 1, 1 - 2.0 * py / (GFX_HEIGHT - 1)};
            Forward f = network_forward(network, x);
            uint8_t red = (uint8_t)(40 + 180 * f.probability);
            uint8_t green = (uint8_t)(85 + 35 * f.probability);
            uint8_t blue = (uint8_t)(210 - 165 * f.probability);
            gfx_pixel(canvas, px, py, gfx_rgb(red, green, blue));
        }
    }
    for (size_t i = 0; i < data->count; ++i) {
        const Sample *sample = &data->samples[i];
        int px = (int)lround((sample->x[0] + 1) * 0.5 * (GFX_WIDTH - 1));
        int py = (int)lround((1 - sample->x[1]) * 0.5 * (GFX_HEIGHT - 1));
        uint32_t color = gfx_rgb(60, 150, 255);
        if (sample->target == 1) {
            color = gfx_rgb(255, 180, 45);
        }
        gfx_circle(canvas, px, py, 2, gfx_rgb(255, 255, 255));
        gfx_pixel(canvas, px, py, color);
    }
}

/**
 * @brief 导出当前决策边界，独占创建 exports 下的 BMP，不覆盖历史图片。
 * @param network 有效只读网络。
 * @param data 要叠加的只读训练数据。
 * @param name 合法 BMP 文件名前缀。
 * @return 成功为 true；写入失败时显示 errno 原因，可能已有其他快照成功保存。
 */
static bool export_boundary(const Network *network, const Dataset *data, const char *name)
{
    static GfxCanvas canvas;
    draw_boundary(network, data, &canvas);
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(&canvas, name, path)) {
        fprintf(stderr, "决策边界导出失败：%s\n", strerror(errno));
        return false;
    }
    printf("已导出：%s\n", path);
    return true;
}

/**
 * @brief 检查 ReLU 激活分支一致，中心差分不应跨过不可微零点。
 * @param a 一个参数扰动方向的只读缓存。
 * @param b 另一个参数扰动方向的只读缓存。
 * @return 两层全部神经元的分支一致且远离零点时为 true。
 */
static bool same_relu_region(const Forward *a, const Forward *b)
{
    for (size_t h = 0; h < WIDTH; ++h) {
        if ((a->z1[h] > 0) != (b->z1[h] > 0) || (a->z2[h] > 0) != (b->z2[h] > 0) ||
            fabs(a->z1[h]) < 1e-7 || fabs(b->z1[h]) < 1e-7 || fabs(a->z2[h]) < 1e-7 ||
            fabs(b->z2[h]) < 1e-7) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 检查全部参数的批量平均梯度、尾批、SGD/Adam 与数据打乱。
 * @param initialized 有效只读初始化网络，测试不修改它。
 */
static void test_numerics(const Network *initialized)
{
    Network zero = {0};
    Sample origin = {{0, 0}, 1};
    Forward inactive = network_forward(&zero, origin.x);
    double inactive_gradients[PARAMETERS] = {0};
    network_backward(&zero, &origin, &inactive, inactive_gradients);
    for (size_t p = 0; p < B3; ++p) {
        assert(inactive_gradients[p] == 0);
    }
    assert(inactive_gradients[B3] == -0.5);

    Dataset small = {.count = 3,
                     .samples = {{{0.23, -0.41}, 1}, {{-0.37, 0.67}, 0}, {{0.81, 0.11}, 0}}};
    size_t order[] = {0, 1, 2};
    double gradients[PARAMETERS] = {0};
    batch_gradients(initialized, &small, order, 0, 2, gradients);
    size_t checked = 0;
    for (size_t p = 0; p < PARAMETERS; ++p) {
        Network plus = *initialized;
        Network minus = *initialized;
        plus.parameters[p] += 1e-5;
        minus.parameters[p] -= 1e-5;
        double numerical = 0;
        bool smooth = true;
        for (size_t i = 0; i < 2; ++i) {
            Forward a = network_forward(&plus, small.samples[i].x);
            Forward b = network_forward(&minus, small.samples[i].x);
            smooth = smooth && same_relu_region(&a, &b);
            numerical += (cross_entropy(a.logit, small.samples[i].target) -
                          cross_entropy(b.logit, small.samples[i].target)) /
                         (2e-5 * 2);
        }
        if (smooth) {
            assert(fabs(numerical - gradients[p]) <
                   1e-8 + 1e-6 * fmax(fabs(numerical), fabs(gradients[p])));
            ++checked;
        }
    }
    assert(checked == PARAMETERS); /* 此固定夹具没有跨零点，确保没有漏检参数。 */
    batch_gradients(initialized, &small, order, 2, 1, gradients);
    Forward tail = network_forward(initialized, small.samples[2].x);
    double single[PARAMETERS] = {0};
    network_backward(initialized, &small.samples[2], &tail, single);
    for (size_t p = 0; p < PARAMETERS; ++p) {
        assert(gradients[p] == single[p]);
    }
    Network sgd = *initialized;
    network_update(&sgd, gradients, OPTIMIZER_SGD, nullptr, 0.05);
    for (size_t p = 0; p < PARAMETERS; ++p) {
        assert(fabs(sgd.parameters[p] - (initialized->parameters[p] - 0.05 * gradients[p])) <
               1e-12);
        gradients[p] = 2;
    }
    Network adam_network = {0};
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    for (size_t step = 1; step <= 2; ++step) {
        network_update(&adam_network, gradients, OPTIMIZER_ADAM, &adam, 0.003);
        for (size_t p = 0; p < PARAMETERS; ++p) {
            assert(fabs(adam_network.parameters[p] + (double)step * 0.003 * 2 / (2 + 1e-8)) <
                   1e-12);
        }
    }
    uint32_t seed = 77;
    size_t shuffled[CAPACITY] = {0};
    bool seen[CAPACITY] = {false};
    shuffle_indices(shuffled, CAPACITY, &seed);
    for (size_t i = 0; i < CAPACITY; ++i) {
        assert(shuffled[i] < CAPACITY && !seen[shuffled[i]]);
        seen[shuffled[i]] = true;
    }
    assert(sigmoid(-1000) == 0 && sigmoid(1000) == 1);
    assert(cross_entropy(1000, 0) == 1000 && cross_entropy(-1000, 1) == 1000);
}

/**
 * @brief 验证数据生成可重复、半径范围及类别平衡，不用测试数据调参。
 */
static void test_data(void)
{
    Dataset first = {0};
    Dataset repeated = {0};
    Dataset independent = {0};
    dataset_generate(&first, CAPACITY, 2025);
    dataset_generate(&repeated, CAPACITY, 2025);
    dataset_generate(&independent, CAPACITY, 2026);
    size_t positives = 0;
    for (size_t i = 0; i < first.count; ++i) {
        const Sample *sample = &first.samples[i];
        assert(sample->x[0] == repeated.samples[i].x[0]);
        assert(sample->x[1] == repeated.samples[i].x[1]);
        assert(sample->target == repeated.samples[i].target);
        double radius_squared = sample->x[0] * sample->x[0] + sample->x[1] * sample->x[1];
        if (sample->target == 1) {
            ++positives;
            assert(radius_squared <= 0.55 * 0.55 + 1e-12);
        } else {
            assert(radius_squared >= 0.75 * 0.75 - 1e-12 && radius_squared <= 1 + 1e-12);
        }
    }
    assert(positives == first.count / 2);
    assert(first.samples[0].x[0] != independent.samples[0].x[0]);
}

/**
 * @brief 训练一个优化器，仅用训练数据计算梯度，以验证损失选择最佳参数副本。
 * @param initial 两个优化器共用的只读起始参数。
 * @param training 非空只读训练集，不修改样本。
 * @param validation 独立只读验证集，只参与参数版本选择。
 * @param optimizer SGD 或 Adam，使用各自明确固定的学习率。
 * @param images 是否在初始、第 100 轮及最佳参数处导出图片。
 * @param quiet 是否省略逐轮输出（测试模式）。
 * @param best 可写最佳网络输出，不转移所有权。
 * @return 成功为 true；图片导出失败返回 false，训练产物不覆盖已有图片。
 */
static bool train(const Network *initial, const Dataset *training, const Dataset *validation,
                  Optimizer optimizer, bool images, bool quiet, Network *best)
{
    Network current = *initial;
    *best = current;
    double best_loss = evaluate(&current, validation).loss;
    size_t best_epoch = 0;
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    uint32_t shuffle_seed = 2028;
    double rate = 0.05;
    const char *name = "SGD";
    if (optimizer == OPTIMIZER_ADAM) {
        rate = 0.003;
        name = "Adam";
    }
    if (!quiet) {
        printf("%s：学习率 %.3f，初始训练损失 %.4f，验证损失 %.4f\n", name, rate,
               evaluate(&current, training).loss, best_loss);
    }
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "mlp2d-%s-initial", name);
    if (images && !export_boundary(&current, training, prefix)) {
        return false;
    }
    for (size_t epoch = 1; epoch <= EPOCHS; ++epoch) {
        size_t order[CAPACITY] = {0};
        shuffle_indices(order, training->count, &shuffle_seed);
        for (size_t start = 0; start < training->count; start += BATCH_SIZE) {
            size_t count = training->count - start;
            if (count > BATCH_SIZE) {
                count = BATCH_SIZE;
            }
            double gradients[PARAMETERS] = {0};
            batch_gradients(&current, training, order, start, count, gradients);
            network_update(&current, gradients, optimizer, &adam, rate);
        }
        Metrics valid = evaluate(&current, validation);
        if (valid.loss < best_loss) {
            best_loss = valid.loss;
            best_epoch = epoch;
            *best = current;
        }
        if (!quiet && epoch % 40 == 0) {
            Metrics learned = evaluate(&current, training);
            printf("%s epoch=%3zu，训练损失=%.4f，验证损失=%.4f，验证准确率=%.1f%%\n", name, epoch,
                   learned.loss, valid.loss, 100 * valid.accuracy);
        }
        if (images && epoch == 100) {
            snprintf(prefix, sizeof(prefix), "mlp2d-%s-epoch100", name);
            if (!export_boundary(&current, training, prefix)) {
                return false;
            }
        }
    }
    if (!quiet) {
        printf("%s 最佳验证轮数：%zu，损失 %.4f\n", name, best_epoch, best_loss);
    }
    snprintf(prefix, sizeof(prefix), "mlp2d-%s-best", name);
    return !images || export_boundary(best, training, prefix);
}

/**
 * @brief 运行二维分类、数值测试和独立评估，默认不写文件。
 * @param argc 命令行参数数量。
 * @param argv 可选 --export 导出 BMP；--test 安静验证；其他参数报错。
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
    Dataset training = {0};
    Dataset validation = {0};
    Dataset testing = {0};
    dataset_generate(&training, 256, 2025);
    dataset_generate(&validation, 128, 2026);
    dataset_generate(&testing, 256, 2027);
    test_data();
    Network initial = {0};
    network_initialize(&initial, 42);
    test_numerics(&initial);
    if (!quiet) {
        puts("=== 二维 MLP：2→16(ReLU)→16(ReLU)→1，337 个参数 ===\n"
             "数据：内盘为 1、外环为 0；训练/验证/测试 = 256/128/256；批量 16。\n"
             "种子：初始化 42，数据 2025/2026/2027，打乱 2028。\n"
             "比较 SGD 与 Adam：相同初始化、批次顺序及 200 轮预算，学习率分别固定。");
    }
    Network sgd_best = {0};
    Network adam_best = {0};
    if (!train(&initial, &training, &validation, OPTIMIZER_SGD, images, quiet, &sgd_best) ||
        !train(&initial, &training, &validation, OPTIMIZER_ADAM, images, quiet, &adam_best)) {
        return EXIT_FAILURE;
    }
    Metrics sgd_valid = evaluate(&sgd_best, &validation);
    Metrics adam_valid = evaluate(&adam_best, &validation);
    const Network *selected = &sgd_best;
    const char *name = "SGD";
    if (adam_valid.loss < sgd_valid.loss) {
        selected = &adam_best;
        name = "Adam";
    }
    /* 测试集到此才参与最终评估，不用于选轮数、选优化器或计算梯度。 */
    Metrics final_train = evaluate(selected, &training);
    Metrics final_test = evaluate(selected, &testing);
    assert(evaluate(&sgd_best, &training).loss < evaluate(&initial, &training).loss);
    assert(evaluate(&adam_best, &training).loss < evaluate(&initial, &training).loss);
    assert(sgd_valid.accuracy >= 0.9 && adam_valid.accuracy >= 0.9);
    assert(final_test.accuracy >= 0.9 && final_test.loss < 0.2);
    printf("验证集选择 %s：训练准确率 %.1f%%；独立测试准确率 %.1f%%，交叉熵 %.4f\n", name,
           100 * final_train.accuracy, 100 * final_test.accuracy, final_test.loss);
    puts("阶段 2 验证通过：337 个参数梯度、尾批、打乱、SGD/Adam 与独立评估。");
    return EXIT_SUCCESS;
}
