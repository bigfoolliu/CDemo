/**
 * @file
 * @brief 阶段 3：纯 C 卷积网络识别合成图形，手写前向、反向与 Adam。
 * 固定形状接口只接受有效数组与有限数值；不依赖 SDL 或深度学习框架。
 */
#include "../graphics/bmp_capture.h"
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 输入为 16×16 单通道灰度图；3×3 不填充卷积输出 14×14。 */
#define SIDE 16  // 输入图像边长
#define PIXELS (SIDE * SIDE)  // 输入图像像素总数
#define KERNEL 3  // 卷积核边长
#define CHANNELS 4  // 卷积输出通道数
#define CONV_SIDE (SIDE - KERNEL + 1)  // 卷积输出边长
#define CONV_PIXELS (CONV_SIDE * CONV_SIDE)  // 卷积输出像素总数
#define POOL_SIDE (CONV_SIDE / 2)  // 池化输出边长

/* 每个通道池化为 7×7，展平后四个通道共 196 个特征。 */
#define FEATURES (CHANNELS * POOL_SIDE * POOL_SIDE)  // 全连接层输入特征数
#define CLASSES 3  // 输出类别数

/* 张量形状依次为 1×16×16 → 4×14×14 → 4×7×7 → 196 → 3。 */

/* 固定容量、批量大小与训练轮数，让示例可重复且便于观察。 */
#define CAPACITY 180  // 数据集容量，必须为 CLASSES 的倍数
#define BATCH_SIZE 12  // 小批量大小，必须为 CLASSES 的倍数
#define EPOCHS 80  // 训练迭代次数，过少可能无法收敛，过多可能过拟合

/* 参数展平到一个数组；枚举标出各参数段起始偏移和总长度。 */
enum {
    CONV_W = 0, /* 四个 3×3 卷积核，共 36 个权重。 */
    CONV_B = CHANNELS * KERNEL * KERNEL, /* 四个卷积偏置从此处开始。 */
    FC_W = CONV_B + CHANNELS, /* 全连接权重紧接卷积偏置。 */
    FC_B = FC_W + CLASSES * FEATURES, /* 分类偏置紧接全连接权重。 */
    PARAMETERS = FC_B + CLASSES
};

/** 一张归一化灰度图和类别编号；像素值在 [0,1]，按行优先排列。 */
typedef struct {
    double image[PIXELS];
    size_t target; /* 0、1、2 分别表示圆形、方形和三角形。 */
} Sample;  // 样本结构体，包含图像数据和目标类别

/** 固定容量的数据集；count 表示 samples 中已生成的样本数。 */
typedef struct {
    size_t count;
    Sample samples[CAPACITY];
} Dataset;  // 数据集结构体，包含样本数和样本数组

/** 按上方偏移枚举顺序存放全部卷积层和分类层参数。 */
typedef struct {
    double parameters[PARAMETERS];
} Network;  // 神经网络结构体，包含所有可训练参数

/** 一次前向传播的缓存；反向传播会用到池化窗口的获胜位置。 */
typedef struct {
    double preactivation[CHANNELS * CONV_PIXELS]; /* 卷积点积加偏置，尚未经过 ReLU。 */
    double activation[CHANNELS * CONV_PIXELS]; /* ReLU 输出，按通道、行、列排列。 */
    double pooled[FEATURES]; /* 池化后展平的全连接输入。 */
    size_t winners[FEATURES]; /* 池化赢家在 activation 中的绝对下标。 */
    double logits[CLASSES]; /* softmax 前的类别分数。 */
    double probabilities[CLASSES]; /* softmax 后的类别概率。 */
} Forward;

/** Adam 的逐参数一阶矩、二阶矩及偏差修正幂次。 */
typedef struct {
    double first[PARAMETERS];
    double second[PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

/** 一次数据集评估结果；混淆矩阵的行是真实类别，列是预测类别。 */
typedef struct {
    double loss; /* 数据集上的平均交叉熵。 */
    size_t correct; /* 预测正确的样本数。 */
    size_t confusion[CLASSES][CLASSES];
} Metrics;

/* 类别编号顺序须与 Sample.target 和混淆矩阵下标一致。 */
static const char *class_names[CLASSES] = {"圆形", "方形", "三角形"};

/**
 * @brief 生成可重复的 [0,1) 伪随机数。
 * @param state 可写独立随机状态，不转移所有权；无符号回绕有明确定义。
 * @return 状态高 24 位映射的实数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 生成灰度图形；各类别共用位置、尺寸、亮度、噪声的采样范围。
 * @param data 可写数据集，旧样本被覆盖。
 * @param count 样本数，3..CAPACITY 且为 CLASSES 的倍数。
 * @param seed 调用者为该数据集提供的种子；函数按值使用，不回写状态。
 */
static void dataset_generate(Dataset *data, size_t count, uint32_t seed)
{
    assert(count >= CLASSES && count <= CAPACITY && count % CLASSES == 0);
    data->count = count;
    for (size_t i = 0; i < count; ++i) {
        Sample *sample = &data->samples[i];
        /* 按类别轮流分配标签，图形参数仍逐张独立采样。 */
        sample->target = i % CLASSES;
        double center_x = 6.5 + 2 * random_unit(&seed);
        double center_y = 6.5 + 2 * random_unit(&seed);
        double radius = 3 + 1.5 * random_unit(&seed);
        double brightness = 0.8 + 0.2 * random_unit(&seed);
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                double dx = (double)column - center_x;
                double dy = (double)row - center_y;
                /* 默认使用圆形方程，再按标签替换为方形或三角形区域。 */
                bool inside = dx * dx + dy * dy <= radius * radius;
                if (sample->target == 1) {
                    inside = fabs(dx) <= radius && fabs(dy) <= radius;
                } else if (sample->target == 2) {
                    inside = dy >= -radius && dy <= radius && fabs(dx) <= (dy + radius) * 0.5;
                }
                double value = 0;
                if (inside) {
                    value = brightness;
                }
                /* 背景和前景都加噪声，再裁切到网络使用的灰度范围。 */
                value += 0.08 * (random_unit(&seed) - 0.5);
                sample->image[row * SIDE + column] = fmin(1, fmax(0, value));
            }
        }
    }
}

/**
 * @brief 计算一个有效区域的 3×3 互相关，不翻转卷积核、不填充。
 * @param image PIXELS 个只读有限灰度元素。
 * @param kernel KERNEL*KERNEL 个只读权重，中间运算须不溢出。
 * @param row 输出行，0..CONV_SIDE-1。
 * @param column 输出列，0..CONV_SIDE-1。
 * @return 对应输入小块与核的点积；深度学习习惯将此操作称为卷积。
 */
static double convolution_at(const double *image, const double *kernel, size_t row, size_t column)
{
    double value = 0;
    /* 输入按 SIDE 行优先索引，核按 KERNEL 行优先索引，逐项累加点积。 */
    for (size_t ky = 0; ky < KERNEL; ++ky) {
        for (size_t kx = 0; kx < KERNEL; ++kx) {
            value += image[(row + ky) * SIDE + column + kx] * kernel[ky * KERNEL + kx];
        }
    }
    return value;
}

/**
 * @brief 减去最大 logit 后计算 softmax，避免指数溢出。
 * @param logits CLASSES 个只读有限分数，分数差须在有限范围内。
 * @param probabilities CLASSES 个可写输出，不与 logits 重叠。
 */
static void softmax(const double logits[CLASSES], double probabilities[CLASSES])
{
    double maximum = logits[0];
    for (size_t c = 1; c < CLASSES; ++c) {
        maximum = fmax(maximum, logits[c]);
    }
    double sum = 0;
    /* 先减最大分数再取指数；最后除以总和得到概率分布。 */
    for (size_t c = 0; c < CLASSES; ++c) {
        probabilities[c] = exp(logits[c] - maximum);
        sum += probabilities[c];
    }
    for (size_t c = 0; c < CLASSES; ++c) {
        probabilities[c] /= sum;
    }
}

/**
 * @brief 用 log-sum-exp 计算三分类交叉熵，不对舍入概率取 log。
 * @param f 有效只读前向缓存。
 * @param target 正确类别，0..CLASSES-1。
 * @return 非负损失；极端正确预测可能舍入为零。
 */
static double cross_entropy(const Forward *f, size_t target)
{
    double maximum = f->logits[0];
    for (size_t c = 1; c < CLASSES; ++c) {
        maximum = fmax(maximum, f->logits[c]);
    }
    double sum = 0;
    for (size_t c = 0; c < CLASSES; ++c) {
        sum += exp(f->logits[c] - maximum);
    }
    return (maximum - f->logits[target]) + log(sum);
}

/**
 * @brief 初始化卷积与分类权重，偏置置零。
 * @param network 有效可写网络，全部旧内容被覆盖。
 * @param seed 初始化种子，不转移资源所有权。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    /* 只覆盖权重区段；先前清零的卷积偏置和分类偏置保持为零。 */
    for (size_t p = CONV_W; p < CONV_B; ++p) {
        network->parameters[p] = (2 * random_unit(&seed) - 1) * sqrt(6.0 / (KERNEL * KERNEL));
    }
    /* 池化尺寸按整数计算；仅在权重初始化公式中转换最终特征数。 */
    const size_t feature_count = FEATURES;
    for (size_t p = FC_W; p < FC_B; ++p) {
        network->parameters[p] = (2 * random_unit(&seed) - 1) * sqrt(3.0 / (double)feature_count);
    }
}

/**
 * @brief 卷积→ReLU→2×2 最大池化→展平→全连接→softmax。
 * @param network 有效只读网络，参数及中间运算须有限且不溢出。
 * @param image PIXELS 个只读有限像素，不转移所有权。
 * @param f 有效可写缓存，不能与 network 或 image 重叠，全部字段覆盖。
 */
static void network_forward(const Network *network, const double *image, Forward *f)
{
    /* 每个输出通道复用一个卷积核，输出索引按通道优先展开。 */
    for (size_t channel = 0; channel < CHANNELS; ++channel) {
        const double *kernel = &network->parameters[CONV_W + channel * KERNEL * KERNEL];
        for (size_t row = 0; row < CONV_SIDE; ++row) {
            for (size_t column = 0; column < CONV_SIDE; ++column) {
                size_t index = channel * CONV_PIXELS + row * CONV_SIDE + column;
                f->preactivation[index] = convolution_at(image, kernel, row, column) +
                                          network->parameters[CONV_B + channel];
                f->activation[index] = fmax(0, f->preactivation[index]);
            }
        }
        /* 不重叠的 2×2 窗口降采样；同时保存赢家供反向传播路由梯度。 */
        for (size_t row = 0; row < POOL_SIDE; ++row) {
            for (size_t column = 0; column < POOL_SIDE; ++column) {
                size_t feature = channel * POOL_SIDE * POOL_SIDE + row * POOL_SIDE + column;
                size_t winner = channel * CONV_PIXELS + 2 * row * CONV_SIDE + 2 * column;
                for (size_t dy = 0; dy < 2; ++dy) {
                    for (size_t dx = 0; dx < 2; ++dx) {
                        size_t candidate =
                            channel * CONV_PIXELS + (2 * row + dy) * CONV_SIDE + 2 * column + dx;
                        /* 严格大于才替换：平局选择行优先的第一个位置。 */
                        if (f->activation[candidate] > f->activation[winner]) {
                            winner = candidate;
                        }
                    }
                }
                f->winners[feature] = winner;
                f->pooled[feature] = f->activation[winner];
            }
        }
    }
    /* pooled 已按通道展平；每个类别用独立权重得到一个 logit。 */
    for (size_t c = 0; c < CLASSES; ++c) {
        f->logits[c] = network->parameters[FC_B + c];
        for (size_t feature = 0; feature < FEATURES; ++feature) {
            f->logits[c] += network->parameters[FC_W + c * FEATURES + feature] * f->pooled[feature];
        }
    }
    softmax(f->logits, f->probabilities);
}

/**
 * @brief 反向计算全部参数与输入梯度；共享核和重叠输入的梯度均累加。
 * @param network 对应 f 的只读旧参数，反向结束前不可修改。
 * @param sample 与 f 对应的只读样本，target 为合法类别。
 * @param f 相同网络版本的只读缓存，含池化获胜位置。
 * @param gradients 至少 PARAMETERS 个可写元素，先清零再累加；
 *        不与其他参数或输出数组重叠。
 * @param image_gradient 至少 PIXELS 个可写元素，不与其他参数或输出数组重叠。
 *        训练过程不使用此梯度。
 */
static void network_backward(const Network *network, const Sample *sample, const Forward *f,
                             double gradients[PARAMETERS], double image_gradient[PIXELS])
{
    /* 输出数组由本函数完整写入，因此先清零，再对共享路径累加贡献。 */
    for (size_t p = 0; p < PARAMETERS; ++p) {
        gradients[p] = 0;
    }
    for (size_t i = 0; i < PIXELS; ++i) {
        image_gradient[i] = 0;
    }
    double pooled_gradient[FEATURES] = {0};
    /* softmax 与交叉熵合并后，logit 梯度是概率减去 one-hot 标签。 */
    for (size_t c = 0; c < CLASSES; ++c) {
        double delta = f->probabilities[c] - (double)(c == sample->target);
        gradients[FC_B + c] = delta;
        for (size_t feature = 0; feature < FEATURES; ++feature) {
            gradients[FC_W + c * FEATURES + feature] = delta * f->pooled[feature];
            pooled_gradient[feature] += delta * network->parameters[FC_W + c * FEATURES + feature];
        }
    }
    /* 最大池化只沿获胜位置回传；未获胜位置的梯度为零。 */
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        size_t winner = f->winners[feature];
        if (f->preactivation[winner] <= 0) {
            continue; /* ReLU 零点采用导数 0。 */
        }
        double delta = pooled_gradient[feature];
        size_t channel = winner / CONV_PIXELS;
        size_t row = (winner % CONV_PIXELS) / CONV_SIDE;
        size_t column = winner % CONV_SIDE;
        gradients[CONV_B + channel] += delta;
        /* 卷积核跨位置共享且图像窗口重叠，参数和输入梯度都需累加。 */
        for (size_t ky = 0; ky < KERNEL; ++ky) {
            for (size_t kx = 0; kx < KERNEL; ++kx) {
                size_t pixel = (row + ky) * SIDE + column + kx;
                size_t weight = CONV_W + channel * KERNEL * KERNEL + ky * KERNEL + kx;
                gradients[weight] += delta * sample->image[pixel];
                image_gradient[pixel] += delta * network->parameters[weight];
            }
        }
    }
}

/**
 * @brief 更新 Adam 状态和参数，步数按批次递增。
 * @param network 有效可写网络，不与其他数组重叠。
 * @param gradients PARAMETERS 个有限只读平均梯度。
 * @param adam 有效可写状态，初始矩为零、beta 的幂为 1。
 * @param rate 有限正学习率，计算须不溢出。
 */
static void network_update(Network *network, const double *gradients, Adam *adam, double rate)
{
    /* beta 幂次每次调用前进一次；一次调用对应一个小批次更新。 */
    adam->beta1_power *= 0.9;
    adam->beta2_power *= 0.999;
    for (size_t p = 0; p < PARAMETERS; ++p) {
        adam->first[p] = 0.9 * adam->first[p] + 0.1 * gradients[p];
        adam->second[p] = 0.999 * adam->second[p] + 0.001 * gradients[p] * gradients[p];
        /* 用幂次修正零初始化矩的偏差，再按自适应步长更新参数。 */
        double first = adam->first[p] / (1 - adam->beta1_power);
        double second = adam->second[p] / (1 - adam->beta2_power);
        network->parameters[p] -= rate * first / (sqrt(second) + 1e-8);
        assert(isfinite(network->parameters[p]));
    }
}

/**
 * @brief 返回概率最大的类别，平局取最小类别下标。
 * @param f 有效只读前向缓存。
 * @return 0..CLASSES-1 内的类别。
 */
static size_t predict(const Forward *f)
{
    size_t selected = 0;
    for (size_t c = 1; c < CLASSES; ++c) {
        if (f->probabilities[c] > f->probabilities[selected]) {
            selected = c;
        }
    }
    return selected;
}

/**
 * @brief 固定参数评估，统计平均损失、正确数和混淆矩阵。
 * @param network 有效只读网络，不进行训练。
 * @param data 非空有效只读数据集。
 * @return 评估指标，不修改网络、数据或优化器状态。
 */
static Metrics evaluate(const Network *network, const Dataset *data)
{
    Metrics metrics = {0};
    for (size_t i = 0; i < data->count; ++i) {
        Forward f;
        network_forward(network, data->samples[i].image, &f);
        size_t predicted = predict(&f);
        metrics.loss += cross_entropy(&f, data->samples[i].target);
        metrics.correct += predicted == data->samples[i].target;
        /* 行下标是真实类别，列下标是预测类别。 */
        ++metrics.confusion[data->samples[i].target][predicted];
    }
    metrics.loss /= (double)data->count;
    assert(isfinite(metrics.loss));
    return metrics;
}

/**
 * @brief 用前向/反向分支相同的夹具验证数值梯度。
 * @param a 正方向扰动的只读前向缓存。
 * @param b 负方向扰动的只读缓存。
 * @return ReLU 符号与池化位置完全一致为 true，不用跨不可微点的差分。
 */
static bool same_region(const Forward *a, const Forward *b)
{
    for (size_t i = 0; i < CHANNELS * CONV_PIXELS; ++i) {
        if ((a->preactivation[i] > 0) != (b->preactivation[i] > 0)) {
            return false;
        }
    }
    for (size_t i = 0; i < FEATURES; ++i) {
        if (a->winners[i] != b->winners[i]) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 在严格正激活、唯一池化极大值的夹具上检查全部参数及输入梯度。
 */
static void test_gradients(void)
{
    Network network = {0};
    network_initialize(&network, 42);
    for (size_t p = 0; p < CONV_B; ++p) {
        network.parameters[p] = 0.02 + 0.005 * (double)p;
    }
    for (size_t c = 0; c < CHANNELS; ++c) {
        network.parameters[CONV_B + c] = 0.1;
    }
    Sample sample = {.target = 1};
    for (size_t i = 0; i < PIXELS; ++i) {
        /* 正输入和正卷积权重令 ReLU 状态与池化赢家对小扰动稳定。 */
        sample.image[i] = 0.1 + (double)i / PIXELS;
    }
    Forward f;
    network_forward(&network, sample.image, &f);
    double gradients[PARAMETERS] = {0};
    double image_gradient[PIXELS] = {0};
    network_backward(&network, &sample, &f, gradients, image_gradient);
    /* 中心差分逐一核对 631 个参数梯度和 256 个输入像素梯度。 */
    const double epsilon = 1e-5;
    for (size_t p = 0; p < PARAMETERS; ++p) {
        Network plus = network;
        Network minus = network;
        plus.parameters[p] += epsilon;
        minus.parameters[p] -= epsilon;
        Forward a;
        Forward b;
        network_forward(&plus, sample.image, &a);
        network_forward(&minus, sample.image, &b);
        assert(same_region(&a, &b));
        double numerical =
            (cross_entropy(&a, sample.target) - cross_entropy(&b, sample.target)) / (2 * epsilon);
        assert(fabs(numerical - gradients[p]) < 1e-8 + 1e-6 * fabs(numerical));
    }
    for (size_t i = 0; i < PIXELS; ++i) {
        Sample plus = sample;
        Sample minus = sample;
        plus.image[i] += epsilon;
        minus.image[i] -= epsilon;
        Forward a;
        Forward b;
        network_forward(&network, plus.image, &a);
        network_forward(&network, minus.image, &b);
        assert(same_region(&a, &b));
        double numerical =
            (cross_entropy(&a, sample.target) - cross_entropy(&b, sample.target)) / (2 * epsilon);
        assert(fabs(numerical - image_gradient[i]) < 1e-8 + 1e-6 * fabs(numerical));
    }
}

/**
 * @brief 验证有效边界卷积、池化平局路由、ReLU 零点及极端 softmax。
 */
static void test_rules(void)
{
    /* 全一卷积核把窗口像素求和，用首尾输出检查有效卷积的边界索引。 */
    double image[PIXELS] = {0};
    double kernel[KERNEL * KERNEL] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    for (size_t i = 0; i < PIXELS; ++i) {
        image[i] = (double)(i + 1);
    }
    assert(convolution_at(image, kernel, 0, 0) == 162);
    assert(convolution_at(image, kernel, CONV_SIDE - 1, CONV_SIDE - 1) == 2151);
    Network network = {0};
    /* 卷积核仅中心为 1；全一输入使池化候选相等，检查首个位置胜出。 */
    for (size_t channel = 0; channel < CHANNELS; ++channel) {
        network.parameters[CONV_W + channel * KERNEL * KERNEL + 4] = 1;
    }
    for (size_t feature = 0; feature < FEATURES; ++feature) {
        network.parameters[FC_W + feature] = 0.01;
    }
    Sample sample = {.target = 1};
    for (size_t i = 0; i < PIXELS; ++i) {
        sample.image[i] = 1;
    }
    Forward f;
    network_forward(&network, sample.image, &f);
    assert(f.winners[0] == 0 && f.winners[1] == 2);
    double gradients[PARAMETERS] = {0};
    double image_gradient[PIXELS] = {0};
    network_backward(&network, &sample, &f, gradients, image_gradient);
    assert(fabs(image_gradient[SIDE + 1] - CHANNELS * 0.01 * f.probabilities[0]) < 1e-12);
    assert(image_gradient[SIDE + 2] == 0);
    network = (Network){0};
    /* 零网络让所有 ReLU 输入为零，验证零点导数约定和预测平局规则。 */
    network_forward(&network, sample.image, &f);
    network_backward(&network, &sample, &f, gradients, image_gradient);
    for (size_t p = 0; p < FC_B; ++p) {
        assert(gradients[p] == 0);
    }
    assert(predict(&f) == 0);
    /* 极端 logits 检查稳定 softmax 与 log-sum-exp 交叉熵。 */
    f.logits[0] = 1000;
    f.logits[1] = 0;
    f.logits[2] = -1000;
    softmax(f.logits, f.probabilities);
    assert(f.probabilities[0] == 1 && f.probabilities[2] == 0);
    assert(cross_entropy(&f, 2) == 2000);
}

/**
 * @brief 绘制一个灰度小图，按块放大；特征图可单独归一化用于展示。
 * @param canvas 有效可写画布，不转移所有权。
 * @param values side*side 个只读有限元素。
 * @param side 图像边长，SIDE 或 CONV_SIDE。
 * @param left 画布内起始 x，须容纳 side*4 像素。
 * @param top 画布内起始 y，须容纳 side*4 像素。
 * @param normalize 是否按本图最小/最大值缩放；不参与训练。
 */
static void draw_tile(GfxCanvas *canvas, const double *values, size_t side, int left, int top,
                      bool normalize)
{
    double minimum = 0;
    double maximum = 1;
    if (normalize) {
        minimum = values[0];
        maximum = values[0];
        for (size_t i = 1; i < side * side; ++i) {
            minimum = fmin(minimum, values[i]);
            maximum = fmax(maximum, values[i]);
        }
    }
    /* 常量特征图没有可拉伸范围，保留全黑，避免除以零。 */
    for (size_t row = 0; row < side; ++row) {
        for (size_t column = 0; column < side; ++column) {
            double scaled = 0;
            if (maximum > minimum) {
                scaled = (values[row * side + column] - minimum) / (maximum - minimum);
            }
            uint8_t gray = (uint8_t)lround(255 * fmin(1, fmax(0, scaled)));
            gfx_rect(canvas, left + (int)column * 4, top + (int)row * 4, 4, 4, true,
                     gfx_rgb(gray, gray, gray));
        }
    }
}

/**
 * @brief 导出样本拼图或一个样本的特征图，保留已有文件。
 * @param network 有效只读网络。
 * @param data 至少九个样本的只读数据；拼图前三行各含三种类别。
 * @param features true 显示输入、固定边缘响应和四个学习特征；
 *        false 显示九张样本。
 * @param name 非空 ASCII 字母、数字、下划线或连字符组成的文件名前缀。
 * @return 成功为 true；失败时显示写入原因，已有图片不受影响。
 */
static bool export_images(const Network *network, const Dataset *data, bool features,
                          const char *name)
{
    static GfxCanvas canvas;
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    if (!features) {
        for (size_t i = 0; i < 9; ++i) {
            draw_tile(&canvas, data->samples[i].image, SIDE, 20 + (int)(i % 3) * 100,
                      5 + (int)(i / 3) * 80, false);
        }
    } else {
        const double *image = data->samples[0].image;
        Forward f;
        network_forward(network, image, &f);
        double edge[CONV_PIXELS] = {0};
        /* 该固定核仅用于可视化对照，不是训练得到的卷积核。 */
        const double kernel[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
        for (size_t row = 0; row < CONV_SIDE; ++row) {
            for (size_t column = 0; column < CONV_SIDE; ++column) {
                edge[row * CONV_SIDE + column] = convolution_at(image, kernel, row, column);
            }
        }
        draw_tile(&canvas, image, SIDE, 20, 40, false);
        draw_tile(&canvas, edge, CONV_SIDE, 120, 40, true);
        for (size_t channel = 0; channel < CHANNELS; ++channel) {
            size_t tile = channel + 2;
            draw_tile(&canvas, &f.activation[channel * CONV_PIXELS], CONV_SIDE,
                      20 + (int)(tile % 3) * 100, 40 + (int)(tile / 3) * 100, true);
        }
    }
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(&canvas, name, path)) {
        fprintf(stderr, "CNN 图片导出失败：%s\n", strerror(errno));
        return false;
    }
    printf("已导出：%s\n", path);
    return true;
}

/**
 * @brief 小批量训练，仅使用训练集梯度和验证集选模。
 * @param initial 有效只读起始参数。
 * @param training 非空只读训练数据，样本数不超过 CAPACITY。
 * @param validation 非空独立验证数据，样本数不超过 CAPACITY；
 *        用于评估和选模，不参与梯度。
 * @param quiet 是否省略逐轮输出。
 * @param best 有效可写的最佳网络输出；只保存推理参数，
 *        不保存优化器状态。
 */
static void train(const Network *initial, const Dataset *training, const Dataset *validation,
                  bool quiet, Network *best)
{
    Network current = *initial;
    *best = current;
    /* 保留初始网络作为候选，防止所有训练轮次的验证损失都更高。 */
    double best_loss = evaluate(&current, validation).loss;
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    uint32_t seed = 2028;
    for (size_t epoch = 1; epoch <= EPOCHS; ++epoch) {
        /* 保存样本下标并打乱顺序，不复制或改写训练样本本身。 */
        size_t order[CAPACITY] = {0};
        for (size_t i = 0; i < training->count; ++i) {
            order[i] = i;
        }
        /* Fisher–Yates 式洗牌重排样本；固定局部种子使顺序可复现。 */
        for (size_t remaining = training->count; remaining > 1; --remaining) {
            size_t selected = (size_t)(random_unit(&seed) * (double)remaining);
            size_t temporary = order[remaining - 1];
            order[remaining - 1] = order[selected];
            order[selected] = temporary;
        }
        for (size_t start = 0; start < training->count; start += BATCH_SIZE) {
            /* 末批可能不足 BATCH_SIZE，梯度按当前实际样本数求平均。 */
            size_t count = training->count - start;
            if (count > BATCH_SIZE) {
                count = BATCH_SIZE;
            }
            double average[PARAMETERS] = {0};
            /* 对当前批次逐样本求梯度，再取平均后只更新一次 Adam。 */
            for (size_t i = 0; i < count; ++i) {
                const Sample *sample = &training->samples[order[start + i]];
                Forward f;
                network_forward(&current, sample->image, &f);
                double gradients[PARAMETERS] = {0};
                double image_gradient[PIXELS] = {0};
                network_backward(&current, sample, &f, gradients, image_gradient);
                for (size_t p = 0; p < PARAMETERS; ++p) {
                    average[p] += gradients[p] / (double)count;
                }
            }
            network_update(&current, average, &adam, 0.003);
        }
        Metrics valid = evaluate(&current, validation);
        /* 验证损失只用于挑选参数，不参与当前批次的梯度计算。 */
        if (valid.loss < best_loss) {
            best_loss = valid.loss;
            *best = current;
        }
        if (!quiet && epoch % 10 == 0) {
            printf("epoch=%2zu，训练损失=%.4f，验证损失=%.4f，验证准确率=%.1f%%\n", epoch,
                   evaluate(&current, training).loss, valid.loss,
                   100.0 * (double)valid.correct / (double)validation->count);
        }
    }
}

/**
 * @brief 运行 CNN 案例和数值断言；默认不创建图片或日志。
 * @param argc 命令行参数数目。
 * @param argv 可选 --export 导出 BMP；--test 安静验证，其他参数报错。
 * @return 成功为 EXIT_SUCCESS；参数或文件失败为 EXIT_FAILURE，断言失败终止。
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
    test_rules();
    test_gradients();
    /* 梯度和离散边界通过检查后，再开始模型训练。 */
    /* 静态数据区避免三个固定容量图像集合占用过多线程栈。 */
    static Dataset training;
    static Dataset validation;
    static Dataset testing;
    /* 三组数据使用独立种子；测试集只在训练和验证选模结束后评估。 */
    dataset_generate(&training, 180, 2025);
    dataset_generate(&validation, 90, 2026);
    dataset_generate(&testing, 90, 2027);
    Network initial = {0};
    network_initialize(&initial, 42);
    if (!quiet) {
        puts("=== CNN：16×16→4×14×14 卷积/ReLU→4×7×7 最大池化→三分类 ===\n"
             "631 个参数；训练/验证/测试=180/90/90；Adam 0.003，批量12，80轮。\n"
             "初始化种子42，数据2025/2026/2027，打乱2028。所有参数及输入梯度检查通过。");
        printf("初始训练交叉熵 %.4f\n", evaluate(&initial, &training).loss);
    }
    if (images && (!export_images(&initial, &training, false, "cnn-shapes-samples") ||
                   !export_images(&initial, &validation, true, "cnn-shapes-initial"))) {
        return EXIT_FAILURE;
    }
    Network best = {0};
    train(&initial, &training, &validation, quiet, &best);
    Metrics learned = evaluate(&best, &training);
    /* 测试集不参与反向传播，也不参与 best 参数的选择。 */
    Metrics tested = evaluate(&best, &testing);
    assert(learned.loss < evaluate(&initial, &training).loss);
    assert((double)tested.correct / (double)testing.count >= 0.8 && tested.loss < 0.6);
    printf("独立测试：正确 %zu/%zu，准确率 %.1f%%，交叉熵 %.4f\n", tested.correct, testing.count,
           100.0 * (double)tested.correct / (double)testing.count, tested.loss);
    if (!quiet) {
        puts("混淆矩阵（行=真实，列=预测；圆形/方形/三角形）：");
        for (size_t c = 0; c < CLASSES; ++c) {
            printf("%s：%zu %zu %zu\n", class_names[c], tested.confusion[c][0],
                   tested.confusion[c][1], tested.confusion[c][2]);
        }
    }
    if (images && !export_images(&best, &validation, true, "cnn-shapes-best")) {
        return EXIT_FAILURE;
    }
    puts("阶段 3 验证通过：边界卷积、池化平局、631个参数和256个输入梯度、独立评估。");
    return EXIT_SUCCESS;
}
