/**
 * @file
 * @brief 纯 C GAN：学习圆周上的八簇二维点，观察对抗训练与模式崩塌。
 * 生成器与判别器交替更新；没有编码器、重建配对或外部计算框架。
 */
#include "../common/logger.h"
#include "../graphics/bmp_capture.h"
#include <assert.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>

#define WIDTH 16
#define BATCH_SIZE 32
#define STEPS 6000
#define MODES 8
#define SAMPLES 1024
#define W1 0
#define B1 (WIDTH * 2)
#define W2 (B1 + WIDTH)
#define B2 (W2 + WIDTH * WIDTH)
#define W3 (B2 + WIDTH)
#define B3 (W3 + WIDTH * 2)
#define PARAMETERS (B3 + 2)

typedef struct {
    size_t outputs; /* 生成器为2，判别器为1；尾层为线性，概率另行计算。 */
    double parameters[PARAMETERS];
} Network;

typedef struct {
    double input[2];
    double first[WIDTH];
    double second[WIDTH];
    double output[2];
} Forward;

typedef struct {
    double first[PARAMETERS];
    double second[PARAMETERS];
    double beta1_power;
    double beta2_power;
} Adam;

typedef struct {
    Logger *logger;
    bool failed;
} Output;

typedef struct {
    double distance;
    double quality;
    size_t covered;
    size_t counts[MODES];
    double real_probability;
    double fake_probability;
} Metrics;

/**
 * @brief 输出报告，可选双输出日志，累计输出失败。
 * @param output 有效可写上下文，借用logger。
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
 * @brief 返回开区间(0,1)伪随机数，避免Box-Muller对零取对数。
 * @param state 有效可写独立随机状态，不用于安全用途。
 * @return 可重复均匀数。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return ((double)(*state >> 8) + 0.5) / 16777216.0;
}

/**
 * @brief 用Box-Muller生成两个标准正态数，训练和评估使用独立状态。
 * @param state 有效可写随机状态。
 * @param epsilon 2个可写值，覆盖旧内容。
 */
static void sample_normal(uint32_t *state, double *epsilon)
{
    double radius = sqrt(-2 * log(random_unit(state)));
    double angle = 2 * acos(-1) * random_unit(state);
    epsilon[0] = radius * cos(angle);
    epsilon[1] = radius * sin(angle);
}

/**
 * @brief 返回第mode簇中心，半径为2，逆时针排列。
 * @param mode 0..MODES-1。
 * @param point 两个可写坐标，不转移所有权。
 */
static void mode_center(size_t mode, double *point)
{
    assert(mode < MODES);
    double angle = 2 * acos(-1) * (double)mode / MODES;
    point[0] = 2 * cos(angle);
    point[1] = 2 * sin(angle);
}

/**
 * @brief 独立抽取八个等权二维高斯之一，坐标噪声标准差0.08。
 * @param state 有效可写随机状态。
 * @param point 两个可写输出坐标。
 */
static void sample_real(uint32_t *state, double *point)
{
    size_t mode = (size_t)(MODES * random_unit(state));
    double noise[2];
    sample_normal(state, noise);
    mode_center(mode, point);
    point[0] += 0.08 * noise[0];
    point[1] += 0.08 * noise[1];
}

/**
 * @brief 初始化两层tanh网络，Xavier权重、零偏置。
 * @param network 有效可写网络，覆盖旧内容。
 * @param outputs 1为判别logit，2为生成坐标。
 * @param seed 固定初始化种子。
 */
static void network_initialize(Network *network, size_t outputs, uint32_t seed)
{
    assert(outputs == 1 || outputs == 2);
    *network = (Network){.outputs = outputs};
    const size_t offsets[] = {W1, W2, W3};
    const size_t inputs[] = {2, WIDTH, WIDTH};
    const size_t sizes[] = {WIDTH, WIDTH, outputs};
    for (size_t layer = 0; layer < 3; ++layer) {
        double scale = sqrt(6.0 / (double)(inputs[layer] + sizes[layer]));
        for (size_t i = 0; i < inputs[layer] * sizes[layer]; ++i) {
            network->parameters[offsets[layer] + i] = scale * (2 * random_unit(&seed) - 1);
        }
    }
}

/**
 * @brief 前向保存缓存，第三层输出线性值；D的sigmoid在损失处计算。
 * @param network 有效只读网络，输出维数须为1或2。
 * @param input 两个有限只读坐标；G输入标准正态噪声，D输入数据点。
 * @param forward 有效可写缓存，覆盖旧内容，不允许与input重叠。
 */
static void network_forward(const Network *network, const double *input, Forward *forward)
{
    *forward = (Forward){.input = {input[0], input[1]}};
    const double *p = network->parameters;
    for (size_t i = 0; i < WIDTH; ++i) {
        double value = p[B1 + i];
        for (size_t j = 0; j < 2; ++j) {
            value += p[W1 + i * 2 + j] * input[j];
        }
        forward->first[i] = tanh(value);
    }
    for (size_t i = 0; i < WIDTH; ++i) {
        double value = p[B2 + i];
        for (size_t j = 0; j < WIDTH; ++j) {
            value += p[W2 + i * WIDTH + j] * forward->first[j];
        }
        forward->second[i] = tanh(value);
    }
    for (size_t i = 0; i < network->outputs; ++i) {
        double value = p[B3 + i];
        for (size_t j = 0; j < WIDTH; ++j) {
            value += p[W3 + i * WIDTH + j] * forward->second[j];
        }
        forward->output[i] = value;
        assert(isfinite(value));
    }
}

/**
 * @brief 根据输出导数回传，参数梯度累加，输入梯度覆盖。
 * @param network 与缓存对应的只读网络；反向不会修改网络。
 * @param forward 有效只读前向缓存。
 * @param upstream 至少outputs个只读输出导数。
 * @param gradient PARAMETERS个可写累加槽，调用者负责初始化为零。
 * @param input_gradient 两个可写输入导数；用于从D回传至G。
 */
static void network_backward(const Network *network, const Forward *forward, const double *upstream,
                             double *gradient, double *input_gradient)
{
    const double *p = network->parameters;
    double second[WIDTH] = {0};
    for (size_t i = 0; i < network->outputs; ++i) {
        gradient[B3 + i] += upstream[i];
        for (size_t j = 0; j < WIDTH; ++j) {
            gradient[W3 + i * WIDTH + j] += upstream[i] * forward->second[j];
            second[j] += p[W3 + i * WIDTH + j] * upstream[i];
        }
    }
    double first[WIDTH] = {0};
    for (size_t i = 0; i < WIDTH; ++i) {
        double delta = second[i] * (1 - forward->second[i] * forward->second[i]);
        gradient[B2 + i] += delta;
        for (size_t j = 0; j < WIDTH; ++j) {
            gradient[W2 + i * WIDTH + j] += delta * forward->first[j];
            first[j] += p[W2 + i * WIDTH + j] * delta;
        }
    }
    input_gradient[0] = 0;
    input_gradient[1] = 0;
    for (size_t i = 0; i < WIDTH; ++i) {
        double delta = first[i] * (1 - forward->first[i] * forward->first[i]);
        gradient[B1 + i] += delta;
        for (size_t j = 0; j < 2; ++j) {
            gradient[W1 + i * 2 + j] += delta * forward->input[j];
            input_gradient[j] += p[W1 + i * 2 + j] * delta;
        }
    }
}

/**
 * @brief 稳定sigmoid，不对大负数直接计算exp(-value)。
 * @param value 有限logit。
 * @return 0..1概率。
 */
static double sigmoid(double value)
{
    if (value >= 0) {
        return 1 / (1 + exp(-value));
    }
    double exponential = exp(value);
    return exponential / (1 + exponential);
}

/**
 * @brief 从logit直接计算稳定BCE，避免log(0)。
 * @param logit 有限判别器输出。
 * @param target 有限0..1目标，本例只使用0或1。
 * @return softplus(logit)-target*logit。
 */
static double binary_loss(double logit, double target)
{
    return fmax(logit, 0) - target * logit + log1p(exp(-fabs(logit)));
}

/**
 * @brief 冻结D参数，把非饱和生成器损失的导数经D传回G。
 * @param generator 有效只读生成器，outputs为2。
 * @param discriminator 有效只读判别器，outputs为1。
 * @param noise 两个只读潜在坐标，数值差分须保持相同值。
 * @param gradient PARAMETERS个可写生成器梯度累加槽。
 * @param noise_gradient 两个可写噪声导数。
 * @return BCE(D(G(z)),1)；不使用真实点作配对重建目标。
 */
static double generator_backward(const Network *generator, const Network *discriminator,
                                 const double *noise, double *gradient, double *noise_gradient)
{
    Forward generated;
    Forward judged;
    network_forward(generator, noise, &generated);
    network_forward(discriminator, generated.output, &judged);
    double upstream = sigmoid(judged.output[0]) - 1;
    double discarded[PARAMETERS] = {0};
    double point_gradient[2];
    /* D需要回传输入导数，但其参数梯度只写入临时槽，不调用D的优化器。 */
    network_backward(discriminator, &judged, &upstream, discarded, point_gradient);
    network_backward(generator, &generated, point_gradient, gradient, noise_gradient);
    return binary_loss(judged.output[0], 1);
}

/**
 * @brief 全局梯度裁剪5，独立Adam状态更新；先验证候选值再提交。
 * @param network 有效可写网络，失败保留所有参数。
 * @param adam 有效可写优化器；初始beta幂为1，其余为0。
 * @param gradient PARAMETERS个只读批量平均梯度。
 * @return 更新有限时为true，失败也保留优化器；学习率0.001、beta1=0.5。
 */
static bool adam_update(Network *network, Adam *adam, const double *gradient)
{
    double norm = 0;
    for (size_t i = 0; i < PARAMETERS; ++i) {
        if (!isfinite(gradient[i])) {
            return false;
        }
        norm = hypot(norm, gradient[i]);
    }
    if (!isfinite(norm)) {
        return false;
    }
    double scale = norm > 5 ? 5 / norm : 1;
    Network candidate = *network;
    Adam next = *adam;
    next.beta1_power *= 0.5;
    next.beta2_power *= 0.999;
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double g = gradient[i] * scale;
        next.first[i] = 0.5 * next.first[i] + 0.5 * g;
        next.second[i] = 0.999 * next.second[i] + 0.001 * g * g;
        double first = next.first[i] / (1 - next.beta1_power);
        double second = next.second[i] / (1 - next.beta2_power);
        candidate.parameters[i] -= 0.001 * first / (sqrt(second) + 1e-8);
        if (!isfinite(candidate.parameters[i]) || !isfinite(next.first[i]) ||
            !isfinite(next.second[i])) {
            return false;
        }
    }
    *network = candidate;
    *adam = next;
    return true;
}

/**
 * @brief 在独立种子上统计最近簇距离、有效样本比例和覆盖，不参与更新。
 * @param generator 有效只读生成器。
 * @param discriminator 有效只读判别器。
 * @param seed 评估随机种子；与训练状态分离。
 * @return SAMPLES个生成点；距离<=0.3为有效，某簇>=8个有效点为覆盖。
 */
static Metrics evaluate(const Network *generator, const Network *discriminator, uint32_t seed)
{
    Metrics result = {0};
    for (size_t i = 0; i < SAMPLES; ++i) {
        double noise[2];
        sample_normal(&seed, noise);
        Forward generated;
        Forward judged;
        network_forward(generator, noise, &generated);
        network_forward(discriminator, generated.output, &judged);
        result.fake_probability += sigmoid(judged.output[0]) / SAMPLES;
        double nearest = INFINITY;
        size_t nearest_mode = 0;
        for (size_t mode = 0; mode < MODES; ++mode) {
            double center[2];
            mode_center(mode, center);
            double distance =
                hypot(generated.output[0] - center[0], generated.output[1] - center[1]);
            if (distance < nearest) {
                nearest = distance;
                nearest_mode = mode;
            }
        }
        result.distance += nearest / SAMPLES;
        if (nearest <= 0.3) {
            ++result.counts[nearest_mode];
            result.quality += 1.0 / SAMPLES;
        }
        double real[2];
        sample_real(&seed, real);
        network_forward(discriminator, real, &judged);
        result.real_probability += sigmoid(judged.output[0]) / SAMPLES;
    }
    for (size_t mode = 0; mode < MODES; ++mode) {
        result.covered += result.counts[mode] >= 8;
    }
    return result;
}

/**
 * @brief 每步先用真实/生成点更新D一次，再用新噪声经更新后的D更新G一次。
 * @param generator 有效可写生成器，不在D阶段更新。
 * @param discriminator 有效可写判别器，不在G阶段更新。
 * @param output 可空报告上下文；空时不打印中间进度。
 * @return 固定预算训练及数值检查成功为true，失败明确报错。
 */
static bool train(Network *generator, Network *discriminator, Output *output)
{
    Adam generator_adam = {.beta1_power = 1, .beta2_power = 1};
    Adam discriminator_adam = {.beta1_power = 1, .beta2_power = 1};
    uint32_t real_seed = 2025;
    uint32_t discriminator_seed = 3025;
    uint32_t generator_seed = 4025;
    for (size_t step = 1; step <= STEPS; ++step) {
        Network frozen_generator = *generator;
        double dg[PARAMETERS] = {0};
        double gg[PARAMETERS] = {0};
        double d_loss = 0;
        double g_loss = 0;
        for (size_t sample = 0; sample < BATCH_SIZE; ++sample) {
            double real[2];
            double noise[2];
            sample_real(&real_seed, real);
            sample_normal(&discriminator_seed, noise);
            Forward generated;
            network_forward(generator, noise, &generated);
            for (size_t kind = 0; kind < 2; ++kind) {
                const double *point = kind == 0 ? real : generated.output;
                double target = kind == 0 ? 1 : 0;
                Forward judged;
                network_forward(discriminator, point, &judged);
                double delta = (sigmoid(judged.output[0]) - target) / BATCH_SIZE;
                double ignored[2];
                network_backward(discriminator, &judged, &delta, dg, ignored);
                d_loss += binary_loss(judged.output[0], target) / BATCH_SIZE;
            }
        }
        if (!adam_update(discriminator, &discriminator_adam, dg)) {
            fputs("GAN判别器更新失败\n", stderr);
            return false;
        }
        assert(memcmp(generator, &frozen_generator, sizeof(frozen_generator)) == 0);
        Network frozen_discriminator = *discriminator;
        for (size_t sample = 0; sample < BATCH_SIZE; ++sample) {
            double noise[2];
            sample_normal(&generator_seed, noise);
            double gradient[PARAMETERS] = {0};
            double ignored[2];
            g_loss +=
                generator_backward(generator, discriminator, noise, gradient, ignored) / BATCH_SIZE;
            for (size_t i = 0; i < PARAMETERS; ++i) {
                gg[i] += gradient[i] / BATCH_SIZE;
            }
        }
        if (!adam_update(generator, &generator_adam, gg)) {
            fputs("GAN生成器更新失败\n", stderr);
            return false;
        }
        assert(memcmp(discriminator, &frozen_discriminator, sizeof(frozen_discriminator)) == 0);
        assert(isfinite(d_loss) && isfinite(g_loss));
        if (output != nullptr && (step == 1 || step % 1000 == 0)) {
            Metrics metrics = evaluate(generator, discriminator, 5025);
            report(output, "step=%zu D_loss=%.4f G_loss=%.4f 距离=%.4f 有效=%.1f%% 覆盖=%zu/8",
                   step, d_loss, g_loss, metrics.distance, 100 * metrics.quality, metrics.covered);
        }
    }
    return true;
}

/**
 * @brief 固定输入验证D的两个标签、输入导数及完整D→G链的参数/噪声导数。
 */
static void test_gradients(void)
{
    Network generator;
    Network discriminator;
    network_initialize(&generator, 2, 42);
    network_initialize(&discriminator, 1, 43);
    double point[2] = {0.7, -0.4};
    const double epsilon = 1e-5;
    for (size_t target = 0; target < 2; ++target) {
        Forward forward;
        network_forward(&discriminator, point, &forward);
        double gradient[PARAMETERS] = {0};
        double dx[2];
        double upstream = sigmoid(forward.output[0]) - (double)target;
        network_backward(&discriminator, &forward, &upstream, gradient, dx);
        for (size_t i = 0; i < PARAMETERS + 2; ++i) {
            double *value = i < PARAMETERS ? &discriminator.parameters[i] : &point[i - PARAMETERS];
            double derivative = i < PARAMETERS ? gradient[i] : dx[i - PARAMETERS];
            double saved = *value;
            *value = saved + epsilon;
            network_forward(&discriminator, point, &forward);
            double plus = binary_loss(forward.output[0], (double)target);
            *value = saved - epsilon;
            network_forward(&discriminator, point, &forward);
            double minus = binary_loss(forward.output[0], (double)target);
            *value = saved;
            double numerical = (plus - minus) / (2 * epsilon);
            assert(fabs(numerical - derivative) < 2e-6 + 2e-4 * fabs(numerical));
        }
    }
    double gradient[PARAMETERS] = {0};
    double dz[2];
    Network frozen = discriminator;
    generator_backward(&generator, &discriminator, point, gradient, dz);
    assert(memcmp(&frozen, &discriminator, sizeof(frozen)) == 0);
    for (size_t i = 0; i < PARAMETERS + 2; ++i) {
        double *value = i < PARAMETERS ? &generator.parameters[i] : &point[i - PARAMETERS];
        double derivative = i < PARAMETERS ? gradient[i] : dz[i - PARAMETERS];
        double saved = *value;
        double scratch[PARAMETERS] = {0};
        double ignored[2];
        *value = saved + epsilon;
        double plus = generator_backward(&generator, &discriminator, point, scratch, ignored);
        *value = saved - epsilon;
        double minus = generator_backward(&generator, &discriminator, point, scratch, ignored);
        *value = saved;
        double numerical = (plus - minus) / (2 * epsilon);
        assert(fabs(numerical - derivative) < 2e-6 + 2e-4 * fabs(numerical));
    }
}

/**
 * @brief 检查稳定损失、随机重复性、单步下降和更新失败的原子性。
 */
static void test_boundaries(void)
{
    assert(binary_loss(0, 1) == log(2));
    assert(binary_loss(1000, 0) == 1000);
    assert(binary_loss(-1000, 1) == 1000);
    assert(sigmoid(1000) == 1 && sigmoid(-1000) == 0);
    uint32_t seed1 = 99;
    uint32_t seed2 = 99;
    double point1[2];
    double point2[2];
    for (size_t i = 0; i < 100; ++i) {
        sample_real(&seed1, point1);
        sample_real(&seed2, point2);
        assert(point1[0] == point2[0] && point1[1] == point2[1]);
    }
    Network network;
    network_initialize(&network, 1, 43);
    Forward forward;
    network_forward(&network, point1, &forward);
    double before = binary_loss(forward.output[0], 1);
    double upstream = sigmoid(forward.output[0]) - 1;
    double gradient[PARAMETERS] = {0};
    double ignored[2];
    network_backward(&network, &forward, &upstream, gradient, ignored);
    Adam adam = {.beta1_power = 1, .beta2_power = 1};
    assert(adam_update(&network, &adam, gradient));
    network_forward(&network, point1, &forward);
    assert(binary_loss(forward.output[0], 1) < before);
    Network saved_network = network;
    Adam saved_adam = adam;
    gradient[PARAMETERS - 1] = NAN;
    assert(!adam_update(&network, &adam, gradient));
    assert(memcmp(&network, &saved_network, sizeof(network)) == 0);
    assert(memcmp(&adam, &saved_adam, sizeof(adam)) == 0);

    /* 固定D和噪声时，G沿非饱和梯度更新应减小该样本的损失。 */
    Network generator;
    network_initialize(&generator, 2, 42);
    memset(gradient, 0, sizeof(gradient));
    before = generator_backward(&generator, &network, point1, gradient, ignored);
    Adam generator_adam = {.beta1_power = 1, .beta2_power = 1};
    assert(adam_update(&generator, &generator_adam, gradient));
    memset(gradient, 0, sizeof(gradient));
    double after = generator_backward(&generator, &network, point1, gradient, ignored);
    assert(after < before);
    assert(memcmp(&network, &saved_network, sizeof(network)) == 0);
}

/**
 * @brief 将[-3,3]坐标映射到画布并画点，忽略非有限或视图外坐标。
 * @param canvas 有效可写画布。
 * @param point 两个只读坐标，绘制时y轴向上。
 * @param color 不透明颜色。
 */
static void draw_point(GfxCanvas *canvas, const double *point, uint32_t color)
{
    if (!isfinite(point[0]) || !isfinite(point[1]) || fabs(point[0]) > 3 || fabs(point[1]) > 3) {
        return;
    }
    int x = 40 + (int)lround((point[0] + 3) * 40);
    int y = 239 - (int)lround((point[1] + 3) * (239.0 / 6));
    gfx_rect(canvas, x - 1, y - 1, 3, 3, true, color);
}

/**
 * @brief 保留旧文件，独占创建BMP，失败报告错误。
 * @param canvas 有效只读画布。
 * @param name 文件名前缀。
 * @param output 有效可写报告上下文。
 * @return 成功为true，失败保留此前导出并停止后续导出。
 */
static bool save_canvas(const GfxCanvas *canvas, const char *name, Output *output)
{
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(canvas, name, path)) {
        fprintf(stderr, "GAN图片导出失败：%s\n", strerror(errno));
        return false;
    }
    report(output, "已导出：%s", path);
    return true;
}

/**
 * @brief 绘制真实绿色点和生成橙色点；固定样本便于前后对照。
 * @param canvas 有效可写画布，覆盖旧内容。
 * @param generator 有效只读生成器。
 */
static void draw_scatter(GfxCanvas *canvas, const Network *generator)
{
    gfx_clear(canvas, gfx_rgb(24, 30, 40));
    gfx_line(canvas, 40, 120, 280, 120, gfx_rgb(60, 70, 80));
    gfx_line(canvas, 160, 0, 160, 239, gfx_rgb(60, 70, 80));
    uint32_t real_seed = 7025;
    uint32_t noise_seed = 8025;
    for (size_t i = 0; i < 256; ++i) {
        double point[2];
        sample_real(&real_seed, point);
        draw_point(canvas, point, gfx_rgb(50, 190, 120));
    }
    for (size_t i = 0; i < 256; ++i) {
        double noise[2];
        sample_normal(&noise_seed, noise);
        Forward generated;
        network_forward(generator, noise, &generated);
        draw_point(canvas, generated.output, gfx_rgb(255, 155, 65));
    }
}

/**
 * @brief 导出前后散点、D热图和G潜在网格，无SDL依赖。
 * @param initial 有效只读初始生成器。
 * @param generator 有效只读训练后生成器。
 * @param discriminator 有效只读训练后判别器。
 * @param output 有效可写报告上下文。
 * @return 四张BMP全部保存成功为true。
 */
static bool export_images(const Network *initial, const Network *generator,
                          const Network *discriminator, Output *output)
{
    static GfxCanvas canvas;
    draw_scatter(&canvas, initial);
    if (!save_canvas(&canvas, "gan-before", output)) {
        return false;
    }
    draw_scatter(&canvas, generator);
    if (!save_canvas(&canvas, "gan-after", output)) {
        return false;
    }
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    for (int row = 0; row < 240; ++row) {
        for (int column = 40; column <= 280; ++column) {
            double point[2] = {(column - 160) / 40.0, 3 - row * (6.0 / 239)};
            Forward judged;
            network_forward(discriminator, point, &judged);
            double probability = sigmoid(judged.output[0]);
            gfx_pixel(&canvas, column, row,
                      gfx_rgb((uint8_t)lround(255 * probability), 40,
                              (uint8_t)lround(255 * (1 - probability))));
        }
    }
    if (!save_canvas(&canvas, "gan-discriminator", output)) {
        return false;
    }
    gfx_clear(&canvas, gfx_rgb(24, 30, 40));
    for (size_t row = 0; row < 25; ++row) {
        for (size_t column = 0; column < 25; ++column) {
            double noise[2] = {-2 + (double)column / 6, -2 + (double)row / 6};
            Forward generated;
            network_forward(generator, noise, &generated);
            draw_point(&canvas, generated.output,
                       gfx_rgb((uint8_t)(40 + column * 8), (uint8_t)(40 + row * 8), 190));
        }
    }
    return save_canvas(&canvas, "gan-latent-grid", output);
}

/**
 * @brief 验证、训练并按选项导出图片和日志；默认不写文件。
 * @param argc 参数数量。
 * @param argv 可选--export和--log，或单独--test。
 * @return 全部数值验证、训练和输出成功为EXIT_SUCCESS，否则为EXIT_FAILURE。
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
    Logger logger = {0};
    if (logging && !logger_open(&logger, "logs/ml/14_gan_2d.log", stdout)) {
        fprintf(stderr, "GAN日志打开失败：%s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    Output output = {.logger = logging ? &logger : nullptr};
    test_gradients();
    test_boundaries();
    Network generator;
    Network discriminator;
    network_initialize(&generator, 2, 42);
    network_initialize(&discriminator, 1, 43);
    Network initial = generator;
    Metrics before = evaluate(&generator, &discriminator, 6025);
    if (!quiet) {
        report(&output, "=== GAN：G 2→16→16→2，D 2→16→16→1；8簇二维点 ===");
        report(&output, "D=真实BCE+生成BCE；G=非饱和BCE目标1；各更新一次，批量32，6000步。");
        report(&output, "初始化42/43，训练真实2025、D噪声3025、G噪声4025；监测5025、测试6025。");
    }
    bool success = train(&generator, &discriminator, quiet ? nullptr : &output);
    if (success) {
        Metrics after = evaluate(&generator, &discriminator, 6025);
        assert(isfinite(after.distance) && after.quality >= 0 && after.quality <= 1);
        assert(after.distance < before.distance);
        report(&output,
               "独立评估：距离 %.4f→%.4f，有效=%.1f%%，覆盖=%zu/8，D(real)=%.3f D(fake)=%.3f",
               before.distance, after.distance, 100 * after.quality, after.covered,
               after.real_probability, after.fake_probability);
        report(&output, "有效簇计数：%zu %zu %zu %zu %zu %zu %zu %zu（每簇至少8点才记作覆盖）",
               after.counts[0], after.counts[1], after.counts[2], after.counts[3], after.counts[4],
               after.counts[5], after.counts[6], after.counts[7]);
        if (images) {
            success = export_images(&initial, &generator, &discriminator, &output);
        }
        report(&output, "GAN数值验证通过：完整D→G链、输入/噪声导数、冻结D与更新边界。");
    }
    if (logging && !logger_close(&logger)) {
        fputs("GAN日志写入或关闭失败\n", stderr);
        success = false;
    }
    if (fflush(stdout) != 0) {
        output.failed = true;
    }
    return success && !output.failed ? EXIT_SUCCESS : EXIT_FAILURE;
}
