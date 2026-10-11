/**
 * @file
 * @brief 阶段 1：单个 sigmoid 神经元与手写 2→4→1 XOR 网络。
 * 固定小数据、固定种子和短训练；仅依赖 C 标准库与数学库。
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUTS 2
#define HIDDEN 4
#define SAMPLES 4
#define EPOCHS 5000
#define LEARNING_RATE 0.1

/* 扁平参数便于逐个检查梯度；W1 按隐藏神经元逐行存放。 */
enum {
    W1 = 0,
    B1 = W1 + HIDDEN * INPUTS,
    W2 = B1 + HIDDEN,
    B2 = W2 + HIDDEN,
    PARAMETERS = B2 + 1
};

typedef struct {
    double parameters[PARAMETERS];
} Network;

typedef struct {
    double hidden[HIDDEN]; /* 保存 tanh 输出，反向时使用 1-h²。 */
    double logit;          /* sigmoid 前的实数分数。 */
    double probability;
} Forward;

static const double xor_inputs[SAMPLES][INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const double xor_targets[SAMPLES] = {0, 1, 1, 0};

/**
 * @brief 分支计算 sigmoid，避免 exp 的大正指数溢出。
 * @param logit 有限实数；不接受 NaN 或无穷作为接口输入。
 * @return [0,1] 内的概率；极端输入可能因舍入到达端点。
 */
static double sigmoid(double logit)
{
    if (logit >= 0) {
        return 1 / (1 + exp(-logit));
    }
    double exponential = exp(logit);
    return exponential / (1 + exponential);
}

/**
 * @brief 从 logit 计算稳定二分类交叉熵，不对舍入后的概率取对数。
 * @param logit 有限实数输出。
 * @param target 标签，必须为 0 或 1。
 * @return 非负损失；正确且极端自信时可能舍入为零。
 */
static double binary_cross_entropy(double logit, double target)
{
    return fmax(logit, 0) - target * logit + log1p(exp(-fabs(logit)));
}

/**
 * @brief 产生可重复的均匀权重，不依赖各平台 rand 的不同实现。
 * @param state 可写随机状态，不转移所有权；uint32_t 回绕有明确定义。
 * @return [-0.8,0.8) 内的初始化权重；不是密码学随机数。
 */
static double random_weight(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    double unit = (double)(*state >> 8) / 16777216.0;
    return 1.6 * unit - 0.8;
}

/**
 * @brief 随机初始化两层权重，偏置置零以保持起步简单。
 * @param network 可写网络，不转移所有权；全部旧参数会被覆盖。
 * @param seed 固定随机种子，用于重复实验。
 */
static void network_initialize(Network *network, uint32_t seed)
{
    *network = (Network){0};
    for (size_t h = 0; h < HIDDEN; ++h) {
        for (size_t i = 0; i < INPUTS; ++i) {
            network->parameters[W1 + h * INPUTS + i] = random_weight(&seed);
        }
        network->parameters[W2 + h] = random_weight(&seed);
    }
}

/**
 * @brief 前向传播：输入→tanh 隐藏层→线性分数→sigmoid。
 * @param network 有效的只读网络；参数及中间运算须有限且不溢出。
 * @param x 包含 INPUTS 个有限特征的只读数组，不转移所有权。
 * @return 隐藏激活、logit 与预测概率；网络参数不变。
 */
static Forward network_forward(const Network *network, const double x[INPUTS])
{
    Forward forward = {0};
    forward.logit = network->parameters[B2];
    for (size_t h = 0; h < HIDDEN; ++h) {
        double preactivation = network->parameters[B1 + h];
        for (size_t i = 0; i < INPUTS; ++i) {
            preactivation += network->parameters[W1 + h * INPUTS + i] * x[i];
        }
        forward.hidden[h] = tanh(preactivation);
        forward.logit += network->parameters[W2 + h] * forward.hidden[h];
    }
    forward.probability = sigmoid(forward.logit);
    return forward;
}

/**
 * @brief 反向传播一个样本，覆盖所有梯度，不在此更新参数。
 * @param network 与 forward 对应的只读旧参数，反向完成前不可修改。
 * @param x 对应前向传播的只读输入，长度为 INPUTS。
 * @param target 0/1 标签。
 * @param forward 当前网络和输入的只读前向缓存。
 * @param gradients 至少 PARAMETERS 个 double 的可写数组，不得与网络参数重叠。
 */
static void network_backward(const Network *network, const double x[INPUTS], double target,
                             const Forward *forward, double gradients[PARAMETERS])
{
    /* sigmoid 与交叉熵合并后 dL/dlogit = probability-target。 */
    double output_delta = forward->probability - target;
    gradients[B2] = output_delta;
    for (size_t h = 0; h < HIDDEN; ++h) {
        gradients[W2 + h] = output_delta * forward->hidden[h];
        double hidden_delta = output_delta * network->parameters[W2 + h] *
                              (1 - forward->hidden[h] * forward->hidden[h]);
        gradients[B1 + h] = hidden_delta;
        for (size_t i = 0; i < INPUTS; ++i) {
            gradients[W1 + h * INPUTS + i] = hidden_delta * x[i];
        }
    }
}

/**
 * @brief 所有梯度就绪后执行一次 SGD 更新。
 * @param network 可写网络；与梯度数组不得重叠，不转移所有权。
 * @param gradients 当前网络上计算的 PARAMETERS 个只读梯度。
 * @param learning_rate 有限正数，更新过程须不溢出。
 */
static void network_update(Network *network, const double gradients[PARAMETERS],
                           double learning_rate)
{
    for (size_t p = 0; p < PARAMETERS; ++p) {
        network->parameters[p] -= learning_rate * gradients[p];
        assert(isfinite(network->parameters[p]));
    }
}

/**
 * @brief 评估同一版本网络在完整 XOR 真值表上的平均交叉熵。
 * @param network 有效的只读网络，不修改参数。
 * @return 四个样本的平均损失；仅用于本固定任务的实现验证。
 */
static double network_loss(const Network *network)
{
    double total = 0;
    for (size_t sample = 0; sample < SAMPLES; ++sample) {
        Forward forward = network_forward(network, xor_inputs[sample]);
        total += binary_cross_entropy(forward.logit, xor_targets[sample]);
    }
    return total / SAMPLES;
}

/**
 * @brief 验证激活和损失的极端值、单神经元手算以及 sigmoid+BCE 的导数。
 */
static void test_neuron(void)
{
    assert(sigmoid(-1000) == 0 && sigmoid(1000) == 1);
    assert(sigmoid(0) == 0.5);
    assert(fabs(binary_cross_entropy(0, 0) - log(2)) < 1e-12);
    assert(binary_cross_entropy(1000, 0) == 1000);
    assert(binary_cross_entropy(-1000, 1) == 1000);
    assert(binary_cross_entropy(1000, 1) == 0);
    assert(binary_cross_entropy(-1000, 0) == 0);

    double logit = 0.5 * 1 + (-0.25) * 2 + 0.1;
    double probability = sigmoid(logit);
    double loss = binary_cross_entropy(logit, 1);
    double epsilon = 1e-5;
    double numerical =
        (binary_cross_entropy(logit + epsilon, 1) - binary_cross_entropy(logit - epsilon, 1)) /
        (2 * epsilon);
    assert(fabs(probability - 0.52497918747894) < 1e-12);
    assert(fabs(loss - 0.644396660073571) < 1e-12);
    assert(fabs(numerical - (probability - 1)) < 1e-8);
}

/**
 * @brief 对四个样本逐一检查全部 17 个参数的中心差分梯度。
 * @param network 有效的只读测试网络，使用副本扰动参数，原网络保持不变。
 * @return 解析梯度与数值梯度的最大绝对误差。
 */
static double test_network_gradients(const Network *network)
{
    double maximum_error = 0;
    const double epsilon = 1e-5;
    for (size_t sample = 0; sample < SAMPLES; ++sample) {
        Forward forward = network_forward(network, xor_inputs[sample]);
        double gradients[PARAMETERS] = {0};
        network_backward(network, xor_inputs[sample], xor_targets[sample], &forward, gradients);
        for (size_t p = 0; p < PARAMETERS; ++p) {
            Network plus = *network;
            Network minus = *network;
            plus.parameters[p] += epsilon;
            minus.parameters[p] -= epsilon;
            Forward plus_forward = network_forward(&plus, xor_inputs[sample]);
            Forward minus_forward = network_forward(&minus, xor_inputs[sample]);
            double numerical = (binary_cross_entropy(plus_forward.logit, xor_targets[sample]) -
                                binary_cross_entropy(minus_forward.logit, xor_targets[sample])) /
                               (2 * epsilon);
            double error = fabs(numerical - gradients[p]);
            maximum_error = fmax(maximum_error, error);
            assert(isfinite(numerical) && isfinite(gradients[p]));
            assert(error < 1e-8 + 1e-6 * fmax(fabs(numerical), fabs(gradients[p])));
        }
    }
    return maximum_error;
}

/**
 * @brief 验证零初始化的对称性陷阱及一次更新的方向。
 * @param initialized 已随机初始化的只读网络，不转移所有权。
 */
static void test_initialization_and_update(const Network *initialized)
{
    Network zero = {0};
    double average[PARAMETERS] = {0};
    for (size_t sample = 0; sample < SAMPLES; ++sample) {
        Forward forward = network_forward(&zero, xor_inputs[sample]);
        double gradients[PARAMETERS] = {0};
        network_backward(&zero, xor_inputs[sample], xor_targets[sample], &forward, gradients);
        assert(forward.probability == 0.5);
        for (size_t p = 0; p < PARAMETERS; ++p) {
            average[p] += gradients[p] / SAMPLES;
        }
    }
    for (size_t p = 0; p < PARAMETERS; ++p) {
        assert(average[p] == 0);
    }
    Network updated = *initialized;
    Forward before = network_forward(&updated, xor_inputs[1]);
    double gradients[PARAMETERS] = {0};
    network_backward(&updated, xor_inputs[1], xor_targets[1], &before, gradients);
    network_update(&updated, gradients, 0.01);
    Forward after = network_forward(&updated, xor_inputs[1]);
    assert(binary_cross_entropy(after.logit, 1) < binary_cross_entropy(before.logit, 1));
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(fabs(updated.parameters[i] - (initialized->parameters[i] - 0.01 * gradients[i])) <
               1e-12);
    }
}

/**
 * @brief 演示单神经元，训练并验证 XOR；输出数量固定，不写日志文件。
 * @return 所有断言通过返回 EXIT_SUCCESS；断言失败时终止，保留调试位置。
 */
int main(void)
{
    test_neuron();
    double logit = 0.5 * 1 + (-0.25) * 2 + 0.1;
    double probability = sigmoid(logit);
    double delta = probability - 1;
    puts("=== 单个 sigmoid 神经元：x=[1,2]，w=[0.5,-0.25]，b=0.1，y=1 ===");
    printf("logit=%.4f，概率=%.6f，交叉熵=%.6f\n", logit, probability,
           binary_cross_entropy(logit, 1));
    printf("dL/dlogit=p-y=%.6f；(dw0,dw1,db)=(%.6f,%.6f,%.6f)\n", delta, delta, 2 * delta, delta);

    Network network = {0};
    network_initialize(&network, UINT32_C(42));
    Network repeated = {0};
    network_initialize(&repeated, UINT32_C(42));
    for (size_t p = 0; p < PARAMETERS; ++p) {
        assert(network.parameters[p] == repeated.parameters[p]);
    }
    double gradient_error = test_network_gradients(&network);
    test_initialization_and_update(&network);
    double initial_loss = network_loss(&network);
    puts("\n=== XOR：2→4(tanh)→1(sigmoid)，17 个参数，固定种子 42 ===");
    printf("全部参数梯度检查通过，最大绝对误差 %.3e\n", gradient_error);
    printf("epoch=%4d，平均交叉熵=%.6f\n", 0, initial_loss);
    for (size_t epoch = 1; epoch <= EPOCHS; ++epoch) {
        /* SGD：每看一个样本就更新一次；此入门例固定顺序，不使用小批量。 */
        for (size_t sample = 0; sample < SAMPLES; ++sample) {
            Forward forward = network_forward(&network, xor_inputs[sample]);
            double gradients[PARAMETERS] = {0};
            network_backward(&network, xor_inputs[sample], xor_targets[sample], &forward,
                             gradients);
            network_update(&network, gradients, LEARNING_RATE);
        }
        if (epoch % 500 == 0) {
            double loss = network_loss(&network);
            assert(isfinite(loss));
            printf("epoch=%4zu，平均交叉熵=%.6f\n", epoch, loss);
        }
    }
    double final_loss = network_loss(&network);
    assert(isfinite(final_loss) && final_loss < initial_loss && final_loss < 0.02);
    puts("\n=== 四组真值表预测（概率 >= 0.5 判为 1）===");
    for (size_t sample = 0; sample < SAMPLES; ++sample) {
        Forward forward = network_forward(&network, xor_inputs[sample]);
        int predicted = forward.probability >= 0.5;
        assert(isfinite(forward.probability));
        assert(predicted == (int)xor_targets[sample]);
        if (xor_targets[sample] == 1) {
            assert(forward.probability > 0.95);
        } else {
            assert(forward.probability < 0.05);
        }
        printf("x=[%.0f,%.0f]，目标=%.0f，概率=%.6f，预测=%d\n", xor_inputs[sample][0],
               xor_inputs[sample][1], xor_targets[sample], forward.probability, predicted);
    }
    puts("阶段 1 验证通过：稳定损失、全部参数梯度、更新、零初始化与 XOR 拟合。\n"
         "四个样本是完整布尔真值表；拟合结果不代表真实数据泛化能力。");
    return EXIT_SUCCESS;
}
