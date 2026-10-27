/**
 * @file
 * @brief M03：固定容量标量计算图、反向自动微分及小型 XOR 网络。
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GRAPH_CAPACITY = 128,
    PARAM_COUNT = 9,
    SAMPLE_COUNT = 4,
    TRAIN_EPOCHS = 800
};

typedef enum {
    OP_LEAF,
    OP_ADD,
    OP_MUL,
    OP_EXP,
    OP_LOG,
    OP_TANH,
    OP_SIGMOID
} Op;

typedef struct {
    Op op;
    size_t left;
    size_t right;
    double value;
    double gradient;
} Node;

typedef struct {
    Node nodes[GRAPH_CAPACITY];
    size_t count;
    size_t capacity; /* 测试可缩小容量，不能超过 GRAPH_CAPACITY。 */
} Graph;

typedef struct {
    double values[PARAM_COUNT];
} Network;

typedef struct {
    double prediction;
    double loss;
    double gradients[PARAM_COUNT];
} SampleResult;

static const double XOR_INPUTS[SAMPLE_COUNT][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const double XOR_TARGETS[SAMPLE_COUNT] = {0, 1, 1, 0};
static const Network INITIAL_NETWORK = {{2, 2, -1, 2, 2, -3, 2, -2, -1}};

/**
 * @brief 清空图的逻辑节点数，旧索引立即失效，不释放外部资源。
 * @param graph 可写图，capacity 须为 1..GRAPH_CAPACITY。
 */
static void graph_reset(Graph *graph)
{
    if (graph != nullptr) {
        graph->count = 0;
    }
}

/**
 * @brief 添加叶或操作节点；新节点只可引用较早的节点。
 * @param graph 可写图，不转移所有权。
 * @param op 操作类型。
 * @param left 左父节点；叶节点传 SIZE_MAX。
 * @param right 右父节点；叶和单目节点传 SIZE_MAX。
 * @param value 有限的前向计算值。
 * @return 成功为节点索引，失败为 SIZE_MAX 且不增加 count。
 */
static size_t graph_push(Graph *graph, Op op, size_t left, size_t right, double value)
{
    if (graph == nullptr || graph->capacity == 0 || graph->capacity > GRAPH_CAPACITY ||
        graph->count >= graph->capacity || !isfinite(value)) {
        return SIZE_MAX;
    }
    if (op != OP_LEAF && left >= graph->count) {
        return SIZE_MAX;
    }
    if ((op == OP_ADD || op == OP_MUL) && right >= graph->count) {
        return SIZE_MAX;
    }
    size_t index = graph->count;
    graph->nodes[index] = (Node){.op = op, .left = left, .right = right, .value = value};
    graph->count++;
    return index;
}

/**
 * @brief 向图中加入可求导的标量叶节点。
 * @param graph 可写图。
 * @param value 有限标量；常数也用叶表示。
 * @return 节点索引或 SIZE_MAX；失败时图不变。
 */
static size_t graph_leaf(Graph *graph, double value)
{
    return graph_push(graph, OP_LEAF, SIZE_MAX, SIZE_MAX, value);
}

/**
 * @brief 创建加法节点。
 * @param graph 可写图。
 * @param left 已存在的左节点索引。
 * @param right 已存在的右节点索引，可与 left 相同。
 * @return 节点索引或 SIZE_MAX；失败时图不变。
 */
static size_t graph_add(Graph *graph, size_t left, size_t right)
{
    if (graph == nullptr || left >= graph->count || right >= graph->count) {
        return SIZE_MAX;
    }
    return graph_push(graph, OP_ADD, left, right,
                      graph->nodes[left].value + graph->nodes[right].value);
}

/**
 * @brief 创建乘法节点。
 * @param graph 可写图。
 * @param left 已存在的左节点索引。
 * @param right 已存在的右节点索引，可与 left 相同。
 * @return 节点索引或 SIZE_MAX；失败时图不变。
 */
static size_t graph_mul(Graph *graph, size_t left, size_t right)
{
    if (graph == nullptr || left >= graph->count || right >= graph->count) {
        return SIZE_MAX;
    }
    return graph_push(graph, OP_MUL, left, right,
                      graph->nodes[left].value * graph->nodes[right].value);
}

/**
 * @brief 创建单目节点，保存前向值以供反向局部求导。
 * @param graph 可写图。
 * @param op OP_EXP、OP_LOG、OP_TANH 或 OP_SIGMOID。
 * @param input 已存在的输入节点索引。
 * @return 节点索引或 SIZE_MAX；对数非正、溢出或容量失败时图不变。
 */
static size_t graph_unary(Graph *graph, Op op, size_t input)
{
    if (graph == nullptr || input >= graph->count) {
        return SIZE_MAX;
    }
    double x = graph->nodes[input].value;
    double value = 0;
    switch (op) {
    case OP_EXP:
        value = exp(x);
        break;
    case OP_LOG:
        if (x <= 0) {
            return SIZE_MAX;
        }
        value = log(x);
        break;
    case OP_TANH:
        value = tanh(x);
        break;
    case OP_SIGMOID:
        value = x >= 0 ? 1.0 / (1.0 + exp(-x)) : exp(x) / (1.0 + exp(x));
        break;
    default:
        return SIZE_MAX;
    }
    return graph_push(graph, op, input, SIZE_MAX, value);
}

/**
 * @brief 清除全部梯度后，按节点创建顺序的逆序累加梯度。
 * @param graph 可写图；新节点只能引用较早节点。
 * @param output 有效的标量输出节点索引。
 * @return 全部梯度有限则为 true；失败时清零梯度。
 */
static bool graph_backward(Graph *graph, size_t output)
{
    if (graph == nullptr || output >= graph->count) {
        return false;
    }
    for (size_t i = 0; i < graph->count; ++i) {
        graph->nodes[i].gradient = 0;
    }
    graph->nodes[output].gradient = 1;
    bool valid = true;
    for (size_t remaining = output + 1; remaining > 0; --remaining) {
        Node *node = &graph->nodes[remaining - 1];
        double upstream = node->gradient;
        if (node->op == OP_LEAF) {
            continue;
        }
        Node *left = &graph->nodes[node->left];
        double left_delta = 0;
        double right_delta = 0;
        switch (node->op) {
        case OP_ADD:
            left_delta = upstream;
            right_delta = upstream;
            break;
        case OP_MUL:
            left_delta = upstream * graph->nodes[node->right].value;
            right_delta = upstream * left->value;
            break;
        case OP_EXP:
            left_delta = upstream * node->value;
            break;
        case OP_LOG:
            left_delta = upstream / left->value;
            break;
        case OP_TANH:
            left_delta = upstream * (1.0 - node->value * node->value);
            break;
        case OP_SIGMOID:
            left_delta = upstream * node->value * (1.0 - node->value);
            break;
        case OP_LEAF:
            break;
        }
        if (!isfinite(left_delta) || !isfinite(right_delta)) {
            valid = false;
            break;
        }
        left->gradient += left_delta;
        if (node->op == OP_ADD || node->op == OP_MUL) {
            graph->nodes[node->right].gradient += right_delta;
        }
        if (!isfinite(left->gradient) ||
            ((node->op == OP_ADD || node->op == OP_MUL) &&
             !isfinite(graph->nodes[node->right].gradient))) {
            valid = false;
            break;
        }
    }
    if (valid) {
        return true;
    }
    for (size_t i = 0; i < graph->count; ++i) {
        graph->nodes[i].gradient = 0;
    }
    return false;
}

/**
 * @brief 在图上构造 w0*x0 + w1*x1 + b，可供隐藏层和输出层复用。
 * @param graph 可写图。
 * @param w0 第一个权重叶索引。
 * @param w1 第二个权重叶索引。
 * @param bias 偏置叶索引。
 * @param x0 第一个输入索引。
 * @param x1 第二个输入索引。
 * @return 仿射输出索引；容量或数值失败时为 SIZE_MAX。
 */
static size_t graph_affine2(Graph *graph, size_t w0, size_t w1, size_t bias,
                            size_t x0, size_t x1)
{
    size_t first = graph_mul(graph, w0, x0);
    size_t second = graph_mul(graph, w1, x1);
    size_t sum = graph_add(graph, first, second);
    return graph_add(graph, sum, bias);
}

/**
 * @brief 为一个 XOR 样本建图，记录参数叶、预测节点和半平方误差节点。
 * @param graph 可写图；调用前可含旧节点，本函数先重置。
 * @param network 有限只读参数，布局为两组隐藏权重/偏置与输出权重/偏置。
 * @param x0 第一输入，须为 0 或 1。
 * @param x1 第二输入，须为 0 或 1。
 * @param target 目标，须为 0 或 1。
 * @param parameter_nodes 可写长度 PARAM_COUNT 的节点索引数组，仅成功时覆盖。
 * @param prediction 可写预测节点索引，仅成功时覆盖。
 * @param loss 可写损失节点索引，仅成功时覆盖。
 * @return 构造成功为 true，容量、数值或非法输入时为 false。
 */
static bool graph_xor(Graph *graph, const Network *network, double x0, double x1,
                      double target, size_t parameter_nodes[PARAM_COUNT], size_t *prediction,
                      size_t *loss)
{
    if (graph == nullptr || network == nullptr || parameter_nodes == nullptr ||
        prediction == nullptr || loss == nullptr || (x0 != 0 && x0 != 1) ||
        (x1 != 0 && x1 != 1) || (target != 0 && target != 1)) {
        return false;
    }
    graph_reset(graph);
    size_t leaves[PARAM_COUNT];
    for (size_t p = 0; p < PARAM_COUNT; ++p) {
        leaves[p] = graph_leaf(graph, network->values[p]);
        if (leaves[p] == SIZE_MAX) {
            return false;
        }
    }
    size_t input0 = graph_leaf(graph, x0);
    size_t input1 = graph_leaf(graph, x1);
    size_t target_node = graph_leaf(graph, target);
    size_t minus_one = graph_leaf(graph, -1);
    size_t half = graph_leaf(graph, 0.5);
    size_t hidden1_pre = graph_affine2(graph, leaves[0], leaves[1], leaves[2], input0, input1);
    size_t hidden2_pre = graph_affine2(graph, leaves[3], leaves[4], leaves[5], input0, input1);
    size_t hidden1 = graph_unary(graph, OP_TANH, hidden1_pre);
    size_t hidden2 = graph_unary(graph, OP_TANH, hidden2_pre);
    size_t logit = graph_affine2(graph, leaves[6], leaves[7], leaves[8], hidden1, hidden2);
    size_t probability = graph_unary(graph, OP_SIGMOID, logit);
    size_t negative_target = graph_mul(graph, target_node, minus_one);
    size_t error = graph_add(graph, probability, negative_target);
    size_t squared = graph_mul(graph, error, error);
    size_t sample_loss = graph_mul(graph, half, squared);
    if (sample_loss == SIZE_MAX) {
        return false;
    }
    memcpy(parameter_nodes, leaves, sizeof(leaves));
    *prediction = probability;
    *loss = sample_loss;
    return true;
}

/**
 * @brief 求一个样本的自动微分结果，不修改模型。
 * @param network 有限只读参数。
 * @param x0 第一输入 0 或 1。
 * @param x1 第二输入 0 或 1。
 * @param target 目标 0 或 1。
 * @param result 可写结果；失败时不修改。
 * @return 构图和反向成功为 true。
 */
static bool xor_sample(const Network *network, double x0, double x1, double target,
                       SampleResult *result)
{
    if (result == nullptr) {
        return false;
    }
    Graph graph = {.capacity = GRAPH_CAPACITY};
    size_t parameter_nodes[PARAM_COUNT] = {0};
    size_t prediction = SIZE_MAX;
    size_t loss = SIZE_MAX;
    if (!graph_xor(&graph, network, x0, x1, target, parameter_nodes, &prediction, &loss) ||
        !graph_backward(&graph, loss)) {
        return false;
    }
    SampleResult candidate = {.prediction = graph.nodes[prediction].value,
                              .loss = graph.nodes[loss].value};
    for (size_t p = 0; p < PARAM_COUNT; ++p) {
        candidate.gradients[p] = graph.nodes[parameter_nodes[p]].gradient;
    }
    *result = candidate;
    return true;
}

/**
 * @brief 手写相同 2→2→1 网络和同一损失的梯度，用于独立对照。
 * @param network 有限只读参数。
 * @param x0 第一输入 0 或 1。
 * @param x1 第二输入 0 或 1。
 * @param target 目标 0 或 1。
 * @param result 可写结果，输入均有效时覆盖。
 */
static void xor_manual(const Network *network, double x0, double x1, double target,
                       SampleResult *result)
{
    const double *p = network->values;
    double h1 = tanh(p[0] * x0 + p[1] * x1 + p[2]);
    double h2 = tanh(p[3] * x0 + p[4] * x1 + p[5]);
    double logit = p[6] * h1 + p[7] * h2 + p[8];
    double prediction = 1.0 / (1.0 + exp(-logit));
    double error = prediction - target;
    double output_delta = error * prediction * (1.0 - prediction);
    double hidden1_delta = output_delta * p[6] * (1.0 - h1 * h1);
    double hidden2_delta = output_delta * p[7] * (1.0 - h2 * h2);
    *result = (SampleResult){
        .prediction = prediction,
        .loss = 0.5 * error * error,
        .gradients = {hidden1_delta * x0, hidden1_delta * x1, hidden1_delta,
                      hidden2_delta * x0, hidden2_delta * x1, hidden2_delta,
                      output_delta * h1, output_delta * h2, output_delta}
    };
}

/**
 * @brief 使用固定输入四样本执行一轮全批量训练，不依赖随机数。
 * @param network 可写参数，成功后更新；非法数值时保持旧参数。
 * @param rate 有限正学习率。
 * @param mean_loss 可写旧参数下的平均损失，失败时保持原值。
 * @return 四个样本和参数更新均成功为 true。
 */
static bool train_epoch(Network *network, double rate, double *mean_loss)
{
    if (network == nullptr || mean_loss == nullptr || !isfinite(rate) || rate <= 0) {
        return false;
    }
    double gradients[PARAM_COUNT] = {0};
    double loss = 0;
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        SampleResult result = {0};
        if (!xor_sample(network, XOR_INPUTS[sample][0], XOR_INPUTS[sample][1],
                        XOR_TARGETS[sample], &result)) {
            return false;
        }
        loss += result.loss / (double)SAMPLE_COUNT;
        for (size_t p = 0; p < PARAM_COUNT; ++p) {
            gradients[p] += result.gradients[p] / (double)SAMPLE_COUNT;
        }
    }
    Network candidate = *network;
    for (size_t p = 0; p < PARAM_COUNT; ++p) {
        candidate.values[p] -= rate * gradients[p];
        if (!isfinite(candidate.values[p])) {
            return false;
        }
    }
    if (!isfinite(loss)) {
        return false;
    }
    *network = candidate;
    *mean_loss = loss;
    return true;
}

/**
 * @brief 返回四条样本在当前参数下的平均半平方损失。
 * @param network 有限只读参数。
 * @param loss 可写结果，失败时保持原值。
 * @return 成功为 true。
 */
static bool xor_mean_loss(const Network *network, double *loss)
{
    if (loss == nullptr) {
        return false;
    }
    double sum = 0;
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        SampleResult result = {0};
        if (!xor_sample(network, XOR_INPUTS[sample][0], XOR_INPUTS[sample][1],
                        XOR_TARGETS[sample], &result)) {
            return false;
        }
        sum += result.loss / (double)SAMPLE_COUNT;
    }
    *loss = sum;
    return true;
}

/** @brief 验证共享节点、容量/数值失败、重复反向和 XOR 梯度。 */
static void test_autodiff(void)
{
    Graph graph = {.capacity = GRAPH_CAPACITY};
    size_t x = graph_leaf(&graph, 2);
    size_t y = graph_leaf(&graph, 3);
    size_t square = graph_mul(&graph, x, x);
    size_t product = graph_mul(&graph, x, y);
    size_t exponential = graph_unary(&graph, OP_EXP, x);
    size_t sum = graph_add(&graph, square, product);
    size_t output = graph_add(&graph, sum, exponential);
    assert(graph_backward(&graph, output));
    double expected_x = 7.0 + exp(2.0);
    assert(fabs(graph.nodes[x].gradient - expected_x) < 1e-12);
    assert(graph.nodes[y].gradient == 2.0);
    assert(graph_backward(&graph, output));
    assert(fabs(graph.nodes[x].gradient - expected_x) < 1e-12);
    assert(graph.nodes[y].gradient == 2.0);
    double step = 1e-5;
    double plus = (2 + step) * (2 + step) + (2 + step) * 3 + exp(2 + step);
    double minus = (2 - step) * (2 - step) + (2 - step) * 3 + exp(2 - step);
    assert(fabs(graph.nodes[x].gradient - (plus - minus) / (2 * step)) < 1e-8);

    size_t logged = graph_unary(&graph, OP_LOG, exponential);
    assert(graph_backward(&graph, logged));
    assert(fabs(graph.nodes[x].gradient - 1.0) < 1e-12);
    size_t activated = graph_unary(&graph, OP_TANH, x);
    assert(graph_backward(&graph, activated));
    assert(fabs(graph.nodes[x].gradient - (1 - tanh(2.0) * tanh(2.0))) < 1e-12);
    size_t sigmoid_node = graph_unary(&graph, OP_SIGMOID, x);
    assert(graph_backward(&graph, sigmoid_node));
    double probability = graph.nodes[sigmoid_node].value;
    assert(fabs(graph.nodes[x].gradient - probability * (1 - probability)) < 1e-12);

    size_t before = graph.count;
    assert(graph_unary(&graph, OP_LOG, graph_leaf(&graph, -1)) == SIZE_MAX);
    assert(graph.count == before + 1); /* 只有显式加入的叶节点保留。 */
    before = graph.count;
    assert(graph_unary(&graph, OP_EXP, graph_leaf(&graph, 1000)) == SIZE_MAX);
    assert(graph.count == before + 1);
    assert(graph_leaf(&graph, NAN) == SIZE_MAX);
    assert(graph_mul(&graph, SIZE_MAX, x) == SIZE_MAX);
    assert(graph_backward(&graph, SIZE_MAX) == false);
    Graph limited = {.capacity = 2};
    assert(graph_leaf(&limited, 1) != SIZE_MAX);
    assert(graph_leaf(&limited, 2) != SIZE_MAX);
    assert(graph_add(&limited, 0, 1) == SIZE_MAX);
    assert(limited.count == 2);
    graph_reset(&limited);
    assert(limited.count == 0);
    assert(graph_leaf(&limited, 3) == 0);

    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        SampleResult automatic = {0};
        SampleResult manual = {0};
        assert(xor_sample(&INITIAL_NETWORK, XOR_INPUTS[sample][0], XOR_INPUTS[sample][1],
                          XOR_TARGETS[sample], &automatic));
        xor_manual(&INITIAL_NETWORK, XOR_INPUTS[sample][0], XOR_INPUTS[sample][1],
                   XOR_TARGETS[sample], &manual);
        assert(fabs(automatic.prediction - manual.prediction) < 1e-12);
        assert(fabs(automatic.loss - manual.loss) < 1e-12);
        for (size_t p = 0; p < PARAM_COUNT; ++p) {
            assert(fabs(automatic.gradients[p] - manual.gradients[p]) < 1e-12);
            Network high = INITIAL_NETWORK;
            Network low = INITIAL_NETWORK;
            high.values[p] += step;
            low.values[p] -= step;
            SampleResult high_result = {0};
            SampleResult low_result = {0};
            assert(xor_sample(&high, XOR_INPUTS[sample][0], XOR_INPUTS[sample][1],
                              XOR_TARGETS[sample], &high_result));
            assert(xor_sample(&low, XOR_INPUTS[sample][0], XOR_INPUTS[sample][1],
                              XOR_TARGETS[sample], &low_result));
            double numerical = (high_result.loss - low_result.loss) / (2 * step);
            assert(fabs(automatic.gradients[p] - numerical) < 1e-8 +
                   1e-6 * fmax(fabs(automatic.gradients[p]), fabs(numerical)));
        }
    }
    SampleResult sentinel = {.prediction = 99};
    Network invalid = INITIAL_NETWORK;
    invalid.values[0] = NAN;
    assert(!xor_sample(&invalid, 0, 1, 1, &sentinel));
    assert(sentinel.prediction == 99);
    double loss_sentinel = 99;
    assert(!train_epoch(&invalid, 0.5, &loss_sentinel));
    assert(loss_sentinel == 99);
}

/**
 * @brief 展示计算图求导和 800 次固定数据训练结果。
 * @param csv 为 true 时输出 field,value，供 C/Python 比较。
 * @return 成功为 EXIT_SUCCESS，否则为 EXIT_FAILURE。
 */
static int print_demo(bool csv)
{
    Network network = INITIAL_NETWORK;
    double initial = 0;
    double old_loss = 0;
    double final = 0;
    if (!xor_mean_loss(&network, &initial)) {
        return EXIT_FAILURE;
    }
    for (size_t epoch = 0; epoch < TRAIN_EPOCHS; ++epoch) {
        if (!train_epoch(&network, 0.5, &old_loss)) {
            return EXIT_FAILURE;
        }
    }
    if (!xor_mean_loss(&network, &final)) {
        return EXIT_FAILURE;
    }
    if (csv) {
        puts("field,value");
        printf("loss_initial,%.17g\nloss_final,%.17g\n", initial, final);
    } else {
        printf("XOR 初始平均损失 %.8f，%d 轮后 %.8f\n", initial, TRAIN_EPOCHS, final);
    }
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        SampleResult result = {0};
        if (!xor_sample(&network, XOR_INPUTS[sample][0], XOR_INPUTS[sample][1],
                        XOR_TARGETS[sample], &result)) {
            return EXIT_FAILURE;
        }
        if (csv) {
            printf("prediction_%zu,%.17g\n", sample, result.prediction);
        } else {
            printf("(%.0f, %.0f) -> %.6f，目标 %.0f\n", XOR_INPUTS[sample][0],
                   XOR_INPUTS[sample][1], result.prediction, XOR_TARGETS[sample]);
        }
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行标量自动求导教学案例。
 * @param argc 参数数量，可传 --test 或 --csv。
 * @param argv 参数数组。
 * @return 成功为 EXIT_SUCCESS；非法参数或数值失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_autodiff();
        puts("标量自动求导测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1) {
        return print_demo(false);
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        return print_demo(true);
    }
    fputs("用法：32_scalar_autodiff [--test | --csv]\n", stderr);
    return EXIT_FAILURE;
}
