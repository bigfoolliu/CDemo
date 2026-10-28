/**
 * @file
 * @brief M08：固定容量张量计算图，复现 M02 的批量全连接层。
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_ELEMENTS = 6, GRAPH_CAPACITY = 32 };

typedef enum { LEAF, ADD, MUL, MATMUL, SUM, MEAN, RELU } Operation;

typedef struct {
    size_t rows;
    size_t cols;
    double values[MAX_ELEMENTS];
    double gradients[MAX_ELEMENTS];
    Operation operation;
    size_t left;
    size_t right;
    bool broadcast_right;
} TensorNode;

typedef struct {
    TensorNode nodes[GRAPH_CAPACITY];
    size_t count;
    size_t capacity;
} TensorGraph;

typedef struct {
    double loss;
    double predictions[4];
    double input_gradients[4];
    double weight_gradients[4];
    double bias_gradients[2];
    double updated_weights[4];
    double updated_bias[2];
} DenseResult;

static const double INPUTS[4] = {1, 2, -1, 1};
static const double WEIGHTS[4] = {1, -1, 2, 0.5};
static const double BIAS[2] = {0.5, -0.5};

/**
 * @brief 释放固定容量图的逻辑节点；旧索引失效，无堆内存。
 * @param graph 可写图；空指针时无操作。
 */
static void graph_release(TensorGraph *graph)
{
    if (graph != nullptr) {
        memset(graph->nodes, 0, sizeof(graph->nodes));
        graph->count = 0;
    }
}

/**
 * @brief 检查小张量形状及可用图容量。
 * @param graph 可写图。
 * @param rows 行数；向量使用一行。
 * @param cols 列数；标量使用 1×1。
 * @return 可添加节点时为 true。
 */
static bool can_push(const TensorGraph *graph, size_t rows, size_t cols)
{
    return graph != nullptr && graph->capacity > 0 &&
           graph->capacity <= GRAPH_CAPACITY && graph->count < graph->capacity &&
           rows > 0 && cols > 0 && rows <= MAX_ELEMENTS / cols;
}

/**
 * @brief 复制有限数组为叶张量；失败时图不变。
 * @param graph 可写图，不转移所有权。
 * @param rows 行数。
 * @param cols 列数。
 * @param values 至少有 rows×cols 项的只读数组。
 * @return 成功为节点索引，失败为 SIZE_MAX。
 */
static size_t graph_leaf(TensorGraph *graph, size_t rows, size_t cols, const double *values)
{
    if (values == nullptr || !can_push(graph, rows, cols)) {
        return SIZE_MAX;
    }
    TensorNode next = {.rows = rows, .cols = cols, .operation = LEAF};
    for (size_t i = 0; i < rows * cols; ++i) {
        if (!isfinite(values[i])) {
            return SIZE_MAX;
        }
        next.values[i] = values[i];
    }
    size_t index = graph->count++;
    graph->nodes[index] = next;
    return index;
}

/**
 * @brief 添加逐元素加法、乘法或矩阵乘法节点。
 * @param graph 可写图。
 * @param operation ADD、MUL 或 MATMUL。
 * @param left 已有左节点索引。
 * @param right 已有右节点索引；ADD 支持一行偏置广播。
 * @return 成功为新索引，形状、容量或数值无效时为 SIZE_MAX 且图不变。
 */
static size_t graph_binary(TensorGraph *graph, Operation operation, size_t left, size_t right)
{
    if (graph == nullptr || left >= graph->count || right >= graph->count ||
        (operation != ADD && operation != MUL && operation != MATMUL)) {
        return SIZE_MAX;
    }
    const TensorNode *a = &graph->nodes[left];
    const TensorNode *b = &graph->nodes[right];
    bool broadcast = operation == ADD && a->rows > 1 && b->rows == 1 && a->cols == b->cols;
    size_t rows = a->rows;
    size_t cols = operation == MATMUL ? b->cols : a->cols;
    if (!can_push(graph, rows, cols)) {
        return SIZE_MAX;
    }
    if (operation == MATMUL ? a->cols != b->rows :
        !(a->rows == b->rows && a->cols == b->cols) && !broadcast) {
        return SIZE_MAX;
    }
    TensorNode next = {.rows = rows, .cols = cols, .operation = operation,
                       .left = left, .right = right, .broadcast_right = broadcast};
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            size_t index = i * cols + j;
            if (operation == MATMUL) {
                for (size_t k = 0; k < a->cols; ++k) {
                    next.values[index] += a->values[i * a->cols + k] *
                                          b->values[k * b->cols + j];
                }
            } else {
                size_t other = broadcast ? j : index;
                next.values[index] = operation == ADD ? a->values[index] + b->values[other] :
                                     a->values[index] * b->values[other];
            }
            if (!isfinite(next.values[index])) {
                return SIZE_MAX;
            }
        }
    }
    size_t output = graph->count++;
    graph->nodes[output] = next;
    return output;
}

/**
 * @brief 添加求和、均值或 ReLU 节点；零点导数约定为零。
 * @param graph 可写图。
 * @param operation SUM、MEAN 或 RELU。
 * @param input 已有输入节点索引。
 * @return 成功为新索引，失败为 SIZE_MAX 且图不变。
 */
static size_t graph_unary(TensorGraph *graph, Operation operation, size_t input)
{
    if (graph == nullptr || input >= graph->count ||
        (operation != SUM && operation != MEAN && operation != RELU)) {
        return SIZE_MAX;
    }
    const TensorNode *source = &graph->nodes[input];
    size_t rows = operation == RELU ? source->rows : 1;
    size_t cols = operation == RELU ? source->cols : 1;
    if (!can_push(graph, rows, cols)) {
        return SIZE_MAX;
    }
    TensorNode next = {.rows = rows, .cols = cols, .operation = operation, .left = input};
    for (size_t i = 0; i < source->rows * source->cols; ++i) {
        if (operation == RELU) {
            next.values[i] = fmax(0, source->values[i]);
        } else {
            next.values[0] += source->values[i];
        }
    }
    if (operation == MEAN) {
        next.values[0] /= (double)(source->rows * source->cols);
    }
    for (size_t i = 0; i < rows * cols; ++i) {
        if (!isfinite(next.values[i])) {
            return SIZE_MAX;
        }
    }
    size_t output = graph->count++;
    graph->nodes[output] = next;
    return output;
}

/**
 * @brief 每次清零后逆序传播，所有父节点梯度都用加法累积。
 * @param graph 有效可写图；节点只引用较早索引。
 * @param output 1×1 标量输出索引。
 * @return 梯度全有限时为 true；失败时梯度清零。
 */
static bool graph_backward(TensorGraph *graph, size_t output)
{
    if (graph == nullptr || output >= graph->count ||
        graph->nodes[output].rows != 1 || graph->nodes[output].cols != 1) {
        return false;
    }
    for (size_t n = 0; n < graph->count; ++n) {
        memset(graph->nodes[n].gradients, 0, sizeof(graph->nodes[n].gradients));
    }
    graph->nodes[output].gradients[0] = 1;
    for (size_t remaining = output + 1; remaining > 0; --remaining) {
        TensorNode *node = &graph->nodes[remaining - 1];
        if (node->operation == LEAF) {
            continue;
        }
        TensorNode *a = &graph->nodes[node->left];
        TensorNode *b = &graph->nodes[node->right];
        for (size_t i = 0; i < node->rows; ++i) {
            for (size_t j = 0; j < node->cols; ++j) {
                size_t index = i * node->cols + j;
                double upstream = node->gradients[index];
                if (node->operation == ADD || node->operation == MUL) {
                    size_t other = node->broadcast_right ? j : index;
                    a->gradients[index] += upstream *
                        (node->operation == MUL ? b->values[other] : 1);
                    b->gradients[other] += upstream *
                        (node->operation == MUL ? a->values[index] : 1);
                } else if (node->operation == MATMUL) {
                    for (size_t k = 0; k < a->cols; ++k) {
                        a->gradients[i * a->cols + k] += upstream * b->values[k * b->cols + j];
                        b->gradients[k * b->cols + j] += upstream * a->values[i * a->cols + k];
                    }
                }
            }
        }
        if (node->operation == SUM || node->operation == MEAN || node->operation == RELU) {
            size_t count = a->rows * a->cols;
            for (size_t i = 0; i < count; ++i) {
                double multiplier = node->operation == MEAN ? 1.0 / (double)count : 1;
                if (node->operation == RELU) {
                    multiplier = a->values[i] > 0 ? 1 : 0;
                    a->gradients[i] += node->gradients[i] * multiplier;
                } else {
                    a->gradients[i] += node->gradients[0] * multiplier;
                }
            }
        }
    }
    for (size_t n = 0; n < graph->count; ++n) {
        size_t count = graph->nodes[n].rows * graph->nodes[n].cols;
        for (size_t i = 0; i < count; ++i) {
            if (!isfinite(graph->nodes[n].gradients[i])) {
                for (size_t k = 0; k < graph->count; ++k) {
                    memset(graph->nodes[k].gradients, 0, sizeof(graph->nodes[k].gradients));
                }
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief 用张量图复现 M02 的前向、平均半平方损失与全部梯度。
 * @param inputs 行优先 2×2 输入。
 * @param weights 行优先 2×2 权重。
 * @param bias 长度为 2 的偏置行向量。
 * @param result 可写结果；失败时保持原值。
 * @return 图构建与反向均成功时为 true。
 */
static bool dense_run(const double inputs[4], const double weights[4],
                      const double bias[2], DenseResult *result)
{
    if (inputs == nullptr || weights == nullptr || bias == nullptr || result == nullptr) {
        return false;
    }
    TensorGraph graph = {.capacity = GRAPH_CAPACITY};
    const double negative_targets[4] = {-2, 0, 0, -1};
    const double half = 0.5;
    size_t x = graph_leaf(&graph, 2, 2, inputs);
    size_t w = graph_leaf(&graph, 2, 2, weights);
    size_t b = graph_leaf(&graph, 1, 2, bias);
    size_t t = graph_leaf(&graph, 2, 2, negative_targets);
    size_t scale = graph_leaf(&graph, 1, 1, &half);
    size_t product = graph_binary(&graph, MATMUL, x, w);
    size_t prediction = graph_binary(&graph, ADD, product, b);
    size_t error = graph_binary(&graph, ADD, prediction, t);
    size_t squared = graph_binary(&graph, MUL, error, error);
    size_t mean = graph_unary(&graph, MEAN, squared);
    size_t loss = graph_binary(&graph, MUL, mean, scale);
    if (loss == SIZE_MAX || !graph_backward(&graph, loss)) {
        graph_release(&graph);
        return false;
    }
    DenseResult next = {.loss = graph.nodes[loss].values[0]};
    for (size_t i = 0; i < 4; ++i) {
        next.predictions[i] = graph.nodes[prediction].values[i];
        next.input_gradients[i] = graph.nodes[x].gradients[i];
        next.weight_gradients[i] = graph.nodes[w].gradients[i];
        next.updated_weights[i] = weights[i] - 0.1 * next.weight_gradients[i];
    }
    for (size_t i = 0; i < 2; ++i) {
        next.bias_gradients[i] = graph.nodes[b].gradients[i];
        next.updated_bias[i] = bias[i] - 0.1 * next.bias_gradients[i];
    }
    *result = next;
    graph_release(&graph);
    return true;
}

/** @brief 检查 M02 手算、十个中心差分、共享节点、广播和图生命周期。 */
static void test_tensor_graph(void)
{
    DenseResult result = {0};
    assert(dense_run(INPUTS, WEIGHTS, BIAS, &result));
    assert(fabs(result.loss - 1.84375) < 1e-12);
    const double expected_prediction[4] = {5.5, -0.5, 1.5, 1};
    const double expected_x[4] = {1, 1.6875, 0.375, 0.75};
    const double expected_w[4] = {0.5, -0.125, 2.125, -0.25};
    const double expected_b[2] = {1.25, -0.125};
    const double expected_updated_w[4] = {0.95, -0.9875, 1.7875, 0.525};
    const double expected_updated_b[2] = {0.375, -0.4875};
    for (size_t i = 0; i < 4; ++i) {
        assert(fabs(result.predictions[i] - expected_prediction[i]) < 1e-12);
        assert(fabs(result.input_gradients[i] - expected_x[i]) < 1e-12);
        assert(fabs(result.weight_gradients[i] - expected_w[i]) < 1e-12);
        assert(fabs(result.updated_weights[i] - expected_updated_w[i]) < 1e-12);
    }
    for (size_t i = 0; i < 2; ++i) {
        assert(fabs(result.bias_gradients[i] - expected_b[i]) < 1e-12);
        assert(fabs(result.updated_bias[i] - expected_updated_b[i]) < 1e-12);
    }
    const double step = 1e-5;
    for (size_t parameter = 0; parameter < 10; ++parameter) {
        double x[4] = {0};
        double w[4] = {0};
        double b[2] = {0};
        memcpy(x, INPUTS, sizeof(x));
        memcpy(w, WEIGHTS, sizeof(w));
        memcpy(b, BIAS, sizeof(b));
        double *item = nullptr;
        if (parameter < 4) {
            item = &x[parameter];
        } else if (parameter < 8) {
            item = &w[parameter - 4];
        } else {
            item = &b[parameter - 8];
        }
        *item += step;
        DenseResult plus = {0};
        assert(dense_run(x, w, b, &plus));
        *item -= 2 * step;
        DenseResult minus = {0};
        assert(dense_run(x, w, b, &minus));
        double numerical = (plus.loss - minus.loss) / (2 * step);
        double analytic = 0;
        if (parameter < 4) {
            analytic = result.input_gradients[parameter];
        } else if (parameter < 8) {
            analytic = result.weight_gradients[parameter - 4];
        } else {
            analytic = result.bias_gradients[parameter - 8];
        }
        assert(fabs(numerical - analytic) < 1e-8);
    }

    TensorGraph graph = {.capacity = 8};
    const double sample[2] = {-1, 2};
    size_t leaf = graph_leaf(&graph, 1, 2, sample);
    size_t doubled = graph_binary(&graph, ADD, leaf, leaf);
    size_t active = graph_unary(&graph, RELU, doubled);
    size_t total = graph_unary(&graph, SUM, active);
    assert(graph_backward(&graph, total));
    assert(graph.nodes[leaf].gradients[0] == 0);
    assert(graph.nodes[leaf].gradients[1] == 2);
    assert(graph_backward(&graph, total));
    assert(graph.nodes[leaf].gradients[1] == 2);
    size_t count = graph.count;
    assert(graph_binary(&graph, MATMUL, leaf, leaf) == SIZE_MAX);
    assert(graph.count == count);
    graph.capacity = graph.count;
    assert(graph_leaf(&graph, 1, 1, sample) == SIZE_MAX);
    graph_release(&graph);
    assert(graph.count == 0);
    assert(!graph_backward(&graph, total));
    graph.capacity = 8;
    const double invalid = NAN;
    assert(graph_leaf(&graph, 1, 1, &invalid) == SIZE_MAX);
    assert(graph.count == 0);
    const double invalid_inputs[4] = {NAN, 2, -1, 1};
    DenseResult untouched = {.loss = 99};
    assert(!dense_run(invalid_inputs, WEIGHTS, BIAS, &untouched));
    assert(untouched.loss == 99);
}

/**
 * @brief 输出 M02 固定案例的张量图结果。
 * @param csv 为 true 时以 field,value 输出供跨语言比较。
 * @return 成功为 EXIT_SUCCESS，否则为 EXIT_FAILURE。
 */
static int print_demo(bool csv)
{
    DenseResult result = {0};
    if (!dense_run(INPUTS, WEIGHTS, BIAS, &result)) {
        return EXIT_FAILURE;
    }
    if (csv) {
        puts("field,value");
        printf("loss,%.17g\n", result.loss);
        for (size_t i = 0; i < 4; ++i) {
            printf("prediction_%zu,%.17g\n", i, result.predictions[i]);
            printf("input_grad_%zu,%.17g\n", i, result.input_gradients[i]);
            printf("weight_grad_%zu,%.17g\n", i, result.weight_gradients[i]);
            printf("updated_weight_%zu,%.17g\n", i, result.updated_weights[i]);
        }
        for (size_t i = 0; i < 2; ++i) {
            printf("bias_grad_%zu,%.17g\n", i, result.bias_gradients[i]);
            printf("updated_bias_%zu,%.17g\n", i, result.updated_bias[i]);
        }
    } else {
        printf("张量图复现 M02：损失=%.5f，预测=[%.2f, %.2f, %.2f, %.2f]\n",
               result.loss, result.predictions[0], result.predictions[1],
               result.predictions[2], result.predictions[3]);
        printf("dW=[%.3f, %.3f, %.3f, %.3f]，db=[%.3f, %.3f]\n",
               result.weight_gradients[0], result.weight_gradients[1],
               result.weight_gradients[2], result.weight_gradients[3],
               result.bias_gradients[0], result.bias_gradients[1]);
        printf("学习率 0.1 更新后：W=[%.4f, %.4f, %.4f, %.4f]，b=[%.4f, %.4f]\n",
               result.updated_weights[0], result.updated_weights[1],
               result.updated_weights[2], result.updated_weights[3],
               result.updated_bias[0], result.updated_bias[1]);
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行 M08 教学实验。
 * @param argc 参数个数，可选 --test 或 --csv。
 * @param argv 参数数组。
 * @return 成功为 EXIT_SUCCESS，否则为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_tensor_graph();
        puts("张量自动求导测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "--csv") == 0)) {
        return print_demo(argc == 2);
    }
    fputs("用法：37_tensor_autodiff [--test | --csv]\n", stderr);
    return EXIT_FAILURE;
}
