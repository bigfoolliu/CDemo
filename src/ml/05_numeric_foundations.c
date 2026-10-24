/**
 * @file
 * @brief 深度学习阶段 0：行优先矩阵、链式法则和一次梯度下降。
 * 数值范围限定为不会使 double 运算溢出的有限小数；不实现通用张量框架。
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    size_t rows;
    size_t columns;
    size_t capacity; /* data 可访问的 double 元素数，不是字节数。 */
    double *data;    /* 独占拥有；禁止复制结构体后分别释放。 */
} Matrix;  // 行优先矩阵描述符，元素按行存储。

/* 生产运行传 calloc；测试传失败分配器，确定性地验证分配失败。 */
typedef void *(*Allocate)(size_t count, size_t size);  // 分配器函数指针类型，返回可 free 释放的内存或 nullptr。

typedef struct {
    double weights[2];
    double bias;
} LinearModel;  // 线性模型，预测值 = w0 * x0 + w1 * x1 + b。

typedef struct {
    double prediction;
    double error;
    double loss;
    double gradients[3]; /* 顺序为 w0、w1、b，均来自同一次前向。 */
} LinearStep;  // 单样本前向计算结果与梯度，梯度来自同一次前向。

/**
 * @brief 检查非零矩阵形状及其存储容量，不读取元素。
 * @param matrix 只读描述符；data 必须指向至少 capacity 个 double。
 * @return 元素数与字节数无溢出且容量足够时为 true。
 */
static bool matrix_valid(const Matrix *matrix)
{
    if (matrix == nullptr || matrix->data == nullptr || matrix->rows == 0 || matrix->columns == 0 ||
        matrix->rows > SIZE_MAX / matrix->columns) {
        return false;
    }
    size_t count = matrix->rows * matrix->columns;
    return count <= SIZE_MAX / sizeof(double) && count <= matrix->capacity;
}

/**
 * @brief 创建行优先矩阵，显式初始化 double 为 0.0。
 * @param matrix 可写、零初始化的空描述符；成功后拥有内存，须调用 matrix_destroy。
 * @param rows 行数，必须大于零。
 * @param columns 列数，必须大于零。
 * @param allocate calloc 或返回可用 free 释放内存的分配器；不得为空。
 * @return 成功为 true；非法形状、非空描述符或分配失败时保持描述符不变。
 */
static bool matrix_create(Matrix *matrix, size_t rows, size_t columns, Allocate allocate)
{
    if (matrix == nullptr || matrix->data != nullptr || matrix->rows != 0 || matrix->columns != 0 ||
        matrix->capacity != 0 || allocate == nullptr || rows == 0 || columns == 0 ||
        rows > SIZE_MAX / columns) {
        return false;
    }
    size_t count = rows * columns;
    if (count > SIZE_MAX / sizeof(double)) {
        return false;
    }
    double *data = allocate(count, sizeof(double));
    if (data == nullptr) {
        return false;
    }
    /* 不把 calloc 的全零字节直接假定为所有 C 实现的浮点零表示。 */
    for (size_t i = 0; i < count; ++i) {
        data[i] = 0.0;
    }
    *matrix = (Matrix){rows, columns, count, data};

    return true;
}

/**
 * @brief 释放拥有的矩阵存储并恢复为空状态。
 * @param matrix 可写拥有者或空指针；只允许释放 matrix_create 创建的内存。
 */
static void matrix_destroy(Matrix *matrix)
{
    if (matrix != nullptr) {
        free(matrix->data);
        *matrix = (Matrix){0};
    }
}

/**
 * @brief 计算 C(m,n) = A(m,k) × B(k,n)，展示三层循环与点积。
 * @param a 只读左矩阵，元素及中间运算须有限且不溢出。
 * @param b 只读右矩阵，与 a 可为同一矩阵。
 * @param output 已分配的可写结果，与输入存储不得重叠；不转移所有权。
 * @return 描述符、形状或同一 data 指针检查失败返回 false，不修改结果。
 */
static bool matrix_multiply(const Matrix *a, const Matrix *b, Matrix *output)
{
    if (!matrix_valid(a) || !matrix_valid(b) || !matrix_valid(output) || a->columns != b->rows ||
        output->rows != a->rows || output->columns != b->columns || output->data == a->data ||
        output->data == b->data) {
        return false;
    }
    for (size_t row = 0; row < output->rows; ++row) {
        for (size_t column = 0; column < output->columns; ++column) {
            double sum = 0;
            for (size_t k = 0; k < a->columns; ++k) {
                /* 行优先：第 r 行第 c 列位于 data[r * columns + c]。 */
                sum += a->data[row * a->columns + k] * b->data[k * b->columns + column];
            }
            output->data[row * output->columns + column] = sum;
        }
    }
    return true;
}

/**
 * @brief 逐元素相加，区别于矩阵乘法。
 * @param a 只读矩阵；元素及求和须有限且不溢出。
 * @param b 与 a 形状相同的只读矩阵。
 * @param output 同形状可写矩阵；允许与输入完全相同，但禁止部分重叠。
 * @return 形状或容量检查失败返回 false，保持结果不变。
 */
static bool matrix_add(const Matrix *a, const Matrix *b, Matrix *output)
{
    if (!matrix_valid(a) || !matrix_valid(b) || !matrix_valid(output) || a->rows != b->rows ||
        a->columns != b->columns || output->rows != a->rows || output->columns != a->columns) {
        return false;
    }
    for (size_t i = 0; i < a->rows * a->columns; ++i) {
        output->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

/**
 * @brief 转置 A(m,n) 为 A^T(n,m)，不改变输入。
 * @param input 只读矩阵，不转移所有权。
 * @param output 已分配且存储不与 input 重叠的可写矩阵。
 * @return 形状、容量或相同 data 指针检查失败返回 false，结果不变。
 */
static bool matrix_transpose(const Matrix *input, Matrix *output)
{
    if (!matrix_valid(input) || !matrix_valid(output) || output->rows != input->columns ||
        output->columns != input->rows || output->data == input->data) {
        return false;
    }
    for (size_t row = 0; row < input->rows; ++row) {
        for (size_t column = 0; column < input->columns; ++column) {
            output->data[column * output->columns + row] =
                input->data[row * input->columns + column];
        }
    }
    return true;
}

/**
 * @brief 打印矩阵形状与元素，便于与手算结果对照。
 * @param name 只读、以零结尾的名称。
 * @param matrix 已验证有效的只读矩阵。
 */
static void matrix_print(const char *name, const Matrix *matrix)
{
    printf("%s (%zu × %zu)\n", name, matrix->rows, matrix->columns);
    for (size_t row = 0; row < matrix->rows; ++row) {
        for (size_t column = 0; column < matrix->columns; ++column) {
            printf(" %7.2f", matrix->data[row * matrix->columns + column]);
        }
        putchar('\n');
    }
}

/**
 * @brief 保存一个样本的前向结果并通过链式法则计算梯度，不修改模型。
 * @param model 有效的只读模型；参数和计算须有限且不溢出。
 * @param x 包含两个有限特征的只读数组，长度必须为 2。
 * @param target 有限目标值；单样本 MSE 为 (prediction-target)^2，不乘 1/2。
 * @return 预测、误差、损失与同一组旧参数下的三个梯度。
 */
static LinearStep linear_step(const LinearModel *model, const double x[2], double target)
{
    LinearStep step = {0};
    step.prediction = model->weights[0] * x[0] + model->weights[1] * x[1] + model->bias;
    step.error = step.prediction - target;
    step.loss = step.error * step.error;

    /* dL/dprediction = 2*error；dprediction/dwi = xi；dprediction/db = 1。 */
    double upstream = 2 * step.error;
    step.gradients[0] = upstream * x[0];
    step.gradients[1] = upstream * x[1];
    step.gradients[2] = upstream;

    return step;
}

/**
 * @brief 在全部梯度计算完成后更新参数，避免混用更新前后的模型。
 * @param model 可写模型；不转移所有权。
 * @param step 由当前模型计算得到的只读梯度，不可在更新后重复使用。
 * @param learning_rate 有限正学习率；更新结果须在 double 的有限范围内。
 */
static void linear_update(LinearModel *model, const LinearStep *step, double learning_rate)
{
    model->weights[0] -= learning_rate * step->gradients[0];
    model->weights[1] -= learning_rate * step->gradients[1];
    model->bias -= learning_rate * step->gradients[2];
}

/**
 * @brief 确定性模拟内存分配失败，不尝试耗尽机器内存。
 * @param count 请求元素数，此测试不使用。
 * @param size 请求元素字节数，此测试不使用。
 * @return 始终为空指针。
 */
static void *fail_allocate(size_t count, size_t size)
{
    (void)count;
    (void)size;
    return nullptr;
}

/**
 * @brief 验证矩阵失败路径与边界，测试缓冲区使用栈内存，不调用 destroy。
 */
static void test_matrix_boundaries(void)
{
    Matrix empty = {0};
    assert(!matrix_create(&empty, 0, 2, calloc));
    assert(!matrix_create(&empty, SIZE_MAX, 2, calloc));
    assert(!matrix_create(&empty, SIZE_MAX / sizeof(double) + 1, 1, calloc));
    assert(!matrix_create(&empty, 2, 2, fail_allocate));
    assert(!matrix_create(&empty, 2, 2, nullptr));
    assert(!matrix_create(nullptr, 2, 2, calloc));
    assert(empty.data == nullptr && empty.rows == 0 && empty.capacity == 0);

    matrix_destroy(&empty);
    matrix_destroy(&empty);
    matrix_destroy(nullptr);

    double input[] = {2, 3};
    double right[] = {4, 5};
    double sentinel[] = {99};
    Matrix a = {1, 2, 2, input};
    Matrix b = {2, 1, 2, right};
    Matrix output = {1, 1, 1, sentinel};
    assert(matrix_multiply(&a, &b, &output));
    assert(sentinel[0] == 23); /* 一行乘一列，就是向量点积。 */

    sentinel[0] = 99;
    b.rows = 1;
    assert(!matrix_multiply(&a, &b, &output));

    b.rows = 2;
    output.capacity = 0;
    assert(!matrix_multiply(&a, &b, &output));

    output.capacity = 1;
    assert(!matrix_multiply(nullptr, &b, &output));
    assert(!matrix_add(&a, &b, &output));
    assert(!matrix_transpose(&a, &output));
    assert(sentinel[0] == 99);
    assert(!matrix_multiply(&a, &b, &a));

    Matrix alias = {2, 1, 2, input};
    assert(!matrix_transpose(&a, &alias));
    assert(input[0] == 2 && input[1] == 3);
    assert(!matrix_create(&a, 1, 2, calloc));
    assert(matrix_add(&a, &a, &a));
    assert(input[0] == 4 && input[1] == 6);
}

/**
 * @brief 验证手算梯度、中心差分与更新，断言不会承诺模型的泛化能力。
 */
static void test_linear_gradients(void)
{
    const double x[] = {1, 2};
    LinearModel model = {{0.5, -0.25}, 0.1};
    LinearStep step = linear_step(&model, x, 1);
    const double expected[] = {-1.8, -3.6, -1.8};
    assert(fabs(step.prediction - 0.1) < 1e-12);
    assert(fabs(step.loss - 0.81) < 1e-12);

    for (size_t i = 0; i < 3; ++i) {
        LinearModel plus = model;
        LinearModel minus = model;
        double epsilon = 1e-5;
        if (i < 2) {
            plus.weights[i] += epsilon;
            minus.weights[i] -= epsilon;
        } else {
            plus.bias += epsilon;
            minus.bias -= epsilon;
        }
        double numerical =
            (linear_step(&plus, x, 1).loss - linear_step(&minus, x, 1).loss) / (2 * epsilon);
        assert(fabs(step.gradients[i] - expected[i]) < 1e-12);
        assert(fabs(numerical - step.gradients[i]) < 1e-8);
    }
    linear_update(&model, &step, 0.05);
    assert(fabs(model.weights[0] - 0.59) < 1e-12);
    assert(fabs(model.weights[1] + 0.07) < 1e-12);
    assert(fabs(model.bias - 0.19) < 1e-12);

    LinearStep updated = linear_step(&model, x, 1);
    assert(fabs(updated.prediction - 0.64) < 1e-12);
    assert(fabs(updated.loss - 0.1296) < 1e-12);

    const double zero[] = {0, 0};
    LinearStep bias_only = linear_step(&model, zero, 1);
    assert(bias_only.gradients[0] == 0 && bias_only.gradients[1] == 0);

    LinearStep perfect = linear_step(&model, x, updated.prediction);
    assert(perfect.loss == 0 && perfect.gradients[2] == 0);
}

/**
 * @brief 用同一初始模型展示三种学习率的误差变化，仅运行少量固定步骤。
 */
static void demonstrate_learning_rates(void)
{
    const double x[] = {1, 2};
    const double rates[] = {0.05, 0.12, 0.20};
    for (size_t i = 0; i < 3; ++i) {
        LinearModel model = {{0.5, -0.25}, 0.1};
        printf("学习率 %.2f，更新前误差：", rates[i]);
        for (size_t iteration = 0; iteration < 6; ++iteration) {
            LinearStep step = linear_step(&model, x, 1);
            assert(isfinite(step.loss));
            printf(" %+.5f", step.error);
            /* 本样本的新误差 = (1 - 12*学习率)*旧误差。 */
            linear_update(&model, &step, rates[i]);
            LinearStep next = linear_step(&model, x, 1);
            assert(fabs(next.error - (1 - 12 * rates[i]) * step.error) < 1e-12);
        }
        putchar('\n');
    }
}

/**
 * @brief 运行阶段 0 案例及断言，所有拥有的矩阵在退出前释放。
 * @return 成功为 EXIT_SUCCESS；分配或运算失败时明确报错并返回 EXIT_FAILURE。
 */
int main(void)
{
    test_matrix_boundaries();
    test_linear_gradients();
    Matrix a = {0};
    Matrix b = {0};
    Matrix product = {0};
    Matrix transposed = {0};
    Matrix sum = {0};
    int result = EXIT_FAILURE;
    if (!matrix_create(&a, 2, 3, calloc) || !matrix_create(&b, 3, 2, calloc) ||
        !matrix_create(&product, 2, 2, calloc) || !matrix_create(&transposed, 3, 2, calloc) ||
        !matrix_create(&sum, 2, 3, calloc)) {
        fputs("阶段 0：矩阵创建失败（形状、容量或内存分配错误）。\n", stderr);
        goto cleanup;
    }
    const double a_values[] = {1, 2, 3, 4, 5, 6};
    const double b_values[] = {7, 8, 9, 10, 11, 12};
    memcpy(a.data, a_values, sizeof(a_values));
    memcpy(b.data, b_values, sizeof(b_values));
    if (!matrix_multiply(&a, &b, &product) || !matrix_transpose(&a, &transposed) ||
        !matrix_add(&a, &a, &sum)) {
        fputs("阶段 0：矩阵运算的形状或容量检查失败。\n", stderr);
        goto cleanup;
    }
    const double expected_product[] = {58, 64, 139, 154};
    const double expected_transpose[] = {1, 4, 2, 5, 3, 6};
    for (size_t i = 0; i < product.capacity; ++i) {
        assert(product.data[i] == expected_product[i]);
    }
    for (size_t i = 0; i < a.capacity; ++i) {
        assert(transposed.data[i] == expected_transpose[i]);
        assert(sum.data[i] == 2 * a_values[i]);
    }
    puts("=== 矩阵：行优先布局，索引 = row * columns + column ===");
    matrix_print("A", &a);
    matrix_print("B", &b);
    matrix_print("A × B", &product);
    matrix_print("A^T", &transposed);
    matrix_print("A + A（逐元素）", &sum);

    const double x[] = {1, 2};
    LinearModel model = {{0.5, -0.25}, 0.1};
    LinearStep step = linear_step(&model, x, 1);
    puts("\n=== 单样本线性模型：x(2)，w(2)，b/y/loss 为标量 ===");
    printf("预测 = 0.5*1 + (-0.25)*2 + 0.1 = %.4f\n", step.prediction);
    printf("误差 = 预测 - 1 = %.4f，单样本 MSE = %.4f\n", step.error, step.loss);
    printf("链式法则：dL/d预测 = 2*误差 = %.4f\n", 2 * step.error);
    printf("梯度 (dw0, dw1, db) = (%.4f, %.4f, %.4f)\n", step.gradients[0], step.gradients[1],
           step.gradients[2]);
    linear_update(&model, &step, 0.05);
    LinearStep updated = linear_step(&model, x, 1);
    printf("学习率 0.05 更新后：(w0, w1, b) = (%.4f, %.4f, %.4f)\n", model.weights[0],
           model.weights[1], model.bias);
    printf("重新前向：预测 = %.4f，MSE = %.4f\n", updated.prediction, updated.loss);
    puts("\n=== 学习率实验：收敛、交替收敛、交替发散 ===");
    demonstrate_learning_rates();
    puts("阶段 0 验证通过：矩阵结果、边界、分配失败、手算梯度与数值梯度。\n");
    result = EXIT_SUCCESS;

cleanup:
    matrix_destroy(&sum);
    matrix_destroy(&transposed);
    matrix_destroy(&product);
    matrix_destroy(&b);
    matrix_destroy(&a);
    return result;
}
