/** @file
 * @brief 函数指针回调与上下文传递。
 */
#include <assert.h>
#include <limits.h>
#include <stdio.h>

/**
 * @brief 元素访问回调，用返回值控制遍历是否继续。
 * @param value 当前元素的整数值。
 * @param context 调用者借用的上下文，具体类型由回调与调用者共同约定。
 * @return true 表示处理成功并继续，false 表示拒绝当前元素并停止。
 */
typedef bool (*Visit)(int value, void *context);

/**
 * @brief 遍历数组并调用可提前停止的回调。
 * @param values 包含 count 个元素的只读数组；count 为零时可为空。
 * @param count 待遍历的元素数。
 * @param visit 有效回调，返回 false 时停止遍历。
 * @param context 原样传给回调的借用上下文；类型由回调约定。
 * @return 回调成功处理的元素数，不包含拒绝的当前元素。
 * @note void * 携带状态，但类型契约由调用者保证；返回 false 提前停止。
 */
static size_t visit_values(const int *values, size_t count, Visit visit, void *context)
{
    for (size_t i = 0; i < count; ++i) {
        if (!visit(values[i], context)) {
            return i; /* 返回成功处理的元素数，不包括拒绝的当前元素。 */
        }
    }
    return count;
}

typedef struct {
    int sum;
    int limit;
} SumContext;

/**
 * @brief 在指定上限内累加非负整数。
 * @param value 待累加的整数，负数将被拒绝。
 * @param context 可读写 SumContext 地址，满足 0 <= sum <= limit <= INT_MAX。
 * @return 累加成功返回 true；负数或超过上限返回 false 且状态不变。
 */
static bool limited_sum(int value, void *context)
{
    SumContext *state = context;
    if (value < 0 || value > state->limit - state->sum) {
        return false; /* 不变量：0 <= sum <= limit <= INT_MAX。 */
    }
    state->sum += value;
    return true;
}

typedef struct {
    int maximum;
    size_t count;
} MaxContext;

/**
 * @brief 记录已访问整数中的最大值。
 * @param value 当前访问的整数。
 * @param context 已初始化的可读写 MaxContext 地址；count 不得溢出。
 * @return 始终返回 true，继续遍历。
 */
static bool find_maximum(int value, void *context)
{
    MaxContext *state = context;
    if (state->count == 0 || value > state->maximum) {
        state->maximum = value;
    }
    ++state->count;
    return true;
}

/**
 * @brief 运行并验证函数指针回调与上下文传递。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    const int values[] = {2, 3, 9, 1};
    SumContext sum = {.limit = 10};
    size_t processed = visit_values(values, 4, limited_sum, &sum);
    assert(processed == 2 && sum.sum == 5);
    MaxContext maximum = {0};
    processed = visit_values(values, 4, find_maximum, &maximum);
    assert(processed == 4 && maximum.maximum == 9);
    processed = visit_values(nullptr, 0, find_maximum, &maximum);
    assert(processed == 0 && maximum.count == 4);
    sum = (SumContext){.sum = INT_MAX, .limit = INT_MAX};
    processed = visit_values(values, 4, limited_sum, &sum);
    assert(processed == 0 && sum.sum == INT_MAX);
    puts("回调接口：函数指针、上下文、策略复用与提前终止验证通过");
    return 0;
}
