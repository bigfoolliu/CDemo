#include <assert.h>
#include <limits.h>
#include <stdio.h>

typedef bool (*Visit)(int value, void *context);

/* void * 携带状态，但类型契约由调用者保证；返回 false 提前停止。 */
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

static bool find_maximum(int value, void *context)
{
    MaxContext *state = context;
    if (state->count == 0 || value > state->maximum) {
        state->maximum = value;
    }
    ++state->count;
    return true;
}

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
