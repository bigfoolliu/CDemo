/** @file
 * @brief 以单步状态机实现四种排序，供绘图界面观察。
 * 所有算法都直接排序固定长度数组；输入复制进状态，不依赖 SDL。
 */
#include "sorting.h"
#include <string.h>

/**
 * @brief 检查排序算法枚举是否受支持。
 * @param algorithm 待检查的算法值。
 * @return 匹配四种实现之一时为 true。
 */
static bool algorithm_is_valid(SortingAlgorithm algorithm)
{
    return algorithm == SORTING_BUBBLE || algorithm == SORTING_INSERTION ||
           algorithm == SORTING_SELECTION || algorithm == SORTING_QUICK;
}

/**
 * @brief 交换两个不同数组槽位，并记录本次写入位置。
 * @param state 可读写的排序状态。
 * @param first 第一个有效槽位索引。
 * @param second 第二个有效且不同的槽位索引。
 */
static void swap_items(SortingState *state, size_t first, size_t second)
{
    SortingItem temporary = state->items[first];
    state->items[first] = state->items[second];
    state->items[second] = temporary;
    state->writes += 2;
    state->changed_a = first;
    state->changed_b = second;
}

/**
 * @brief 将快速排序的非平凡子区间加入显式栈。
 * @param state 正在执行快速排序的可读写状态。
 * @param low 子区间左端点，包含。
 * @param high 子区间右端点，包含，且须满足 low <= high。
 * @note 单元素区间无需继续处理，会立即标记为已排定。待处理的非平凡区间互不重叠，
 *       因此显式栈不会超过固定数组容量。
 */
static void push_quick_range(SortingState *state, size_t low, size_t high)
{
    if (low == high) {
        state->settled[low] = true;
        return;
    }
    state->quick_stack_low[state->quick_stack_size] = low;
    state->quick_stack_high[state->quick_stack_size] = high;
    ++state->quick_stack_size;
}

/**
 * @brief 将当前枢轴归位，并安排左右两个待处理区间。
 * @param state 正在执行快速排序的可读写状态。
 * @note 左右区间只在边界有效时构造，避免无符号索引下溢或溢出。
 */
static void finish_quick_partition(SortingState *state)
{
    size_t pivot = state->quick_boundary;
    if (pivot != state->quick_high) {
        swap_items(state, pivot, state->quick_high);
    }
    state->settled[pivot] = true;
    state->pivot_index = SIZE_MAX;
    state->quick_active = false;
    if (pivot > state->quick_low) {
        push_quick_range(state, state->quick_low, pivot - 1);
    }
    if (pivot < state->quick_high) {
        push_quick_range(state, pivot + 1, state->quick_high);
    }
    if (state->quick_stack_size == 0) {
        for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
            state->settled[index] = true;
        }
        state->phase = SORTING_COMPLETE;
    }
}

/**
 * @brief 对当前子区间执行 Lomuto 分区中的一次键比较。
 * @param state 正在执行快速排序的可读写状态。
 * @note 区间末元素始终保留为枢轴；扫描值不大于枢轴时移入左侧区域。
 */
static void quick_step(SortingState *state)
{
    if (!state->quick_active) {
        if (state->quick_stack_size == 0) {
            state->phase = SORTING_COMPLETE;
            return;
        }
        --state->quick_stack_size;
        state->quick_low = state->quick_stack_low[state->quick_stack_size];
        state->quick_high = state->quick_stack_high[state->quick_stack_size];
        state->quick_cursor = state->quick_low;
        state->quick_boundary = state->quick_low;
        state->quick_active = true;
        state->pivot_index = state->quick_high;
    }

    if (state->quick_cursor == state->quick_high) {
        finish_quick_partition(state);
        return;
    }

    state->pivot_index = state->quick_high;
    state->active_a = state->quick_cursor;
    state->active_b = state->quick_high;
    ++state->comparisons;
    if (state->items[state->quick_cursor].value <= state->items[state->quick_high].value) {
        if (state->quick_cursor != state->quick_boundary) {
            swap_items(state, state->quick_cursor, state->quick_boundary);
        }
        ++state->quick_boundary;
    }
    ++state->quick_cursor;
}

/**
 * @brief 标记插入排序已完成的前缀，并准备下一个 key。
 * @param state 正在执行插入排序的可读写状态。
 */
static void finish_insertion(SortingState *state)
{
    for (size_t index = 0; index <= state->outer; ++index) {
        state->settled[index] = true;
    }
    state->key_active = false;
    ++state->outer;
    if (state->outer == SORTING_ITEM_COUNT) {
        state->phase = SORTING_COMPLETE;
        return;
    }
    state->hole = state->outer;
    state->key = state->items[state->outer];
    state->key_active = true;
}

/** @copydoc sorting_start */
bool sorting_start(const SortingItem input[SORTING_ITEM_COUNT], SortingAlgorithm algorithm,
                   SortingState *state)
{
    if (input == nullptr || state == nullptr || !algorithm_is_valid(algorithm)) {
        return false;
    }
    for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
        if (input[index].value > SORTING_VALUE_MAX) {
            return false;
        }
    }
    SortingState result = {.algorithm = algorithm,
                           .phase = SORTING_RUNNING,
                           .active_a = SIZE_MAX,
                           .active_b = SIZE_MAX,
                           .changed_a = SIZE_MAX,
                           .changed_b = SIZE_MAX};
    memcpy(result.items, input, sizeof result.items);
    if (algorithm == SORTING_INSERTION) {
        result.settled[0] = true;
        result.outer = 1;
        result.hole = result.outer;
        result.key = result.items[result.outer];
        result.key_active = true;
    } else if (algorithm == SORTING_SELECTION) {
        result.cursor = 1;
    } else if (algorithm == SORTING_QUICK) {
        result.quick_stack_low[0] = 0;
        result.quick_stack_high[0] = SORTING_ITEM_COUNT - 1;
        result.quick_stack_size = 1;
        result.pivot_index = SIZE_MAX;
    }
    *state = result;
    return true;
}

/** @copydoc sorting_step */
void sorting_step(SortingState *state)
{
    if (state == nullptr || state->phase != SORTING_RUNNING) {
        return;
    }
    state->active_a = SIZE_MAX;
    state->active_b = SIZE_MAX;
    state->changed_a = SIZE_MAX;
    state->changed_b = SIZE_MAX;

    if (state->algorithm == SORTING_QUICK) {
        quick_step(state);
        return;
    }

    if (state->algorithm == SORTING_BUBBLE) {
        size_t end = SORTING_ITEM_COUNT - 1 - state->outer;
        state->active_a = state->cursor;
        state->active_b = state->cursor + 1;
        ++state->comparisons;
        if (state->items[state->cursor].value > state->items[state->cursor + 1].value) {
            swap_items(state, state->cursor, state->cursor + 1);
            state->swapped_in_pass = true;
        }
        ++state->cursor;
        if (state->cursor == end) {
            state->settled[end] = true;
            ++state->outer;
            state->cursor = 0;
            bool finished = state->outer == SORTING_ITEM_COUNT - 1 ||
                            !state->swapped_in_pass;
            state->swapped_in_pass = false;
            if (finished) {
                for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
                    state->settled[index] = true;
                }
                state->phase = SORTING_COMPLETE;
            }
        }
        return;
    }

    if (state->algorithm == SORTING_INSERTION) {
        if (state->hole == 0) {
            state->items[0] = state->key;
            ++state->writes;
            state->changed_a = 0;
            finish_insertion(state);
            return;
        }
        state->active_a = state->hole - 1;
        state->active_b = state->hole;
        ++state->comparisons;
        if (state->items[state->hole - 1].value > state->key.value) {
            state->items[state->hole] = state->items[state->hole - 1];
            ++state->writes;
            state->changed_b = state->hole;
            state->settled[state->hole - 1] = false;
            --state->hole;
            if (state->hole == 0) {
                state->items[0] = state->key;
                ++state->writes;
                state->changed_a = 0;
                finish_insertion(state);
            }
            return;
        }
        if (state->hole != state->outer) {
            state->items[state->hole] = state->key;
            ++state->writes;
            state->changed_a = state->hole;
        }
        finish_insertion(state);
        return;
    }

    state->active_a = state->selected;
    state->active_b = state->cursor;
    ++state->comparisons;
    if (state->items[state->cursor].value < state->items[state->selected].value) {
        state->selected = state->cursor;
    }
    ++state->cursor;
    if (state->cursor == SORTING_ITEM_COUNT) {
        if (state->selected != state->outer) {
            swap_items(state, state->outer, state->selected);
        }
        state->settled[state->outer] = true;
        ++state->outer;
        if (state->outer == SORTING_ITEM_COUNT - 1) {
            state->settled[state->outer] = true;
            state->phase = SORTING_COMPLETE;
            return;
        }
        state->selected = state->outer;
        state->cursor = state->outer + 1;
    }
}
