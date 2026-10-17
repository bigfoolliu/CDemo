/** @file @brief 验证排序结果、重复值次序、单步状态和参数边界。 */
#include "../src/graphics/sorting.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief 构造含重复键的固定样本，保留每个元素的原始次序。
 * @param items 输出数组，容量须为 SORTING_ITEM_COUNT。
 */
static void make_items(SortingItem items[SORTING_ITEM_COUNT])
{
    for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
        items[index].value = (uint8_t)((index * 17 + 9) % 13 + 1);
        items[index].original_order = (uint8_t)index;
    }
}

/**
 * @brief 逐步完成排序，并限制最大步数以避免错误状态无限循环。
 * @param state 已初始化的可读写排序状态。
 */
static void finish_sort(SortingState *state)
{
    for (size_t step = 0; step < SORTING_ITEM_COUNT * SORTING_ITEM_COUNT * 2 &&
                           state->phase == SORTING_RUNNING;
         ++step) {
        sorting_step(state);
    }
    assert(state->phase == SORTING_COMPLETE);
}

/**
 * @brief 检查键值有序，并按需检查相等键保持原始次序。
 * @param state 已完成的只读排序状态。
 * @param require_stable 为 true 时还要求相等键的原始次序递增。
 */
static void check_sorted(const SortingState *state, bool require_stable)
{
    for (size_t index = 1; index < SORTING_ITEM_COUNT; ++index) {
        assert(state->items[index - 1].value <= state->items[index].value);
        if (require_stable && state->items[index - 1].value == state->items[index].value) {
            assert(state->items[index - 1].original_order < state->items[index].original_order);
        }
    }
    for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
        assert(state->settled[index]);
    }
}

/** @brief 验证四种算法都完成排序，并检查其比较次数特征。 */
static void test_algorithms(void)
{
    SortingItem input[SORTING_ITEM_COUNT];
    make_items(input);
    for (size_t sample = 0; sample < 5; ++sample) {
        if (sample != 0) {
            for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
                input[index].value = (uint8_t)((index * 19 + sample * 23 + index * index) % 31 + 1);
                input[index].original_order = (uint8_t)index;
            }
        }
        SortingItem original[SORTING_ITEM_COUNT];
        memcpy(original, input, sizeof original);
        for (int algorithm = SORTING_BUBBLE; algorithm <= SORTING_QUICK; ++algorithm) {
            SortingState state;
            assert(sorting_start(input, (SortingAlgorithm)algorithm, &state));
            assert(state.phase == SORTING_RUNNING);
            assert(state.comparisons == 0 && state.writes == 0);
            assert(state.active_a == SIZE_MAX && state.active_b == SIZE_MAX);
            finish_sort(&state);
            bool stable = algorithm == SORTING_BUBBLE || algorithm == SORTING_INSERTION;
            check_sorted(&state, stable);
            assert(memcmp(input, original, sizeof input) == 0);
            if (algorithm == SORTING_SELECTION) {
                assert(state.comparisons == SORTING_ITEM_COUNT * (SORTING_ITEM_COUNT - 1) / 2);
            }
        }
    }
}

/** @brief 验证单次调用只执行一个冒泡比较，并准确记录交换与写入。 */
static void test_single_step(void)
{
    SortingItem input[SORTING_ITEM_COUNT];
    for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
        input[index].value = (uint8_t)(SORTING_ITEM_COUNT - index);
        input[index].original_order = (uint8_t)index;
    }
    SortingState state;
    assert(sorting_start(input, SORTING_BUBBLE, &state));
    sorting_step(&state);
    assert(state.comparisons == 1 && state.writes == 2);
    assert(state.active_a == 0 && state.active_b == 1);
    assert(state.changed_a == 0 && state.changed_b == 1);
    assert(state.items[0].value == SORTING_ITEM_COUNT - 1);
    assert(state.items[1].value == SORTING_ITEM_COUNT);

    assert(sorting_start(input, SORTING_QUICK, &state));
    sorting_step(&state);
    assert(state.comparisons == 1 && state.writes == 0);
    assert(state.active_a == 0 && state.active_b == SORTING_ITEM_COUNT - 1);
    assert(state.pivot_index == SORTING_ITEM_COUNT - 1);
    assert(state.items[0].value == SORTING_ITEM_COUNT);
    finish_sort(&state);
    check_sorted(&state, false);
}

/** @brief 验证有序输入的边界行为及选择排序不保证稳定。 */
static void test_boundaries_and_stability(void)
{
    SortingItem input[SORTING_ITEM_COUNT];
    for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
        input[index].value = (uint8_t)(index + 1);
        input[index].original_order = (uint8_t)index;
    }
    SortingState insertion;
    SortingState bubble;
    assert(sorting_start(input, SORTING_BUBBLE, &bubble));
    finish_sort(&bubble);
    assert(bubble.comparisons == SORTING_ITEM_COUNT - 1);
    assert(bubble.writes == 0);
    check_sorted(&bubble, true);
    assert(sorting_start(input, SORTING_INSERTION, &insertion));
    finish_sort(&insertion);
    assert(insertion.comparisons == SORTING_ITEM_COUNT - 1);
    assert(insertion.writes == 0);
    check_sorted(&insertion, true);

    SortingState quick;
    assert(sorting_start(input, SORTING_QUICK, &quick));
    finish_sort(&quick);
    check_sorted(&quick, false);
    assert(quick.comparisons == SORTING_ITEM_COUNT * (SORTING_ITEM_COUNT - 1) / 2);
    assert(quick.writes == 0);

    SortingState completed = insertion;
    sorting_step(&insertion);
    assert(insertion.phase == completed.phase);
    assert(insertion.comparisons == completed.comparisons);
    assert(memcmp(insertion.items, completed.items, sizeof insertion.items) == 0);

    input[0].value = 2;
    input[1].value = 2;
    input[2].value = 1;
    for (size_t index = 3; index < SORTING_ITEM_COUNT; ++index) {
        input[index].value = (uint8_t)index;
    }
    SortingState selection;
    assert(sorting_start(input, SORTING_SELECTION, &selection));
    finish_sort(&selection);
    check_sorted(&selection, false);
    assert(selection.items[1].original_order == 1);
    assert(selection.items[2].original_order == 0);

    SortingState quick_duplicates;
    assert(sorting_start(input, SORTING_QUICK, &quick_duplicates));
    finish_sort(&quick_duplicates);
    check_sorted(&quick_duplicates, false);
    assert(quick_duplicates.items[1].original_order == 1);
    assert(quick_duplicates.items[2].original_order == 0);

    SortingState unchanged = {.algorithm = SORTING_BUBBLE,
                              .phase = SORTING_COMPLETE,
                              .comparisons = 123};
    assert(!sorting_start(nullptr, SORTING_BUBBLE, &unchanged));
    assert(!sorting_start(input, (SortingAlgorithm)99, &unchanged));
    assert(!sorting_start(input, SORTING_BUBBLE, nullptr));
    SortingItem invalid[SORTING_ITEM_COUNT];
    memcpy(invalid, input, sizeof invalid);
    invalid[0].value = SORTING_VALUE_MAX + 1;
    assert(!sorting_start(invalid, SORTING_BUBBLE, &unchanged));
    assert(unchanged.algorithm == SORTING_BUBBLE);
    assert(unchanged.phase == SORTING_COMPLETE && unchanged.comparisons == 123);
    sorting_step(nullptr);
}

int main(void)
{
    test_algorithms();
    test_single_step();
    test_boundaries_and_stability();
    puts("排序测试通过：四种算法、单步状态、稳定性和边界。");
    return 0;
}
