/** @file @brief 四种可逐步观察的排序核心，不依赖 SDL 或绘图。 */
#ifndef CDEMO_SORTING_H
#define CDEMO_SORTING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    SORTING_ITEM_COUNT = 48,
    SORTING_VALUE_MAX = 100
};

typedef enum {
    SORTING_BUBBLE,
    SORTING_INSERTION,
    SORTING_SELECTION,
    SORTING_QUICK
} SortingAlgorithm;

typedef enum {
    SORTING_RUNNING,
    SORTING_COMPLETE
} SortingPhase;

/** 排序键及原始次序；次序编号用于验证稳定排序不会打乱相等键。 */
typedef struct {
    uint8_t value;
    uint8_t original_order;
} SortingItem;

/** 算法状态可逐步推进；调用者只读数组来绘制当前比较与移动。 */
typedef struct {
    SortingAlgorithm algorithm; /* 当前排序算法。 */
    SortingPhase phase;           /* 运行中或已完成。 */
    SortingItem items[SORTING_ITEM_COUNT]; /* 当前数组；value 是排序键。 */
    bool settled[SORTING_ITEM_COUNT];      /* 已确定位置或当前有序前缀。 */
    size_t outer;               /* 趟序号、待插入位置或未排序后缀首索引。 */
    size_t cursor;              /* 相邻比较左索引或当前扫描位置。 */
    size_t selected;            /* 选择排序当前找到的最小值位置。 */
    size_t hole;                /* 插入排序中 key 暂存时的空位。 */
    SortingItem key;            /* 插入排序暂存的待插入项。 */
    bool key_active;            /* true 时绘制时用 key 覆盖 hole 位置。 */
    size_t quick_low;           /* 当前快速排序分区的左边界，包含。 */
    size_t quick_high;          /* 当前分区的右边界，枢轴位于此处。 */
    size_t quick_cursor;        /* 正在与枢轴比较的元素索引。 */
    size_t quick_boundary;      /* 小于等于枢轴的区域末端边界。 */
    size_t quick_stack_low[SORTING_ITEM_COUNT];  /* 待处理区间左端点栈。 */
    size_t quick_stack_high[SORTING_ITEM_COUNT]; /* 待处理区间右端点栈。 */
    size_t quick_stack_size;    /* 显式区间栈中的待处理区间数。 */
    bool quick_active;          /* 当前是否正在扫描并分区一个区间。 */
    size_t pivot_index;         /* 当前枢轴索引；无枢轴时为 SIZE_MAX。 */
    size_t active_a;            /* 当前比较的第一个索引；无索引时为 SIZE_MAX。 */
    size_t active_b;            /* 当前比较的第二个索引；无索引时为 SIZE_MAX。 */
    size_t changed_a;           /* 最近写入的第一个索引；无写入时为 SIZE_MAX。 */
    size_t changed_b;           /* 最近写入的第二个索引；无写入时为 SIZE_MAX。 */
    bool swapped_in_pass;       /* 冒泡排序本趟是否发生交换，用于提前结束。 */
    size_t comparisons;         /* 按键值执行的比较次数。 */
    size_t writes;              /* 写入数组槽位的次数。 */
} SortingState;

/**
 * @brief 从固定长度输入初始化一轮排序。
 * @param input 恰含 SORTING_ITEM_COUNT 个元素的只读数组；每个 value 范围为
 *        0..SORTING_VALUE_MAX。
 * @param algorithm 冒泡、插入、选择或快速排序。
 * @param state 有效输出地址；成功后整体替换，不接管 input 所有权。
 * @return 参数有效时为 true；空指针、非法算法或越界键值返回 false，
 *         并保持 state 不变。
 */
bool sorting_start(const SortingItem input[SORTING_ITEM_COUNT], SortingAlgorithm algorithm,
                   SortingState *state);

/**
 * @brief 执行一次排序微步骤，至多比较一对键并进行相关移动。
 * @param state 由 sorting_start 初始化的可读写状态。
 * @note 完成或参数为空时不改变状态；数组可通过 state->items 观察。
 */
void sorting_step(SortingState *state);

#endif
