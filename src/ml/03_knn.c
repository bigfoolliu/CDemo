/** @file
 * @brief K 近邻二分类。
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

enum { SAMPLE_COUNT = 6 };
typedef struct { double x; double y; int label; } Sample;
typedef struct { double distance; size_t index; int label; } Neighbor;

/**
 * @brief 按距离和原始索引比较邻居，供 qsort 使用。
 * @param left 第一个只读 Neighbor 对象地址。
 * @param right 第二个只读 Neighbor 对象地址。
 * @return 负数、零、正数分别表示 left 小于、等于、大于 right。
 */
static int compare_neighbors(const void *left, const void *right)
{
    const Neighbor *a = left;
    const Neighbor *b = right;
    if (a->distance != b->distance) {
        return a->distance < b->distance ? -1 : 1;
    }
    /* 同距离时按原始索引排序，结果不依赖 qsort 的稳定性。 */
    return (a->index > b->index) - (a->index < b->index);
}

/**
 * @brief 使用最近的 k 个样本进行二分类。
 * @param samples 含 SAMPLE_COUNT 个样本的只读数组，标签为 0/1，坐标有限。
 * @param x 待分类点的有限横坐标。
 * @param y 待分类点的有限纵坐标。
 * @param k 邻居数，范围为 1..SAMPLE_COUNT。
 * @param output 有效类别输出地址，票数平局时写入 0。
 * @return 分类成功返回 true；k 非法返回 false 且输出不变。
 * @note 教学接口固定为六个样本、二分类；输入坐标要求有限。
 */
static bool classify(const Sample samples[SAMPLE_COUNT], double x, double y,
                     size_t k, int *output)
{
    if (k == 0 || k > SAMPLE_COUNT) {
        return false;
    }
    Neighbor neighbors[SAMPLE_COUNT];
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        double dx = samples[i].x - x;
        double dy = samples[i].y - y;
        neighbors[i] = (Neighbor){dx * dx + dy * dy, i, samples[i].label};
    }
    /* 距离平方的排序与距离相同，无需计算平方根。 */
    qsort(neighbors, SAMPLE_COUNT, sizeof(neighbors[0]), compare_neighbors);
    size_t positives = 0;
    for (size_t i = 0; i < k; ++i) {
        positives += neighbors[i].label == 1;
    }
    *output = positives > k - positives ? 1 : 0; /* 票数平局时选 0。 */
    return true;
}

/**
 * @brief 运行并验证K 近邻二分类。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    const Sample samples[SAMPLE_COUNT] = {
        {0, 0, 0}, {0, 1, 0}, {1, 0, 0},
        {4, 4, 1}, {4, 5, 1}, {5, 4, 1}
    };
    int label = -1;
    bool success = classify(samples, 0.2, 0.3, 3, &label);
    assert(success && label == 0);
    success = classify(samples, 4.2, 4.3, 3, &label);
    assert(success && label == 1);
    success = classify(samples, 2, 2, 6, &label);
    assert(success && label == 0); /* 3 比 3 平局。 */
    label = -1;
    success = classify(samples, 0, 0, 0, &label);
    assert(!success && label == -1);
    success = classify(samples, 0, 0, 7, &label);
    assert(!success && label == -1);
    puts("K 近邻：k=3，(0.2,0.3) → 类别0，(4.2,4.3) → 类别1");
    return 0;
}
