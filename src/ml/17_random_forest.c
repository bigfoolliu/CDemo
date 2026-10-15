/**
 * @file
 * @brief 随机森林分类：自助采样、节点随机选特征、投票与袋外评估。
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    FEATURE_COUNT = 2,
    SAMPLE_COUNT = 64,
    MAX_NODES = 2 * SAMPLE_COUNT - 1,
    TREE_COUNT = 31,
    EXPERIMENT_SEEDS = 3
};

typedef struct {
    double features[FEATURE_COUNT];
    int label;
} Sample;

typedef struct {
    int feature;
    double threshold;
    int left;
    int right;
    int label;
    size_t samples;
    size_t positives;
} Node;

typedef struct {
    Node nodes[MAX_NODES];
    size_t node_count;
} Tree;

typedef struct {
    Tree tree;
    bool in_bag[SAMPLE_COUNT];
} ForestMember;

typedef struct {
    ForestMember members[TREE_COUNT];
    size_t tree_count;
} Forest;

typedef struct {
    double value;
    int label;
    size_t index;
} SortedSample;

typedef struct {
    int feature;
    double threshold;
    double gain;
} Candidate;

typedef struct {
    size_t errors;
    size_t count;
} Metrics;

/**
 * @brief 推进确定性线性同余随机数状态。
 * @param state 有效的可写随机状态；调用者独占维护。
 * @return 更新后状态的高 16 位；用于可重复教学，不用于安全用途。
 */
static uint32_t random_next(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state >> 16;
}

/**
 * @brief 将整数状态映射到 [0, upper_bound) 的下标。
 * @param state 有效的可写随机状态。
 * @param upper_bound 正整数范围上界。
 * @return 范围内伪随机下标。
 */
static size_t random_index(uint32_t *state, size_t upper_bound)
{
    return (size_t)random_next(state) % upper_bound;
}

/**
 * @brief 计算二分类基尼不纯度。
 * @param positives 正类数量，不超过 count。
 * @param count 节点样本数，大于零。
 * @return 范围 [0, 0.5] 内的不纯度。
 */
static double impurity(size_t positives, size_t count)
{
    double probability = (double)positives / (double)count;
    return 2.0 * probability * (1.0 - probability);
}

/**
 * @brief 按特征值和原数组下标排序，稳定确定相同值的扫描顺序。
 * @param left 指向只读 SortedSample。
 * @param right 指向只读 SortedSample。
 * @return qsort 所需的负数、零或正数。
 */
static int compare_sorted(const void *left, const void *right)
{
    const SortedSample *a = left;
    const SortedSample *b = right;
    if (a->value < b->value) {
        return -1;
    }
    if (a->value > b->value) {
        return 1;
    }
    return (a->index > b->index) - (a->index < b->index);
}

/**
 * @brief 在随机抽取的一个特征上寻找最佳基尼切分。
 * @param samples 当前树的只读 bootstrap 样本数组。
 * @param indices 当前节点的样本下标，至少 count 个。
 * @param count 当前节点样本数，范围 1..SAMPLE_COUNT。
 * @param positives 当前节点正类数，不超过 count。
 * @param feature 本节点随机抽中的特征编号，范围 0..FEATURE_COUNT-1。
 * @param result 有效可写候选结果；找不到增益时保持不变。
 * @return 存在严格正增益的切分时为 true。
 */
static bool find_candidate(const Sample *samples, const size_t *indices, size_t count,
                           size_t positives, int feature, Candidate *result)
{
    SortedSample sorted[SAMPLE_COUNT];
    for (size_t i = 0; i < count; ++i) {
        size_t index = indices[i];
        sorted[i] = (SortedSample){samples[index].features[feature], samples[index].label, index};
    }
    qsort(sorted, count, sizeof(sorted[0]), compare_sorted);

    double parent = impurity(positives, count);
    double best_gain = 0.0;
    size_t left_positives = 0;
    Candidate best = {0};
    for (size_t i = 0; i + 1 < count; ++i) {
        left_positives += sorted[i].label == 1;
        size_t left_count = i + 1;
        size_t right_count = count - left_count;
        if (sorted[i].value == sorted[i + 1].value) {
            continue;
        }
        double threshold = sorted[i].value +
                           (sorted[i + 1].value - sorted[i].value) / 2.0;
        if (threshold >= sorted[i + 1].value) {
            continue;
        }
        size_t right_positives = positives - left_positives;
        double child_impurity =
            ((double)left_count * impurity(left_positives, left_count) +
             (double)right_count * impurity(right_positives, right_count)) /
            (double)count;
        double gain = parent - child_impurity;
        if (gain > best_gain + 1e-12) {
            best = (Candidate){feature, threshold, gain};
            best_gain = gain;
        }
    }
    if (best_gain <= 0.0) {
        return false;
    }
    *result = best;
    return true;
}

/**
 * @brief 递归构建一棵随机森林成员树，每个节点随机挑选一个特征。
 * @param tree 正在写入的空树。
 * @param samples 当前成员树的 bootstrap 样本数组。
 * @param indices 当前节点的样本下标。
 * @param count 当前节点样本数量，大于零。
 * @param depth 根深度为零。
 * @param max_depth 最大切分深度。
 * @param random_state 可写且由该树独占的随机状态。
 * @param output 成功时写入节点下标。
 * @return 构建成功为 true；固定节点容量溢出或内部切分不一致时为 false。
 */
static bool build_tree_node(Tree *tree, const Sample *samples, const size_t *indices,
                            size_t count, size_t depth, size_t max_depth,
                            uint32_t *random_state, int *output)
{
    if (count == 0 || tree->node_count >= MAX_NODES) {
        return false;
    }
    size_t positives = 0;
    for (size_t i = 0; i < count; ++i) {
        positives += samples[indices[i]].label == 1;
    }
    int node_index = (int)tree->node_count++;
    tree->nodes[node_index] = (Node){.feature = -1,
                                    .left = -1,
                                    .right = -1,
                                    .label = positives > count - positives ? 1 : 0,
                                    .samples = count,
                                    .positives = positives};
    if (positives == 0 || positives == count || depth >= max_depth) {
        *output = node_index;
        return true;
    }

    /* 随机选择一个特征，构成教学版随机子空间；之后在该维枚举全部阈值。 */
    int feature = (int)(random_next(random_state) % FEATURE_COUNT);
    Candidate candidate;
    if (!find_candidate(samples, indices, count, positives, feature, &candidate)) {
        *output = node_index;
        return true;
    }

    size_t left_indices[SAMPLE_COUNT];
    size_t right_indices[SAMPLE_COUNT];
    size_t left_count = 0;
    size_t right_count = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t index = indices[i];
        if (samples[index].features[feature] <= candidate.threshold) {
            left_indices[left_count++] = index;
        } else {
            right_indices[right_count++] = index;
        }
    }
    if (left_count == 0 || right_count == 0) {
        return false;
    }
    int left = -1;
    int right = -1;
    if (!build_tree_node(tree, samples, left_indices, left_count, depth + 1, max_depth,
                         random_state, &left) ||
        !build_tree_node(tree, samples, right_indices, right_count, depth + 1, max_depth,
                         random_state, &right)) {
        return false;
    }
    tree->nodes[node_index].feature = feature;
    tree->nodes[node_index].threshold = candidate.threshold;
    tree->nodes[node_index].left = left;
    tree->nodes[node_index].right = right;
    *output = node_index;
    return true;
}

/**
 * @brief 构造每次都检查全部特征的确定性 CART 基线。
 * @param tree 正在写入的可写树。
 * @param samples 只读训练样本。
 * @param indices 当前节点样本下标。
 * @param count 当前节点样本数，大于零。
 * @param depth 当前深度，根为零。
 * @param max_depth 最大切分深度。
 * @param output 成功后写入当前节点索引。
 * @return 成功为 true；节点容量或切分不变量失败为 false。
 */
static bool build_cart_node(Tree *tree, const Sample *samples, const size_t *indices,
                            size_t count, size_t depth, size_t max_depth, int *output)
{
    if (count == 0 || tree->node_count >= MAX_NODES) {
        return false;
    }
    size_t positives = 0;
    for (size_t i = 0; i < count; ++i) {
        positives += samples[indices[i]].label == 1;
    }
    int node_index = (int)tree->node_count++;
    tree->nodes[node_index] = (Node){.feature = -1,
                                    .left = -1,
                                    .right = -1,
                                    .label = positives > count - positives ? 1 : 0,
                                    .samples = count,
                                    .positives = positives};
    if (positives == 0 || positives == count || depth >= max_depth) {
        *output = node_index;
        return true;
    }

    Candidate best = {0};
    bool found = false;
    for (int feature = 0; feature < FEATURE_COUNT; ++feature) {
        Candidate candidate;
        if (find_candidate(samples, indices, count, positives, feature, &candidate) &&
            (!found || candidate.gain > best.gain + 1e-12)) {
            best = candidate;
            found = true;
        }
    }
    if (!found) {
        *output = node_index;
        return true;
    }

    size_t left_indices[SAMPLE_COUNT];
    size_t right_indices[SAMPLE_COUNT];
    size_t left_count = 0;
    size_t right_count = 0;
    for (size_t i = 0; i < count; ++i) {
        size_t index = indices[i];
        if (samples[index].features[best.feature] <= best.threshold) {
            left_indices[left_count++] = index;
        } else {
            right_indices[right_count++] = index;
        }
    }
    if (left_count == 0 || right_count == 0) {
        return false;
    }
    int left = -1;
    int right = -1;
    if (!build_cart_node(tree, samples, left_indices, left_count, depth + 1,
                         max_depth, &left) ||
        !build_cart_node(tree, samples, right_indices, right_count, depth + 1,
                         max_depth, &right)) {
        return false;
    }
    tree->nodes[node_index].feature = best.feature;
    tree->nodes[node_index].threshold = best.threshold;
    tree->nodes[node_index].left = left;
    tree->nodes[node_index].right = right;
    *output = node_index;
    return true;
}

/**
 * @brief 用指定 bootstrap 样本和种子训练一个树成员。
 * @param member 可写成员；成功后包含树和原训练集的袋内标记。
 * @param training 原始只读训练集，恰含 SAMPLE_COUNT 条样本。
 * @param seed 每棵树独立的确定性随机种子。
 * @param max_depth 最大切分深度，范围 1..SAMPLE_COUNT。
 * @return 成功为 true；固定容量约束失败时 false。
 */
static bool member_train(ForestMember *member, const Sample training[SAMPLE_COUNT],
                         uint32_t seed, size_t max_depth)
{
    if (member == nullptr || training == nullptr || max_depth == 0 ||
        max_depth > SAMPLE_COUNT) {
        return false;
    }
    ForestMember next = {0};
    Sample bootstrap[SAMPLE_COUNT];
    size_t indices[SAMPLE_COUNT];
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        size_t selected = random_index(&seed, SAMPLE_COUNT);
        bootstrap[i] = training[selected];
        indices[i] = i;
        next.in_bag[selected] = true;
    }
    int root = -1;
    if (!build_tree_node(&next.tree, bootstrap, indices, SAMPLE_COUNT, 0, max_depth,
                         &seed, &root) || root != 0) {
        return false;
    }
    *member = next;
    return true;
}

/**
 * @brief 预测一条样本，输出叶类别。
 * @param tree 已训练且节点索引有效的只读树。
 * @param features 两个有限输入坐标，均在 [-1,1]。
 * @param output 有效输出地址；失败时不修改。
 * @return 预测成功为 true，否则为 false。
 */
static bool tree_predict(const Tree *tree, const double features[FEATURE_COUNT], int *output)
{
    if (tree == nullptr || features == nullptr || output == nullptr || tree->node_count == 0 ||
        !isfinite(features[0]) || !isfinite(features[1]) || fabs(features[0]) > 1.0 ||
        fabs(features[1]) > 1.0) {
        return false;
    }
    int index = 0;
    while (tree->nodes[index].feature >= 0) {
        const Node *node = &tree->nodes[index];
        index = features[node->feature] <= node->threshold ? node->left : node->right;
        if (index < 0 || (size_t)index >= tree->node_count) {
            return false;
        }
    }
    *output = tree->nodes[index].label;
    return true;
}

/**
 * @brief 对树投票并将类别票数转换为正类投票比例。
 * @param forest 至少含一棵有效成员树的只读森林。
 * @param features 两个有限且位于 [-1,1] 的输入坐标。
 * @param output 可写类别结果；平票选择 0，失败时不修改。
 * @param probability 可写正类投票比例结果；失败时不修改。
 * @return 所有成员预测成功为 true，否则为 false。
 */
static bool forest_predict(const Forest *forest, const double features[FEATURE_COUNT],
                           int *output, double *probability)
{
    if (forest == nullptr || forest->tree_count == 0 || output == nullptr ||
        probability == nullptr) {
        return false;
    }
    size_t votes[2] = {0, 0};
    for (size_t i = 0; i < forest->tree_count; ++i) {
        int label = -1;
        if (!tree_predict(&forest->members[i].tree, features, &label)) {
            return false;
        }
        ++votes[label];
    }
    int predicted = votes[1] > votes[0] ? 1 : 0;
    double positive_fraction = (double)votes[1] / (double)forest->tree_count;
    *output = predicted;
    *probability = positive_fraction;
    return true;
}

/**
 * @brief 训练一个确定性 CART 基线，使用全量训练数据和所有特征。
 * @param tree 可写输出树。
 * @param training 恰含 SAMPLE_COUNT 条合法训练样本。
 * @param max_depth 最大切分深度。
 * @return 成功为 true；输入约定违反时 false。
 */
static bool baseline_tree_train(Tree *tree, const Sample training[SAMPLE_COUNT],
                                size_t max_depth)
{
    size_t indices[SAMPLE_COUNT];
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        indices[i] = i;
    }
    Tree result = {0};
    int root = -1;
    if (!build_cart_node(&result, training, indices, SAMPLE_COUNT, 0, max_depth, &root)) {
        return false;
    }
    *tree = result;
    return true;
}

/**
 * @brief 训练森林并累计袋外投票；每个成员从独立子种子开始。
 * @param forest 可写输出森林；成功时拥有 TREE_COUNT 棵树。
 * @param training 恰含 SAMPLE_COUNT 个有限二维二分类样本。
 * @param seed 主种子，决定成员种子、bootstrap 与随机特征序列。
 * @param max_depth 每棵树的最大切分深度。
 * @param oob_votes 输出每个训练样本的两类袋外票数，恰有 SAMPLE_COUNT 项。
 * @return 全部成员训练并预测成功为 true，否则为 false。
 */
static bool forest_train(Forest *forest, const Sample training[SAMPLE_COUNT], uint32_t seed,
                         size_t max_depth, size_t oob_votes[SAMPLE_COUNT][2])
{
    if (forest == nullptr || training == nullptr || oob_votes == nullptr || max_depth == 0 ||
        max_depth > SAMPLE_COUNT) {
        return false;
    }
    Forest next = {0};
    size_t votes[SAMPLE_COUNT][2] = {0};
    next.tree_count = TREE_COUNT;
    for (size_t i = 0; i < TREE_COUNT; ++i) {
        uint32_t member_seed = random_next(&seed) ^ seed ^ (uint32_t)(i * UINT32_C(2654435761));
        if (!member_train(&next.members[i], training, member_seed, max_depth)) {
            return false;
        }
        for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
            if (next.members[i].in_bag[sample]) {
                continue;
            }
            int label = -1;
            if (!tree_predict(&next.members[i].tree, training[sample].features, &label)) {
                return false;
            }
            ++votes[sample][label];
        }
    }
    *forest = next;
    memcpy(oob_votes, votes, sizeof(votes));
    return true;
}

/**
 * @brief 计算单棵树在指定样本上的错误数。
 * @param tree 已训练的只读树。
 * @param samples 只读评估样本数组。
 * @param count 评估样本数。
 * @param metrics 可写错误数与样本数；成功后整体更新。
 * @return 全部预测成功为 true，否则为 false。
 */
static bool tree_evaluate(const Tree *tree, const Sample *samples, size_t count,
                          Metrics *metrics)
{
    if (tree == nullptr || samples == nullptr || metrics == nullptr) {
        return false;
    }
    Metrics result = {0};
    for (size_t i = 0; i < count; ++i) {
        int predicted = -1;
        if (!tree_predict(tree, samples[i].features, &predicted)) {
            return false;
        }
        result.errors += predicted != samples[i].label;
        ++result.count;
    }
    *metrics = result;
    return true;
}

/**
 * @brief 统计森林对一组样本的多数投票错误。
 * @param forest 已训练的只读森林。
 * @param samples 只读评估样本数组。
 * @param count 评估样本数。
 * @param metrics 可写错误数与样本数；失败时保持不变。
 * @return 全部预测成功为 true，否则为 false。
 */
static bool forest_evaluate(const Forest *forest, const Sample *samples, size_t count,
                            Metrics *metrics)
{
    if (forest == nullptr || samples == nullptr || metrics == nullptr) {
        return false;
    }
    Metrics result = {0};
    for (size_t i = 0; i < count; ++i) {
        int predicted = -1;
        double probability = 0.0;
        if (!forest_predict(forest, samples[i].features, &predicted, &probability)) {
            return false;
        }
        (void)probability;
        result.errors += predicted != samples[i].label;
        ++result.count;
    }
    *metrics = result;
    return true;
}

/**
 * @brief 从逐样本袋外投票中统计有覆盖的数量和分类错误数。
 * @param votes 两列分别为类别 0、1 的袋外票数，恰含 SAMPLE_COUNT 行。
 * @param metrics 可写错误数及有覆盖样本数；失败时不修改。
 * @return 至少一个样本有袋外票时为 true。
 */
static bool oob_evaluate(const size_t votes[SAMPLE_COUNT][2], const Sample training[SAMPLE_COUNT],
                         Metrics *metrics)
{
    if (votes == nullptr || training == nullptr || metrics == nullptr) {
        return false;
    }
    Metrics result = {0};
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        size_t total = votes[i][0] + votes[i][1];
        if (total == 0) {
            continue;
        }
        int predicted = votes[i][1] > votes[i][0] ? 1 : 0;
        result.errors += predicted != training[i].label;
        ++result.count;
    }
    if (result.count == 0) {
        return false;
    }
    *metrics = result;
    return true;
}

/**
 * @brief 生成可复现的圆形分类网格；可选地翻转少量训练标签。
 * @param samples 有效可写样本数组，至少 SAMPLE_COUNT 个。
 * @param offset 坐标网格偏移量，须处于 [-0.1,0.1]。
 * @param add_noise true 时按固定索引规则加入标签噪声。
 */
static void make_samples(Sample samples[SAMPLE_COUNT], double offset, bool add_noise)
{
    for (size_t row = 0; row < 8; ++row) {
        for (size_t column = 0; column < 8; ++column) {
            size_t index = row * 8 + column;
            double x = -0.875 + 0.25 * (double)column + offset;
            double y = -0.875 + 0.25 * (double)row + offset;
            int label = x * x + y * y < 0.55 ? 1 : 0;
            if (add_noise && (index * 29 + 7) % 13 == 0) {
                label = 1 - label;
            }
            samples[index] = (Sample){{x, y}, label};
        }
    }
}

/**
 * @brief 运行三种固定随机种子的树与森林对比实验。
 * @return 全部训练、评估和输出成功为 0，否则为 1。
 */
int main(void)
{
    const uint32_t seeds[EXPERIMENT_SEEDS] = {2025, 2026, 2027};
    Sample training[SAMPLE_COUNT];
    Sample test[SAMPLE_COUNT];
    make_samples(training, 0.0, true);
    make_samples(test, 0.0625, false);

    puts("任务：圆形区域内为类别1；训练标签加入固定噪声，测试标签保持干净。");
    puts("每个森林含31棵深度6的树；每棵树对64个训练样本有放回抽样，"
         "每个节点随机选择一个坐标特征。");
    size_t positive_count = 0;
    size_t test_positive_count = 0;
    for (size_t i = 0; i < SAMPLE_COUNT; ++i) {
        positive_count += training[i].label == 1;
        test_positive_count += test[i].label == 1;
    }
    int majority_label = positive_count > SAMPLE_COUNT - positive_count ? 1 : 0;
    size_t majority_train_errors =
        majority_label == 1 ? SAMPLE_COUNT - positive_count : positive_count;
    size_t majority_test_errors =
        majority_label == 1 ? SAMPLE_COUNT - test_positive_count : test_positive_count;
    printf("训练集多数类基线错误：训练 %zu/%d，测试 %zu/%d\n", majority_train_errors,
           SAMPLE_COUNT, majority_test_errors, SAMPLE_COUNT);
    puts("种子 | 单树训练/测试错误 | 森林训练/测试错误 | 袋外错误/覆盖样本");
    double forest_accuracy_sum = 0.0;
    double oob_accuracy_sum = 0.0;
    for (size_t run = 0; run < EXPERIMENT_SEEDS; ++run) {
        Tree baseline = {0};
        Forest forest = {0};
        size_t votes[SAMPLE_COUNT][2] = {0};
        Metrics baseline_train = {0};
        Metrics baseline_test = {0};
        Metrics forest_train_metrics = {0};
        Metrics forest_test = {0};
        Metrics oob = {0};
        if (!baseline_tree_train(&baseline, training, 6) ||
            !forest_train(&forest, training, seeds[run], 6, votes) ||
            !tree_evaluate(&baseline, training, SAMPLE_COUNT, &baseline_train) ||
            !tree_evaluate(&baseline, test, SAMPLE_COUNT, &baseline_test) ||
            !forest_evaluate(&forest, training, SAMPLE_COUNT, &forest_train_metrics) ||
            !forest_evaluate(&forest, test, SAMPLE_COUNT, &forest_test) ||
            !oob_evaluate(votes, training, &oob)) {
            fputs("随机森林训练或评估失败\n", stderr);
            return 1;
        }
        printf("%u | %zu/%zu, %zu/%zu | %zu/%zu, %zu/%zu | %zu/%zu\n", seeds[run],
               baseline_train.errors, baseline_train.count, baseline_test.errors,
               baseline_test.count, forest_train_metrics.errors, forest_train_metrics.count,
               forest_test.errors, forest_test.count, oob.errors, oob.count);
        forest_accuracy_sum += (double)(forest_test.count - forest_test.errors) /
                               (double)forest_test.count;
        oob_accuracy_sum += (double)(oob.count - oob.errors) / (double)oob.count;
    }
    printf("森林测试准确率均值：%.1f%%；袋外准确率均值：%.1f%%\n",
           100.0 * forest_accuracy_sum / EXPERIMENT_SEEDS,
           100.0 * oob_accuracy_sum / EXPERIMENT_SEEDS);
    return !ferror(stdout) ? 0 : 1;
}

