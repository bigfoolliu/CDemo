/**
 * @file
 * @brief 二维二分类决策树：基尼增益、深度限制与可选决策边界导出。
 */
#include "../graphics/bmp_capture.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TREE_FEATURES = 2,
    TREE_MAX_SAMPLES = 64,
    TREE_MAX_NODES = 2 * TREE_MAX_SAMPLES - 1
};

typedef struct {
    double x[TREE_FEATURES];
    int label;
} TreeSample;

typedef struct {
    size_t max_depth;
    size_t min_leaf;
} TreeConfig;

typedef struct {
    int feature;
    double threshold;
    int left;
    int right;
    int label;
    size_t count;
    size_t positives;
    double gain;
} TreeNode;

typedef struct {
    TreeNode nodes[TREE_MAX_NODES];
    size_t count;
} DecisionTree;

typedef struct {
    double value;
    size_t index;
    int label;
} FeatureValue;

typedef struct {
    int feature;
    double threshold;
    double gain;
} Split;

/**
 * @brief 计算一个节点的基尼不纯度，二分类时为 2p(1-p)。
 * @param positives 正类样本数，不超过 count。
 * @param count 节点样本数，大于零。
 * @return 范围为 [0,0.5] 的不纯度。
 */
static double gini(size_t positives, size_t count)
{
    double p = (double)positives / (double)count;
    return 2.0 * p * (1.0 - p);
}

/**
 * @brief 按特征值与原始索引排序，保证相同输入产生稳定候选顺序。
 * @param left 指向一个只读 FeatureValue。
 * @param right 指向另一个只读 FeatureValue。
 * @return qsort 所需的负、零或正比较结果。
 */
static int compare_values(const void *left, const void *right)
{
    const FeatureValue *a = left;
    const FeatureValue *b = right;
    if (a->value < b->value) {
        return -1;
    }
    if (a->value > b->value) {
        return 1;
    }
    return (a->index > b->index) - (a->index < b->index);
}

/**
 * @brief 枚举两维相邻不同值的中点，选择严格正增益的最佳切分。
 * @param samples 只读完整样本数组，索引均有效。
 * @param indices 当前节点的 n 个只读样本索引。
 * @param n 当前节点样本数，1..TREE_MAX_SAMPLES。
 * @param positives 当前节点正类数，不超过 n。
 * @param min_leaf 子节点所需最小样本数，至少为 1。
 * @param result 有效可写结果；没有有效切分时不修改。
 * @return 找到有效切分为 true，否则为 false。
 */
static bool find_split(const TreeSample *samples, const size_t *indices, size_t n,
                       size_t positives, size_t min_leaf, Split *result)
{
    double parent_impurity = gini(positives, n);
    double best_gain = 0.0;
    Split best = {0};
    for (int feature = 0; feature < TREE_FEATURES; ++feature) {
        FeatureValue values[TREE_MAX_SAMPLES];
        for (size_t i = 0; i < n; ++i) {
            size_t index = indices[i];
            values[i] = (FeatureValue){samples[index].x[feature], index, samples[index].label};
        }
        qsort(values, n, sizeof(values[0]), compare_values);
        size_t left_positives = 0;
        for (size_t i = 0; i + 1 < n; ++i) {
            left_positives += values[i].label == 1;
            size_t left_count = i + 1;
            size_t right_count = n - left_count;
            if (left_count < min_leaf || right_count < min_leaf ||
                values[i].value == values[i + 1].value) {
                continue;
            }
            double threshold = values[i].value +
                               (values[i + 1].value - values[i].value) / 2.0;
            if (threshold >= values[i + 1].value) {
                continue;
            }
            size_t right_positives = positives - left_positives;
            double weighted = ((double)left_count * gini(left_positives, left_count) +
                               (double)right_count * gini(right_positives, right_count)) /
                              (double)n;
            double gain = parent_impurity - weighted;
            if (gain > best_gain + 1e-12) {
                best = (Split){feature, threshold, gain};
                best_gain = gain;
            }
        }
    }
    if (best_gain <= 0.0) {
        return false;
    }
    *result = best;
    return true;
}

/**
 * @brief 递归构造当前节点；每次有效切分都产生两个非空子节点。
 * @param tree 正在构造的可写树，不转移所有权。
 * @param samples 只读训练样本。
 * @param indices 当前节点的 n 个只读样本索引。
 * @param n 当前节点样本数，大于零。
 * @param depth 根节点深度为零。
 * @param config 深度与叶节点样本数限制。
 * @param output 成功时写入新节点索引；失败时不修改。
 * @return 构造成功为 true；容量或内部不变量失败为 false。
 */
static bool build_node(DecisionTree *tree, const TreeSample *samples, const size_t *indices,
                       size_t n, size_t depth, TreeConfig config, int *output)
{
    if (n == 0 || tree->count >= TREE_MAX_NODES) {
        return false;
    }
    size_t positives = 0;
    for (size_t i = 0; i < n; ++i) {
        positives += samples[indices[i]].label == 1;
    }
    int node_index = (int)tree->count++;
    tree->nodes[node_index] = (TreeNode){.feature = -1, .left = -1, .right = -1,
                                        .label = positives > n - positives ? 1 : 0,
                                        .count = n, .positives = positives};
    if (positives == 0 || positives == n || depth >= config.max_depth ||
        config.min_leaf > n / 2) {
        *output = node_index;
        return true;
    }
    Split split;
    if (!find_split(samples, indices, n, positives, config.min_leaf, &split)) {
        *output = node_index;
        return true;
    }

    size_t left_indices[TREE_MAX_SAMPLES];
    size_t right_indices[TREE_MAX_SAMPLES];
    size_t left_count = 0;
    size_t right_count = 0;
    for (size_t i = 0; i < n; ++i) {
        size_t index = indices[i];
        if (samples[index].x[split.feature] <= split.threshold) {
            left_indices[left_count++] = index;
        } else {
            right_indices[right_count++] = index;
        }
    }
    if (left_count < config.min_leaf || right_count < config.min_leaf) {
        return false;
    }
    int left = -1;
    int right = -1;
    if (!build_node(tree, samples, left_indices, left_count, depth + 1, config, &left) ||
        !build_node(tree, samples, right_indices, right_count, depth + 1, config, &right)) {
        return false;
    }
    tree->nodes[node_index].feature = split.feature;
    tree->nodes[node_index].threshold = split.threshold;
    tree->nodes[node_index].gain = split.gain;
    tree->nodes[node_index].left = left;
    tree->nodes[node_index].right = right;
    *output = node_index;
    return true;
}

/**
 * @brief 用有限、位于 [-1,1] 的二维样本训练决策树。
 * @param tree 可写树；首用时零初始化，成功替换旧树，失败保留旧树。
 * @param samples 至少 count 个只读样本，标签只能为 0 或 1。
 * @param count 样本数，1..TREE_MAX_SAMPLES。
 * @param config max_depth 为 0..TREE_MAX_SAMPLES；min_leaf 为 1..count。
 * @return 参数和构造有效为 true，否则为 false 且旧树不变。
 */
static bool tree_fit(DecisionTree *tree, const TreeSample *samples, size_t count,
                     TreeConfig config)
{
    if (tree == nullptr || samples == nullptr || count == 0 || count > TREE_MAX_SAMPLES ||
        config.max_depth > TREE_MAX_SAMPLES || config.min_leaf == 0 ||
        config.min_leaf > count) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (samples[i].label < 0 || samples[i].label > 1 ||
            !isfinite(samples[i].x[0]) || !isfinite(samples[i].x[1]) ||
            fabs(samples[i].x[0]) > 1.0 || fabs(samples[i].x[1]) > 1.0) {
            return false;
        }
    }
    size_t indices[TREE_MAX_SAMPLES];
    for (size_t i = 0; i < count; ++i) {
        indices[i] = i;
    }
    DecisionTree next = {0};
    int root = -1;
    if (!build_node(&next, samples, indices, count, 0, config, &root) || root != 0) {
        return false;
    }
    *tree = next;
    return true;
}

/**
 * @brief 按 `x[feature] <= threshold` 遍历树，平局叶节点预测为类别 0。
 * @param tree 已训练的只读树，节点索引在有效范围内。
 * @param x 两个有限、位于 [-1,1] 的输入坐标。
 * @param output 有效类别输出地址；失败时不修改。
 * @return 有效预测为 true，否则为 false。
 */
static bool tree_predict(const DecisionTree *tree, const double x[TREE_FEATURES], int *output)
{
    if (tree == nullptr || x == nullptr || output == nullptr || tree->count == 0 ||
        !isfinite(x[0]) || !isfinite(x[1]) || fabs(x[0]) > 1.0 || fabs(x[1]) > 1.0) {
        return false;
    }
    int index = 0;
    while (tree->nodes[index].feature >= 0) {
        const TreeNode *node = &tree->nodes[index];
        index = x[node->feature] <= node->threshold ? node->left : node->right;
        if (index < 0 || (size_t)index >= tree->count) {
            return false;
        }
    }
    *output = tree->nodes[index].label;
    return true;
}

/**
 * @brief 计算固定数据集上的错误数。
 * @param tree 已训练的只读树。
 * @param samples 至少 count 个有效只读样本。
 * @param count 样本数量。
 * @param errors 可写错误数，失败时不修改。
 * @return 全部预测成功为 true，否则为 false。
 */
static bool count_errors(const DecisionTree *tree, const TreeSample *samples, size_t count,
                         size_t *errors)
{
    if (samples == nullptr || errors == nullptr) {
        return false;
    }
    size_t wrong = 0;
    for (size_t i = 0; i < count; ++i) {
        int label = -1;
        if (!tree_predict(tree, samples[i].x, &label)) {
            return false;
        }
        wrong += label != samples[i].label;
    }
    *errors = wrong;
    return true;
}

/**
 * @brief 以缩进文本打印一个节点及其左右子树。
 * @param tree 已训练的只读树。
 * @param index 有效节点索引。
 * @param depth 当前打印深度，根节点为零。
 */
static void print_node(const DecisionTree *tree, int index, size_t depth)
{
    const TreeNode *node = &tree->nodes[index];
    printf("%*s", (int)(2 * depth), "");
    if (node->feature < 0) {
        printf("叶: 类别%d (%zu样本, %zu正类)\n", node->label, node->count,
               node->positives);
        return;
    }
    printf("x%d <= %.3f? 基尼增益 %.3f\n", node->feature, node->threshold, node->gain);
    print_node(tree, node->left, depth + 1);
    print_node(tree, node->right, depth + 1);
}

/**
 * @brief 绘制树桩与深树各占半幅的预测背景，并标出训练样本。
 * @param stump 已训练的深度 1 树。
 * @param deep 已训练的更深树。
 * @param samples 只读训练样本数组。
 * @param count 训练样本数。
 * @return BMP 独占创建成功为 true；预测或文件写入失败为 false。
 */
static bool export_boundary(const DecisionTree *stump, const DecisionTree *deep,
                            const TreeSample *samples, size_t count)
{
    GfxCanvas canvas;
    int panel_width = GFX_WIDTH / 2;
    for (int py = 0; py < GFX_HEIGHT; ++py) {
        double y = 1.0 - 2.0 * ((double)py + 0.5) / (double)GFX_HEIGHT;
        for (int px = 0; px < GFX_WIDTH; ++px) {
            int panel = px / panel_width;
            int local_x = px % panel_width;
            double x[2] = {2.0 * ((double)local_x + 0.5) / (double)panel_width - 1.0, y};
            int label = -1;
            const DecisionTree *tree = panel == 0 ? stump : deep;
            if (!tree_predict(tree, x, &label)) {
                return false;
            }
            canvas.pixels[py][px] = label == 0 ? gfx_rgb(180, 210, 245) :
                                                 gfx_rgb(250, 205, 160);
        }
    }
    for (int panel = 0; panel < 2; ++panel) {
        for (size_t i = 0; i < count; ++i) {
            int px = panel * panel_width +
                     (int)((samples[i].x[0] + 1.0) * (double)panel_width / 2.0);
            int py = (int)((1.0 - samples[i].x[1]) * (double)GFX_HEIGHT / 2.0);
            uint32_t color = samples[i].label == 0 ? gfx_rgb(25, 60, 145) :
                                                     gfx_rgb(170, 65, 20);
            gfx_circle(&canvas, px, py, 3, gfx_rgb(255, 255, 255));
            gfx_circle(&canvas, px, py, 2, color);
        }
    }
    gfx_line(&canvas, panel_width, 0, panel_width, GFX_HEIGHT - 1, gfx_rgb(40, 40, 40));
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(&canvas, "decision-tree", path)) {
        return false;
    }
    printf("决策边界已保存：%s（左：树桩；右：深度2）\n", path);
    return true;
}

/**
 * @brief 训练固定网格的树桩与深树，报告独立测试误差；显式参数导出 BMP。
 * @param argc 命令行参数个数，仅允许无参数或 --export。
 * @param argv 命令行参数数组，借用且不修改。
 * @return 训练、预测和可选导出成功为 0，否则为 1。
 */
int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--export") != 0)) {
        fprintf(stderr, "用法：%s [--export]\n", argv[0]);
        return 1;
    }
    const double grid[] = {-0.8, -0.4, 0.4, 0.8};
    TreeSample train[16];
    for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
            size_t index = row * 4 + column;
            train[index] = (TreeSample){{grid[column], grid[row]},
                                        grid[column] > 0.0 && grid[row] > 0.0 ? 1 : 0};
        }
    }
    const TreeSample test[] = {
        {{-0.6, -0.6}, 0}, {{-0.6, 0.6}, 0}, {{0.6, -0.6}, 0}, {{0.6, 0.6}, 1},
        {{-0.2, 0.2}, 0}, {{0.2, -0.2}, 0}, {{0.2, 0.2}, 1}, {{-0.2, -0.2}, 0}
    };
    DecisionTree stump = {0};
    DecisionTree deep = {0};
    if (!tree_fit(&stump, train, 16, (TreeConfig){1, 1}) ||
        !tree_fit(&deep, train, 16, (TreeConfig){2, 1})) {
        fputs("决策树训练失败\n", stderr);
        return 1;
    }
    size_t stump_train = 0;
    size_t stump_test = 0;
    size_t deep_train = 0;
    size_t deep_test = 0;
    if (!count_errors(&stump, train, 16, &stump_train) ||
        !count_errors(&stump, test, sizeof(test) / sizeof(test[0]), &stump_test) ||
        !count_errors(&deep, train, 16, &deep_train) ||
        !count_errors(&deep, test, sizeof(test) / sizeof(test[0]), &deep_test)) {
        fputs("决策树预测失败\n", stderr);
        return 1;
    }
    puts("深度1树桩：");
    print_node(&stump, 0, 0);
    puts("深度2决策树：");
    print_node(&deep, 0, 0);
    printf("树桩错误数：训练 %zu/16，测试 %zu/8\n", stump_train, stump_test);
    printf("深树错误数：训练 %zu/16，测试 %zu/8\n", deep_train, deep_test);
    if (argc == 2 && !export_boundary(&stump, &deep, train, 16)) {
        perror("导出决策边界失败");
        return 1;
    }
    return stump_train == 4 && stump_test == 2 && deep_train == 0 && deep_test == 0 &&
                   !ferror(stdout) ? 0 : 1;
}
