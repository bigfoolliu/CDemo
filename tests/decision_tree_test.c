/**
 * @file
 * @brief 决策树的基尼增益、确定性、停止条件、边界与失败保留测试。
 */
#define main decision_tree_demo_main
#include "../src/ml/16_decision_tree.c"
#undef main
#include <assert.h>

/**
 * @brief 核对手算不纯度、首个分裂、树桩与两层树的分类差异。
 */
static void test_gain_and_depth(void)
{
    const TreeSample samples[] = {
        {{-0.8, -0.8}, 0}, {{-0.8, 0.8}, 0},
        {{0.8, -0.8}, 0}, {{0.8, 0.8}, 1}
    };
    DecisionTree stump = {0};
    DecisionTree deep = {0};
    assert(fabs(gini(1, 4) - 0.375) < 1e-12);
    assert(tree_fit(&stump, samples, 4, (TreeConfig){1, 1}));
    assert(stump.count == 3 && stump.nodes[0].feature == 0);
    assert(stump.nodes[0].threshold == 0.0);
    assert(fabs(stump.nodes[0].gain - 0.125) < 1e-12);
    assert(tree_fit(&deep, samples, 4, (TreeConfig){2, 1}));
    assert(deep.count == 5 && deep.nodes[0].feature == 0);
    double positive[2] = {0.8, 0.8};
    int label = -1;
    assert(tree_predict(&stump, positive, &label) && label == 0);
    assert(tree_predict(&deep, positive, &label) && label == 1);
    double boundary[2] = {0.0, 0.8};
    assert(tree_predict(&deep, boundary, &label) && label == 0);
    DecisionTree repeated = {0};
    assert(tree_fit(&repeated, samples, 4, (TreeConfig){2, 1}));
    assert(repeated.count == deep.count);
    assert(repeated.nodes[0].feature == deep.nodes[0].feature);
    assert(repeated.nodes[0].threshold == deep.nodes[0].threshold);
}

/**
 * @brief 检查纯节点、重复坐标与最小叶规模触发的停止条件。
 */
static void test_stopping_rules(void)
{
    const TreeSample pure[] = {{{-0.5, 0.0}, 1}, {{0.5, 0.0}, 1}};
    const TreeSample repeated[] = {{{0.0, 0.0}, 0}, {{0.0, 0.0}, 1}};
    const TreeSample mixed[] = {
        {{-0.8, -0.8}, 0}, {{-0.8, 0.8}, 0},
        {{0.8, -0.8}, 0}, {{0.8, 0.8}, 1}
    };
    DecisionTree tree = {0};
    assert(tree_fit(&tree, pure, 2, (TreeConfig){2, 1}));
    assert(tree.count == 1 && tree.nodes[0].label == 1);
    assert(tree_fit(&tree, repeated, 2, (TreeConfig){2, 1}));
    assert(tree.count == 1 && tree.nodes[0].label == 0);
    assert(tree_fit(&tree, mixed, 4, (TreeConfig){0, 1}));
    assert(tree.count == 1 && tree.nodes[0].label == 0);
    assert(tree_fit(&tree, mixed, 4, (TreeConfig){2, 3}));
    assert(tree.count == 1);
    assert(tree_fit(&tree, mixed, 4, (TreeConfig){2, 2}));
    assert(tree.count == 3);
}

/**
 * @brief 无效样本与参数不替换已训练树，无效预测不修改输出。
 */
static void test_invalid_inputs(void)
{
    const TreeSample valid[] = {{{-0.5, 0.0}, 0}, {{0.5, 0.0}, 1}};
    const TreeSample invalid_label[] = {{{-0.5, 0.0}, 0}, {{0.5, 0.0}, 2}};
    const TreeSample invalid_coordinate[] = {{{-0.5, 0.0}, 0}, {{NAN, 0.0}, 1}};
    DecisionTree tree = {0};
    assert(tree_fit(&tree, valid, 2, (TreeConfig){1, 1}));
    assert(!tree_fit(&tree, invalid_label, 2, (TreeConfig){1, 1}));
    assert(!tree_fit(&tree, invalid_coordinate, 2, (TreeConfig){1, 1}));
    assert(!tree_fit(&tree, valid, 2, (TreeConfig){1, 0}));
    assert(!tree_fit(&tree, valid, 2, (TreeConfig){1, 3}));
    assert(!tree_fit(&tree, valid, 0, (TreeConfig){1, 1}));
    assert(tree.count == 3 && tree.nodes[0].feature == 0);
    int label = 42;
    double bad[2] = {INFINITY, 0.0};
    assert(!tree_predict(&tree, bad, &label) && label == 42);
    assert(!tree_predict(&tree, nullptr, &label) && label == 42);
    double good[2] = {0.5, 0.0};
    assert(tree_predict(&tree, good, &label) && label == 1);
}

/**
 * @brief 运行所有断言。
 * @return 全部通过为 0；断言失败时终止程序。
 */
int main(void)
{
    test_gain_and_depth();
    test_stopping_rules();
    test_invalid_inputs();
    puts("决策树测试通过");
    return 0;
}
