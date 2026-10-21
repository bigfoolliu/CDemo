/** @file @brief 验证 DBSCAN 的邻域、簇扩展和噪声重分类。 */
#define main dbscan_demo_main
#include "../src/ml/26_dbscan.c"
#undef main
#include <assert.h>

/**
 * @brief 检查半径边界包含等号，并且邻域计数包含中心点自身。
 */
static void test_region_query_includes_boundary_and_self(void)
{
    const Point points[] = {{0.0, 0.0}, {1.0, 0.0}, {1.01, 0.0}};
    size_t neighbors[MAX_POINT_COUNT] = {0};
    size_t neighbor_count = 0;
    assert(region_query(points, 3, 0, 1.0, neighbors, &neighbor_count));
    assert(neighbor_count == 2);
    assert(neighbors[0] == 0 && neighbors[1] == 1);
}

/**
 * @brief 检查先被访问为噪声的边界点可被核心点扩展重新归入簇。
 */
static void test_clusters_border_points_and_noise(void)
{
    const Point points[] = {
        {1.50, 0.00}, {0.00, 0.00}, {0.70, 0.00}, {0.35, 0.60}, {0.35, -0.60},
        {7.50, 0.00}, {6.00, 0.00}, {6.70, 0.00}, {6.35, 0.60}, {6.35, -0.60},
        {3.50, 3.50}};
    const size_t count = sizeof(points) / sizeof(points[0]);
    int labels[MAX_POINT_COUNT] = {0};
    size_t cluster_count = 99;
    assert(dbscan_cluster(points, count, 0.9, 4, labels, &cluster_count));
    assert(cluster_count == 2);
    assert(labels[0] == 0);
    for (size_t index = 1; index <= 4; ++index) {
        assert(labels[index] == 0);
    }
    assert(labels[5] == 1);
    for (size_t index = 6; index <= 9; ++index) {
        assert(labels[index] == 1);
    }
    assert(labels[10] == LABEL_NOISE);
}

/**
 * @brief 检查非法参数失败时不改写标签和簇数量。
 */
static void test_invalid_input_preserves_outputs(void)
{
    const Point points[] = {{0.0, 0.0}, {1.0, 1.0}};
    int labels[MAX_POINT_COUNT] = {0};
    labels[0] = 42;
    size_t cluster_count = 17;
    assert(!dbscan_cluster(points, 2, 0.0, 1, labels, &cluster_count));
    assert(labels[0] == 42 && cluster_count == 17);
    assert(!dbscan_cluster(points, 2, 1.0, 3, labels, &cluster_count));
    assert(labels[0] == 42 && cluster_count == 17);

    Point invalid[] = {{0.0, 0.0}, {NAN, 1.0}};
    assert(!dbscan_cluster(invalid, 2, 1.0, 1, labels, &cluster_count));
    assert(labels[0] == 42 && cluster_count == 17);
}

/**
 * @brief 运行 DBSCAN 的确定性行为断言。
 * @return 所有断言通过时返回 0。
 */
int main(void)
{
    test_region_query_includes_boundary_and_self();
    test_clusters_border_points_and_noise();
    test_invalid_input_preserves_outputs();
    puts("DBSCAN 测试通过");
    return 0;
}
