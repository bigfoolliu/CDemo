#include <assert.h>
#include <math.h>
#include <stdio.h>

enum { POINT_COUNT = 6, CLUSTER_COUNT = 2 };
typedef struct { double x; double y; } Point;

static double distance_squared(Point a, Point b)
{
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

static size_t nearest(Point point, const Point centers[CLUSTER_COUNT])
{
    return distance_squared(point, centers[0]) <= distance_squared(point, centers[1])
        ? 0 : 1; /* 等距离时选编号较小的簇。 */
}

static double objective(const Point points[POINT_COUNT], const Point centers[CLUSTER_COUNT])
{
    double total = 0;
    for (size_t i = 0; i < POINT_COUNT; ++i) {
        total += distance_squared(points[i], centers[nearest(points[i], centers)]);
    }
    return total;
}

static size_t train(const Point points[POINT_COUNT], Point centers[CLUSTER_COUNT])
{
    for (size_t iteration = 0; iteration < 100; ++iteration) {
        Point sums[CLUSTER_COUNT] = {0};
        size_t counts[CLUSTER_COUNT] = {0};
        for (size_t i = 0; i < POINT_COUNT; ++i) {
            size_t cluster = nearest(points[i], centers);
            sums[cluster].x += points[i].x;
            sums[cluster].y += points[i].y;
            ++counts[cluster];
        }
        double previous = objective(points, centers);
        double movement = 0;
        for (size_t cluster = 0; cluster < CLUSTER_COUNT; ++cluster) {
            if (counts[cluster] == 0) {
                continue; /* 空簇保留原中心，不除以零。 */
            }
            Point updated = {sums[cluster].x / (double)counts[cluster],
                             sums[cluster].y / (double)counts[cluster]};
            movement += distance_squared(centers[cluster], updated);
            centers[cluster] = updated;
        }
        double current = objective(points, centers);
        assert(isfinite(current) && current <= previous + 1e-9);
        if (movement < 1e-12) {
            return iteration + 1;
        }
    }
    return 100; /* 教学实现返回迭代数；工程接口应另报告是否收敛。 */
}

int main(void)
{
    /* 无类别标签，算法自行将点划分为两个簇。 */
    const Point points[POINT_COUNT] = {{0, 0}, {0, 1}, {1, 0}, {8, 8}, {8, 9}, {9, 8}};
    Point centers[CLUSTER_COUNT] = {points[0], points[3]};
    double initial = objective(points, centers);
    size_t iterations = train(points, centers);
    assert(iterations < 100 && objective(points, centers) < initial);
    assert(fabs(centers[0].x - 1.0 / 3) < 1e-9);
    assert(fabs(centers[1].x - 25.0 / 3) < 1e-9);
    for (size_t i = 0; i < POINT_COUNT; ++i) {
        assert(nearest(points[i], centers) == (i < 3 ? 0u : 1u));
    }
    const Point identical[POINT_COUNT] = {{1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    Point empty_test[CLUSTER_COUNT] = {{1, 1}, {100, 100}};
    (void)train(identical, empty_test);
    assert(empty_test[1].x == 100 && isfinite(objective(identical, empty_test)));
    printf("K-means：%zu 次迭代，中心0=(%.3f,%.3f)，中心1=(%.3f,%.3f)\n",
           iterations, centers[0].x, centers[0].y, centers[1].x, centers[1].y);
    return 0;
}
