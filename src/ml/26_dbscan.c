/**
 * @file
 * @brief 用纯 C 实现二维 DBSCAN，展示核心点、边界点和噪声点。
 */
#include <math.h>
#include <stdio.h>

enum {
    MAX_POINT_COUNT = 16,
    LABEL_UNVISITED = -2,
    LABEL_NOISE = -1
};

typedef struct {
    double x;
    double y;
} Point;

/**
 * @brief 统计一个点闭邻域中的样本数，包含点自身。
 * @param points 至少 count 个有限二维点的只读数组。
 * @param count 点数，范围为 1..MAX_POINT_COUNT。
 * @param center 邻域中心点的数组下标。
 * @param epsilon 有限正数邻域半径。
 * @param neighbors 至少 count 个可写下标位置。
 * @param neighbor_count 有效可写计数；成功时写入邻域点数量。
 * @return 参数有效时为 true；输出下标及数量随成功结果写入。
 */
static bool region_query(const Point *points, size_t count, size_t center, double epsilon,
                         size_t neighbors[MAX_POINT_COUNT], size_t *neighbor_count)
{
    if (points == nullptr || neighbors == nullptr || neighbor_count == nullptr || center >= count ||
        count == 0 || count > MAX_POINT_COUNT || !isfinite(epsilon) || epsilon <= 0.0) {
        return false;
    }

    size_t found[MAX_POINT_COUNT] = {0};
    size_t found_count = 0;
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(points[index].x) || !isfinite(points[index].y)) {
            return false;
        }
        double distance = hypot(points[center].x - points[index].x,
                                points[center].y - points[index].y);
        if (distance <= epsilon) {
            found[found_count] = index;
            ++found_count;
        }
    }
    for (size_t index = 0; index < found_count; ++index) {
        neighbors[index] = found[index];
    }
    *neighbor_count = found_count;
    return true;
}

/**
 * @brief 检查输入点是否有限，避免无效坐标进入距离判断。
 * @param points 待检查的只读点数组。
 * @param count 点数，范围为 1..MAX_POINT_COUNT。
 * @return 所有坐标均为有限数时为 true。
 */
static bool points_are_finite(const Point *points, size_t count)
{
    if (points == nullptr || count == 0 || count > MAX_POINT_COUNT) {
        return false;
    }
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(points[index].x) || !isfinite(points[index].y)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief 根据密度可达关系聚类，邻域判定使用闭区间距离 <= epsilon。
 * @param points count 个有限二维点的只读数组。
 * @param count 点数，范围为 1..MAX_POINT_COUNT。
 * @param epsilon 有限正数邻域半径。
 * @param min_points 成为核心点所需的最小邻域样本数，包含点自身。
 * @param labels 至少 count 个可写标签；噪声为 -1，簇编号从 0 开始。
 * @param cluster_count 有效可写结果；成功时写入簇数。
 * @return 参数和输入有效时为 true；失败时 labels 与 cluster_count 均不修改。
 */
static bool dbscan_cluster(const Point *points, size_t count, double epsilon, size_t min_points,
                           int labels[MAX_POINT_COUNT], size_t *cluster_count)
{
    if (!points_are_finite(points, count) || labels == nullptr || cluster_count == nullptr ||
        !isfinite(epsilon) || epsilon <= 0.0 || min_points == 0 || min_points > count) {
        return false;
    }

    int found_labels[MAX_POINT_COUNT];
    bool visited[MAX_POINT_COUNT] = {false};
    for (size_t index = 0; index < count; ++index) {
        found_labels[index] = LABEL_UNVISITED;
    }

    size_t found_cluster_count = 0;
    for (size_t point_index = 0; point_index < count; ++point_index) {
        if (visited[point_index]) {
            continue;
        }
        visited[point_index] = true;

        size_t initial_neighbors[MAX_POINT_COUNT] = {0};
        size_t initial_count = 0;
        if (!region_query(points, count, point_index, epsilon, initial_neighbors, &initial_count)) {
            return false;
        }
        if (initial_count < min_points) {
            found_labels[point_index] = LABEL_NOISE;
            continue;
        }

        /* 核心点创建新簇；队列继续寻找密度可达的核心点。 */
        int current_cluster = (int)found_cluster_count;
        ++found_cluster_count;
        size_t queue[MAX_POINT_COUNT] = {0};
        bool enqueued[MAX_POINT_COUNT] = {false};
        size_t queue_count = 0;
        for (size_t neighbor = 0; neighbor < initial_count; ++neighbor) {
            size_t neighbor_index = initial_neighbors[neighbor];
            if (!enqueued[neighbor_index]) {
                queue[queue_count] = neighbor_index;
                ++queue_count;
                enqueued[neighbor_index] = true;
            }
            if (found_labels[neighbor_index] == LABEL_UNVISITED ||
                found_labels[neighbor_index] == LABEL_NOISE) {
                found_labels[neighbor_index] = current_cluster;
            }
        }

        /* 一个先被标为噪声的点，若后来由核心点触达，就成为边界点。 */
        for (size_t queue_head = 0; queue_head < queue_count; ++queue_head) {
            size_t candidate = queue[queue_head];
            if (!visited[candidate]) {
                visited[candidate] = true;
                size_t candidate_neighbors[MAX_POINT_COUNT] = {0};
                size_t candidate_count = 0;
                if (!region_query(points, count, candidate, epsilon, candidate_neighbors,
                                  &candidate_count)) {
                    return false;
                }
                if (candidate_count >= min_points) {
                    for (size_t neighbor = 0; neighbor < candidate_count; ++neighbor) {
                        size_t neighbor_index = candidate_neighbors[neighbor];
                        if (!enqueued[neighbor_index]) {
                            queue[queue_count] = neighbor_index;
                            ++queue_count;
                            enqueued[neighbor_index] = true;
                        }
                    }
                }
            }
            if (found_labels[candidate] == LABEL_UNVISITED ||
                found_labels[candidate] == LABEL_NOISE) {
                found_labels[candidate] = current_cluster;
            }
        }
    }

    for (size_t index = 0; index < count; ++index) {
        labels[index] = found_labels[index];
    }
    *cluster_count = found_cluster_count;
    return true;
}

/**
 * @brief 展示两组密集点、一组边界点和孤立噪声点的聚类结果。
 * @return 聚类成功时返回 0，否则返回非零值。
 */
int main(void)
{
    const Point points[] = {
        {1.50, 0.00}, {0.00, 0.00}, {0.70, 0.00}, {0.35, 0.60}, {0.35, -0.60},
        {7.50, 0.00}, {6.00, 0.00}, {6.70, 0.00}, {6.35, 0.60}, {6.35, -0.60},
        {3.50, 3.50}};
    const size_t point_count = sizeof(points) / sizeof(points[0]);
    int labels[MAX_POINT_COUNT] = {0};
    size_t cluster_count = 0;
    if (!dbscan_cluster(points, point_count, 0.9, 4, labels, &cluster_count)) {
        return 1;
    }

    printf("DBSCAN：epsilon=0.90，min_points=4（邻域计数包含自身）\n");
    printf("找到 %zu 个簇；簇编号从 0 开始，-1 表示噪声\n", cluster_count);
    for (size_t index = 0; index < point_count; ++index) {
        printf("点 %2zu (%5.2f, %5.2f) -> ", index, points[index].x, points[index].y);
        if (labels[index] == LABEL_NOISE) {
            puts("噪声");
        } else {
            printf("簇 %d\n", labels[index]);
        }
    }
    return 0;
}
