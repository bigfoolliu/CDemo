/** @file
 * @brief 以可逐步观察的方式实现无权 BFS、加权 Dijkstra 与启发式 A*。
 * 搜索不读写画布，也不依赖窗口；UI 通过 PathfindingSearch 读取状态。
 */
#include "pathfinding.h"
#include <limits.h>
#include <string.h>

/**
 * @brief 判断算法枚举是否受支持。
 * @param algorithm 待检查的算法值。
 * @return 匹配 BFS、Dijkstra 或 A* 时为 true。
 */
static bool algorithm_is_valid(PathfindingAlgorithm algorithm)
{
    return algorithm == PATHFINDING_BFS || algorithm == PATHFINDING_DIJKSTRA ||
           algorithm == PATHFINDING_A_STAR;
}

/**
 * @brief 计算两格的曼哈顿距离，并乘以全图最小通行代价。
 * @param first 第一个格子编号。
 * @param second 第二个格子编号。
 * @param minimum_cost 图中最小的非零格代价。
 * @return A* 的可采纳启发式，不会高估剩余代价。
 */
static uint32_t heuristic(int first, int second, uint8_t minimum_cost)
{
    int first_x = first % PATHFINDING_WIDTH;
    int first_y = first / PATHFINDING_WIDTH;
    int second_x = second % PATHFINDING_WIDTH;
    int second_y = second / PATHFINDING_WIDTH;
    unsigned int dx = (unsigned int)(first_x > second_x ? first_x - second_x : second_x - first_x);
    unsigned int dy = (unsigned int)(first_y > second_y ? first_y - second_y : second_y - first_y);
    return (dx + dy) * minimum_cost;
}

/**
 * @brief 选出当前未展开节点中优先级最低的格子。
 * @param search 已初始化且尚在运行的搜索。
 * @return 下一个格子编号；没有可展开节点时返回 -1。
 * @note Dijkstra/A* 使用线性扫描，固定小网格下实现简单，整体最坏 O(V²)。
 */
static int select_lowest_priority(const PathfindingSearch *search)
{
    int selected = -1;
    uint32_t selected_priority = UINT32_MAX;
    for (int cell = 0; cell < PATHFINDING_CELLS; ++cell) {
        if (!search->discovered[cell] || search->expanded[cell]) {
            continue;
        }
        uint32_t priority = search->best_cost[cell];
        if (search->algorithm == PATHFINDING_A_STAR) {
            priority += heuristic(cell, search->goal, search->minimum_cost);
        }
        if (priority < selected_priority) {
            selected = cell;
            selected_priority = priority;
        }
    }
    return selected;
}

/**
 * @brief 从终点沿前驱回溯，计算路径边数和进入格子的代价总和。
 * @param grid 当前只读网格。
 * @param search 终点已展开且前驱链有效的可读写搜索。
 */
static void mark_path(const PathfindingGrid *grid, PathfindingSearch *search)
{
    int cell = search->goal;
    while (cell != -1) {
        search->path[cell] = true;
        if (cell != search->start) {
            ++search->path_length;
            search->path_cost += grid->cost[cell];
        }
        cell = search->parent[cell];
    }
    search->phase = PATHFINDING_FOUND;
}

/** @copydoc pathfinding_start */
bool pathfinding_start(const PathfindingGrid *grid, int start, int goal,
                       PathfindingAlgorithm algorithm, PathfindingSearch *search)
{
    if (grid == nullptr || search == nullptr || start < 0 || start >= PATHFINDING_CELLS ||
        goal < 0 || goal >= PATHFINDING_CELLS || !algorithm_is_valid(algorithm) ||
        grid->cost[start] == 0 || grid->cost[goal] == 0) {
        return false;
    }
    PathfindingSearch result = {.algorithm = algorithm,
                               .phase = PATHFINDING_RUNNING,
                               .start = start,
                               .goal = goal,
                               .current = -1,
                               .minimum_cost = UINT8_MAX};
    for (int cell = 0; cell < PATHFINDING_CELLS; ++cell) {
        result.parent[cell] = -1;
        result.best_cost[cell] = UINT32_MAX;
        if (grid->cost[cell] != 0 && grid->cost[cell] < result.minimum_cost) {
            result.minimum_cost = grid->cost[cell];
        }
    }
    result.discovered[start] = true;
    result.best_cost[start] = 0;
    if (algorithm == PATHFINDING_BFS) {
        result.frontier[result.tail++] = start;
    }
    *search = result;
    return true;
}

/** @copydoc pathfinding_step */
void pathfinding_step(const PathfindingGrid *grid, PathfindingSearch *search)
{
    if (grid == nullptr || search == nullptr || search->phase != PATHFINDING_RUNNING) {
        return;
    }
    int cell;
    if (search->algorithm == PATHFINDING_BFS) {
        if (search->head == search->tail) {
            search->phase = PATHFINDING_NO_PATH;
            return;
        }
        cell = search->frontier[search->head++];
    } else {
        cell = select_lowest_priority(search);
        if (cell == -1) {
            search->phase = PATHFINDING_NO_PATH;
            return;
        }
    }
    search->current = cell;
    search->expanded[cell] = true;
    ++search->expanded_count;
    if (cell == search->goal) {
        mark_path(grid, search);
        return;
    }
    const int dx[4] = {0, 1, 0, -1};
    const int dy[4] = {-1, 0, 1, 0};
    int x = cell % PATHFINDING_WIDTH;
    int y = cell / PATHFINDING_WIDTH;
    for (size_t direction = 0; direction < 4; ++direction) {
        int next_x = x + dx[direction];
        int next_y = y + dy[direction];
        if (next_x < 0 || next_x >= PATHFINDING_WIDTH || next_y < 0 ||
            next_y >= PATHFINDING_HEIGHT) {
            continue;
        }
        int next = next_y * PATHFINDING_WIDTH + next_x;
        if (grid->cost[next] == 0 || search->expanded[next]) {
            continue;
        }
        uint32_t step_cost = search->algorithm == PATHFINDING_BFS ? 1 : grid->cost[next];
        uint32_t candidate = search->best_cost[cell] + step_cost;
        if (search->discovered[next] && candidate >= search->best_cost[next]) {
            continue;
        }
        search->discovered[next] = true;
        search->best_cost[next] = candidate;
        search->parent[next] = cell;
        if (search->algorithm == PATHFINDING_BFS) {
            search->frontier[search->tail++] = next;
        }
    }
}
