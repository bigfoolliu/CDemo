/** @file @brief 验证 BFS、Dijkstra 与 A* 的寻路结果和边界。 */
#include "../src/graphics/pathfinding.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/**
 * @brief 完成有限网格上的搜索，防止测试错误时无限循环。
 * @param grid 当前只读网格。
 * @param search 已初始化的可读写搜索状态。
 */
static void finish_search(const PathfindingGrid *grid, PathfindingSearch *search)
{
    for (size_t step = 0; step <= PATHFINDING_CELLS &&
                           search->phase == PATHFINDING_RUNNING;
         ++step) {
        pathfinding_step(grid, search);
    }
    assert(search->phase == PATHFINDING_FOUND || search->phase == PATHFINDING_NO_PATH);
}

/**
 * @brief 校验找到路径的前驱链、相邻关系、移动步数和进入格子代价。
 * @param grid 当前只读网格。
 * @param search 已完成且找到路径的只读搜索。
 */
static void check_path(const PathfindingGrid *grid, const PathfindingSearch *search)
{
    assert(search->phase == PATHFINDING_FOUND);
    assert(search->path[search->start] && search->path[search->goal]);
    size_t length = 0;
    uint32_t cost = 0;
    int cell = search->goal;
    while (cell != search->start) {
        int previous = search->parent[cell];
        assert(previous >= 0 && previous < PATHFINDING_CELLS);
        assert(grid->cost[cell] != 0 && search->path[cell]);
        int dx = cell % PATHFINDING_WIDTH - previous % PATHFINDING_WIDTH;
        int dy = cell / PATHFINDING_WIDTH - previous / PATHFINDING_WIDTH;
        assert((dx == 0 && (dy == -1 || dy == 1)) ||
               (dy == 0 && (dx == -1 || dx == 1)));
        cost += grid->cost[cell];
        ++length;
        assert(length < PATHFINDING_CELLS);
        cell = previous;
    }
    assert(search->parent[search->start] == -1);
    assert(length == search->path_length);
    assert(cost == search->path_cost);
}

/**
 * @brief 比较搜索结构的值字段，忽略 C 结构体填充字节。
 * @param first 第一个只读搜索对象。
 * @param second 第二个只读搜索对象。
 * @return 算法状态、计数和全部数组相同时为 true。
 */
static bool searches_equal(const PathfindingSearch *first, const PathfindingSearch *second)
{
    return first->algorithm == second->algorithm && first->phase == second->phase &&
           first->start == second->start && first->goal == second->goal &&
           first->current == second->current && first->minimum_cost == second->minimum_cost &&
           first->head == second->head && first->tail == second->tail &&
           first->expanded_count == second->expanded_count &&
           first->path_length == second->path_length && first->path_cost == second->path_cost &&
           memcmp(first->discovered, second->discovered, sizeof first->discovered) == 0 &&
           memcmp(first->expanded, second->expanded, sizeof first->expanded) == 0 &&
           memcmp(first->path, second->path, sizeof first->path) == 0 &&
           memcmp(first->parent, second->parent, sizeof first->parent) == 0 &&
           memcmp(first->frontier, second->frontier, sizeof first->frontier) == 0 &&
           memcmp(first->best_cost, second->best_cost, sizeof first->best_cost) == 0;
}

/**
 * @brief 让短而昂贵的直路与长而便宜的绕路产生不同算法结果。
 * @return 构造好的测试网格，墙与地形代价写入固定数组。
 */
static PathfindingGrid make_weighted_grid(void)
{
    PathfindingGrid grid = {0};
    for (int x = 1; x <= 8; ++x) {
        grid.cost[4 * PATHFINDING_WIDTH + x] = x == 1 || x == 8 ? 1 : 9;
        grid.cost[2 * PATHFINDING_WIDTH + x] = 1;
    }
    grid.cost[3 * PATHFINDING_WIDTH + 1] = 1;
    grid.cost[3 * PATHFINDING_WIDTH + 8] = 1;
    return grid;
}

/** @brief 验证 BFS 最少步数，Dijkstra/A* 最低总代价和可重复结果。 */
static void test_weighted_paths(void)
{
    PathfindingGrid grid = make_weighted_grid();
    int start = 4 * PATHFINDING_WIDTH + 1;
    int goal = 4 * PATHFINDING_WIDTH + 8;
    PathfindingSearch bfs;
    PathfindingSearch dijkstra;
    PathfindingSearch astar;
    assert(pathfinding_start(&grid, start, goal, PATHFINDING_BFS, &bfs));
    assert(pathfinding_start(&grid, start, goal, PATHFINDING_DIJKSTRA, &dijkstra));
    assert(pathfinding_start(&grid, start, goal, PATHFINDING_A_STAR, &astar));
    assert(bfs.phase == PATHFINDING_RUNNING && bfs.expanded_count == 0);
    assert(dijkstra.phase == PATHFINDING_RUNNING && dijkstra.expanded_count == 0);
    assert(astar.phase == PATHFINDING_RUNNING && astar.expanded_count == 0);
    assert(!bfs.expanded[start] && !dijkstra.expanded[start] && !astar.expanded[start]);
    pathfinding_step(&grid, &astar);
    assert(astar.expanded[start] && astar.expanded_count == 1);
    finish_search(&grid, &bfs);
    finish_search(&grid, &dijkstra);
    finish_search(&grid, &astar);
    check_path(&grid, &bfs);
    check_path(&grid, &dijkstra);
    check_path(&grid, &astar);
    assert(bfs.path_length == 7 && bfs.path_cost == 55);
    assert(dijkstra.path_length == 11 && dijkstra.path_cost == 11);
    assert(astar.path_length == dijkstra.path_length && astar.path_cost == dijkstra.path_cost);
    assert(astar.expanded_count <= dijkstra.expanded_count);
    PathfindingSearch repeated;
    assert(pathfinding_start(&grid, start, goal, PATHFINDING_A_STAR, &repeated));
    finish_search(&grid, &repeated);
    assert(memcmp(repeated.path, astar.path, sizeof astar.path) == 0);
    assert(repeated.expanded_count == astar.expanded_count);
}

/** @brief 验证无路、起终点相同、非法参数及失败时输出不被改写。 */
static void test_boundaries(void)
{
    PathfindingGrid grid = {0};
    for (size_t cell = 0; cell < PATHFINDING_CELLS; ++cell) {
        grid.cost[cell] = 1;
    }
    for (int y = 0; y < PATHFINDING_HEIGHT; ++y) {
        grid.cost[y * PATHFINDING_WIDTH + PATHFINDING_WIDTH / 2] = 0;
    }
    PathfindingSearch search;
    for (int algorithm = PATHFINDING_BFS; algorithm <= PATHFINDING_A_STAR; ++algorithm) {
        assert(pathfinding_start(&grid, 0, PATHFINDING_WIDTH - 1,
                                 (PathfindingAlgorithm)algorithm, &search));
        finish_search(&grid, &search);
        assert(search.phase == PATHFINDING_NO_PATH && search.path_cost == 0);
    }
    grid.cost[0] = 1;
    assert(pathfinding_start(&grid, 0, 0, PATHFINDING_A_STAR, &search));
    finish_search(&grid, &search);
    assert(search.phase == PATHFINDING_FOUND && search.path_length == 0 && search.path_cost == 0);
    check_path(&grid, &search);

    search = (PathfindingSearch){.algorithm = PATHFINDING_A_STAR,
                                 .phase = PATHFINDING_FOUND,
                                 .start = 12,
                                 .goal = 19,
                                 .current = 17,
                                 .minimum_cost = 7,
                                 .head = 3,
                                 .tail = 4,
                                 .expanded_count = 6,
                                 .path_length = 8,
                                 .path_cost = UINT32_C(1234567)};
    search.discovered[0] = true;
    search.expanded[1] = true;
    search.path[2] = true;
    search.parent[3] = 9;
    search.frontier[0] = 10;
    search.best_cost[4] = 81;
    PathfindingSearch unchanged = search;
    assert(!pathfinding_start(&grid, -1, 0, PATHFINDING_BFS, &search));
    assert(searches_equal(&search, &unchanged));
    assert(!pathfinding_start(&grid, 0, 0, (PathfindingAlgorithm)99, &search));
    assert(searches_equal(&search, &unchanged));
    grid.cost[0] = 0;
    assert(!pathfinding_start(&grid, 0, 1, PATHFINDING_BFS, &search));
    assert(searches_equal(&search, &unchanged));
    assert(!pathfinding_start(nullptr, 0, 1, PATHFINDING_BFS, &search));
    assert(!pathfinding_start(&grid, 0, 1, PATHFINDING_BFS, nullptr));
}

/**
 * @brief 运行寻路算法核心测试。
 * @return 所有断言通过时返回零。
 */
int main(void)
{
    test_weighted_paths();
    test_boundaries();
    puts("寻路测试通过：BFS 步数、Dijkstra/A* 加权最优、无路、边界和稳定性。");
    return 0;
}
