/** @file @brief 固定网格上的 BFS、Dijkstra 与 A* 寻路核心，不依赖绘图或 SDL。 */
#ifndef CDEMO_PATHFINDING_H
#define CDEMO_PATHFINDING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    PATHFINDING_WIDTH = 16,
    PATHFINDING_HEIGHT = 10,
    PATHFINDING_CELLS = PATHFINDING_WIDTH * PATHFINDING_HEIGHT
};

typedef enum {
    PATHFINDING_BFS,
    PATHFINDING_DIJKSTRA,
    PATHFINDING_A_STAR
} PathfindingAlgorithm;

typedef enum {
    PATHFINDING_RUNNING,
    PATHFINDING_FOUND,
    PATHFINDING_NO_PATH
} PathfindingPhase;

/** 每格代价为零表示墙；1..255 表示进入该格的移动代价。 */
typedef struct {
    uint8_t cost[PATHFINDING_CELLS];
} PathfindingGrid;

/** 搜索状态由核心独占维护，调用者可读取数组用于显示和教学观察。 */
typedef struct {
    PathfindingAlgorithm algorithm;
    PathfindingPhase phase;
    int start;
    int goal;
    int current;
    uint8_t minimum_cost;
    bool discovered[PATHFINDING_CELLS];
    bool expanded[PATHFINDING_CELLS];
    bool path[PATHFINDING_CELLS];
    int parent[PATHFINDING_CELLS];
    int frontier[PATHFINDING_CELLS];
    size_t head;
    size_t tail;
    uint32_t best_cost[PATHFINDING_CELLS];
    size_t expanded_count;
    size_t path_length;
    uint32_t path_cost;
} PathfindingSearch;

/**
 * @brief 初始化一轮搜索。
 * @param grid 固定尺寸网格；搜索期间不得修改，零代价格视为墙。
 * @param start 起点编号，范围为 [0, PATHFINDING_CELLS)，且须可通行。
 * @param goal 终点编号，范围为 [0, PATHFINDING_CELLS)，且须可通行。
 * @param algorithm BFS、Dijkstra 或 A*。
 * @param search 有效输出地址；成功后整体替换，不转移网格所有权。
 * @return 参数和端点有效时为 true；失败时不修改 search。
 */
bool pathfinding_start(const PathfindingGrid *grid, int start, int goal,
                       PathfindingAlgorithm algorithm, PathfindingSearch *search);

/**
 * @brief 推进一个搜索步骤，最多展开一个格子。
 * @param grid pathfinding_start 使用的同一张只读网格。
 * @param search 已初始化的可读写搜索状态。
 * @note 找到终点、无路或参数为空时不再扩展；首次成功时填好 path 和指标。
 */
void pathfinding_step(const PathfindingGrid *grid, PathfindingSearch *search);

#endif
