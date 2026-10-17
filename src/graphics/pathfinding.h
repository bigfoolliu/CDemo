/** @file @brief 固定网格上的 BFS、Dijkstra 与 A* 寻路核心，不依赖绘图或 SDL。 */
#ifndef CDEMO_PATHFINDING_H
#define CDEMO_PATHFINDING_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    /* 格子按行优先编号：index = y * WIDTH + x。 */
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

/** 固定尺寸、按行优先编号的网格。 */
typedef struct {
    uint8_t cost[PATHFINDING_CELLS]; /* 进入格子的代价；0 表示墙。 */
} PathfindingGrid;

/** 搜索状态由核心独占维护，调用者可读取数组用于显示和教学观察。 */
typedef struct {
    PathfindingAlgorithm algorithm; /* 当前选择的 BFS、Dijkstra 或 A*。 */
    PathfindingPhase phase;         /* 搜索进行中、找到路径或确认无路。 */
    int start;                      /* 起点格编号。 */
    int goal;                       /* 目标格编号。 */
    int current;                    /* 最近一次展开的格子；开始前为 -1。 */
    uint8_t minimum_cost;           /* 全图最小通行代价，用于 A* 启发式。 */
    bool discovered[PATHFINDING_CELLS]; /* 已发现，等待展开或已经展开。 */
    bool expanded[PATHFINDING_CELLS];   /* 已从待处理集合取出并检查邻居。 */
    bool path[PATHFINDING_CELLS];       /* 最终路径上的格子。 */
    int parent[PATHFINDING_CELLS];      /* 最优前驱；起点和未发现格为 -1。 */
    int frontier[PATHFINDING_CELLS];    /* BFS 的先进先出队列。 */
    size_t head;                        /* BFS 下一个出队位置。 */
    size_t tail;                        /* BFS 下一个入队位置。 */
    uint32_t best_cost[PATHFINDING_CELLS]; /* 起点到该格的当前最小已知代价 g。 */
    size_t expanded_count;              /* 已展开格子数。 */
    size_t path_length;                 /* 路径边数，不包含起点本身。 */
    uint32_t path_cost;                 /* 进入路径格子的总代价，不计起点。 */
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
