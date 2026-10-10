/** @file @brief 固定尺寸迷宫：DFS 回溯生成，BFS/DFS 逐步寻路，无 SDL 依赖。 */
#ifndef CDEMO_MAZE_H
#define CDEMO_MAZE_H
#include "graphics.h"
#include <stddef.h>

enum {
    MAZE_WIDTH = 39, MAZE_HEIGHT = 29, MAZE_CELLS = MAZE_WIDTH * MAZE_HEIGHT,
    MAZE_CELL_SIZE = 8, MAZE_MARGIN = 4, MAZE_HZ = 120,
    MAZE_START = MAZE_WIDTH + 1,
    MAZE_GOAL = (MAZE_HEIGHT - 2) * MAZE_WIDTH + MAZE_WIDTH - 2
};
typedef enum { MAZE_BFS, MAZE_DFS } MazeAlgorithm;
typedef enum { MAZE_GENERATING, MAZE_SEARCHING, MAZE_FOUND, MAZE_NO_PATH } MazePhase;

/**
 * 所有数组由世界自己拥有，无动态内存。编号 id = y * MAZE_WIDTH + x。
 * 调用者先 reset，再 step；除测试构造地图外，不应手动修改内部状态。
 */
typedef struct {
    bool open[MAZE_CELLS]; /* true 为通道，false 为墙。 */
    int stack[MAZE_CELLS]; /* 生成算法的 DFS 回溯栈，内容为奇数坐标的房间编号。 */
    size_t depth;
    uint32_t seed, rng; /* 初始种子与当前独立随机流状态。 */
    MazeAlgorithm algorithm;
    MazePhase phase;
    bool discovered[MAZE_CELLS], expanded[MAZE_CELLS], path[MAZE_CELLS];
    int parent[MAZE_CELLS]; /* 首次发现时记录前驱，起点前驱为 -1。 */
    int frontier[MAZE_CELLS]; /* BFS 的 FIFO 队列，或 DFS 的 LIFO 栈。 */
    size_t head, tail;
    size_t visited_count, path_length; /* 已展开节点数；最终路径的边数，不是节点数。 */
    int current; /* 当前房间或当前展开格子；-1 表示没有。 */
} MazeWorld;

/**
 * @brief 全部重置并从起点开始生成迷宫。
 * @param world 有效的可写世界，旧内容被替换。
 * @param seed 随机种子，所有 uint32_t 值有效；同种子同操作可重现。
 * @param algorithm 生成完成后的寻路算法；未知值使用 BFS。
 */
void maze_reset(MazeWorld *world, uint32_t seed, MazeAlgorithm algorithm);
/**
 * @brief 推进一步：生成阶段挖通或回退一次，寻路阶段展开一个节点。
 * @param world 已初始化的可读写世界；完成/无路状态下无操作。
 * @note 生成结束自动开始寻路。节点在入队/入栈时标记，避免重复加入。
 */
void maze_step(MazeWorld *world);
/**
 * @brief 在现有地图上重新寻路，清除旧搜索和路径状态。
 * @param world 已初始化的可读写世界，地图保持不变。
 * @param algorithm BFS 或 DFS。
 * @return 生成尚未完成或算法无效时为 false，世界不变；否则为 true。
 */
bool maze_restart_search(MazeWorld *world, MazeAlgorithm algorithm);
/**
 * @brief 查询一个坐标是否为通道，地图外视为墙。
 * @param world 已初始化的只读世界。
 * @param x 列坐标，有效范围 [0,MAZE_WIDTH)。
 * @param y 行坐标，有效范围 [0,MAZE_HEIGHT)。
 * @return 坐标合法且为通道时为 true。
 */
bool maze_is_open(const MazeWorld *world, int x, int y);
/**
 * @brief 清屏并绘制通道、搜索状态、当前格和最终路径。
 * @param world 已初始化的只读世界。
 * @param canvas 有效的可写画布，所有权不变。
 */
void maze_draw(const MazeWorld *world, GfxCanvas *canvas);
#endif
