/** @file
 * @brief DFS 生成一棵覆盖全部房间的树，再用队列或栈探索通道图。
 * 一次 step 只做一个可观察步骤，窗口控制与墙钟计时不参与算法。
 */
#include "maze.h"
#include <string.h>

_Static_assert(MAZE_WIDTH % 2 == 1 && MAZE_HEIGHT % 2 == 1, "迷宫尺寸必须为奇数");
_Static_assert(MAZE_MARGIN * 2 + MAZE_WIDTH * MAZE_CELL_SIZE == GFX_WIDTH, "宽度不匹配");
_Static_assert(MAZE_MARGIN * 2 + MAZE_HEIGHT * MAZE_CELL_SIZE == GFX_HEIGHT, "高度不匹配");

/** @copydoc maze_is_open */
bool maze_is_open(const MazeWorld *world, int x, int y)
{
    /* 先检查边界，再访问数组；上下左右不环绕连接。 */
    return x >= 0 && x < MAZE_WIDTH && y >= 0 && y < MAZE_HEIGHT && world->open[y * MAZE_WIDTH + x];
}

/** @copydoc maze_reset */
void maze_reset(MazeWorld *world, uint32_t seed, MazeAlgorithm algorithm)
{
    *world = (MazeWorld){.seed = seed,
                         .rng = seed,
                         .algorithm = algorithm == MAZE_DFS ? MAZE_DFS : MAZE_BFS,
                         .phase = MAZE_GENERATING,
                         .depth = 1,
                         .current = MAZE_START};
    world->open[MAZE_START] = true;
    world->stack[0] = MAZE_START;
}

/** @copydoc maze_restart_search */
bool maze_restart_search(MazeWorld *world, MazeAlgorithm algorithm)
{
    if (world->phase == MAZE_GENERATING || (algorithm != MAZE_BFS && algorithm != MAZE_DFS)) {
        return false;
    }
    memset(world->discovered, 0, sizeof world->discovered);
    memset(world->expanded, 0, sizeof world->expanded);
    memset(world->path, 0, sizeof world->path);
    for (int i = 0; i < MAZE_CELLS; ++i) {
        world->parent[i] = -1;
    }
    world->algorithm = algorithm;
    world->head = world->tail = 0;
    world->visited_count = world->path_length = 0;
    world->current = -1;
    /* 地图被测试或以后编辑功能修改时，封闭起点/终点直接报告无路径。 */
    world->phase =
        world->open[MAZE_START] && world->open[MAZE_GOAL] ? MAZE_SEARCHING : MAZE_NO_PATH;
    if (world->phase == MAZE_SEARCHING) {
        world->frontier[world->tail++] = MAZE_START;
        world->discovered[MAZE_START] = true;
    }
    return true;
}

/**
 * @brief 挖通一个新房间，或从死路退回上一个房间。
 * @param world 生成阶段的有效可读写世界。
 */
static void generate_step(MazeWorld *world)
{
    int cell = world->stack[world->depth - 1];
    int x = cell % MAZE_WIDTH, y = cell / MAZE_WIDTH;
    /* 奇数坐标是房间；相邻房间相隔两格，中间一格是可打通的墙。 */
    const int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
    int candidates[4], walls[4];
    size_t count = 0;
    for (int i = 0; i < 4; ++i) {
        int nx = x + 2 * dx[i], ny = y + 2 * dy[i];
        if (nx > 0 && nx < MAZE_WIDTH - 1 && ny > 0 && ny < MAZE_HEIGHT - 1 &&
            !world->open[ny * MAZE_WIDTH + nx]) {
            candidates[count] = ny * MAZE_WIDTH + nx;
            walls[count++] = (y + dy[i]) * MAZE_WIDTH + x + dx[i];
        }
    }
    if (count != 0) {
        /* 独立 LCG 用于视觉示例，不用于安全；取模选择可能有极小分布偏差。
         * 只连向尚未访问房间，保证不形成环，也不会覆盖栈容量。
         */
        world->rng = world->rng * UINT32_C(1664525) + UINT32_C(1013904223);
        size_t choice = (world->rng >> 8) % count;
        world->open[walls[choice]] = world->open[candidates[choice]] = true;
        world->stack[world->depth++] = candidates[choice];
        world->current = candidates[choice];
    } else {
        --world->depth; /* 无未访问邻居就回溯；显式栈代替递归调用栈。 */
        world->current = world->depth == 0 ? -1 : world->stack[world->depth - 1];
        if (world->depth == 0) {
            world->phase = MAZE_SEARCHING;
            maze_restart_search(world, world->algorithm);
        }
    }
}

/**
 * @brief 沿终点的前驱链回溯到起点并标记最终路径。
 * @param world 终点已发现且前驱链有效的可读写世界。
 */
static void mark_path(MazeWorld *world)
{
    int cell = MAZE_GOAL;
    while (cell != -1) {
        world->path[cell] = true;
        int previous = world->parent[cell];
        if (previous != -1) {
            ++world->path_length;
        }
        cell = previous;
    }
    world->phase = MAZE_FOUND;
}

/** @copydoc maze_step */
void maze_step(MazeWorld *world)
{
    if (world->phase == MAZE_GENERATING) {
        generate_step(world);
        return;
    }
    if (world->phase != MAZE_SEARCHING) {
        return;
    }
    if (world->head == world->tail) {
        world->phase = MAZE_NO_PATH;
        return;
    }
    /* BFS 从队首取，按距离逐层展开；DFS 从栈顶取，先深入再回退。
     * BFS 的 tail 单调增加，每个格最多入队一次，所以固定数组足够，无需环形队列。
     */
    int cell = world->algorithm == MAZE_BFS ? world->frontier[world->head++]
                                            : world->frontier[--world->tail];
    world->expanded[cell] = true;
    world->current = cell;
    ++world->visited_count;
    if (cell == MAZE_GOAL) {
        mark_path(world);
        return;
    }
    int x = cell % MAZE_WIDTH, y = cell / MAZE_WIDTH;
    const int dx[4] = {0, 1, 0, -1}, dy[4] = {-1, 0, 1, 0};
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i], ny = y + dy[i];
        if (!maze_is_open(world, nx, ny)) {
            continue;
        }
        int next = ny * MAZE_WIDTH + nx;
        if (world->discovered[next]) {
            continue;
        }
        /* 入队/入栈就标记，而非出队时才标记，防止有环地图重复加入节点。
         * 首次发现时只记录一次前驱：BFS 因 FIFO 顺序得到无权图的最短路。
         */
        world->discovered[next] = true;
        world->parent[next] = cell;
        world->frontier[world->tail++] = next;
    }
}

/** @copydoc maze_draw */
void maze_draw(const MazeWorld *world, GfxCanvas *canvas)
{
    gfx_clear(canvas, gfx_rgb(18, 23, 32));
    for (int id = 0; id < MAZE_CELLS; ++id) {
        uint32_t color = gfx_rgb(40, 48, 62);
        if (world->open[id]) {
            color = gfx_rgb(215, 220, 225);
        }
        if (world->discovered[id]) {
            color = gfx_rgb(90, 150, 230);
        }
        if (world->expanded[id]) {
            color = gfx_rgb(65, 85, 120);
        }
        if (world->path[id]) {
            color = gfx_rgb(255, 210, 70);
        }
        if (id == world->current && world->phase != MAZE_FOUND) {
            color = gfx_rgb(245, 150, 55);
        }
        if (id == MAZE_START && world->open[id]) {
            color = gfx_rgb(70, 220, 110);
        }
        if (id == MAZE_GOAL && world->open[id]) {
            color = gfx_rgb(245, 80, 90);
        }
        gfx_rect(canvas, MAZE_MARGIN + id % MAZE_WIDTH * MAZE_CELL_SIZE,
                 MAZE_MARGIN + id / MAZE_WIDTH * MAZE_CELL_SIZE, MAZE_CELL_SIZE - 1,
                 MAZE_CELL_SIZE - 1, true, color);
    }
}
