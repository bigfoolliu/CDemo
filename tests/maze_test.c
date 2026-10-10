/** @file @brief 验证生成地图的树结构、可重复性、寻路最短性与边界。 */
#include "../src/graphics/maze.h"
#include <assert.h>
#include <stdio.h>

/**
 * @brief 有界地生成地图，避免缺陷导致测试无限运行。
 * @param world 已 reset 的可读写世界。
 */
static void finish_generation(MazeWorld *world)
{
    size_t steps = 0;
    while (world->phase == MAZE_GENERATING && steps < MAZE_CELLS) {
        maze_step(world); ++steps;
        assert(world->depth <= ((MAZE_WIDTH - 1) / 2) * ((MAZE_HEIGHT - 1) / 2));
    }
    assert(world->phase == MAZE_SEARCHING);
    assert(steps == 2 * ((MAZE_WIDTH - 1) / 2) * ((MAZE_HEIGHT - 1) / 2) - 1);
}

/**
 * @brief 有界地完成搜索，校验前沿边界。
 * @param world 已开始搜索的可读写世界。
 */
static void finish_search(MazeWorld *world)
{
    for (int i = 0; i <= MAZE_CELLS && world->phase == MAZE_SEARCHING; ++i) {
        maze_step(world);
        assert(world->head <= world->tail && world->tail <= MAZE_CELLS);
    }
    assert(world->phase == MAZE_FOUND || world->phase == MAZE_NO_PATH);
}

/**
 * @brief 沿前驱验证路径只经过通道、每步相邻、长度与标记一致。
 * @param world 找到路径的只读世界。
 */
static void check_path(const MazeWorld *world)
{
    assert(world->phase == MAZE_FOUND && world->path[MAZE_START] && world->path[MAZE_GOAL]);
    int id = MAZE_GOAL;
    size_t length = 0, marked = 0;
    while (id != MAZE_START) {
        assert(id >= 0 && id < MAZE_CELLS && world->open[id] && world->path[id]);
        int parent = world->parent[id];
        assert(parent >= 0 && parent < MAZE_CELLS);
        int dx = id % MAZE_WIDTH - parent % MAZE_WIDTH;
        int dy = id / MAZE_WIDTH - parent / MAZE_WIDTH;
        assert((dx == 0 && (dy == 1 || dy == -1)) || (dy == 0 && (dx == 1 || dx == -1)));
        id = parent;
        assert(++length < MAZE_CELLS);
    }
    for (int i = 0; i < MAZE_CELLS; ++i) { marked += world->path[i]; }
    assert(length == world->path_length && marked == length + 1);
}

/** @brief 独立连通性遍历及边数检查，验证全部房间形成树且无越界通道。 */
static void test_generation(void)
{
    for (uint32_t seed = 0; seed < 50; ++seed) {
        MazeWorld world, same;
        maze_reset(&world, seed, MAZE_BFS); maze_reset(&same, seed, MAZE_BFS);
        assert(!maze_restart_search(&world, MAZE_DFS) && world.depth == 1);
        finish_generation(&world); finish_generation(&same);
        size_t open = 0, edges = 0;
        for (int y = 0; y < MAZE_HEIGHT; ++y) {
            for (int x = 0; x < MAZE_WIDTH; ++x) {
                int id = y * MAZE_WIDTH + x;
                assert(world.open[id] == same.open[id]);
                if (x == 0 || y == 0 || x == MAZE_WIDTH - 1 || y == MAZE_HEIGHT - 1) {
                    assert(!world.open[id]);
                }
                if (x % 2 == 1 && y % 2 == 1) { assert(world.open[id]); }
                if (world.open[id]) {
                    ++open;
                    edges += maze_is_open(&world, x + 1, y);
                    edges += maze_is_open(&world, x, y + 1);
                }
            }
        }
        assert(open == 2 * ((MAZE_WIDTH - 1) / 2) * ((MAZE_HEIGHT - 1) / 2) - 1 && edges == open - 1);
        bool seen[MAZE_CELLS] = {0};
        int queue[MAZE_CELLS]; size_t head = 0, tail = 1;
        queue[0] = MAZE_START; seen[MAZE_START] = true;
        while (head < tail) {
            int id = queue[head++], x = id % MAZE_WIDTH, y = id / MAZE_WIDTH;
            int nx[4] = {x, x + 1, x, x - 1}, ny[4] = {y - 1, y, y + 1, y};
            for (int i = 0; i < 4; ++i) {
                if (!maze_is_open(&world, nx[i], ny[i])) { continue; }
                int next = ny[i] * MAZE_WIDTH + nx[i];
                if (!seen[next]) { seen[next] = true; queue[tail++] = next; }
            }
        }
        assert(tail == open); /* 连通 + 边数为节点数减一 => 树，起终点路径唯一。 */
        finish_search(&world); check_path(&world);
        same = world;
        assert(maze_restart_search(&world, MAZE_DFS));
        finish_search(&world); check_path(&world);
        assert(world.path_length == same.path_length);
        for (int i = 0; i < MAZE_CELLS; ++i) { assert(world.path[i] == same.path[i]); }
        size_t visited = world.visited_count;
        maze_step(&world); assert(world.visited_count == visited); /* 结束态不再改变。 */
    }
    MazeWorld edge;
    maze_reset(&edge, UINT32_MAX, MAZE_BFS); finish_generation(&edge); finish_search(&edge); check_path(&edge);
    assert(!maze_is_open(&edge, -1, 1) && !maze_is_open(&edge, 1, MAZE_HEIGHT));
}

/** @brief 有环地图中 DFS 走较长绕路，BFS 得到独立可计算的最短路径。 */
static void test_shortest_and_failure(void)
{
    MazeWorld world;
    maze_reset(&world, 42, MAZE_BFS);
    for (int i = 0; i < MAZE_CELLS; ++i) { world.open[i] = false; }
    /* 上方短路 36+26=62；下方绕路 24+34+22+2+24=106。 */
    for (int x = 1; x <= 37; ++x) { world.open[MAZE_WIDTH + x] = true; }
    for (int y = 1; y <= 27; ++y) { world.open[y * MAZE_WIDTH + 37] = true; }
    for (int y = 1; y <= 25; ++y) { world.open[y * MAZE_WIDTH + 1] = true; }
    for (int x = 1; x <= 35; ++x) { world.open[25 * MAZE_WIDTH + x] = true; }
    for (int y = 3; y <= 25; ++y) { world.open[y * MAZE_WIDTH + 35] = true; }
    for (int x = 35; x <= 37; ++x) { world.open[3 * MAZE_WIDTH + x] = true; }
    world.phase = MAZE_SEARCHING;
    assert(maze_restart_search(&world, MAZE_BFS)); finish_search(&world); check_path(&world);
    assert(world.path_length == 62);
    assert(maze_restart_search(&world, MAZE_DFS)); finish_search(&world); check_path(&world);
    assert(world.path_length == 106);
    assert(!maze_restart_search(&world, (MazeAlgorithm)99) && world.phase == MAZE_FOUND);
    for (int i = 0; i < MAZE_CELLS; ++i) { world.open[i] = false; }
    world.open[MAZE_START] = world.open[MAZE_GOAL] = true;
    assert(maze_restart_search(&world, MAZE_BFS)); finish_search(&world);
    assert(world.phase == MAZE_NO_PATH && world.visited_count == 1 && world.path_length == 0);
    world.open[MAZE_GOAL] = false;
    assert(maze_restart_search(&world, MAZE_DFS) && world.phase == MAZE_NO_PATH);
}

/** @brief 验证画布边界、端点颜色及找到路径后的高亮。 */
static void test_draw(void)
{
    MazeWorld world;
    maze_reset(&world, 42, MAZE_BFS); finish_generation(&world); finish_search(&world);
    static GfxCanvas canvas;
    maze_draw(&world, &canvas);
    assert(canvas.pixels[0][0] == gfx_rgb(18, 23, 32));
    assert(canvas.pixels[4][4] == gfx_rgb(40, 48, 62));
    assert(canvas.pixels[12][12] == gfx_rgb(70, 220, 110));
    assert(canvas.pixels[220][300] == gfx_rgb(245, 80, 90));
    for (int id = 0; id < MAZE_CELLS; ++id) {
        if (world.path[id] && id != MAZE_START && id != MAZE_GOAL) {
            assert(canvas.pixels[4 + id / MAZE_WIDTH * 8][4 + id % MAZE_WIDTH * 8] == gfx_rgb(255, 210, 70));
        }
    }
}

/** @brief 执行全部纯 C 迷宫断言。 @return 成功返回零。 */
int main(void)
{
    test_generation(); test_shortest_and_failure(); test_draw();
    puts("迷宫测试通过：可重复生成、连通无环、BFS 最短路、DFS、无路与绘制。");
    return 0;
}
