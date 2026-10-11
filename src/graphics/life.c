/** @file @brief 康威生命游戏：邻居计数与双缓冲，完全不依赖 SDL。 */
#include "life.h"

/* 网格必须恰好覆盖画布，改变尺寸时在编译期发现不匹配。 */
_Static_assert(LIFE_WIDTH *LIFE_CELL_SIZE == GFX_WIDTH, "网格宽度不匹配");
_Static_assert(LIFE_HEIGHT *LIFE_CELL_SIZE == GFX_HEIGHT, "网格高度不匹配");

/** @copydoc life_reset */
void life_reset(LifeWorld *world, LifePattern pattern)
{
    /* 聚合初始化把两张网格、索引、代数全部置零，不遗留上次演化的细胞。 */
    *world = (LifeWorld){0};
    const int x = LIFE_WIDTH / 2 - 1, y = LIFE_HEIGHT / 2 - 1;
    if (pattern == LIFE_GLIDER) {
        /* .O. / ..O / OOO：四代后向右下移动一格。 */
        world->cells[0][y][x + 1] = true;
        world->cells[0][y + 1][x + 2] = true;
        world->cells[0][y + 2][x] = true;
        world->cells[0][y + 2][x + 1] = true;
        world->cells[0][y + 2][x + 2] = true;
    } else if (pattern == LIFE_BLINKER) {
        for (int dx = 0; dx < 3; ++dx) {
            world->cells[0][y + 1][x + dx] = true;
        }
    } else if (pattern == LIFE_RANDOM) {
        /* 独立线性同余随机流，不影响其他功能的随机状态。
         * 重置时固定种子，相同调用顺序生成相同图案；只用于教学。
         * 高两位为零的概率约为 1/4，避开低位的短周期。
         */
        uint32_t rng = 42;
        for (int row = 0; row < LIFE_HEIGHT; ++row) {
            for (int col = 0; col < LIFE_WIDTH; ++col) {
                rng = rng * UINT32_C(1664525) + UINT32_C(1013904223);
                world->cells[0][row][col] = (rng >> 30) == 0;
            }
        }
    }
}

/** @copydoc life_get */
bool life_get(const LifeWorld *world, int x, int y)
{
    /* && 短路求值：通过边界检查后才访问数组，不将负数用作下标。
     * 这里决定边界规则，没有对坐标取模，所以上下左右不会首尾相连。
     */
    return x >= 0 && x < LIFE_WIDTH && y >= 0 && y < LIFE_HEIGHT &&
           world->cells[world->current][y][x];
}

/** @copydoc life_step */
void life_step(LifeWorld *world)
{
    unsigned int next = 1 - world->current;
    for (int y = 0; y < LIFE_HEIGHT; ++y) {
        for (int x = 0; x < LIFE_WIDTH; ++x) {
            unsigned int neighbors = 0;
            /* 3×3 邻域跳过中心格，只统计八个方向；bool 可转为 0/1 累加。 */
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx != 0 || dy != 0) {
                        neighbors += life_get(world, x + dx, y + dy);
                    }
                }
            }
            bool alive = life_get(world, x, y);
            /* B3：三个邻居时出生；S23：已存活且两个/三个邻居时存活。
             * 每一格都写入 next，包括死亡结果，因此无需预先清空目标网格。
             * 不能写回 current，否则后面的格会读到本代刚更新的数据。
             */
            world->cells[next][y][x] = neighbors == 3 || (alive && neighbors == 2);
        }
    }
    world->current = next; /* 所有格计算完后，统一切换“当前世界”。 */
    ++world->generation;
}

/** @copydoc life_toggle */
bool life_toggle(LifeWorld *world, int x, int y)
{
    if (x < 0 || x >= LIFE_WIDTH || y < 0 || y >= LIFE_HEIGHT) {
        return false;
    }
    world->cells[world->current][y][x] = !world->cells[world->current][y][x];
    world->generation = 0;
    return true;
}

/** @copydoc life_count */
size_t life_count(const LifeWorld *world)
{
    size_t count = 0;
    for (int y = 0; y < LIFE_HEIGHT; ++y) {
        for (int x = 0; x < LIFE_WIDTH; ++x) {
            count += life_get(world, x, y);
        }
    }
    return count;
}

/** @copydoc life_draw */
void life_draw(const LifeWorld *world, GfxCanvas *canvas)
{
    gfx_clear(canvas, gfx_rgb(18, 23, 32));
    for (int y = 0; y < LIFE_HEIGHT; ++y) {
        for (int x = 0; x < LIFE_WIDTH; ++x) {
            if (life_get(world, x, y)) {
                /* 每格 4×4，画其中的 3×3，留下顶部和左侧的一像素间隔。
                 * 最右、最下的矩形也在画布内。先清屏避免死亡格留下旧图像。
                 */
                gfx_rect(canvas, x * LIFE_CELL_SIZE + 1, y * LIFE_CELL_SIZE + 1, LIFE_CELL_SIZE - 1,
                         LIFE_CELL_SIZE - 1, true, gfx_rgb(90, 220, 130));
            }
        }
    }
}
