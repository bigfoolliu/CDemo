/** @file @brief 离散局部规则模拟颗粒与水；不求解真实流体力学方程。 */
#include "sand.h"
#include <string.h>

/** @brief 判断可编辑内圈。 @param x 列坐标。 @param y 行坐标。 */
static bool inside(int x, int y)
{
    return x > 0 && x < SAND_WIDTH - 1 && y > 0 && y < SAND_HEIGHT - 1;
}

/** @copydoc sand_reset */
void sand_reset(SandWorld *world)
{
    if (world == nullptr) { return; }
    *world = (SandWorld){0};
    for (int y = 0; y < SAND_HEIGHT; ++y) {
        for (int x = 0; x < SAND_WIDTH; ++x) {
            if (!inside(x, y)) { world->cells[y][x] = SAND_WALL; }
        }
    }
}

/** @copydoc sand_paint */
bool sand_paint(SandWorld *world, int x, int y, int radius, SandMaterial material)
{
    if (world == nullptr || !inside(x, y) || radius < 0 || radius > SAND_MAX_BRUSH ||
        material < SAND_AIR || material > SAND_WALL) { return false; }
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            if (dx * dx + dy * dy > radius * radius || !inside(x + dx, y + dy)) { continue; }
            uint8_t *cell = &world->cells[y + dy][x + dx];
            if (material == SAND_AIR || *cell == SAND_AIR) { *cell = (uint8_t)material; }
        }
    }
    return true;
}

/** @brief 尝试交换源格和目标格，并标记本步已经处理。
 * @param world 已初始化可写状态。
 * @param x 源列。 @param y 源行。
 * @param target_x 目标列。 @param target_y 目标行。
 * @return 目标可进入且未被本步移动标记占用时 true。
 */
static bool try_move(SandWorld *world, int x, int y, int target_x, int target_y)
{
    if (!inside(target_x, target_y) || world->moved[target_y][target_x]) { return false; }
    uint8_t source = world->cells[y][x], target = world->cells[target_y][target_x];
    if (target != SAND_AIR && !(source == SAND_GRAIN && target == SAND_WATER)) { return false; }
    world->cells[target_y][target_x] = source;
    world->cells[y][x] = target;
    /* 沙水交换时标记源格，防止被抬起的水再次移动；空出来的空气
     * 不标记，让上方物质在本步正常填入，保持连续下落。
     */
    world->moved[y][x] = target != SAND_AIR;
    world->moved[target_y][target_x] = true;
    return true;
}

/** @brief 寻找同一行空气通道内最近的可下落位置，避免左右交替导致水原地摆动。
 * @param world 已初始化的只读状态。
 * @param x 水粒子列。 @param y 水粒子行。
 * @param fallback 等距或没有出口时的默认方向，-1 或 1。
 * @return 优先水平移动方向，仍只移动一格，不直接跳到出口。
 */
static int water_direction(const SandWorld *world, int x, int y, int fallback)
{
    if (y >= SAND_HEIGHT - 2) { return fallback; } /* 底部不可能再下落。 */
    int best_distance = SAND_WIDTH, chosen = fallback;
    for (int side = 0; side < 2; ++side) {
        int direction = side == 0 ? fallback : -fallback;
        for (int distance = 1; distance < SAND_WIDTH; ++distance) {
            int target_x = x + distance * direction;
            if (!inside(target_x, y) || world->cells[y][target_x] != SAND_AIR) { break; }
            if (inside(target_x, y + 1) && world->cells[y + 1][target_x] == SAND_AIR) {
                if (distance < best_distance) { best_distance = distance; chosen = direction; }
                break;
            }
        }
    }
    return chosen;
}

/** @copydoc sand_step */
bool sand_step(SandWorld *world)
{
    if (world == nullptr) { return false; }
    memset(world->moved, 0, sizeof world->moved);
    int direction = world->right_first ? 1 : -1;
    /* 自底向上使下落后的粒子不再被遍历；水平方向交替减小单侧偏置。
     * moved 额外保护水的水平移动与沙水交换，避免一次移动多格。
     */
    for (int y = SAND_HEIGHT - 2; y >= 1; --y) {
        for (int column = 1; column < SAND_WIDTH - 1; ++column) {
            int x = world->right_first ? SAND_WIDTH - 1 - column : column;
            uint8_t material = world->cells[y][x];
            if (world->moved[y][x] || (material != SAND_GRAIN && material != SAND_WATER)) { continue; }
            if (try_move(world, x, y, x, y + 1) ||
                try_move(world, x, y, x + direction, y + 1) ||
                try_move(world, x, y, x - direction, y + 1)) { continue; }
            if (material == SAND_WATER) {
                int flow = water_direction(world, x, y, direction);
                if (try_move(world, x, y, x + flow, y) ||
                    try_move(world, x, y, x - flow, y)) { continue; }
            }
        }
    }
    world->right_first = !world->right_first;
    if (world->steps != UINT64_MAX) { ++world->steps; }
    return true;
}

/** @copydoc sand_draw */
bool sand_draw(const SandWorld *world, GfxCanvas *canvas)
{
    if (world == nullptr || canvas == nullptr) { return false; }
    const uint32_t colors[] = {gfx_rgb(18, 22, 32), gfx_rgb(235, 195, 80),
                               gfx_rgb(45, 135, 235), gfx_rgb(115, 125, 140)};
    for (int y = 0; y < SAND_HEIGHT; ++y) {
        for (int x = 0; x < SAND_WIDTH; ++x) {
            uint8_t material = world->cells[y][x];
            /* 正常接口保证物质编号有效；损坏编号用背景色显示，避免数组越界。 */
            uint32_t color = colors[material <= SAND_WALL ? material : SAND_AIR];
            gfx_rect(canvas, x * SAND_CELL_SIZE, y * SAND_CELL_SIZE,
                     SAND_CELL_SIZE, SAND_CELL_SIZE, true, color);
        }
    }
    return true;
}
