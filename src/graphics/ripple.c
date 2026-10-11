/** @file @brief 中心差分波动方程；两层高度保存位置和隐含速度。 */
#include "ripple.h"
#include <math.h>

/** @brief 限制有限数值。 @param value 数值。 @param low 下界。 @param high 上界。 */
static float clamp_value(float value, float low, float high)
{
    return value < low ? low : value > high ? high : value;
}

/** @copydoc ripple_reset */
void ripple_reset(RippleWorld *world)
{
    if (world != nullptr) { *world = (RippleWorld){0}; }
}

/** @copydoc ripple_disturb */
bool ripple_disturb(RippleWorld *world, int x, int y, int radius, float strength)
{
    if (world == nullptr || world->current > 1 || x < 1 || x >= RIPPLE_WIDTH - 1 ||
        y < 1 || y >= RIPPLE_HEIGHT - 1 || radius < 1 || radius > RIPPLE_MAX_RADIUS ||
        !isfinite(strength) || strength < -8 || strength > 8) { return false; }
    int left = x - radius < 1 ? 1 : x - radius;
    int right = x + radius > RIPPLE_WIDTH - 2 ? RIPPLE_WIDTH - 2 : x + radius;
    int top = y - radius < 1 ? 1 : y - radius;
    int bottom = y + radius > RIPPLE_HEIGHT - 2 ? RIPPLE_HEIGHT - 2 : y + radius;
    for (int row = top; row <= bottom; ++row) {
        for (int col = left; col <= right; ++col) {
            float dx = (float)(col - x), dy = (float)(row - y);
            float distance = sqrtf(dx * dx + dy * dy);
            if (distance >= radius) { continue; }
            /* 余弦包络在中心最大、边缘平滑降到零，减少单像素尖峰的锯齿。 */
            float offset = strength * 0.5f * (1 + cosf(3.14159265358979323846f * distance / radius));
            for (unsigned layer = 0; layer < 2; ++layer) {
                world->height[layer][row][col] = clamp_value(world->height[layer][row][col] + offset,
                    -RIPPLE_HEIGHT_LIMIT, RIPPLE_HEIGHT_LIMIT);
            }
        }
    }
    return true;
}

/** @copydoc ripple_step */
bool ripple_step(RippleWorld *world)
{
    if (world == nullptr || world->current > 1) { return false; }
    unsigned old = 1 - world->current;
    const float (*current)[RIPPLE_WIDTH] = world->height[world->current];
    float (*previous)[RIPPLE_WIDTH] = world->height[old];
    for (int y = 1; y < RIPPLE_HEIGHT - 1; ++y) {
        for (int x = 1; x < RIPPLE_WIDTH - 1; ++x) {
            float center = current[y][x];
            float laplacian = current[y][x - 1] + current[y][x + 1] +
                              current[y - 1][x] + current[y + 1][x] - 4 * center;
            /* 速度近似为 current-previous；阻尼抑制速度，而非直接清空高度。
             * 这里只读取旧层的同一格，不读取它的邻居，因此可以安全覆盖旧层。
             * 所有空间邻居来自只读的当前层，不能将结果写回当前层。
             */
            float next = 2 * center - previous[y][x] + RIPPLE_COURANT_SQUARED * laplacian -
                         RIPPLE_DAMPING * (center - previous[y][x]);
            previous[y][x] = clamp_value(next, -RIPPLE_HEIGHT_LIMIT, RIPPLE_HEIGHT_LIMIT);
        }
    }
    world->current = old;
    if (world->steps != UINT64_MAX) { ++world->steps; }
    return true;
}

/** @copydoc ripple_draw */
bool ripple_draw(const RippleWorld *world, GfxCanvas *canvas)
{
    if (world == nullptr || world->current > 1 || canvas == nullptr) { return false; }
    const float (*height)[RIPPLE_WIDTH] = world->height[world->current];
    for (int y = 0; y < RIPPLE_HEIGHT; ++y) {
        for (int x = 0; x < RIPPLE_WIDTH; ++x) {
            float slope = 0;
            if (x > 0 && x < RIPPLE_WIDTH - 1 && y > 0 && y < RIPPLE_HEIGHT - 1) {
                slope = height[y][x - 1] - height[y][x + 1] +
                        height[y - 1][x] - height[y + 1][x];
            }
            /* 坡度模拟固定光照的明暗，少量高度项帮助区分波峰和波谷。
             * 转整数前将颜色限制在 [0,1]，避免越界浮点转整数。
             */
            float light = clamp_value(0.45f + 0.25f * slope + 0.025f * height[y][x], 0, 1);
            uint32_t color = gfx_rgb((uint8_t)(15 + 40 * light), (uint8_t)(50 + 120 * light),
                                     (uint8_t)(100 + 150 * light));
            gfx_rect(canvas, x * RIPPLE_CELL_SIZE, y * RIPPLE_CELL_SIZE,
                     RIPPLE_CELL_SIZE, RIPPLE_CELL_SIZE, true, color);
        }
    }
    return true;
}
