/** @file @brief 平静水面、传播、对称性、衰减、边界与非法请求验证。 */
#include "../src/graphics/ripple.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/** @brief 读取当前高度。 @param world 初始化后的状态。 @param x 列。 @param y 行。 */
static float value(const RippleWorld *world, int x, int y)
{
    return world->height[world->current][y][x];
}

/** @brief 验证两层零边界。 @param world 初始化后的状态。 */
static void check_edges(const RippleWorld *world)
{
    for (unsigned layer = 0; layer < 2; ++layer) {
        for (int x = 0; x < RIPPLE_WIDTH; ++x) {
            assert(world->height[layer][0][x] == 0);
            assert(world->height[layer][RIPPLE_HEIGHT - 1][x] == 0);
        }
        for (int y = 0; y < RIPPLE_HEIGHT; ++y) {
            assert(world->height[layer][y][0] == 0);
            assert(world->height[layer][y][RIPPLE_WIDTH - 1] == 0);
        }
    }
}

/** @brief 一个单格凸起应同步传给四邻居，之后保持镜像对称并逐渐衰减。 */
static void test_propagation(void)
{
    static RippleWorld world;
    ripple_reset(&world);
    assert(ripple_step(&world) && world.steps == 1);
    assert(value(&world, 80, 60) == 0);
    assert(ripple_disturb(&world, 80, 60, 1, 8));
    assert(value(&world, 80, 60) == 8);
    assert(world.height[1 - world.current][60][80] == 8);
    assert(ripple_step(&world));
    assert(fabsf(value(&world, 80, 60) - 1.6f) < 1e-5f);
    assert(fabsf(value(&world, 79, 60) - 1.6f) < 1e-5f);
    assert(fabsf(value(&world, 81, 60) - 1.6f) < 1e-5f);
    assert(fabsf(value(&world, 80, 59) - 1.6f) < 1e-5f);
    assert(value(&world, 78, 60) == 0); /* 每步只能影响相邻一格。 */
    for (int step = 0; step < 30; ++step) { assert(ripple_step(&world)); }
    for (int offset = 0; offset < 25; ++offset) {
        assert(fabsf(value(&world, 80 - offset, 60) - value(&world, 80 + offset, 60)) < 1e-5f);
        assert(fabsf(value(&world, 80, 60 - offset) - value(&world, 80, 60 + offset)) < 1e-5f);
    }
    for (int step = 0; step < 1800; ++step) { assert(ripple_step(&world)); }
    float peak = 0;
    for (int y = 0; y < RIPPLE_HEIGHT; ++y) {
        for (int x = 0; x < RIPPLE_WIDTH; ++x) {
            float height = value(&world, x, y);
            assert(isfinite(height) && fabsf(height) <= RIPPLE_HEIGHT_LIMIT);
            peak = fmaxf(peak, fabsf(height));
        }
    }
    assert(peak < 0.05f);
    check_edges(&world);
    world.steps = UINT64_MAX;
    assert(ripple_step(&world) && world.steps == UINT64_MAX);
}

/** @brief 正负波纹在限幅前线性叠加，边缘扰动保留固定边界。 */
static void test_disturbances(void)
{
    static RippleWorld positive, negative;
    ripple_reset(&positive);
    ripple_reset(&negative);
    assert(ripple_disturb(&positive, 1, 1, 6, 8));
    assert(ripple_disturb(&negative, 1, 1, 6, -8));
    for (int i = 0; i < 20; ++i) {
        assert(ripple_step(&positive) && ripple_step(&negative));
    }
    for (int y = 0; y < RIPPLE_HEIGHT; ++y) {
        for (int x = 0; x < RIPPLE_WIDTH; ++x) {
            assert(fabsf(value(&positive, x, y) + value(&negative, x, y)) < 1e-5f);
        }
    }
    check_edges(&positive);
    for (int i = 0; i < 100; ++i) { assert(ripple_disturb(&positive, 1, 1, 1, 8)); }
    assert(value(&positive, 1, 1) == RIPPLE_HEIGHT_LIMIT);
}

/** @brief 非法参数不修改状态或画布；重置和绘制可用于空水面。 */
static void test_failures_and_draw(void)
{
    static RippleWorld world, before;
    static GfxCanvas canvas, image_before;
    ripple_reset(&world);
    before = world;
    assert(!ripple_disturb(&world, 0, 1, 6, 8));
    assert(!ripple_disturb(&world, RIPPLE_WIDTH, 1, 6, 8));
    assert(!ripple_disturb(&world, 1, RIPPLE_HEIGHT - 1, 6, 8));
    assert(!ripple_disturb(&world, 1, 1, 0, 8));
    assert(!ripple_disturb(&world, 1, 1, 17, 8));
    assert(!ripple_disturb(&world, 1, 1, 6, NAN));
    assert(!ripple_disturb(&world, 1, 1, 6, INFINITY));
    assert(!ripple_disturb(&world, 1, 1, 6, 9));
    assert(memcmp(&world, &before, sizeof world) == 0);
    assert(!ripple_step(nullptr));
    assert(!ripple_draw(nullptr, &canvas));
    assert(ripple_draw(&world, &canvas));
    for (int y = 0; y < GFX_HEIGHT; ++y) {
        for (int x = 0; x < GFX_WIDTH; ++x) { assert(canvas.pixels[y][x] == canvas.pixels[0][0]); }
    }
    assert(ripple_disturb(&world, 80, 60, 6, 8));
    assert(ripple_draw(&world, &canvas));
    assert(canvas.pixels[120][160] != canvas.pixels[0][0]);
    assert(canvas.pixels[120][160] == canvas.pixels[121][161]);
    image_before = canvas;
    world.current = 2;
    before = world;
    assert(!ripple_step(&world) && !ripple_draw(&world, &canvas));
    assert(!ripple_disturb(&world, 80, 60, 6, 8));
    assert(memcmp(&world, &before, sizeof world) == 0);
    assert(memcmp(&canvas, &image_before, sizeof canvas) == 0);
    ripple_reset(&world);
    assert(world.current == 0 && world.steps == 0 && value(&world, 80, 60) == 0);
}

/** @brief 运行无 SDL 的验证。 @return 成功返回零，失败由断言终止。 */
int main(void)
{
    test_propagation();
    test_disturbances();
    test_failures_and_draw();
    puts("波纹核心测试通过：传播、对称、衰减、边界、限幅与失败行为。");
    return 0;
}
