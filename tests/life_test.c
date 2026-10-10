/** @file @brief 纯 C 验证生命规则、经典图案、双缓冲、边界和绘制。 */
#include "../src/graphics/life.h"
#include <assert.h>
#include <stdio.h>

/** @brief 穷举中心格的存活状态和邻居数，验证 B3/S23 真值表。 */
static void test_rules(void)
{
    const int offsets[8][2] = {{-1,-1},{0,-1},{1,-1},{-1,0},{1,0},{-1,1},{0,1},{1,1}};
    LifeWorld world;
    for (int alive = 0; alive <= 1; ++alive) {
        for (int count = 0; count <= 8; ++count) {
            life_reset(&world, LIFE_EMPTY);
            world.cells[0][10][10] = alive != 0;
            for (int i = 0; i < count; ++i) {
                world.cells[0][10 + offsets[i][1]][10 + offsets[i][0]] = true;
            }
            life_step(&world);
            assert(life_get(&world, 10, 10) == (count == 3 || (alive && count == 2)));
            assert(world.generation == 1 && world.current == 1);
        }
    }
    life_reset(&world, LIFE_EMPTY);
    life_step(&world);
    assert(life_count(&world) == 0);
    /* 角落只有三个有效邻居；三个邻居均存活会使角落出生。 */
    life_toggle(&world, 1, 0); life_toggle(&world, 0, 1); life_toggle(&world, 1, 1);
    life_step(&world);
    assert(life_get(&world, 0, 0) && life_count(&world) == 4);
    life_reset(&world, LIFE_EMPTY);
    life_toggle(&world, LIFE_WIDTH - 1, 0);
    life_toggle(&world, LIFE_WIDTH - 1, LIFE_HEIGHT - 1);
    life_toggle(&world, 0, LIFE_HEIGHT - 1);
    life_step(&world);
    assert(!life_get(&world, 0, 0)); /* 对面的格子不通过边界环绕成为邻居。 */
    assert(!life_get(&world, -1, 0) && !life_get(&world, LIFE_WIDTH, 0));
    assert(!life_get(&world, 0, -1) && !life_get(&world, 0, LIFE_HEIGHT));
}

/** @brief 稳定方块、两周期振荡器和滑翔机验证同步演化。 */
static void test_patterns(void)
{
    LifeWorld world, initial;
    life_reset(&world, LIFE_EMPTY);
    for (int y = 20; y < 22; ++y) {
        for (int x = 30; x < 32; ++x) { life_toggle(&world, x, y); }
    }
    initial = world;
    for (int step = 0; step < 10; ++step) {
        life_step(&world);
        for (int y = 0; y < LIFE_HEIGHT; ++y) {
            for (int x = 0; x < LIFE_WIDTH; ++x) {
                assert(life_get(&world, x, y) == life_get(&initial, x, y));
            }
        }
    }
    life_reset(&world, LIFE_BLINKER);
    initial = world;
    life_step(&world);
    assert(life_count(&world) == 3 && life_get(&world, 40, 29) && life_get(&world, 40, 31));
    assert(!life_get(&world, 39, 30));
    life_step(&world);
    assert(world.current == 0 && world.generation == 2);
    for (int y = 0; y < LIFE_HEIGHT; ++y) {
        for (int x = 0; x < LIFE_WIDTH; ++x) {
            assert(life_get(&world, x, y) == life_get(&initial, x, y));
        }
    }
    life_reset(&world, LIFE_GLIDER);
    initial = world;
    for (int step = 0; step < 4; ++step) { life_step(&world); }
    assert(life_count(&world) == 5);
    for (int y = 0; y < LIFE_HEIGHT; ++y) {
        for (int x = 0; x < LIFE_WIDTH; ++x) {
            assert(life_get(&world, x, y) == life_get(&initial, x - 1, y - 1));
        }
    }
}

/** @brief 反向遍历的独立参考结果应与实际更新一致，排除扫描顺序依赖。 */
static void test_scan_order(void)
{
    LifeWorld world;
    life_reset(&world, LIFE_RANDOM);
    bool expected[LIFE_HEIGHT][LIFE_WIDTH];
    for (int y = LIFE_HEIGHT - 1; y >= 0; --y) {
        for (int x = LIFE_WIDTH - 1; x >= 0; --x) {
            unsigned int count = 0;
            for (int row = y - 1; row <= y + 1; ++row) {
                for (int col = x - 1; col <= x + 1; ++col) {
                    if (row != y || col != x) { count += life_get(&world, col, row); }
                }
            }
            expected[y][x] = life_get(&world, x, y) ? (count == 2 || count == 3) : count == 3;
        }
    }
    life_step(&world);
    for (int y = 0; y < LIFE_HEIGHT; ++y) {
        for (int x = 0; x < LIFE_WIDTH; ++x) { assert(life_get(&world, x, y) == expected[y][x]); }
    }
}

/** @brief 固定随机预设可重现；切换、复用双缓冲和像素间隔正确。 */
static void test_edit_and_draw(void)
{
    LifeWorld a, b;
    life_reset(&a, LIFE_RANDOM); life_reset(&b, LIFE_RANDOM);
    assert(life_count(&a) > 0 && life_count(&a) < LIFE_WIDTH * LIFE_HEIGHT);
    for (int step = 0; step < 20; ++step) {
        for (int y = 0; y < LIFE_HEIGHT; ++y) {
            for (int x = 0; x < LIFE_WIDTH; ++x) { assert(life_get(&a, x, y) == life_get(&b, x, y)); }
        }
        life_step(&a); life_step(&b);
    }
    uint64_t old_generation = a.generation;
    assert(!life_toggle(&a, -1, 0) && !life_toggle(&a, 0, LIFE_HEIGHT));
    assert(a.generation == old_generation);
    life_reset(&a, LIFE_EMPTY);
    life_step(&a); /* 在 current == 1 时编辑，下一步必须覆盖旧缓冲的每个格子。 */
    assert(life_toggle(&a, 0, 0) && life_get(&a, 0, 0) && a.generation == 0);
    assert(life_toggle(&a, 0, 0) && !life_get(&a, 0, 0));
    life_toggle(&a, LIFE_WIDTH - 1, LIFE_HEIGHT - 1);
    static GfxCanvas canvas;
    life_draw(&a, &canvas);
    assert(canvas.pixels[239][319] == gfx_rgb(90, 220, 130));
    assert(canvas.pixels[236][316] == gfx_rgb(18, 23, 32));
    life_step(&a);
    life_draw(&a, &canvas);
    assert(life_count(&a) == 0 && canvas.pixels[239][319] == gfx_rgb(18, 23, 32));
}

/** @brief 运行全部生命游戏核心断言。 @return 成功返回零。 */
int main(void)
{
    test_rules(); test_patterns(); test_scan_order(); test_edit_and_draw();
    puts("生命游戏核心测试通过：规则、图案、边界、双缓冲与绘制。");
    return 0;
}
