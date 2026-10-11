/** @file @brief 下落、堆积、水流、沙水交换、守恒及边界验证。 */
#include "../src/graphics/sand.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/** @brief 统计物质格数。 @param world 有效状态。 @param material 物质类型。 */
static size_t count_material(const SandWorld *world, SandMaterial material)
{
    size_t count = 0;
    for (int y = 0; y < SAND_HEIGHT; ++y) {
        for (int x = 0; x < SAND_WIDTH; ++x) { count += world->cells[y][x] == material; }
    }
    return count;
}

/** @brief 一次只下落一格；沙遇墙斜落，左右优先随步交替。 */
static void test_gravity(void)
{
    static SandWorld world;
    sand_reset(&world);
    assert(sand_paint(&world, 80, 20, 0, SAND_GRAIN));
    assert(sand_step(&world));
    assert(world.cells[20][80] == SAND_AIR && world.cells[21][80] == SAND_GRAIN);
    assert(world.cells[22][80] == SAND_AIR);
    sand_reset(&world);
    sand_paint(&world, 80, 20, 0, SAND_GRAIN);
    sand_paint(&world, 80, 21, 0, SAND_WALL);
    sand_step(&world);
    assert(world.cells[21][79] == SAND_GRAIN);
    sand_reset(&world);
    sand_step(&world); /* 空步切换左右优先。 */
    sand_paint(&world, 80, 20, 0, SAND_GRAIN);
    sand_paint(&world, 80, 21, 0, SAND_WALL);
    sand_step(&world);
    assert(world.cells[21][81] == SAND_GRAIN);
    sand_reset(&world);
    sand_paint(&world, 80, 20, 0, SAND_GRAIN);
    for (int x = 79; x <= 81; ++x) { sand_paint(&world, x, 21, 0, SAND_WALL); }
    sand_step(&world);
    assert(world.cells[20][80] == SAND_GRAIN);
}

/** @brief 水在底部水平移动一格；沙与静止水交换后水不再次移动。 */
static void test_water(void)
{
    static SandWorld world;
    sand_reset(&world);
    sand_paint(&world, 80, SAND_HEIGHT - 2, 0, SAND_WATER);
    sand_step(&world);
    assert(world.cells[SAND_HEIGHT - 2][79] == SAND_WATER);
    assert(world.cells[SAND_HEIGHT - 2][78] == SAND_AIR);
    sand_reset(&world);
    sand_paint(&world, 80, SAND_HEIGHT - 2, 0, SAND_WATER);
    sand_paint(&world, 79, SAND_HEIGHT - 2, 0, SAND_WALL);
    sand_step(&world); /* 向扫描前方移动也不能被再次处理。 */
    assert(world.cells[SAND_HEIGHT - 2][81] == SAND_WATER);
    assert(world.cells[SAND_HEIGHT - 2][82] == SAND_AIR);
    sand_reset(&world);
    sand_paint(&world, 80, 20, 0, SAND_GRAIN);
    sand_paint(&world, 80, 21, 0, SAND_WATER);
    for (int x = 79; x <= 81; ++x) { sand_paint(&world, x, 22, 0, SAND_WALL); }
    sand_paint(&world, 79, 21, 0, SAND_WALL);
    sand_paint(&world, 81, 21, 0, SAND_WALL);
    sand_step(&world);
    assert(world.cells[21][80] == SAND_GRAIN);
    assert(world.cells[20][80] == SAND_WATER);
    assert(world.cells[20][79] == SAND_AIR); /* 被交换的水不在本步再流动。 */
    assert(count_material(&world, SAND_GRAIN) == 1 && count_material(&world, SAND_WATER) == 1);
    sand_reset(&world);
    sand_paint(&world, 80, 20, 0, SAND_WATER);
    for (int x = 75; x <= 90; ++x) { sand_paint(&world, x, 21, 0, SAND_WALL); }
    sand_step(&world);
    assert(world.cells[20][79] == SAND_WATER);
    sand_step(&world); /* 左侧出口更近，切换优先方向后仍向左流动。 */
    assert(world.cells[20][78] == SAND_WATER && world.cells[20][80] == SAND_AIR);
}

/** @brief 重复演化保持物质量与固定墙不变，相同初始状态得到相同结果。 */
static void test_conservation(void)
{
    static SandWorld a, b;
    sand_reset(&a);
    sand_paint(&a, 60, 20, 8, SAND_GRAIN);
    sand_paint(&a, 90, 20, 8, SAND_WATER);
    for (int x = 40; x < 120; ++x) { sand_paint(&a, x, 80, 0, SAND_WALL); }
    b = a;
    size_t sand = count_material(&a, SAND_GRAIN), water = count_material(&a, SAND_WATER);
    size_t walls = count_material(&a, SAND_WALL);
    for (int step = 0; step < 300; ++step) {
        assert(sand_step(&a) && sand_step(&b));
        assert(count_material(&a, SAND_GRAIN) == sand);
        assert(count_material(&a, SAND_WATER) == water);
        assert(count_material(&a, SAND_WALL) == walls);
    }
    assert(memcmp(&a, &b, sizeof a) == 0);
    for (int y = 0; y < SAND_HEIGHT; ++y) {
        assert(a.cells[y][0] == SAND_WALL && a.cells[y][SAND_WIDTH - 1] == SAND_WALL);
    }
    for (int x = 0; x < SAND_WIDTH; ++x) {
        assert(a.cells[0][x] == SAND_WALL && a.cells[SAND_HEIGHT - 1][x] == SAND_WALL);
    }
    a.steps = UINT64_MAX;
    bool direction = a.right_first;
    assert(sand_step(&a) && a.steps == UINT64_MAX && a.right_first != direction);
}

/** @brief 笔刷裁切、擦除、无效输入与物质颜色正确。 */
static void test_paint_and_draw(void)
{
    static SandWorld world, before;
    static GfxCanvas canvas, image;
    sand_reset(&world);
    before = world;
    assert(!sand_paint(&world, 0, 1, 3, SAND_AIR));
    assert(!sand_paint(&world, 1, SAND_HEIGHT, 3, SAND_GRAIN));
    assert(!sand_paint(&world, 1, 1, -1, SAND_GRAIN));
    assert(!sand_paint(&world, 1, 1, 9, SAND_GRAIN));
    assert(!sand_paint(&world, 1, 1, 1, (SandMaterial)99));
    assert(!sand_paint(nullptr, 1, 1, 1, SAND_GRAIN));
    assert(memcmp(&world, &before, sizeof world) == 0);
    assert(sand_paint(&world, 1, 1, 8, SAND_GRAIN));
    assert(world.cells[0][0] == SAND_WALL && world.cells[1][1] == SAND_GRAIN);
    assert(sand_paint(&world, 1, 1, 0, SAND_WATER));
    assert(world.cells[1][1] == SAND_GRAIN); /* 添加不会覆盖原有物质。 */
    assert(sand_paint(&world, 1, 1, 8, SAND_AIR));
    assert(count_material(&world, SAND_GRAIN) == 0 && world.cells[0][0] == SAND_WALL);
    sand_paint(&world, 80, 60, 0, SAND_GRAIN);
    assert(sand_draw(&world, &canvas));
    assert(canvas.pixels[120][160] == gfx_rgb(235, 195, 80));
    assert(canvas.pixels[120][160] == canvas.pixels[121][161]);
    assert(canvas.pixels[0][0] == gfx_rgb(115, 125, 140));
    image = canvas;
    assert(!sand_draw(nullptr, &canvas) && !sand_draw(&world, nullptr));
    assert(memcmp(&image, &canvas, sizeof image) == 0);
    assert(!sand_step(nullptr));
}

/** @brief 运行纯 C 验证。 @return 成功返回零，失败由断言终止。 */
int main(void)
{
    test_gravity();
    test_water();
    test_conservation();
    test_paint_and_draw();
    puts("落沙核心测试通过：重力、斜落、水流、交换、守恒、边界和笔刷。");
    return 0;
}
