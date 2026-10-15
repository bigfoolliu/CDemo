/** @file @brief 验证玫瑰粒子可重复性、运动状态与竖屏像素区域。 */
#include "../src/graphics/rose3d.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

/** @brief 相同种子应生成相同粒子；每个字段均保持有限。 */
static void test_repeatable_reset(void)
{
    static Rose3dSystem first;
    static Rose3dSystem second;
    rose3d_reset(&first, UINT32_C(71));
    rose3d_reset(&second, UINT32_C(71));
    assert(first.random_state == second.random_state);
    for (size_t i = 0; i < ROSE3D_PARTICLE_COUNT; ++i) {
        const Rose3dParticle *a = &first.particles[i];
        const Rose3dParticle *b = &second.particles[i];
        assert(a->x == b->x && a->y == b->y && a->z == b->z);
        assert(a->phase == b->phase && a->lifetime == b->lifetime);
        assert(isfinite(a->x) && isfinite(a->y) && isfinite(a->z));
        assert(a->lifetime >= 2.0 && a->lifetime <= 10.0);
        assert(a->red >= 205 && a->green >= 126 && a->blue >= 148);
    }
}

/** @brief 固定步进应使两个同种子系统保持一致并产生缓慢漂移。 */
static void test_fixed_step(void)
{
    static Rose3dSystem first;
    static Rose3dSystem second;
    rose3d_reset(&first, UINT32_C(19));
    rose3d_reset(&second, UINT32_C(19));
    double initial_x = first.particles[0].x;
    double initial_y = first.particles[0].y;
    double initial_yaw = first.yaw;
    for (int i = 0; i < 180; ++i) {
        rose3d_step(&first);
        rose3d_step(&second);
    }
    assert(fabs(first.time - 3.0) < 1e-12);
    assert(first.particles[0].x == second.particles[0].x);
    assert(first.particles[0].y == second.particles[0].y);
    assert(first.particles[0].x != initial_x || first.particles[0].y != initial_y);
    assert(first.yaw > initial_yaw);
}

/** @brief 绘制只修改指定竖屏矩形，重复绘制结果一致且粒子云可见。 */
static void test_portrait_render(void)
{
    static Rose3dSystem system;
    static GfxCanvas canvas;
    rose3d_reset(&system, UINT32_C(42));
    rose3d_draw(&system, &canvas);
    uint32_t center = canvas.pixels[ROSE3D_VIEW_HEIGHT / 2][ROSE3D_VIEW_X +
                                                            ROSE3D_VIEW_WIDTH / 2];
    assert(center != gfx_rgb(5, 6, 13));
    assert(canvas.pixels[0][0] == gfx_rgb(5, 6, 13));
    assert(canvas.pixels[0][GFX_WIDTH - 1] == gfx_rgb(5, 6, 13));
    rose3d_draw(&system, &canvas);
    assert(canvas.pixels[ROSE3D_VIEW_HEIGHT / 2][ROSE3D_VIEW_X + ROSE3D_VIEW_WIDTH / 2] == center);
    rose3d_step(&system);
    rose3d_draw(&system, &canvas);
    assert(canvas.pixels[ROSE3D_VIEW_HEIGHT / 2][ROSE3D_VIEW_X + ROSE3D_VIEW_WIDTH / 2] != center);
}

/** @brief 运行纯 C 玫瑰核心断言。 */
int main(void)
{
    test_repeatable_reset();
    test_fixed_step();
    test_portrait_render();
    puts("三维粒子玫瑰核心测试通过。");
    return 0;
}
