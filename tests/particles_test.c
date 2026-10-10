/** @file @brief 无窗口验证粒子运动、对象池和颜色裁切。 */
#include "../src/graphics/particles.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

/** @brief 构造寿命充足的有效粒子，便于单独验证更新。 */
static Particle sample(void)
{
    return (Particle){.x = 100, .y = 100, .vx = 30, .vy = -60,
        .remaining = 150, .lifetime = 150, .green = 200};
}

/** @brief 同种子重复模拟必须逐字段一致，不比较结构体填充字节。 */
static void test_repeatability(void)
{
    ParticleSystem a, b;
    particles_reset(&a, 42);
    particles_reset(&b, 42);
    for (int step = 0; step < 2000; ++step) {
        particles_step(&a);
        particles_step(&b);
        assert(a.rng == b.rng);
        assert(particles_count(&a) <= PARTICLE_CAPACITY);
        for (size_t i = 0; i < PARTICLE_CAPACITY; ++i) {
            Particle p = a.pool[i], q = b.pool[i];
            assert(p.x == q.x && p.y == q.y && p.vx == q.vx && p.vy == q.vy);
            assert(p.remaining == q.remaining && p.lifetime == q.lifetime && p.green == q.green);
            if (p.remaining != 0) {
                assert(p.x >= 0 && p.x < GFX_WIDTH && p.y >= 0 && p.y < GFX_HEIGHT);
                assert(p.remaining <= p.lifetime && p.lifetime >= 90 && p.lifetime <= 150);
            }
        }
    }
    particles_reset(&a, 42);
    assert(particles_count(&a) == 0 && a.rng == 42);
    particles_step(&a);
    assert(particles_count(&a) == 2);
    for (size_t i = 0; i < 2; ++i) {
        Particle p = a.pool[i];
        assert(p.x == 160 && p.y == 210 && p.remaining == p.lifetime);
        assert(p.vx >= -35 && p.vx <= 35 && p.vy >= -110 && p.vy <= -70);
        assert(p.green >= 180 && p.green <= 230);
    }
}

/** @brief 满池不覆盖存活粒子；到期和越界槽位能在同一步复用。 */
static void test_pool_and_motion(void)
{
    ParticleSystem system;
    particles_reset(&system, 42);
    for (size_t i = 0; i < PARTICLE_CAPACITY; ++i) { system.pool[i] = sample(); }
    particles_step(&system);
    assert(system.rng == 42 && particles_count(&system) == PARTICLE_CAPACITY);
    assert(fabs(system.pool[0].vy - (-58.5)) < 1e-12);
    assert(fabs(system.pool[0].x - 100.5) < 1e-12);
    assert(fabs(system.pool[0].y - 99.025) < 1e-12);
    assert(system.pool[0].remaining == 149);
    system.pool[0].remaining = 1;
    system.pool[1].x = 319.9;
    system.pool[1].vx = 60;
    particles_step(&system);
    assert(system.pool[0].x == 160 && system.pool[1].x == 160);
    assert(system.pool[0].remaining == system.pool[0].lifetime);
    /* 四条边都回收；每步最多补充两个，所以两个槽位仍为空。 */
    system.pool[0].x = 0; system.pool[0].vx = -60;
    system.pool[1].x = 319.9; system.pool[1].vx = 60;
    system.pool[2].y = 0; system.pool[2].vy = -60;
    system.pool[3].y = 239.9; system.pool[3].vy = 60;
    particles_step(&system);
    assert(system.pool[2].remaining == 0 && system.pool[3].remaining == 0);
    assert(particles_count(&system) == PARTICLE_CAPACITY - 2);
    particles_step(&system);
    assert(particles_count(&system) == PARTICLE_CAPACITY);
}

/** @brief 寿命改变亮度，2×2 绘制在右下角安全裁切。 */
static void test_pixels(void)
{
    static GfxCanvas canvas;
    ParticleSystem system;
    particles_reset(&system, 42);
    system.pool[0] = sample();
    system.pool[0].x = 10.5; system.pool[0].y = 20.5;
    gfx_clear(&canvas, 0);
    particles_draw(&system, &canvas);
    assert(canvas.pixels[20][10] == gfx_rgb(255, 200, 60));
    assert(canvas.pixels[21][11] == gfx_rgb(255, 200, 60));
    assert(canvas.pixels[20][12] == 0);
    system.pool[0].remaining = 75;
    particles_draw(&system, &canvas);
    assert(canvas.pixels[20][10] == gfx_rgb(127, 100, 30));
    system.pool[0].x = GFX_WIDTH - 0.1; system.pool[0].y = GFX_HEIGHT - 0.1;
    particles_draw(&system, &canvas);
    assert(canvas.pixels[GFX_HEIGHT - 1][GFX_WIDTH - 1] == gfx_rgb(127, 100, 30));
    system.pool[0].remaining = 0;
    gfx_clear(&canvas, 0);
    particles_draw(&system, &canvas);
    assert(canvas.pixels[20][10] == 0);
}

/** @brief 运行全部纯 C 粒子断言。 */
int main(void)
{
    test_repeatability();
    test_pool_and_motion();
    test_pixels();
    puts("粒子核心测试通过。");
    return 0;
}
