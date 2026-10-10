/** @file @brief SDL dummy 驱动下验证控制、时间累积与有界主循环。 */
#define main particles_demo_main
#include "../src/graphics/particles_demo.c"
#undef main
#include <assert.h>

/** @brief 验证暂停、单步、重置与时间截断的可观察结果。 */
static void test_controls_and_time(void)
{
    ParticleDemoState state = {.running = true};
    particles_reset(&state.particles, 42);
    advance_time(&state, 0.01);
    assert(particles_count(&state.particles) == 0);
    advance_time(&state, 0.01);
    assert(particles_count(&state.particles) == 2);
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_SPACE;
    assert(handle_event(&state, &event) && state.paused && state.accumulator == 0);
    uint32_t rng = state.particles.rng;
    advance_time(&state, 100);
    assert(state.particles.rng == rng && state.accumulator == 0);
    event.key.keysym.sym = SDLK_n;
    assert(!handle_event(&state, &event));
    assert(particles_count(&state.particles) == 4);
    event.key.repeat = 1;
    handle_event(&state, &event);
    assert(particles_count(&state.particles) == 4);
    event.key.repeat = 0;
    event.key.keysym.sym = SDLK_r;
    assert(handle_event(&state, &event) && state.paused);
    assert(particles_count(&state.particles) == 0 && state.particles.rng == 42);
    event.key.keysym.sym = SDLK_SPACE;
    assert(handle_event(&state, &event) && !state.paused);
    event.key.keysym.sym = SDLK_n;
    handle_event(&state, &event);
    assert(particles_count(&state.particles) == 0);
    advance_time(&state, NAN);
    advance_time(&state, -1);
    assert(particles_count(&state.particles) == 0);
    advance_time(&state, 100);
    assert(particles_count(&state.particles) == 12);
    assert(state.accumulator < 1.0 / PARTICLE_HZ);
    event.key.keysym.sym = SDLK_ESCAPE;
    handle_event(&state, &event);
    assert(!state.running);
    state.running = true;
    event.type = SDL_QUIT;
    handle_event(&state, &event);
    assert(!state.running);
}

/** @brief 无实体窗口运行演示并验证非法命令行参数。 */
int main(void)
{
    test_controls_and_time();
    char *args[] = {"particles_demo", "--frames", "2", nullptr};
    assert(particles_demo_main(3, args) == EXIT_SUCCESS);
    args[2] = "0";
    assert(particles_demo_main(3, args) == EXIT_FAILURE);
    args[2] = "2x";
    assert(particles_demo_main(3, args) == EXIT_FAILURE);
    args[1] = "--unknown";
    assert(particles_demo_main(3, args) == EXIT_FAILURE);
    puts("粒子窗口控制测试通过。");
    return 0;
}
