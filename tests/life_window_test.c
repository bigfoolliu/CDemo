/** @file @brief SDL dummy 下测试输入状态、坐标映射、时间与有界主循环。 */
#define main life_demo_main
#include "../src/graphics/life_demo.c"
#undef main
#include <assert.h>
#include "bmp_window_helpers.h"

/**
 * @brief 发送一次合成键盘事件。
 * @param state 可读写演示状态。
 * @param key SDL 键码。
 * @return 是否要求丢弃本帧时间。
 */
static bool key(LifeDemoState *state, SDL_Keycode key)
{
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = key;
    return handle_event(state, &event, 640, 480);
}

/** @brief 验证暂停、预设、编辑、时间截断与退出。 */
static void test_controls(void)
{
    LifeDemoState state = {.running = true};
    load_preset(&state, LIFE_GLIDER);
    SDL_Event screenshot = {.type = SDL_KEYDOWN};
    screenshot.key.keysym.sym = SDLK_s; screenshot.key.repeat = 1;
    handle_event(&state, &screenshot, 640, 480);
    assert(!state.capture);
    screenshot.key.repeat = 0;
    handle_event(&state, &screenshot, 640, 480);
    assert(state.capture && state.world.generation == 0);
    state.capture = false;
    advance_time(&state, 0.06);
    assert(state.world.generation == 0);
    advance_time(&state, 0.06);
    assert(state.world.generation == 1 && state.accumulator < 0.1);
    assert(key(&state, SDLK_SPACE) && state.paused && state.accumulator == 0);
    advance_time(&state, 100);
    assert(state.world.generation == 1);
    key(&state, SDLK_n);
    assert(state.world.generation == 2);
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_n; event.key.repeat = 1;
    handle_event(&state, &event, 640, 480);
    assert(state.world.generation == 2);
    key(&state, SDLK_2);
    assert(state.paused && state.preset == LIFE_BLINKER && state.world.generation == 0);
    key(&state, SDLK_3);
    LifeWorld random = state.world;
    key(&state, SDLK_c);
    assert(state.paused && state.custom && life_count(&state.world) == 0);
    event = (SDL_Event){.type = SDL_MOUSEBUTTONDOWN};
    event.button.button = SDL_BUTTON_LEFT;
    event.button.x = 639; event.button.y = 479;
    assert(handle_event(&state, &event, 640, 480));
    assert(life_get(&state.world, 79, 59) && strcmp(pattern_name(&state), "custom") == 0);
    event.button.x = 0; event.button.y = 0;
    assert(handle_event(&state, &event, 320, 240) && life_get(&state.world, 0, 0));
    /* 非标准窗口尺寸也正确映射，右边界本身则越界。 */
    event.button.x = 799; event.button.y = 599;
    assert(handle_event(&state, &event, 800, 600) && !life_get(&state.world, 79, 59));
    event.button.x = 800;
    assert(!handle_event(&state, &event, 800, 600));
    event.button.x = -1;
    assert(!handle_event(&state, &event, 800, 600));
    event.button.x = 1;
    assert(!handle_event(&state, &event, 0, 600));
    event.button.button = SDL_BUTTON_RIGHT;
    assert(!handle_event(&state, &event, 800, 600));
    key(&state, SDLK_r);
    assert(!state.custom && state.paused && state.preset == LIFE_RANDOM);
    for (int y = 0; y < LIFE_HEIGHT; ++y) {
        for (int x = 0; x < LIFE_WIDTH; ++x) { assert(life_get(&state.world, x, y) == life_get(&random, x, y)); }
    }
    key(&state, SDLK_1);
    assert(life_count(&state.world) == 5);
    key(&state, SDLK_SPACE);
    assert(!state.paused && state.accumulator == 0);
    key(&state, SDLK_n);
    assert(state.world.generation == 0);
    event.button.button = SDL_BUTTON_LEFT;
    assert(!handle_event(&state, &event, 800, 600));
    advance_time(&state, NAN); advance_time(&state, -1);
    assert(state.world.generation == 0);
    advance_time(&state, 100);
    assert(state.world.generation == 1);
    key(&state, SDLK_r);
    assert(!state.paused && state.world.generation == 0 && state.accumulator == 0);
    key(&state, SDLK_ESCAPE);
    assert(!state.running);
    state.running = true;
    event.type = SDL_QUIT;
    handle_event(&state, &event, 640, 480);
    assert(!state.running);
}

/** @brief 验证有界窗口主循环及非法参数。 @return 成功返回零。 */
int main(void)
{
    test_capture_loop(life_demo_main);
    test_controls();
    char *args[] = {"life_demo", "--frames", "2", nullptr};
    assert(life_demo_main(3, args) == EXIT_SUCCESS);
    const char *invalid[] = {"0", "10001", "2x", "-1", "999999999999999999999999999999"};
    for (size_t i = 0; i < sizeof invalid / sizeof invalid[0]; ++i) {
        args[2] = (char *)invalid[i];
        assert(life_demo_main(3, args) == EXIT_FAILURE);
    }
    puts("生命游戏窗口控制测试通过。");
    return 0;
}
