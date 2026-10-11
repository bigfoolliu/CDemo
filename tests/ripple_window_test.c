/** @file @brief 波纹窗口的输入、固定步长、暂停及截图主循环验证。 */
#define main ripple_demo_entry
#include "../src/graphics/ripple_demo.c"
#undef main
#include "bmp_window_helpers.h"
#include <assert.h>

/** @brief 创建按键事件。 @param key SDL 键码。 @return 无重复的按下事件。 */
static SDL_Event key_event(SDL_Keycode key)
{
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = key;
    return event;
}

/** @brief 验证暂停、单步、重置、截图与有效坐标映射。 */
static void test_input(void)
{
    static RippleDemoState state;
    state.running = true;
    reset_scene(&state, false);
    SDL_Event event = key_event(SDLK_SPACE);
    event.key.repeat = 1;
    assert(!handle_event(&state, &event, 640, 480) && !state.paused);
    event.key.repeat = 0;
    assert(handle_event(&state, &event, 640, 480) && state.paused);
    event = key_event(SDLK_n);
    handle_event(&state, &event, 640, 480);
    assert(state.world.steps == 1);
    event = (SDL_Event){.type = SDL_MOUSEBUTTONDOWN};
    event.button.button = SDL_BUTTON_LEFT;
    event.button.x = 320;
    event.button.y = 240;
    handle_event(&state, &event, 640, 480);
    assert(state.world.height[state.world.current][60][80] == 8);
    reset_scene(&state, false);
    event.button.button = SDL_BUTTON_RIGHT;
    event.button.x = 160;
    event.button.y = 120;
    handle_event(&state, &event, 320, 240);
    assert(state.world.height[state.world.current][60][80] == -8);
    reset_scene(&state, false);
    event.button.x = -1;
    handle_event(&state, &event, 320, 240);
    event.button.x = 320;
    handle_event(&state, &event, 320, 240);
    event.button.x = 160;
    handle_event(&state, &event, 0, 0);
    assert(state.world.height[0][60][80] == 0);
    event = key_event(SDLK_r);
    assert(handle_event(&state, &event, 640, 480));
    assert(state.world.steps == 0 && state.world.height[0][60][80] == 8 && state.paused);
    event = key_event(SDLK_c);
    assert(handle_event(&state, &event, 640, 480));
    assert(state.world.height[0][60][80] == 0 && state.paused);
    event = key_event(SDLK_s);
    handle_event(&state, &event, 640, 480);
    assert(state.capture);
    event = key_event(SDLK_ESCAPE);
    handle_event(&state, &event, 640, 480);
    assert(!state.running);
    state.running = true;
    event.type = SDL_QUIT;
    handle_event(&state, &event, 640, 480);
    assert(!state.running);
}

/** @brief 分段时间累积应一致；暂停不积累时间，长停顿限制追赶。 */
static void test_time(void)
{
    static RippleDemoState a, b;
    reset_scene(&a, true);
    reset_scene(&b, true);
    advance_time(&a, 1.0 / RIPPLE_HZ);
    advance_time(&b, 0.5 / RIPPLE_HZ);
    assert(b.world.steps == 0);
    advance_time(&b, 0.5 / RIPPLE_HZ);
    assert(a.world.steps == 1 && b.world.steps == 1);
    assert(memcmp(&a.world, &b.world, sizeof a.world) == 0);
    advance_time(&a, NAN);
    advance_time(&a, -1);
    assert(a.world.steps == 1);
    advance_time(&a, 10);
    assert(a.world.steps == 7);
    a.paused = true;
    advance_time(&a, 10);
    assert(a.world.steps == 7 && a.accumulator == 0);
    a.paused = false;
    SDL_Event event = key_event(SDLK_n);
    handle_event(&a, &event, 640, 480);
    assert(a.world.steps == 7);
}

/** @brief 运行合成事件与 dummy 驱动主循环。 @return 成功返回零。 */
int main(void)
{
    test_input();
    test_time();
    test_capture_loop(ripple_demo_entry);
    char *bad[] = {"ripple_demo", "--frames", "0", nullptr};
    assert(ripple_demo_entry(3, bad) == EXIT_FAILURE);
    puts("波纹窗口测试通过。");
    return 0;
}
