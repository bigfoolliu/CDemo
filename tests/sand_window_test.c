/** @file @brief 连续笔刷、拖动路径、暂停计时及截图主循环验证。 */
#define main sand_demo_entry
#include "../src/graphics/sand_demo.c"
#undef main
#include "bmp_window_helpers.h"
#include <assert.h>

/** @brief 构造按键事件。 @param key SDL 键码。 @return 无重复的按下事件。 */
static SDL_Event key_event(SDL_Keycode key)
{
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = key;
    return event;
}

/** @brief 检查坐标映射、快速拖动、连续补充、擦除及输入取消。 */
static void test_brush(void)
{
    static SandDemoState state;
    state.material = SAND_GRAIN;
    state.radius = 0;
    reset_scene(&state, false);
    SDL_Event event = {.type = SDL_MOUSEBUTTONDOWN};
    event.button.button = SDL_BUTTON_LEFT;
    event.button.x = 160;
    event.button.y = 80;
    handle_event(&state, &event, 640, 480);
    assert(state.world.cells[20][40] == SAND_GRAIN && state.painting);
    event = (SDL_Event){.type = SDL_MOUSEMOTION};
    event.motion.state = SDL_BUTTON_LMASK;
    event.motion.x = 200;
    event.motion.y = 80;
    handle_event(&state, &event, 640, 480);
    for (int x = 40; x <= 50; ++x) { assert(state.world.cells[20][x] == SAND_GRAIN); }
    advance_time(&state, 2.0 / SAND_HZ);
    assert(state.world.steps == 2);
    assert(state.world.cells[21][50] == SAND_GRAIN && state.world.cells[22][50] == SAND_GRAIN);
    event.type = SDL_MOUSEBUTTONUP;
    event.button.button = SDL_BUTTON_LEFT;
    handle_event(&state, &event, 640, 480);
    assert(!state.painting);
    state.paused = true;
    event = (SDL_Event){.type = SDL_MOUSEBUTTONDOWN};
    event.button.button = SDL_BUTTON_RIGHT;
    event.button.x = 100;
    event.button.y = 44;
    handle_event(&state, &event, 320, 240);
    assert(state.erasing && state.world.cells[22][50] == SAND_AIR);
    event.type = SDL_WINDOWEVENT;
    event.window.event = SDL_WINDOWEVENT_FOCUS_LOST;
    handle_event(&state, &event, 640, 480);
    assert(!state.erasing && !state.painting);
    state.painting = true;
    event.window.event = SDL_WINDOWEVENT_LEAVE;
    handle_event(&state, &event, 640, 480);
    assert(!state.painting);
    state.painting = true;
    event = (SDL_Event){.type = SDL_MOUSEMOTION};
    handle_event(&state, &event, 640, 480);
    assert(!state.painting);
    state.painting = true;
    paint_mouse(&state, -1, 0, 640, 480, false);
    assert(!state.painting);
    state.painting = true;
    paint_mouse(&state, 1, 1, 0, 0, false);
    assert(!state.painting);
}

/** @brief 控制键不因自动重复触发；固定时间步与暂停不积累时间。 */
static void test_controls(void)
{
    static SandDemoState state;
    state.running = true;
    state.radius = 3;
    reset_scene(&state, true);
    SDL_Event event = key_event(SDLK_SPACE);
    event.key.repeat = 1;
    assert(!handle_event(&state, &event, 640, 480) && !state.paused);
    event.key.repeat = 0;
    assert(handle_event(&state, &event, 640, 480) && state.paused);
    event = key_event(SDLK_n);
    handle_event(&state, &event, 640, 480);
    assert(state.world.steps == 1);
    advance_time(&state, 10);
    assert(state.world.steps == 1 && state.accumulator == 0);
    event = key_event(SDLK_c);
    assert(handle_event(&state, &event, 640, 480));
    assert(state.world.cells[20][50] == SAND_AIR && state.paused && state.world.steps == 0);
    event = key_event(SDLK_2);
    handle_event(&state, &event, 640, 480);
    assert(state.material == SAND_WATER);
    event = key_event(SDLK_3);
    handle_event(&state, &event, 640, 480);
    assert(state.material == SAND_WALL);
    event = key_event(SDLK_LEFTBRACKET);
    for (int i = 0; i < 20; ++i) { handle_event(&state, &event, 640, 480); }
    assert(state.radius == 1);
    event = key_event(SDLK_RIGHTBRACKET);
    for (int i = 0; i < 20; ++i) { handle_event(&state, &event, 640, 480); }
    assert(state.radius == SAND_MAX_BRUSH);
    event = key_event(SDLK_r);
    assert(handle_event(&state, &event, 640, 480) && state.world.cells[20][50] == SAND_GRAIN);
    state.paused = false;
    advance_time(&state, NAN);
    advance_time(&state, -1);
    assert(state.world.steps == 0);
    advance_time(&state, 0.5 / SAND_HZ);
    assert(state.world.steps == 0);
    advance_time(&state, 0.5 / SAND_HZ);
    assert(state.world.steps == 1);
    advance_time(&state, 10);
    assert(state.world.steps == 7);
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

/** @brief 运行合成输入与 dummy 驱动主循环。 @return 成功返回零。 */
int main(void)
{
    test_brush();
    test_controls();
    test_capture_loop(sand_demo_entry);
    char *bad[] = {"sand_demo", "--frames", "0", nullptr};
    assert(sand_demo_entry(3, bad) == EXIT_FAILURE);
    puts("落沙窗口测试通过。");
    return 0;
}
