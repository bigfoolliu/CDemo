/** @file @brief SDL dummy 下验证鼠标映射、重绘标记、缩放边界与截图主循环。 */
#define main mandelbrot_demo_main
#include "../src/graphics/mandelbrot_demo.c"
#undef main
#include <assert.h>
#include <math.h>
#include "bmp_window_helpers.h"

/** @brief 合成键盘事件。 @param state 可读写状态。 @param code 键码。 @return 视图是否改变。 */
static bool key(MandelbrotDemoState *state, SDL_Keycode code)
{
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = code;
    return handle_event(state, &event, 640, 480, 320, 240);
}

/** @brief 验证按键、鼠标、自然滚动方向、越界反馈及缓存画布标记。 */
static void test_controls(void)
{
    MandelbrotDemoState state = {.running = true};
    mandelbrot_reset(&state.view);
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_PLUS; event.key.repeat = 1;
    handle_event(&state, &event, 640, 480, 0, 0);
    assert(state.view.max_iterations == 128 && !state.dirty);
    assert(key(&state, SDLK_PLUS) && state.view.max_iterations == 256 && state.dirty);
    key(&state, SDLK_PLUS); key(&state, SDLK_PLUS);
    state.dirty = false;
    assert(!key(&state, SDLK_PLUS) && state.view.max_iterations == 1024 && state.limited && !state.dirty);
    for (int i = 0; i < 5; ++i) { key(&state, SDLK_MINUS); }
    assert(state.view.max_iterations == 32 && !key(&state, SDLK_MINUS));
    assert(key(&state, SDLK_r) && state.view.span == 3.5 && state.view.max_iterations == 128 && !state.limited);
    assert(key(&state, SDLK_RIGHT) && state.view.center_real == 0.375);
    assert(key(&state, SDLK_LEFT) && state.view.center_real == -0.5);
    assert(key(&state, SDLK_UP) && state.view.center_imag == 0.65625);
    assert(key(&state, SDLK_DOWN) && state.view.center_imag == 0);
    event = (SDL_Event){.type = SDL_MOUSEBUTTONDOWN};
    event.button.button = SDL_BUTTON_LEFT; event.button.x = 200; event.button.y = 100;
    GfxPoint before, after;
    assert(mandelbrot_map(&state.view, 200.5 / 800, 100.5 / 600, &before));
    assert(handle_event(&state, &event, 800, 600, 0, 0) && state.view.span == 1.75);
    assert(mandelbrot_map(&state.view, 200.5 / 800, 100.5 / 600, &after));
    assert(fabs(before.x - after.x) < 1e-14 && fabs(before.y - after.y) < 1e-14);
    event.button.button = SDL_BUTTON_RIGHT;
    assert(handle_event(&state, &event, 800, 600, 0, 0) && state.view.span == 3.5);
    state.dirty = false;
    assert(!handle_event(&state, &event, 800, 600, 0, 0) && state.limited && !state.dirty);
    event.button.button = SDL_BUTTON_LEFT; event.button.x = 800;
    assert(!handle_event(&state, &event, 800, 600, 0, 0));
    event.button.x = -1;
    assert(!handle_event(&state, &event, 800, 600, 0, 0));
    event.button.x = 1;
    assert(!handle_event(&state, &event, 0, 600, 0, 0));
    key(&state, SDLK_r);
    event = (SDL_Event){.type = SDL_MOUSEWHEEL};
    event.wheel.y = 1;
    assert(handle_event(&state, &event, 640, 480, 320, 240) && state.view.span == 1.75);
    event.wheel.direction = SDL_MOUSEWHEEL_FLIPPED;
    assert(handle_event(&state, &event, 640, 480, 320, 240) && state.view.span == 3.5);
    event.wheel.y = 0;
    assert(!handle_event(&state, &event, 640, 480, 320, 240));
    key(&state, SDLK_s); assert(state.capture);
    key(&state, SDLK_ESCAPE); assert(!state.running);
    state.running = true; event.type = SDL_QUIT;
    handle_event(&state, &event, 640, 480, 0, 0); assert(!state.running);
}

/** @brief 验证控制、自动退出、实际 BMP 截图和导出失败不退出。 @return 成功为零。 */
int main(void)
{
    test_controls(); test_capture_loop(mandelbrot_demo_main);
    char *args[] = {"mandelbrot_demo", "--frames", "2", nullptr};
    assert(mandelbrot_demo_main(3, args) == EXIT_SUCCESS);
    args[2] = "0"; assert(mandelbrot_demo_main(3, args) == EXIT_FAILURE);
    args[2] = "2x"; assert(mandelbrot_demo_main(3, args) == EXIT_FAILURE);
    args[2] = "10001"; assert(mandelbrot_demo_main(3, args) == EXIT_FAILURE);
    puts("曼德勃罗窗口控制测试通过。");
    return 0;
}
