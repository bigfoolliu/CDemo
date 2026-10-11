/** @file @brief 合成 SDL 事件验证拖动、边界、重置、截图及真实主循环。 */
#define main bezier_demo_entry
#include "../src/graphics/bezier_demo.c"
#undef main
#include "bmp_window_helpers.h"
#include <assert.h>

/** @brief 构造鼠标按下事件。
 * @param x 窗口横坐标。 @param y 窗口纵坐标。
 * @return 左键事件。
 */
static SDL_Event press(int x, int y)
{
    SDL_Event event = {0};
    event.type = SDL_MOUSEBUTTONDOWN;
    event.button.button = SDL_BUTTON_LEFT;
    event.button.x = x;
    event.button.y = y;
    return event;
}

/** @brief 验证坐标映射、抓取偏移、缓存标记和拖动取消。 */
static void test_dragging(void)
{
    BezierDemoState state = {.running = true, .selected = -1};
    bezier_reset(&state.curve);
    SDL_Event event = press(64, 380); /* 640×480 下接近 P0，有点击偏移。 */
    handle_event(&state, &event, 640, 480);
    assert(state.selected == 0 && state.dragging && state.dirty);
    assert(state.curve.points[0].x == 30 && state.curve.points[0].y == 190);
    state.dirty = false;
    event = (SDL_Event){0};
    event.type = SDL_MOUSEMOTION;
    event.motion.state = SDL_BUTTON_LMASK;
    event.motion.x = 64;
    event.motion.y = 380;
    handle_event(&state, &event, 640, 480);
    assert(!state.dirty); /* 未移动鼠标不重绘，也不跳到鼠标中心。 */
    event.motion.x = 84;
    event.motion.y = 400;
    handle_event(&state, &event, 640, 480);
    assert(state.curve.points[0].x == 40 && state.curve.points[0].y == 200);
    assert(state.dirty);
    event.motion.x = -100;
    event.motion.y = -100;
    handle_event(&state, &event, 640, 480);
    assert(state.curve.points[0].x == 0 && state.curve.points[0].y == 0);
    event.motion.x = 1000;
    event.motion.y = 1000;
    handle_event(&state, &event, 640, 480);
    assert(state.curve.points[0].x == 319 && state.curve.points[0].y == 239);
    event.motion.state = 0;
    handle_event(&state, &event, 640, 480);
    assert(!state.dragging);
    const Uint8 cancellations[] = {SDL_WINDOWEVENT_FOCUS_LOST, SDL_WINDOWEVENT_LEAVE};
    for (unsigned i = 0; i < sizeof cancellations / sizeof cancellations[0]; ++i) {
        state.dragging = true;
        event.type = SDL_WINDOWEVENT;
        event.window.event = cancellations[i];
        handle_event(&state, &event, 640, 480);
        assert(!state.dragging);
    }
    state.dragging = true;
    event.type = SDL_MOUSEBUTTONUP;
    event.button.button = SDL_BUTTON_LEFT;
    handle_event(&state, &event, 640, 480);
    assert(!state.dragging);
    bezier_reset(&state.curve);
    event = press(80, 30); /* 1:1 缩放仍正确命中。 */
    handle_event(&state, &event, 320, 240);
    assert(state.selected == 1 && state.dragging);
    event = press(0, 0);
    handle_event(&state, &event, 0, 0);
    assert(!state.dragging);
    handle_event(&state, &event, 320, 240);
    assert(state.selected == -1);
}

/** @brief 重复键不触发动作，截图与重置各有独立状态。 */
static void test_keys(void)
{
    BezierDemoState state = {.running = true, .selected = 2, .dragging = true};
    bezier_reset(&state.curve);
    state.curve.points[0].x = 100;
    SDL_Event event = {0};
    event.type = SDL_KEYDOWN;
    event.key.keysym.sym = SDLK_r;
    event.key.repeat = 1;
    handle_event(&state, &event, 640, 480);
    assert(state.curve.points[0].x == 100 && !state.dirty);
    event.key.repeat = 0;
    handle_event(&state, &event, 640, 480);
    assert(state.curve.points[0].x == 30 && state.selected == -1 && !state.dragging && state.dirty);
    state.dirty = false;
    event.key.keysym.sym = SDLK_s;
    handle_event(&state, &event, 640, 480);
    assert(state.capture && !state.dirty);
    event.key.keysym.sym = SDLK_ESCAPE;
    handle_event(&state, &event, 640, 480);
    assert(!state.running);
    state.running = true;
    event.type = SDL_QUIT;
    handle_event(&state, &event, 640, 480);
    assert(!state.running);
}

/** @brief 运行事件测试与 dummy 视频驱动下的主循环。 @return 成功返回零。 */
int main(void)
{
    test_dragging();
    test_keys();
    test_capture_loop(bezier_demo_entry);
    char *bad[] = {"bezier_demo", "--frames", "0", nullptr};
    assert(bezier_demo_entry(3, bad) == EXIT_FAILURE);
    puts("贝塞尔曲线窗口测试通过。");
    return 0;
}
