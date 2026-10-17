/** @file @brief SDL dummy 下检查排序窗口控件、图例、计时与截图。 */
#define main sorting_demo_main
#include "../src/graphics/sorting_demo.c"
#undef main
#include <assert.h>
#include "bmp_window_helpers.h"

/**
 * @brief 合成一个按键事件。
 * @param state 可读写窗口状态。
 * @param code 要发送的 SDL 按键代码。
 * @return 事件要求丢弃本帧时间时为 true。
 */
static bool key(SortingDemoState *state, SDL_Keycode code)
{
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = code;
    return handle_event(state, &event);
}

/** @brief 检查图例的颜色色块和文字标签确实绘制到画布。 */
static void test_legend(void)
{
    static GfxCanvas canvas;
    SortingDemoState state = {.seed = 1};
    assert(restart_sort(&state, SORTING_BUBBLE, state.seed));
    draw_scene(&state, &canvas);

    static const int positions[][2] = {{12, 211}, {71, 211}, {130, 211}, {189, 211},
                                       {248, 211}};
    const uint32_t colors[] = {
        gfx_rgb(70, 145, 230), gfx_rgb(235, 155, 55), gfx_rgb(195, 115, 225),
        gfx_rgb(65, 215, 135), gfx_rgb(55, 105, 80)
    };
    for (size_t item = 0; item < sizeof colors / sizeof colors[0]; ++item) {
        assert(canvas.pixels[positions[item][1]][positions[item][0]] == colors[item]);
    }
    assert(canvas.pixels[212][23] == gfx_rgb(228, 232, 239));

    assert(restart_sort(&state, SORTING_QUICK, state.seed));
    draw_scene(&state, &canvas);
    assert(canvas.pixels[9][13] == gfx_rgb(228, 232, 239));
    sorting_step(&state.sorting);
    draw_scene(&state, &canvas);
    uint8_t pivot_value = state.sorting.items[state.sorting.pivot_index].value;
    int pivot_height = (int)pivot_value * SORT_BAR_MAX_HEIGHT / SORTING_VALUE_MAX;
    if (pivot_height == 0) {
        pivot_height = 1;
    }
    int pivot_y = SORT_BAR_BASELINE - pivot_height;
    int pivot_x = SORT_BAR_LEFT + (int)state.sorting.pivot_index *
                  (SORT_BAR_WIDTH + SORT_BAR_GAP);
    assert(canvas.pixels[pivot_y][pivot_x] == gfx_rgb(195, 115, 225));
}

/** @brief 检查算法切换、暂停单步、重置和事件边界。 */
static void test_controls(void)
{
    SortingDemoState state = {.running = true, .seed = 1};
    assert(restart_sort(&state, SORTING_BUBBLE, state.seed));
    SortingItem original[SORTING_ITEM_COUNT];
    memcpy(original, state.sorting.items, sizeof original);
    assert(key(&state, SDLK_2));
    assert(state.sorting.algorithm == SORTING_INSERTION);
    assert(state.sorting.comparisons == 0);
    assert(memcmp(original, state.sorting.items, sizeof original) == 0);
    assert(key(&state, SDLK_3));
    assert(state.sorting.algorithm == SORTING_SELECTION);
    assert(key(&state, SDLK_4));
    assert(state.sorting.algorithm == SORTING_QUICK);
    assert(state.sorting.comparisons == 0);
    assert(memcmp(original, state.sorting.items, sizeof original) == 0);
    assert(key(&state, SDLK_KP_4));
    assert(state.sorting.algorithm == SORTING_QUICK);
    assert(state.sorting.comparisons == 0);
    assert(key(&state, SDLK_1));
    assert(state.sorting.algorithm == SORTING_BUBBLE);
    assert(key(&state, SDLK_SPACE) && state.paused && state.accumulator == 0);
    advance_time(&state, 1.0);
    assert(state.sorting.comparisons == 0);
    key(&state, SDLK_n);
    assert(state.sorting.comparisons == 1);
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_n;
    event.key.repeat = 1;
    handle_event(&state, &event);
    assert(state.sorting.comparisons == 1);

    uint32_t previous_seed = state.seed;
    key(&state, SDLK_r);
    assert(state.seed == previous_seed + 1);
    assert(state.sorting.comparisons == 0 && state.paused);
    advance_time(&state, NAN);
    advance_time(&state, -1.0);
    assert(state.sorting.comparisons == 0);
    key(&state, SDLK_SPACE);
    advance_time(&state, 0.1);
    assert(state.sorting.comparisons > 0 && state.sorting.comparisons <= 6);
    key(&state, SDLK_s);
    assert(state.capture);
    key(&state, SDLK_ESCAPE);
    assert(!state.running);
    state.running = true;
    event.type = SDL_QUIT;
    handle_event(&state, &event);
    assert(!state.running);
}

/**
 * @brief 执行窗口控件、截图、有界主循环和参数验证。
 * @return 所有断言通过时返回零。
 */
int main(void)
{
    test_legend();
    test_controls();
    test_capture_loop(sorting_demo_main);
    char *arguments[] = {"sorting_demo", "--frames", "2", nullptr};
    assert(sorting_demo_main(3, arguments) == EXIT_SUCCESS);
    arguments[2] = "0";
    assert(sorting_demo_main(3, arguments) == EXIT_FAILURE);
    arguments[2] = "2x";
    assert(sorting_demo_main(3, arguments) == EXIT_FAILURE);
    arguments[1] = "--unknown";
    assert(sorting_demo_main(3, arguments) == EXIT_FAILURE);
    puts("排序窗口测试通过：算法切换、单步、图例、计时、截图和退出。");
    return 0;
}
