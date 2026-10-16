/** @file @brief SDL dummy 下检查寻路窗口控制、鼠标编辑与截图。 */
#define main pathfinding_demo_main
#include "../src/graphics/pathfinding_demo.c"
#undef main
#include <assert.h>
#include "bmp_window_helpers.h"

/**
 * @brief 合成一个按键事件。
 * @param state 可读写窗口状态。
 * @param code 要发送的 SDL 按键代码。
 * @return 事件要求丢弃本帧时间时为 true。
 */
static bool key(PathfindingDemoState *state, SDL_Keycode code)
{
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = code;
    return handle_event(state, &event);
}

/**
 * @brief 合成点击指定网格格子的鼠标事件。
 * @param state 可读写窗口状态。
 * @param cell 要点击的网格编号。
 * @param button SDL 左、中或右鼠标按钮。
 * @return 网格处理了该点击时为 true。
 */
static bool click_cell(PathfindingDemoState *state, int cell, Uint8 button)
{
    SDL_Event event = {.type = SDL_MOUSEBUTTONDOWN};
    int column = cell % PATHFINDING_WIDTH;
    int row = cell / PATHFINDING_WIDTH;
    event.button.button = button;
    event.button.x = (PATH_GRID_LEFT + column * PATH_CELL_SIZE + 5) * PATH_WINDOW_SCALE;
    event.button.y = (PATH_GRID_TOP + row * PATH_CELL_SIZE + 5) * PATH_WINDOW_SCALE;
    return handle_event(state, &event);
}

/** @brief 检查画布底部图例中的颜色色块和文字标签。 */
static void test_legend(void)
{
    static GfxCanvas canvas;
    PathfindingDemoState state = {.running = true, .algorithm = PATHFINDING_BFS};
    reset_grid(&state);
    draw_scene(&state, &canvas);

    static const int positions[][2] = {
        {14, 193}, {112, 193}, {210, 193}, {14, 206}, {112, 206},
        {210, 206}, {14, 219}, {112, 219}, {210, 219}
    };
    const uint32_t colors[] = {
        gfx_rgb(72, 215, 118), gfx_rgb(238, 82, 91), gfx_rgb(215, 220, 225),
        gfx_rgb(45, 51, 62), gfx_rgb(220, 132, 55), gfx_rgb(70, 145, 230),
        gfx_rgb(65, 82, 112), gfx_rgb(220, 105, 225), gfx_rgb(250, 206, 66)
    };
    for (size_t item = 0; item < sizeof colors / sizeof colors[0]; ++item) {
        assert(canvas.pixels[positions[item][1]][positions[item][0]] == colors[item]);
    }
    assert(canvas.pixels[194][27] == gfx_rgb(228, 232, 239));
}

/** @brief 检查算法切换、暂停单步、鼠标编辑和事件边界。 */
static void test_controls(void)
{
    PathfindingDemoState state = {.running = true, .algorithm = PATHFINDING_BFS};
    reset_grid(&state);
    assert(state.search.phase == PATHFINDING_RUNNING);
    assert(key(&state, SDLK_2) && state.algorithm == PATHFINDING_DIJKSTRA);
    assert(key(&state, SDLK_3) && state.algorithm == PATHFINDING_A_STAR);
    assert(key(&state, SDLK_1) && state.algorithm == PATHFINDING_BFS);
    assert(key(&state, SDLK_SPACE) && state.paused && state.accumulator == 0);
    advance_time(&state, 1.0);
    assert(state.search.expanded_count == 0);
    key(&state, SDLK_n);
    assert(state.search.expanded_count == 1);
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_n;
    event.key.repeat = 1;
    handle_event(&state, &event);
    assert(state.search.expanded_count == 1);
    assert(click_cell(&state, 2 * PATHFINDING_WIDTH + 2, SDL_BUTTON_LEFT));
    assert(state.grid.cost[2 * PATHFINDING_WIDTH + 2] == 0);
    assert(state.search.expanded_count == 0 && state.paused);
    assert(click_cell(&state, 2 * PATHFINDING_WIDTH + 2, SDL_BUTTON_RIGHT));
    assert(state.grid.cost[2 * PATHFINDING_WIDTH + 2] == PATH_EXPENSIVE_COST);
    assert(!click_cell(&state, PATH_START, SDL_BUTTON_LEFT));
    assert(state.grid.cost[PATH_START] == 1);
    assert(!click_cell(&state, PATH_GOAL, SDL_BUTTON_RIGHT));
    assert(!click_cell(&state, 2 * PATHFINDING_WIDTH + 2, SDL_BUTTON_MIDDLE));
    assert(key(&state, SDLK_r) && state.paused && state.search.expanded_count == 0);
    assert(state.grid.cost[2 * PATHFINDING_WIDTH + 2] == 1);
    advance_time(&state, NAN);
    advance_time(&state, -1.0);
    assert(state.search.expanded_count == 0);
    key(&state, SDLK_SPACE);
    advance_time(&state, 0.1);
    assert(state.search.expanded_count == 2);
    key(&state, SDLK_ESCAPE);
    assert(!state.running);
    state.running = true;
    event.type = SDL_QUIT;
    handle_event(&state, &event);
    assert(!state.running);
}

/**
 * @brief 执行窗口输入、截图、有界主循环和参数验证。
 * @return 所有断言通过时返回零。
 */
int main(void)
{
    test_legend();
    test_controls();
    test_capture_loop(pathfinding_demo_main);
    char *arguments[] = {"pathfinding_demo", "--frames", "2", nullptr};
    assert(pathfinding_demo_main(3, arguments) == EXIT_SUCCESS);
    arguments[2] = "0";
    assert(pathfinding_demo_main(3, arguments) == EXIT_FAILURE);
    arguments[2] = "2x";
    assert(pathfinding_demo_main(3, arguments) == EXIT_FAILURE);
    arguments[1] = "--unknown";
    assert(pathfinding_demo_main(3, arguments) == EXIT_FAILURE);
    puts("寻路窗口测试通过：算法切换、网格编辑、时间步进、截图和退出。");
    return 0;
}
