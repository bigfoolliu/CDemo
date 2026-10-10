/** @file @brief SDL dummy 下验证控制、计时、截图与有界窗口主循环。 */
#define main maze_demo_main
#include "../src/graphics/maze_demo.c"
#undef main
#include <assert.h>
#include "bmp_window_helpers.h"

/** @brief 合成按键。 @param state 可读写状态。 @param code 键码。 @return 是否丢弃时间。 */
static bool key(MazeDemoState *state, SDL_Keycode code)
{
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = code;
    return handle_event(state, &event);
}

/** @brief 验证暂停、单步、同种子重建、新种子、算法切换和两种退出事件。 */
static void test_controls(void)
{
    MazeDemoState state = {.running = true};
    maze_reset(&state.world, 42, MAZE_BFS);
    advance_time(&state, 0.004); assert(state.world.depth == 1);
    advance_time(&state, 0.005); assert(state.world.depth == 2);
    assert(!key(&state, SDLK_2) && state.world.algorithm == MAZE_BFS);
    assert(key(&state, SDLK_SPACE) && state.paused && state.accumulator == 0);
    size_t depth = state.world.depth;
    advance_time(&state, 100); assert(state.world.depth == depth);
    key(&state, SDLK_n); assert(state.world.depth == depth + 1);
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_n; event.key.repeat = 1;
    handle_event(&state, &event); assert(state.world.depth == depth + 1);
    event.key.keysym.sym = SDLK_s;
    handle_event(&state, &event); assert(!state.capture);
    event.key.repeat = 0;
    handle_event(&state, &event); assert(state.capture);
    assert(key(&state, SDLK_r) && state.paused && state.world.seed == 42 && state.world.depth == 1);
    assert(key(&state, SDLK_g) && state.world.seed == 43 && state.paused);
    for (int i = 0; i < MAZE_CELLS && state.world.phase == MAZE_GENERATING; ++i) { maze_step(&state.world); }
    assert(state.world.phase == MAZE_SEARCHING);
    bool map[MAZE_CELLS]; memcpy(map, state.world.open, sizeof map);
    assert(key(&state, SDLK_2) && state.world.algorithm == MAZE_DFS && state.paused);
    for (int i = 0; i < MAZE_CELLS; ++i) { assert(map[i] == state.world.open[i]); }
    assert(key(&state, SDLK_1) && state.world.algorithm == MAZE_BFS);
    assert(key(&state, SDLK_SPACE) && !state.paused);
    key(&state, SDLK_n); assert(state.world.visited_count == 0);
    advance_time(&state, NAN); advance_time(&state, -1); assert(state.world.visited_count == 0);
    advance_time(&state, 100); assert(state.world.visited_count == 12);
    key(&state, SDLK_ESCAPE); assert(!state.running);
    state.running = true; event.type = SDL_QUIT;
    handle_event(&state, &event); assert(!state.running);
}

/** @brief 执行截图、控制和非法参数测试。 @return 成功返回零。 */
int main(void)
{
    test_controls();
    test_capture_loop(maze_demo_main);
    char *args[] = {"maze_demo", "--frames", "2", nullptr};
    assert(maze_demo_main(3, args) == EXIT_SUCCESS);
    args[2] = "0"; assert(maze_demo_main(3, args) == EXIT_FAILURE);
    args[2] = "2x"; assert(maze_demo_main(3, args) == EXIT_FAILURE);
    args[1] = "--unknown"; assert(maze_demo_main(3, args) == EXIT_FAILURE);
    puts("迷宫窗口控制测试通过。");
    return 0;
}
