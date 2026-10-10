/** @file @brief 迷宫窗口：DFS 生成动画、BFS/DFS 搜索动画与 BMP 截图。 */
#include "graphics_sdl.h"
#include "bmp_capture.h"
#include "maze.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAZE_DRAW_HZ = 60 };
/** 算法世界与窗口控制分离，暂停时仍处理事件和显示。 */
typedef struct {
    MazeWorld world;
    bool running, paused, capture;
    double accumulator;
} MazeDemoState;

/**
 * @brief 处理一个键盘或退出事件，忽略自动重复。
 * @param state 有效的可读写演示状态。
 * @param event 有效的只读 SDL 事件。
 * @return 暂停切换、重建或重寻路时为 true，要求丢弃本帧时间。
 */
static bool handle_event(MazeDemoState *state, const SDL_Event *event)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
    } else if (event->type == SDL_KEYDOWN && event->key.repeat == 0) {
        switch (event->key.keysym.sym) {
        case SDLK_ESCAPE: state->running = false; break;
        case SDLK_s: state->capture = true; break;
        case SDLK_SPACE:
            state->paused = !state->paused; state->accumulator = 0; return true;
        case SDLK_n: if (state->paused) { maze_step(&state->world); } break;
        case SDLK_r:
            maze_reset(&state->world, state->world.seed, state->world.algorithm);
            state->accumulator = 0; return true;
        case SDLK_g:
            /* 递增固定种子提供不同地图；同种子仍可通过 R 重现。 */
            maze_reset(&state->world, state->world.seed + UINT32_C(1), state->world.algorithm);
            state->accumulator = 0; return true;
        case SDLK_1:
        case SDLK_2:
            /* 保留地图，切换前沿结构；生成尚未完成时忽略，避免搜索半成品。 */
            if (maze_restart_search(&state->world, event->key.keysym.sym == SDLK_1 ? MAZE_BFS : MAZE_DFS)) {
                state->accumulator = 0; return true;
            }
            break;
        default: break;
        }
    }
    return false;
}

/**
 * @brief 使用固定 1/120 秒时间步推进算法，可在一帧推进多步。
 * @param state 有效的可读写状态。
 * @param elapsed 秒数，负数/非有限数忽略，最多计入 0.1 秒。
 */
static void advance_time(MazeDemoState *state, double elapsed)
{
    if (state->paused) { state->accumulator = 0; return; }
    if (!isfinite(elapsed) || elapsed < 0) { return; }
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / MAZE_HZ;
    while (state->accumulator >= tick) {
        maze_step(&state->world);
        state->accumulator -= tick;
    }
}

/**
 * @brief 提供窗口标题的阶段名称。
 * @param phase 世界阶段。
 * @return 静态字符串，不需释放。
 */
static const char *phase_name(MazePhase phase)
{
    switch (phase) {
    case MAZE_GENERATING: return "generating";
    case MAZE_SEARCHING: return "searching";
    case MAZE_FOUND: return "found";
    case MAZE_NO_PATH: return "no-path";
    default: return "unknown";
    }
}

/**
 * @brief 创建窗口并演示迷宫；统一清理部分创建或完整创建的资源。
 * @param argc 参数数量，包含程序名。
 * @param argv 参数字符串数组，支持 --frames 1..10000 自动退出。
 * @return 正常退出为 EXIT_SUCCESS，参数或 SDL 失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    unsigned long limit = 0; /* 零表示不限帧数。 */
    if (argc != 1) {
        char *end = nullptr;
        /* 短路求值确保 argc 合法后才读取后续参数。 */
        if (argc != 3 || strcmp(argv[1], "--frames") != 0 ||
            argv[2][0] < '0' || argv[2][0] > '9') {
            fputs("用法：maze_demo [--frames 1..10000]\n", stderr);
            return EXIT_FAILURE;
        }
        errno = 0;
        limit = strtoul(argv[2], &end, 10);
        if (errno != 0 || *end != '\0' || limit == 0 || limit > 10000) {
            fputs("帧数必须为 1..10000。\n", stderr);
            return EXIT_FAILURE;
        }
    }
    GfxWindow display = {0};
    int status = EXIT_FAILURE;
    if (!gfx_sdl_open(&display, "Maze | DFS generation")) { goto cleanup; }
    static GfxCanvas canvas; /* 大画布放在静态存储区，避免占用约 300 KiB 栈。 */
    MazeDemoState state = {.running = true};
    maze_reset(&state.world, 42, MAZE_BFS);
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) { SDL_SetError("无法获取计时频率"); goto cleanup; }
    unsigned long frames = 0;
    puts("迷宫：Space 暂停/继续，N 单步，R 同种子重建，G 新迷宫，1 BFS / 2 DFS 重寻路，S 保存 BMP，Esc 退出。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        /* 计数差除以每秒计数得到秒数；暂停也更新 previous，避免恢复时追赶。 */
        double elapsed = (double)(started - previous) / (double)frequency;
        previous = started;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (handle_event(&state, &event)) { elapsed = 0; }
        }
        if (!state.running) { break; }
        advance_time(&state, elapsed);
        maze_draw(&state.world, &canvas);
        char title[256];
        snprintf(title, sizeof title,
                 "Maze | seed=%" PRIu32 " | %s/%s | visited=%zu | path=%zu | %s | Space/N R/G 1/2 S Esc",
                 state.world.seed, phase_name(state.world.phase),
                 state.world.algorithm == MAZE_BFS ? "BFS" : "DFS",
                 state.world.visited_count, state.world.path_length,
                 state.paused ? "paused" : "running");
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            /* 此时画布已完整绘制，保存逻辑像素；失败报错后继续演示。 */
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "maze", path)) {
                printf("已保存 BMP：%s\n", path);
            } else {
                fprintf(stderr, "BMP 截图失败：%s\n", strerror(errno));
            }
            state.capture = false;
        }
        if (!gfx_sdl_present(&display, &canvas)) { goto cleanup; }
        if (limit != 0 && ++frames >= limit) { break; } /* 显示次数与算法步数不同。 */
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        /* 目标显示 60 FPS，算法推进为 120 步/秒；延时精度和调度影响实际 FPS。
         * 超时帧不等待；单帧时间截断意味着长停顿的额外模拟时间被丢弃。
         */
        if (spent < 1.0 / MAZE_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / MAZE_DRAW_HZ - spent) * 1000));
        }
    }
    status = EXIT_SUCCESS;
cleanup:
    if (status != EXIT_SUCCESS) { fprintf(stderr, "SDL 失败：%s\n", SDL_GetError()); }
    gfx_sdl_close(&display);
    return status;
}
