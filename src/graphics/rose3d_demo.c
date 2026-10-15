/**
 * @file
 * @brief 三维粒子玫瑰窗口：固定步长动画、相机控制与 BMP 截图。
 */
#include "bmp_capture.h"
#include "graphics_sdl.h"
#include "rose3d.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    Rose3dSystem rose;
    bool running;
    bool paused;
    bool capture;
    double accumulator;
} Rose3dDemoState;

/**
 * @brief 处理关闭、相机旋转、暂停、单步、重置和截图按键。
 * @param state 有效的可读写演示状态。
 * @param event 有效的只读 SDL 事件。
 * @return 暂停或重置时为 true，调用方应丢弃本帧时间余量。
 */
static bool handle_event(Rose3dDemoState *state, const SDL_Event *event)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
    } else if (event->type == SDL_KEYDOWN && event->key.repeat == 0) {
        switch (event->key.keysym.sym) {
        case SDLK_ESCAPE:
            state->running = false;
            break;
        case SDLK_SPACE:
            state->paused = !state->paused;
            state->accumulator = 0.0;
            return true;
        case SDLK_n:
            if (state->paused) {
                rose3d_step(&state->rose);
            }
            break;
        case SDLK_r:
            rose3d_reset(&state->rose, UINT32_C(20261015));
            state->accumulator = 0.0;
            return true;
        case SDLK_s:
            state->capture = true;
            break;
        case SDLK_LEFT:
            rose3d_rotate(&state->rose, -0.12, 0.0);
            break;
        case SDLK_RIGHT:
            rose3d_rotate(&state->rose, 0.12, 0.0);
            break;
        case SDLK_UP:
            rose3d_rotate(&state->rose, 0.0, 0.10);
            break;
        case SDLK_DOWN:
            rose3d_rotate(&state->rose, 0.0, -0.10);
            break;
        default:
            break;
        }
    }
    return false;
}

/**
 * @brief 将经过限制的墙钟增量累积为固定 1/60 秒模拟步。
 * @param state 有效的可读写演示状态。
 * @param elapsed 秒数；非有限或负值忽略，单帧最多计入 0.1 秒。
 */
static void advance_time(Rose3dDemoState *state, double elapsed)
{
    if (state->paused) {
        state->accumulator = 0.0;
        return;
    }
    if (!isfinite(elapsed) || elapsed < 0.0) {
        return;
    }
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / ROSE3D_HZ;
    while (state->accumulator >= tick) {
        rose3d_step(&state->rose);
        state->accumulator -= tick;
    }
}

/**
 * @brief 解析 --frames 参数；无参数时持续交互，有限模式用于短时启动。
 * @param argc 命令行参数个数。
 * @param argv 命令行参数数组，借用且不修改。
 * @param output 有效可写帧数；零表示无限运行。
 * @return 参数有效为 true，否则不修改输出。
 */
static bool parse_frame_limit(int argc, char *argv[], unsigned long *output)
{
    if (output == nullptr) {
        return false;
    }
    if (argc == 1) {
        *output = 0;
        return true;
    }
    if (argc != 3 || strcmp(argv[1], "--frames") != 0 ||
        argv[2][0] < '0' || argv[2][0] > '9') {
        return false;
    }
    char *end = nullptr;
    errno = 0;
    unsigned long frames = strtoul(argv[2], &end, 10);
    if (errno != 0 || *end != '\0' || frames == 0 || frames > 10000) {
        return false;
    }
    *output = frames;
    return true;
}

/**
 * @brief 创建窗口并运行三维玫瑰动画，所有 SDL 资源由统一出口释放。
 * @param argc 命令行参数数量，支持可选 --frames 1..10000。
 * @param argv 命令行参数数组。
 * @return 正常退出为 EXIT_SUCCESS，参数或 SDL 失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    unsigned long frame_limit = 0;
    if (!parse_frame_limit(argc, argv, &frame_limit)) {
        fputs("用法：rose3d_demo [--frames 1..10000]\n", stderr);
        return EXIT_FAILURE;
    }
    GfxWindow display = {0};
    int status = EXIT_FAILURE;
    if (!gfx_sdl_open(&display, "C 3D Particle Rose")) {
        goto cleanup;
    }
    SDL_SetWindowSize(display.window, ROSE3D_VIEW_WIDTH * 2, ROSE3D_VIEW_HEIGHT * 2);
    static GfxCanvas canvas;
    Rose3dDemoState state = {.running = true};
    rose3d_reset(&state.rose, UINT32_C(20261015));
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        SDL_SetError("无法获取计时频率");
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("三维粒子玫瑰：左右/上下方向键旋转，Space 暂停，暂停时 N 单步，"
         "R 重置，S 截图，Esc 退出。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        double elapsed = (double)(started - previous) / (double)frequency;
        previous = started;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (handle_event(&state, &event)) {
                elapsed = 0.0;
            }
        }
        if (!state.running) {
            break;
        }
        advance_time(&state, elapsed);
        rose3d_draw(&state.rose, &canvas);
        SDL_SetWindowTitle(display.window,
                           "3D Particle Rose | arrows: rotate | Space: pause | "
                           "N: step | S: BMP | Esc: quit");
        if (state.capture) {
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture_region(&canvas, ROSE3D_VIEW_X, ROSE3D_VIEW_Y,
                                       ROSE3D_VIEW_WIDTH, ROSE3D_VIEW_HEIGHT,
                                       "particle-rose-3d", path)) {
                printf("已保存 BMP：%s\n", path);
            } else {
                fprintf(stderr, "BMP 截图失败：%s\n", strerror(errno));
            }
            state.capture = false;
        }
        const SDL_Rect viewport = {ROSE3D_VIEW_X, ROSE3D_VIEW_Y,
                                   ROSE3D_VIEW_WIDTH, ROSE3D_VIEW_HEIGHT};
        if (!gfx_sdl_present_region(&display, &canvas, &viewport)) {
            goto cleanup;
        }
        if (frame_limit != 0 && ++frames >= frame_limit) {
            break;
        }
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        if (spent < 1.0 / ROSE3D_HZ) {
            SDL_Delay((Uint32)((1.0 / ROSE3D_HZ - spent) * 1000.0));
        }
    }
    status = EXIT_SUCCESS;
cleanup:
    if (status != EXIT_SUCCESS) {
        fprintf(stderr, "SDL 失败：%s\n", SDL_GetError());
    }
    gfx_sdl_close(&display);
    return status;
}
