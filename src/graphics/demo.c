/** @file
 * @brief SDL2 只负责窗口、输入、计时和显示；图形由纯 C 核心绘制。
 */
#include "graphics_sdl.h"
#include <SDL.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const double PI = 3.14159265358979323846;

/** 主循环状态；暂停只停止角度更新，不停止事件处理或显示。 */
typedef struct { bool running; bool paused; double angle; } DemoState;

/**
 * @brief 处理一次窗口事件，便于在无窗口环境验证操作规则。
 * @param state 有效的可读写主循环状态。
 * @param event 有效的只读 SDL 事件；自动重复按键会被忽略。
 * @return 本次事件要求清零时间增量时为 true（暂停切换或重置）。
 */
static bool handle_event(DemoState *state, const SDL_Event *event)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
    } else if (event->type == SDL_KEYDOWN && event->key.repeat == 0) {
        switch (event->key.keysym.sym) {
        case SDLK_ESCAPE: state->running = false; break;
        case SDLK_SPACE: state->paused = !state->paused; return true;
        case SDLK_n: if (state->paused) { state->angle += PI / 240; } break;
        case SDLK_r: state->angle = 0; return true;
        default: break;
        }
    }
    return false;
}

/**
 * @brief 每帧绘制静态图形及变换后的矩形。
 * @param canvas 有效的可写画布，所有像素会先清空。
 * @param angle 旋转角度，单位为弧度；本演示保持在 [0, 2π)。
 */
static void draw_scene(GfxCanvas *canvas, double angle)
{
    uint32_t gray = gfx_rgb(45, 55, 70), white = gfx_rgb(230, 235, 245);
    gfx_clear(canvas, gfx_rgb(18, 23, 32));
    for (int x = 0; x < GFX_WIDTH; x += 20) {
        gfx_line(canvas, x, 0, x, GFX_HEIGHT - 1, gray);
    }
    for (int y = 0; y < GFX_HEIGHT; y += 20) {
        gfx_line(canvas, 0, y, GFX_WIDTH - 1, y, gray);
    }
    gfx_pixel(canvas, 10, 10, white);
    gfx_line(canvas, 15, 25, 115, 60, gfx_rgb(255, 100, 90));
    gfx_rect(canvas, 15, 80, 45, 30, false, gfx_rgb(100, 200, 255));
    gfx_rect(canvas, 75, 80, 40, 30, true, gfx_rgb(80, 170, 110));
    gfx_circle(canvas, 65, 170, 30, gfx_rgb(255, 210, 90));

    const GfxPoint local[] = {{-40, -25}, {40, -25}, {40, 25}, {-40, 25}};
    GfxPoint screen[4];
    for (size_t i = 0; i < 4; ++i) {
        if (!gfx_transform_point(local[i], (GfxPoint){225, 130}, angle, 1, &screen[i])) {
            return; /* 演示使用有限的小坐标；无效变换不提交给整数绘图接口。 */
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        size_t next = (i + 1) % 4;
        gfx_line(canvas, (int)lround(screen[i].x), (int)lround(screen[i].y),
                 (int)lround(screen[next].x), (int)lround(screen[next].y), white);
    }
    gfx_pixel(canvas, 225, 130, gfx_rgb(255, 100, 90));
}

/**
 * @brief 创建窗口并运行事件、更新、绘制和显示循环。
 * @param argc 命令行参数数量，包含程序名。
 * @param argv 参数字符串数组；可传 --frames N 限制演示帧数用于显示验证。
 * @return 正常退出返回 EXIT_SUCCESS；参数或 SDL 操作失败返回 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    unsigned long frame_limit = 0;
    if (argc != 1) {
        char *end = nullptr;
        if (argc != 3 || strcmp(argv[1], "--frames") != 0 || argv[2][0] < '0' || argv[2][0] > '9') {
            fputs("用法：graphics_demo [--frames 1..10000]\n", stderr);
            return EXIT_FAILURE;
        }
        frame_limit = strtoul(argv[2], &end, 10);
        if (*end != '\0' || frame_limit == 0 || frame_limit > 10000) {
            fputs("帧数必须为 1..10000。\n", stderr);
            return EXIT_FAILURE;
        }
    }
    GfxWindow display = {0};
    int status = EXIT_FAILURE;
    if (!gfx_sdl_open(&display, "C 2D | Space: pause | N: step | R: reset | Esc: quit")) {
        goto cleanup;
    }
    static GfxCanvas canvas; /* 避免把约 300 KiB 画布放进较小的调用栈。 */
    DemoState state = {.running = true};
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("二维绘图：Space 暂停/继续，暂停时 N 前进一帧，R 重置，Esc 退出。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        double dt = (double)(started - previous) / (double)frequency;
        previous = started;
        if (dt > 0.1) {
            dt = 0.1; /* 防止断点或系统挂起后出现大幅跳变。 */
        }
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (handle_event(&state, &event)) {
                dt = 0;
            }
        }
        if (!state.running) {
            break;
        }
        if (!state.paused) {
            state.angle += (PI / 4) * dt;
        }
        state.angle = fmod(state.angle, 2 * PI);
        draw_scene(&canvas, state.angle);
        if (!gfx_sdl_present(&display, &canvas)) {
            goto cleanup;
        }
        if (frame_limit != 0 && ++frames >= frame_limit) {
            break;
        }
        double elapsed = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        if (elapsed < 1.0 / 60) {
            SDL_Delay((Uint32)((1.0 / 60 - elapsed) * 1000));
        }
    }
    status = EXIT_SUCCESS;
cleanup:
    if (status != EXIT_SUCCESS) {
        fprintf(stderr, "SDL 图形演示失败：%s\n", SDL_GetError());
    }
    gfx_sdl_close(&display);
    return status;
}
