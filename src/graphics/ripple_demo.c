/** @file @brief 波纹窗口：点击扰动、固定步长模拟、暂停与 BMP 导出。 */
#include "graphics_sdl.h"
#include "bmp_capture.h"
#include "ripple.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    RIPPLE_DRAW_HZ = 60
};
/** 暂停只停止时间演进，仍可添加扰动、显示与截图。 */
typedef struct {
    RippleWorld world;
    bool running, paused, capture;
    double accumulator;
} RippleDemoState;

/** @brief 重建水面并清除时间余量，保留暂停状态。
 * @param state 有效可写窗口状态。
 * @param initial true 为中央初始波纹，false 为平静水面。
 */
static void reset_scene(RippleDemoState *state, bool initial)
{
    ripple_reset(&state->world);
    if (initial) {
        ripple_disturb(&state->world, RIPPLE_WIDTH / 2, RIPPLE_HEIGHT / 2, 6, 8);
    }
    state->accumulator = 0;
}

/** @brief 处理事件，忽略自动重复键；支持暂停时添加扰动。
 * @param state 有效可写窗口状态。
 * @param event 有效只读 SDL 事件。
 * @param width 当前窗口宽度，单位与鼠标事件一致。
 * @param height 当前窗口高度。
 * @return 暂停切换、重置或清空时 true，要求丢弃本帧已耗时间。
 */
static bool handle_event(RippleDemoState *state, const SDL_Event *event, int width, int height)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
    } else if (event->type == SDL_KEYDOWN && event->key.repeat == 0) {
        switch (event->key.keysym.sym) {
        case SDLK_ESCAPE:
            state->running = false;
            break;
        case SDLK_s:
            state->capture = true;
            break;
        case SDLK_SPACE:
            state->paused = !state->paused;
            state->accumulator = 0;
            return true;
        case SDLK_n:
            if (state->paused) {
                ripple_step(&state->world);
            }
            break;
        case SDLK_r:
            reset_scene(state, true);
            return true;
        case SDLK_c:
            reset_scene(state, false);
            return true;
        default:
            break;
        }
    } else if (event->type == SDL_MOUSEBUTTONDOWN &&
               (event->button.button == SDL_BUTTON_LEFT ||
                event->button.button == SDL_BUTTON_RIGHT) &&
               width > 0 && height > 0 && event->button.x >= 0 && event->button.x < width &&
               event->button.y >= 0 && event->button.y < height) {
        /* 先扩展到 int64_t 再乘，避免极大窗口尺寸下整数溢出。
         * 最外圈不接受扰动；靠近边界的圆形扰动只修改内圈。
         */
        int x = (int)((int64_t)event->button.x * RIPPLE_WIDTH / width);
        int y = (int)((int64_t)event->button.y * RIPPLE_HEIGHT / height);
        ripple_disturb(&state->world, x, y, 6, event->button.button == SDL_BUTTON_LEFT ? 8 : -8);
    }
    return false;
}

/** @brief 将经过的秒数累积为固定步；暂停时不积累时间。
 * @param state 有效可写窗口状态。
 * @param elapsed 秒数；负数/非有限数忽略，每帧最多接收 0.1 秒，最多追赶六步。
 */
static void advance_time(RippleDemoState *state, double elapsed)
{
    if (state->paused) {
        state->accumulator = 0;
        return;
    }
    if (!isfinite(elapsed) || elapsed < 0) {
        return;
    }
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / RIPPLE_HZ;
    int count = 0;
    while (state->accumulator >= tick && count < 6) {
        ripple_step(&state->world);
        state->accumulator -= tick;
        ++count;
    }
}

/**
 * @brief 创建窗口并演示波纹；统一清理部分创建或完整创建的资源。
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
        if (argc != 3 || strcmp(argv[1], "--frames") != 0 || argv[2][0] < '0' || argv[2][0] > '9') {
            fputs("用法：ripple_demo [--frames 1..10000]\n", stderr);
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
    if (!gfx_sdl_open(&display, "Ripple")) {
        goto cleanup;
    }
    static GfxCanvas canvas;      /* 大画布放在静态存储区，避免占用约 300 KiB 栈。 */
    static RippleDemoState state; /* 两层高度场约 150 KiB，不放在栈上。 */
    state = (RippleDemoState){.running = true};
    reset_scene(&state, true);
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        SDL_SetError("无法获取计时频率");
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("波纹模拟：左键凸起/右键凹陷，Space 暂停/继续，N 单步，R 初始波纹，C 清空，S 保存 "
         "BMP，Esc 退出。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        /* 计数差除以每秒计数得到秒数；暂停也更新 previous，避免恢复时追赶。 */
        double elapsed = (double)(started - previous) / (double)frequency;
        previous = started;
        int width, height;
        SDL_GetWindowSize(display.window, &width, &height);
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (handle_event(&state, &event, width, height)) {
                elapsed = 0;
            }
        }
        if (!state.running) {
            break;
        }
        advance_time(&state, elapsed);
        if (!ripple_draw(&state.world, &canvas)) {
            SDL_SetError("非法波纹状态");
            goto cleanup;
        }
        char title[256];
        snprintf(title, sizeof title,
                 "Ripple | steps=%" PRIu64
                 " | %s | left/right click | Space/N | R/C | S: BMP | Esc",
                 state.world.steps, state.paused ? "paused" : "running");
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            /* 此时画布已完整绘制，保存逻辑像素；失败报错后继续演示。 */
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "ripple", path)) {
                printf("已保存 BMP：%s\n", path);
            } else {
                fprintf(stderr, "BMP 截图失败：%s\n", strerror(errno));
            }
            state.capture = false;
        }
        if (!gfx_sdl_present(&display, &canvas)) {
            goto cleanup;
        }
        if (limit != 0 && ++frames >= limit) {
            break;
        } /* 显示次数与模拟步数分别统计。 */
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        /* 目标显示 60 FPS，模拟采用每秒 60 步；延时精度和调度影响实际 FPS。
         * 超时帧不等待；单帧时间截断意味着长停顿的额外模拟时间被丢弃。
         */
        if (spent < 1.0 / RIPPLE_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / RIPPLE_DRAW_HZ - spent) * 1000));
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
