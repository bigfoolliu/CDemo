/** @file @brief 落沙窗口：连续笔刷、材料选择、暂停与固定步长。 */
#include "graphics_sdl.h"
#include "bmp_capture.h"
#include "sand.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SAND_DRAW_HZ = 60 };
/** 绘制状态与物质网格分开；右键擦除优先于左键添加。 */
typedef struct {
    SandWorld world;
    SandMaterial material;
    int radius, mouse_x, mouse_y;
    bool running, paused, capture, painting, erasing;
    double accumulator;
} SandDemoState;

/** @brief 加载初始场景或空场景，保留暂停、材料和笔刷大小。
 * @param state 有效可写窗口状态。
 * @param initial true 添加示例物质和平台，false 仅保留外圈墙。
 */
static void reset_scene(SandDemoState *state, bool initial)
{
    sand_reset(&state->world);
    state->accumulator = 0;
    state->painting = state->erasing = false;
    if (!initial) { return; }
    for (int x = 30; x < 130; ++x) { sand_paint(&state->world, x, 80, 0, SAND_WALL); }
    for (int y = 20; y < 35; ++y) {
        for (int x = 50; x < 70; ++x) { sand_paint(&state->world, x, y, 0, SAND_GRAIN); }
        for (int x = 90; x < 115; ++x) { sand_paint(&state->world, x, y, 0, SAND_WATER); }
    }
}

/** @brief 在缓存的鼠标格子绘制，合法性由核心检查。
 * @param state 有效可写窗口状态。
 */
static void paint_brush(SandDemoState *state)
{
    if (state->painting || state->erasing) {
        sand_paint(&state->world, state->mouse_x, state->mouse_y, state->radius,
                   state->erasing ? SAND_AIR : state->material);
    }
}

/** @brief 映射鼠标并连接拖动路径，防止快速拖动遗漏中间格子。
 * @param state 有效可写窗口状态。
 * @param x 鼠标窗口横坐标。 @param y 鼠标窗口纵坐标。
 * @param width 当前窗口宽度。 @param height 当前窗口高度。
 * @param connect true 从上次位置画连续笔刷，false 只绘制本次位置。
 */
static void paint_mouse(SandDemoState *state, int x, int y, int width, int height, bool connect)
{
    if (width <= 0 || height <= 0 || x < 0 || x >= width || y < 0 || y >= height) {
        state->painting = state->erasing = false;
        return;
    }
    int target_x = (int)((int64_t)x * SAND_WIDTH / width);
    int target_y = (int)((int64_t)y * SAND_HEIGHT / height);
    int start_x = connect ? state->mouse_x : target_x;
    int start_y = connect ? state->mouse_y : target_y;
    int dx = abs(target_x - start_x), dy = -abs(target_y - start_y);
    int sx = start_x < target_x ? 1 : -1, sy = start_y < target_y ? 1 : -1;
    int error = dx + dy;
    /* Bresenham 同时用于笔刷中心路径；所有端点先映射到有限网格。 */
    while (true) {
        state->mouse_x = start_x;
        state->mouse_y = start_y;
        paint_brush(state);
        if (start_x == target_x && start_y == target_y) { break; }
        int twice = 2 * error;
        if (twice >= dy) { error += dy; start_x += sx; }
        if (twice <= dx) { error += dx; start_y += sy; }
    }
}

/** @brief 处理事件；重复键忽略，失焦/离开窗口取消连续绘制。
 * @param state 有效可写窗口状态。
 * @param event 有效只读 SDL 事件。
 * @param width 当前窗口宽度，和鼠标同单位。
 * @param height 当前窗口高度。
 * @return 暂停切换、重置、清空时 true，丢弃本帧已耗时间。
 */
static bool handle_event(SandDemoState *state, const SDL_Event *event, int width, int height)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
    } else if (event->type == SDL_KEYDOWN && event->key.repeat == 0) {
        switch (event->key.keysym.sym) {
        case SDLK_ESCAPE: state->running = false; break;
        case SDLK_s: state->capture = true; break;
        case SDLK_1: state->material = SAND_GRAIN; break;
        case SDLK_2: state->material = SAND_WATER; break;
        case SDLK_3: state->material = SAND_WALL; break;
        case SDLK_LEFTBRACKET: if (state->radius > 1) { --state->radius; } break;
        case SDLK_RIGHTBRACKET: if (state->radius < SAND_MAX_BRUSH) { ++state->radius; } break;
        case SDLK_SPACE:
            state->paused = !state->paused;
            state->accumulator = 0;
            return true;
        case SDLK_n:
            if (state->paused) { sand_step(&state->world); }
            break;
        case SDLK_r: reset_scene(state, true); return true;
        case SDLK_c: reset_scene(state, false); return true;
        default: break;
        }
    } else if (event->type == SDL_WINDOWEVENT &&
               (event->window.event == SDL_WINDOWEVENT_FOCUS_LOST || event->window.event == SDL_WINDOWEVENT_LEAVE)) {
        state->painting = state->erasing = false;
    } else if (event->type == SDL_MOUSEBUTTONDOWN) {
        if (event->button.button == SDL_BUTTON_LEFT) { state->painting = true; }
        else if (event->button.button == SDL_BUTTON_RIGHT) { state->erasing = true; }
        else { return false; }
        paint_mouse(state, event->button.x, event->button.y, width, height, false);
    } else if (event->type == SDL_MOUSEBUTTONUP) {
        if (event->button.button == SDL_BUTTON_LEFT) { state->painting = false; }
        if (event->button.button == SDL_BUTTON_RIGHT) { state->erasing = false; }
    } else if (event->type == SDL_MOUSEMOTION && (state->painting || state->erasing)) {
        state->painting = state->painting && (event->motion.state & SDL_BUTTON_LMASK) != 0;
        state->erasing = state->erasing && (event->motion.state & SDL_BUTTON_RMASK) != 0;
        if (state->painting || state->erasing) {
            paint_mouse(state, event->motion.x, event->motion.y, width, height, true);
        }
    }
    return false;
}

/** @brief 固定步长演化；持续按住左键时每个模拟步先补充笔刷物质。
 * @param state 有效可写窗口状态。
 * @param elapsed 秒数；非有限数/负数忽略，单帧最多 0.1 秒和六步。
 */
static void advance_time(SandDemoState *state, double elapsed)
{
    if (state->paused) { state->accumulator = 0; return; }
    if (!isfinite(elapsed) || elapsed < 0) { return; }
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / SAND_HZ;
    int count = 0;
    while (state->accumulator >= tick && count < 6) {
        paint_brush(state);
        sand_step(&state->world);
        state->accumulator -= tick;
        ++count;
    }
}

/**
 * @brief 创建窗口并演示落沙；统一清理部分创建或完整创建的资源。
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
            fputs("用法：sand_demo [--frames 1..10000]\n", stderr);
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
    if (!gfx_sdl_open(&display, "Sand")) { goto cleanup; }
    static GfxCanvas canvas; /* 大画布放在静态存储区，避免占用约 300 KiB 栈。 */
    static SandDemoState state; /* 网格及访问标记约 40 KiB，不放在栈上。 */
    state = (SandDemoState){.running = true, .material = SAND_GRAIN, .radius = 3};
    reset_scene(&state, true);
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) { SDL_SetError("无法获取计时频率"); goto cleanup; }
    unsigned long frames = 0;
    puts("落沙模拟：1 沙/2 水/3 墙，左键绘制，右键擦除，[ / ] 调整笔刷，Space 暂停，N 单步，R 重置，C 清空，S 保存 BMP，Esc 退出。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        /* 计数差除以每秒计数得到秒数；暂停也更新 previous，避免恢复时追赶。 */
        double elapsed = (double)(started - previous) / (double)frequency;
        previous = started;
        int width, height;
        SDL_GetWindowSize(display.window, &width, &height);
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (handle_event(&state, &event, width, height)) { elapsed = 0; }
        }
        if (!state.running) { break; }
        advance_time(&state, elapsed);
        if (!sand_draw(&state.world, &canvas)) { SDL_SetError("非法落沙状态"); goto cleanup; }
        char title[256];
        const char *name = state.material == SAND_GRAIN ? "sand" : state.material == SAND_WATER ? "water" : "wall";
        snprintf(title, sizeof title,
                 "Falling sand | steps=%" PRIu64 " | %s | %s radius=%d | 1/2/3 | L: paint R: erase | [ ] | Space/N R/C S Esc",
                 state.world.steps, state.paused ? "paused" : "running", name, state.radius);
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            /* 此时画布已完整绘制，保存逻辑像素；失败报错后继续演示。 */
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "sand", path)) {
                printf("已保存 BMP：%s\n", path);
            } else {
                fprintf(stderr, "BMP 截图失败：%s\n", strerror(errno));
            }
            state.capture = false;
        }
        if (!gfx_sdl_present(&display, &canvas)) { goto cleanup; }
        if (limit != 0 && ++frames >= limit) { break; } /* 显示次数与模拟步数分别统计。 */
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        /* 目标显示 60 FPS，模拟采用每秒 60 步；延时精度和调度影响实际 FPS。
         * 超时帧不等待；单帧时间截断意味着长停顿的额外模拟时间被丢弃。
         */
        if (spent < 1.0 / SAND_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / SAND_DRAW_HZ - spent) * 1000));
        }
    }
    status = EXIT_SUCCESS;
cleanup:
    if (status != EXIT_SUCCESS) { fprintf(stderr, "SDL 失败：%s\n", SDL_GetError()); }
    gfx_sdl_close(&display);
    return status;
}
