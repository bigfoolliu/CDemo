/** @file
 * @brief 生命游戏窗口：事件 → 固定步长演化 → 绘制 → 显示/限速。
 * SDL 只负责输入、计时和呈现，生命规则位于纯 C 模块 life.c。
 */
#include "graphics_sdl.h"
#include "bmp_capture.h"
#include "life.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { LIFE_DRAW_HZ = 60 };
/** 世界与窗口控制分离：暂停不影响显示，鼠标编辑不替换最近选择的预设。 */
typedef struct {
    LifeWorld world;
    LifePattern preset;
    bool running, paused, custom;
    bool capture; /* S 请求保存本帧；重复按键忽略，同帧多次请求合并。 */
    double accumulator; /* 尚未用于演化的秒数，不积累暂停时间。 */
} LifeDemoState;

/**
 * @brief 重新加载预设，保留暂停状态。
 * @param state 有效的可读写演示状态。
 * @param preset 待加载的合法预设。
 */
static void load_preset(LifeDemoState *state, LifePattern preset)
{
    state->preset = preset;
    life_reset(&state->world, preset);
    state->custom = false;
    state->accumulator = 0;
}

/**
 * @brief 处理键盘、鼠标或退出事件；鼠标仅在暂停时可编辑。
 * @param state 有效的可读写状态。
 * @param event 有效的只读 SDL 事件，忽略键盘自动重复。
 * @param width 当前窗口宽度，单位与 SDL 鼠标事件坐标一致。
 * @param height 当前窗口高度；尺寸无效时忽略鼠标。
 * @return 要求丢弃本帧时间增量时为 true。
 */
static bool handle_event(LifeDemoState *state, const SDL_Event *event, int width, int height)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
    } else if (event->type == SDL_KEYDOWN && event->key.repeat == 0) {
        switch (event->key.keysym.sym) {
        case SDLK_s: state->capture = true; break;
        case SDLK_ESCAPE: state->running = false; break;
        case SDLK_SPACE:
            state->paused = !state->paused;
            state->accumulator = 0;
            return true;
        case SDLK_n:
            if (state->paused) { life_step(&state->world); }
            break;
        case SDLK_1: load_preset(state, LIFE_GLIDER); return true;
        case SDLK_2: load_preset(state, LIFE_BLINKER); return true;
        case SDLK_3: load_preset(state, LIFE_RANDOM); return true;
        case SDLK_r: load_preset(state, state->preset); return true;
        case SDLK_c:
            life_reset(&state->world, LIFE_EMPTY);
            state->paused = true;
            state->custom = true;
            state->accumulator = 0;
            return true;
        default: break;
        }
    } else if (event->type == SDL_MOUSEBUTTONDOWN && event->button.button == SDL_BUTTON_LEFT &&
               state->paused && width > 0 && height > 0 &&
               event->button.x >= 0 && event->button.x < width &&
               event->button.y >= 0 && event->button.y < height) {
        /* 整个窗口显示整张画布；按比例映射到格子，不硬编码两倍放大。
         * 先转 int64_t 再乘，避免较大窗口坐标的整数乘法溢出。
         */
        int x = (int)((int64_t)event->button.x * LIFE_WIDTH / width);
        int y = (int)((int64_t)event->button.y * LIFE_HEIGHT / height);
        if (life_toggle(&state->world, x, y)) {
            state->custom = true;
            state->accumulator = 0;
            return true;
        }
    }
    return false;
}

/**
 * @brief 将真实时间累积为固定 0.1 秒的演化步。
 * @param state 有效的可读写状态。
 * @param elapsed 秒数；负数和非有限数忽略，单帧最多接收 0.1 秒。
 */
static void advance_time(LifeDemoState *state, double elapsed)
{
    if (state->paused) { state->accumulator = 0; return; }
    if (!isfinite(elapsed) || elapsed < 0) { return; }
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / LIFE_HZ; /* 1.0 保证浮点除法，整数 1/10 会得到零。 */
    while (state->accumulator >= tick) {
        life_step(&state->world);
        state->accumulator -= tick; /* 不足一代的余量留给下一帧。 */
    }
}

/**
 * @brief 返回窗口标题使用的图案名称。
 * @param state 有效的只读状态。
 * @return 静态字符串，调用者不释放。
 */
static const char *pattern_name(const LifeDemoState *state)
{
    if (state->custom) { return "custom"; }
    switch (state->preset) {
    case LIFE_GLIDER: return "glider";
    case LIFE_BLINKER: return "blinker";
    case LIFE_RANDOM: return "random";
    default: return "empty";
    }
}

/**
 * @brief 创建窗口并演示生命游戏；统一清理部分创建或完整创建的资源。
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
            fputs("用法：life_demo [--frames 1..10000]\n", stderr);
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
    if (!gfx_sdl_open(&display, "Conway Life")) { goto cleanup; }
    static GfxCanvas canvas; /* 大画布放在静态存储区，避免占用约 300 KiB 栈。 */
    LifeDemoState state = {.running = true};
    load_preset(&state, LIFE_GLIDER);
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) { SDL_SetError("无法获取计时频率"); goto cleanup; }
    unsigned long frames = 0;
    puts("生命游戏：Space 暂停/继续，N 单步，1/2/3 图案，R 重置，C 清空，暂停时左键编辑，S 保存 BMP，Esc 退出。");
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
        life_draw(&state.world, &canvas);
        char title[256];
        /* PRIu64 匹配 uint64_t，%zu 匹配 size_t；snprintf 限制写入长度。 */
        snprintf(title, sizeof title,
                 "Life | %s | gen=%" PRIu64 " | alive=%zu | %s | Space/N | 1/2/3 | R/C | click | S: BMP | Esc",
                 pattern_name(&state), state.world.generation, life_count(&state.world),
                 state.paused ? "paused" : "running");
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            /* 此时画布已完整绘制，保存逻辑像素；失败报错后继续演示。 */
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "life", path)) {
                printf("已保存 BMP：%s\n", path);
            } else {
                fprintf(stderr, "BMP 截图失败：%s\n", strerror(errno));
            }
            state.capture = false;
        }
        if (!gfx_sdl_present(&display, &canvas)) { goto cleanup; }
        if (limit != 0 && ++frames >= limit) { break; } /* 显示次数与演化代数不同。 */
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        /* 目标显示 60 FPS，模拟只有 10 代/秒；延时精度和调度影响实际 FPS。
         * 超时帧不等待；单帧时间截断意味着长停顿的额外模拟时间被丢弃。
         */
        if (spent < 1.0 / LIFE_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / LIFE_DRAW_HZ - spent) * 1000));
        }
    }
    status = EXIT_SUCCESS;
cleanup:
    if (status != EXIT_SUCCESS) { fprintf(stderr, "SDL 失败：%s\n", SDL_GetError()); }
    gfx_sdl_close(&display);
    return status;
}
