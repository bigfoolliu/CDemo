/** @file @brief 在柱状图上逐步比较四种基础排序算法。 */
#include "bmp_capture.h"
#include "graphics_sdl.h"
#include "sorting.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SORT_BAR_LEFT = 16,
    SORT_BAR_WIDTH = 5,
    SORT_BAR_GAP = 1,
    SORT_BAR_BASELINE = 190,
    SORT_BAR_MAX_HEIGHT = 160,
    SORT_STEP_HZ = 60,
    SORT_DRAW_HZ = 60
};

/** 窗口管理输入和时间；排序状态由独立核心维护。 */
typedef struct {
    SortingState sorting; /* 当前数组、算法进度和操作计数。 */
    uint32_t seed;        /* 当前可重复数据集的种子。 */
    bool running;         /* false 时退出主循环。 */
    bool paused;          /* 暂停自动排序，但保留手动单步。 */
    bool capture;         /* 下一帧画布完成后请求截图。 */
    double accumulator;   /* 尚未凑满一个排序步骤的秒数。 */
} SortingDemoState;

/**
 * @brief 生成下一个确定性伪随机整数。
 * @param state 可读写随机状态；零状态会替换为固定非零种子。
 * @return 由 xorshift32 产生的 32 位伪随机值。
 */
static uint32_t next_random(uint32_t *state)
{
    if (*state == 0) {
        *state = UINT32_C(0x9e3779b9);
    }
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

/**
 * @brief 根据种子生成含重复值的演示数组。
 * @param seed 非零数据集种子；相同种子产生相同数组。
 * @param items 容量为 SORTING_ITEM_COUNT 的输出数组。
 */
static void make_input(uint32_t seed, SortingItem items[SORTING_ITEM_COUNT])
{
    uint32_t random_state = seed;
    for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
        items[index].value = (uint8_t)(next_random(&random_state) % SORTING_VALUE_MAX + 1);
        items[index].original_order = (uint8_t)index;
    }
}

/**
 * @brief 使用当前数据种子重启指定算法。
 * @param state 有效的可读写演示状态。
 * @param algorithm 冒泡、插入、选择或快速排序。
 * @param seed 当前演示数据种子。
 * @return 核心初始化成功时为 true。
 */
static bool restart_sort(SortingDemoState *state, SortingAlgorithm algorithm, uint32_t seed)
{
    SortingItem items[SORTING_ITEM_COUNT];
    make_input(seed, items);
    if (!sorting_start(items, algorithm, &state->sorting)) {
        return false;
    }
    state->seed = seed;
    state->accumulator = 0;
    return true;
}

/**
 * @brief 返回排序算法的窗口显示名称。
 * @param algorithm 当前算法。
 * @return 静态字符串，不需要释放。
 */
static const char *algorithm_name(SortingAlgorithm algorithm)
{
    switch (algorithm) {
    case SORTING_BUBBLE:
        return "Bubble";
    case SORTING_INSERTION:
        return "Insertion";
    case SORTING_SELECTION:
        return "Selection";
    case SORTING_QUICK:
        return "Quick";
    default:
        return "unknown";
    }
}

/**
 * @brief 返回画布上显示的大写算法名称。
 * @param algorithm 当前算法。
 * @return 仅包含 glyph3x5 支持字符的静态字符串。
 */
static const char *algorithm_label(SortingAlgorithm algorithm)
{
    switch (algorithm) {
    case SORTING_BUBBLE:
        return "BUBBLE";
    case SORTING_INSERTION:
        return "INSERTION";
    case SORTING_SELECTION:
        return "SELECTION";
    case SORTING_QUICK:
        return "QUICK";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief 将 3×5 大写点阵写入输出缓冲区。
 * @param letter ASCII 大写英文字母。
 * @param rows 输出的五行位图，每行低三位从左到右代表像素。
 */
static void glyph3x5(char letter, uint8_t rows[5])
{
    static const uint8_t font[26][5] = {
        ['A' - 'A'] = {2, 5, 7, 5, 5},   ['B' - 'A'] = {6, 5, 6, 5, 6},
        ['C' - 'A'] = {3, 4, 4, 4, 3},   ['D' - 'A'] = {6, 5, 5, 5, 6},
        ['E' - 'A'] = {7, 4, 6, 4, 7},   ['H' - 'A'] = {5, 5, 7, 5, 5},
        ['I' - 'A'] = {7, 2, 2, 2, 7},   ['K' - 'A'] = {5, 5, 6, 5, 5},
        ['L' - 'A'] = {4, 4, 4, 4, 7},   ['N' - 'A'] = {5, 7, 7, 7, 5},
        ['O' - 'A'] = {2, 5, 5, 5, 2},   ['Q' - 'A'] = {2, 5, 5, 3, 1},
        ['R' - 'A'] = {6, 5, 6, 5, 5},   ['S' - 'A'] = {3, 4, 2, 1, 6},
        ['T' - 'A'] = {7, 2, 2, 2, 2},   ['U' - 'A'] = {5, 5, 5, 5, 7},
        ['W' - 'A'] = {5, 5, 7, 7, 5}
    };
    if (letter < 'A' || letter > 'Z') {
        memset(rows, 0, 5 * sizeof rows[0]);
        return;
    }
    memcpy(rows, font[letter - 'A'], 5 * sizeof rows[0]);
}

/**
 * @brief 在画布上绘制无字体依赖的 3×5 标签。
 * @param canvas 有效的可写画布。
 * @param x 标签左侧像素坐标。
 * @param y 标签顶部像素坐标。
 * @param label 只含 ASCII 大写字母的 NUL 结尾字符串。
 * @param color 标签像素颜色。
 */
static void draw_label(GfxCanvas *canvas, int x, int y, const char *label, uint32_t color)
{
    for (size_t letter = 0; label[letter] != '\0'; ++letter) {
        uint8_t rows[5];
        glyph3x5(label[letter], rows);
        for (size_t row = 0; row < 5; ++row) {
            for (size_t column = 0; column < 3; ++column) {
                uint8_t mask = (uint8_t)(1u << (2u - (unsigned int)column));
                if ((rows[row] & mask) != 0) {
                    gfx_pixel(canvas, x + (int)letter * 4 + (int)column,
                              y + (int)row, color);
                }
            }
        }
    }
}

/**
 * @brief 绘制数组柱形、操作状态和颜色图例。
 * @param state 已初始化的只读演示状态。
 * @param canvas 有效的可写画布。
 */
static void draw_scene(const SortingDemoState *state, GfxCanvas *canvas)
{
    gfx_clear(canvas, gfx_rgb(18, 22, 31));
    gfx_rect(canvas, 8, 5, 58, 13, true, gfx_rgb(30, 36, 48));
    draw_label(canvas, 12, 9, algorithm_label(state->sorting.algorithm),
               gfx_rgb(228, 232, 239));
    gfx_rect(canvas, 8, 204, GFX_WIDTH - 16, 22, true, gfx_rgb(30, 36, 48));
    gfx_line(canvas, 12, SORT_BAR_BASELINE, GFX_WIDTH - 12, SORT_BAR_BASELINE,
             gfx_rgb(110, 120, 135));
    for (size_t index = 0; index < SORTING_ITEM_COUNT; ++index) {
        uint8_t value = state->sorting.items[index].value;
        if (state->sorting.key_active && index == state->sorting.hole) {
            value = state->sorting.key.value;
        }
        int height = (int)value * SORT_BAR_MAX_HEIGHT / SORTING_VALUE_MAX;
        if (height == 0) {
            height = 1;
        }
        uint32_t color = gfx_rgb(70, 145, 230);
        if (state->sorting.settled[index]) {
            color = gfx_rgb(55, 105, 80);
        }
        if (index == state->sorting.active_a || index == state->sorting.active_b) {
            color = gfx_rgb(235, 155, 55);
        }
        if (state->sorting.algorithm == SORTING_QUICK &&
            index == state->sorting.pivot_index) {
            color = gfx_rgb(195, 115, 225);
        }
        if (index == state->sorting.changed_a || index == state->sorting.changed_b) {
            color = gfx_rgb(65, 215, 135);
        }
        int x = SORT_BAR_LEFT + (int)index * (SORT_BAR_WIDTH + SORT_BAR_GAP);
        int y = SORT_BAR_BASELINE - height;
        gfx_rect(canvas, x, y, SORT_BAR_WIDTH, height, true, color);
    }

    static const char *labels[] = {"DATA", "CHECK", "PIVOT", "WRITE", "DONE"};
    const uint32_t colors[] = {
        gfx_rgb(70, 145, 230), gfx_rgb(235, 155, 55), gfx_rgb(195, 115, 225),
        gfx_rgb(65, 215, 135), gfx_rgb(55, 105, 80)
    };
    for (size_t item = 0; item < sizeof labels / sizeof labels[0]; ++item) {
        int x = 12 + (int)item * 59;
        gfx_rect(canvas, x, 211, 7, 7, true, colors[item]);
        draw_label(canvas, x + 10, 212, labels[item], gfx_rgb(228, 232, 239));
    }
}

/**
 * @brief 处理算法切换、暂停、单步、重置、截图与退出事件。
 * @param state 有效的可读写演示状态。
 * @param event 有效的只读 SDL 事件，忽略键盘自动重复。
 * @return 状态重置时为 true，调用者应丢弃本帧时间增量。
 */
static bool handle_event(SortingDemoState *state, const SDL_Event *event)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
        return false;
    }
    if (event->type != SDL_KEYDOWN || event->key.repeat != 0) {
        return false;
    }
    switch (event->key.keysym.sym) {
    case SDLK_ESCAPE:
        state->running = false;
        break;
    case SDLK_SPACE:
        state->paused = !state->paused;
        state->accumulator = 0;
        return true;
    case SDLK_n:
        if (state->paused) {
            sorting_step(&state->sorting);
        }
        break;
    case SDLK_1:
    case SDLK_KP_1:
    case SDLK_2:
    case SDLK_KP_2:
    case SDLK_3:
    case SDLK_KP_3:
    case SDLK_4:
    case SDLK_KP_4: {
        SortingAlgorithm algorithm = SORTING_BUBBLE;
        if (event->key.keysym.sym == SDLK_2 || event->key.keysym.sym == SDLK_KP_2) {
            algorithm = SORTING_INSERTION;
        } else if (event->key.keysym.sym == SDLK_3 ||
                   event->key.keysym.sym == SDLK_KP_3) {
            algorithm = SORTING_SELECTION;
        } else if (event->key.keysym.sym == SDLK_4 ||
                   event->key.keysym.sym == SDLK_KP_4) {
            algorithm = SORTING_QUICK;
        }
        restart_sort(state, algorithm, state->seed);
        return true;
    }
    case SDLK_r: {
        uint32_t seed = state->seed + 1;
        if (seed == 0) {
            seed = 1;
        }
        restart_sort(state, state->sorting.algorithm, seed);
        return true;
    }
    case SDLK_s:
        state->capture = true;
        break;
    default:
        break;
    }
    return false;
}

/**
 * @brief 按固定步频推进排序，绘制帧率独立于算法速度。
 * @param state 有效的可读写状态。
 * @param elapsed 帧间秒数；负数与非有限数忽略，单帧最多累计 0.1 秒。
 */
static void advance_time(SortingDemoState *state, double elapsed)
{
    if (state->paused || state->sorting.phase != SORTING_RUNNING) {
        state->accumulator = 0;
        return;
    }
    if (!isfinite(elapsed) || elapsed < 0) {
        return;
    }
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / SORT_STEP_HZ;
    while (state->accumulator >= tick && state->sorting.phase == SORTING_RUNNING) {
        sorting_step(&state->sorting);
        state->accumulator -= tick;
    }
}

/**
 * @brief 创建窗口并运行动画；统一清理部分创建或完整创建的 SDL 资源。
 * @param argc 参数数量，包含程序名。
 * @param argv 支持无参数或 --frames 1..10000。
 * @return 正常退出为 EXIT_SUCCESS，参数错误或 SDL 失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    unsigned long frame_limit = 0;
    if (argc != 1) {
        char *end = nullptr;
        if (argc != 3 || strcmp(argv[1], "--frames") != 0 || argv[2][0] < '0' ||
            argv[2][0] > '9') {
            fputs("用法：sorting_demo [--frames 1..10000]\n", stderr);
            return EXIT_FAILURE;
        }
        errno = 0;
        frame_limit = strtoul(argv[2], &end, 10);
        if (errno != 0 || *end != '\0' || frame_limit == 0 || frame_limit > 10000) {
            fputs("帧数必须为 1..10000。\n", stderr);
            return EXIT_FAILURE;
        }
    }
    GfxWindow display = {0};
    int status = EXIT_FAILURE;
    if (!gfx_sdl_open(&display, "Sorting | Bubble / Insertion / Selection / Quick")) {
        goto cleanup;
    }
    static GfxCanvas canvas;
    SortingDemoState state = {.running = true, .seed = 1};
    if (!restart_sort(&state, SORTING_BUBBLE, state.seed)) {
        goto cleanup;
    }
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        SDL_SetError("无法获取计时频率");
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("排序：主键盘或小键盘 1 冒泡、2 插入、3 选择、4 快速；Space 暂停/继续，N 单步，"
         "R 换数据，S 截图，Esc 退出。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        double elapsed = (double)(started - previous) / (double)frequency;
        previous = started;
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (handle_event(&state, &event)) {
                elapsed = 0;
            }
        }
        if (!state.running) {
            break;
        }
        advance_time(&state, elapsed);
        draw_scene(&state, &canvas);
        char title[256];
        snprintf(title, sizeof title,
                 "Sorting | %s | %s | compare=%zu writes=%zu | Space/N 1/2/3/4 R S Esc",
                 algorithm_name(state.sorting.algorithm),
                 state.sorting.phase == SORTING_COMPLETE ? "complete" : "running",
                 state.sorting.comparisons, state.sorting.writes);
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "sorting", path)) {
                printf("已保存 BMP：%s\n", path);
            } else {
                fprintf(stderr, "BMP 截图失败：%s\n", strerror(errno));
            }
            state.capture = false;
        }
        if (!gfx_sdl_present(&display, &canvas)) {
            goto cleanup;
        }
        if (frame_limit != 0 && ++frames >= frame_limit) {
            break;
        }
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        if (spent < 1.0 / SORT_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / SORT_DRAW_HZ - spent) * 1000));
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
