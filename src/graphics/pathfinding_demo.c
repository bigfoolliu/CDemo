/** @file @brief 在同一张网格上逐步比较 BFS、Dijkstra 和 A*。 */
#include "bmp_capture.h"
#include "graphics_sdl.h"
#include "pathfinding.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PATH_CELL_SIZE = 18,
    PATH_GRID_LEFT = 16,
    PATH_GRID_TOP = 6,
    PATH_WINDOW_SCALE = 2,
    PATH_START = 5 * PATHFINDING_WIDTH + 1,
    PATH_GOAL = 5 * PATHFINDING_WIDTH + 14,
    PATH_STEP_HZ = 24,
    PATH_DRAW_HZ = 60,
    PATH_EXPENSIVE_COST = 9
};

/** 窗口只保存交互、时间和绘制状态；图搜索数据由纯 C 核心处理。 */
typedef struct {
    PathfindingGrid grid;
    PathfindingSearch search;
    PathfindingAlgorithm algorithm;
    bool running;
    bool paused;
    bool capture;
    double accumulator;
} PathfindingDemoState;

/**
 * @brief 按选定算法重置搜索，不改变当前地图或暂停状态。
 * @param state 已初始化的可读写演示状态。
 * @return 搜索成功初始化时为 true；当前预设保证起点和终点可通行。
 */
static bool restart_search(PathfindingDemoState *state)
{
    return pathfinding_start(&state->grid, PATH_START, PATH_GOAL, state->algorithm,
                             &state->search);
}

/**
 * @brief 建立固定障碍与加权地形，便于观察步数最短路和代价最低路的差别。
 * @param state 有效的可读写演示状态；旧网格将被覆盖。
 */
static void reset_grid(PathfindingDemoState *state)
{
    for (size_t cell = 0; cell < PATHFINDING_CELLS; ++cell) {
        state->grid.cost[cell] = 1;
    }
    /* 两道墙限定过路缺口；中央高代价区让无权与加权最短路出现分歧。 */
    for (int y = 0; y < PATHFINDING_HEIGHT; ++y) {
        if (y != 5) {
            state->grid.cost[y * PATHFINDING_WIDTH + 5] = 0;
            state->grid.cost[y * PATHFINDING_WIDTH + 10] = 0;
        }
    }
    for (int y = 4; y <= 6; ++y) {
        for (int x = 6; x <= 9; ++x) {
            state->grid.cost[y * PATHFINDING_WIDTH + x] = PATH_EXPENSIVE_COST;
        }
    }
    state->grid.cost[PATH_START] = 1;
    state->grid.cost[PATH_GOAL] = 1;
    restart_search(state);
}

/**
 * @brief 把窗口坐标换算为网格编号；窗口固定按逻辑画布两倍显示。
 * @param window_x SDL 鼠标窗口横坐标。
 * @param window_y SDL 鼠标窗口纵坐标。
 * @return 命中网格时返回编号，否则返回 -1。
 */
static int cell_from_window(int window_x, int window_y)
{
    int x = window_x / PATH_WINDOW_SCALE;
    int y = window_y / PATH_WINDOW_SCALE;
    x -= PATH_GRID_LEFT;
    y -= PATH_GRID_TOP;
    if (x < 0 || y < 0) {
        return -1;
    }
    int column = x / PATH_CELL_SIZE;
    int row = y / PATH_CELL_SIZE;
    if (column >= PATHFINDING_WIDTH || row >= PATHFINDING_HEIGHT ||
        x % PATH_CELL_SIZE == PATH_CELL_SIZE - 1 ||
        y % PATH_CELL_SIZE == PATH_CELL_SIZE - 1) {
        return -1;
    }
    return row * PATHFINDING_WIDTH + column;
}

/**
 * @brief 按鼠标按钮切换墙格或普通/高代价格。
 * @param state 有效的可读写演示状态。
 * @param event 有效的鼠标按下事件。
 * @return 地图被修改并重启搜索时为 true。
 */
static bool edit_cell(PathfindingDemoState *state, const SDL_MouseButtonEvent *event)
{
    int cell = cell_from_window(event->x, event->y);
    if (cell < 0 || cell == PATH_START || cell == PATH_GOAL) {
        return false;
    }
    if (event->button == SDL_BUTTON_LEFT) {
        state->grid.cost[cell] = state->grid.cost[cell] == 0 ? 1 : 0;
    } else if (event->button == SDL_BUTTON_RIGHT) {
        if (state->grid.cost[cell] == 0 || state->grid.cost[cell] == 1) {
            state->grid.cost[cell] = PATH_EXPENSIVE_COST;
        } else {
            state->grid.cost[cell] = 1;
        }
    } else {
        return false;
    }
    restart_search(state);
    state->accumulator = 0;
    return true;
}

/**
 * @brief 处理键盘和鼠标事件；自动重复按键不重复触发。
 * @param state 有效的可读写演示状态。
 * @param event 有效的只读 SDL 事件。
 * @return 状态重置时为 true，调用者应丢弃本帧累计时间。
 */
static bool handle_event(PathfindingDemoState *state, const SDL_Event *event)
{
    if (event->type == SDL_QUIT) {
        state->running = false;
        return false;
    }
    if (event->type == SDL_MOUSEBUTTONDOWN) {
        return edit_cell(state, &event->button);
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
            pathfinding_step(&state->grid, &state->search);
        }
        break;
    case SDLK_1:
    case SDLK_2:
    case SDLK_3:
        state->algorithm = event->key.keysym.sym == SDLK_1 ? PATHFINDING_BFS :
                           event->key.keysym.sym == SDLK_2 ? PATHFINDING_DIJKSTRA : PATHFINDING_A_STAR;
        restart_search(state);
        state->accumulator = 0;
        return true;
    case SDLK_r:
        reset_grid(state);
        state->accumulator = 0;
        return true;
    case SDLK_s:
        state->capture = true;
        break;
    default:
        break;
    }
    return false;
}

/**
 * @brief 按固定 24 步/秒推进搜索，窗口仍独立按 60 FPS 绘制。
 * @param state 有效的可读写演示状态。
 * @param elapsed 单位为秒；负数和非有限数忽略，单次最多累计 0.1 秒。
 */
static void advance_time(PathfindingDemoState *state, double elapsed)
{
    if (state->paused || state->search.phase != PATHFINDING_RUNNING) {
        state->accumulator = 0;
        return;
    }
    if (!isfinite(elapsed) || elapsed < 0) {
        return;
    }
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / PATH_STEP_HZ;
    while (state->accumulator >= tick && state->search.phase == PATHFINDING_RUNNING) {
        pathfinding_step(&state->grid, &state->search);
        state->accumulator -= tick;
    }
}

/**
 * @brief 返回算法显示名称。
 * @param algorithm 已选择的搜索算法。
 * @return 静态字符串，不需要释放。
 */
static const char *algorithm_name(PathfindingAlgorithm algorithm)
{
    switch (algorithm) {
    case PATHFINDING_BFS:
        return "BFS";
    case PATHFINDING_DIJKSTRA:
        return "Dijkstra";
    case PATHFINDING_A_STAR:
        return "A*";
    default:
        return "unknown";
    }
}

/**
 * @brief 返回窗口标题用的搜索阶段名称。
 * @param phase 当前搜索阶段。
 * @return 静态字符串，不需要释放。
 */
static const char *phase_name(PathfindingPhase phase)
{
    switch (phase) {
    case PATHFINDING_RUNNING:
        return "searching";
    case PATHFINDING_FOUND:
        return "found";
    case PATHFINDING_NO_PATH:
        return "no path";
    default:
        return "unknown";
    }
}

/**
 * @brief 取得小型 3×5 大写字母点阵，用于不依赖字体库绘制图例。
 * @param letter ASCII 大写英文字母。
 * @param rows 输出的五行位图，每行低三位表示从左到右的像素。
 */
static void glyph3x5(char letter, uint8_t rows[5])
{
    static const uint8_t font[26][5] = {
        ['A' - 'A'] = {2, 5, 7, 5, 5},   ['C' - 'A'] = {3, 4, 4, 4, 3},
        ['D' - 'A'] = {6, 5, 5, 5, 6},   ['E' - 'A'] = {7, 4, 6, 4, 7},
        ['F' - 'A'] = {7, 4, 6, 4, 4},   ['G' - 'A'] = {3, 4, 5, 5, 3},
        ['H' - 'A'] = {5, 5, 7, 5, 5},   ['I' - 'A'] = {7, 2, 2, 2, 7},
        ['L' - 'A'] = {4, 4, 4, 4, 7},   ['M' - 'A'] = {5, 7, 7, 5, 5},
        ['N' - 'A'] = {5, 7, 7, 7, 5},   ['O' - 'A'] = {2, 5, 5, 5, 2},
        ['P' - 'A'] = {6, 5, 6, 4, 4},   ['R' - 'A'] = {6, 5, 6, 5, 5},
        ['S' - 'A'] = {3, 4, 2, 1, 6},   ['T' - 'A'] = {7, 2, 2, 2, 2},
        ['U' - 'A'] = {5, 5, 5, 5, 7},   ['W' - 'A'] = {5, 5, 7, 7, 5},
        ['X' - 'A'] = {5, 5, 2, 5, 5}
    };
    if (letter < 'A' || letter > 'Z') {
        memset(rows, 0, 5 * sizeof rows[0]);
        return;
    }
    memcpy(rows, font[letter - 'A'], 5 * sizeof rows[0]);
}

/**
 * @brief 在画布上绘制无字体依赖的 3×5 大写图例标签。
 * @param canvas 有效的可写画布。
 * @param x 标签左侧像素坐标。
 * @param y 标签顶部像素坐标。
 * @param label 只含 ASCII 大写字母的 NUL 结尾字符串。
 * @param color 字符像素颜色。
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
 * @brief 用不同颜色画墙、地形代价、前沿、已展开格与最终路径。
 * @param state 已初始化的只读演示状态。
 * @param canvas 有效的可写画布，所有权不变。
 */
static void draw_scene(const PathfindingDemoState *state, GfxCanvas *canvas)
{
    gfx_clear(canvas, gfx_rgb(18, 22, 31));
    gfx_rect(canvas, 8, 190, GFX_WIDTH - 16, 44, true, gfx_rgb(30, 36, 48));
    for (int cell = 0; cell < PATHFINDING_CELLS; ++cell) {
        uint32_t color = gfx_rgb(215, 220, 225);
        if (state->grid.cost[cell] == 0) {
            color = gfx_rgb(45, 51, 62);
        } else if (state->grid.cost[cell] > 1) {
            color = gfx_rgb(220, 132, 55);
        }
        if (state->search.discovered[cell] && !state->search.expanded[cell]) {
            color = gfx_rgb(70, 145, 230);
        }
        if (state->search.expanded[cell]) {
            color = gfx_rgb(65, 82, 112);
        }
        if (state->search.path[cell]) {
            color = gfx_rgb(250, 206, 66);
        }
        if (cell == state->search.current && state->search.phase == PATHFINDING_RUNNING) {
            color = gfx_rgb(220, 105, 225);
        }
        if (cell == PATH_START) {
            color = gfx_rgb(72, 215, 118);
        }
        if (cell == PATH_GOAL) {
            color = gfx_rgb(238, 82, 91);
        }
        int x = PATH_GRID_LEFT + cell % PATHFINDING_WIDTH * PATH_CELL_SIZE;
        int y = PATH_GRID_TOP + cell / PATHFINDING_WIDTH * PATH_CELL_SIZE;
        gfx_rect(canvas, x, y, PATH_CELL_SIZE - 1, PATH_CELL_SIZE - 1, true, color);
    }
    static const char *labels[] = {
        "START", "GOAL", "NORMAL", "WALL", "COST", "FRONTIER", "EXPANDED", "CURRENT", "PATH"
    };
    const uint32_t colors[] = {
        gfx_rgb(72, 215, 118), gfx_rgb(238, 82, 91),  gfx_rgb(215, 220, 225),
        gfx_rgb(45, 51, 62),   gfx_rgb(220, 132, 55), gfx_rgb(70, 145, 230),
        gfx_rgb(65, 82, 112),  gfx_rgb(220, 105, 225), gfx_rgb(250, 206, 66)
    };
    for (size_t item = 0; item < sizeof labels / sizeof labels[0]; ++item) {
        int column = (int)(item % 3);
        int row = (int)(item / 3);
        int x = 14 + column * 98;
        int y = 193 + row * 13;
        gfx_rect(canvas, x, y, 8, 8, true, colors[item]);
        draw_label(canvas, x + 12, y + 1, labels[item], gfx_rgb(228, 232, 239));
    }
}

/**
 * @brief 创建窗口并运行动画；统一清理部分创建或完整创建的 SDL 资源。
 * @param argc 参数数量，包含程序名。
 * @param argv 支持无参数或 --frames 1..10000，字符串由系统借用。
 * @return 正常退出为 EXIT_SUCCESS，参数错误或 SDL 失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    unsigned long frame_limit = 0;
    if (argc != 1) {
        char *end = nullptr;
        if (argc != 3 || strcmp(argv[1], "--frames") != 0 || argv[2][0] < '0' ||
            argv[2][0] > '9') {
            fputs("用法：pathfinding_demo [--frames 1..10000]\n", stderr);
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
    if (!gfx_sdl_open(&display, "Pathfinding | BFS / Dijkstra / A*")) {
        goto cleanup;
    }
    static GfxCanvas canvas;
    PathfindingDemoState state = {.running = true, .algorithm = PATHFINDING_BFS};
    reset_grid(&state);
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        SDL_SetError("无法获取计时频率");
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("寻路：1 BFS，2 Dijkstra，3 A*；Space 暂停/继续，N 单步，R 重置。"
         "左键切换墙，右键切换高代价地形，S 截图，Esc 退出。");
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
                 "Pathfinding | %s | %s | expanded=%zu | steps=%zu cost=%" PRIu32
                 " | Space/N 1/2/3 L/R R S Esc",
                 algorithm_name(state.algorithm), phase_name(state.search.phase),
                 state.search.expanded_count, state.search.path_length, state.search.path_cost);
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "pathfinding", path)) {
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
        if (spent < 1.0 / PATH_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / PATH_DRAW_HZ - spent) * 1000));
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
