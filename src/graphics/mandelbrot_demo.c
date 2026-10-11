/** @file @brief 曼德勃罗分形窗口：锚点缩放、平移、迭代上限和 BMP 导出。 */
#include "graphics_sdl.h"
#include "bmp_capture.h"
#include "mandelbrot.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    MANDELBROT_DRAW_HZ = 60
};
/** dirty 控制重新计算；窗口始终响应事件并呈现缓存画布。 */
typedef struct {
    MandelbrotView view;
    bool running, dirty, capture, limited;
} MandelbrotDemoState;

/**
 * @brief 将一次数值操作的结果转为重绘请求或边界反馈。
 * @param state 有效的可读写窗口状态。
 * @param changed 数值核心是否接受这次修改。
 * @return changed 原值，便于事件测试。
 */
static bool changed_view(MandelbrotDemoState *state, bool changed)
{
    state->limited = !changed;
    if (changed) {
        state->dirty = true;
    }
    return changed;
}

/**
 * @brief 把鼠标坐标转为归一化锚点，再请求缩放。
 * @param state 有效的可读写窗口状态。
 * @param x 窗口鼠标横坐标。
 * @param y 窗口鼠标纵坐标。
 * @param width 当前窗口宽度，与鼠标坐标使用相同单位。
 * @param height 当前窗口高度。
 * @param factor 宽度倍率，0.5 放大、2 缩小。
 * @return 有效坐标且缩放成功时 true；窗口外的坐标直接忽略。
 */
static bool zoom_mouse(MandelbrotDemoState *state, int x, int y, int width, int height,
                       double factor)
{
    if (width <= 0 || height <= 0 || x < 0 || x >= width || y < 0 || y >= height) {
        return false;
    }
    return changed_view(
        state, mandelbrot_zoom(&state->view, (x + 0.5) / width, (y + 0.5) / height, factor));
}

/**
 * @brief 处理输入；自动重复键忽略，滚轮每个事件最多缩放一级。
 * @param state 有效的可读写窗口状态。
 * @param event 有效的只读 SDL 事件。
 * @param width 当前窗口宽度。
 * @param height 当前窗口高度。
 * @param mouse_x 滚轮事件时鼠标横坐标；按钮事件使用自己的坐标。
 * @param mouse_y 滚轮事件时鼠标纵坐标。
 * @return 视图改变为 true；截图/退出和无效操作返回 false。
 */
static bool handle_event(MandelbrotDemoState *state, const SDL_Event *event, int width, int height,
                         int mouse_x, int mouse_y)
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
        case SDLK_r:
            mandelbrot_reset(&state->view);
            return changed_view(state, true);
        case SDLK_LEFT:
            return changed_view(state, mandelbrot_pan(&state->view, -0.25, 0));
        case SDLK_RIGHT:
            return changed_view(state, mandelbrot_pan(&state->view, 0.25, 0));
        case SDLK_UP:
            return changed_view(state, mandelbrot_pan(&state->view, 0, 0.25));
        case SDLK_DOWN:
            return changed_view(state, mandelbrot_pan(&state->view, 0, -0.25));
        case SDLK_PLUS:
        case SDLK_EQUALS:
        case SDLK_KP_PLUS:
            if (state->view.max_iterations >= MANDELBROT_MAX_ITERATIONS) {
                return changed_view(state, false);
            }
            state->view.max_iterations = state->view.max_iterations > MANDELBROT_MAX_ITERATIONS / 2
                                             ? MANDELBROT_MAX_ITERATIONS
                                             : state->view.max_iterations * 2;
            return changed_view(state, true);
        case SDLK_MINUS:
        case SDLK_KP_MINUS:
            if (state->view.max_iterations <= MANDELBROT_MIN_ITERATIONS) {
                return changed_view(state, false);
            }
            state->view.max_iterations = state->view.max_iterations < MANDELBROT_MIN_ITERATIONS * 2
                                             ? MANDELBROT_MIN_ITERATIONS
                                             : state->view.max_iterations / 2;
            return changed_view(state, true);
        default:
            break;
        }
    } else if (event->type == SDL_MOUSEBUTTONDOWN) {
        if (event->button.button == SDL_BUTTON_LEFT || event->button.button == SDL_BUTTON_RIGHT) {
            return zoom_mouse(state, event->button.x, event->button.y, width, height,
                              event->button.button == SDL_BUTTON_LEFT ? 0.5 : 2);
        }
    } else if (event->type == SDL_MOUSEWHEEL && event->wheel.y != 0) {
        bool zoom_in = event->wheel.y > 0;
        if (event->wheel.direction == SDL_MOUSEWHEEL_FLIPPED) {
            zoom_in = !zoom_in;
        }
        return zoom_mouse(state, mouse_x, mouse_y, width, height, zoom_in ? 0.5 : 2);
    }
    return false;
}

/**
 * @brief 创建窗口并演示曼德勃罗分形；统一清理部分创建或完整创建的资源。
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
            fputs("用法：mandelbrot_demo [--frames 1..10000]\n", stderr);
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
    if (!gfx_sdl_open(&display, "Mandelbrot")) {
        goto cleanup;
    }
    static GfxCanvas canvas; /* 大画布放在静态存储区，避免占用约 300 KiB 栈。 */
    MandelbrotDemoState state = {.running = true};
    mandelbrot_reset(&state.view);
    state.dirty = true;
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        SDL_SetError("无法获取计时频率");
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("曼德勃罗分形：左键/滚轮向上放大，右键/滚轮向下缩小，方向键平移，+/- 迭代，R 重置，S 保存 "
         "BMP，Esc 退出。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            int width, height, mouse_x, mouse_y;
            SDL_GetWindowSize(display.window, &width, &height);
            SDL_GetMouseState(&mouse_x, &mouse_y);
            handle_event(&state, &event, width, height, mouse_x, mouse_y);
        }
        if (!state.running) {
            break;
        }
        /* 静态分形只有视图/迭代上限改变时才重新计算，避免每帧重复大量迭代。
         * 绘制同步执行；最坏计算量受 320×240 和 1024 次上限约束。
         */
        if (state.dirty) {
            if (!mandelbrot_draw(&state.view, &canvas)) {
                SDL_SetError("非法分形视图");
                goto cleanup;
            }
            state.dirty = false;
        }
        char title[256];
        snprintf(title, sizeof title,
                 "Mandelbrot | center=(%.12g,%.12g) | span=%.3g | iter=%u | %s | click/wheel "
                 "arrows +/- R S Esc",
                 state.view.center_real, state.view.center_imag, state.view.span,
                 state.view.max_iterations, state.limited ? "limit reached" : "ready");
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            /* 此时画布已完整绘制，保存逻辑像素；失败报错后继续演示。 */
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "mandelbrot", path)) {
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
        } /* 统计显示帧，不是每像素迭代次数。 */
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        /* 目标显示 60 FPS；只有操作后重算图像，计算耗时可能超过一帧。
         * 无动画时间步，也不承诺缩放重绘时维持 60 FPS。
         */
        if (spent < 1.0 / MANDELBROT_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / MANDELBROT_DRAW_HZ - spent) * 1000));
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
