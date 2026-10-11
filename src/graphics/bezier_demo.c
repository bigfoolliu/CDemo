/** @file @brief 贝塞尔曲线窗口：鼠标编辑、坐标映射和 BMP 导出。 */
#include "graphics_sdl.h"
#include "bmp_capture.h"
#include "bezier.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    BEZIER_DRAW_HZ = 60,
    BEZIER_HIT_RADIUS = 8
};
/** 选择与拖动分开：松开鼠标保留高亮；offset 保存点击时的抓取偏移。 */
typedef struct {
    BezierCurve curve;
    GfxPoint offset;
    int selected;
    bool running, dirty, capture, dragging;
} BezierDemoState;

/** @brief 将窗口坐标映射到逻辑像素中心，窗口外坐标允许暂时越界。
 * @param x 窗口横坐标。
 * @param y 窗口纵坐标。
 * @param width 正数窗口宽度，和鼠标坐标采用相同单位。
 * @param height 正数窗口高度。
 * @return 尚未裁切的逻辑坐标；浮点运算避免整数乘法溢出。
 */
static GfxPoint logical_mouse(int x, int y, int width, int height)
{
    return (GfxPoint){(x + 0.5) * GFX_WIDTH / width - 0.5, (y + 0.5) * GFX_HEIGHT / height - 0.5};
}

/** @brief 将有限坐标限制在闭区间内。
 * @param value 待裁切数值。
 * @param maximum 上界，下界固定为零。
 * @return 裁切后的坐标。
 */
static double clamp_coordinate(double value, double maximum)
{
    return value < 0 ? 0 : value > maximum ? maximum : value;
}

/** @brief 处理一次输入；取消拖动后仍保留选择，重置则清除选择。
 * @param state 有效的可读写窗口状态。
 * @param event 有效的只读 SDL 事件。
 * @param width 当前窗口宽度。
 * @param height 当前窗口高度。
 */
static void handle_event(BezierDemoState *state, const SDL_Event *event, int width, int height)
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
            bezier_reset(&state->curve);
            state->selected = -1;
            state->dragging = false;
            state->dirty = true;
            break;
        default:
            break;
        }
    } else if (event->type == SDL_WINDOWEVENT &&
               (event->window.event == SDL_WINDOWEVENT_FOCUS_LOST ||
                event->window.event == SDL_WINDOWEVENT_LEAVE)) {
        state->dragging = false; /* 防止窗口外松开鼠标后继续拖动。 */
    } else if (event->type == SDL_MOUSEBUTTONUP && event->button.button == SDL_BUTTON_LEFT) {
        state->dragging = false;
    } else if (event->type == SDL_MOUSEBUTTONDOWN && event->button.button == SDL_BUTTON_LEFT) {
        state->dragging = false;
        if (width <= 0 || height <= 0 || event->button.x < 0 || event->button.x >= width ||
            event->button.y < 0 || event->button.y >= height) {
            return;
        }
        GfxPoint mouse = logical_mouse(event->button.x, event->button.y, width, height);
        mouse.x = clamp_coordinate(mouse.x, GFX_WIDTH - 1);
        mouse.y = clamp_coordinate(mouse.y, GFX_HEIGHT - 1);
        state->selected = bezier_hit(&state->curve, mouse, BEZIER_HIT_RADIUS);
        state->dirty = true;
        if (state->selected >= 0) {
            GfxPoint point = state->curve.points[state->selected];
            state->offset = (GfxPoint){point.x - mouse.x, point.y - mouse.y};
            state->dragging = true;
        }
    } else if (event->type == SDL_MOUSEMOTION && state->dragging) {
        if ((event->motion.state & SDL_BUTTON_LMASK) == 0) {
            state->dragging = false;
            return;
        }
        if (width <= 0 || height <= 0) {
            return;
        }
        GfxPoint mouse = logical_mouse(event->motion.x, event->motion.y, width, height);
        GfxPoint point = {clamp_coordinate(mouse.x + state->offset.x, GFX_WIDTH - 1),
                          clamp_coordinate(mouse.y + state->offset.y, GFX_HEIGHT - 1)};
        GfxPoint old = state->curve.points[state->selected];
        if ((point.x != old.x || point.y != old.y) &&
            bezier_move(&state->curve, state->selected, point)) {
            state->dirty = true;
        }
    }
}

/**
 * @brief 创建窗口并演示贝塞尔曲线编辑器；统一清理部分创建或完整创建的资源。
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
            fputs("用法：bezier_demo [--frames 1..10000]\n", stderr);
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
    if (!gfx_sdl_open(&display, "Bezier")) {
        goto cleanup;
    }
    static GfxCanvas canvas; /* 大画布放在静态存储区，避免占用约 300 KiB 栈。 */
    BezierDemoState state = {.running = true};
    bezier_reset(&state.curve);
    state.selected = -1;
    state.dirty = true;
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        SDL_SetError("无法获取计时频率");
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("贝塞尔曲线编辑器：左键拖动控制点，R 重置，S 保存 BMP，Esc 退出；标题显示 P0..P3 坐标。");
    while (state.running) {
        Uint64 started = SDL_GetPerformanceCounter();
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            int width, height;
            SDL_GetWindowSize(display.window, &width, &height);
            handle_event(&state, &event, width, height);
        }
        if (!state.running) {
            break;
        }
        /* 控制点或选择状态改变时重绘，其余帧呈现缓存画布。 */
        if (state.dirty) {
            if (!bezier_draw(&state.curve, state.selected, &canvas)) {
                SDL_SetError("非法曲线状态");
                goto cleanup;
            }
            state.dirty = false;
        }
        char title[256];
        snprintf(title, sizeof title,
                 "Bezier | P0=(%.0f,%.0f) P1=(%.0f,%.0f) P2=(%.0f,%.0f) P3=(%.0f,%.0f) | "
                 "selected=%d | drag R S Esc",
                 state.curve.points[0].x, state.curve.points[0].y, state.curve.points[1].x,
                 state.curve.points[1].y, state.curve.points[2].x, state.curve.points[2].y,
                 state.curve.points[3].x, state.curve.points[3].y, state.selected);
        SDL_SetWindowTitle(display.window, title);
        if (state.capture) {
            /* 此时画布已完整绘制，保存逻辑像素；失败报错后继续演示。 */
            char path[BMP_CAPTURE_PATH_SIZE];
            if (gfx_bmp_capture(&canvas, "bezier", path)) {
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
        } /* 统计显示帧，用于自动退出验证。 */
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        /* 目标显示 60 FPS；操作时才重绘，计时和系统调度会影响实际帧率。 */
        if (spent < 1.0 / BEZIER_DRAW_HZ) {
            SDL_Delay((Uint32)((1.0 / BEZIER_DRAW_HZ - spent) * 1000));
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
