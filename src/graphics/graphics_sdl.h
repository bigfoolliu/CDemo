/** @file
 * @brief 图形教学演示共用的 SDL 窗口适配层；纯 C 绘图核心无需此头文件。
 */
#ifndef CDEMO_GRAPHICS_SDL_H
#define CDEMO_GRAPHICS_SDL_H
#include "graphics.h"
#include <SDL.h>

/** 适配层拥有三个 SDL 资源，调用者负责最终 close。 */
typedef struct {
    SDL_Window *window;
    SDL_Renderer *renderer;
    SDL_Texture *texture;
} GfxWindow;

/**
 * @brief 初始化 SDL 并创建两倍放大窗口和 ARGB 流式纹理。
 * @param display 有效的输出地址，不能覆盖尚未关闭的窗口。
 * @param title 有效的 NUL 结尾标题。
 * @return 成功为 true；失败为 false，调用者仍须 close 释放部分资源。
 */
bool gfx_sdl_open(GfxWindow *display, const char *title);

/**
 * @brief 按 SDL 提供的行跨度复制画布并显示，不调用 SDL 图元绘制。
 * @param display 已创建的显示资源，借用。
 * @param canvas 有效的只读画布，借用。
 * @return 纹理锁定与提交成功为 true，否则为 false，可用 SDL_GetError 查看原因。
 */
bool gfx_sdl_present(const GfxWindow *display, const GfxCanvas *canvas);

/**
 * @brief 将画布指定矩形放大填满窗口，适合竖屏或局部构图。
 * @param display 已创建的显示资源，借用。
 * @param canvas 有效的只读画布，借用。
 * @param source 非空有效矩形，必须完全位于逻辑画布内。
 * @return 成功为 true；参数、纹理或呈现失败为 false。
 */
bool gfx_sdl_present_region(const GfxWindow *display, const GfxCanvas *canvas,
                            const SDL_Rect *source);

/**
 * @brief 释放已创建资源，清空句柄并退出 SDL；可清理创建失败的部分状态。
 * @param display 有效的可读写资源对象。
 * @note 各演示独占一个进程的 SDL 生命周期，不用于同时管理多个窗口。
 */
void gfx_sdl_close(GfxWindow *display);
#endif
