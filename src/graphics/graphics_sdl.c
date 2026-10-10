/** @file
 * @brief SDL 资源创建、像素提交与清理；几何算法仍由 graphics.c 提供。
 */
#include "graphics_sdl.h"
#include <string.h>

/** @copydoc gfx_sdl_open */
bool gfx_sdl_open(GfxWindow *display, const char *title)
{
    /* 顺序创建资源：SDL → 窗口 → 渲染器 → 纹理。
     * 每次成功创建后立即保存句柄，便于失败时释放此前已创建的部分资源。
     */
    *display = (GfxWindow){0};
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        return false;
    }
    /* 最近邻放大保持像素块边缘清晰；在创建纹理前设置缩放方式。 */
    if (SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0") == SDL_FALSE) {
        SDL_SetError("无法设置最近邻像素放大");
        return false;
    }
    /* 逻辑画布 320×240，实际窗口 640×480；几何计算仍使用逻辑坐标。 */
    display->window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       GFX_WIDTH * 2, GFX_HEIGHT * 2, 0);
    if (display->window == nullptr) { return false; }
    /* 使用软件渲染器显示整张画布；粒子位置和像素颜色仍由纯 C 代码计算。 */
    display->renderer = SDL_CreateRenderer(display->window, -1, SDL_RENDERER_SOFTWARE);
    if (display->renderer == nullptr) { return false; }
    /* STREAMING 允许每帧锁定并写入；ARGB8888 匹配画布的 0xFFRRGGBB 数值格式。 */
    display->texture = SDL_CreateTexture(display->renderer, SDL_PIXELFORMAT_ARGB8888,
                                         SDL_TEXTUREACCESS_STREAMING, GFX_WIDTH, GFX_HEIGHT);
    return display->texture != nullptr;
}

/** @copydoc gfx_sdl_present */
bool gfx_sdl_present(const GfxWindow *display, const GfxCanvas *canvas)
{
    void *pixels; /* SDL 借出的可写纹理地址，只在锁定期间使用，不由调用者 free。 */
    int pitch; /* 相邻两行起始地址相隔的字节数，可能包含行末填充。 */
    if (SDL_LockTexture(display->texture, nullptr, &pixels, &pitch) != 0) {
        return false;
    }
    for (int y = 0; y < GFX_HEIGHT; ++y) {
        /* 按字节计算目标行地址，再复制一行画布。
         * 不能假设 pitch == 宽度 * 4，否则在存在填充时会写错行。
         * unsigned char * 可按单字节偏移；不对 void * 做非标准指针运算。
         */
        memcpy((unsigned char *)pixels + (size_t)y * (size_t)pitch,
               canvas->pixels[y], sizeof(canvas->pixels[y]));
    }
    SDL_UnlockTexture(display->texture); /* 提交写入后不再使用 pixels 指针。 */
    /* 清理渲染目标，再将整张纹理缩放到窗口；这里没有画单个粒子。 */
    if (SDL_RenderClear(display->renderer) != 0 ||
        SDL_RenderCopy(display->renderer, display->texture, nullptr, nullptr) != 0) {
        return false;
    }
    SDL_RenderPresent(display->renderer); /* 将本帧渲染结果交给窗口显示。 */
    return true;
}

/** @copydoc gfx_sdl_close */
void gfx_sdl_close(GfxWindow *display)
{
    /* 按创建的逆序释放，避免纹理仍引用已销毁的渲染器。
     * 检查空句柄使创建失败的部分状态也能安全清理。
     */
    if (display->texture != nullptr) { SDL_DestroyTexture(display->texture); }
    if (display->renderer != nullptr) { SDL_DestroyRenderer(display->renderer); }
    if (display->window != nullptr) { SDL_DestroyWindow(display->window); }
    *display = (GfxWindow){0}; /* 清除失效句柄，避免调用者继续使用悬空资源。 */
    SDL_Quit();
}
