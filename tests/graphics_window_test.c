/** @file
 * @brief 用 SDL dummy 驱动验证窗口显示路径和操作状态，不产生可见窗口。
 */
#define main graphics_demo_main
#include "../src/graphics/demo.c"
#undef main
#include <assert.h>

/** @brief 验证暂停、单步、重复键、重置、继续和两种退出事件。 */
static void test_controls(void)
{
    DemoState state = {.running = true, .angle = 1};
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_n;
    bool reset = handle_event(&state, &event);
    assert(!reset && state.angle == 1); /* 未暂停时不接受单步。 */
    event.key.keysym.sym = SDLK_SPACE;
    reset = handle_event(&state, &event);
    assert(reset && state.paused && state.angle == 1);
    event.key.keysym.sym = SDLK_n;
    reset = handle_event(&state, &event);
    assert(!reset && fabs(state.angle - (1 + PI / 240)) < 1e-12);
    event.key.repeat = 1;
    reset = handle_event(&state, &event);
    assert(!reset && fabs(state.angle - (1 + PI / 240)) < 1e-12);
    event.key.repeat = 0;
    event.key.keysym.sym = SDLK_r;
    reset = handle_event(&state, &event);
    assert(reset && state.angle == 0 && state.paused);
    event.key.keysym.sym = SDLK_SPACE;
    reset = handle_event(&state, &event);
    assert(reset && !state.paused);
    event.key.keysym.sym = SDLK_ESCAPE;
    reset = handle_event(&state, &event);
    assert(!reset && !state.running);
    state.running = true;
    event.type = SDL_QUIT;
    reset = handle_event(&state, &event);
    assert(!reset && !state.running);
}

/** @brief 验证 ARGB 像素复制、最近邻放大和有界主循环。 @return 成功为 0。 */
int main(void)
{
    test_controls();
    int result = SDL_Init(SDL_INIT_VIDEO);
    assert(result == 0);
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, GFX_WIDTH * 2, GFX_HEIGHT * 2,
                                                        32, SDL_PIXELFORMAT_ARGB8888);
    assert(surface != nullptr);
    SDL_Renderer *renderer = SDL_CreateSoftwareRenderer(surface);
    assert(renderer != nullptr);
    SDL_bool hint = SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
    assert(hint == SDL_TRUE);
    SDL_Texture *texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888,
                                            SDL_TEXTUREACCESS_STREAMING, GFX_WIDTH, GFX_HEIGHT);
    assert(texture != nullptr);
    static GfxCanvas buffer;
    uint32_t background = gfx_rgb(1, 2, 3), foreground = gfx_rgb(100, 150, 200);
    gfx_clear(&buffer, background);
    gfx_pixel(&buffer, 5, 7, foreground);
    bool ok = gfx_sdl_present(&(GfxWindow){.renderer = renderer, .texture = texture}, &buffer);
    assert(ok);
    for (int y = 14; y <= 15; ++y) {
        for (int x = 10; x <= 11; ++x) {
            const uint32_t *row = (const uint32_t *)((const unsigned char *)surface->pixels +
                                                   (size_t)y * (size_t)surface->pitch);
            assert(row[x] == foreground && row[x - 2] == background);
        }
    }
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_FreeSurface(surface);
    SDL_Quit();
    char *arguments[] = {"graphics_demo", "--frames", "2", nullptr};
    result = graphics_demo_main(3, arguments);
    assert(result == EXIT_SUCCESS);
    puts("graphics-window：ARGB 呈现、两倍像素、暂停/单步/重置/退出及主循环验证通过");
    return 0;
}
