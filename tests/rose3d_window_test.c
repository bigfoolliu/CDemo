/** @file @brief SDL dummy 驱动下验证玫瑰控制、竖屏截图及窗口区域。 */
#define main rose3d_demo_main
#include "../src/graphics/rose3d_demo.c"
#undef main
#include <assert.h>
#include <dirent.h>
#include <unistd.h>

/** @brief 暂停、单步、相机控制与重置遵循交互约定。 */
static void test_controls(void)
{
    static Rose3dDemoState state = {.running = true};
    rose3d_reset(&state.rose, UINT32_C(55));
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.repeat = 0;
    event.key.keysym.sym = SDLK_SPACE;
    assert(handle_event(&state, &event));
    assert(state.paused);
    double before = state.rose.time;
    event.key.keysym.sym = SDLK_n;
    assert(!handle_event(&state, &event));
    assert(state.rose.time > before);
    event.key.keysym.sym = SDLK_LEFT;
    handle_event(&state, &event);
    assert(state.rose.yaw < -0.20);
    event.key.keysym.sym = SDLK_r;
    assert(handle_event(&state, &event));
    assert(state.rose.time == 0.0 && state.paused);
}

/** @brief 发出截图按键并验证导出的 BMP 尺寸与文件长度。 */
static void test_portrait_capture(void)
{
    char previous[4096];
    char directory[] = "/tmp/cdemo-rose-window-XXXXXX";
    assert(getcwd(previous, sizeof previous) != nullptr);
    assert(mkdtemp(directory) != nullptr && chdir(directory) == 0);
    assert(SDL_Init(SDL_INIT_VIDEO) == 0);
    SDL_Event event = {.type = SDL_KEYDOWN};
    event.key.keysym.sym = SDLK_s;
    assert(SDL_PushEvent(&event) == 1);
    char *args[] = {"rose3d_demo", "--frames", "2", nullptr};
    assert(rose3d_demo_main(3, args) == EXIT_SUCCESS);
    DIR *images = opendir("exports");
    assert(images != nullptr);
    size_t count = 0;
    struct dirent *item;
    while ((item = readdir(images)) != nullptr) {
        if (strcmp(item->d_name, ".") == 0 || strcmp(item->d_name, "..") == 0) {
            continue;
        }
        char path[512];
        int length = snprintf(path, sizeof path, "exports/%s", item->d_name);
        assert(length > 0 && (size_t)length < sizeof path);
        FILE *image = fopen(path, "rb");
        assert(image != nullptr);
        unsigned char header[54];
        assert(fread(header, 1, sizeof header, image) == sizeof header);
        assert(header[0] == 'B' && header[1] == 'M');
        assert(header[18] == ROSE3D_VIEW_WIDTH && header[22] == ROSE3D_VIEW_HEIGHT);
        assert(fseek(image, 0, SEEK_END) == 0 && ftell(image) == 122934);
        assert(fclose(image) == 0 && remove(path) == 0);
        ++count;
    }
    assert(closedir(images) == 0 && count == 1);
    assert(rmdir("exports") == 0);
    assert(chdir(previous) == 0 && rmdir(directory) == 0);
}

/** @brief 执行 SDL 窗口适配层控制与截图集成测试。 */
int main(void)
{
    test_controls();
    test_portrait_capture();
    puts("三维粒子玫瑰窗口测试通过。");
    return 0;
}
