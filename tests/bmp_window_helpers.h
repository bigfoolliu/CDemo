/** @file @brief 仅供窗口测试：在临时目录内发送 S 事件并验证实际截图。 */
#ifndef CDEMO_BMP_WINDOW_HELPERS_H
#define CDEMO_BMP_WINDOW_HELPERS_H
#include <assert.h>
#include <dirent.h>
#include <unistd.h>

/**
 * @brief 验证演示主循环的截图路径，并确保导出失败不终止演示。
 * @param entry 被测试演示的 main 函数，支持 --frames N。
 */
static void test_capture_loop(int (*entry)(int, char *[]))
{
    char previous[4096], directory[] = "/tmp/cdemo-bmp-window-XXXXXX";
    assert(getcwd(previous, sizeof previous) != nullptr);
    assert(mkdtemp(directory) != nullptr && chdir(directory) == 0);
    for (int failure = 0; failure < 2; ++failure) {
        if (failure) {
            FILE *block = fopen("exports", "wb");
            assert(block != nullptr && fclose(block) == 0);
        }
        assert(SDL_Init(SDL_INIT_VIDEO) == 0);
        SDL_Event event = {.type = SDL_KEYDOWN};
        event.key.keysym.sym = SDLK_s;
        assert(SDL_PushEvent(&event) == 1);
        char *args[] = {"demo", "--frames", "2", nullptr};
        assert(entry(3, args) == EXIT_SUCCESS);
        if (!failure) {
            DIR *images = opendir("exports");
            assert(images != nullptr);
            size_t count = 0;
            struct dirent *item;
            while ((item = readdir(images)) != nullptr) {
                if (strcmp(item->d_name, ".") == 0 || strcmp(item->d_name, "..") == 0) { continue; }
                char path[512];
                int n = snprintf(path, sizeof path, "exports/%s", item->d_name);
                assert(n > 0 && (size_t)n < sizeof path);
                FILE *image = fopen(path, "rb");
                assert(image != nullptr && fseek(image, 0, SEEK_END) == 0 && ftell(image) == 230454);
                assert(fclose(image) == 0 && remove(path) == 0);
                ++count;
            }
            assert(closedir(images) == 0 && count == 1 && rmdir("exports") == 0);
        } else {
            assert(remove("exports") == 0); /* 截图失败也没有移除阻挡文件。 */
        }
    }
    assert(chdir(previous) == 0 && rmdir(directory) == 0);
}
#endif
