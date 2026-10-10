/** @file @brief 无需 SDL 的 BMP 示例：绘制彩色图形并导出原始画布。 */
#include "bmp_capture.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** @brief 绘制不对称色块，便于观察 BMP 行方向与颜色顺序。 @return 成功为零。 */
int main(void)
{
    static GfxCanvas canvas;
    gfx_clear(&canvas, gfx_rgb(18, 23, 32));
    gfx_rect(&canvas, 10, 10, 100, 60, true, gfx_rgb(255, 60, 40));
    gfx_rect(&canvas, 210, 160, 100, 60, true, gfx_rgb(30, 90, 255));
    gfx_circle(&canvas, 160, 120, 45, gfx_rgb(255, 220, 70));
    gfx_line(&canvas, 0, 239, 319, 0, gfx_rgb(80, 220, 130));
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(&canvas, "demo", path)) {
        fprintf(stderr, "BMP 导出失败：%s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    printf("已导出：%s（320×240，24 位 BMP）\n", path);
    return EXIT_SUCCESS;
}
