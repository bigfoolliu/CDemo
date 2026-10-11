/** @file
 * @brief 从像素写入构建直线、矩形、圆和二维点变换。
 */
#include "graphics.h"
#include <limits.h>
#include <math.h>
#include <stdlib.h>

static_assert(INT_MAX >= 131072, "绘图教学坐标范围要求 int 至少支持 131072");

/**
 * @brief 判断整数是否处于教学接口支持的坐标范围。
 * @param value 待检查的坐标，不使用 abs 以避免 INT_MIN 溢出。
 * @return 位于闭区间 ±GFX_COORD_LIMIT 时为 true。
 */
static bool valid_coordinate(int value)
{
    return value >= -GFX_COORD_LIMIT && value <= GFX_COORD_LIMIT;
}

/** @copydoc gfx_rgb 参数及返回值契约见 graphics.h。 */
uint32_t gfx_rgb(uint8_t red, uint8_t green, uint8_t blue)
{
    return UINT32_C(0xff000000) | ((uint32_t)red << 16) | ((uint32_t)green << 8) | blue;
}

/** @copydoc gfx_clear 参数及所有权契约见 graphics.h。 */
void gfx_clear(GfxCanvas *canvas, uint32_t color)
{
    for (int y = 0; y < GFX_HEIGHT; ++y) {
        for (int x = 0; x < GFX_WIDTH; ++x) {
            canvas->pixels[y][x] = color;
        }
    }
}

/** @copydoc gfx_pixel 参数及越界行为见 graphics.h。 */
void gfx_pixel(GfxCanvas *canvas, int x, int y, uint32_t color)
{
    if (x >= 0 && x < GFX_WIDTH && y >= 0 && y < GFX_HEIGHT) {
        canvas->pixels[y][x] = color;
    }
}

/** @copydoc gfx_line 参数范围及端点规则见 graphics.h。 */
void gfx_line(GfxCanvas *canvas, int x0, int y0, int x1, int y1, uint32_t color)
{
    if (!valid_coordinate(x0) || !valid_coordinate(y0) || !valid_coordinate(x1) ||
        !valid_coordinate(y1)) {
        return;
    }
    int dx = abs(x1 - x0), dy = -abs(y1 - y0);
    int sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;
    for (;;) {
        gfx_pixel(canvas, x0, y0, color);
        if (x0 == x1 && y0 == y1) {
            break;
        }
        /* 误差比较决定本步沿 x、y 或两个方向移动，无需计算斜率。 */
        int twice = 2 * error;
        if (twice >= dy) {
            error += dy;
            x0 += sx;
        }
        if (twice <= dx) {
            error += dx;
            y0 += sy;
        }
    }
}

/** @copydoc gfx_rect 参数范围及尺寸规则见 graphics.h。 */
void gfx_rect(GfxCanvas *canvas, int x, int y, int width, int height, bool filled, uint32_t color)
{
    if (!valid_coordinate(x) || !valid_coordinate(y) || width <= 0 || height <= 0 ||
        width > GFX_COORD_LIMIT || height > GFX_COORD_LIMIT) {
        return;
    }
    int right = x + width - 1, bottom = y + height - 1;
    if (!valid_coordinate(right) || !valid_coordinate(bottom)) {
        return;
    }
    if (!filled) {
        gfx_line(canvas, x, y, right, y, color);
        gfx_line(canvas, right, y, right, bottom, color);
        gfx_line(canvas, right, bottom, x, bottom, color);
        gfx_line(canvas, x, bottom, x, y, color);
        return;
    }
    /* 填充前只截取可见区域，避免为巨大的画布外矩形进行无效扫描。 */
    int first_x = x < 0 ? 0 : x, first_y = y < 0 ? 0 : y;
    int last_x = right >= GFX_WIDTH ? GFX_WIDTH - 1 : right;
    int last_y = bottom >= GFX_HEIGHT ? GFX_HEIGHT - 1 : bottom;
    for (int row = first_y; row <= last_y; ++row) {
        for (int column = first_x; column <= last_x; ++column) {
            gfx_pixel(canvas, column, row, color);
        }
    }
}

/** @copydoc gfx_circle 参数范围及半径规则见 graphics.h。 */
void gfx_circle(GfxCanvas *canvas, int center_x, int center_y, int radius, uint32_t color)
{
    if (!valid_coordinate(center_x) || !valid_coordinate(center_y) || radius < 0 ||
        radius > GFX_COORD_LIMIT) {
        return;
    }
    int x = radius, y = 0, error = 1 - radius;
    while (x >= y) {
        /* 计算八分之一圆周，其余像素由八向对称得到。 */
        gfx_pixel(canvas, center_x + x, center_y + y, color);
        gfx_pixel(canvas, center_x + y, center_y + x, color);
        gfx_pixel(canvas, center_x - y, center_y + x, color);
        gfx_pixel(canvas, center_x - x, center_y + y, color);
        gfx_pixel(canvas, center_x - x, center_y - y, color);
        gfx_pixel(canvas, center_x - y, center_y - x, color);
        gfx_pixel(canvas, center_x + y, center_y - x, color);
        gfx_pixel(canvas, center_x + x, center_y - y, color);
        ++y;
        if (error < 0) {
            error += 2 * y + 1;
        } else {
            --x;
            error += 2 * (y - x) + 1;
        }
    }
}

/** @copydoc gfx_transform_point 参数、变换顺序及失败行为见 graphics.h。 */
bool gfx_transform_point(GfxPoint point, GfxPoint translation, double radians, double scale,
                         GfxPoint *output)
{
    if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(translation.x) ||
        !isfinite(translation.y) || !isfinite(radians) || !isfinite(scale)) {
        return false;
    }
    double x = point.x * scale, y = point.y * scale;
    double cosine = cos(radians), sine = sin(radians);
    GfxPoint result = {x * cosine - y * sine + translation.x,
                       x * sine + y * cosine + translation.y};
    if (!isfinite(result.x) || !isfinite(result.y) || fabs(result.x) > GFX_COORD_LIMIT ||
        fabs(result.y) > GFX_COORD_LIMIT) {
        return false;
    }
    *output = result;
    return true;
}
