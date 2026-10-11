/** @file
 * @brief 最小软件绘制接口；不依赖窗口库，所有像素由调用者持有。
 */
#ifndef CDEMO_GRAPHICS_H
#define CDEMO_GRAPHICS_H

#include <stdint.h>

enum {
    GFX_WIDTH = 320,
    GFX_HEIGHT = 240,
    GFX_COORD_LIMIT = 16384
};

/** 固定画布：左上角为原点，x 向右、y 向下；颜色数值为 0xFFRRGGBB。 */
typedef struct {
    uint32_t pixels[GFX_HEIGHT][GFX_WIDTH];
} GfxCanvas;

/** 变换阶段的浮点坐标；提交直线绘制前四舍五入为整数像素。 */
typedef struct {
    double x;
    double y;
} GfxPoint;

/**
 * @brief 构造不透明 RGB 颜色。
 * @param red 红色分量，0..255。
 * @param green 绿色分量，0..255。
 * @param blue 蓝色分量，0..255。
 * @return 数值格式为 0xFFRRGGBB 的颜色。
 */
uint32_t gfx_rgb(uint8_t red, uint8_t green, uint8_t blue);

/**
 * @brief 将整幅画布填为指定颜色。
 * @param canvas 有效的可写画布，不转移所有权。
 * @param color 由 gfx_rgb 构造的不透明颜色。
 */
void gfx_clear(GfxCanvas *canvas, uint32_t color);

/**
 * @brief 写入一个像素，画布外坐标直接忽略。
 * @param canvas 有效的可写画布。
 * @param x 像素横坐标。
 * @param y 像素纵坐标。
 * @param color 不透明颜色。
 */
void gfx_pixel(GfxCanvas *canvas, int x, int y, uint32_t color);

/**
 * @brief 使用 Bresenham 算法画线，包含两个端点；范围外的请求忽略。
 * @param canvas 有效的可写画布。
 * @param x0 起点横坐标，范围为 ±GFX_COORD_LIMIT。
 * @param y0 起点纵坐标，同上。
 * @param x1 终点横坐标，同上。
 * @param y1 终点纵坐标，同上。
 * @param color 不透明颜色。
 */
void gfx_line(GfxCanvas *canvas, int x0, int y0, int x1, int y1, uint32_t color);

/**
 * @brief 绘制描边或填充矩形，非正尺寸或范围外请求忽略。
 * @param canvas 有效的可写画布。
 * @param x 左上角横坐标，矩形两端均须位于 ±GFX_COORD_LIMIT。
 * @param y 左上角纵坐标，同上。
 * @param width 像素宽度，1..GFX_COORD_LIMIT。
 * @param height 像素高度，1..GFX_COORD_LIMIT。
 * @param filled true 填充，false 只描边。
 * @param color 不透明颜色。
 */
void gfx_rect(GfxCanvas *canvas, int x, int y, int width, int height, bool filled, uint32_t color);

/**
 * @brief 使用中点算法画圆周，半径零画一个点；非法范围请求忽略。
 * @param canvas 有效的可写画布。
 * @param center_x 圆心横坐标，范围为 ±GFX_COORD_LIMIT。
 * @param center_y 圆心纵坐标，同上。
 * @param radius 半径，0..GFX_COORD_LIMIT。
 * @param color 不透明颜色。
 */
void gfx_circle(GfxCanvas *canvas, int center_x, int center_y, int radius, uint32_t color);

/**
 * @brief 将局部点依次缩放、旋转和平移；屏幕坐标中正角度呈顺时针。
 * @param point 按值传入的局部坐标。
 * @param translation 平移量，即局部原点的屏幕位置。
 * @param radians 旋转角度，单位为弧度。
 * @param scale 均匀缩放系数，允许零及负值。
 * @param output 有效的结果地址，仅成功时修改；无资源分配。
 * @return 输入及结果有限且输出处于绘图坐标范围时为 true，否则为 false。
 */
bool gfx_transform_point(GfxPoint point, GfxPoint translation, double radians, double scale,
                         GfxPoint *output);

#endif
