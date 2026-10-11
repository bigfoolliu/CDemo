/** @file @brief 曼德勃罗分形的纯 C 数值、视图和像素绘制接口。 */
#ifndef CDEMO_MANDELBROT_H
#define CDEMO_MANDELBROT_H
#include "graphics.h"

enum {
    MANDELBROT_DEFAULT_ITERATIONS = 128,
    MANDELBROT_MIN_ITERATIONS = 32,
    MANDELBROT_MAX_ITERATIONS = 1024
};
/**
 * 实轴向右，虚轴向上；span 是整个画面的实轴宽度。
 * 虚轴高度按画布 4:3 比例计算，保证复平面上的单位长度显示为正方形像素。
 */
typedef struct {
    double center_real, center_imag, span;
    unsigned int max_iterations;
} MandelbrotView;
/** 逃逸发生在最后一次允许迭代时，也必须用 escaped 区分“未逃逸”。 */
typedef struct {
    unsigned int iterations;
    bool escaped;
} MandelbrotEscape;

/** @brief 恢复中心 (-0.5,0)、宽度 3.5、迭代上限 128。 @param view 有效的可写视图。 */
void mandelbrot_reset(MandelbrotView *view);
/**
 * @brief 将归一化画布位置映射到复平面，纵向反转使虚轴向上。
 * @param view 合法只读视图；宽度 1e-10..4，整个可见范围位于 [-4,4]²。
 * @param u 水平比例 0..1，0 为左边缘，1 为右边缘。
 * @param v 垂直比例 0..1，0 为上边缘，1 为下边缘。
 * @param point 有效的输出地址，x 为实部、y 为虚部。
 * @return 成功为 true，非法/非有限输入返回 false 且不修改输出。
 */
bool mandelbrot_map(const MandelbrotView *view, double u, double v, GfxPoint *point);
/**
 * @brief 执行 z=0 起始的 z=z²+c，模长严格大于 2 时判定逃逸。
 * @param c 复数常量，x/y 分别为实/虚部，均须有限且位于 [-4,4]。
 * @param limit 最大迭代数，1..MANDELBROT_MAX_ITERATIONS。
 * @param result 有效的输出地址，失败时不修改。
 * @return 参数有效且计算完成为 true，否则 false。
 * @note 未逃逸时 iterations=limit；这不是严格证明 c 属于集合。
 */
bool mandelbrot_escape(GfxPoint c, unsigned int limit, MandelbrotEscape *result);
/**
 * @brief 围绕指定位置缩放，保持其复平面坐标不变。
 * @param view 合法的可读写视图。
 * @param u 水平锚点比例，0..1。
 * @param v 垂直锚点比例，0..1。
 * @param factor 正的有限宽度倍率；0.5 放大两倍，2 缩小两倍。
 * @return 新视图合法为 true；超限或非法参数返回 false，原视图不变。
 */
bool mandelbrot_zoom(MandelbrotView *view, double u, double v, double factor);
/**
 * @brief 平移视图，偏移按当前可见宽度和高度的比例计算。
 * @param view 合法的可读写视图。
 * @param real_fraction 实轴偏移比例，正值向右。
 * @param imag_fraction 虚轴偏移比例，正值向上。
 * @return 新视图合法为 true；非法/超限时 false 且视图不变。
 */
bool mandelbrot_pan(MandelbrotView *view, double real_fraction, double imag_fraction);
/**
 * @brief 对所有像素中心执行逃逸测试并绘制，不读取时钟或分配内存。
 * @param view 合法的只读视图，迭代上限 32..1024。
 * @param canvas 有效的可写画布。
 * @return 成功为 true；视图/地址非法返回 false，画布不变。
 */
bool mandelbrot_draw(const MandelbrotView *view, GfxCanvas *canvas);
#endif
