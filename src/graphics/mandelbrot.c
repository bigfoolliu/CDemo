/** @file
 * @brief 将 z²+c 拆成实数运算，用有限次迭代绘制分形近似。
 * 数值核心与 SDL、事件、文件保存分离，便于独立断言验证。
 */
#include "mandelbrot.h"
#include <math.h>

/**
 * @brief 验证视图边界，限制浮点精度和最坏计算量。
 * @param view 待验证的只读视图，允许 nullptr。
 * @return 满足全部约束为 true。
 */
static bool valid_view(const MandelbrotView *view)
{
    return view != nullptr && isfinite(view->center_real) && isfinite(view->center_imag) &&
           isfinite(view->span) && view->span >= 1e-10 && view->span <= 4 &&
           fabs(view->center_real) <= 4 && fabs(view->center_imag) <= 4 &&
           fabs(view->center_real) + view->span / 2 <= 4 &&
           fabs(view->center_imag) + view->span * GFX_HEIGHT / GFX_WIDTH / 2 <= 4 &&
           view->max_iterations >= MANDELBROT_MIN_ITERATIONS &&
           view->max_iterations <= MANDELBROT_MAX_ITERATIONS;
}

/** @copydoc mandelbrot_reset */
void mandelbrot_reset(MandelbrotView *view)
{
    *view = (MandelbrotView){.center_real = -0.5, .center_imag = 0, .span = 3.5,
                            .max_iterations = MANDELBROT_DEFAULT_ITERATIONS};
}

/** @copydoc mandelbrot_map */
bool mandelbrot_map(const MandelbrotView *view, double u, double v, GfxPoint *point)
{
    if (!valid_view(view) || point == nullptr || !isfinite(u) || !isfinite(v) ||
        u < 0 || u > 1 || v < 0 || v > 1) { return false; }
    *point = (GfxPoint){view->center_real + (u - 0.5) * view->span,
                       view->center_imag + (0.5 - v) * view->span * GFX_HEIGHT / GFX_WIDTH};
    return true;
}

/**
 * @brief 已验证参数后的逃逸计算；渲染循环无需每个像素重复检查。
 * @param c 有限且位于支持范围的复数常量。
 * @param limit 有效迭代上限。
 * @return 迭代次数及是否逃逸。
 */
static MandelbrotEscape escape_unchecked(GfxPoint c, unsigned int limit)
{
    double real = 0, imag = 0; /* 每个 c 都从 z0=0 开始，而不是 z0=c。 */
    for (unsigned int i = 1; i <= limit; ++i) {
        /* (a+bi)² = (a²-b²) + 2ab i。
         * 两个新分量都必须使用旧 a、b；提前覆盖 real 会破坏虚部公式。
         */
        double next_real = real * real - imag * imag + c.x;
        imag = 2 * real * imag + c.y;
        real = next_real;
        /* 比较平方模长，避免每轮调用 sqrt。严格 >4，等于 4 尚未逃逸。
         * 每轮开始时旧 |z|<=2，且 |c.x|、|c.y|<=4，中间运算不会爆成无穷大。
         */
        if (real * real + imag * imag > 4) { return (MandelbrotEscape){i, true}; }
    }
    return (MandelbrotEscape){limit, false};
}

/** @copydoc mandelbrot_escape */
bool mandelbrot_escape(GfxPoint c, unsigned int limit, MandelbrotEscape *result)
{
    if (result == nullptr || !isfinite(c.x) || !isfinite(c.y) || fabs(c.x) > 4 || fabs(c.y) > 4 ||
        limit == 0 || limit > MANDELBROT_MAX_ITERATIONS) { return false; }
    *result = escape_unchecked(c, limit);
    return true;
}

/** @copydoc mandelbrot_zoom */
bool mandelbrot_zoom(MandelbrotView *view, double u, double v, double factor)
{
    GfxPoint anchor;
    if (!isfinite(factor) || factor <= 0 || !mandelbrot_map(view, u, v, &anchor)) { return false; }
    /* 先限制倍率，再乘，避免极大/极小输入造成浮点溢出或下溢。 */
    if (factor > 4 / view->span || factor < 1e-10 / view->span) { return false; }
    MandelbrotView next = *view;
    next.span *= factor;
    /* 从“锚点 = 中心 + 相对位置 × 新跨度”反解新中心，使鼠标所指位置不漂移。 */
    next.center_real = anchor.x - (u - 0.5) * next.span;
    next.center_imag = anchor.y - (0.5 - v) * next.span * GFX_HEIGHT / GFX_WIDTH;
    if (!valid_view(&next)) { return false; }
    *view = next; /* 成功后整体提交；失败不留下半更新状态。 */
    return true;
}

/** @copydoc mandelbrot_pan */
bool mandelbrot_pan(MandelbrotView *view, double real_fraction, double imag_fraction)
{
    if (!valid_view(view) || !isfinite(real_fraction) || !isfinite(imag_fraction)) { return false; }
    /* 合法中心最多移动 8 个复平面单位，先拒绝不可能成立的巨大比例。 */
    if (fabs(real_fraction) > 8 / view->span ||
        fabs(imag_fraction) > 8 / (view->span * GFX_HEIGHT / GFX_WIDTH)) { return false; }
    MandelbrotView next = *view;
    next.center_real += real_fraction * next.span;
    next.center_imag += imag_fraction * next.span * GFX_HEIGHT / GFX_WIDTH;
    if (!valid_view(&next)) { return false; }
    *view = next;
    return true;
}

/**
 * @brief 用迭代比例映射 RGB，未逃逸点用黑色。
 * @param escape 一次有效的逃逸结果。
 * @param limit 对应计算的迭代上限。
 * @return 不透明颜色；此多项式配色仅为视觉表达，没有集合判定含义。
 */
static uint32_t escape_color(MandelbrotEscape escape, unsigned int limit)
{
    if (!escape.escaped) { return gfx_rgb(0, 0, 0); }
    double t = (double)escape.iterations / limit, inverse = 1 - t;
    /* 基础多项式调色，不实现平滑逃逸着色；有限迭代会显示颜色分带。
     * 最后一步才逃逸时 t=1，单独给亮色，不能把它误画成“未逃逸”的黑色。
     */
    if (escape.iterations == limit) { return gfx_rgb(255, 220, 120); }
    return gfx_rgb((uint8_t)(9 * inverse * t * t * t * 255),
                   (uint8_t)(15 * inverse * inverse * t * t * 255),
                   (uint8_t)(8.5 * inverse * inverse * inverse * t * 255));
}

/** @copydoc mandelbrot_draw */
bool mandelbrot_draw(const MandelbrotView *view, GfxCanvas *canvas)
{
    if (!valid_view(view) || canvas == nullptr) { return false; }
    double pixel_step = view->span / GFX_WIDTH;
    for (int y = 0; y < GFX_HEIGHT; ++y) {
        for (int x = 0; x < GFX_WIDTH; ++x) {
            /* 采样像素中心而非左上角，保证上下行对应共轭位置，纵向单位比例相同。 */
            GfxPoint c = {view->center_real + (x + 0.5 - GFX_WIDTH / 2.0) * pixel_step,
                          view->center_imag + (GFX_HEIGHT / 2.0 - y - 0.5) * pixel_step};
            canvas->pixels[y][x] = escape_color(escape_unchecked(c, view->max_iterations),
                                              view->max_iterations);
        }
    }
    return true;
}
