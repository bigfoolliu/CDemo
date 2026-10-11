/** @file @brief 三次贝塞尔曲线：纯 C 插值、控制点编辑和软件绘制。 */
#ifndef CDEMO_BEZIER_H
#define CDEMO_BEZIER_H

#include "graphics.h"

enum { BEZIER_POINT_COUNT = 4, BEZIER_SEGMENTS = 128 };
/** P0/P3 是端点，P1/P2 控制切线；所有点属于调用者，坐标限制在画布内。 */
typedef struct {
    GfxPoint points[BEZIER_POINT_COUNT];
} BezierCurve;

/** @brief 恢复初始曲线。
 * @param curve 可写曲线；nullptr 时不操作。
 */
void bezier_reset(BezierCurve *curve);

/** @brief 用 de Casteljau 算法计算曲线上一点。
 * @param curve 有效曲线，控制点须有限且位于画布内。
 * @param t 曲线参数，有限数且在 [0,1] 内；不是动画时间或弧长比例。
 * @param output 可写结果地址，仅成功时修改；可与控制点别名。
 * @return 输入有效为 true，否则 false；不分配资源。
 */
bool bezier_evaluate(const BezierCurve *curve, double t, GfxPoint *output);

/** @brief 移动一个控制点；不自动裁切非法坐标。
 * @param curve 可写的有效曲线，失败时保持原值。
 * @param index 控制点编号，0..3。
 * @param point 新位置，有限且在 [0,319] × [0,239] 内。
 * @return 修改成功为 true，否则 false。
 */
bool bezier_move(BezierCurve *curve, int index, GfxPoint point);

/** @brief 查找命中半径内距离最近的控制点，等距时选择编号较小者。
 * @param curve 有效曲线。
 * @param point 查询位置，有限且位于画布内。
 * @param radius 逻辑像素半径，有限且在 [0,GFX_COORD_LIMIT] 内。
 * @return 控制点编号；未命中或输入无效返回 -1。
 */
int bezier_hit(const BezierCurve *curve, GfxPoint point, double radius);

/** @brief 清屏并绘制控制多边形、曲线和控制点；边缘标记由绘图核心裁切。
 * @param curve 有效的只读曲线。
 * @param selected 高亮的控制点编号，-1 表示无选择，其他允许值为 0..3。
 * @param canvas 可写画布，仍归调用者所有；输入无效时不修改。
 * @return 绘制成功为 true，否则 false。
 */
bool bezier_draw(const BezierCurve *curve, int selected, GfxCanvas *canvas);

#endif
