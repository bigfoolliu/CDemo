/** @file @brief 二维阻尼波：纯 C 高度场、局部扰动和水面明暗绘制。 */
#ifndef CDEMO_RIPPLE_H
#define CDEMO_RIPPLE_H
#include "graphics.h"

enum {
    RIPPLE_CELL_SIZE = 2,
    RIPPLE_WIDTH = GFX_WIDTH / RIPPLE_CELL_SIZE,
    RIPPLE_HEIGHT = GFX_HEIGHT / RIPPLE_CELL_SIZE,
    RIPPLE_HZ = 60,
    RIPPLE_MAX_RADIUS = 16,
    RIPPLE_HEIGHT_LIMIT = 64
};
/** 网格单位下的固定数值系数；改变它们需重新检查稳定条件。 */
#define RIPPLE_COURANT_SQUARED 0.20f
#define RIPPLE_DAMPING 0.01f

/** 调用者持有的两层高度场，不动态分配；先 reset，之后仅通过接口修改。
 * current 指向当前层，另一层保存上一时刻；最外圈始终为零。
 * 可只读观察 height/steps；不要直接修改，否则可能破坏有限数与边界不变量。
 */
typedef struct {
    float height[2][RIPPLE_HEIGHT][RIPPLE_WIDTH];
    unsigned current;
    uint64_t steps;
} RippleWorld;

/** @brief 清空水面并将步数归零。
 * @param world 可写状态，nullptr 时不操作；大对象宜放静态区或堆上。
 */
void ripple_reset(RippleWorld *world);

/** @brief 添加圆形平滑位移，同时修改两层以保持原有速度。
 * @param world 已初始化的可写状态，非法请求保持原值。
 * @param x 网格中心横坐标，1..RIPPLE_WIDTH-2。
 * @param y 网格中心纵坐标，1..RIPPLE_HEIGHT-2。
 * @param radius 网格半径，1..RIPPLE_MAX_RADIUS；靠边部分裁切到内圈。
 * @param strength 有限位移，范围 [-8,8]；正负对应凸起与凹陷。
 * @return 成功为 true，否则 false；累计高度限制在 ±RIPPLE_HEIGHT_LIMIT。
 */
bool ripple_disturb(RippleWorld *world, int x, int y, int radius, float strength);

/** @brief 前进一个固定 1/RIPPLE_HZ 秒的数值步，固定零高度边界会反射波。
 * @param world 已初始化的可写状态；读取当前层，覆盖旧层，再交换层编号。
 * @return 状态地址/层编号合法为 true；失败不修改状态，步数溢出时饱和。
 */
bool ripple_step(RippleWorld *world);

/** @brief 用高度与坡度生成蓝色明暗，每个网格覆盖 2×2 逻辑像素。
 * @param world 已初始化的只读状态。
 * @param canvas 调用者持有的可写画布，非法地址/层编号时不修改。
 * @return 绘制成功为 true，否则 false；这是示意着色，不是流体光学计算。
 */
bool ripple_draw(const RippleWorld *world, GfxCanvas *canvas);
#endif
