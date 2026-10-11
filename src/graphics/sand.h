/** @file @brief 落沙元胞模拟：沙、水、固定墙及纯 C 绘图。 */
#ifndef CDEMO_SAND_H
#define CDEMO_SAND_H
#include "graphics.h"

enum {
    SAND_CELL_SIZE = 2,
    SAND_WIDTH = GFX_WIDTH / SAND_CELL_SIZE,
    SAND_HEIGHT = GFX_HEIGHT / SAND_CELL_SIZE,
    SAND_HZ = 60,
    SAND_MAX_BRUSH = 8
};
/** 空气可被移动物质占据；沙可与水交换，墙永不自行移动。 */
typedef enum {
    SAND_AIR,
    SAND_GRAIN,
    SAND_WATER,
    SAND_WALL
} SandMaterial;

/** 调用者持有网格；reset 后仅通过接口修改，可只读观察 cells 和 steps。
 * 外圈为不可编辑墙，moved 是单步访问标记，不属于持久物质状态。
 */
typedef struct {
    uint8_t cells[SAND_HEIGHT][SAND_WIDTH];
    bool moved[SAND_HEIGHT][SAND_WIDTH];
    uint64_t steps;
    bool right_first; /* 独立方向位，即使步数饱和也继续左右交替。 */
} SandWorld;

/** @brief 清空内部并建立固定边界，恢复确定的更新顺序。
 * @param world 可写状态，nullptr 时不操作。
 */
void sand_reset(SandWorld *world);

/** @brief 以圆形笔刷绘制；添加只填空气，橡皮擦可清除内部任何物质。
 * @param world 已初始化可写状态，无资源分配。
 * @param x 中心列，1..SAND_WIDTH-2。
 * @param y 中心行，1..SAND_HEIGHT-2。
 * @param radius 网格半径，0..SAND_MAX_BRUSH；越界圆部分裁切。
 * @param material 合法物质，SAND_AIR 为橡皮擦。
 * @return 请求合法为 true（即使没有改变）；非法请求不修改状态。
 */
bool sand_paint(SandWorld *world, int x, int y, int radius, SandMaterial material);

/** @brief 自底向上更新一次，交换格子且保证同一步每个粒子最多移动一次。
 * @param world 已初始化可写状态；不要外部修改网格或固定边界。
 * @return 非空状态为 true，nullptr 为 false；步数溢出时饱和。
 */
bool sand_step(SandWorld *world);

/** @brief 把物质映射到颜色，每格显示为 2×2 像素。
 * @param world 已初始化只读状态。
 * @param canvas 调用者持有的可写画布。
 * @return 地址合法为 true；失败不修改画布。
 */
bool sand_draw(const SandWorld *world, GfxCanvas *canvas);
#endif
