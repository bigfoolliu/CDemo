/** @file @brief 生命游戏的纯 C 接口：固定网格、双缓冲与像素绘制。 */
#ifndef CDEMO_LIFE_H
#define CDEMO_LIFE_H
#include "graphics.h"
#include <stddef.h>

enum { LIFE_WIDTH = 80, LIFE_HEIGHT = 60, LIFE_CELL_SIZE = 4, LIFE_HZ = 10 };
/** 预设种类；EMPTY 供清空和规则测试使用。 */
typedef enum { LIFE_EMPTY, LIFE_GLIDER, LIFE_BLINKER, LIFE_RANDOM } LifePattern;

/**
 * 两张网格属于此对象，无动态内存。current 只能为 0 或 1。
 * 更新时只读 cells[current]，写另一张，最后交换索引实现同步演化。
 */
typedef struct {
    bool cells[2][LIFE_HEIGHT][LIFE_WIDTH];
    unsigned int current;
    uint64_t generation; /* 已执行的代数；无符号计数达到上限后按标准规则回绕。 */
} LifeWorld;

/**
 * @brief 清空双缓冲并加载预设，代数归零。
 * @param world 有效的可写世界，旧状态被替换。
 * @param pattern 预设；未知值按空网格处理。随机预设总使用种子 42。
 */
void life_reset(LifeWorld *world, LifePattern pattern);
/**
 * @brief 按 B3/S23 规则同步推进一代；网格外始终死亡。
 * @param world 已初始化的可读写世界，current 必须为 0 或 1。
 */
void life_step(LifeWorld *world);
/**
 * @brief 读取细胞，网格外返回 false。
 * @param world 已初始化的只读世界。
 * @param x 列坐标，有效范围 [0,LIFE_WIDTH)。
 * @param y 行坐标，有效范围 [0,LIFE_HEIGHT)。
 * @return 指定细胞是否存活。
 */
bool life_get(const LifeWorld *world, int x, int y);
/**
 * @brief 切换一个细胞，并将代数归零；非法坐标不修改状态。
 * @param world 已初始化的可读写世界。
 * @param x 列坐标。
 * @param y 行坐标。
 * @return 坐标有效且已切换时返回 true。
 */
bool life_toggle(LifeWorld *world, int x, int y);
/** @brief 统计当前存活数。 @param world 已初始化的只读世界。 @return 存活数量。 */
size_t life_count(const LifeWorld *world);
/**
 * @brief 清屏并绘制网格；存活细胞为绿色，格间留一像素空隙。
 * @param world 已初始化的只读世界，不修改模拟状态。
 * @param canvas 有效的可写画布，所有权仍归调用者。
 */
void life_draw(const LifeWorld *world, GfxCanvas *canvas);
#endif
