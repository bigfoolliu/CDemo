/** @file
 * @brief 固定粒子池与喷泉模拟，使用纯 C，不依赖 SDL。
 *
 * 阅读顺序：Particle 保存单个粒子的状态，ParticleSystem 管理全部槽位，
 * reset 初始化系统，step 推进模拟，draw 将当前状态转换为像素。
 * 调用者拥有系统和画布；本模块不分配内存，也不保存调用者的指针。
 */
#ifndef CDEMO_PARTICLES_H
#define CDEMO_PARTICLES_H
#include "graphics.h"
#include <stddef.h>

/* 容量限制同时约束内存和每步扫描量；模拟频率决定一个寿命单位的时长。
 * 演示主循环也借用 PARTICLE_HZ 作为绘制限速目标，但绘制与更新分别执行。
 */
enum { PARTICLE_CAPACITY = 256, PARTICLE_HZ = 60 };

/**
 * 单粒子状态。位置用浮点数累积，绘制时才取整，避免慢速移动被整数截断。
 * 存活粒子必须满足：位置、速度有限，位置在画布内，
 * 0 < remaining <= lifetime。remaining == 0 时其他字段不参与更新或绘制。
 */
typedef struct {
    double x, y; /* 像素坐标；原点在左上角，x 向右，y 向下。 */
    double vx, vy; /* 像素/秒；vy < 0 向上，vy > 0 向下。 */
    unsigned int remaining, lifetime; /* 剩余/初始寿命，单位为 1/PARTICLE_HZ 秒。 */
    uint8_t green; /* 初始 RGB 为 (255, green, 60)，三通道按寿命比例渐暗。 */
} Particle;

/**
 * 固定对象池：数组元素始终存在，粒子死亡只把槽位标记为空闲。
 * rng 属于当前系统，使两个系统的随机序列互不干扰。
 */
typedef struct {
    Particle pool[PARTICLE_CAPACITY]; /* 不压缩数组，空槽可在下次发射时复用。 */
    uint32_t rng; /* 每次取随机数都会推进此状态；相同种子和调用顺序可重现。 */
} ParticleSystem;

/**
 * @brief 清空粒子池并设置可重复的随机种子。
 * @param system 有效的可写系统，原有粒子全部丢弃。
 * @param seed 随机种子，包括零在内的全部 uint32_t 值均有效。
 * @note 必须先 reset 再 step/draw；重置不涉及窗口或暂停状态。
 */
void particles_reset(ParticleSystem *system, uint32_t seed);

/**
 * @brief 推进 1/60 秒，先更新/回收旧粒子，再最多发射两个新粒子。
 * @param system 有效且已初始化的系统；存活粒子的字段须保持初始化时的不变量。
 * @note 池满时跳过发射，不覆盖存活粒子；不分配内存。
 * 新粒子出生在本步末尾，保留完整寿命，到下一步才开始运动。
 * 此函数不读取墙钟时间，一次调用始终代表一个固定模拟步。
 */
void particles_step(ParticleSystem *system);

/**
 * @brief 统计存活粒子数。
 * @param system 有效的只读系统。
 * @return 剩余寿命非零的槽位数，最多 PARTICLE_CAPACITY。
 */
size_t particles_count(const ParticleSystem *system);

/**
 * @brief 按剩余寿命渐暗绘制 2×2 粒子，不负责清屏。
 * @param system 有效的只读系统，存活粒子须具有合法坐标及寿命。
 * @param canvas 有效的可写画布，所有权仍归调用者。
 * @note 绘制不修改粒子。调用方应每帧先清屏，否则旧位置会留下痕迹。
 * 右边和下边的 2×2 方块由图形核心裁切；颜色不透明，重叠时后画覆盖先画。
 */
void particles_draw(const ParticleSystem *system, GfxCanvas *canvas);
#endif
