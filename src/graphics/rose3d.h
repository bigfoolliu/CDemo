/** @file
 * @brief 三维粒子玫瑰的固定粒子池、旋转状态与投影绘制接口。
 */
#ifndef CDEMO_ROSE3D_H
#define CDEMO_ROSE3D_H
#include "graphics.h"
#include <stddef.h>
#include <stdint.h>

enum {
    ROSE3D_PARTICLE_COUNT = 16000,
    ROSE3D_HZ = 60,
    ROSE3D_VIEW_X = 75,
    ROSE3D_VIEW_Y = 0,
    ROSE3D_VIEW_WIDTH = 170,
    ROSE3D_VIEW_HEIGHT = 240
};

/**
 * 花瓣中的固定粒子，位置属于模型空间。
 * 调用者不得直接修改存活粒子字段。
 */
typedef struct {
    double x;
    double y;
    double z;
    double phase;
    double drift_x;
    double drift_y;
    double drift_z;
    double lifetime;
    uint8_t red;
    uint8_t green;
    uint8_t blue;
} Rose3dParticle;

/**
 * 固定容量的玫瑰点云和动画状态；不动态分配，
 * 也不取得外部资源所有权。
 */
typedef struct {
    Rose3dParticle particles[ROSE3D_PARTICLE_COUNT];
    double time;
    double yaw;
    double pitch;
    uint32_t random_state;
} Rose3dSystem;

/**
 * @brief 用确定性随机种子生成粉白体积粒子云并复位相机。
 * @param system 有效的可写系统；成功后整体替换，不持有外部资源。
 * @param seed 任意 uint32_t 种子；相同种子产生相同点云。
 */
void rose3d_reset(Rose3dSystem *system, uint32_t seed);

/**
 * @brief 按固定 1/60 秒推进相机旋转、粒子漂移和生命周期。
 * @param system 有效且已由 rose3d_reset 初始化的可写系统。
 */
void rose3d_step(Rose3dSystem *system);

/**
 * @brief 调整观察角度，限制俯仰范围以避免视图翻转。
 * @param system 有效且已初始化的可写系统。
 * @param yaw_delta 水平旋转弧度，可为任意有限值。
 * @param pitch_delta 俯仰变化弧度，可为任意有限值。
 */
void rose3d_rotate(Rose3dSystem *system, double yaw_delta, double pitch_delta);

/**
 * @brief 清屏并将三维粒子云投影为柔和发光的竖屏玫瑰效果。
 * @param system 有效的只读玫瑰状态。
 * @param canvas 有效的可写画布，仍由调用者拥有。
 * @note 采用 CPU 透视投影和加色光晕；绘制不修改系统，不使用 SDL 图元。
 */
void rose3d_draw(const Rose3dSystem *system, GfxCanvas *canvas);

#endif
