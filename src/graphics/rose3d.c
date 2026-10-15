/** @file
 * @brief 采样、漂移、投影并柔和合成三维粉白粒子花云。
 */
#include "rose3d.h"
#include <math.h>
#include <string.h>

enum {
    ROSE3D_BACKGROUND_RED = 5,
    ROSE3D_BACKGROUND_GREEN = 6,
    ROSE3D_BACKGROUND_BLUE = 13
};

typedef struct {
    double x;
    double y;
    double z;
} Point3d;

static uint16_t light_buffer[ROSE3D_VIEW_HEIGHT][ROSE3D_VIEW_WIDTH][3];

/**
 * @brief 推进系统自有的线性同余随机状态。
 * @param system 已初始化的可写粒子系统。
 * @return [0,1) 内的可重复伪随机数。
 */
static double random_unit(Rose3dSystem *system)
{
    system->random_state = system->random_state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(system->random_state >> 8) / 16777216.0;
}

/**
 * @brief 从花瓣主体或向下散开的尘雾中采样一个粒子。
 * @param system 提供随机状态的可写系统。
 * @param particle 有效粒子槽位；成功后完全初始化。
 */
static void respawn_particle(Rose3dSystem *system, Rose3dParticle *particle)
{
    double angle = 2.0 * acos(-1.0) * random_unit(system);
    double selector = random_unit(system);
    double radius;
    double x;
    double y;
    double z;
    if (selector < 0.14) {
        /* 尾部采用逐渐变窄的随机锥，形成参考图中的向下飘散区域。 */
        double down = random_unit(system);
        double spread = 0.10 + 0.47 * down;
        radius = sqrt(random_unit(system)) * spread;
        x = radius * cos(angle) + 0.08 * sin(5.0 * down);
        y = -0.25 - 1.22 * down;
        z = 0.25 * (2.0 * random_unit(system) - 1.0) * spread;
    } else {
        /* 五瓣玫瑰曲线决定轮廓，径向和深度扰动将曲线扩成体积云。 */
        double petal = fabs(cos(2.5 * angle));
        double fill = sqrt(random_unit(system));
        radius = (0.10 + 0.98 * petal) * fill;
        double fold = 0.16 * sin(5.0 * angle) * radius;
        x = 0.92 * radius * cos(angle) + 0.10 * fold;
        y = 0.77 * radius * sin(angle) + 0.06 * fold;
        z = 0.30 * sin(2.0 * angle) * radius +
            0.34 * (2.0 * random_unit(system) - 1.0) * (0.25 + 0.75 * fill);
    }

    double warm = random_unit(system);
    *particle = (Rose3dParticle){
        .x = x,
        .y = y,
        .z = z,
        .phase = 2.0 * acos(-1.0) * random_unit(system),
        .drift_x = 0.0015 * (2.0 * random_unit(system) - 1.0),
        .drift_y = 0.0012 * (2.0 * random_unit(system) - 1.0),
        .drift_z = 0.0015 * (2.0 * random_unit(system) - 1.0),
        .lifetime = 2.0 + 8.0 * random_unit(system),
        .red = (uint8_t)(205.0 + 50.0 * warm),
        .green = (uint8_t)(126.0 + 92.0 * warm),
        .blue = (uint8_t)(148.0 + 90.0 * warm)};
}

/**
 * @brief 将模型空间点绕 Y、X 轴旋转并透视投影到竖屏区域。
 * @param cos_yaw 水平角余弦，由每帧调用方预计算。
 * @param sin_yaw 水平角正弦，由每帧调用方预计算。
 * @param cos_pitch 俯仰角余弦，由每帧调用方预计算。
 * @param sin_pitch 俯仰角正弦，由每帧调用方预计算。
 * @param point 有限的模型坐标。
 * @param screen_x 有效输出地址，成功时写入局部横坐标。
 * @param screen_y 有效输出地址，成功时写入局部纵坐标。
 * @param depth 有效输出地址，成功时写入相机深度。
 * @return 点位于相机前方且投影有限时为 true。
 */
static bool project_point(double cos_yaw, double sin_yaw, double cos_pitch, double sin_pitch,
                          Point3d point, double *screen_x, double *screen_y, double *depth)
{
    double rotated_x = cos_yaw * point.x + sin_yaw * point.z;
    double rotated_z = -sin_yaw * point.x + cos_yaw * point.z;
    double rotated_y = cos_pitch * point.y - sin_pitch * rotated_z;
    rotated_z = sin_pitch * point.y + cos_pitch * rotated_z;
    double camera_depth = 3.9 - rotated_z;
    if (!isfinite(camera_depth) || camera_depth < 1.0) {
        return false;
    }
    double focal_length = 214.0;
    double projected_x = ROSE3D_VIEW_WIDTH * 0.5 + focal_length * rotated_x / camera_depth;
    double projected_y = ROSE3D_VIEW_HEIGHT * 0.43 - focal_length * rotated_y / camera_depth;
    if (!isfinite(projected_x) || !isfinite(projected_y) || fabs(projected_x) > 10000.0 ||
        fabs(projected_y) > 10000.0) {
        return false;
    }
    *screen_x = projected_x;
    *screen_y = projected_y;
    *depth = camera_depth;
    return true;
}

/**
 * @brief 向裁剪后的浮点光照缓冲写入一个有界 RGB 光斑。
 * @param x 光斑中心横坐标。
 * @param y 光斑中心纵坐标。
 * @param color RGB 通道强度。
 * @param strength 当前粒子的亮度权重。
 */
static void add_light(int x, int y, const uint8_t color[3], unsigned int strength)
{
    static const unsigned int kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        int pixel_y = y + offset_y;
        if (pixel_y < 0 || pixel_y >= ROSE3D_VIEW_HEIGHT) {
            continue;
        }
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            int pixel_x = x + offset_x;
            if (pixel_x < 0 || pixel_x >= ROSE3D_VIEW_WIDTH) {
                continue;
            }
            unsigned int weight = kernel[offset_y + 1][offset_x + 1] * strength;
            for (size_t channel = 0; channel < 3; ++channel) {
                unsigned int value = light_buffer[pixel_y][pixel_x][channel] +
                                     ((unsigned int)color[channel] * weight) / 4;
                light_buffer[pixel_y][pixel_x][channel] =
                    (uint16_t)(value > UINT16_MAX ? UINT16_MAX : value);
            }
        }
    }
}

/** @copydoc rose3d_reset */
void rose3d_reset(Rose3dSystem *system, uint32_t seed)
{
    *system = (Rose3dSystem){.yaw = -0.20, .pitch = 0.10, .random_state = seed};
    for (size_t i = 0; i < ROSE3D_PARTICLE_COUNT; ++i) {
        respawn_particle(system, &system->particles[i]);
    }
}

/** @copydoc rose3d_step */
void rose3d_step(Rose3dSystem *system)
{
    const double delta_time = 1.0 / ROSE3D_HZ;
    system->time += delta_time;
    system->yaw += 0.0018;
    for (size_t i = 0; i < ROSE3D_PARTICLE_COUNT; ++i) {
        Rose3dParticle *particle = &system->particles[i];
        particle->lifetime -= delta_time;
        if (particle->lifetime <= 0.0) {
            respawn_particle(system, particle);
            continue;
        }
        particle->x += particle->drift_x + 0.0008 * sin(system->time + particle->phase);
        particle->y += particle->drift_y + 0.0006 * cos(system->time * 0.7 + particle->phase);
        particle->z += particle->drift_z;
    }
    if (system->time >= 100000.0) {
        system->time = fmod(system->time, 100000.0);
    }
}

/** @copydoc rose3d_rotate */
void rose3d_rotate(Rose3dSystem *system, double yaw_delta, double pitch_delta)
{
    if (!isfinite(yaw_delta) || !isfinite(pitch_delta)) {
        return;
    }
    system->yaw = remainder(system->yaw + yaw_delta, 2.0 * acos(-1.0));
    system->pitch = fmin(1.2, fmax(-1.2, system->pitch + pitch_delta));
}

/** @copydoc rose3d_draw */
void rose3d_draw(const Rose3dSystem *system, GfxCanvas *canvas)
{
    gfx_clear(canvas, gfx_rgb(ROSE3D_BACKGROUND_RED, ROSE3D_BACKGROUND_GREEN,
                              ROSE3D_BACKGROUND_BLUE));
    memset(light_buffer, 0, sizeof light_buffer);

    /* 中心轴和细微刻度提供空间参照，颜色刻意压暗以免抢过粒子云。 */
    for (int y = 12; y < ROSE3D_VIEW_HEIGHT - 10; y += 1) {
        unsigned int alpha = (unsigned int)(5 + 4 * (1.0 + sin((double)y * 0.08)));
        add_light(ROSE3D_VIEW_WIDTH / 2, y, (const uint8_t[3]){104, 92, 110}, alpha / 4);
    }

    double cos_yaw = cos(system->yaw);
    double sin_yaw = sin(system->yaw);
    double cos_pitch = cos(system->pitch);
    double sin_pitch = sin(system->pitch);
    double cosine = cos(system->time * 2.0);
    for (size_t i = 0; i < ROSE3D_PARTICLE_COUNT; ++i) {
        const Rose3dParticle *particle = &system->particles[i];
        double screen_x;
        double screen_y;
        double depth;
        Point3d point = {particle->x, particle->y, particle->z};
        if (!project_point(cos_yaw, sin_yaw, cos_pitch, sin_pitch, point, &screen_x, &screen_y,
                           &depth)) {
            continue;
        }
        if (screen_x < -2.0 || screen_x >= ROSE3D_VIEW_WIDTH + 2.0 || screen_y < -2.0 ||
            screen_y >= ROSE3D_VIEW_HEIGHT + 2.0) {
            continue;
        }
        double twinkle = 0.72 + 0.28 * sin(system->time * 2.8 + particle->phase);
        double depth_glow = fmin(1.25, 3.1 / depth);
        uint8_t color[3] = {particle->red, particle->green, particle->blue};
        unsigned int strength = (unsigned int)(twinkle * depth_glow * (1.0 + 0.3 * cosine));
        if (strength == 0) {
            strength = 1;
        }
        add_light((int)screen_x, (int)screen_y, color, strength);
    }

    /* 压缩高亮并叠加到底色，避免粒子重叠处出现生硬的纯白色块。 */
    for (int y = 0; y < ROSE3D_VIEW_HEIGHT; ++y) {
        for (int x = 0; x < ROSE3D_VIEW_WIDTH; ++x) {
            uint16_t *light = light_buffer[y][x];
            if (light[0] == 0 && light[1] == 0 && light[2] == 0) {
                continue;
            }
            unsigned int red = ROSE3D_BACKGROUND_RED + light[0] / (12 + light[0] / 150);
            unsigned int green = ROSE3D_BACKGROUND_GREEN + light[1] / (12 + light[1] / 150);
            unsigned int blue = ROSE3D_BACKGROUND_BLUE + light[2] / (12 + light[2] / 150);
            if (red > 255) {
                red = 255;
            }
            if (green > 255) {
                green = 255;
            }
            if (blue > 255) {
                blue = 255;
            }
            canvas->pixels[y + ROSE3D_VIEW_Y][x + ROSE3D_VIEW_X] =
                gfx_rgb((uint8_t)red, (uint8_t)green, (uint8_t)blue);
        }
    }
}
