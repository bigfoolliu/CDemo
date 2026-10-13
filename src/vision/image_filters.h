/**
 * @file
 * @brief 灰度图像与3×3空间滤波；不依赖窗口或第三方图像库。
 */
#ifndef CDEMO_IMAGE_FILTERS_H
#define CDEMO_IMAGE_FILTERS_H

#include "../graphics/graphics.h"
#include <stddef.h>

/** 有效区域为[0,width)×[0,height)，像素为有限double，可保存带符号响应。 */
typedef struct {
    size_t width;
    size_t height;
    double pixels[GFX_HEIGHT][GFX_WIDTH];
} VisionImage;

typedef enum {
    VISION_BORDER_CLAMP, /* 超出图像的坐标取最近边缘像素。 */
    VISION_BORDER_ZERO   /* 超出图像的像素取零。 */
} VisionBorder;

typedef enum {
    VISION_BOX_BLUR,
    VISION_GAUSSIAN_BLUR,
    VISION_SHARPEN,
    VISION_EMBOSS,
    VISION_SOBEL,
    VISION_FILTER_COUNT
} VisionFilter;

/**
 * @brief RGB画布转灰度，使用(299R+587G+114B)/1000，忽略alpha。
 * @param canvas 有效只读320×240画布，不能与output存储重叠。
 * @param output 有效可写图像，结果范围0..255，覆盖尺寸和有效像素。
 * @return 成功为true；空指针时为false且不修改output。
 */
bool vision_from_canvas(const GfxCanvas *canvas, VisionImage *output);

/**
 * @brief 3×3互相关：核不翻转，保留带符号浮点响应。
 * @param source 有效只读图像，尺寸1..画布上限，有效像素须为有限数。
 * @param kernel 九个有限系数，按行排列，调用期间不可修改。
 * @param bias 有限响应偏移；滤波后再加，不自动归一化或限幅。
 * @param border 合法边界模式。
 * @param output 有效可写结果，可与source为同一图像；无所有权转移。
 * @return 成功为true；非法输入、分配失败或运算非有限时为false，output不变。
 */
bool vision_correlate3(const VisionImage *source, const double kernel[9], double bias,
                       VisionBorder border, VisionImage *output);

/**
 * @brief 应用预置滤镜；Sobel返回hypot(Gx,Gy)，不提前裁掉负导数。
 * @param source 同vision_correlate3的只读输入契约。
 * @param filter 合法滤镜枚举；浮雕加128，其他滤镜不加偏移。
 * @param border 合法边界模式。
 * @param output 可写结果，允许原地处理；像素可超出0..255。
 * @return 成功为true；失败保留output，临时缓冲在函数内释放。
 */
bool vision_filter(const VisionImage *source, VisionFilter filter, VisionBorder border,
                   VisionImage *output);

/**
 * @brief 灰度响应显示为RGB画布，限幅到0..255再四舍五入，不做自动对比度拉伸。
 * @param image 有效只读有限图像，不可与canvas存储重叠。
 * @param canvas 有效可写画布，图像从左上角开始，未覆盖区域填黑。
 * @return 成功为true；输入非法时为false且画布不变。
 */
bool vision_to_canvas(const VisionImage *image, GfxCanvas *canvas);

#endif
