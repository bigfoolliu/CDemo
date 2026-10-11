/** @file @brief 演示程序共用的截图路径管理；目录创建使用 POSIX API。 */
#ifndef CDEMO_BMP_CAPTURE_H
#define CDEMO_BMP_CAPTURE_H
#include "graphics.h"

enum {
    BMP_CAPTURE_PATH_SIZE = 256
};
/**
 * @brief 创建 exports 目录，并以时间与序号新建一张完整画布图片。
 * @param canvas 有效的只读画布，导出原始 320×240 像素，不含标题栏。
 * @param name 文件名前缀，只接受非空 ASCII 字母、数字、下划线和连字符。
 * @param path 有效的可写数组，容量至少 BMP_CAPTURE_PATH_SIZE，成功后为保存路径。
 * @return 成功为 true；目录/命名/写入失败为 false，errno 提供原因。
 * @note 从仓库根目录运行；exports 是图片目录，不是日志目录。
 */
bool gfx_bmp_capture(const GfxCanvas *canvas, const char *name, char path[BMP_CAPTURE_PATH_SIZE]);
#endif
