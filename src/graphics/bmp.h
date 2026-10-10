/** @file @brief 纯 C 的 24 位 BMP 编码；不使用 SDL 或平台位图结构体。 */
#ifndef CDEMO_BMP_H
#define CDEMO_BMP_H
#include "graphics.h"
#include <stdio.h>

/**
 * @brief 将画布左上角区域编码为未压缩 24 位 BMP，并刷新输出流。
 * @param stream 有效的可写二进制流，从当前位置写入；调用者负责关闭。
 * @param canvas 有效的只读画布；取 RGB 通道，忽略最高字节的 alpha。
 * @param width 区域宽度，1..GFX_WIDTH。
 * @param height 区域高度，1..GFX_HEIGHT。
 * @return 成功为 true；参数/写入/刷新失败为 false，errno 提供原因。
 * @note 失败可能留下部分数据，不回滚调用者的流。当前流位置应为零才能形成完整文件。
 */
bool gfx_bmp_write(FILE *stream, const GfxCanvas *canvas, int width, int height);

/**
 * @brief 新建 BMP 文件并保存完整画布，禁止覆盖已有文件。
 * @param canvas 有效的只读画布。
 * @param path 非空文件路径，父目录须已存在。
 * @return 成功为 true；失败为 false 并设置 errno，包含关闭时的写入错误。
 * @note 使用 C11 起支持的 fopen("wbx")；失败时尝试删除本次创建的不完整文件。
 * 已有文件不会删除；若清理也失败，不完整文件可能保留。
 */
bool gfx_bmp_save(const GfxCanvas *canvas, const char *path);
#endif
