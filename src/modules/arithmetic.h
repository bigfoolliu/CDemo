/** @file
 * @brief 整数加法模块的公共接口。
 */
#ifndef CDEMO_ARITHMETIC_H
#define CDEMO_ARITHMETIC_H

/* 头文件声明公共接口；保护宏避免同一翻译单元重复包含。 */
/**
 * @brief 计算两个整数的和。
 * @param left 左操作数。
 * @param right 右操作数；两数之和须在 int 范围内。
 * @return 两个操作数的和。
 */
int arithmetic_add(int left, int right);

#endif
