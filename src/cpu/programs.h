/** @file @brief 两个内置教学机器码程序；不是 RISC-V 或 x86 编码。 */
#ifndef CDEMO_CPU_PROGRAMS_H
#define CDEMO_CPU_PROGRAMS_H
#include "cpu.h"

/* r0 为累加器，r1 为剩余次数，r2 为常量 1，r3 为当前加数。 */
static const uint8_t cpu_sum_program[] = {
    CPU_MOVI, 0, 0,  0, /* 00: r0 = 0 */
    CPU_MOVI, 1, 10, 0, /* 04: r1 = 10 */
    CPU_MOVI, 2, 1,  0, /* 08: r2 = 1 */
    CPU_MOVI, 3, 1,  0, /* 12: r3 = 1 */
    CPU_ADD,  0, 3,  0, /* 16: r0 += r3 */
    CPU_ADD,  3, 2,  0, /* 20: r3 += 1 */
    CPU_SUB,  1, 2,  0, /* 24: r1 -= 1 */
    CPU_JZ,   1, 36, 0, /* 28: r1 为零时到 OUT */
    CPU_JMP,  0, 16, 0, /* 32: 回到循环 */
    CPU_OUT,  0, 0,  0, /* 36: 输出 55 */
    CPU_HALT, 0, 0,  0  /* 40: 停止 */
};

/* 地址 0x0101 未按两字节对齐，用逐字节小端访问仍可正确读写。 */
static const uint8_t cpu_memory_program[] = {CPU_MOVI,  0, 55, 0, /* r0 = 55 */
                                             CPU_STORE, 0, 1,  1, /* memory[0x0101..0x0102] = r0 */
                                             CPU_MOVI,  0, 0,  0, /* 清零寄存器，保证结果来自内存 */
                                             CPU_LOAD,  4, 1,  1, /* r4 = memory[0x0101..0x0102] */
                                             CPU_OUT,   4, 0,  0, CPU_HALT, 0, 0, 0};
#endif
