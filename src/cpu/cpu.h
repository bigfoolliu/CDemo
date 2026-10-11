/** @file @brief 教学 CPU：16 位寄存器、64 KiB 内存和固定四字节指令。 */
#ifndef CDEMO_CPU_H
#define CDEMO_CPU_H

#include <stddef.h>
#include <stdint.h>

enum {
    CPU_REGISTERS = 8,
    CPU_MEMORY_SIZE = 65536,
    CPU_INSTRUCTION_SIZE = 4
};

typedef enum {
    CPU_MOVI = 1,
    CPU_MOV,
    CPU_ADD,
    CPU_SUB,
    CPU_LOAD,
    CPU_STORE,
    CPU_JMP,
    CPU_JZ,
    CPU_OUT,
    CPU_HALT
} CpuOpcode;

typedef enum {
    CPU_RUNNING,
    CPU_HALTED,
    CPU_FAULTED,
    CPU_LIMIT_REACHED /* 可续跑，不是客体异常。 */
} CpuStatus;

typedef enum {
    CPU_FAULT_NONE,
    CPU_FAULT_FETCH,
    CPU_FAULT_ALIGNMENT,
    CPU_FAULT_OPCODE,
    CPU_FAULT_REGISTER,
    CPU_FAULT_ENCODING,
    CPU_FAULT_DATA_ADDRESS,
    CPU_FAULT_JUMP_TARGET
} CpuFault;

typedef struct {
    uint16_t registers[CPU_REGISTERS];
    uint32_t pc; /* 可表示内存末尾之后的 65536，不让顺序取指地址回绕。 */
    uint8_t memory[CPU_MEMORY_SIZE];
    CpuStatus status;
    CpuFault fault;
} Cpu;

typedef struct {
    uint32_t pc_before;
    uint8_t bytes[CPU_INSTRUCTION_SIZE];
    CpuStatus status;
    bool has_output;
    uint16_t output;
} CpuStep;

typedef struct {
    CpuStatus status;
    size_t executed; /* 成功执行的指令数，含 HALT，不含故障指令。 */
} CpuRun;

/** 观察一次指令尝试；借用只读状态，不可重入或修改 CPU，输出由调用方处理。 */
typedef void (*CpuObserver)(const Cpu *cpu, const CpuStep *step, void *context);

/**
 * @brief 清零寄存器、内存、PC 和故障，恢复运行状态。
 * @param cpu 有效可写 CPU，不转移所有权。
 */
void cpu_init(Cpu *cpu);

/**
 * @brief 校验程序范围后重置 CPU 并装载字节，PC 指向起始地址。
 * @param cpu 有效可写 CPU。
 * @param program 非空只读字节数组，至少 size 字节，不与 cpu 存储重叠。
 * @param size 非零、四字节倍数的程序大小。
 * @param address 四字节对齐的装载地址，完整程序必须位于模拟内存内。
 * @return 成功为 true；参数错误为 false，CPU 完全不变。
 */
bool cpu_load(Cpu *cpu, const uint8_t *program, size_t size, uint32_t address);

/**
 * @brief 取指、解码、校验后执行一步；步数上限状态允许续跑。
 * @param cpu 有效可写 CPU，停止或故障状态不会再次取指。
 * @return 指令尝试和可选 OUT 值；故障不改 PC、寄存器及内存，仅记录故障状态。
 */
CpuStep cpu_step(Cpu *cpu);

/**
 * @brief 有界执行，达到上限时保留状态以便继续执行。
 * @param cpu 有效可写 CPU。
 * @param max_steps 最多尝试次数，零表示不执行；停止和故障状态保持不变。
 * @param observer 每次尝试后的观察回调，可为空；核心不打印或写文件。
 * @param context 传给回调的借用上下文，可为空。
 * @return 最终状态及成功执行的指令数。
 */
CpuRun cpu_run(Cpu *cpu, size_t max_steps, CpuObserver observer, void *context);

/**
 * @brief 获取操作码助记符。
 * @param opcode 原始操作码。
 * @return 借用静态字符串，非法值为 INVALID，不得释放。
 */
const char *cpu_opcode_name(uint8_t opcode);

/**
 * @brief 获取中文故障原因。
 * @param fault 故障码。
 * @return 借用静态字符串，不得释放。
 */
const char *cpu_fault_name(CpuFault fault);

#endif
