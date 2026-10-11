/** @file @brief 指令解释器：先完成全部校验，再修改客体状态。 */
#include "cpu.h"
#include <limits.h>
#include <string.h>

static_assert(CHAR_BIT == 8, "教学机器码要求八位字节");

void cpu_init(Cpu *cpu)
{
    *cpu = (Cpu){0};
}

bool cpu_load(Cpu *cpu, const uint8_t *program, size_t size, uint32_t address)
{
    if (program == nullptr || size == 0 || size % CPU_INSTRUCTION_SIZE != 0 ||
        address % CPU_INSTRUCTION_SIZE != 0 || address >= CPU_MEMORY_SIZE ||
        size > CPU_MEMORY_SIZE - address) {
        return false;
    }
    cpu_init(cpu);
    memcpy(&cpu->memory[address], program, size);
    cpu->pc = address;
    return true;
}

/**
 * @brief 校验已取出的编码，非法指令没有可见副作用。
 * @param bytes 四个只读指令字节。
 * @param operand 最后两字节组成的小端立即数或地址。
 * @return 无错误为 CPU_FAULT_NONE；JZ 无论是否跳转都校验目标。
 */
static CpuFault validate(const uint8_t bytes[CPU_INSTRUCTION_SIZE], uint16_t operand)
{
    uint8_t opcode = bytes[0];
    uint8_t a = bytes[1];
    if (opcode < CPU_MOVI || opcode > CPU_HALT) {
        return CPU_FAULT_OPCODE;
    }
    if (opcode == CPU_JMP || opcode == CPU_HALT) {
        if (a != 0) {
            return CPU_FAULT_ENCODING;
        }
    } else if (a >= CPU_REGISTERS) {
        return CPU_FAULT_REGISTER;
    }
    if (opcode == CPU_MOV || opcode == CPU_ADD || opcode == CPU_SUB) {
        if (bytes[2] >= CPU_REGISTERS) {
            return CPU_FAULT_REGISTER;
        }
        if (bytes[3] != 0) {
            return CPU_FAULT_ENCODING;
        }
    }
    if ((opcode == CPU_OUT || opcode == CPU_HALT) && operand != 0) {
        return CPU_FAULT_ENCODING;
    }
    if ((opcode == CPU_LOAD || opcode == CPU_STORE) && operand > CPU_MEMORY_SIZE - 2) {
        return CPU_FAULT_DATA_ADDRESS;
    }
    if ((opcode == CPU_JMP || opcode == CPU_JZ) &&
        (operand % CPU_INSTRUCTION_SIZE != 0 || operand > CPU_MEMORY_SIZE - CPU_INSTRUCTION_SIZE)) {
        return CPU_FAULT_JUMP_TARGET;
    }
    return CPU_FAULT_NONE;
}

CpuStep cpu_step(Cpu *cpu)
{
    CpuStep step = {.pc_before = cpu->pc, .status = cpu->status};
    if (cpu->status == CPU_HALTED || cpu->status == CPU_FAULTED) {
        return step;
    }
    CpuFault fault = CPU_FAULT_NONE;
    if (cpu->pc > CPU_MEMORY_SIZE - CPU_INSTRUCTION_SIZE) {
        fault = CPU_FAULT_FETCH;
    } else if (cpu->pc % CPU_INSTRUCTION_SIZE != 0) {
        fault = CPU_FAULT_ALIGNMENT;
    }
    uint16_t operand = 0;
    if (fault == CPU_FAULT_NONE) {
        memcpy(step.bytes, &cpu->memory[cpu->pc], sizeof(step.bytes));
        operand = (uint16_t)((uint16_t)step.bytes[2] | ((uint16_t)step.bytes[3] << 8));
        fault = validate(step.bytes, operand);
    }
    if (fault != CPU_FAULT_NONE) {
        cpu->fault = fault;
        cpu->status = CPU_FAULTED;
        step.status = cpu->status;
        return step;
    }

    uint8_t a = step.bytes[1];
    uint8_t b = step.bytes[2];
    uint32_t next_pc = cpu->pc + CPU_INSTRUCTION_SIZE;
    cpu->status = CPU_RUNNING;
    switch (step.bytes[0]) {
    case CPU_MOVI:
        cpu->registers[a] = operand;
        break;
    case CPU_MOV:
        cpu->registers[a] = cpu->registers[b];
        break;
    case CPU_ADD:
        cpu->registers[a] = (uint16_t)((uint32_t)cpu->registers[a] + cpu->registers[b]);
        break;
    case CPU_SUB:
        cpu->registers[a] = (uint16_t)((uint32_t)cpu->registers[a] - (uint32_t)cpu->registers[b]);
        break;
    case CPU_LOAD:
        cpu->registers[a] =
            (uint16_t)((uint16_t)cpu->memory[operand] | ((uint16_t)cpu->memory[operand + 1] << 8));
        break;
    case CPU_STORE:
        cpu->memory[operand] = (uint8_t)(cpu->registers[a] & 0xff);
        cpu->memory[operand + 1] = (uint8_t)(cpu->registers[a] >> 8);
        break;
    case CPU_JMP:
        next_pc = operand;
        break;
    case CPU_JZ:
        if (cpu->registers[a] == 0) {
            next_pc = operand;
        }
        break;
    case CPU_OUT:
        step.has_output = true;
        step.output = cpu->registers[a];
        break;
    case CPU_HALT:
        cpu->status = CPU_HALTED;
        break;
    }
    /* HALT 也算成功执行，PC 指向它之后，可能等于内存大小。 */
    cpu->pc = next_pc;
    step.status = cpu->status;
    return step;
}

CpuRun cpu_run(Cpu *cpu, size_t max_steps, CpuObserver observer, void *context)
{
    CpuRun result = {.status = cpu->status};
    if (cpu->status == CPU_HALTED || cpu->status == CPU_FAULTED) {
        return result;
    }
    for (size_t attempt = 0; attempt < max_steps; ++attempt) {
        CpuStep step = cpu_step(cpu);
        if (step.status != CPU_FAULTED) {
            ++result.executed;
        }
        if (observer != nullptr) {
            observer(cpu, &step, context);
        }
        if (cpu->status == CPU_HALTED || cpu->status == CPU_FAULTED) {
            result.status = cpu->status;
            return result;
        }
    }
    cpu->status = CPU_LIMIT_REACHED;
    result.status = cpu->status;
    return result;
}

const char *cpu_opcode_name(uint8_t opcode)
{
    static const char *names[] = {"INVALID", "MOVI", "MOV", "ADD", "SUB", "LOAD",
                                  "STORE",   "JMP",  "JZ",  "OUT", "HALT"};
    if (opcode > CPU_HALT) {
        return names[0];
    }
    return names[opcode];
}

const char *cpu_fault_name(CpuFault fault)
{
    switch (fault) {
    case CPU_FAULT_NONE:
        return "无故障";
    case CPU_FAULT_FETCH:
        return "取指超出内存范围";
    case CPU_FAULT_ALIGNMENT:
        return "取指地址未按四字节对齐";
    case CPU_FAULT_OPCODE:
        return "非法操作码";
    case CPU_FAULT_REGISTER:
        return "非法寄存器编号";
    case CPU_FAULT_ENCODING:
        return "保留字段非零";
    case CPU_FAULT_DATA_ADDRESS:
        return "数据访问超出内存范围";
    case CPU_FAULT_JUMP_TARGET:
        return "跳转目标非法或未对齐";
    }
    return "未知故障";
}
