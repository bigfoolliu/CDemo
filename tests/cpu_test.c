/** @file @brief 教学 CPU 的指令语义、原子故障和运行边界验证。 */
#include "../src/cpu/programs.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    size_t attempts;
    size_t outputs;
    uint16_t value;
} Observation;

/**
 * @brief 收集回调事件，验证回调得到执行后的状态。
 * @param cpu 借用只读 CPU。
 * @param step 借用只读指令事件。
 * @param context 有效可写 Observation。
 */
static void observe(const Cpu *cpu, const CpuStep *step, void *context)
{
    Observation *observation = context;
    assert(cpu->status == step->status);
    ++observation->attempts;
    if (step->has_output) {
        ++observation->outputs;
        observation->value = step->output;
    }
}

/**
 * @brief 验证失败只更新状态与故障原因，并且再次执行没有副作用。
 * @param bytes 有效四字节指令。
 * @param fault 预期故障。
 */
static void check_fault(const uint8_t bytes[4], CpuFault fault)
{
    Cpu cpu;
    assert(cpu_load(&cpu, bytes, 4, 0));
    cpu.registers[0] = 1; /* JZ 不跳转时也必须拒绝非法目标。 */
    cpu.memory[500] = 123;
    Cpu before = cpu;
    CpuStep step = cpu_step(&cpu);
    assert(step.status == CPU_FAULTED && !step.has_output);
    assert(cpu.fault == fault && cpu.pc == before.pc);
    assert(memcmp(cpu.registers, before.registers, sizeof(cpu.registers)) == 0);
    assert(memcmp(cpu.memory, before.memory, sizeof(cpu.memory)) == 0);
    assert(cpu_step(&cpu).status == CPU_FAULTED);
    assert(cpu_run(&cpu, 10, nullptr, nullptr).executed == 0);
}

/** @brief 验证全部指令、别名寄存器、无符号回绕与小端读写。 */
static void test_instructions(void)
{
    const uint8_t program[] = {
        CPU_MOVI, 7, 0xff, 0xff, CPU_MOV,   0, 7, 0, CPU_MOVI, 1, 1, 0, CPU_ADD,   0, 1,    0,
        CPU_SUB,  0, 1,    0,    CPU_ADD,   1, 1, 0, CPU_SUB,  1, 1, 0, CPU_STORE, 7, 0xfe, 0xff,
        CPU_LOAD, 2, 0xfe, 0xff, CPU_STORE, 2, 1, 1, CPU_LOAD, 3, 1, 1, CPU_HALT,  0, 0,    0};
    Cpu cpu;
    assert(cpu_load(&cpu, program, sizeof(program), 0));
    for (size_t i = 0; i < 4; ++i) {
        assert(cpu_step(&cpu).status == CPU_RUNNING);
    }
    assert(cpu.registers[0] == 0); /* 65535 + 1 */
    cpu_step(&cpu);
    assert(cpu.registers[0] == UINT16_MAX); /* 0 - 1 */
    cpu_step(&cpu);
    assert(cpu.registers[1] == 2);
    cpu_step(&cpu);
    assert(cpu.registers[1] == 0);
    assert(cpu_run(&cpu, 20, nullptr, nullptr).status == CPU_HALTED);
    assert(cpu.registers[2] == UINT16_MAX && cpu.registers[3] == UINT16_MAX);
    assert(cpu.memory[65534] == 255 && cpu.memory[65535] == 255);
    assert(cpu.memory[257] == 255 && cpu.memory[258] == 255);
    uint32_t stopped_pc = cpu.pc;
    assert(!cpu_step(&cpu).has_output && cpu.pc == stopped_pc);

    const uint8_t endian[] = {CPU_MOVI, 0, 0x34, 0x12, CPU_STORE, 0, 1, 1,
                              CPU_LOAD, 1, 1,    1,    CPU_HALT,  0, 0, 0};
    assert(cpu_load(&cpu, endian, sizeof(endian), 0));
    assert(cpu_run(&cpu, 4, nullptr, nullptr).status == CPU_HALTED);
    assert(cpu.registers[1] == 0x1234);
    assert(cpu.memory[257] == 0x34 && cpu.memory[258] == 0x12);

    const uint8_t branches[] = {CPU_JZ, 0, 8, 0, CPU_HALT, 0, 0, 0, CPU_JMP, 0, 4, 0};
    assert(cpu_load(&cpu, branches, sizeof(branches), 0));
    cpu_step(&cpu);
    assert(cpu.pc == 8);
    cpu_step(&cpu);
    assert(cpu.pc == 4);
    assert(cpu_step(&cpu).status == CPU_HALTED);
    assert(cpu_load(&cpu, branches, sizeof(branches), 0));
    cpu.registers[0] = 1;
    cpu_step(&cpu);
    assert(cpu.pc == 4);

    const uint8_t end[] = {CPU_HALT, 0, 0, 0};
    assert(cpu_load(&cpu, end, sizeof(end), 65532));
    assert(cpu_step(&cpu).status == CPU_HALTED && cpu.pc == 65536);
    assert(cpu_step(&cpu).status == CPU_HALTED && cpu.pc == 65536);
    const uint8_t jump[] = {CPU_JMP, 0, 0xfc, 0xff};
    assert(cpu_load(&cpu, jump, sizeof(jump), 0));
    memcpy(&cpu.memory[65532], end, 4);
    assert(cpu_run(&cpu, 2, nullptr, nullptr).status == CPU_HALTED);
}

/** @brief 验证非法编码、寄存器及内存边界，故障不部分提交。 */
static void test_faults(void)
{
    check_fault((uint8_t[]){0, 0, 0, 0}, CPU_FAULT_OPCODE);
    check_fault((uint8_t[]){255, 0, 0, 0}, CPU_FAULT_OPCODE);
    check_fault((uint8_t[]){CPU_MOVI, 8, 0, 0}, CPU_FAULT_REGISTER);
    check_fault((uint8_t[]){CPU_MOV, 0, 8, 0}, CPU_FAULT_REGISTER);
    check_fault((uint8_t[]){CPU_ADD, 0, 0, 1}, CPU_FAULT_ENCODING);
    check_fault((uint8_t[]){CPU_JMP, 1, 0, 0}, CPU_FAULT_ENCODING);
    check_fault((uint8_t[]){CPU_OUT, 0, 1, 0}, CPU_FAULT_ENCODING);
    check_fault((uint8_t[]){CPU_HALT, 1, 0, 0}, CPU_FAULT_ENCODING);
    check_fault((uint8_t[]){CPU_HALT, 0, 0, 1}, CPU_FAULT_ENCODING);
    check_fault((uint8_t[]){CPU_LOAD, 0, 255, 255}, CPU_FAULT_DATA_ADDRESS);
    check_fault((uint8_t[]){CPU_STORE, 0, 255, 255}, CPU_FAULT_DATA_ADDRESS);
    check_fault((uint8_t[]){CPU_JMP, 0, 2, 0}, CPU_FAULT_JUMP_TARGET);
    check_fault((uint8_t[]){CPU_JZ, 0, 255, 255}, CPU_FAULT_JUMP_TARGET);
    const uint32_t addresses[] = {1, 65533, 65536, UINT32_MAX};
    for (size_t i = 0; i < sizeof(addresses) / sizeof(addresses[0]); ++i) {
        Cpu cpu;
        cpu_init(&cpu);
        cpu.pc = addresses[i];
        Cpu before = cpu;
        cpu_step(&cpu);
        assert(cpu.status == CPU_FAULTED && cpu.pc == before.pc);
        assert(cpu.fault == (i == 0 ? CPU_FAULT_ALIGNMENT : CPU_FAULT_FETCH));
        assert(memcmp(cpu.memory, before.memory, sizeof(cpu.memory)) == 0);
        assert(memcmp(cpu.registers, before.registers, sizeof(cpu.registers)) == 0);
    }
}

/** @brief 验证装载拒绝、重置、预算耗尽及续跑。 */
static void test_lifecycle(void)
{
    const uint8_t loop[] = {CPU_JMP, 0, 0, 0};
    Cpu cpu;
    cpu_init(&cpu);
    cpu.registers[7] = 123;
    cpu.memory[500] = 42;
    Cpu before = cpu;
    assert(!cpu_load(&cpu, nullptr, 4, 0));
    assert(!cpu_load(&cpu, loop, 0, 0));
    assert(!cpu_load(&cpu, loop, 3, 0));
    assert(!cpu_load(&cpu, loop, SIZE_MAX - 3, 0));
    assert(!cpu_load(&cpu, loop, 4, 1));
    assert(!cpu_load(&cpu, loop, 4, 65536));
    assert(!cpu_load(&cpu, loop, 8, 65532));
    assert(cpu.registers[7] == 123 && cpu.status == CPU_RUNNING);
    assert(cpu.pc == before.pc && cpu.fault == before.fault);
    assert(memcmp(cpu.registers, before.registers, sizeof(cpu.registers)) == 0);
    assert(memcmp(cpu.memory, before.memory, sizeof(cpu.memory)) == 0);
    assert(cpu_load(&cpu, loop, 4, 0) && cpu.registers[7] == 0);
    assert(cpu.memory[500] == 0);
    Observation observation = {0};
    CpuRun run = cpu_run(&cpu, 0, observe, &observation);
    assert(run.status == CPU_LIMIT_REACHED && run.executed == 0);
    assert(observation.attempts == 0);
    run = cpu_run(&cpu, 5, observe, &observation);
    assert(run.status == CPU_LIMIT_REACHED && run.executed == 5 && cpu.pc == 0);
    assert(cpu_step(&cpu).status == CPU_RUNNING);
    cpu.memory[0] = 255;
    run = cpu_run(&cpu, 5, observe, &observation);
    assert(run.status == CPU_FAULTED && run.executed == 0 && observation.attempts == 6);

    assert(cpu_load(&cpu, cpu_sum_program, sizeof(cpu_sum_program), 0));
    observation = (Observation){0};
    assert(cpu_run(&cpu, 1, observe, &observation).status == CPU_LIMIT_REACHED);
    run = cpu_run(&cpu, 100, observe, &observation);
    assert(run.status == CPU_HALTED && run.executed == 54);
    assert(observation.outputs == 1 && observation.value == 55 && observation.attempts == 55);
    assert(cpu.registers[0] == 55 && cpu.registers[1] == 0 && cpu.registers[3] == 11);
    assert(cpu_run(&cpu, 100, observe, &observation).executed == 0);
    assert(observation.outputs == 1);
    assert(cpu_load(&cpu, cpu_memory_program, sizeof(cpu_memory_program), 0));
    observation = (Observation){0};
    run = cpu_run(&cpu, 100, observe, &observation);
    assert(run.status == CPU_HALTED && run.executed == 6);
    assert(observation.outputs == 1 && observation.value == 55);
    assert(cpu.memory[257] == 55 && cpu.memory[258] == 0 && cpu.registers[4] == 55);
    assert(strcmp(cpu_opcode_name(255), "INVALID") == 0);
    assert(strcmp(cpu_opcode_name(CPU_LOAD), "LOAD") == 0);
    assert(strlen(cpu_fault_name(CPU_FAULT_FETCH)) > 0);
}

/** @brief 运行无交互、无文件写入的核心断言。 @return 成功返回零。 */
int main(void)
{
    test_instructions();
    test_faults();
    test_lifecycle();
    puts("教学 CPU 测试通过");
    return 0;
}
