/** @file @brief 演示内置机器码和逐指令追踪，日志仅写入 logs/cpu/。 */
#include "programs.h"
#include "../common/logger.h"
#include <inttypes.h>

typedef struct {
    Logger *logger;
    bool output_failed;
    size_t attempts;
} Display;

/**
 * @brief 显示指令执行后的状态，核心的 OUT 值由此交给终端。
 * @param cpu 有效只读执行后 CPU。
 * @param step 有效只读指令尝试，包含执行前 PC。
 * @param context 有效可写 Display，不转移所有权；累计输出错误。
 */
static void display_step(const Cpu *cpu, const CpuStep *step, void *context)
{
    Display *display = context;
    ++display->attempts;
    if (display->logger != nullptr) {
        logger_write(display->logger,
                     "step=%zu PC=%04" PRIx32 "→%04" PRIx32 " bytes=%02x %02x %02x %02x %s "
                     "r=[%u,%u,%u,%u,%u,%u,%u,%u] status=%d",
                     display->attempts, step->pc_before, cpu->pc, (unsigned)step->bytes[0],
                     (unsigned)step->bytes[1], (unsigned)step->bytes[2], (unsigned)step->bytes[3],
                     cpu_opcode_name(step->bytes[0]), (unsigned)cpu->registers[0],
                     (unsigned)cpu->registers[1], (unsigned)cpu->registers[2],
                     (unsigned)cpu->registers[3], (unsigned)cpu->registers[4],
                     (unsigned)cpu->registers[5], (unsigned)cpu->registers[6],
                     (unsigned)cpu->registers[7], (int)cpu->status);
    }
    if (step->has_output) {
        if (display->logger != nullptr) {
            logger_write(display->logger, "OUT=%u", (unsigned)step->output);
        } else if (printf("OUT=%u\n", (unsigned)step->output) < 0) {
            display->output_failed = true;
        }
    }
}

/**
 * @brief 装载并有界运行一个内置程序，明确报告故障、步数上限和输出失败。
 * @param name 有效只读案例名称。
 * @param program 至少 size 字节的有效只读机器码。
 * @param size 程序字节数。
 * @param logger 已打开日志记录器或空指针，借用而不关闭。
 * @return 正常停止且输出成功为 true。
 */
static bool run_example(const char *name, const uint8_t *program, size_t size, Logger *logger)
{
    Cpu cpu;
    cpu_init(&cpu);
    if (!cpu_load(&cpu, program, size, 0)) {
        fprintf(stderr, "%s：程序装载失败\n", name);
        return false;
    }
    if (logger != nullptr) {
        logger_write(logger, "=== %s ===", name);
    } else if (printf("=== %s ===\n", name) < 0) {
        return false;
    }
    Display display = {.logger = logger};
    CpuRun run = cpu_run(&cpu, 10000, display_step, &display);
    if (run.status != CPU_HALTED) {
        fprintf(stderr, "%s：运行失败，PC=%" PRIu32 "，%s\n", name, cpu.pc,
                run.status == CPU_FAULTED ? cpu_fault_name(cpu.fault) : "达到步数上限");
        return false;
    }
    if (logger != nullptr) {
        logger_write(logger, "正常停止：%zu 条指令", run.executed);
        return !logger->failed;
    }
    return !display.output_failed && printf("正常停止：%zu 条指令\n", run.executed) >= 0;
}

/**
 * @brief 运行两个 CPU 案例；--trace 同时追加终端与 logs/cpu/trace.log。
 * @param argc 参数数量。
 * @param argv 可选 --trace；其他参数报错。
 * @return 正常完成为 EXIT_SUCCESS；装载、运行或输出失败为 EXIT_FAILURE。
 */
int main(int argc, char **argv)
{
    bool trace = argc == 2 && strcmp(argv[1], "--trace") == 0;
    if (argc != 1 && !trace) {
        fprintf(stderr, "用法：%s [--trace]\n", argv[0]);
        return EXIT_FAILURE;
    }
    Logger logger = {0};
    if (trace && !logger_open(&logger, "logs/cpu/trace.log", stdout)) {
        fprintf(stderr, "CPU 日志打开失败：%s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    Logger *active = trace ? &logger : nullptr;
    bool success =
        run_example("累加 1..10", cpu_sum_program, sizeof(cpu_sum_program), active) &&
        run_example("内存写入与读回", cpu_memory_program, sizeof(cpu_memory_program), active);
    if (trace && !logger_close(&logger)) {
        fputs("CPU 日志写入或关闭失败\n", stderr);
        success = false;
    }
    if (fflush(stdout) != 0) {
        success = false;
    }
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
