/** @file
 * @brief SDL 粒子喷泉：事件输入、固定步长模拟与像素显示。
 *
 * 每帧依次执行：读取时间 → 处理事件 → 若干次固定更新 → 清屏/绘制 → 显示/限速。
 * particles.c 负责模拟，graphics.c 负责像素，graphics_sdl.c 负责窗口资源。
 * 阅读时可先看 main 的整体流程，再进入 handle_event 和 advance_time。
 */
#include "graphics_sdl.h"
#include "particles.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** 暂停不积累时间；重置只清空粒子和随机状态，保留暂停状态。 */
typedef struct {
    ParticleSystem particles; /* 本演示自己拥有的粒子池。 */
    bool running, paused; /* running 控制退出；paused 只控制自动模拟。 */
    double accumulator; /* 尚未用于模拟的秒数，正常推进后小于 1/60 秒。 */
} ParticleDemoState;

/**
 * @brief 执行一个输入事件，忽略自动重复按键。
 * @param state 有效的可读写演示状态。
 * @param event 有效的只读 SDL 事件。
 * @return 暂停切换或重置时返回 true，要求调用方丢弃当前帧时间。
 */
static bool handle_event(ParticleDemoState *state, const SDL_Event *event)
{
    /* 关闭按钮和 Esc 都只设置退出标记，资源由 main 的统一出口释放。 */
    if (event->type == SDL_QUIT) {
        state->running = false;
    } else if (event->type == SDL_KEYDOWN && event->key.repeat == 0) {
        /* SDL 在按住键时会生成重复事件；忽略它们，避免连续切换暂停或单步。 */
        switch (event->key.keysym.sym) {
        case SDLK_ESCAPE: state->running = false; break;
        case SDLK_SPACE:
            state->paused = !state->paused;
            /* 清掉前一阶段不足一步的余量；返回 true 还会丢弃当前帧 elapsed，
             * 使暂停前或恢复前的时间不会在切换后被自动模拟使用。
             */
            state->accumulator = 0;
            return true;
        case SDLK_n:
            /* 单步直接调用核心，不依赖累积时间；运行状态下按 N 不额外加速。 */
            if (state->paused) { particles_step(&state->particles); }
            break;
        case SDLK_r:
            /* 固定种子重现相同发射序列；不改 paused，方便暂停后从头观察。 */
            particles_reset(&state->particles, 42);
            state->accumulator = 0;
            return true;
        default: break;
        }
    }
    return false;
}

/**
 * @brief 将墙钟时间转换为若干次 1/60 秒更新，保留不足一步的余量。
 * @param state 有效的可读写演示状态。
 * @param elapsed 秒数；负数或非有限值忽略，超过 0.1 秒按 0.1 处理。
 */
static void advance_time(ParticleDemoState *state, double elapsed)
{
    if (state->paused) {
        /* 暂停仍会显示和处理输入，但不积累需要在恢复后“补跑”的模拟时间。 */
        state->accumulator = 0;
        return;
    }
    /* 非有限数会污染累加器，负数也不是有效时长；在入口直接忽略。 */
    if (!isfinite(elapsed) || elapsed < 0) { return; }
    /* 最多接收 0.1 秒，避免断点/系统挂起后追赶大量更新。
     * 因此长停顿的额外时间被丢弃，模拟不保证严格追随真实时间。
     */
    state->accumulator += fmin(elapsed, 0.1);
    const double tick = 1.0 / PARTICLE_HZ;
    while (state->accumulator >= tick) {
        /* 显示帧耗时与模拟步长不必相同：慢帧可能推进多步，快帧可能零步。
         * 减去已使用的时间，保留不足一步的余量，避免每帧丢掉这部分时间。
         */
        particles_step(&state->particles);
        state->accumulator -= tick;
    }
}

/**
 * @brief 创建窗口并运行粒子演示，所有 SDL 资源在退出时释放。
 * @param argc 参数数量，包含程序名。
 * @param argv 参数数组；支持 --frames 1..10000 自动退出。
 * @return 成功为 EXIT_SUCCESS，参数或 SDL 失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    unsigned long limit = 0; /* 0 表示不限制显示帧数，直到用户退出。 */
    if (argc != 1) {
        char *end = nullptr; /* C23 的空指针常量；strtoul 写入首个未解析字符的地址。 */
        /* 先验证参数数量再访问 argv[1]/argv[2]，利用 || 的短路求值保证安全。
         * 首字符必须为数字，拒绝 strtoul 本来可以接受的空白或正负号。
         */
        if (argc != 3 || strcmp(argv[1], "--frames") != 0 ||
            argv[2][0] < '0' || argv[2][0] > '9') {
            fputs("用法：particles_demo [--frames 1..10000]\n", stderr);
            return EXIT_FAILURE;
        }
        /* errno 检测溢出，end 检测尾部垃圾，范围检查拒绝零和过大的帧数。 */
        errno = 0;
        limit = strtoul(argv[2], &end, 10);
        if (errno != 0 || *end != '\0' || limit == 0 || limit > 10000) {
            fputs("帧数必须为 1..10000。\n", stderr);
            return EXIT_FAILURE;
        }
    }
    /* 零初始化句柄，让创建途中失败时也能走同一个 cleanup 出口。
     * 默认按失败处理，只有主循环正常结束后才改为成功。
     */
    GfxWindow display = {0};
    int status = EXIT_FAILURE;
    if (!gfx_sdl_open(&display, "C particles")) { goto cleanup; }
    static GfxCanvas canvas; /* 约 300 KiB 的画布放在静态存储区，减少栈占用。 */
    ParticleDemoState state = {.running = true};
    particles_reset(&state.particles, 42);
    /* 高精度计数器返回的是计数，不是秒；差值除以 frequency 才得到秒数。 */
    Uint64 previous = SDL_GetPerformanceCounter();
    Uint64 frequency = SDL_GetPerformanceFrequency();
    if (frequency == 0) {
        SDL_SetError("无法获取计时频率");
        goto cleanup;
    }
    unsigned long frames = 0;
    puts("粒子喷泉：Space 暂停/继续，暂停时 N 前进一步，R 重置，Esc 退出。");
    while (state.running) {
        /* 两次帧开始时间的差包含上一帧处理、显示和延时的全部时间。 */
        Uint64 started = SDL_GetPerformanceCounter();
        double elapsed = (double)(started - previous) / (double)frequency;
        previous = started; /* 暂停时也更新，防止恢复后把整个暂停时长算进去。 */
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            /* 一帧处理完当前事件队列；暂停切换和重置要求丢弃本帧增量。 */
            if (handle_event(&state, &event)) { elapsed = 0; }
        }
        if (!state.running) { break; } /* 收到退出请求后不再更新或显示下一帧。 */
        advance_time(&state, elapsed);
        /* 每帧重画背景，再根据当前粒子状态绘图；否则旧位置会留下拖尾。 */
        gfx_clear(&canvas, gfx_rgb(18, 23, 32));
        particles_draw(&state.particles, &canvas);
        char title[160]; /* snprintf 限制写入长度；%zu 对应 size_t 类型的存活数。 */
        snprintf(title, sizeof title,
                 "C particles | %zu/256 | %s | Space: pause | N: step | R: reset | Esc: quit",
                 particles_count(&state.particles), state.paused ? "paused" : "running");
        SDL_SetWindowTitle(display.window, title);
        /* 将 CPU 像素数组复制到 SDL 纹理；绘图核心不直接调用 SDL 图元 API。 */
        if (!gfx_sdl_present(&display, &canvas)) { goto cleanup; }
        /* --frames 统计显示次数，不是模拟步数；它们不一定一一对应。
         * && 的短路求值使无限运行模式下不必增加 frames。
         */
        if (limit != 0 && ++frames >= limit) { break; }
        /* 默认目标 60 FPS：先扣除本帧已经耗费的秒数，再等待剩余时间。
         * SDL_Delay 接受毫秒，转换为 Uint32 会截掉小数；调度也会引入误差，
         * 所以这里只是主动限速，不能保证精确 60 FPS。超时帧不再额外等待。
         */
        double spent = (double)(SDL_GetPerformanceCounter() - started) / (double)frequency;
        if (spent < 1.0 / PARTICLE_HZ) {
            SDL_Delay((Uint32)((1.0 / PARTICLE_HZ - spent) * 1000));
        }
    }
    status = EXIT_SUCCESS;
cleanup:
    /* goto 只用于统一清理路径；SDL 错误要在清理前读取，避免错误信息被改变。 */
    if (status != EXIT_SUCCESS) { fprintf(stderr, "SDL 失败：%s\n", SDL_GetError()); }
    gfx_sdl_close(&display);
    return status;
}
