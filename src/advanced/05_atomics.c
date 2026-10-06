/** @file
 * @brief POSIX 线程与 ISO C 原子操作。
 */
#include <assert.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    atomic_bool ready;
    atomic_uint ticks;
    int payload; /* 普通对象，通过 release/acquire 建立读写顺序。 */
} Shared;

/**
 * @brief 在线程中递增原子计数并发布载荷。
 * @param context 借用的 Shared 地址，须至少存活至线程结束。
 * @return nullptr；主线程通过共享状态获取结果。
 */
static void *produce(void *context)
{
    Shared *shared = context;
    for (unsigned int i = 0; i < 10000; ++i) {
        atomic_fetch_add_explicit(&shared->ticks, 1u, memory_order_relaxed);
    }
    shared->payload = 42;
    atomic_store_explicit(&shared->ready, true, memory_order_release);
    return nullptr;
}

/**
 * @brief 运行并验证POSIX 线程与 ISO C 原子操作。
 * @return 正常完成返回 0；资源操作失败可能返回非零，断言失败会终止程序。
 */
int main(void)
{
    Shared shared = {.ready = false, .ticks = 0, .payload = 0};
    pthread_t thread;
    int error = pthread_create(&thread, nullptr, produce, &shared);
    if (error != 0) {
        fprintf(stderr, "创建线程失败：%s\n", strerror(error));
        return EXIT_FAILURE;
    }
    for (unsigned int i = 0; i < 10000; ++i) {
        atomic_fetch_add_explicit(&shared.ticks, 1u, memory_order_relaxed);
    }
    /* acquire 读取到 release 发布的 true，才允许读取普通 payload。
     * relaxed 保证计数增量的原子性，但不能用于发布其他普通对象。 */
    while (!atomic_load_explicit(&shared.ready, memory_order_acquire)) {
        sched_yield(); /* 仅教学用等待；工程中考虑条件变量或事件通知。 */
    }
    assert(shared.payload == 42);
    error = pthread_join(thread, nullptr);
    if (error != 0) {
        fprintf(stderr, "等待线程失败：%s\n", strerror(error));
        return EXIT_FAILURE;
    }
    unsigned int ticks = atomic_load_explicit(&shared.ticks, memory_order_relaxed);
    assert(ticks == 20000);
    printf("原子操作：双线程计数=%u，发布载荷=%d\n", ticks, shared.payload);
    return EXIT_SUCCESS;
}
