/** @file
 * @brief 区分日历时间、时间格式化和进程 CPU 时间。
 *
 * 学习范围：<time.h> 日历时间、时间格式化与 CPU 计时。
 * 接口概览（常规函数）：
 * - 获取与运算：time、clock、difftime、mktime、timespec_get。
 * - 日历转换：localtime、gmtime；文本转换：strftime、asctime、ctime。
 * - C23 新增：localtime_r、gmtime_r、timespec_getres；需单独验证平台支持。
 * - 类型与宏：time_t、clock_t、struct tm、struct timespec、CLOCKS_PER_SEC、TIME_UTC、NULL。
 * 本文件已演示：time、localtime、strftime、difftime、timespec_get、clock，
 * 使用上述时间类型、CLOCKS_PER_SEC 和 TIME_UTC。
 * 本文件未演示：mktime、gmtime、asctime、ctime 和 C23 新增接口。
 * 观察重点：复制库管理的日历存储；区分本地日历时间、CPU 时间与单调计时。
 * 本阶段优先使用 strftime；asctime/ctime 在 C23 中属于过时接口。
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/** @brief 验证时间接口，不依赖固定时区或执行速度。 @return 成功为 0，接口失败为非零。 */
int main(void)
{
    time_t now = time(nullptr);
    if (now == (time_t)-1) {
        fputs("time 获取日历时间失败。\n", stderr);
        return EXIT_FAILURE;
    }
    struct tm *calendar = localtime(&now);
    if (calendar == nullptr) {
        fputs("localtime 转换失败。\n", stderr);
        return EXIT_FAILURE;
    }
    struct tm copied = *calendar; /* 返回存储由库管理，后续调用可能覆盖它。 */
    char text[64];
    size_t length = strftime(text, sizeof(text), "%Y-%m-%d %H:%M:%S", &copied);
    if (length == 0) {
        fputs("strftime 格式化失败或缓冲区不足。\n", stderr);
        return EXIT_FAILURE;
    }
    assert(length > 0 && difftime(now, now) == 0);
    struct timespec precise;
    if (timespec_get(&precise, TIME_UTC) != TIME_UTC) {
        fputs("timespec_get 获取时间失败。\n", stderr);
        return EXIT_FAILURE;
    }
    assert(precise.tv_nsec >= 0 && precise.tv_nsec < 1000000000L);
    clock_t cpu = clock();
    if (cpu == (clock_t)-1) {
        puts("time：日历与格式化验证通过；此实现无法提供 CPU 时间");
        return EXIT_SUCCESS;
    }
    printf("time：本地时间=%s，纳秒字段=%ld，CPU 秒数=%.6f\n",
           text, precise.tv_nsec, (double)cpu / CLOCKS_PER_SEC);
    /* clock 是进程 CPU 时间；time/timespec_get 不保证是单调计时源。 */
    return EXIT_SUCCESS;
}
