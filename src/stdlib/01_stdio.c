/** @file
 * @brief 使用临时文件学习文本、二进制输入输出及文件定位。
 *
 * 学习范围：<stdio.h> 输入输出库。
 * 接口概览（主要常规接口，不含可选边界检查接口）：
 * - 流与文件：fopen、freopen、fclose、fflush、setbuf、setvbuf、remove、rename、tmpfile、tmpnam。
 * - 格式化输出：printf/fprintf/sprintf/snprintf 及对应的 vprintf/vfprintf/vsprintf/vsnprintf。
 * - 格式化输入：scanf/fscanf/sscanf 及对应的 vscanf/vfscanf/vsscanf。
 * - 字符与行：fgetc/getc/getchar、fputc/putc/putchar、fgets、fputs、puts、ungetc。
 * - 二进制读写：fread、fwrite；定位：fseek、ftell、rewind、fgetpos、fsetpos。
 * - 状态与错误：feof、ferror、clearerr、perror。
 * - 类型与宏：FILE、fpos_t、size_t、stdin/stdout/stderr、EOF、BUFSIZ、SEEK_SET 等。
 * 本文件已演示：snprintf、tmpfile、fputs、fgets、fread、fwrite、fseek、fgetc、
 * feof、ferror、clearerr、fclose，并用 puts/fputs/perror 输出状态或错误。
 * 本文件未演示：普通文件打开、缓冲配置、格式化输入、其他定位及文件管理。
 * 观察重点：容量包含终止符；读写返回元素数；读取失败后区分 EOF 和错误。
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * @brief 写入并读取两行文本，验证 EOF 与流错误的区别。
 * @return 成功返回 true，文件创建、读写或关闭失败返回 false。
 */
static bool text_io(void)
{
    FILE *file = tmpfile(); /* 自动删除的临时流，不覆盖用户文件。 */
    if (file == nullptr) {
        perror("创建文本临时文件失败");
        return false;
    }
    bool ok = fputs("apple\npear\n", file) >= 0;
    /* 更新流从写切换到读前，执行文件定位；不能直接混用。 */
    if (ok) {
        ok = fseek(file, 0, SEEK_SET) == 0;
    }
    char line[16];
    size_t lines = 0;
    while (ok && fgets(line, sizeof(line), file) != nullptr) {
        assert(strcmp(line, lines == 0 ? "apple\n" : "pear\n") == 0);
        ++lines;
    }
    if (ok) {
        ok = !ferror(file);
        assert(feof(file) && lines == 2);
        clearerr(file);
        assert(!feof(file) && !ferror(file));
    }
    if (fclose(file) == EOF) {
        ok = false;
    }
    return ok;
}

/**
 * @brief 验证 fread/fwrite 按元素计数及 fgetc 的 EOF 返回值。
 * @return 成功返回 true，文件操作失败返回 false。
 */
static bool binary_io(void)
{
    FILE *file = tmpfile();
    if (file == nullptr) {
        perror("创建二进制临时文件失败");
        return false;
    }
    const unsigned char bytes[] = {0, 127, 255};
    unsigned char restored[sizeof(bytes)] = {0};
    bool ok = fwrite(bytes, sizeof(bytes[0]), sizeof(bytes), file) == sizeof(bytes);
    if (ok) {
        ok = fseek(file, 0, SEEK_SET) == 0;
    }
    if (ok) {
        ok = fread(restored, sizeof(restored[0]), sizeof(restored), file) == sizeof(restored);
        assert(!ok || memcmp(bytes, restored, sizeof(bytes)) == 0);
        int next = fgetc(file); /* 必须用 int 保留 EOF，不能先转成 char。 */
        assert(next == EOF);
        ok = ok && !ferror(file) && feof(file);
    }
    if (fclose(file) == EOF) {
        ok = false;
    }
    return ok;
}

/** @brief 验证格式化、文本和二进制读写。 @return 成功为 0，文件操作失败为非零。 */
int main(void)
{
    char text[32];
    int required = snprintf(text, sizeof(text), "count=%zu", (size_t)3);
    assert(required == 7 && strcmp(text, "count=3") == 0);
    char small[4];
    required = snprintf(small, sizeof(small), "%s", text);
    assert(required == 7 && (size_t)required >= sizeof(small));
    assert(strcmp(small, "cou") == 0); /* 返回所需长度，可据此检测截断。 */
    if (!text_io() || !binary_io()) {
        fputs("标准库输入输出验证失败。\n", stderr);
        return EXIT_FAILURE;
    }
    puts("stdio：格式化截断、逐行读取、二进制读写、EOF 与文件定位验证通过");
    return EXIT_SUCCESS;
}
