/**
 * @file
 * @brief 逐字节分词与词频统计；默认内置短文本，文件和stdin显式选择。
 */
#include "../common/word_map.h"
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>

typedef struct {
    WordMap map;
    char *token;
    size_t length;
    size_t capacity;
    uint64_t total;
} WordCounter;

/**
 * @brief 初始化计数器和默认分配器；本函数不分配内存。
 * @param counter 有效可写新对象，不得包含未释放资源。
 */
static void counter_init(WordCounter *counter)
{
    *counter = (WordCounter){0};
    (void)word_map_init(&counter->map, nullptr, nullptr);
}

/**
 * @brief 释放正在拼接的单词与哈希表；重复销毁安全。
 * @param counter 有效可写、已初始化计数器。
 */
static void counter_destroy(WordCounter *counter)
{
    free(counter->token);
    counter->token = nullptr;
    counter->length = 0;
    counter->capacity = 0;
    word_map_destroy(&counter->map);
    counter->total = 0;
}

/**
 * @brief 将已拼接单词提交到表，词频和总词数不会在失败后部分增加。
 * @param counter 有效可写计数器，token是lowercase ASCII字母序列。
 * @return 成功或没有待处理单词时为true；分配或计数溢出时为false。
 */
static bool counter_flush(WordCounter *counter)
{
    if (counter->length == 0) {
        return true;
    }
    if (counter->total == UINT64_MAX ||
        !word_map_increment(&counter->map, counter->token, counter->length)) {
        return false;
    }
    ++counter->total;
    counter->length = 0;
    return true;
}

/**
 * @brief 追加字母到可扩容缓冲，避免固定长度截断单词。
 * @param counter 有效可写计数器。
 * @param letter 一个a..z字节。
 * @return 追加成功为true；容量溢出或分配失败时保持原单词。
 */
static bool counter_append(WordCounter *counter, char letter)
{
    if (counter->length == SIZE_MAX - 1) {
        return false;
    }
    if (counter->length + 1 >= counter->capacity) {
        size_t capacity = counter->capacity == 0 ? 32 : counter->capacity;
        if (counter->capacity != 0) {
            if (capacity > SIZE_MAX / 2) {
                return false;
            }
            capacity *= 2;
        }
        char *grown = realloc(counter->token, capacity);
        if (grown == nullptr) {
            return false;
        }
        counter->token = grown;
        counter->capacity = capacity;
    }
    counter->token[counter->length++] = letter;
    return true;
}

/**
 * @brief 增量扫描字节；仅A..Z/a..z组词并转小写，其余字节为分隔符。
 * @param counter 有效可写计数器，失败后可销毁，不保证整段回滚。
 * @param bytes 至少length个只读字节；length为零时可空。
 * @param length 输入长度，不依赖NUL终止。
 * @param final 最后一块时为true，提交末尾没有分隔符的单词。
 * @return 本块全部处理成功为true；参数、内存或计数失败为false。
 */
static bool counter_feed(WordCounter *counter, const unsigned char *bytes, size_t length,
                         bool final)
{
    if (counter == nullptr || (length > 0 && bytes == nullptr)) {
        return false;
    }
    for (size_t i = 0; i < length; ++i) {
        unsigned char byte = bytes[i];
        if (byte >= 'A' && byte <= 'Z') {
            byte = (unsigned char)(byte - 'A' + 'a');
        }
        if (byte >= 'a' && byte <= 'z') {
            if (!counter_append(counter, (char)byte)) {
                return false;
            }
        } else if (!counter_flush(counter)) {
            return false;
        }
    }
    return !final || counter_flush(counter);
}

/**
 * @brief 从流读入任意数量的数据块；跨块单词在计数器中保留。
 * @param counter 有效可写计数器。
 * @param input 已打开可读流，调用者负责关闭；本函数不回退文件位置。
 * @return 扫描与读取均成功为true，失败可保留已提交的前缀词频。
 */
static bool counter_stream(WordCounter *counter, FILE *input)
{
    if (counter == nullptr || input == nullptr) {
        return false;
    }
    unsigned char buffer[4096];
    size_t size;
    while ((size = fread(buffer, 1, sizeof(buffer), input)) > 0) {
        if (!counter_feed(counter, buffer, size, false)) {
            return false;
        }
    }
    return !ferror(input) && counter_feed(counter, nullptr, 0, true);
}

/**
 * @brief 为qsort比较词条，词频降序，平局按ASCII字典序。
 * @param left 指向WordEntry指针的只读地址。
 * @param right 同上。
 * @return 小于、等于或大于零的排序结果。
 */
static int compare_entries(const void *left, const void *right)
{
    const WordEntry *a = *(const WordEntry *const *)left;
    const WordEntry *b = *(const WordEntry *const *)right;
    if (a->count != b->count) {
        return a->count > b->count ? -1 : 1;
    }
    return strcmp(a->key, b->key);
}

/**
 * @brief 输出总词数、不同词数、表容量和排序后的前top项。
 * @param counter 有效只读计数器，打印期间不得修改哈希表。
 * @param top 最多展示的条目数，大于不同词数时只展示已有条目。
 * @param output 有效可写文本流，调用者负责关闭与最终flush。
 * @return 分配和写入成功为true；失败保留调用者拥有的表。
 */
static bool print_report(const WordCounter *counter, size_t top, FILE *output)
{
    if (counter->map.size > SIZE_MAX / sizeof(WordEntry *)) {
        return false;
    }
    const WordEntry **sorted = nullptr;
    if (counter->map.size > 0) {
        sorted = malloc(counter->map.size * sizeof(*sorted));
        if (sorted == nullptr) {
            return false;
        }
    }
    size_t used = 0;
    for (size_t i = 0; i < counter->map.capacity; ++i) {
        if (counter->map.entries[i].key != nullptr) {
            sorted[used++] = &counter->map.entries[i];
        }
    }
    if (used > 1) {
        qsort(sorted, used, sizeof(*sorted), compare_entries);
    }
    bool success =
        fprintf(output, "总词数=%" PRIu64 " 不同词=%zu 槽数=%zu 插入碰撞=%zu\n", counter->total,
                used, counter->map.capacity, counter->map.collisions) >= 0;
    if (success) {
        success = fprintf(output, "词频前%zu项：\n", top < used ? top : used) >= 0;
    }
    for (size_t i = 0; success && i < top && i < used; ++i) {
        success = fprintf(output, "%-18s %" PRIu64 "\n", sorted[i]->key, sorted[i]->count) >= 0;
    }
    free(sorted);
    return success;
}

/**
 * @brief 严格解析十进制正整数，不接受符号、空白或尾随字符。
 * @param text 有效NUL终止字符串。
 * @param output 有效可写结果，仅成功时修改。
 * @return 数值为1..SIZE_MAX时为true，否则为false。
 */
static bool parse_top(const char *text, size_t *output)
{
    if (text == nullptr || *text == '\0') {
        return false;
    }
    size_t value = 0;
    for (const unsigned char *cursor = (const unsigned char *)text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') {
            return false;
        }
        size_t digit = (size_t)(*cursor - '0');
        if (value > (SIZE_MAX - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
    }
    if (value == 0) {
        return false;
    }
    *output = value;
    return true;
}

/**
 * @brief 统计内置文本、指定文件或标准输入；默认内置样本避免make run阻塞。
 * @param argc 参数数量。
 * @param argv 可选--file 路径、--stdin或--top 正整数；两种来源互斥。
 * @return 统计及输出成功时为EXIT_SUCCESS；失败输出原因并返回失败。
 */
int main(int argc, char **argv)
{
    const char *path = nullptr;
    bool use_stdin = false;
    bool top_set = false;
    size_t top = 10;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--file") == 0 && path == nullptr && !use_stdin && i + 1 < argc) {
            path = argv[++i];
        } else if (strcmp(argv[i], "--stdin") == 0 && path == nullptr && !use_stdin) {
            use_stdin = true;
        } else if (strcmp(argv[i], "--top") == 0 && !top_set && i + 1 < argc &&
                   parse_top(argv[i + 1], &top)) {
            top_set = true;
            ++i;
        } else {
            fprintf(stderr, "用法：%s [--file 路径 | --stdin] [--top 正整数]\n", argv[0]);
            return EXIT_FAILURE;
        }
    }

    WordCounter counter;
    counter_init(&counter);
    bool success = true;
    if (path != nullptr || use_stdin) {
        FILE *input = use_stdin ? stdin : fopen(path, "rb");
        if (input == nullptr) {
            fprintf(stderr, "无法打开输入文件：%s\n", strerror(errno));
            success = false;
        } else {
            success = counter_stream(&counter, input);
            if (!success) {
                fputs("读取或统计词频失败：输入流、内存或计数越界\n", stderr);
            }
            if (!use_stdin && fclose(input) != 0) {
                fprintf(stderr, "关闭输入文件失败：%s\n", strerror(errno));
                success = false;
            }
        }
    } else {
        const char text[] = "C is small. C is fast. Learning C means learning memory. "
                            "Memory needs care; C rewards care.";
        success = counter_feed(&counter, (const unsigned char *)text, strlen(text), true);
        if (!success) {
            fputs("内置文本统计失败：内存或计数越界\n", stderr);
        }
    }
    if (success) {
        success = print_report(&counter, top, stdout);
        if (!success) {
            fputs("词频报告输出失败\n", stderr);
        }
    }
    if (fflush(stdout) != 0) {
        fputs("标准输出刷新失败\n", stderr);
        success = false;
    }
    counter_destroy(&counter);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
