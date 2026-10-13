/**
 * @file
 * @brief 哈希冲突、动态扩容、复制键、流式分词和失败状态测试。
 */
#define main word_frequency_demo_main
#include "../src/advanced/07_word_frequency.c"
#undef main
#include <assert.h>

static size_t allocation_calls;
static size_t fail_at;

/**
 * @brief 按序号注入一次分配失败，成功时返回malloc拥有的内存。
 * @param bytes 请求字节数。
 * @return 到达fail_at时为空，否则为可用内存。
 */
static void *test_allocate(size_t bytes)
{
    ++allocation_calls;
    return allocation_calls == fail_at ? nullptr : malloc(bytes);
}

/**
 * @brief 寻找两个哈希初始槽相同但内容不同的单字母词。
 * @param first 有效可写第一个字母。
 * @param second 有效可写第二个字母。
 */
static void collision_pair(char *first, char *second)
{
    for (char a = 'a'; a <= 'z'; ++a) {
        for (char b = a + 1; b <= 'z'; ++b) {
            if ((word_map_hash(&a, 1) & 7) == (word_map_hash(&b, 1) & 7)) {
                *first = a;
                *second = b;
                return;
            }
        }
    }
    assert(false);
}

/**
 * @brief 验证冲突仍区分键、复用输入缓冲与多次扩容后可查回每个值。
 */
static void test_map_growth(void)
{
    WordMap map;
    assert(word_map_init(&map, nullptr, nullptr));
    char first;
    char second;
    collision_pair(&first, &second);
    assert(word_map_increment(&map, &first, 1));
    assert(word_map_increment(&map, &second, 1));
    assert(map.size == 2 && map.capacity == 8 && map.collisions >= 1);
    char key[4];
    uint64_t count = 0;
    for (size_t i = 0; i < 250; ++i) {
        key[0] = 'a' + (char)(i / 26);
        key[1] = 'a' + (char)(i % 26);
        assert(word_map_increment(&map, key, 2));
        if (i % 2 == 0) {
            assert(word_map_increment(&map, key, 2));
        }
        key[0] = '?';
        key[1] = '?';
    }
    assert(map.size == 252 && map.capacity >= 512);
    for (size_t i = 0; i < 250; ++i) {
        key[0] = 'a' + (char)(i / 26);
        key[1] = 'a' + (char)(i % 26);
        assert(word_map_get(&map, key, 2, &count));
        assert(count == (i % 2 == 0 ? 2u : 1u));
    }
    assert(word_map_get(&map, &first, 1, &count) && count == 1);
    assert(word_map_get(&map, &second, 1, &count) && count == 1);
    count = 99;
    assert(!word_map_get(&map, "missing", 7, &count) && count == 99);
    assert(!word_map_increment(&map, "UPPER", 5));
    assert(!word_map_increment(&map, "", 0));
    assert(map.size == 252);
    word_map_destroy(&map);
    word_map_destroy(&map);
    assert(map.size == 0 && map.capacity == 0);
}

/**
 * @brief 在复制新键和扩容分配时分别失败，检查旧指针及内容不变。
 */
static void test_allocation_failure(void)
{
    WordMap map;
    allocation_calls = 0;
    fail_at = SIZE_MAX;
    assert(word_map_init(&map, test_allocate, free));
    for (char key = 'a'; key <= 'f'; ++key) {
        assert(word_map_increment(&map, &key, 1));
    }
    assert(map.size == 6 && map.capacity == 8);
    WordMap old = map;
    WordEntry snapshot[8];
    memcpy(snapshot, map.entries, sizeof(snapshot));
    fail_at = allocation_calls + 1;
    assert(!word_map_increment(&map, "g", 1));
    assert(map.entries == old.entries && map.size == old.size && map.capacity == old.capacity &&
           map.collisions == old.collisions);
    assert(memcmp(snapshot, map.entries, sizeof(snapshot)) == 0);
    fail_at = allocation_calls + 2;
    assert(!word_map_increment(&map, "g", 1));
    assert(map.entries == old.entries && map.size == old.size && map.capacity == old.capacity &&
           map.collisions == old.collisions);
    assert(memcmp(snapshot, map.entries, sizeof(snapshot)) == 0);
    fail_at = SIZE_MAX;
    assert(word_map_increment(&map, "g", 1));
    assert(map.size == 7 && map.capacity == 16);
    uint64_t count = 0;
    bool found = false;
    size_t slot = word_map_slot(&map, "a", 1, word_map_hash("a", 1), &found);
    assert(found);
    map.entries[slot].count = UINT64_MAX;
    assert(!word_map_increment(&map, "a", 1));
    assert(word_map_get(&map, "a", 1, &count) && count == UINT64_MAX);
    word_map_destroy(&map);
}

/**
 * @brief 小块输入跨单词边界、大小写、数字标点和无分隔符的末尾。
 */
static void test_streaming(void)
{
    WordCounter counter;
    counter_init(&counter);
    assert(counter_feed(&counter, (const unsigned char *)"Al", 2, false));
    assert(counter_feed(&counter, (const unsigned char *)"pha ALP", 7, false));
    assert(counter_feed(&counter, (const unsigned char *)"HA42beta!", 9, false));
    assert(counter_feed(&counter, (const unsigned char *)"BeTa", 4, true));
    uint64_t count = 0;
    assert(word_map_get(&counter.map, "alpha", 5, &count) && count == 2);
    assert(word_map_get(&counter.map, "beta", 4, &count) && count == 2);
    assert(counter.total == 4 && counter.map.size == 2);
    assert(!counter_feed(&counter, nullptr, 1, false));
    assert(counter.total == 4);
    counter_destroy(&counter);

    counter_init(&counter);
    FILE *input = tmpfile();
    assert(input != nullptr);
    for (size_t i = 0; i < 5000; ++i) {
        assert(fputc('A', input) != EOF);
    }
    assert(fputc(' ', input) != EOF);
    assert(fputs("b B", input) >= 0);
    rewind(input);
    assert(counter_stream(&counter, input));
    assert(fclose(input) == 0);
    assert(counter.total == 3 && counter.map.size == 2);
    assert(word_map_get(&counter.map, "b", 1, &count) && count == 2);
    counter_destroy(&counter);
}

/**
 * @brief 空输入、排序平局、严格top解析与错误输入不修改结果。
 */
static void test_reporting(void)
{
    WordCounter counter;
    counter_init(&counter);
    assert(counter_feed(&counter, nullptr, 0, true));
    FILE *output = tmpfile();
    assert(output != nullptr);
    assert(print_report(&counter, 10, output));
    assert(fclose(output) == 0);
    const unsigned char text[] = "zeta alpha zeta alpha beta";
    assert(counter_feed(&counter, text, sizeof(text) - 1, true));
    output = tmpfile();
    assert(output != nullptr);
    assert(print_report(&counter, 3, output));
    rewind(output);
    char line[160];
    assert(fgets(line, sizeof(line), output) != nullptr);
    assert(strstr(line, "总词数=5 不同词=3") != nullptr);
    assert(fgets(line, sizeof(line), output) != nullptr);
    assert(fgets(line, sizeof(line), output) != nullptr);
    assert(strncmp(line, "alpha", 5) == 0);
    assert(fgets(line, sizeof(line), output) != nullptr);
    assert(strncmp(line, "zeta", 4) == 0);
    assert(fgets(line, sizeof(line), output) != nullptr);
    assert(strncmp(line, "beta", 4) == 0);
    assert(fclose(output) == 0);
    size_t top = 7;
    assert(parse_top("12", &top) && top == 12);
    const char *invalid[] = {"", "0", "-1", "+1", " 1", "1x", "999999999999999999999999"};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        assert(!parse_top(invalid[i], &top) && top == 12);
    }
    counter_destroy(&counter);
}

/**
 * @brief 执行可重复的哈希表和词频核心验证。
 * @return 全部断言通过时为零。
 */
int main(void)
{
    test_map_growth();
    test_allocation_failure();
    test_streaming();
    test_reporting();
    puts("哈希表测试通过：碰撞、扩容、所有权、失败回滚、跨块分词与排序。");
    return 0;
}
