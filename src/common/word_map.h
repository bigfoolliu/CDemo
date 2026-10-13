/**
 * @file
 * @brief 词频哈希表：FNV-1a、线性探测、75%负载扩容和键的所有权。
 * 头文件实现便于单文件示例复用；本模块为单线程使用，不可浅复制WordMap。
 */
#ifndef CDEMO_WORD_MAP_H
#define CDEMO_WORD_MAP_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *key;
    size_t length;
    uint64_t hash;
    uint64_t count;
} WordEntry;

typedef struct {
    WordEntry *entries;
    size_t size;
    size_t capacity;
    size_t collisions;
    void *(*allocate)(size_t);
    void (*release)(void *);
} WordMap;

/**
 * @brief 初始化空表；传入成对分配/释放函数可注入失败。
 * @param map 有效可写对象，初始化前不得拥有未释放资源。
 * @param allocate 分配回调，空指针时使用malloc；需返回可写内存。
 * @param release 与allocate匹配的释放回调，空指针时使用free。
 * @return 参数有效时为true；回调必须同时为空或同时非空。
 */
static inline bool word_map_init(WordMap *map, void *(*allocate)(size_t), void (*release)(void *))
{
    if (map == nullptr || (allocate == nullptr) != (release == nullptr)) {
        return false;
    }
    *map = (WordMap){.allocate = allocate == nullptr ? malloc : allocate,
                     .release = release == nullptr ? free : release};
    return true;
}

/**
 * @brief 释放所有键与槽数组；重复调用安全，之后可再次插入。
 * @param map 已初始化可写表；空指针无操作。
 */
static inline void word_map_destroy(WordMap *map)
{
    if (map == nullptr) {
        return;
    }
    for (size_t i = 0; i < map->capacity; ++i) {
        if (map->entries[i].key != nullptr) {
            map->release(map->entries[i].key);
        }
    }
    if (map->entries != nullptr) {
        map->release(map->entries);
    }
    map->entries = nullptr;
    map->size = 0;
    map->capacity = 0;
    map->collisions = 0;
}

/**
 * @brief 检查非空ASCII小写单词，避免键中NUL或不可打印字节。
 * @param key 至少length个可读字节，可无终止符。
 * @param length 字节数，必须大于零且小于SIZE_MAX。
 * @return 全部字节为a..z时为true。
 */
static inline bool word_map_valid_key(const char *key, size_t length)
{
    if (key == nullptr || length == 0 || length == SIZE_MAX) {
        return false;
    }
    for (size_t i = 0; i < length; ++i) {
        if (key[i] < 'a' || key[i] > 'z') {
            return false;
        }
    }
    return true;
}

/**
 * @brief 计算FNV-1a的64位哈希；冲突仍须比较长度和内容。
 * @param key 至少length个有效只读字节。
 * @param length 字节数。
 * @return 完整64位哈希值。
 */
static inline uint64_t word_map_hash(const char *key, size_t length)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < length; ++i) {
        hash ^= (unsigned char)key[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

/**
 * @brief 在非满表中寻找键或空槽；容量总为2的幂。
 * @param map 已分配槽且至少有一个空槽的有效只读表。
 * @param key 至少length个只读规范单词字节。
 * @param length 字节数。
 * @param hash 该键的FNV-1a哈希。
 * @param found 有效可写结果，返回槽是已有键时置true。
 * @return 匹配键或首个空槽的索引。
 */
static inline size_t word_map_slot(const WordMap *map, const char *key, size_t length,
                                   uint64_t hash, bool *found)
{
    size_t slot = (size_t)hash & (map->capacity - 1);
    while (map->entries[slot].key != nullptr) {
        const WordEntry *entry = &map->entries[slot];
        if (entry->hash == hash && entry->length == length &&
            memcmp(entry->key, key, length) == 0) {
            *found = true;
            return slot;
        }
        slot = (slot + 1) & (map->capacity - 1);
    }
    *found = false;
    return slot;
}

/**
 * @brief 几何扩容并迁移键指针，不重新复制键；分配失败保留旧表。
 * @param map 已初始化可写表，容量为零或2的幂且负载不超过75%。
 * @return 扩容成功为true；空间不足或分配失败为false且状态不变。
 */
static inline bool word_map_grow(WordMap *map)
{
    if (map->capacity > SIZE_MAX / 2) {
        return false;
    }
    size_t capacity = map->capacity == 0 ? 8 : map->capacity * 2;
    if (capacity > SIZE_MAX / sizeof(WordEntry)) {
        return false;
    }
    WordEntry *grown = map->allocate(capacity * sizeof(*grown));
    if (grown == nullptr) {
        return false;
    }
    memset(grown, 0, capacity * sizeof(*grown));
    for (size_t i = 0; i < map->capacity; ++i) {
        WordEntry entry = map->entries[i];
        if (entry.key == nullptr) {
            continue;
        }
        size_t slot = (size_t)entry.hash & (capacity - 1);
        while (grown[slot].key != nullptr) {
            slot = (slot + 1) & (capacity - 1);
        }
        grown[slot] = entry;
    }
    if (map->entries != nullptr) {
        map->release(map->entries);
    }
    map->entries = grown;
    map->capacity = capacity;
    return true;
}

/**
 * @brief 单词计数加一；新增键复制到表私有内存，调用者可复用原缓冲。
 * @param map 已初始化的可写表，不可浅复制或并发修改。
 * @param key 至少length字节的只读a..z单词，可无NUL终止符。
 * @param length 字节数，必须大于零。
 * @return 成功为true；非法键、计数溢出或内存失败时原表状态不变。
 */
static inline bool word_map_increment(WordMap *map, const char *key, size_t length)
{
    if (map == nullptr || map->allocate == nullptr || !word_map_valid_key(key, length)) {
        return false;
    }
    uint64_t hash = word_map_hash(key, length);
    if (map->capacity != 0) {
        bool found = false;
        size_t slot = word_map_slot(map, key, length, hash, &found);
        if (found) {
            if (map->entries[slot].count == UINT64_MAX) {
                return false;
            }
            ++map->entries[slot].count;
            return true;
        }
    }

    char *owned = map->allocate(length + 1);
    if (owned == nullptr) {
        return false;
    }
    memcpy(owned, key, length);
    owned[length] = '\0';
    if (map->capacity == 0 || map->size + 1 > map->capacity - map->capacity / 4) {
        if (!word_map_grow(map)) {
            map->release(owned);
            return false;
        }
    }
    bool found = false;
    size_t slot = word_map_slot(map, owned, length, hash, &found);
    size_t start = (size_t)hash & (map->capacity - 1);
    map->collisions += slot != start;
    map->entries[slot] = (WordEntry){.key = owned, .length = length, .hash = hash, .count = 1};
    ++map->size;
    return true;
}

/**
 * @brief 查询单词计数；返回的只有整数，没有内部键指针所有权。
 * @param map 已初始化的只读表。
 * @param key 至少length字节的只读a..z单词。
 * @param length 字节数，大于零。
 * @param count 有效可写整数，只有查到键时才修改。
 * @return 键存在时为true；缺失或参数非法时为false。
 */
static inline bool word_map_get(const WordMap *map, const char *key, size_t length, uint64_t *count)
{
    if (map == nullptr || count == nullptr || map->capacity == 0 ||
        !word_map_valid_key(key, length)) {
        return false;
    }
    bool found = false;
    size_t slot = word_map_slot(map, key, length, word_map_hash(key, length), &found);
    if (found) {
        *count = map->entries[slot].count;
    }
    return found;
}

#endif
