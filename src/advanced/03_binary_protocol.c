/** @file
 * @brief 二进制报文与借用视图。
 */
#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>

static_assert(CHAR_BIT == 8, "此协议要求 8 位字节");

typedef struct {
    uint8_t kind;
    const unsigned char *payload;
    size_t length;
} Packet;

/**
 * @brief 校验并解码二进制报文，返回零拷贝视图。
 * @param bytes 至少 size 字节的只读输入；size 为零时不解引用。
 * @param size 输入缓冲区的字节数。
 * @param output 成功时写入报文视图；载荷借用 bytes 的存储。
 * @return 报文完整且版本正确返回 true；失败返回 false 且输出不变。
 * @note 格式：[版本=1][类型][长度高字节][长度低字节][载荷]。
 *       返回借用视图，不复制数据；输入缓冲区必须比视图活得更久。
 *       不将原始字节强转成结构体，避免填充、对齐、字节序和别名问题。
 */
static bool decode(const unsigned char *bytes, size_t size, Packet *output)
{
    if (size < 4 || bytes[0] != 1) {
        return false;
    }
    size_t length = ((size_t)bytes[2] << 8) | bytes[3];
    if (length != size - 4) { /* 先减再比较，避免 length + 4 溢出。 */
        return false;
    }
    *output = (Packet){.kind = bytes[1], .payload = bytes + 4, .length = length};
    return true;
}

/**
 * @brief 运行并验证二进制报文与借用视图。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    const unsigned char bytes[] = {1, 7, 0, 3, 'a', 'b', 'c'};
    Packet packet = {0};
    bool success = decode(bytes, sizeof(bytes), &packet);
    assert(success && packet.kind == 7 && packet.length == 3);
    assert(packet.payload[0] == 'a' && packet.payload[2] == 'c');
    for (size_t size = 0; size < sizeof(bytes); ++size) {
        success = decode(bytes, size, &packet);
        assert(!success && packet.length == 3); /* 失败不改输出。 */
    }
    const unsigned char wrong_version[] = {2, 0, 0, 0};
    success = decode(wrong_version, sizeof(wrong_version), &packet);
    assert(!success);
    const unsigned char empty[] = {1, 0, 0, 0};
    success = decode(empty, sizeof(empty), &packet);
    assert(success && packet.length == 0); /* 尾后指针不可解引用。 */
    puts("二进制协议：大端解码、截断检查、零拷贝借用视图验证通过");
    return 0;
}
