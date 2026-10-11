/** @file
 * @brief 结构体、联合体与枚举。
 */
#include <assert.h>
#include <stdio.h>

typedef struct {
    const char *name;
    int age;
} Person; /* typedef 提供类型别名，不创建对象。 */

enum ValueKind {
    INTEGER_VALUE,
    REAL_VALUE
};
struct Value {
    enum ValueKind kind;
    union {
        int integer;
        double real;
    } data; /* 成员共享存储；标签记录当前使用的成员。 */
};

/**
 * @brief 运行并验证结构体、联合体与枚举。
 * @return 正常完成返回 0；断言失败会终止程序。
 */
int main(void)
{
    Person person = {.name = "小明", .age = 18}; /* 指定初始化。 */
    Person copy = person;                        /* 结构体可整体赋值；指针成员只是复制地址。 */
    Person *pointer = &copy;
    pointer->age = 20; /* -> 等价于 (*pointer).age。 */
    struct Value value = {.kind = REAL_VALUE, .data.real = 3.5};
    assert(person.age == 18 && copy.age == 20);
    assert(value.kind == REAL_VALUE && value.data.real == 3.5);
    Person temporary = (Person){.name = "小红", .age = 21}; /* 复合字面量。 */
    assert(temporary.age == 21);
    printf("聚合类型：%s，年龄=%d\n", pointer->name, pointer->age);
    return 0;
}
