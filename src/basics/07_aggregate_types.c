/** @file
 * @brief 通过可运行示例学习结构体、联合体、枚举和类型别名。
 */
#include <assert.h>
#include <stddef.h>
#include <stdio.h>

typedef struct Point {
    int x;
    int y;
} Point;

typedef struct Person {
    const char *name;
    int age;
    Point location;
} Person;

typedef enum {
    INTEGER_VALUE,
    REAL_VALUE
} ValueKind;

typedef union {
    int integer;
    double real;
} ValueData;

/* kind 是判别标签；它和 data 中当前有效的成员必须保持一致。 */
typedef struct {
    ValueKind kind;
    ValueData data;
} Value;

typedef struct {
    char flag;
    int count;
    double ratio;
} LayoutExample;

/**
 * @brief 运行并验证聚合类型示例。
 * @return 示例正常完成时返回 0；断言失败时程序终止。
 */
int main(void)
{
    Person person = {
        .name = "小明",
        .age = 18,
        .location = {.x = 3, .y = 5}
    };
    Person copy = person; /* 结构体整体赋值会复制每个成员。 */
    Person *person_pointer = &copy;
    person_pointer->age = 20; /* -> 等价于 (*person_pointer).age。 */

    /* 未指定的成员会进行零初始化，因此 age 为 0。 */
    Person partial = {.name = "小林"};

    assert(person.age == 18);
    assert(copy.age == 20);
    /* 指针成员复制的是地址，不是字符串内容。 */
    assert(copy.name == person.name);
    assert(copy.location.x == 3 && copy.location.y == 5);
    assert(partial.age == 0);

    /* 复合字面量创建一个未命名的 Person 对象。 */
    Person temporary = (Person){
        .name = "小红",
        .age = 21,
        .location = {.x = 8, .y = 2}
    };
    assert(temporary.age == 21);

    /* 指定初始化同时选定联合体成员，并设置对应的判别标签。 */
    Value integer_value = {
        .kind = INTEGER_VALUE,
        .data.integer = 42
    };
    Value real_value = {
        .kind = REAL_VALUE,
        .data.real = 3.5
    };

    switch (integer_value.kind) {
    case INTEGER_VALUE:
        assert(integer_value.data.integer == 42);
        printf("整数值：%d\n", integer_value.data.integer);
        break;
    case REAL_VALUE:
        assert(0); /* 当前标签与实际初始化的成员不匹配。 */
        break;
    }

    switch (real_value.kind) {
    case INTEGER_VALUE:
        assert(0); /* 当前标签与实际初始化的成员不匹配。 */
        break;
    case REAL_VALUE:
        assert(real_value.data.real == 3.5);
        printf("浮点值：%.1f\n", real_value.data.real);
        break;
    }

    /* sizeof 显示本机布局；不要据此假设不同平台的大小相同。 */
    printf("结构体访问：%s，年龄=%d，位置=(%d, %d)\n",
           person_pointer->name,
           person_pointer->age,
           person_pointer->location.x,
           person_pointer->location.y);
    printf("sizeof(Person)=%zu，sizeof(ValueData)=%zu，sizeof(LayoutExample)=%zu\n",
           sizeof(Person),
           sizeof(ValueData),
           sizeof(LayoutExample));
    printf("LayoutExample 成员偏移：flag=%zu，count=%zu，ratio=%zu\n",
           offsetof(LayoutExample, flag),
           offsetof(LayoutExample, count),
           offsetof(LayoutExample, ratio));

    return 0;
}
