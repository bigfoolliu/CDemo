#ifndef CDEMO_SNAKE_ENV_H
#define CDEMO_SNAKE_ENV_H

/* 纯规则层：不依赖屏幕、计时或文件，人工游戏和训练共用同一实现。 */

#include <stdint.h>
#include <stdlib.h>

enum { WIDTH = 20, HEIGHT = 10, CAPACITY = WIDTH * HEIGHT, FEATURES = 11, ACTIONS = 3 };
typedef struct { int x; int y; } Point;
/* body[0] 是蛇头；仅 [0,length) 有效；rng 属于这局环境。 */
typedef struct {
    Point body[CAPACITY];
    size_t length;
    Point direction;
    Point food;
    uint32_t rng;
} Game;

/* 每个环境和训练器拥有独立的随机流；零种子映射到固定非零状态。 */
static inline uint32_t snake_random(uint32_t *state)
{
    uint32_t x = *state == 0 ? UINT32_C(0x9e3779b9) : *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* 生成 [0,1) 内的数，供训练器判断是否随机探索。 */
static inline double snake_uniform(uint32_t *state)
{
    return (double)snake_random(state) / 4294967296.0;
}

/* 比较坐标值，不比较结构体的内存字节（其中可能有填充）。 */
static inline bool equal(Point a, Point b)
{
    return a.x == b.x && a.y == b.y;
}

/* 只扫描前 count 节身体，碰撞检测可借此排除即将离开的尾节。 */
static inline bool occupied(const Game *game, Point point, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (equal(game->body[i], point)) {
            return true;
        }
    }
    return false;
}

/* 随机选择第几个空格，再扫描定位；不会把食物放在蛇身上。 */
static inline void place_food(Game *game)
{
    if (game->length == CAPACITY) {
        return;
    }
    size_t choice = snake_random(&game->rng) % (CAPACITY - game->length);
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            Point point = {x, y};
            if (!occupied(game, point, game->length)) {
                if (choice == 0) {
                    game->food = point;
                    return;
                }
                --choice;
            }
        }
    }
}

/* 固定初始蛇形与方向；相同种子产生相同的初始食物。 */
static inline Game snake_reset(uint32_t seed)
{
    Game game = {.body = {{10, 5}, {9, 5}, {8, 5}}, .length = 3,
                 .direction = {1, 0}, .rng = seed};
    place_food(&game);
    return game;
}

/* 只预测下一步是否碰撞，不移动蛇；direction 应为单位方向。 */
static inline bool snake_collision(const Game *game, Point direction)
{
    Point head = {game->body[0].x + direction.x, game->body[0].y + direction.y};
    bool eating = equal(head, game->food);
    return head.x < 0 || head.x >= WIDTH || head.y < 0 || head.y >= HEIGHT ||
           occupied(game, head, game->length - (eating ? 0u : 1u));
}

/* 不吃食物时旧尾格可以进入；碰撞不修改环境；满棋盘后不可继续调用。 */
static inline bool step(Game *game, Point direction)
{
    if (game->length == CAPACITY) {
        return false;
    }
    if (direction.x == -game->direction.x && direction.y == -game->direction.y) {
        direction = game->direction;
    }
    if (snake_collision(game, direction)) {
        return false;
    }
    Point head = {game->body[0].x + direction.x, game->body[0].y + direction.y};
    bool eating = equal(head, game->food);
    size_t length = game->length + (eating ? 1u : 0u);
    for (size_t i = length - 1; i > 0; --i) {
        /* 从尾向头移动，避免覆盖仍需要复制的旧坐标。 */
        game->body[i] = game->body[i - 1];
    }
    game->body[0] = head;
    game->length = length;
    game->direction = direction;
    if (eating) {
        place_food(game);
    }
    return true;
}

/* 0 直行、1 左转、2 右转；屏幕的 y 轴向下。 */
static inline Point snake_direction(Point current, int action)
{
    if (action == 1) {
        return (Point){current.y, -current.x};
    }
    if (action == 2) {
        return (Point){-current.y, current.x};
    }
    return current;
}

/* 11 个 0/1 特征：三方向危险 + 当前四方向 + 食物四方位。
 * 这是简化观测，无法表示完整蛇身布局。 */
static inline void snake_observe(const Game *game, double state[FEATURES])
{
    for (int action = 0; action < ACTIONS; ++action) {
        state[action] = snake_collision(game, snake_direction(game->direction, action));
    }
    state[3] = game->direction.x == -1;
    state[4] = game->direction.x == 1;
    state[5] = game->direction.y == -1;
    state[6] = game->direction.y == 1;
    state[7] = game->food.x < game->body[0].x;
    state[8] = game->food.x > game->body[0].x;
    state[9] = game->food.y < game->body[0].y;
    state[10] = game->food.y > game->body[0].y;
}

/* 负的归一化曼哈顿距离：离食物越近，势越接近零。 */
static inline double snake_potential(const Game *game)
{
    int distance = abs(game->food.x - game->body[0].x) + abs(game->food.y - game->body[0].y);
    return -(double)distance / (WIDTH + HEIGHT);
}

/* terminated 只表示死亡或获胜；时间上限由训练器单独处理。 */
typedef struct { double reward; bool terminated; bool won; } SnakeResult;

/* 将合法相对动作（0..2）转成方向，执行一步并计算基础奖励与塑形奖励。 */
static inline SnakeResult snake_act(Game *game, int action)
{
    double previous = snake_potential(game);
    size_t length = game->length;
    bool moved = step(game, snake_direction(game->direction, action));
    bool won = moved && game->length == CAPACITY;
    bool terminated = !moved || won;
    double reward = !moved ? -10 : (game->length > length ? 10 : -0.01);
    if (won) {
        reward += 100;
    }
    /* 势函数塑形；真正终止后势为零，截断不改变这一规则。 */
    reward += 0.95 * (terminated ? 0 : snake_potential(game)) - previous;
    return (SnakeResult){reward, terminated, won};
}

#endif
