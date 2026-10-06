/** @file
 * @brief 人工与 AI 共用的贪吃蛇规则。
 */
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

/**
 * @brief 更新独立随机状态并生成伪随机整数。
 * @param state 有效的可读写随机状态地址，零状态会映射到非零状态。
 * @return 生成的 32 位无符号整数。
 * @note 每个环境和训练器拥有独立的随机流；零种子映射到固定非零状态。
 */
static inline uint32_t snake_random(uint32_t *state)
{
    uint32_t x = *state == 0 ? UINT32_C(0x9e3779b9) : *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * @brief 生成用于随机探索的均匀伪随机数。
 * @param state 有效的可读写随机状态地址。
 * @return [0, 1) 范围内的伪随机数。
 * @note 生成 [0,1) 内的数，供训练器判断是否随机探索。
 */
static inline double snake_uniform(uint32_t *state)
{
    return (double)snake_random(state) / 4294967296.0;
}

/**
 * @brief 按坐标值判断两点是否相同。
 * @param a 第一个坐标。
 * @param b 第二个坐标。
 * @return 坐标相同返回 true。
 * @note 比较坐标值，不比较结构体的内存字节（其中可能有填充）。
 */
static inline bool equal(Point a, Point b)
{
    return a.x == b.x && a.y == b.y;
}

/**
 * @brief 检查指定格子是否被身体前若干节占据。
 * @param game 有效的只读游戏状态。
 * @param point 待检查的棋盘坐标。
 * @param count 待扫描的身体节数，须不大于 game->length。
 * @return 被扫描范围内的身体占据时返回 true。
 * @note 只扫描前 count 节身体，碰撞检测可借此排除即将离开的尾节。
 */
static inline bool occupied(const Game *game, Point point, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (equal(game->body[i], point)) {
            return true;
        }
    }
    return false;
}

/**
 * @brief 在未被蛇身占据的格子中生成食物。
 * @param game 有效的可读写游戏状态，同时更新 food 和 rng；棋盘已满时不修改。
 * @note 随机选择第几个空格，再扫描定位；不会把食物放在蛇身上。
 */
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

/**
 * @brief 创建初始长度为三、方向向右的新局。
 * @param seed 环境随机种子，相同种子产生相同初始食物。
 * @return 按值返回初始化后的游戏状态。
 * @note 固定初始蛇形与方向；相同种子产生相同的初始食物。
 */
static inline Game snake_reset(uint32_t seed)
{
    Game game = {.body = {{10, 5}, {9, 5}, {8, 5}}, .length = 3,
                 .direction = {1, 0}, .rng = seed};
    place_food(&game);
    return game;
}

/**
 * @brief 预测沿指定方向移动是否发生碰撞。
 * @param game 未满棋盘的有效只读状态。
 * @param direction 上下左右之一的单位方向。
 * @return 下一步撞墙或身体时返回 true；不修改游戏。
 * @note 只预测下一步是否碰撞，不移动蛇；direction 应为单位方向。
 */
static inline bool snake_collision(const Game *game, Point direction)
{
    Point head = {game->body[0].x + direction.x, game->body[0].y + direction.y};
    bool eating = equal(head, game->food);
    return head.x < 0 || head.x >= WIDTH || head.y < 0 || head.y >= HEIGHT ||
           occupied(game, head, game->length - (eating ? 0u : 1u));
}

/**
 * @brief 执行一次移动，吃食物时增长身体。
 * @param game 有效的可读写游戏状态。
 * @param direction 请求的单位方向；反向请求会沿原方向前进。
 * @return 移动成功返回 true；碰撞或棋盘已满返回 false 且状态不变。
 * @note 不吃食物时旧尾格可以进入；碰撞不修改环境；满棋盘后不可继续调用。
 */
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

/**
 * @brief 将相对动作转换为绝对方向。
 * @param current 当前单位方向。
 * @param action 0 直行、1 左转、2 右转；其他值按直行处理。
 * @return 执行动作后的单位方向。
 * @note 0 直行、1 左转、2 右转；屏幕的 y 轴向下。
 */
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

/**
 * @brief 将游戏状态编码为简化的二值特征。
 * @param game 未满棋盘的有效只读状态。
 * @param state 可写输出数组，至少容纳 FEATURES 个 double。
 * @note 11 个 0/1 特征：三方向危险 + 当前四方向 + 食物四方位。
 *       这是简化观测，无法表示完整蛇身布局。
 */
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

/**
 * @brief 计算基于蛇头与食物距离的奖励势函数。
 * @param game 有效的只读游戏状态。
 * @return 负的归一化曼哈顿距离，越接近食物越接近零。
 * @note 负的归一化曼哈顿距离：离食物越近，势越接近零。
 */
static inline double snake_potential(const Game *game)
{
    int distance = abs(game->food.x - game->body[0].x) + abs(game->food.y - game->body[0].y);
    return -(double)distance / (WIDTH + HEIGHT);
}

/* terminated 只表示死亡或获胜；时间上限由训练器单独处理。 */
typedef struct { double reward; bool terminated; bool won; } SnakeResult;

/**
 * @brief 执行相对动作并计算奖励和终止状态。
 * @param game 未终止的可读写游戏状态。
 * @param action 合法动作编号 0..ACTIONS-1。
 * @return 本步奖励、是否终止和是否获胜；时间上限截断由调用者处理。
 * @note 将合法相对动作（0..2）转成方向，执行一步并计算基础奖励与塑形奖励。
 */
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
