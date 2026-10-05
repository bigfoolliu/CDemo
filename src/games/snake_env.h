#ifndef CDEMO_SNAKE_ENV_H
#define CDEMO_SNAKE_ENV_H

#include <stdint.h>
#include <stdlib.h>

enum { WIDTH = 20, HEIGHT = 10, CAPACITY = WIDTH * HEIGHT, FEATURES = 11, ACTIONS = 3 };
typedef struct { int x; int y; } Point;
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

static inline double snake_uniform(uint32_t *state)
{
    return (double)snake_random(state) / 4294967296.0;
}

static inline bool equal(Point a, Point b)
{
    return a.x == b.x && a.y == b.y;
}

static inline bool occupied(const Game *game, Point point, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (equal(game->body[i], point)) {
            return true;
        }
    }
    return false;
}

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

static inline Game snake_reset(uint32_t seed)
{
    Game game = {.body = {{10, 5}, {9, 5}, {8, 5}}, .length = 3,
                 .direction = {1, 0}, .rng = seed};
    place_food(&game);
    return game;
}

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

static inline double snake_potential(const Game *game)
{
    int distance = abs(game->food.x - game->body[0].x) + abs(game->food.y - game->body[0].y);
    return -(double)distance / (WIDTH + HEIGHT);
}

typedef struct { double reward; bool terminated; bool won; } SnakeResult;

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
