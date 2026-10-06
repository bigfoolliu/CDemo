/** @file
 * @brief 人工实时贪吃蛇与规则自测。
 */
#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "snake_terminal.h"

/**
 * @brief 验证人工贪吃蛇的规则、按键解析和内存得分统计。
 */
static void self_test(void)
{
    Game game = {.body = {{3, 2}, {2, 2}, {1, 2}}, .length = 3,
                 .direction = {1, 0}, .food = {4, 2}};
    bool moved = step(&game, (Point){1, 0});
    assert(moved && game.length == 4 && equal(game.body[0], (Point){4, 2}));
    assert(!occupied(&game, game.food, game.length));
    game.food = (Point){0, 0};
    moved = step(&game, (Point){-1, 0});
    assert(moved && game.body[0].x == 5); /* 掉头被忽略。 */
    game.body[0] = (Point){WIDTH - 1, 2};
    moved = step(&game, (Point){1, 0});
    assert(!moved && game.body[0].x == WIDTH - 1);
    game = (Game){.body = {{2, 2}, {2, 3}, {3, 3}, {3, 2}}, .length = 4,
                  .direction = {0, -1}, .food = {0, 0}};
    moved = step(&game, (Point){1, 0});
    assert(moved && equal(game.body[0], (Point){3, 2})); /* 进入旧尾格。 */
    game = (Game){.body = {{2, 2}, {2, 3}, {3, 3}, {3, 2}, {4, 2}}, .length = 5,
                  .direction = {0, -1}, .food = {0, 0}};
    moved = step(&game, (Point){1, 0});
    assert(!moved); /* 撞到非尾部身体。 */
    game = new_game();
    game.food = (Point){0, 0};
    assert(game.length == 3 && equal(game.direction, (Point){1, 0}));
    moved = step(&game, game.direction);
    assert(moved && game.body[0].x == 11);
    moved = step(&game, game.direction);
    assert(moved && game.body[0].x == 12);
    Point pending = game.direction;
    bool changed = change_direction('w', game.direction, &pending);
    assert(changed && equal(pending, (Point){0, -1}));
    changed = change_direction('a', game.direction, &pending);
    assert(!changed);
    const unsigned char codes[] = {'A', 'B', 'C', 'D'};
    const int keys[] = {'w', 's', 'd', 'a'};
    for (size_t i = 0; i < 4; ++i) {
        int escape = 0;
        int key = decode_key(27, &escape);
        assert(key == 0);
        key = decode_key('[', &escape);
        assert(key == 0);
        key = decode_key(codes[i], &escape);
        assert(key == keys[i] && escape == 0);
    }
    Scores scores = {0};
    add_score(&scores, 3);
    add_score(&scores, 8);
    add_score(&scores, 5);
    assert(scores.best == 8 && scores.games == 3);
    assert(scores.recent[0] == 5 && scores.recent[1] == 8 && scores.recent[2] == 3);
    puts("贪吃蛇测试通过：持续移动、碰撞、重置、方向键解析及历史分数。");
}

/**
 * @brief 解析命令行并运行人工游戏或规则自测。
 * @param argc 命令行参数数量，包含程序名。
 * @param argv 参数字符串数组，可指定历史得分路径或 --test。
 * @return 正常完成返回 EXIT_SUCCESS；参数或运行错误返回 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        self_test();
        return EXIT_SUCCESS;
    }
    if (argc > 2) {
        fprintf(stderr, "用法：%s [历史得分文件路径 | --test]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *score_path = argc == 2 ? argv[1] : "snake_scores.log";
    Scores scores = {0};
    if (!load_scores(score_path, &scores)) {
        perror("读取历史得分失败");
        return EXIT_FAILURE;
    }
    if (!start_terminal()) {
        fputs("请在支持 ANSI 的交互终端中运行游戏。\n", stderr);
        return EXIT_FAILURE;
    }
    srand((unsigned int)time(nullptr));
    Game game = new_game();
    Point pending = game.direction;
    bool turned = false;
    bool restarting = false;
    int escape = 0;
    int result = EXIT_SUCCESS;
    const char *message = "每 200 毫秒自动前进一格；吃食物加一分。";
    double now = monotonic_time();
    double deadline = now + 0.2;
    if (now < 0) {
        result = EXIT_FAILURE;
        stopped = 1;
    }
    draw(&game, &scores, message);
    while (!stopped) {
        now = monotonic_time();
        if (now < 0) {
            result = EXIT_FAILURE;
            break;
        }
        if (now >= deadline) {
            if (restarting) {
                game = new_game();
                pending = game.direction;
                turned = false;
                restarting = false;
                escape = 0;
                message = "新一局开始！";
            } else {
                bool moved = step(&game, pending);
                turned = false;
                if (!moved || game.length == CAPACITY) {
                    if (!save_score(score_path, &scores, game.length - 3)) {
                        result = EXIT_FAILURE;
                        break;
                    }
                    restarting = true;
                    message = moved ? "填满棋盘，获胜！1 秒后自动重开。" :
                                      "撞墙或撞到自己，死亡！1 秒后自动重开。";
                }
            }
            deadline = now + (restarting ? 1.0 : 0.2);
            draw(&game, &scores, message);
            continue;
        }
        double wait = deadline - now;
        struct timeval timeout = {.tv_sec = (time_t)wait,
            .tv_usec = (suseconds_t)((wait - (double)(time_t)wait) * 1e6)};
        fd_set input;
        FD_ZERO(&input);
        FD_SET(STDIN_FILENO, &input);
        int ready = select(STDIN_FILENO + 1, &input, nullptr, nullptr, &timeout);
        if (ready < 0) {
            if (errno == EINTR) {
                continue;
            }
            result = EXIT_FAILURE;
            break;
        }
        if (ready > 0) {
            unsigned char bytes[64];
            ssize_t count = read(STDIN_FILENO, bytes, sizeof(bytes));
            if (count < 0 && errno == EINTR) {
                continue;
            }
            if (count <= 0) {
                if (count < 0) {
                    result = EXIT_FAILURE;
                }
                break;
            }
            for (ssize_t i = 0; i < count; ++i) {
                int key = decode_key(bytes[i], &escape);
                if (key == 'q' || key == 'Q' || key == 4) {
                    stopped = 1;
                    break;
                }
                /* 每个移动周期仅一次转向，防止快速连按造成隐式掉头。 */
                if (!restarting && !turned) {
                    turned = change_direction(key, game.direction, &pending);
                }
            }
        }
    }
    if (result == EXIT_SUCCESS && !restarting &&
        !save_score(score_path, &scores, game.length - 3)) {
        result = EXIT_FAILURE;
    }
    restore_terminal();
    if (result != EXIT_SUCCESS) {
        fputs("游戏运行或得分文件写入失败，请检查终端和文件路径。\n", stderr);
    }
    printf("游戏已退出。历史最高：%zu；得分记录：%s\n", scores.best, score_path);
    return result;
}
