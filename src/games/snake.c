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

enum { WIDTH = 20, HEIGHT = 10, CAPACITY = WIDTH * HEIGHT };
typedef struct { int x; int y; } Point;
typedef struct {
    Point body[CAPACITY];
    size_t length;
    Point direction;
    Point food;
} Game;

static bool equal(Point a, Point b)
{
    return a.x == b.x && a.y == b.y;
}

static bool occupied(const Game *game, Point point, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        if (equal(game->body[i], point)) {
            return true;
        }
    }
    return false;
}

static void place_food(Game *game)
{
    /* 从空格中选一个位置，避免蛇很长时反复随机重试。 */
    size_t choice = (size_t)rand() % (CAPACITY - game->length);
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

/* 一步移动：碰撞时不修改状态；不吃食物时允许进入即将离开的尾格。 */
static bool step(Game *game, Point direction)
{
    if (direction.x == -game->direction.x && direction.y == -game->direction.y) {
        direction = game->direction; /* 禁止直接掉头。 */
    }
    Point head = {game->body[0].x + direction.x, game->body[0].y + direction.y};
    bool eating = equal(head, game->food);
    if (head.x < 0 || head.x >= WIDTH || head.y < 0 || head.y >= HEIGHT ||
        occupied(game, head, game->length - (eating ? 0u : 1u))) {
        return false;
    }
    size_t length = game->length + (eating ? 1u : 0u);
    for (size_t i = length - 1; i > 0; --i) {
        game->body[i] = game->body[i - 1];
    }
    game->body[0] = head;
    game->length = length;
    game->direction = direction;
    if (eating && length < CAPACITY) {
        place_food(game);
    }
    return true;
}

typedef struct {
    size_t best;
    size_t games;
    size_t recent[5];
} Scores;

static void add_score(Scores *scores, size_t score)
{
    for (size_t i = 4; i > 0; --i) {
        scores->recent[i] = scores->recent[i - 1];
    }
    scores->recent[0] = score;
    ++scores->games;
    if (score > scores->best) {
        scores->best = score;
    }
}

static bool load_scores(const char *path, Scores *scores)
{
    FILE *file = fopen(path, "r");
    if (file == nullptr) {
        return errno == ENOENT;
    }
    char line[64];
    while (fgets(line, sizeof(line), file) != nullptr) {
        if (strchr(line, '\n') == nullptr && !feof(file)) {
            int c;
            while ((c = fgetc(file)) != '\n' && c != EOF) {}
            continue;
        }
        char *end;
        errno = 0;
        unsigned long score = strtoul(line, &end, 10);
        if (end == line || errno != 0 || score > CAPACITY - 3) {
            continue;
        }
        while (*end == '\n' || *end == '\r' || *end == ' ' || *end == '\t') {
            ++end;
        }
        if (*end == '\0') {
            add_score(scores, (size_t)score);
        }
    }
    bool success = !ferror(file);
    if (fclose(file) == EOF) {
        success = false;
    }
    return success;
}

static bool save_score(const char *path, Scores *scores, size_t score)
{
    FILE *file = fopen(path, "a");
    if (file == nullptr) {
        return false;
    }
    bool success = fprintf(file, "%zu\n", score) >= 0;
    if (fclose(file) == EOF) {
        success = false;
    }
    if (success) {
        add_score(scores, score);
    }
    return success;
}

static struct termios saved_terminal;
static bool terminal_active;
static volatile sig_atomic_t stopped;

static void stop_game(int signal_number)
{
    (void)signal_number;
    stopped = 1;
}

static void restore_terminal(void)
{
    if (terminal_active) {
        (void)tcsetattr(STDIN_FILENO, TCSANOW, &saved_terminal);
        fputs("\033[?25h\033[?1049l", stdout);
        fflush(stdout);
        terminal_active = false;
    }
}

static bool start_terminal(void)
{
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO) ||
        tcgetattr(STDIN_FILENO, &saved_terminal) != 0 || atexit(restore_terminal) != 0) {
        return false;
    }
    struct sigaction action = {0};
    action.sa_handler = stop_game;
    sigemptyset(&action.sa_mask);
    const int signals[] = {SIGINT, SIGTERM, SIGHUP, SIGQUIT, SIGTSTP};
    for (size_t i = 0; i < sizeof(signals) / sizeof(signals[0]); ++i) {
        if (sigaction(signals[i], &action, nullptr) != 0) {
            return false;
        }
    }
    struct termios raw = saved_terminal;
    raw.c_lflag &= (tcflag_t)~(ICANON | ECHO);
    raw.c_iflag &= (tcflag_t)~IXON;
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
        return false;
    }
    terminal_active = true;
    fputs("\033[?1049h\033[?25l", stdout);
    return true;
}

static Game new_game(void)
{
    Game game = {.body = {{10, 5}, {9, 5}, {8, 5}}, .length = 3, .direction = {1, 0}};
    place_food(&game);
    return game;
}

/* 转义序列可跨多次读取，状态必须保留；兼容 ESC [ 和 ESC O 方向键。 */
static int decode_key(unsigned char byte, int *escape)
{
    if (byte == 27) {
        *escape = 1;
        return 0;
    }
    if (*escape == 1) {
        *escape = (byte == '[' || byte == 'O') ? 2 : 0;
        return 0;
    }
    if (*escape == 2) {
        *escape = 0;
        switch (byte) {
        case 'A': return 'w';
        case 'B': return 's';
        case 'C': return 'd';
        case 'D': return 'a';
        default: return 0;
        }
    }
    return byte;
}

static bool change_direction(int key, Point current, Point *pending)
{
    Point next;
    switch (key) {
    case 'w': case 'W': next = (Point){0, -1}; break;
    case 's': case 'S': next = (Point){0, 1}; break;
    case 'a': case 'A': next = (Point){-1, 0}; break;
    case 'd': case 'D': next = (Point){1, 0}; break;
    default: return false;
    }
    if (equal(next, current) || (next.x == -current.x && next.y == -current.y)) {
        return false;
    }
    *pending = next;
    return true;
}

static double monotonic_time(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static void draw(const Game *game, const Scores *scores, const char *message)
{
    /* ANSI 转义序列清屏并回到左上角，常见终端均支持。 */
    fputs("\033[2J\033[H", stdout);
    printf("贪吃蛇 | 当前：%zu | 历史最高：%zu | 已记录：%zu 局\n",
           game->length - 3, scores->best, scores->games);
    fputs("方向键 / WASD 转向，Q 或 Ctrl+C 退出（无需回车）\n最近得分：", stdout);
    for (size_t i = 0; i < 5 && i < scores->games; ++i) {
        printf(" %zu", scores->recent[i]);
    }
    printf("\n%s\n", message);
    for (int y = -1; y <= HEIGHT; ++y) {
        for (int x = -1; x <= WIDTH; ++x) {
            Point point = {x, y};
            char cell = ' ';
            if (x == -1 || x == WIDTH || y == -1 || y == HEIGHT) {
                cell = '#';
            } else if (equal(point, game->body[0])) {
                cell = '@';
            } else if (occupied(game, point, game->length)) {
                cell = 'o';
            } else if (equal(point, game->food)) {
                cell = '*';
            }
            putchar(cell);
        }
        putchar('\n');
    }
    fflush(stdout);
}

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
