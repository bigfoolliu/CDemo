#ifndef CDEMO_SNAKE_TERMINAL_H
#define CDEMO_SNAKE_TERMINAL_H

#include "snake_env.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    size_t best;
    size_t games;
    size_t recent[5];
} Scores;

static inline void add_score(Scores *scores, size_t score)
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

static inline bool load_scores(const char *path, Scores *scores)
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

static inline bool save_score(const char *path, Scores *scores, size_t score)
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

static inline void stop_game(int signal_number)
{
    (void)signal_number;
    stopped = 1;
}

static inline void restore_terminal(void)
{
    if (terminal_active) {
        (void)tcsetattr(STDIN_FILENO, TCSANOW, &saved_terminal);
        fputs("\033[?25h\033[?1049l", stdout);
        fflush(stdout);
        terminal_active = false;
    }
}

static inline bool start_terminal(void)
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

static inline Game new_game(void)
{
    return snake_reset((uint32_t)rand());
}

/* 转义序列可跨多次读取，状态必须保留；兼容 ESC [ 和 ESC O 方向键。 */
static inline int decode_key(unsigned char byte, int *escape)
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

static inline bool change_direction(int key, Point current, Point *pending)
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

static inline double monotonic_time(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

static inline void draw_mode(const Game *game, const Scores *scores, const char *message, bool ai)
{
    /* ANSI 转义序列清屏并回到左上角，常见终端均支持。 */
    fputs("\033[2J\033[H", stdout);
    printf("贪吃蛇 | 当前：%zu | 历史最高：%zu | 已记录：%zu 局\n",
           game->length - 3, scores->best, scores->games);
    fputs(ai ? "AI 控制方向，Q 或 Ctrl+C 退出\n最近得分：" :
               "方向键 / WASD 转向，Q 或 Ctrl+C 退出（无需回车）\n最近得分：", stdout);
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

static inline void draw(const Game *game, const Scores *scores, const char *message)
{
    draw_mode(game, scores, message, false);
}


#endif
