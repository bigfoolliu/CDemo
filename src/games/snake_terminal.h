/** @file
 * @brief POSIX 终端输入、渲染与得分接口。
 */
#ifndef CDEMO_SNAKE_TERMINAL_H
#define CDEMO_SNAKE_TERMINAL_H

/* POSIX 终端层：得分记录、即时输入和渲染；不参与模型参数学习。 */

#include "snake_env.h"
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/select.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

/* 最近五局按新到旧排列；完整历史保留在文件里。 */
typedef struct {
    size_t best;
    size_t games;
    size_t recent[5];
} Scores;

/**
 * @brief 更新内存中的最高分、局数和最近得分。
 * @param scores 已初始化的可读写统计对象。
 * @param score 本局得分。
 * @note 更新内存中的统计；不执行文件写入。
 */
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

/**
 * @brief 读取得分历史并追加到内存统计。
 * @param path 有效的 NUL 结尾历史文件路径。
 * @param scores 已初始化的可读写统计对象，通常先清零。
 * @return 读取成功或文件不存在返回 true；读取或关闭失败返回 false，统计可能已部分更新。
 * @note 文件不存在视为没有历史；跳过无效记录，其他读取错误返回失败。
 */
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

/**
 * @brief 追加保存一局得分，成功后更新内存统计。
 * @param path 有效的 NUL 结尾历史文件路径，父目录须已存在。
 * @param scores 可读写统计对象；写入失败时不更新。
 * @param score 待追加的得分。
 * @return 写入并关闭成功返回 true；失败返回 false，文件可能已部分写入。
 * @note 追加并关闭文件成功后才更新内存统计，避免显示未保存的得分。
 */
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
static volatile sig_atomic_t stopped; /* 信号处理器与主循环共用的退出标志。 */

/**
 * @brief 请求主循环退出，供 POSIX 信号处理器调用。
 * @param signal_number 收到的信号编号；本函数不区分具体信号。
 * @note 信号中只设置标志，恢复终端和保存文件留给主循环处理。
 */
static inline void stop_game(int signal_number)
{
    (void)signal_number;
    stopped = 1;
}

/**
 * @brief 恢复终端输入模式、光标和原屏幕。
 * @note 恢复输入模式、光标和原屏幕；可重复调用，也注册为退出回调。
 */
static inline void restore_terminal(void)
{
    if (terminal_active) {
        (void)tcsetattr(STDIN_FILENO, TCSANOW, &saved_terminal);
        fputs("\033[?25h\033[?1049l", stdout);
        fflush(stdout);
        terminal_active = false;
    }
}

/**
 * @brief 设置交互终端并注册退出恢复及信号处理。
 * @return 设置成功返回 true；非交互终端或 POSIX 调用失败返回 false。
 * @note 关闭逐行输入与回显，但保留 Ctrl+C 信号；备用屏幕避免污染原终端。
 */
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

/**
 * @brief 使用人工游戏的随机流创建新局。
 * @return 初始化后的游戏状态；调用者可先通过 srand 设置种子。
 * @note 人工模式借用 rand 选择种子；训练模式直接调用带种子的 snake_reset。
 */
static inline Game new_game(void)
{
    return snake_reset((uint32_t)rand());
}

/**
 * @brief 逐字节解析普通按键和方向键转义序列。
 * @param byte 本次读取的输入字节。
 * @param escape 可读写解析状态，首次置零并跨读取保留。
 * @return 普通字符或映射后的 WASD；序列未完成或无效时返回 0。
 * @note 转义序列可跨多次读取，状态必须保留；兼容 ESC [ 和 ESC O 方向键。
 */
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

/**
 * @brief 校验转向并更新待执行方向。
 * @param key WASD 按键编码，大小写均可。
 * @param current 本帧已执行的单位方向，用于拒绝反向操作。
 * @param pending 有效的方向输出地址，仅接受转向时修改。
 * @return 接受新方向返回 true；无效、同向或反向输入返回 false。
 * @note 转向成功才更新 pending；current 是本帧已执行方向，用于拒绝掉头。
 */
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

/**
 * @brief 读取不受系统时间调整影响的单调时钟。
 * @return 单调时钟秒数；读取失败返回 -1。
 * @note 返回单调时钟秒数，不受系统时间调整影响；失败返回 -1。
 */
static inline double monotonic_time(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return -1;
    }
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}

/**
 * @brief 在终端渲染棋盘、统计和操作提示。
 * @param game 有效的只读游戏状态。
 * @param scores 有效的只读得分统计。
 * @param message 有效的 NUL 结尾提示文本。
 * @param ai true 显示 AI 操作提示，false 显示人工操作提示。
 * @note 渲染只读取状态；ai 只改变操作提示，不影响游戏规则。
 */
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

/**
 * @brief 渲染人工游戏界面。
 * @param game 有效的只读游戏状态。
 * @param scores 有效的只读得分统计。
 * @param message 有效的 NUL 结尾提示文本。
 * @note 人工游戏使用的渲染入口。
 */
static inline void draw(const Game *game, const Scores *scores, const char *message)
{
    draw_mode(game, scores, message, false);
}


#endif
