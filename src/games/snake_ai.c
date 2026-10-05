#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif

#include "snake_ai_network.h"
#include "snake_terminal.h"
#include "../common/logger.h"
#include <sys/stat.h>

enum { REPLAY_CAPACITY = 10000, BATCH = 32, MAX_STEPS = 4000 };
typedef struct {
    double state[FEATURES];
    double next[FEATURES];
    double reward;
    int action;
    bool terminated;
} Transition;
typedef struct {
    Transition data[REPLAY_CAPACITY];
    size_t count;
    size_t cursor;
} Replay;
typedef struct {
    Network online;
    Network target;
    Adam adam;
    Replay replay;
    uint32_t actions_rng;
    uint32_t sample_rng;
    uint32_t environment_rng;
} Learner;
typedef struct {
    double mean;
    size_t maximum;
    size_t truncated;
    size_t won;
    size_t games;
} Evaluation;
typedef struct {
    double seconds;
    uint32_t seed;
    size_t episodes;
    size_t steps;
    const char *model;
} Options;

static void replay_push(Replay *replay, const Transition *transition)
{
    replay->data[replay->cursor] = *transition;
    replay->cursor = (replay->cursor + 1) % REPLAY_CAPACITY;
    if (replay->count < REPLAY_CAPACITY) {
        ++replay->count;
    }
}

static double td_target(const Transition *transition, const Network *target)
{
    if (transition->terminated) {
        return transition->reward;
    }
    Forward next;
    network_forward(target, transition->next, &next);
    double maximum = next.q[0];
    for (int action = 1; action < ACTIONS; ++action) {
        maximum = fmax(maximum, next.q[action]);
    }
    /* 4000 步的时间上限是截断，不是死亡，仍然进行 bootstrap。 */
    return transition->reward + 0.95 * maximum;
}

static bool learn_batch(Learner *learner, double *loss)
{
    Network gradient = {0};
    double total = 0;
    for (int i = 0; i < BATCH; ++i) {
        size_t index = snake_random(&learner->sample_rng) % learner->replay.count;
        const Transition *transition = &learner->replay.data[index];
        Forward cache;
        network_forward(&learner->online, transition->state, &cache);
        total += network_gradient(&learner->online, &cache, transition->action,
                                  td_target(transition, &learner->target), &gradient);
    }
    *loss = total / BATCH;
    if (!isfinite(*loss) || !adam_update(&learner->online, &learner->adam, &gradient, BATCH)) {
        return false;
    }
    if (learner->adam.updates % 500 == 0) {
        learner->target = learner->online;
    }
    return true;
}

static double exploration(size_t steps)
{
    return 1 - 0.95 * fmin(1, (double)steps / 50000);
}

/* 每局单独的环境随机流，所有策略使用相同种子列表和步数上限。
 * 测试种子不用于训练或挑选模型；随机策略也不更改食物随机流。 */
static Evaluation evaluate(const Network *net, bool random_policy, uint32_t seed, size_t games)
{
    Evaluation result = {.games = games};
    double total = 0;
    for (size_t episode = 0; episode < games; ++episode) {
        uint32_t environment_seed = seed + (uint32_t)episode * UINT32_C(2654435761);
        uint32_t actions_rng = environment_seed ^ UINT32_C(0x83ac4f17);
        Game game = snake_reset(environment_seed);
        bool terminated = false;
        for (size_t i = 0; i < MAX_STEPS; ++i) {
            double state[FEATURES];
            snake_observe(&game, state);
            int action = random_policy ? (int)(snake_random(&actions_rng) % ACTIONS) :
                                         greedy_action(net, state);
            SnakeResult outcome = snake_act(&game, action);
            if (outcome.terminated) {
                terminated = true;
                result.won += outcome.won;
                break;
            }
        }
        size_t score = game.length - 3;
        total += (double)score;
        if (score > result.maximum) {
            result.maximum = score;
        }
        result.truncated += !terminated;
    }
    result.mean = total / (double)games;
    return result;
}

static void log_evaluation(Logger *logger, const char *label, Evaluation result)
{
    logger_write(logger, "%s：%zu 局，平均分=%.3f，最高分=%zu，截断=%.1f%%，满分=%zu\n",
                 label, result.games, result.mean, result.maximum,
                 100.0 * (double)result.truncated / (double)result.games, result.won);
}

static bool make_session(char directory[256])
{
    if (mkdir("sessions", 0755) != 0 && errno != EEXIST) {
        return false;
    }
    if (mkdir("sessions/snake-ai", 0755) != 0 && errno != EEXIST) {
        return false;
    }
    strcpy(directory, "sessions/snake-ai/run-XXXXXX");
    return mkdtemp(directory) != nullptr;
}

static bool artifact(char path[512], const char *directory, const char *name)
{
    int length = snprintf(path, 512, "%s/%s", directory, name);
    return length >= 0 && length < 512;
}

static bool write_config(const char *directory, const Options *options)
{
    char path[512];
    if (!artifact(path, directory, "config.txt")) {
        return false;
    }
    FILE *file = fopen(path, "wx");
    if (file == nullptr) {
        return false;
    }
    bool ok = fprintf(file,
        "seed=%u\nseconds=%.3f\nepisodes=%zu\nsteps_limit=%zu\n"
        "board=20x10\nnetwork=11,64,64,3\nfeatures_version=1\n"
        "replay=10000\nbatch=32\nwarmup=1000\nupdate_every=4\ntarget_every=500\n"
        "gamma=0.95\nlearning_rate=0.001\ngradient_clip=10\n"
        "epsilon=1.0->0.05 over 50000 steps\nmax_episode_steps=4000\n"
        "reward=food:10,death:-10,move:-0.01,win_bonus:100,potential:gamma*next-prev\n"
        "validation_seed=1000001\nvalidation_games=30\ntest_seed=2000001\ntest_games=100\n",
        options->seed, options->seconds, options->episodes, options->steps) >= 0;
    if (fclose(file) == EOF) {
        ok = false;
    }
    return ok;
}

static int train(const Options *options)
{
    char directory[256], log_path[512], csv_path[512], best_path[512], final_path[512];
    if (!make_session(directory) || !write_config(directory, options) ||
        !artifact(log_path, directory, "train.log") || !artifact(csv_path, directory, "metrics.csv") ||
        !artifact(best_path, directory, "best.model") || !artifact(final_path, directory, "final.model")) {
        perror("创建训练目录失败");
        return EXIT_FAILURE;
    }
    Logger logger;
    if (!logger_open(&logger, log_path, stdout)) {
        perror("打开训练日志失败");
        return EXIT_FAILURE;
    }
    FILE *csv = fopen(csv_path, "wx");
    Learner *learner = calloc(1, sizeof(*learner));
    if (csv == nullptr || learner == nullptr) {
        if (csv != nullptr) {
            fclose(csv);
        }
        free(learner);
        logger_close(&logger);
        return EXIT_FAILURE;
    }
    uint32_t init_rng = options->seed;
    network_init(&learner->online, &init_rng);
    Network initial = learner->online;
    learner->target = learner->online;
    learner->actions_rng = options->seed ^ UINT32_C(0x342de871);
    learner->sample_rng = options->seed ^ UINT32_C(0xcafef00d);
    learner->environment_rng = options->seed ^ UINT32_C(0x785f3391);
    Evaluation best = evaluate(&learner->online, false, 1000001, 30);
    bool ok = network_save(&learner->online, best_path);
    ok = fprintf(csv, "episode,steps,score,reward,epsilon,loss,updates,terminated,truncated,elapsed\n") >= 0 && ok;
    logger_write(&logger, "纯 C DQN：11→64→64→3，训练预算 %.1f 秒，种子 %u\n运行目录：%s\n",
                 options->seconds, options->seed, directory);
    log_evaluation(&logger, "初始验证", best);
    double start = monotonic_time();
    size_t total_steps = 0;
    size_t episodes = 0;
    size_t training_best = 0;
    double last_loss = 0;
    bool interrupted = false;
    if (start < 0) {
        ok = false;
    }
    while (ok && !stopped && episodes < options->episodes && total_steps < options->steps) {
        Game game = snake_reset(snake_random(&learner->environment_rng));
        double total_reward = 0;
        bool terminated = false;
        size_t episode_steps = 0;
        while (episode_steps < MAX_STEPS && total_steps < options->steps && !stopped) {
            double now = monotonic_time();
            if (now < 0) {
                ok = false;
                break;
            }
            if (now - start >= options->seconds) {
                interrupted = true;
                break;
            }
            Transition transition = {0};
            snake_observe(&game, transition.state);
            double epsilon = exploration(total_steps);
            transition.action = snake_uniform(&learner->actions_rng) < epsilon ?
                (int)(snake_random(&learner->actions_rng) % ACTIONS) :
                greedy_action(&learner->online, transition.state);
            SnakeResult outcome = snake_act(&game, transition.action);
            transition.reward = outcome.reward;
            transition.terminated = outcome.terminated;
            snake_observe(&game, transition.next);
            replay_push(&learner->replay, &transition);
            total_reward += outcome.reward;
            ++total_steps;
            ++episode_steps;
            if (total_steps >= 1000 && total_steps % 4 == 0) {
                if (!learn_batch(learner, &last_loss)) {
                    ok = false;
                    break;
                }
            }
            if (outcome.terminated) {
                terminated = true;
                break;
            }
        }
        bool truncated = !terminated && episode_steps == MAX_STEPS;
        bool partial = !terminated && !truncated;
        if (episode_steps == 0 || !ok) {
            break;
        }
        ++episodes;
        size_t score = game.length - 3;
        if (score > training_best) {
            training_best = score;
        }
        double elapsed = monotonic_time() - start;
        ok = fprintf(csv, "%zu,%zu,%zu,%.9g,%.9g,%.9g,%zu,%d,%d,%.3f\n",
            episodes, episode_steps, score, total_reward, exploration(total_steps), last_loss,
            learner->adam.updates, terminated, truncated, elapsed) >= 0 && fflush(csv) == 0;
        logger.console = episodes % 20 == 0 ? stdout : nullptr;
        logger_write(&logger, "局=%zu 步=%zu 分=%zu 训练最高=%zu ε=%.3f 损失=%.5f 更新=%zu 时间=%.1fs%s\n",
            episodes, total_steps, score, training_best, exploration(total_steps), last_loss,
            learner->adam.updates, elapsed, partial ? "（预算/信号截断的未完成局）" : "");
        logger.console = stdout;
        ok = ok && !logger.failed;
        if (ok && episodes % 200 == 0 && !partial && !stopped) {
            Evaluation validation = evaluate(&learner->online, false, 1000001, 30);
            log_evaluation(&logger, "验证集", validation);
            if (validation.mean > best.mean) {
                best = validation;
                ok = network_save(&learner->online, best_path);
                logger_write(&logger, "已更新最佳模型：%s\n", best_path);
            }
        }
        if (interrupted || partial) {
            break;
        }
    }
    if (ok) {
        Evaluation validation = evaluate(&learner->online, false, 1000001, 30);
        log_evaluation(&logger, "最终验证集", validation);
        if (validation.mean > best.mean) {
            ok = network_save(&learner->online, best_path);
        }
        ok = network_save(&learner->online, final_path) && ok;
    }
    Network selected;
    if (ok && network_load(&selected, best_path)) {
        char report_path[512];
        ok = artifact(report_path, directory, "report.txt");
        Logger report;
        if (ok && logger_open(&report, report_path, stdout)) {
            logger_write(&report, "训练完成：%zu 局，%zu 环境步，%zu 更新，训练中最高分=%zu\n",
                         episodes, total_steps, learner->adam.updates, training_best);
            log_evaluation(&report, "随机策略（独立测试集）", evaluate(&initial, true, 2000001, 100));
            log_evaluation(&report, "未训练网络（独立测试集）", evaluate(&initial, false, 2000001, 100));
            log_evaluation(&report, "最佳模型（独立测试集）", evaluate(&selected, false, 2000001, 100));
            logger_write(&report, "最佳模型：%s\n最终模型：%s\n训练与评估均未使用人工得分文件。\n",
                         best_path, final_path);
            ok = logger_close(&report);
        } else {
            ok = false;
        }
    } else {
        ok = false;
    }
    if (fclose(csv) == EOF) {
        ok = false;
    }
    if (!logger_close(&logger)) {
        ok = false;
    }
    free(learner);
    if (!ok) {
        fputs("训练失败：请检查数值、磁盘或文件权限；已有运行记录仍保留。\n", stderr);
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

static int play(const Network *net, uint32_t seed)
{
    char directory[256], path[512];
    if (!make_session(directory) || !artifact(path, directory, "ai_scores.log") || !start_terminal()) {
        fputs("无法启动 AI 演示，请在交互终端中运行。\n", stderr);
        return EXIT_FAILURE;
    }
    Scores scores = {0};
    uint32_t rng = seed;
    Game game = snake_reset(snake_random(&rng));
    bool restarting = false;
    size_t steps = 0;
    int result = EXIT_SUCCESS;
    const char *message = "AI 自动游戏：动作由网络直接选择，无自动避险；Q / Ctrl+C 退出。";
    double deadline = monotonic_time() + 0.2;
    draw_mode(&game, &scores, message, true);
    while (!stopped) {
        double now = monotonic_time();
        if (now < 0) {
            result = EXIT_FAILURE;
            break;
        }
        if (now >= deadline) {
            if (restarting) {
                game = snake_reset(snake_random(&rng));
                steps = 0;
                restarting = false;
                message = "AI 新一局开始。Q / Ctrl+C 退出。";
            } else {
                double state[FEATURES];
                snake_observe(&game, state);
                SnakeResult outcome = snake_act(&game, greedy_action(net, state));
                ++steps;
                if (outcome.terminated || steps == MAX_STEPS) {
                    if (!save_score(path, &scores, game.length - 3)) {
                        result = EXIT_FAILURE;
                        break;
                    }
                    restarting = true;
                    message = outcome.won ? "AI 满分！一秒后重开。" :
                        (outcome.terminated ? "AI 死亡，一秒后重开。" : "达到步数上限，一秒后重开。");
                }
            }
            deadline = now + (restarting ? 1.0 : 0.2);
            draw_mode(&game, &scores, message, true);
            continue;
        }
        double wait = deadline - now;
        struct timeval timeout = {.tv_sec = (time_t)wait,
            .tv_usec = (suseconds_t)((wait - (double)(time_t)wait) * 1e6)};
        fd_set input;
        FD_ZERO(&input);
        FD_SET(STDIN_FILENO, &input);
        int ready = select(STDIN_FILENO + 1, &input, nullptr, nullptr, &timeout);
        if (ready < 0 && errno != EINTR) {
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
                result = count < 0 ? EXIT_FAILURE : EXIT_SUCCESS;
                break;
            }
            for (ssize_t i = 0; i < count; ++i) {
                if (bytes[i] == 'q' || bytes[i] == 'Q' || bytes[i] == 4) {
                    stopped = 1;
                }
            }
        }
    }
    if (result == EXIT_SUCCESS && !restarting && !save_score(path, &scores, game.length - 3)) {
        result = EXIT_FAILURE;
    }
    restore_terminal();
    printf("AI 演示结束，记录：%s\n", path);
    return result;
}

static bool parse_number(const char *text, unsigned long *value)
{
    if (*text < '0' || *text > '9') {
        return false;
    }
    char *end;
    errno = 0;
    *value = strtoul(text, &end, 10);
    return errno == 0 && *end == '\0';
}

int main(int argc, char *argv[])
{
    Options options = {.seconds = 180, .seed = 42, .episodes = 20000, .steps = SIZE_MAX};
    if (argc < 2 || (strcmp(argv[1], "train") != 0 && strcmp(argv[1], "eval") != 0 &&
                     strcmp(argv[1], "play") != 0)) {
        fputs("用法：snake_ai train [--seconds 180] [--seed 42] [--episodes 20000] [--steps N]\n"
              "      snake_ai eval|play --model <模型路径> [--seed 42]\n", stderr);
        return EXIT_FAILURE;
    }
    bool training = strcmp(argv[1], "train") == 0;
    if (strcmp(argv[1], "eval") == 0) {
        options.seed = 2000001;
    }
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) {
            fputs("参数缺少值。\n", stderr);
            return EXIT_FAILURE;
        }
        if (!training && strcmp(argv[i], "--model") == 0) {
            options.model = argv[i + 1];
            continue;
        }
        unsigned long value;
        if (!parse_number(argv[i + 1], &value)) {
            fputs("数值参数必须是非负整数。\n", stderr);
            return EXIT_FAILURE;
        }
        if (strcmp(argv[i], "--seed") == 0 && value <= UINT32_MAX) {
            options.seed = (uint32_t)value;
        } else if (training && strcmp(argv[i], "--seconds") == 0 && value > 0 && value <= 86400) {
            options.seconds = (double)value;
        } else if (training && strcmp(argv[i], "--episodes") == 0 && value > 0) {
            options.episodes = value;
        } else if (training && strcmp(argv[i], "--steps") == 0 && value > 0) {
            options.steps = value;
        } else {
            fprintf(stderr, "未知或越界参数：%s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    if (training) {
        struct sigaction action = {0};
        action.sa_handler = stop_game;
        sigemptyset(&action.sa_mask);
        if (sigaction(SIGINT, &action, nullptr) != 0 || sigaction(SIGTERM, &action, nullptr) != 0) {
            return EXIT_FAILURE;
        }
        return train(&options);
    }
    Network net;
    if (options.model == nullptr || !network_load(&net, options.model)) {
        fputs("请提供有效模型：--model <路径>；模型版本、尺寸和数值必须匹配。\n", stderr);
        return EXIT_FAILURE;
    }
    if (strcmp(argv[1], "play") == 0) {
        return play(&net, options.seed);
    }
    Evaluation result = evaluate(&net, false, options.seed, 100);
    Logger logger = {.console = stdout};
    log_evaluation(&logger, "模型评估", result);
    return logger.failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
