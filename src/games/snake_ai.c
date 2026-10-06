/** @file
 * @brief 纯 C DQN 训练、评估与终端演示。
 */
/* 阅读顺序：main → train → learn_batch → 网络头文件；play 展示训练后的策略。
 * train 学习参数，eval 独立测分，play 只推理、不更新网络。 */
#define _POSIX_C_SOURCE 200809L
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif

#include "snake_ai_network.h"
#include "snake_terminal.h"
#include "../common/logger.h"
#include <sys/stat.h>

enum { REPLAY_CAPACITY = 10000, BATCH = 32, MAX_STEPS = 4000 };
/* 一条经验记录一次动作前后的观测；terminated 不包含时间上限截断。 */
typedef struct {
    double state[FEATURES];
    double next[FEATURES];
    double reward;
    int action;
    bool terminated;
} Transition;
/* 环形经验池：count 是有效条数，cursor 是下次写入位置。 */
typedef struct {
    Transition data[REPLAY_CAPACITY];
    size_t count;
    size_t cursor;
} Replay;
/* online 参与学习，target 提供稳定目标；三条随机流互不干扰。 */
typedef struct {
    Network online;
    Network target;
    Adam adam;
    Replay replay;
    uint32_t actions_rng;
    uint32_t sample_rng;
    uint32_t environment_rng;
} Learner;
/* 评分统计；平均分用于比较策略，最高分只表示本次样本中的最好结果。 */
typedef struct {
    double mean;
    size_t maximum;
    size_t truncated;
    size_t won;
    size_t games;
} Evaluation;
/* 命令行配置；训练达到任一预算即停止，model 是借用的参数字符串。 */
typedef struct {
    double seconds;
    uint32_t seed;
    size_t episodes;
    size_t steps;
    const char *model;
} Options;

/**
 * @brief 向环形经验池追加一条经验。
 * @param replay 已初始化的可读写经验池。
 * @param transition 只读经验，内容按值复制，不保留指针。
 * @note 满容量后覆盖最旧经验，不再分配内存。
 */
static void replay_push(Replay *replay, const Transition *transition)
{
    replay->data[replay->cursor] = *transition;
    replay->cursor = (replay->cursor + 1) % REPLAY_CAPACITY;
    if (replay->count < REPLAY_CAPACITY) {
        ++replay->count;
    }
}

/**
 * @brief 计算含终止状态处理的贝尔曼目标。
 * @param transition 待学习的只读经验。
 * @param target 只读目标网络；经验未终止时用于预测下一状态。
 * @return 终止时仅返回奖励，否则加上折扣后的最大下一状态 Q 值。
 * @note 贝尔曼目标：即时奖励 + 折扣后的下一状态最大 Q 值。
 */
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

/**
 * @brief 抽样一个批次并更新在线网络和目标网络。
 * @param learner 已初始化的学习器，经验池必须非空。
 * @param loss 有效的平均损失输出地址；数值失败时也可能已写入。
 * @return 学习成功返回 true；非有限损失或更新失败返回 false。
 * @note 有效经验池非空时调用：抽样 → 前向 → 累积梯度 → Adam 更新。
 */
static bool learn_batch(Learner *learner, double *loss)
{
    Network gradient = {0};
    double total = 0;
    for (int i = 0; i < BATCH; ++i) {
        /* 有放回抽样：一个批次可重复选到同一条经验。 */
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
        /* 同步周期按参数更新次数计数，而不是环境步数。 */
        learner->target = learner->online;
    }
    return true;
}

/**
 * @brief 计算随训练步数线性衰减的探索概率。
 * @param steps 已执行的环境步数。
 * @return 从 1.0 降至 0.05 的随机动作概率。
 * @note ε 是随机动作概率：前 50000 步线性下降，此后保持 0.05。
 */
static double exploration(size_t steps)
{
    return 1 - 0.95 * fmin(1, (double)steps / 50000);
}

/**
 * @brief 使用固定种子序列评估网络或随机策略。
 * @param net 只读网络；random_policy 为 true 时不使用，可为空。
 * @param random_policy 是否采用随机动作而不是网络策略。
 * @param seed 评估环境的基础种子。
 * @param games 评估局数，须大于零。
 * @return 平均分、最高分、截断数及获胜数等统计。
 * @note 每局单独的环境随机流，所有策略使用相同种子列表和步数上限。
 *       测试种子不用于训练或挑选模型；随机策略也不更改食物随机流。
 */
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

/**
 * @brief 将评估统计写入日志摘要。
 * @param logger 已初始化的可读写日志记录器。
 * @param label 有效的 NUL 结尾摘要名称。
 * @param result 按值传入的评估结果，games 须大于零。
 * @note 将评估统计转换为可读摘要，不参与训练。
 */
static void log_evaluation(Logger *logger, const char *label, Evaluation result)
{
    logger_write(logger, "%s：%zu 局，平均分=%.3f，最高分=%zu，截断=%.1f%%，满分=%zu\n",
                 label, result.games, result.mean, result.maximum,
                 100.0 * (double)result.truncated / (double)result.games, result.won);
}

/**
 * @brief 为一次实验创建独立产物目录。
 * @param directory 可写路径缓冲区，至少 256 字节；失败时内容不可作为有效路径。
 * @return 目录创建成功返回 true；失败返回 false。
 * @note 创建本次运行的独立产物目录，避免覆盖已有实验。
 */
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

/**
 * @brief 将产物目录和文件名拼接为路径。
 * @param path 可写输出缓冲区，至少 512 字节；失败时可能含截断内容。
 * @param directory 有效的 NUL 结尾目录路径。
 * @param name 有效的 NUL 结尾产物文件名。
 * @return 路径完整写入返回 true；格式化失败或截断返回 false。
 * @note 拼接路径并检查是否截断；path 必须至少容纳 512 字节。
 */
static bool artifact(char path[512], const char *directory, const char *name)
{
    int length = snprintf(path, 512, "%s/%s", directory, name);
    return length >= 0 && length < 512;
}

/**
 * @brief 保存训练预算与超参数供实验复现。
 * @param directory 已存在的实验目录路径。
 * @param options 有效的只读训练选项。
 * @return 创建、写入并关闭成功返回 true；失败返回 false。
 * @note 保存预算与超参数，方便将训练结果与配置对应起来。
 */
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

/**
 * @brief 执行 DQN 采样、学习、选模和独立测试。
 * @param options 已校验的只读训练预算和随机种子配置。
 * @return 训练及产物写入成功返回 EXIT_SUCCESS，否则返回 EXIT_FAILURE。
 * @note 训练主流程：初始化 → 采样学习 → 验证选模 → 独立测试 → 释放资源。
 */
static int train(const Options *options)
{
    char directory[256], log_path[512], csv_path[512], best_path[512], final_path[512];
    char log_directory[512];
    if (!make_session(directory)) {
        perror("创建训练目录失败");
        return EXIT_FAILURE;
    }
    /* 模型仍在 sessions/，日志按相同运行编号放到 logs/。 */
    snprintf(log_directory, sizeof(log_directory), "logs/snake-ai/%s", strrchr(directory, '/') + 1);
    if (!write_config(directory, options) ||
        !artifact(log_path, log_directory, "train.log") || !artifact(csv_path, directory, "metrics.csv") ||
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
    /* 经验池较大，放在堆上；calloc 同时清零 Adam 和环形池状态。 */
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
    Network initial = learner->online; /* 留下未训练副本用于最终对照。 */
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
    /* 初始验证不计入预算；训练期间的周期验证计入预算。 */
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
            /* ε-greedy：有时随机探索，否则选择当前网络最看好的动作。 */
            transition.action = snake_uniform(&learner->actions_rng) < epsilon ?
                (int)(snake_random(&learner->actions_rng) % ACTIONS) :
                greedy_action(&learner->online, transition.state);
            /* 环境执行动作并给奖励；下一状态必须在动作执行后编码。 */
            SnakeResult outcome = snake_act(&game, transition.action);
            transition.reward = outcome.reward;
            transition.terminated = outcome.terminated;
            snake_observe(&game, transition.next);
            replay_push(&learner->replay, &transition);
            total_reward += outcome.reward;
            ++total_steps;
            ++episode_steps;
            /* 先收集 1000 步经验，再每四步训练一次，减少相邻样本相关性。 */
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
        bool partial = !terminated && !truncated; /* 总预算或信号导致的未完成局。 */
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
        /* 文件记录每局，终端每 20 局打印，兼顾完整性与可读性。 */
        logger.console = episodes % 20 == 0 ? stdout : nullptr;
        logger_write(&logger, "局=%zu 步=%zu 分=%zu 训练最高=%zu ε=%.3f 损失=%.5f 更新=%zu 时间=%.1fs%s\n",
            episodes, total_steps, score, training_best, exploration(total_steps), last_loss,
            learner->adam.updates, elapsed, partial ? "（预算/信号截断的未完成局）" : "");
        logger.console = stdout;
        ok = ok && !logger.failed;
        if (ok && episodes % 200 == 0 && !partial && !stopped) {
            Evaluation validation = evaluate(&learner->online, false, 1000001, 30);
            log_evaluation(&logger, "验证集", validation);
            /* 仅按验证平均分选模；不根据训练最高分或测试集选模。 */
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
    /* 最佳模型可能不是最后的参数；三种策略使用同一独立测试种子列表。 */
    Network selected;
    if (ok && network_load(&selected, best_path)) {
        char report_path[512];
        ok = artifact(report_path, log_directory, "report.txt");
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

/**
 * @brief 使用已训练网络在终端自动玩贪吃蛇。
 * @param net 已加载的只读网络，演示时不训练。
 * @param seed 演示环境的初始随机种子。
 * @return 正常结束返回 EXIT_SUCCESS；终端或文件操作失败返回 EXIT_FAILURE。
 * @note 演示不训练：每个时钟周期预测动作，终止后记分并自动重开。
 */
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
        /* 输入不会重置 deadline，因此按键不会改变蛇的移动速度。 */
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
    /* 等待重开的局已记录；主动退出只补记仍在进行的一局。 */
    if (result == EXIT_SUCCESS && !restarting && !save_score(path, &scores, game.length - 3)) {
        result = EXIT_FAILURE;
    }
    restore_terminal();
    printf("AI 演示结束，记录：%s\n", path);
    return result;
}

/**
 * @brief 解析完整的非负十进制无符号整数。
 * @param text 有效的 NUL 结尾数值文本，不接受符号或空白。
 * @param value 有效的输出地址；转换失败时也可能被修改。
 * @return 文本完整且未超出 unsigned long 范围返回 true，否则返回 false。
 * @note 要求整个字符串都是非负十进制数，并检查转换溢出。
 */
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

/**
 * @brief 校验命令行并分派训练、评估或演示。
 * @param argc 命令行参数数量，包含程序名。
 * @param argv 参数字符串数组，包含 train/eval/play 子命令及对应选项。
 * @return 正常完成返回 EXIT_SUCCESS；参数或运行错误返回 EXIT_FAILURE。
 * @note 参数按“选项 值”解析；先检查输入，再分派到训练、评估或演示。
 */
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
