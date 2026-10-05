/* 编译在同一翻译单元中直接验证内部算法，训练与测试使用同一实现。 */
#define main snake_ai_program_main
#include "../src/games/snake_ai.c"
#undef main
#include <assert.h>

static void test_environment(void)
{
    Game a = snake_reset(42);
    Game b = snake_reset(42);
    assert(equal(a.food, b.food) && a.rng == b.rng);
    double state[FEATURES];
    snake_observe(&a, state);
    assert(state[0] == 0 && state[1] == 0 && state[2] == 0 && state[4] == 1);
    assert(equal(snake_direction(a.direction, 1), (Point){0, -1}));
    assert(equal(snake_direction(a.direction, 2), (Point){0, 1}));
    a.food = (Point){11, 5};
    size_t count = 1;
    for (int y = 0; y < HEIGHT; ++y) {
        for (int x = 0; x < WIDTH; ++x) {
            Point point = {x, y};
            if (!equal(point, a.body[0]) && !equal(point, a.food)) {
                a.body[count++] = point;
            }
        }
    }
    a.length = count;
    assert(count == CAPACITY - 1);
    SnakeResult result = snake_act(&a, 0);
    assert(result.terminated && result.won && a.length == CAPACITY && result.reward > 100);
    bool moved = step(&a, a.direction);
    assert(!moved);
}

static void test_gradient(void)
{
    uint32_t rng = 123;
    Network net;
    network_init(&net, &rng);
    double state[FEATURES];
    for (int i = 0; i < FEATURES; ++i) {
        state[i] = (double)(i + 1) / 13;
    }
    Forward cache;
    network_forward(&net, state, &cache);
    const double targets[] = {cache.q[1] + 0.3, cache.q[1] + 2};
    for (size_t t = 0; t < 2; ++t) {
        Network gradient = {0};
        double loss = network_gradient(&net, &cache, 1, targets[t], &gradient);
        assert(isfinite(loss));
        /* 所有 5123 个参数，覆盖 ReLU、全部层及 Huber 两个分支。 */
        for (int i = 0; i < PARAMETERS; ++i) {
            double original = net.p[i];
            net.p[i] = original + 1e-6;
            Forward plus;
            network_forward(&net, state, &plus);
            net.p[i] = original - 1e-6;
            Forward minus;
            network_forward(&net, state, &minus);
            net.p[i] = original;
            double numeric = (huber(plus.q[1] - targets[t]) - huber(minus.q[1] - targets[t])) / 2e-6;
            assert(fabs(numeric - gradient.p[i]) < 1e-5);
        }
    }
    Network gradient = {0};
    Adam adam = {0};
    Network zero = {0};
    gradient.p[B3] = 1;
    bool ok = adam_update(&zero, &adam, &gradient, 1);
    assert(ok && fabs(zero.p[B3] + 0.001) < 1e-8 && adam.updates == 1);
    gradient = (Network){0};
    for (int i = 0; i < PARAMETERS; ++i) {
        gradient.p[i] = 100;
    }
    adam = (Adam){0};
    zero = (Network){0};
    ok = adam_update(&zero, &adam, &gradient, 1);
    double moment_norm = 0;
    for (int i = 0; i < PARAMETERS; ++i) {
        moment_norm += adam.m[i] * adam.m[i];
    }
    assert(ok && fabs(sqrt(moment_norm) - 1) < 1e-8);
}

static void test_replay(void)
{
    Learner *learner = calloc(1, sizeof(*learner));
    assert(learner != nullptr);
    Transition transition = {.reward = 2, .terminated = true};
    learner->target.p[B3] = 7;
    double target = td_target(&transition, &learner->target);
    assert(target == 2);
    transition.terminated = false;
    target = td_target(&transition, &learner->target);
    assert(fabs(target - (2 + 0.95 * 7)) < 1e-12);
    for (int i = 0; i < REPLAY_CAPACITY + 1; ++i) {
        transition.reward = i;
        replay_push(&learner->replay, &transition);
    }
    assert(learner->replay.count == REPLAY_CAPACITY && learner->replay.cursor == 1);
    assert(learner->replay.data[0].reward == REPLAY_CAPACITY);
    learner->replay = (Replay){0};
    transition = (Transition){.reward = 1, .terminated = true};
    replay_push(&learner->replay, &transition);
    learner->adam.updates = 499;
    learner->adam.beta1_power = pow(0.9, 499);
    learner->adam.beta2_power = pow(0.999, 499);
    double loss;
    bool ok = learn_batch(learner, &loss);
    assert(ok && learner->adam.updates == 500);
    assert(memcmp(&learner->online, &learner->target, sizeof(Network)) == 0);
    free(learner);
}

static void test_files(void)
{
    char directory[] = "/tmp/cdemo-ai-unit-XXXXXX";
    char *created = mkdtemp(directory);
    assert(created != nullptr);
    char model_path[512], scores_path[512];
    bool ok = artifact(model_path, directory, "model.txt") && artifact(scores_path, directory, "scores.txt");
    assert(ok);
    uint32_t rng = 19;
    Network original, loaded;
    network_init(&original, &rng);
    ok = network_save(&original, model_path) && network_load(&loaded, model_path);
    assert(ok && memcmp(&original, &loaded, sizeof(Network)) == 0);
    FILE *file = fopen(model_path, "w");
    assert(file != nullptr);
    int written = fputs("CDEMO_SNAKE_DQN 1 11 64 64 3 1 20 10\nnan\n", file);
    int closed = fclose(file);
    assert(written >= 0 && closed == 0);
    ok = network_load(&loaded, model_path);
    assert(!ok && memcmp(&original, &loaded, sizeof(Network)) == 0);
    ok = network_save(&original, model_path);
    assert(ok);
    file = fopen(model_path, "a");
    assert(file != nullptr);
    written = fputs("extra\n", file);
    closed = fclose(file);
    assert(written >= 0 && closed == 0);
    ok = network_load(&loaded, model_path);
    assert(!ok);
    Scores scores = {0};
    ok = save_score(scores_path, &scores, 4) && save_score(scores_path, &scores, 9);
    assert(ok);
    Scores restored = {0};
    ok = load_scores(scores_path, &restored);
    assert(ok && restored.games == 2 && restored.best == 9);
    int removed = remove(model_path);
    assert(removed == 0);
    removed = remove(scores_path);
    assert(removed == 0);
    removed = rmdir(directory);
    assert(removed == 0);
}

int main(void)
{
    test_environment();
    test_gradient();
    test_replay();
    test_files();
    puts("AI 单元测试通过：环境/满分、全部参数梯度、Adam/裁剪、TD/截断、回放池、模型和历史文件。");
    return EXIT_SUCCESS;
}
