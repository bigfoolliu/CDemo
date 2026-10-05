#ifndef CDEMO_SNAKE_AI_NETWORK_H
#define CDEMO_SNAKE_AI_NETWORK_H

#include "snake_env.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { HIDDEN = 64, W1 = 0, B1 = 704, W2 = 768, B2 = 4864,
       W3 = 4928, B3 = 5120, PARAMETERS = 5123 };
/* 各层矩阵按输出行连续存放，其后跟偏置。修改维度也需升级模型格式。 */
static_assert(B1 == FEATURES * HIDDEN && W2 == B1 + HIDDEN);
static_assert(B2 == W2 + HIDDEN * HIDDEN && W3 == B2 + HIDDEN);
static_assert(B3 == W3 + ACTIONS * HIDDEN && PARAMETERS == B3 + ACTIONS);
typedef struct { double p[PARAMETERS]; } Network;
typedef struct {
    double input[FEATURES];
    double h1[HIDDEN];
    double h2[HIDDEN];
    double q[ACTIONS];
} Forward;
typedef struct {
    double m[PARAMETERS];
    double v[PARAMETERS];
    double beta1_power;
    double beta2_power;
    size_t updates;
} Adam;

static inline double gaussian(uint32_t *rng)
{
    double u = ((double)snake_random(rng) + 1) / 4294967297.0;
    return sqrt(-2 * log(u)) * cos(6.283185307179586 * snake_uniform(rng));
}

static inline void network_init(Network *net, uint32_t *rng)
{
    *net = (Network){0};
    for (int i = W1; i < B1; ++i) {
        net->p[i] = gaussian(rng) * sqrt(2.0 / FEATURES);
    }
    for (int i = W2; i < B2; ++i) {
        net->p[i] = gaussian(rng) * sqrt(2.0 / HIDDEN);
    }
    for (int i = W3; i < B3; ++i) {
        net->p[i] = gaussian(rng) * sqrt(1.0 / HIDDEN);
    }
}

static inline void dense(const double *weights, const double *bias, const double *input,
                         int inputs, int outputs, double *result, bool relu)
{
    for (int row = 0; row < outputs; ++row) {
        double value = bias[row];
        for (int column = 0; column < inputs; ++column) {
            value += weights[row * inputs + column] * input[column];
        }
        result[row] = relu ? fmax(0, value) : value;
    }
}

static inline void network_forward(const Network *net, const double state[FEATURES], Forward *cache)
{
    memcpy(cache->input, state, sizeof(cache->input));
    dense(net->p + W1, net->p + B1, state, FEATURES, HIDDEN, cache->h1, true);
    dense(net->p + W2, net->p + B2, cache->h1, HIDDEN, HIDDEN, cache->h2, true);
    dense(net->p + W3, net->p + B3, cache->h2, HIDDEN, ACTIONS, cache->q, false);
}

static inline int greedy_action(const Network *net, const double state[FEATURES])
{
    Forward cache;
    network_forward(net, state, &cache);
    int best = 0;
    for (int action = 1; action < ACTIONS; ++action) {
        if (cache.q[action] > cache.q[best]) {
            best = action;
        }
    }
    return best;
}

static inline double huber(double error)
{
    return fabs(error) <= 1 ? 0.5 * error * error : fabs(error) - 0.5;
}

/* 只对所选动作的 Q 值求导。TD 目标是常数，不能沿目标网络反传。 */
static inline double network_gradient(const Network *net, const Forward *cache,
                                     int action, double target, Network *gradient)
{
    double error = cache->q[action] - target;
    double derivative = fmax(-1, fmin(1, error));
    gradient->p[B3 + action] += derivative;
    double delta2[HIDDEN];
    for (int i = 0; i < HIDDEN; ++i) {
        gradient->p[W3 + action * HIDDEN + i] += derivative * cache->h2[i];
        delta2[i] = cache->h2[i] > 0 ? derivative * net->p[W3 + action * HIDDEN + i] : 0;
        gradient->p[B2 + i] += delta2[i];
        for (int j = 0; j < HIDDEN; ++j) {
            gradient->p[W2 + i * HIDDEN + j] += delta2[i] * cache->h1[j];
        }
    }
    for (int j = 0; j < HIDDEN; ++j) {
        double delta1 = 0;
        for (int i = 0; i < HIDDEN; ++i) {
            delta1 += delta2[i] * net->p[W2 + i * HIDDEN + j];
        }
        if (cache->h1[j] <= 0) {
            delta1 = 0;
        }
        gradient->p[B1 + j] += delta1;
        for (int i = 0; i < FEATURES; ++i) {
            gradient->p[W1 + j * FEATURES + i] += delta1 * cache->input[i];
        }
    }
    return huber(error);
}

static inline bool adam_update(Network *net, Adam *adam, Network *gradient, size_t batch)
{
    double norm_squared = 0;
    for (int i = 0; i < PARAMETERS; ++i) {
        gradient->p[i] /= (double)batch;
        if (!isfinite(gradient->p[i])) {
            return false;
        }
        norm_squared += gradient->p[i] * gradient->p[i];
    }
    if (!isfinite(norm_squared)) {
        return false;
    }
    double scale = norm_squared > 100 ? 10 / sqrt(norm_squared) : 1;
    if (adam->updates == 0) {
        adam->beta1_power = 1;
        adam->beta2_power = 1;
    }
    adam->beta1_power *= 0.9;
    adam->beta2_power *= 0.999;
    ++adam->updates;
    for (int i = 0; i < PARAMETERS; ++i) {
        double g = gradient->p[i] * scale;
        adam->m[i] = 0.9 * adam->m[i] + 0.1 * g;
        adam->v[i] = 0.999 * adam->v[i] + 0.001 * g * g;
        double m = adam->m[i] / (1 - adam->beta1_power);
        double v = adam->v[i] / (1 - adam->beta2_power);
        net->p[i] -= 0.001 * m / (sqrt(v) + 1e-8);
        if (!isfinite(net->p[i])) {
            return false;
        }
    }
    return true;
}

/* 模型仅保存推理参数，不是可恢复 Adam / 回放池的训练检查点。 */
static inline bool network_save(const Network *net, const char *path)
{
    char temporary[1024];
    int length = snprintf(temporary, sizeof(temporary), "%s.tmp", path);
    if (length < 0 || (size_t)length >= sizeof(temporary)) {
        return false;
    }
    FILE *file = fopen(temporary, "wx");
    if (file == nullptr) {
        return false;
    }
    bool ok = fprintf(file, "CDEMO_SNAKE_DQN 1 11 64 64 3 1 20 10\n") >= 0;
    for (int i = 0; i < PARAMETERS && ok; ++i) {
        ok = isfinite(net->p[i]) && fprintf(file, "%.17g\n", net->p[i]) >= 0;
    }
    if (fclose(file) == EOF) {
        ok = false;
    }
    if (ok) {
        ok = rename(temporary, path) == 0;
    }
    if (!ok) {
        (void)remove(temporary);
    }
    return ok;
}

static inline bool network_load(Network *net, const char *path)
{
    FILE *file = fopen(path, "r");
    if (file == nullptr) {
        return false;
    }
    char header[128];
    Network candidate;
    bool ok = fgets(header, sizeof(header), file) != nullptr &&
        strcmp(header, "CDEMO_SNAKE_DQN 1 11 64 64 3 1 20 10\n") == 0;
    for (int i = 0; i < PARAMETERS && ok; ++i) {
        ok = fscanf(file, "%lf", &candidate.p[i]) == 1 && isfinite(candidate.p[i]);
    }
    if (ok) {
        char extra;
        ok = fscanf(file, " %c", &extra) == EOF && !ferror(file);
    }
    if (fclose(file) == EOF) {
        ok = false;
    }
    if (ok) {
        *net = candidate;
    }
    return ok;
}

#endif
