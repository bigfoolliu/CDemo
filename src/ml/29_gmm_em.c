/** @file
 * @brief 用二维固定数据演示高斯混合模型与期望最大化。
 */
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

enum {
    SAMPLE_COUNT = 6,
    COMPONENT_COUNT = 2,
    MAX_ITERATIONS = 20
};

#define PI 3.14159265358979323846
#define COVARIANCE_REGULARIZATION 1e-6
#define EFFECTIVE_COUNT_MINIMUM 1e-12
#define CONVERGENCE_TOLERANCE 1e-10

/** 保存一个二维样本或均值向量。 */
typedef struct {
    double x;
    double y;
} Point;

/** 保存对称二维协方差矩阵的三个独立元素。 */
typedef struct {
    double xx;
    double xy;
    double yy;
} Covariance;

/** 保存两个高斯成分的权重、均值和协方差。 */
typedef struct {
    double weights[COMPONENT_COUNT];
    Point means[COMPONENT_COUNT];
    Covariance covariances[COMPONENT_COUNT];
} GaussianMixture;

/* 两组成团的二维样本与 K-means 案例使用相同数据。 */
static const Point POINTS[SAMPLE_COUNT] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {8.0, 8.0},
    {8.0, 9.0},
    {9.0, 8.0}
};

/**
 * @brief 检查二维协方差是否有限且正定。
 * @param covariance 待检查的对称协方差矩阵。
 * @return 对角元素和行列式均满足正定条件时返回 true。
 */
static bool covariance_is_valid(Covariance covariance)
{
    double determinant = covariance.xx * covariance.yy - covariance.xy * covariance.xy;

    return isfinite(covariance.xx) && isfinite(covariance.xy) && isfinite(covariance.yy) &&
           isfinite(determinant) && covariance.xx > 0.0 && determinant > 0.0;
}

/**
 * @brief 检查 GMM 参数是否满足基本概率约束。
 * @param model 待检查的模型参数。
 * @return 权重为正且和为 1、均值有限、协方差正定时返回 true。
 */
static bool model_is_valid(const GaussianMixture *model)
{
    if (model == NULL) {
        return false;
    }

    double weight_sum = 0.0;
    for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
        const Point mean = model->means[component];

        if (!isfinite(model->weights[component]) || model->weights[component] <= 0.0 ||
            !isfinite(mean.x) || !isfinite(mean.y) ||
            !covariance_is_valid(model->covariances[component])) {
            return false;
        }

        weight_sum += model->weights[component];
    }

    return fabs(weight_sum - 1.0) <= 1e-9;
}

/**
 * @brief 计算二维高斯分布的对数概率密度。
 * @param point 当前观测点。
 * @param mean 高斯成分的均值向量。
 * @param covariance 对称正定协方差矩阵。
 * @param log_density 用于接收对数密度的可写地址。
 * @return 输入有效且结果有限时返回 true；失败时不修改输出。
 */
static bool log_gaussian_2d(
    Point point,
    Point mean,
    Covariance covariance,
    double *log_density
)
{
    if (log_density == NULL || !isfinite(point.x) || !isfinite(point.y) ||
        !isfinite(mean.x) || !isfinite(mean.y) || !covariance_is_valid(covariance)) {
        return false;
    }

    double determinant = covariance.xx * covariance.yy - covariance.xy * covariance.xy;
    double dx = point.x - mean.x;
    double dy = point.y - mean.y;
    /* 二次型等于残差转置乘协方差逆矩阵再乘残差。 */
    double quadratic = (
        covariance.yy * dx * dx - 2.0 * covariance.xy * dx * dy + covariance.xx * dy * dy
    ) / determinant;
    /* 在对数空间相加，避免先算很小的高斯密度而下溢。 */
    double result = -log(2.0 * PI) - 0.5 * log(determinant) - 0.5 * quadratic;

    if (!isfinite(result)) {
        return false;
    }

    *log_density = result;
    return true;
}

/**
 * @brief E 步：计算责任度矩阵和当前模型的观测对数似然。
 * @param model 当前有效 GMM 参数。
 * @param responsibilities 用于接收 SAMPLE_COUNT × COMPONENT_COUNT 个责任度。
 * @param log_likelihood 用于接收所有样本的观测对数似然。
 * @return 计算成功时返回 true；失败时不修改输出。
 */
static bool expectation_step(
    const GaussianMixture *model,
    double responsibilities[SAMPLE_COUNT][COMPONENT_COUNT],
    double *log_likelihood
)
{
    if (responsibilities == NULL || log_likelihood == NULL || !model_is_valid(model)) {
        return false;
    }

    double updated_responsibilities[SAMPLE_COUNT][COMPONENT_COUNT] = {0};
    double total_log_likelihood = 0.0;

    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        double log_scores[COMPONENT_COUNT] = {0};
        for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
            double log_density = 0.0;
            if (!log_gaussian_2d(
                    POINTS[sample],
                    model->means[component],
                    model->covariances[component],
                    &log_density
                )) {
                return false;
            }

            /* 对数分数 = 对数先验权重 + 当前成分的对数密度。 */
            log_scores[component] = log(model->weights[component]) + log_density;
        }

        double maximum_score = fmax(log_scores[0], log_scores[1]);
        /* 减去最大分数是 log-sum-exp 的稳定化步骤。 */
        double log_normalizer = maximum_score + log(
            exp(log_scores[0] - maximum_score) + exp(log_scores[1] - maximum_score)
        );
        total_log_likelihood += log_normalizer;

        for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
            /* 除以所有成分的总分数后，得到和为 1 的责任度。 */
            updated_responsibilities[sample][component] =
                exp(log_scores[component] - log_normalizer);
        }
    }

    if (!isfinite(total_log_likelihood)) {
        return false;
    }

    memcpy(responsibilities, updated_responsibilities, sizeof(updated_responsibilities));
    *log_likelihood = total_log_likelihood;
    return true;
}

/**
 * @brief M 步：按责任度更新混合权重、均值和协方差。
 * @param responsibilities 上一轮 E 步计算出的责任度矩阵。
 * @param updated_model 用于接收新参数的可写地址。
 * @return 更新成功时返回 true；空成分或无效协方差时不修改输出。
 */
static bool maximization_step(
    const double responsibilities[SAMPLE_COUNT][COMPONENT_COUNT],
    GaussianMixture *updated_model
)
{
    if (responsibilities == NULL || updated_model == NULL) {
        return false;
    }

    /* N_k 是第 k 个成分收到的软分配总量，通常不是整数。 */
    double effective_counts[COMPONENT_COUNT] = {0};
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        double row_sum = 0.0;
        for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
            double responsibility = responsibilities[sample][component];
            if (!isfinite(responsibility) || responsibility < 0.0 || responsibility > 1.0) {
                return false;
            }

            effective_counts[component] += responsibility;
            row_sum += responsibility;
        }

        if (fabs(row_sum - 1.0) > 1e-9) {
            return false;
        }
    }

    GaussianMixture candidate = {0};
    for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
        double effective_count = effective_counts[component];
        if (effective_count <= EFFECTIVE_COUNT_MINIMUM) {
            return false;
        }

        candidate.weights[component] = effective_count / (double)SAMPLE_COUNT;
        for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
            double responsibility = responsibilities[sample][component];
            /* 均值是按每个样本责任度加权后的坐标平均。 */
            candidate.means[component].x += responsibility * POINTS[sample].x;
            candidate.means[component].y += responsibility * POINTS[sample].y;
        }

        candidate.means[component].x /= effective_count;
        candidate.means[component].y /= effective_count;
    }

    for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
        for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
            double responsibility = responsibilities[sample][component];
            double dx = POINTS[sample].x - candidate.means[component].x;
            double dy = POINTS[sample].y - candidate.means[component].y;

            candidate.covariances[component].xx += responsibility * dx * dx;
            candidate.covariances[component].xy += responsibility * dx * dy;
            candidate.covariances[component].yy += responsibility * dy * dy;
        }

        /* 加权外积给出完整协方差；再给对角线增加很小的正则量。 */
        candidate.covariances[component].xx /= effective_counts[component];
        candidate.covariances[component].xy /= effective_counts[component];
        candidate.covariances[component].yy /= effective_counts[component];
        candidate.covariances[component].xx += COVARIANCE_REGULARIZATION;
        candidate.covariances[component].yy += COVARIANCE_REGULARIZATION;
    }

    if (!model_is_valid(&candidate)) {
        return false;
    }

    *updated_model = candidate;
    return true;
}

/**
 * @brief 输出一轮 EM 的似然和全部成分参数。
 * @param iteration 当前迭代编号，初始模型使用 0。
 * @param model 当前 GMM 参数。
 * @param log_likelihood 当前观测对数似然。
 * @param change 与上一轮相比的对数似然变化。
 * @return 无。
 */
static void print_model(
    size_t iteration,
    const GaussianMixture *model,
    double log_likelihood,
    double change
)
{
    printf(
        "迭代 %zu：对数似然=% .9f，变化=%+.3e\n",
        iteration,
        log_likelihood,
        change
    );

    for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
        Point mean = model->means[component];
        Covariance covariance = model->covariances[component];
        printf(
            "  成分 %zu：权重=%.6f，均值=(%.6f, %.6f)，"
            "协方差=[[%.6f, %.6f], [%.6f, %.6f]]\n",
            component,
            model->weights[component],
            mean.x,
            mean.y,
            covariance.xx,
            covariance.xy,
            covariance.xy,
            covariance.yy
        );
    }
}

/**
 * @brief 输出每个样本属于两个成分的后验概率。
 * @param responsibilities E 步生成的责任度矩阵。
 * @return 无。
 */
static void print_responsibilities(
    const double responsibilities[SAMPLE_COUNT][COMPONENT_COUNT]
)
{
    for (size_t sample = 0; sample < SAMPLE_COUNT; ++sample) {
        printf(
            "  样本 %zu (%g, %g)：责任度=(%.9f, %.9f)\n",
            sample,
            POINTS[sample].x,
            POINTS[sample].y,
            responsibilities[sample][0],
            responsibilities[sample][1]
        );
    }
}

/**
 * @brief 用固定二维点运行 GMM 与 EM 教学示例。
 * @return 训练正常结束时返回 0；数值或模型更新失败时返回 1。
 */
int main(void)
{
    GaussianMixture model = {
        .weights = {0.5, 0.5},
        .means = {{0.0, 0.0}, {8.0, 8.0}},
        .covariances = {{1.0, 0.0, 1.0}, {1.0, 0.0, 1.0}}
    };
    double responsibilities[SAMPLE_COUNT][COMPONENT_COUNT] = {0};
    double log_likelihood = 0.0;

    if (!expectation_step(&model, responsibilities, &log_likelihood)) {
        fprintf(stderr, "初始 GMM 参数无效，无法计算 E 步。\n");
        return 1;
    }

    print_model(0, &model, log_likelihood, 0.0);
    print_responsibilities(responsibilities);
    bool converged = false;

    /* 每轮先用当前责任度做 M 步，再对新参数执行 E 步。 */
    for (size_t iteration = 1; iteration <= MAX_ITERATIONS; ++iteration) {
        double previous_log_likelihood = log_likelihood;
        GaussianMixture updated_model = {0};

        if (!maximization_step(responsibilities, &updated_model)) {
            fprintf(
                stderr,
                "第 %zu 轮 M 步失败：成分为空或协方差无效。\n",
                iteration
            );
            return 1;
        }

        if (!expectation_step(&updated_model, responsibilities, &log_likelihood)) {
            fprintf(stderr, "第 %zu 轮更新后无法计算 E 步。\n", iteration);
            return 1;
        }

        double change = log_likelihood - previous_log_likelihood;
        model = updated_model;
        print_model(iteration, &model, log_likelihood, change);
        print_responsibilities(responsibilities);

        if (fabs(change) < CONVERGENCE_TOLERANCE) {
            converged = true;
            break;
        }
    }

    assert(model_is_valid(&model));
    printf("训练结束：%s。\n", converged ? "达到似然变化阈值" : "达到迭代上限");
    return 0;
}
