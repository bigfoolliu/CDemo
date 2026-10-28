/**
 * @file
 * @brief M04：硬币的 MLE/Beta 后验与一维高斯方差估计。
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846

typedef struct {
    double mle;
    double posterior_mean;
    double map;
    double posterior_alpha;
    double posterior_beta;
} CoinEstimate;

typedef struct {
    double mean;
    double mle_variance;
    double unbiased_variance;
    bool has_unbiased_variance;
} GaussianEstimate;

/**
 * @brief 从二项计数计算 MLE、Beta 后验均值和内点 MAP。
 * @param heads 正面次数，与 tails 之和必须大于零且不溢出。
 * @param tails 反面次数。
 * @param alpha Beta 先验第一个形状参数，必须为有限且大于 1 的数。
 * @param beta Beta 先验第二个形状参数，必须为有限且大于 1 的数。
 * @param result 可写结果，不转移所有权；失败时不修改。
 * @return 参数和结果均有效时为 true，否则为 false。
 */
static bool estimate_coin(size_t heads, size_t tails, double alpha, double beta,
                          CoinEstimate *result)
{
    if (result == nullptr || heads > SIZE_MAX - tails || heads + tails == 0 ||
        !isfinite(alpha) || !isfinite(beta) || alpha <= 1 || beta <= 1) {
        return false;
    }
    double count = (double)(heads + tails);
    double posterior_alpha = alpha + (double)heads;
    double posterior_beta = beta + (double)tails;
    double posterior_sum = posterior_alpha + posterior_beta;
    double mle = (double)heads / count;
    double posterior_mean = posterior_alpha / posterior_sum;
    double map = (posterior_alpha - 1) / (posterior_sum - 2);
    if (!isfinite(count) || !isfinite(posterior_sum) || !isfinite(mle) ||
        !isfinite(posterior_mean) || !isfinite(map)) {
        return false;
    }
    *result = (CoinEstimate){mle, posterior_mean, map, posterior_alpha, posterior_beta};
    return true;
}

/**
 * @brief 计算给定硬币概率下有序观测的平均负对数似然。
 * @param heads 正面个数。
 * @param tails 反面个数，样本总数必须大于零。
 * @param probability 正面概率，范围为 [0,1]；不可能事件返回失败。
 * @param result 可写结果，失败时不修改。
 * @return 结果有限时为 true；非法概率或不可能观测时为 false。
 */
static bool coin_mean_nll(size_t heads, size_t tails, double probability, double *result)
{
    if (result == nullptr || heads > SIZE_MAX - tails || heads + tails == 0 ||
        !isfinite(probability) || probability < 0 || probability > 1 ||
        (heads > 0 && probability == 0) || (tails > 0 && probability == 1)) {
        return false;
    }
    double total = 0;
    if (heads > 0) {
        total -= (double)heads * log(probability);
    }
    if (tails > 0) {
        total -= (double)tails * log1p(-probability);
    }
    double candidate = total / (double)(heads + tails);
    if (!isfinite(candidate)) {
        return false;
    }
    *result = candidate;
    return true;
}

/**
 * @brief 用两遍扫描计算一维高斯的均值、MLE 方差和无偏样本方差。
 * @param values 长度 count 的只读有限值数组，不转移所有权。
 * @param count 样本数，至少为 1；只有至少 2 个样本才有无偏方差。
 * @param result 可写结果，失败时不修改。
 * @return 数值有效时为 true；空输入或溢出时为 false。
 */
static bool estimate_gaussian(const double *values, size_t count, GaussianEstimate *result)
{
    if (values == nullptr || result == nullptr || count == 0) {
        return false;
    }
    double sum = 0;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return false;
        }
        sum += values[i];
        if (!isfinite(sum)) {
            return false;
        }
    }
    double mean = sum / (double)count;
    double squared = 0;
    for (size_t i = 0; i < count; ++i) {
        double delta = values[i] - mean;
        squared += delta * delta;
        if (!isfinite(squared)) {
            return false;
        }
    }
    double mle_variance = squared / (double)count;
    double unbiased_variance = count > 1 ? squared / (double)(count - 1) : 0;
    if (!isfinite(mean) || !isfinite(mle_variance) || !isfinite(unbiased_variance)) {
        return false;
    }
    *result = (GaussianEstimate){mean, mle_variance, unbiased_variance, count > 1};
    return true;
}

/**
 * @brief 计算非退化一维高斯在指定点的概率密度。
 * @param estimate 有限均值与严格正的 MLE 方差。
 * @param x 有限的评估点。
 * @param result 可写密度，失败时不修改。
 * @return 结果有限且参数有效时为 true。
 */
static bool gaussian_density(const GaussianEstimate *estimate, double x, double *result)
{
    if (estimate == nullptr || result == nullptr || !isfinite(x) ||
        !isfinite(estimate->mean) || !isfinite(estimate->mle_variance) ||
        estimate->mle_variance <= 0) {
        return false;
    }
    double delta = x - estimate->mean;
    double exponent = -(delta * delta) / (2 * estimate->mle_variance);
    double candidate = exp(exponent) / sqrt(2 * PI * estimate->mle_variance);
    if (!isfinite(candidate)) {
        return false;
    }
    *result = candidate;
    return true;
}

/** @brief 验证闭式结果、边界计数与不修改输出的失败路径。 */
static void test_estimation(void)
{
    CoinEstimate coin = {0};
    assert(estimate_coin(7, 3, 2, 2, &coin));
    assert(fabs(coin.mle - 0.7) < 1e-12);
    assert(fabs(coin.posterior_mean - 9.0 / 14.0) < 1e-12);
    assert(fabs(coin.map - 2.0 / 3.0) < 1e-12);
    CoinEstimate sentinel = {.mle = 99};
    assert(!estimate_coin(0, 0, 2, 2, &sentinel));
    assert(!estimate_coin(7, 3, 1, 2, &sentinel));
    assert(!estimate_coin(SIZE_MAX, 1, 2, 2, &sentinel));
    assert(sentinel.mle == 99);
    assert(estimate_coin(0, 10, 2, 2, &coin));
    assert(coin.mle == 0 && coin.posterior_mean > 0 && coin.map > 0);
    assert(estimate_coin(10, 0, 2, 2, &coin));
    assert(coin.mle == 1 && coin.posterior_mean < 1 && coin.map < 1);
    double score = 99;
    assert(!coin_mean_nll(1, 0, 0, &score));
    assert(!coin_mean_nll(0, 1, 1, &score));
    assert(score == 99);
    assert(coin_mean_nll(0, 3, 0, &score) && score == 0);

    const double values[] = {1, 2, 3, 4};
    GaussianEstimate gaussian = {0};
    assert(estimate_gaussian(values, 4, &gaussian));
    assert(gaussian.mean == 2.5);
    assert(gaussian.mle_variance == 1.25);
    assert(fabs(gaussian.unbiased_variance - 5.0 / 3.0) < 1e-12);
    assert(gaussian.has_unbiased_variance);
    assert(gaussian_density(&gaussian, 2.5, &score));
    assert(fabs(score - 1 / sqrt(2 * PI * 1.25)) < 1e-12);
    const double same[] = {4, 4, 4};
    assert(estimate_gaussian(same, 3, &gaussian));
    assert(gaussian.mle_variance == 0);
    score = 99;
    assert(!gaussian_density(&gaussian, 4, &score) && score == 99);
    const double one[] = {4};
    assert(estimate_gaussian(one, 1, &gaussian));
    assert(!gaussian.has_unbiased_variance);
    const double invalid[] = {1, NAN};
    gaussian.mean = 99;
    assert(!estimate_gaussian(nullptr, 0, &gaussian));
    assert(!estimate_gaussian(invalid, 2, &gaussian));
    assert(gaussian.mean == 99);
}

/**
 * @brief 输出固定训练和独立测试示例。
 * @param csv 为 true 时输出 field,value，供跨语言比较。
 * @return 成功为 EXIT_SUCCESS，计算失败为 EXIT_FAILURE。
 */
static int print_demo(bool csv)
{
    CoinEstimate coin = {0};
    GaussianEstimate gaussian = {0};
    const double values[] = {1, 2, 3, 4};
    double test_mle = 0;
    double test_mean = 0;
    double test_map = 0;
    double test_baseline = 0;
    double density = 0;
    if (!estimate_coin(7, 3, 2, 2, &coin) ||
        !coin_mean_nll(4, 2, coin.mle, &test_mle) ||
        !coin_mean_nll(4, 2, coin.posterior_mean, &test_mean) ||
        !coin_mean_nll(4, 2, coin.map, &test_map) ||
        !coin_mean_nll(4, 2, 0.5, &test_baseline) ||
        !estimate_gaussian(values, 4, &gaussian) ||
        !gaussian_density(&gaussian, gaussian.mean, &density)) {
        return EXIT_FAILURE;
    }
    if (csv) {
        puts("field,value");
        printf("coin_mle,%.17g\ncoin_posterior_mean,%.17g\ncoin_map,%.17g\n",
               coin.mle, coin.posterior_mean, coin.map);
        printf("coin_test_nll_mle,%.17g\ncoin_test_nll_mean,%.17g\n",
               test_mle, test_mean);
        printf("coin_test_nll_map,%.17g\ncoin_test_nll_baseline,%.17g\n",
               test_map, test_baseline);
        printf("gaussian_mean,%.17g\ngaussian_mle_variance,%.17g\n",
               gaussian.mean, gaussian.mle_variance);
        printf("gaussian_unbiased_variance,%.17g\ngaussian_density_at_mean,%.17g\n",
               gaussian.unbiased_variance, density);
    } else {
        printf("硬币：MLE %.6f，后验均值 %.6f，MAP %.6f\n",
               coin.mle, coin.posterior_mean, coin.map);
        printf("独立测试平均负对数似然：MLE %.6f，后验均值 %.6f，MAP %.6f，"
               "公平硬币基线 %.6f\n", test_mle, test_mean, test_map, test_baseline);
        printf("高斯：均值 %.6f，MLE 方差 %.6f，无偏样本方差 %.6f，"
               "均值处密度 %.6f\n", gaussian.mean, gaussian.mle_variance,
               gaussian.unbiased_variance, density);
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行概率与统计估计教学示例。
 * @param argc 参数数目；可传 --test 或 --csv。
 * @param argv 参数数组。
 * @return 正常完成为 EXIT_SUCCESS，参数或计算失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_estimation();
        puts("概率与统计估计测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1) {
        return print_demo(false);
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        return print_demo(true);
    }
    fputs("用法：33_probability_estimation [--test | --csv]\n", stderr);
    return EXIT_FAILURE;
}
