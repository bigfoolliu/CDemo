/**
 * @file
 * @brief 主成分分析：训练集中心化、协方差、幂迭代与二维重建误差。
 */
#include "../graphics/bmp_capture.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    FEATURE_COUNT = 4,
    COMPONENT_COUNT = 2,
    TRAIN_COUNT = 64,
    TEST_COUNT = 32,
    POWER_ITERATIONS = 256
};

typedef struct {
    size_t count;
    double values[TRAIN_COUNT][FEATURE_COUNT];
} Dataset;

typedef struct {
    double mean[FEATURE_COUNT];
    double covariance[FEATURE_COUNT][FEATURE_COUNT];
    double components[COMPONENT_COUNT][FEATURE_COUNT];
    double eigenvalues[COMPONENT_COUNT];
    double total_variance;
} PcaModel;

typedef struct {
    double projection[COMPONENT_COUNT];
    double reconstruction[FEATURE_COUNT];
} PcaResult;

/**
 * @brief 产生固定种子的 [0,1) 伪随机数。
 * @param state 可写随机状态，由调用者独占维护。
 * @return 更新状态映射到 [0,1) 的实数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 生成四个相关特征的合成样本，两个潜在方向叠加小噪声。
 * @param dataset 有效可写数据集；count 和全部值都会被覆盖。
 * @param count 样本数量，2..TRAIN_COUNT。
 * @param seed 随机种子；训练和测试使用不同种子。
 * @return 参数有效且生成值有限为 true，否则不修改数据集。
 */
static bool dataset_generate(Dataset *dataset, size_t count, uint32_t seed)
{
    if (dataset == nullptr || count < 2 || count > TRAIN_COUNT) {
        return false;
    }
    Dataset generated = {0};
    generated.count = count;
    for (size_t row = 0; row < count; ++row) {
        double first = 2.0 * random_unit(&seed) - 1.0;
        double second = 2.0 * random_unit(&seed) - 1.0;
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double noise = 0.2 * random_unit(&seed) - 0.1;
            double first_weight[] = {2.0, 1.0, 3.0, -1.0};
            double second_weight[] = {0.2, -0.4, 0.1, 0.5};
            generated.values[row][feature] = first_weight[feature] * first +
                                             second_weight[feature] * second + noise;
            if (!isfinite(generated.values[row][feature])) {
                return false;
            }
        }
    }
    *dataset = generated;
    return true;
}

/**
 * @brief 归一化向量，拒绝零长度和非有限范数。
 * @param vector 可写向量，恰有 FEATURE_COUNT 个元素。
 * @return 归一化成功为 true；失败时保留原向量。
 */
static bool normalize(double vector[FEATURE_COUNT])
{
    double norm_squared = 0.0;
    for (size_t i = 0; i < FEATURE_COUNT; ++i) {
        norm_squared += vector[i] * vector[i];
    }
    if (!isfinite(norm_squared) || norm_squared <= DBL_MIN) {
        return false;
    }
    double inverse_norm = 1.0 / sqrt(norm_squared);
    for (size_t i = 0; i < FEATURE_COUNT; ++i) {
        vector[i] *= inverse_norm;
    }
    return true;
}

/**
 * @brief 从向量中移除在已知单位方向上的投影分量。
 * @param vector 可写向量，恰有 FEATURE_COUNT 个元素。
 * @param direction 已归一化的只读方向，恰有 FEATURE_COUNT 个元素。
 */
static void remove_projection(double vector[FEATURE_COUNT],
                              const double direction[FEATURE_COUNT])
{
    double coefficient = 0.0;
    for (size_t i = 0; i < FEATURE_COUNT; ++i) {
        coefficient += vector[i] * direction[i];
    }
    for (size_t i = 0; i < FEATURE_COUNT; ++i) {
        vector[i] -= coefficient * direction[i];
    }
}

/**
 * @brief 固定一个方向后用幂迭代求最大特征方向，并通过残差检查收敛。
 * @param covariance 有限对称协方差矩阵，形状 FEATURE_COUNT×FEATURE_COUNT。
 * @param orthogonal_to 已求出的单位方向；首主成分时为空。
 * @param output 有效可写单位向量；成功时写入规范符号方向。
 * @param eigenvalue 有效可写特征值输出，成功时写入 Rayleigh 商。
 * @return 迭代、归一化与残差检查成功为 true；失败不修改输出。
 */
static bool dominant_component(const double covariance[FEATURE_COUNT][FEATURE_COUNT],
                               const double *orthogonal_to,
                               double output[FEATURE_COUNT], double *eigenvalue)
{
    double vector[FEATURE_COUNT] = {1.0, 0.5, -0.25, 0.75};
    if (orthogonal_to != nullptr) {
        remove_projection(vector, orthogonal_to);
    }
    if (!normalize(vector)) {
        return false;
    }

    for (size_t iteration = 0; iteration < POWER_ITERATIONS; ++iteration) {
        double next[FEATURE_COUNT] = {0};
        for (size_t row = 0; row < FEATURE_COUNT; ++row) {
            for (size_t column = 0; column < FEATURE_COUNT; ++column) {
                next[row] += covariance[row][column] * vector[column];
            }
        }
        if (orthogonal_to != nullptr) {
            remove_projection(next, orthogonal_to);
        }
        if (!normalize(next)) {
            return false;
        }
        double alignment = 0.0;
        for (size_t i = 0; i < FEATURE_COUNT; ++i) {
            alignment += next[i] * vector[i];
        }
        if (alignment < 0.0) {
            for (size_t i = 0; i < FEATURE_COUNT; ++i) {
                next[i] = -next[i];
            }
        }
        for (size_t i = 0; i < FEATURE_COUNT; ++i) {
            vector[i] = next[i];
        }
    }

    double value = 0.0;
    double residual_squared = 0.0;
    for (size_t row = 0; row < FEATURE_COUNT; ++row) {
        double product = 0.0;
        for (size_t column = 0; column < FEATURE_COUNT; ++column) {
            product += covariance[row][column] * vector[column];
        }
        value += vector[row] * product;
    }
    if (!isfinite(value) || value <= 0.0) {
        return false;
    }
    for (size_t row = 0; row < FEATURE_COUNT; ++row) {
        double product = 0.0;
        for (size_t column = 0; column < FEATURE_COUNT; ++column) {
            product += covariance[row][column] * vector[column];
        }
        double residual = product - value * vector[row];
        residual_squared += residual * residual;
    }
    if (!isfinite(residual_squared) || sqrt(residual_squared) > 1e-8 * (1.0 + value)) {
        return false;
    }

    size_t largest = 0;
    for (size_t i = 1; i < FEATURE_COUNT; ++i) {
        if (fabs(vector[i]) > fabs(vector[largest])) {
            largest = i;
        }
    }
    if (vector[largest] < 0.0) {
        for (size_t i = 0; i < FEATURE_COUNT; ++i) {
            vector[i] = -vector[i];
        }
    }
    memcpy(output, vector, sizeof(vector));
    *eigenvalue = value;
    return true;
}

/**
 * @brief 仅使用训练集拟合均值、总体协方差和两个正交主成分。
 * @param model 可写模型；成功后整体替换，不转移外部资源。
 * @param dataset 至少两个有限样本的只读训练数据集。
 * @return 两个正特征值及收敛方向均有效时为 true，失败时模型不变。
 */
static bool pca_fit(PcaModel *model, const Dataset *dataset)
{
    if (model == nullptr || dataset == nullptr || dataset->count < 2 ||
        dataset->count > TRAIN_COUNT) {
        return false;
    }
    PcaModel fitted = {0};
    for (size_t row = 0; row < dataset->count; ++row) {
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double value = dataset->values[row][feature];
            if (!isfinite(value)) {
                return false;
            }
            fitted.mean[feature] += value;
        }
    }
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        fitted.mean[feature] /= (double)dataset->count;
    }
    for (size_t row = 0; row < dataset->count; ++row) {
        for (size_t left = 0; left < FEATURE_COUNT; ++left) {
            double centered_left = dataset->values[row][left] - fitted.mean[left];
            for (size_t right = 0; right < FEATURE_COUNT; ++right) {
                double centered_right = dataset->values[row][right] - fitted.mean[right];
                fitted.covariance[left][right] += centered_left * centered_right;
            }
        }
    }
    for (size_t left = 0; left < FEATURE_COUNT; ++left) {
        for (size_t right = 0; right < FEATURE_COUNT; ++right) {
            fitted.covariance[left][right] /= (double)dataset->count;
            if (!isfinite(fitted.covariance[left][right])) {
                return false;
            }
        }
        fitted.total_variance += fitted.covariance[left][left];
    }
    if (!isfinite(fitted.total_variance) || fitted.total_variance <= 0.0 ||
        !dominant_component(fitted.covariance, nullptr, fitted.components[0],
                            &fitted.eigenvalues[0]) ||
        !dominant_component(fitted.covariance, fitted.components[0], fitted.components[1],
                            &fitted.eigenvalues[1])) {
        return false;
    }
    *model = fitted;
    return true;
}

/**
 * @brief 将一个样本中心化后投影到两个主成分，并重建回原特征空间。
 * @param model 已拟合的只读 PCA 模型。
 * @param input 有效只读特征向量，恰有 FEATURE_COUNT 个有限值。
 * @param output 有效可写结果；投影与重建均成功后才整体修改。
 * @return 计算结果有限为 true，否则为 false 且输出保持不变。
 */
static bool pca_transform(const PcaModel *model, const double input[FEATURE_COUNT],
                          PcaResult *output)
{
    if (model == nullptr || input == nullptr || output == nullptr) {
        return false;
    }
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        if (!isfinite(input[feature]) || !isfinite(model->mean[feature])) {
            return false;
        }
    }
    PcaResult result = {0};
    for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            result.projection[component] +=
                (input[feature] - model->mean[feature]) * model->components[component][feature];
        }
    }
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        result.reconstruction[feature] = model->mean[feature];
        for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
            result.reconstruction[feature] +=
                result.projection[component] * model->components[component][feature];
        }
        if (!isfinite(result.reconstruction[feature])) {
            return false;
        }
    }
    *output = result;
    return true;
}

/**
 * @brief 计算数据集的逐特征平均平方重建误差。
 * @param model 已拟合的只读模型。
 * @param dataset 只读评估数据集。
 * @param output 有效可写误差输出；失败时不修改。
 * @return 全部重建成功且误差有限为 true，否则为 false。
 */
static bool reconstruction_mse(const PcaModel *model, const Dataset *dataset, double *output)
{
    if (model == nullptr || dataset == nullptr || output == nullptr || dataset->count == 0 ||
        dataset->count > TRAIN_COUNT) {
        return false;
    }
    double total_squared_error = 0.0;
    for (size_t row = 0; row < dataset->count; ++row) {
        PcaResult result;
        if (!pca_transform(model, dataset->values[row], &result)) {
            return false;
        }
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            double difference = dataset->values[row][feature] - result.reconstruction[feature];
            total_squared_error += difference * difference;
        }
    }
    double mse = total_squared_error / (double)(dataset->count * FEATURE_COUNT);
    if (!isfinite(mse)) {
        return false;
    }
    *output = mse;
    return true;
}

/**
 * @brief 导出训练集与测试集的二维主成分散点图。
 * @param model 仅由训练集拟合的只读模型。
 * @param training 只读训练样本，用于确定图表坐标范围。
 * @param test 只读测试样本，只投影并显示，不参与拟合或定轴。
 * @return 图片独占写入 exports/ 成功为 true，否则为 false。
 */
static bool export_projection(const PcaModel *model, const Dataset *training,
                              const Dataset *test)
{
    if (model == nullptr || training == nullptr || test == nullptr || training->count == 0 ||
        test->count == 0) {
        return false;
    }
    double minimum[COMPONENT_COUNT] = {INFINITY, INFINITY};
    double maximum[COMPONENT_COUNT] = {-INFINITY, -INFINITY};
    for (size_t i = 0; i < training->count; ++i) {
        PcaResult result;
        if (!pca_transform(model, training->values[i], &result)) {
            return false;
        }
        for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
            if (result.projection[component] < minimum[component]) {
                minimum[component] = result.projection[component];
            }
            if (result.projection[component] > maximum[component]) {
                maximum[component] = result.projection[component];
            }
        }
    }
    for (size_t component = 0; component < COMPONENT_COUNT; ++component) {
        double padding = (maximum[component] - minimum[component]) * 0.08;
        if (padding < 0.1) {
            padding = 0.1;
        }
        minimum[component] -= padding;
        maximum[component] += padding;
    }

    GfxCanvas canvas;
    gfx_clear(&canvas, gfx_rgb(250, 250, 250));
    int left = 28;
    int right = GFX_WIDTH - 12;
    int top = 12;
    int bottom = GFX_HEIGHT - 24;
    uint32_t axis_color = gfx_rgb(190, 190, 190);
    if (minimum[0] <= 0.0 && maximum[0] >= 0.0) {
        double fraction = (0.0 - minimum[0]) / (maximum[0] - minimum[0]);
        int x = left + (int)(fraction * (double)(right - left));
        gfx_line(&canvas, x, top, x, bottom, axis_color);
    }
    if (minimum[1] <= 0.0 && maximum[1] >= 0.0) {
        double fraction = (maximum[1] - 0.0) / (maximum[1] - minimum[1]);
        int y = top + (int)(fraction * (double)(bottom - top));
        gfx_line(&canvas, left, y, right, y, axis_color);
    }

    const Dataset *sets[] = {training, test};
    const uint32_t colors[] = {0xff2459a6, 0xffdf6b18};
    for (size_t set = 0; set < 2; ++set) {
        for (size_t i = 0; i < sets[set]->count; ++i) {
            PcaResult result;
            if (!pca_transform(model, sets[set]->values[i], &result)) {
                return false;
            }
            double x_fraction = (result.projection[0] - minimum[0]) /
                                (maximum[0] - minimum[0]);
            double y_fraction = (maximum[1] - result.projection[1]) /
                                (maximum[1] - minimum[1]);
            int x = left + (int)(x_fraction * (double)(right - left));
            int y = top + (int)(y_fraction * (double)(bottom - top));
            gfx_circle(&canvas, x, y, set == 0 ? 2 : 3, colors[set]);
        }
    }
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(&canvas, "pca-projection", path)) {
        return false;
    }
    printf("二维投影图已保存：%s（蓝色：训练；橙色：测试）\n", path);
    return true;
}

/**
 * @brief 运行可复现的 PCA 拟合、投影和独立重建评估。
 * @param argc 命令行参数个数；仅允许无参数或 --export。
 * @param argv 命令行参数数组，借用且不修改。
 * @return 生成、拟合、评估或输出失败为 1；成功为 0。
 */
int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--export") != 0)) {
        fprintf(stderr, "用法：%s [--export]\n", argv[0]);
        return 1;
    }
    Dataset training = {0};
    Dataset test = {0};
    PcaModel model = {0};
    if (!dataset_generate(&training, TRAIN_COUNT, UINT32_C(2025)) ||
        !dataset_generate(&test, TEST_COUNT, UINT32_C(2026)) ||
        !pca_fit(&model, &training)) {
        fputs("PCA 数据生成或拟合失败\n", stderr);
        return 1;
    }

    double training_mse = 0.0;
    double test_mse = 0.0;
    if (!reconstruction_mse(&model, &training, &training_mse) ||
        !reconstruction_mse(&model, &test, &test_mse)) {
        fputs("PCA 投影或重建失败\n", stderr);
        return 1;
    }
    double explained = (model.eigenvalues[0] + model.eigenvalues[1]) /
                       model.total_variance;
    puts("4个相关特征压缩为2个主成分；主成分仅由训练集拟合。");
    printf("训练集均值: [%.3f, %.3f, %.3f, %.3f]\n", model.mean[0], model.mean[1],
           model.mean[2], model.mean[3]);
    printf("主成分1 (方差 %.4f): [%.4f, %.4f, %.4f, %.4f]\n", model.eigenvalues[0],
           model.components[0][0], model.components[0][1], model.components[0][2],
           model.components[0][3]);
    printf("主成分2 (方差 %.4f): [%.4f, %.4f, %.4f, %.4f]\n", model.eigenvalues[1],
           model.components[1][0], model.components[1][1], model.components[1][2],
           model.components[1][3]);
    printf("总方差 %.4f；二维保留方差比例 %.2f%%\n", model.total_variance, 100.0 * explained);
    printf("逐特征平均平方重建误差：训练 %.6f，独立测试 %.6f\n", training_mse, test_mse);
    puts("前3条测试样本投影 (PC1, PC2)：");
    for (size_t i = 0; i < 3; ++i) {
        PcaResult result;
        if (!pca_transform(&model, test.values[i], &result)) {
            return 1;
        }
        printf("[% .3f, % .3f, % .3f, % .3f] -> [% .3f, % .3f]\n",
               test.values[i][0], test.values[i][1], test.values[i][2], test.values[i][3],
               result.projection[0], result.projection[1]);
    }
    if (argc == 2 && !export_projection(&model, &training, &test)) {
        perror("导出 PCA 投影图失败");
        return 1;
    }
    return isfinite(explained) && explained > 0.0 && explained <= 1.0 + 1e-10 &&
                   !ferror(stdout) ? 0 : 1;
}
