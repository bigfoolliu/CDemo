/**
 * @file
 * @brief 线性支持向量机：hinge 损失、L2 正则、小批量 SGD 与逻辑回归对照。
 */
#include "../graphics/bmp_capture.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    FEATURE_COUNT = 2,
    TRAIN_COUNT = 96,
    TEST_COUNT = 48,
    BATCH_SIZE = 12,
    EPOCH_COUNT = 500
};

static const double REGULARIZATION = 0.03;

typedef struct {
    double features[FEATURE_COUNT];
    int label;
} Sample;

typedef struct {
    Sample samples[TRAIN_COUNT];
    size_t count;
} Dataset;

typedef struct {
    double weights[FEATURE_COUNT];
    double bias;
} LinearModel;

typedef struct {
    double objective;
    double data_loss;
    size_t errors;
} Metrics;

/**
 * @brief 生成固定种子的 [0,1) 伪随机数。
 * @param state 可写随机状态，由调用者独占维护。
 * @return 更新状态映射到 [0,1) 的实数，不用于安全用途。
 */
static double random_unit(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return (double)(*state >> 8) / 16777216.0;
}

/**
 * @brief 生成二维线性分类数据，加入少量标签翻转以展示软间隔。
 * @param dataset 有效可写数据集；成功后整体替换。
 * @param count 样本数，范围 2..TRAIN_COUNT。
 * @param seed 固定伪随机种子；训练和测试应使用不同种子。
 * @return 生成成功为 true；参数错误时不修改数据集。
 */
static bool dataset_generate(Dataset *dataset, size_t count, uint32_t seed)
{
    if (dataset == nullptr || count < 2 || count > TRAIN_COUNT) {
        return false;
    }
    Dataset generated = {0};
    generated.count = count;
    for (size_t i = 0; i < count; ++i) {
        double x = 2.0 * random_unit(&seed) - 1.0;
        double y = 2.0 * random_unit(&seed) - 1.0;
        double boundary_noise = 0.22 * (2.0 * random_unit(&seed) - 1.0);
        int label = x + 0.65 * y + boundary_noise >= 0.0 ? 1 : -1;
        if (i % 29 == 0) {
            label = -label;
        }
        generated.samples[i] = (Sample){{x, y}, label};
    }
    *dataset = generated;
    return true;
}

/**
 * @brief 计算线性模型的有符号决策分数。
 * @param model 有效只读线性模型。
 * @param features 两个有限输入特征。
 * @return w·x+b；正数预测正类，负数预测负类。
 */
static double decision_value(const LinearModel *model, const double features[FEATURE_COUNT])
{
    double value = model->bias;
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        value += model->weights[feature] * features[feature];
    }
    return value;
}

/**
 * @brief 计算平均 hinge 数据损失与加入 L2 权重惩罚后的 SVM 目标。
 * @param model 有效只读线性模型。
 * @param dataset 非空只读数据集，标签为 -1 或 +1。
 * @param output 有效可写指标；失败时不修改。
 * @return 计算结果有限为 true，否则为 false。
 */
static bool svm_metrics(const LinearModel *model, const Dataset *dataset, Metrics *output)
{
    if (model == nullptr || dataset == nullptr || output == nullptr || dataset->count == 0) {
        return false;
    }
    Metrics metrics = {0};
    for (size_t i = 0; i < dataset->count; ++i) {
        double margin = (double)dataset->samples[i].label *
                        decision_value(model, dataset->samples[i].features);
        metrics.data_loss += fmax(0.0, 1.0 - margin);
        metrics.errors += margin <= 0.0;
    }
    metrics.data_loss /= (double)dataset->count;
    metrics.objective = metrics.data_loss;
    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
        metrics.objective += 0.5 * REGULARIZATION *
                             model->weights[feature] * model->weights[feature];
    }
    if (!isfinite(metrics.objective) || !isfinite(metrics.data_loss)) {
        return false;
    }
    *output = metrics;
    return true;
}

/**
 * @brief 用确定性打乱的小批量次梯度下降拟合软间隔线性 SVM。
 * @param model 可写模型；成功后写入训练结果，不转移资源所有权。
 * @param dataset 至少两个有限特征样本，标签为 -1 或 +1。
 * @return 训练完成且参数有限为 true；失败时模型保持不变。
 */
static bool svm_fit(LinearModel *model, const Dataset *dataset)
{
    if (model == nullptr || dataset == nullptr || dataset->count < 2 ||
        dataset->count > TRAIN_COUNT) {
        return false;
    }
    for (size_t i = 0; i < dataset->count; ++i) {
        if ((dataset->samples[i].label != -1 && dataset->samples[i].label != 1) ||
            !isfinite(dataset->samples[i].features[0]) ||
            !isfinite(dataset->samples[i].features[1])) {
            return false;
        }
    }

    LinearModel fitted = {0};
    size_t order[TRAIN_COUNT];
    uint32_t random_state = UINT32_C(71237);
    for (size_t i = 0; i < dataset->count; ++i) {
        order[i] = i;
    }
    for (size_t epoch = 0; epoch < EPOCH_COUNT; ++epoch) {
        for (size_t i = dataset->count - 1; i > 0; --i) {
            size_t swap_index = (size_t)(random_unit(&random_state) * (double)(i + 1));
            size_t temporary = order[i];
            order[i] = order[swap_index];
            order[swap_index] = temporary;
        }
        double learning_rate = 0.08 / (1.0 + 0.004 * (double)epoch);
        for (size_t start = 0; start < dataset->count; start += BATCH_SIZE) {
            size_t batch_count = dataset->count - start;
            if (batch_count > BATCH_SIZE) {
                batch_count = BATCH_SIZE;
            }
            double weight_gradient[FEATURE_COUNT] = {0};
            double bias_gradient = 0.0;
            for (size_t item = 0; item < batch_count; ++item) {
                const Sample *sample = &dataset->samples[order[start + item]];
                double margin = (double)sample->label *
                                decision_value(&fitted, sample->features);
                if (margin < 1.0) {
                    for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                        weight_gradient[feature] -=
                            (double)sample->label * sample->features[feature];
                    }
                    bias_gradient -= (double)sample->label;
                }
            }
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                weight_gradient[feature] =
                    REGULARIZATION * fitted.weights[feature] +
                    weight_gradient[feature] / (double)batch_count;
                fitted.weights[feature] -= learning_rate * weight_gradient[feature];
            }
            fitted.bias -= learning_rate * bias_gradient / (double)batch_count;
        }
    }
    if (!isfinite(fitted.weights[0]) || !isfinite(fitted.weights[1]) ||
        !isfinite(fitted.bias)) {
        return false;
    }
    *model = fitted;
    return true;
}

/**
 * @brief 稳定计算逻辑回归的 sigmoid，作为相同数据上的对照模型。
 * @param value 有限线性输出。
 * @return [0,1] 概率估计。
 */
static double sigmoid(double value)
{
    if (value >= 0.0) {
        return 1.0 / (1.0 + exp(-value));
    }
    double exponential = exp(value);
    return exponential / (1.0 + exponential);
}

/**
 * @brief 在同一训练数据上用小批量 SGD 拟合带 L2 的逻辑回归。
 * @param model 可写模型；成功后写入训练结果。
 * @param dataset 至少两个有限二维样本，标签为 -1 或 +1。
 * @return 训练成功且参数有限为 true；失败时模型不变。
 */
static bool logistic_fit(LinearModel *model, const Dataset *dataset)
{
    if (model == nullptr || dataset == nullptr || dataset->count < 2 ||
        dataset->count > TRAIN_COUNT) {
        return false;
    }
    for (size_t i = 0; i < dataset->count; ++i) {
        if ((dataset->samples[i].label != -1 && dataset->samples[i].label != 1) ||
            !isfinite(dataset->samples[i].features[0]) ||
            !isfinite(dataset->samples[i].features[1])) {
            return false;
        }
    }
    LinearModel fitted = {0};
    size_t order[TRAIN_COUNT];
    uint32_t random_state = UINT32_C(982451653);
    for (size_t i = 0; i < dataset->count; ++i) {
        order[i] = i;
    }
    for (size_t epoch = 0; epoch < EPOCH_COUNT; ++epoch) {
        for (size_t i = dataset->count - 1; i > 0; --i) {
            size_t swap_index = (size_t)(random_unit(&random_state) * (double)(i + 1));
            size_t temporary = order[i];
            order[i] = order[swap_index];
            order[swap_index] = temporary;
        }
        double learning_rate = 0.06 / (1.0 + 0.004 * (double)epoch);
        for (size_t start = 0; start < dataset->count; start += BATCH_SIZE) {
            size_t batch_count = dataset->count - start;
            if (batch_count > BATCH_SIZE) {
                batch_count = BATCH_SIZE;
            }
            double weight_gradient[FEATURE_COUNT] = {0};
            double bias_gradient = 0.0;
            for (size_t item = 0; item < batch_count; ++item) {
                const Sample *sample = &dataset->samples[order[start + item]];
                double target = sample->label > 0 ? 1.0 : 0.0;
                double error = sigmoid(decision_value(&fitted, sample->features)) - target;
                for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                    weight_gradient[feature] += error * sample->features[feature];
                }
                bias_gradient += error;
            }
            for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
                weight_gradient[feature] =
                    REGULARIZATION * fitted.weights[feature] +
                    weight_gradient[feature] / (double)batch_count;
                fitted.weights[feature] -= learning_rate * weight_gradient[feature];
            }
            fitted.bias -= learning_rate * bias_gradient / (double)batch_count;
        }
    }
    if (!isfinite(fitted.weights[0]) || !isfinite(fitted.weights[1]) ||
        !isfinite(fitted.bias)) {
        return false;
    }
    *model = fitted;
    return true;
}

/**
 * @brief 绘制 SVM 与逻辑回归的分类区域、训练样本和独立测试样本。
 * @param svm 已拟合的 SVM 模型，左侧面板显示其 ±1 间隔。
 * @param logistic 已拟合的逻辑回归模型，右侧面板作分类对照。
 * @param training 只读训练样本。
 * @param test 只读测试样本。
 * @return 图片独占写入 exports/ 成功为 true，否则为 false。
 */
static bool export_comparison(const LinearModel *svm, const LinearModel *logistic,
                              const Dataset *training, const Dataset *test)
{
    GfxCanvas canvas;
    int panel_width = GFX_WIDTH / 2;
    for (int py = 0; py < GFX_HEIGHT; ++py) {
        double y = 1.0 - 2.0 * ((double)py + 0.5) / (double)GFX_HEIGHT;
        for (int px = 0; px < GFX_WIDTH; ++px) {
            size_t panel = (size_t)(px / panel_width);
            int local_x = px % panel_width;
            double features[FEATURE_COUNT] = {
                2.0 * ((double)local_x + 0.5) / (double)panel_width - 1.0, y};
            const LinearModel *model = panel == 0 ? svm : logistic;
            int prediction = decision_value(model, features) >= 0.0 ? 1 : -1;
            canvas.pixels[py][px] = prediction > 0 ? gfx_rgb(251, 218, 194) :
                                                     gfx_rgb(195, 220, 245);
        }
    }

    const LinearModel *models[] = {svm, logistic};
    for (size_t panel = 0; panel < 2; ++panel) {
        for (int level = panel == 0 ? -1 : 0; level <= 1; ++level) {
            if (panel == 1 && level != 0) {
                continue;
            }
            const LinearModel *model = models[panel];
            double offset = panel == 0 ? (double)level : 0.0;
            uint32_t color = level == 0 ? gfx_rgb(35, 35, 35) : gfx_rgb(120, 120, 120);
            if (fabs(model->weights[1]) > 1e-10) {
                double left_y = -(model->weights[0] * -1.0 + model->bias - offset) /
                                model->weights[1];
                double right_y = -(model->weights[0] * 1.0 + model->bias - offset) /
                                 model->weights[1];
                int x1 = (int)(panel * (size_t)panel_width);
                int x2 = x1 + panel_width - 1;
                int y1 = (int)((1.0 - left_y) * (double)(GFX_HEIGHT - 1) / 2.0);
                int y2 = (int)((1.0 - right_y) * (double)(GFX_HEIGHT - 1) / 2.0);
                gfx_line(&canvas, x1, y1, x2, y2, color);
            } else if (fabs(model->weights[0]) > 1e-10) {
                double x_value = -(model->bias - offset) / model->weights[0];
                int x = (int)(panel * (size_t)panel_width +
                              (x_value + 1.0) * (double)panel_width / 2.0);
                gfx_line(&canvas, x, 0, x, GFX_HEIGHT - 1, color);
            }
        }
    }

    const Dataset *sets[] = {training, test};
    for (size_t set = 0; set < 2; ++set) {
        for (size_t panel = 0; panel < 2; ++panel) {
            for (size_t i = 0; i < sets[set]->count; ++i) {
                const Sample *sample = &sets[set]->samples[i];
                int x = (int)(panel * (size_t)panel_width +
                              (sample->features[0] + 1.0) * (double)panel_width / 2.0);
                int y = (int)((1.0 - sample->features[1]) *
                              (double)(GFX_HEIGHT - 1) / 2.0);
                uint32_t color = sample->label > 0 ? gfx_rgb(170, 65, 20) :
                                                     gfx_rgb(25, 60, 145);
                gfx_circle(&canvas, x, y, set == 0 ? 3 : 4,
                           set == 0 ? color : gfx_rgb(255, 255, 255));
                if (set == 1) {
                    gfx_circle(&canvas, x, y, 2, color);
                }
            }
        }
    }
    gfx_line(&canvas, panel_width, 0, panel_width, GFX_HEIGHT - 1, gfx_rgb(50, 50, 50));
    char path[BMP_CAPTURE_PATH_SIZE];
    if (!gfx_bmp_capture(&canvas, "linear-svm", path)) {
        return false;
    }
    printf("分类边界对比图已保存：%s（左：SVM边界与间隔线；右：逻辑回归）\n", path);
    return true;
}

/**
 * @brief 训练并比较线性 SVM 与逻辑回归，可选导出决策边界图。
 * @param argc 命令行参数个数，仅允许无参数或 --export。
 * @param argv 借用的命令行参数数组，不修改。
 * @return 训练、评估或可选导出成功为 0，否则为 1。
 */
int main(int argc, char **argv)
{
    if (argc > 2 || (argc == 2 && strcmp(argv[1], "--export") != 0)) {
        fprintf(stderr, "用法：%s [--export]\n", argv[0]);
        return 1;
    }
    Dataset training = {0};
    Dataset test = {0};
    LinearModel svm = {0};
    LinearModel logistic = {0};
    if (!dataset_generate(&training, TRAIN_COUNT, UINT32_C(20261015)) ||
        !dataset_generate(&test, TEST_COUNT, UINT32_C(20261016)) ||
        !svm_fit(&svm, &training) || !logistic_fit(&logistic, &training)) {
        fputs("线性分类器训练失败\n", stderr);
        return 1;
    }
    Metrics svm_train = {0};
    Metrics svm_test = {0};
    Metrics logistic_train = {0};
    Metrics logistic_test = {0};
    if (!svm_metrics(&svm, &training, &svm_train) ||
        !svm_metrics(&svm, &test, &svm_test) ||
        !svm_metrics(&logistic, &training, &logistic_train) ||
        !svm_metrics(&logistic, &test, &logistic_test)) {
        fputs("分类器评估失败\n", stderr);
        return 1;
    }
    double norm = hypot(svm.weights[0], svm.weights[1]);
    printf("线性 SVM：w=[% .3f, % .3f], b=% .3f, ||w||=%.3f, 几何间隔=%.3f\n",
           svm.weights[0], svm.weights[1], svm.bias, norm, 1.0 / norm);
    printf("SVM hinge 数据损失 %.4f，含 L2 目标 %.4f；训练错误 %zu/%zu，测试错误 %zu/%zu\n",
           svm_train.data_loss, svm_train.objective, svm_train.errors, training.count,
           svm_test.errors, test.count);
    printf("逻辑回归：w=[% .3f, % .3f], b=% .3f；训练错误 %zu/%zu，测试错误 %zu/%zu\n",
           logistic.weights[0], logistic.weights[1], logistic.bias, logistic_train.errors,
           training.count, logistic_test.errors, test.count);
    puts("标签噪声使数据非完全可分；分类错误和间隔用于观察软间隔的取舍。");
    if (argc == 2 && !export_comparison(&svm, &logistic, &training, &test)) {
        perror("导出线性 SVM 图像失败");
        return 1;
    }
    return norm > 0.0 && isfinite(norm) && !ferror(stdout) ? 0 : 1;
}
