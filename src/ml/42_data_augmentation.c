/**
 * @file
 * @brief D02：用可重复的图像分类实验比较数据增强及标签破坏。
 * 固定容量、纯 C 实现；以小型线性 Softmax 分类器隔离增强数据流，不依赖框架。
 */
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SIDE 8
#define PIXELS (SIDE * SIDE)
#define CLASSES 3
#define TRAIN_COUNT 72
#define EVALUATION_COUNT 36
#define MAX_SAMPLES TRAIN_COUNT
#define BATCH_SIZE 9
#define EPOCHS 120
#define RUN_COUNT 5
#define WEIGHTS (CLASSES * PIXELS)
#define PARAMETERS (WEIGHTS + CLASSES)
#define NOISE_AMPLITUDE 0.12
#define LEARNING_RATE 0.08

typedef struct {
    double pixels[PIXELS];
    size_t target;
    size_t source_id;
} Image;

typedef struct {
    size_t count;
    Image samples[MAX_SAMPLES];
} Dataset;

typedef struct {
    double weights[WEIGHTS];
    double biases[CLASSES];
} Model;

typedef struct {
    double logits[CLASSES];
    double probabilities[CLASSES];
    double loss;
} Forward;

typedef struct {
    double loss;
    double accuracy;
} Metrics;

typedef enum {
    AUGMENT_NONE,
    AUGMENT_TRANSLATION,
    AUGMENT_NOISE,
    AUGMENT_COMBINED,
    AUGMENT_BAD_ROTATION,
    AUGMENTATION_COUNT
} Augmentation;

typedef enum {
    VIEW_CLEAN,
    VIEW_SHIFTED,
    VIEW_NOISY,
    VIEW_SHIFTED_NOISY,
    VIEW_COUNT
} EvaluationView;

typedef struct {
    Metrics validation;
    Metrics test[VIEW_COUNT];
} RunResult;

static const char *augmentation_names[AUGMENTATION_COUNT] = {
    "none", "translation", "noise", "combined", "bad_rotate"
};

static uint32_t random_next(uint32_t *state)
{
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static double random_unit(uint32_t *state)
{
    return (double)(random_next(state) >> 8) / 16777216.0;
}

static bool image_is_valid(const Image *image)
{
    if (image == nullptr || image->target >= CLASSES) {
        return false;
    }

    for (size_t i = 0; i < PIXELS; ++i) {
        if (!isfinite(image->pixels[i]) || image->pixels[i] < 0 || image->pixels[i] > 1) {
            return false;
        }
    }
    return true;
}

static bool dataset_generate(Dataset *dataset, size_t count, uint32_t seed, size_t source_group)
{
    if (dataset == nullptr || count == 0 || count > MAX_SAMPLES || count % CLASSES != 0) {
        return false;
    }
    Dataset candidate = {0};
    candidate.count = count;
    for (size_t i = 0; i < count; ++i) {
        Image *image = &candidate.samples[i];
        uint32_t state = seed + (uint32_t)(i + 1) * UINT32_C(2654435761);
        image->target = i % CLASSES;
        image->source_id = source_group * 1024 + i;
        double center_x = 3.5 + 0.5 * (random_unit(&state) - 0.5);
        double center_y = 3.5 + 0.5 * (random_unit(&state) - 0.5);
        double brightness = 0.8 + 0.2 * random_unit(&state);
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                double dx = (double)column - center_x;
                double dy = (double)row - center_y;
                bool inside = fabs(dx) <= 1 && fabs(dy) <= 1;
                if (image->target == 1) {
                    inside = fabs(dx) <= 0.5 && fabs(dy) <= 1.5;
                } else if (image->target == 2) {
                    inside = fabs(dx) <= 1.5 && fabs(dy) <= 0.5;
                }
                image->pixels[row * SIDE + column] = inside ? brightness : 0;
            }
        }
    }
    *dataset = candidate;
    return true;
}

static bool translate_pixels(const double source[PIXELS], double output[PIXELS], int dx, int dy)
{
    if (source == nullptr || output == nullptr || dx < -1 || dx > 1 || dy < -1 || dy > 1) {
        return false;
    }
    for (size_t i = 0; i < PIXELS; ++i) {
        if (!isfinite(source[i]) || source[i] < 0 || source[i] > 1) {
            return false;
        }
    }
    double translated[PIXELS] = {0};
    for (size_t row = 0; row < SIDE; ++row) {
        for (size_t column = 0; column < SIDE; ++column) {
            double value = source[row * SIDE + column];
            int target_row = (int)row + dy;
            int target_column = (int)column + dx;
            if (value != 0 && (target_row < 0 || target_row >= SIDE ||
                               target_column < 0 || target_column >= SIDE)) {
                return false;
            }
            if (target_row >= 0 && target_row < SIDE &&
                target_column >= 0 && target_column < SIDE) {
                translated[(size_t)target_row * SIDE + (size_t)target_column] = value;
            }
        }
    }
    memcpy(output, translated, sizeof translated);
    return true;
}

static void add_noise(double pixels[PIXELS], uint32_t *state, double amplitude)
{
    for (size_t i = 0; i < PIXELS; ++i) {
        double noise = amplitude * (2 * random_unit(state) - 1);
        pixels[i] = fmin(1, fmax(0, pixels[i] + noise));
    }
}

static bool augment_image(const Image *source, Image *output, Augmentation augmentation,
                          uint32_t seed)
{
    if (output == nullptr || !image_is_valid(source) ||
        augmentation < AUGMENT_NONE || augmentation >= AUGMENTATION_COUNT) {
        return false;
    }
    Image candidate = *source;
    uint32_t state = seed;
    if (augmentation == AUGMENT_TRANSLATION || augmentation == AUGMENT_COMBINED) {
        int dx = (int)(random_unit(&state) * 3) - 1;
        int dy = (int)(random_unit(&state) * 3) - 1;
        if (!translate_pixels(source->pixels, candidate.pixels, dx, dy)) {
            return false;
        }
    } else if (augmentation == AUGMENT_BAD_ROTATION && source->target != 0) {
        for (size_t row = 0; row < SIDE; ++row) {
            for (size_t column = 0; column < SIDE; ++column) {
                candidate.pixels[column * SIDE + (SIDE - 1 - row)] =
                    source->pixels[row * SIDE + column];
            }
        }
    }
    if (augmentation == AUGMENT_NOISE || augmentation == AUGMENT_COMBINED) {
        add_noise(candidate.pixels, &state, NOISE_AMPLITUDE);
    }
    *output = candidate;
    return true;
}

static bool evaluation_image(const Image *source, Image *output, EvaluationView view, size_t index)
{
    if (output == nullptr || !image_is_valid(source) || view < VIEW_CLEAN || view >= VIEW_COUNT) {
        return false;
    }
    Image candidate = *source;
    if (view == VIEW_SHIFTED || view == VIEW_SHIFTED_NOISY) {
        int dx = (int)(index % 3) - 1;
        int dy = (int)((index / 3) % 3) - 1;
        if (!translate_pixels(source->pixels, candidate.pixels, dx, dy)) {
            return false;
        }
    }
    if (view == VIEW_NOISY || view == VIEW_SHIFTED_NOISY) {
        uint32_t state = UINT32_C(0xD02A0000) + (uint32_t)index;
        add_noise(candidate.pixels, &state, NOISE_AMPLITUDE);
    }
    *output = candidate;
    return true;
}

static bool model_is_valid(const Model *model)
{
    if (model == nullptr) {
        return false;
    }
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double value = i < WEIGHTS ? model->weights[i] : model->biases[i - WEIGHTS];
        if (!isfinite(value)) {
            return false;
        }
    }
    return true;
}

static bool model_forward(const Model *model, const Image *image, Forward *output)
{
    if (output == nullptr || !model_is_valid(model) || !image_is_valid(image)) {
        return false;
    }
    Forward candidate = {0};
    double maximum = -INFINITY;
    for (size_t category = 0; category < CLASSES; ++category) {
        candidate.logits[category] = model->biases[category];
        for (size_t pixel = 0; pixel < PIXELS; ++pixel) {
            candidate.logits[category] +=
                model->weights[category * PIXELS + pixel] * image->pixels[pixel];
        }
        if (!isfinite(candidate.logits[category])) {
            return false;
        }
        maximum = fmax(maximum, candidate.logits[category]);
    }
    double sum = 0;
    for (size_t category = 0; category < CLASSES; ++category) {
        candidate.probabilities[category] = exp(candidate.logits[category] - maximum);
        sum += candidate.probabilities[category];
    }
    if (!isfinite(sum) || sum <= 0) {
        return false;
    }
    for (size_t category = 0; category < CLASSES; ++category) {
        candidate.probabilities[category] /= sum;
    }
    candidate.loss = maximum - candidate.logits[image->target] + log(sum);
    if (!isfinite(candidate.loss)) {
        return false;
    }
    *output = candidate;
    return true;
}

static bool model_gradient(const Model *model, const Image *image, double gradient[PARAMETERS],
                           double *loss)
{
    if (gradient == nullptr || loss == nullptr) {
        return false;
    }
    Forward forward;
    if (!model_forward(model, image, &forward)) {
        return false;
    }
    double candidate[PARAMETERS] = {0};
    for (size_t category = 0; category < CLASSES; ++category) {
        double delta = forward.probabilities[category] - (double)(category == image->target);
        candidate[WEIGHTS + category] = delta;
        for (size_t pixel = 0; pixel < PIXELS; ++pixel) {
            candidate[category * PIXELS + pixel] = delta * image->pixels[pixel];
        }
    }
    memcpy(gradient, candidate, sizeof candidate);
    *loss = forward.loss;
    return true;
}

static size_t predict(const Forward *forward)
{
    size_t prediction = 0;
    for (size_t category = 1; category < CLASSES; ++category) {
        if (forward->probabilities[category] > forward->probabilities[prediction]) {
            prediction = category;
        }
    }
    return prediction;
}

static Metrics evaluate(const Model *model, const Dataset *dataset, EvaluationView view)
{
    Metrics metrics = {0};
    for (size_t i = 0; i < dataset->count; ++i) {
        Image image;
        Forward forward;
        bool transformed = evaluation_image(&dataset->samples[i], &image, view, i);
        assert(transformed);
        bool valid = model_forward(model, &image, &forward);
        assert(valid);
        metrics.loss += forward.loss;
        metrics.accuracy += (double)(predict(&forward) == image.target);
    }
    metrics.loss /= (double)dataset->count;
    metrics.accuracy /= (double)dataset->count;
    return metrics;
}

static void model_initialize(Model *model, uint32_t seed)
{
    *model = (Model){0};
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double value = 0.04 * (2 * random_unit(&seed) - 1);
        if (i < WEIGHTS) {
            model->weights[i] = value;
        } else {
            model->biases[i - WEIGHTS] = value;
        }
    }
}

static void train_model(Model *model, const Dataset *training, Augmentation augmentation,
                        uint32_t run_seed)
{
    uint32_t shuffle_state = run_seed ^ UINT32_C(0x2468ACE0);
    for (size_t epoch = 0; epoch < EPOCHS; ++epoch) {
        size_t order[TRAIN_COUNT];
        for (size_t i = 0; i < training->count; ++i) {
            order[i] = i;
        }
        for (size_t remaining = training->count; remaining > 1; --remaining) {
            size_t selected = (size_t)(random_unit(&shuffle_state) * (double)remaining);
            size_t last = remaining - 1;
            size_t temporary = order[last];
            order[last] = order[selected];
            order[selected] = temporary;
        }
        for (size_t start = 0; start < training->count; start += BATCH_SIZE) {
            double gradient[PARAMETERS] = {0};
            size_t count = training->count - start;
            if (count > BATCH_SIZE) {
                count = BATCH_SIZE;
            }
            for (size_t batch_index = 0; batch_index < count; ++batch_index) {
                size_t sample_index = order[start + batch_index];
                const Image *source = &training->samples[sample_index];
                Image transformed;
                uint32_t augmentation_seed = run_seed ^
                    ((uint32_t)(epoch + 1) * UINT32_C(0x9E3779B9)) ^
                    ((uint32_t)(sample_index + 1) * UINT32_C(0x85EBCA6B));
                bool augmented = augment_image(source, &transformed, augmentation, augmentation_seed);
                assert(augmented);
                double sample_gradient[PARAMETERS];
                double loss = 0;
                bool valid = model_gradient(model, &transformed, sample_gradient, &loss);
                assert(valid && isfinite(loss));
                for (size_t parameter = 0; parameter < PARAMETERS; ++parameter) {
                    gradient[parameter] += sample_gradient[parameter] / (double)count;
                }
            }
            for (size_t parameter = 0; parameter < PARAMETERS; ++parameter) {
                if (parameter < WEIGHTS) {
                    model->weights[parameter] -= LEARNING_RATE * gradient[parameter];
                } else {
                    model->biases[parameter - WEIGHTS] -= LEARNING_RATE * gradient[parameter];
                }
            }
        }
    }
}

static RunResult run_one(const Dataset *training, const Dataset *validation, const Dataset *testing,
                         Augmentation augmentation, uint32_t seed)
{
    Model model;
    model_initialize(&model, seed ^ UINT32_C(0x13579BDF));
    train_model(&model, training, augmentation, seed);
    RunResult result = {0};
    result.validation = evaluate(&model, validation, VIEW_CLEAN);
    for (size_t view = 0; view < VIEW_COUNT; ++view) {
        result.test[view] = evaluate(&model, testing, (EvaluationView)view);
    }
    return result;
}

static RunResult all_results[AUGMENTATION_COUNT][RUN_COUNT];

static void run_experiment(const Dataset *training, const Dataset *validation,
                           const Dataset *testing)
{
    for (size_t augmentation = 0; augmentation < AUGMENTATION_COUNT; ++augmentation) {
        for (size_t run = 0; run < RUN_COUNT; ++run) {
            uint32_t seed = 42 + (uint32_t)run;
            all_results[augmentation][run] = run_one(training, validation, testing,
                                                      (Augmentation)augmentation, seed);
        }
    }
}

static Metrics aggregate_metric(size_t augmentation, size_t view, bool validation, bool accuracy)
{
    double values[RUN_COUNT];
    double mean = 0;
    for (size_t run = 0; run < RUN_COUNT; ++run) {
        const RunResult *result = &all_results[augmentation][run];
        double value = validation ? result->validation.loss :
                       (accuracy ? result->test[view].accuracy : result->test[view].loss);
        values[run] = value;
        mean += value / (double)RUN_COUNT;
    }
    double variance = 0;
    for (size_t run = 0; run < RUN_COUNT; ++run) {
        double difference = values[run] - mean;
        variance += difference * difference / (double)RUN_COUNT;
    }
    return (Metrics){.loss = mean, .accuracy = sqrt(variance)};
}

static void print_csv(void)
{
    puts("augmentation,seed,validation_loss,validation_accuracy,clean_loss,clean_accuracy,"
         "shifted_loss,shifted_accuracy,noisy_loss,noisy_accuracy,shifted_noisy_loss,"
         "shifted_noisy_accuracy");
    for (size_t augmentation = 0; augmentation < AUGMENTATION_COUNT; ++augmentation) {
        for (size_t run = 0; run < RUN_COUNT; ++run) {
            const RunResult *result = &all_results[augmentation][run];
            printf("%s,%u,%.12f,%.12f", augmentation_names[augmentation],
                   42u + (unsigned)run, result->validation.loss, result->validation.accuracy);
            for (size_t view = 0; view < VIEW_COUNT; ++view) {
                printf(",%.12f,%.12f", result->test[view].loss, result->test[view].accuracy);
            }
            putchar('\n');
        }
    }
}

static void print_summary(void)
{
    printf("固定划分：训练/验证/测试=%d/%d/%d；每项 %d 轮、%d 批/轮、%d 次种子重复。\n",
           TRAIN_COUNT, EVALUATION_COUNT, EVALUATION_COUNT, EPOCHS,
           TRAIN_COUNT / BATCH_SIZE, RUN_COUNT);
    puts("增强方法              验证损失均值±标准差   干净准确率   平移准确率   噪声准确率   平移+噪声准确率");
    for (size_t augmentation = 0; augmentation < AUGMENTATION_COUNT; ++augmentation) {
        Metrics validation = aggregate_metric(augmentation, 0, true, false);
        printf("%-20s %.4f ± %.4f", augmentation_names[augmentation],
               validation.loss, validation.accuracy);
        for (size_t view = 0; view < VIEW_COUNT; ++view) {
            Metrics metric = aggregate_metric(augmentation, view, false, true);
            printf("     %.3f ± %.3f", metric.loss, metric.accuracy);
        }
        putchar('\n');
    }
    puts("测试视图在所有训练和验证选择之外固定；标准差只反映五个训练种子的波动。");
}

static void print_hand_example(void)
{
    Image grid = {0};
    grid.pixels[3 * SIDE + 3] = 1;
    double translated[PIXELS] = {0};
    bool moved = translate_pixels(grid.pixels, translated, 1, 0);
    assert(moved);
    uint32_t noise_state = 1;
    add_noise(translated, &noise_state, NOISE_AMPLITUDE);
    puts("变换手算：像素 (3,3) 右移到 (3,4)，再用种子 1 加噪；局部 3×3：");
    for (size_t row = 3; row < 6; ++row) {
        for (size_t column = 3; column < 6; ++column) {
            printf("%.6f%s", translated[row * SIDE + column], column == 5 ? "\n" : " ");
        }
    }

    Model model = {0};
    Image image = {0};
    image.target = 0;
    image.pixels[3 * SIDE + 3] = 1;
    model.weights[3 * SIDE + 3] = 0.2;
    model.weights[PIXELS + 3 * SIDE + 3] = 0;
    model.weights[2 * PIXELS + 3 * SIDE + 3] = -0.2;
    Forward forward;
    double gradients[PARAMETERS];
    double loss = 0;
    bool valid = model_forward(&model, &image, &forward) &&
                 model_gradient(&model, &image, gradients, &loss);
    assert(valid);
    printf("logits=[%.12f, %.12f, %.12f]\n", forward.logits[0], forward.logits[1],
           forward.logits[2]);
    printf("probabilities=[%.12f, %.12f, %.12f], loss=%.12f\n",
           forward.probabilities[0], forward.probabilities[1],
           forward.probabilities[2], loss);
    printf("active-pixel gradient=[%.12f, %.12f, %.12f]\n",
           gradients[3 * SIDE + 3], gradients[PIXELS + 3 * SIDE + 3],
           gradients[2 * PIXELS + 3 * SIDE + 3]);
    printf("one SGD step (rate=0.08)=[%.12f, %.12f, %.12f]\n",
           model.weights[3 * SIDE + 3] - LEARNING_RATE * gradients[3 * SIDE + 3],
           model.weights[PIXELS + 3 * SIDE + 3] -
               LEARNING_RATE * gradients[PIXELS + 3 * SIDE + 3],
           model.weights[2 * PIXELS + 3 * SIDE + 3] -
               LEARNING_RATE * gradients[2 * PIXELS + 3 * SIDE + 3]);
}

static void test_transformations(void)
{
    Image source = {0};
    source.target = 2;
    source.source_id = 17;
    source.pixels[3 * SIDE + 3] = 1;
    Image moved = {0};
    assert(translate_pixels(source.pixels, moved.pixels, 1, 0));
    assert(moved.pixels[3 * SIDE + 4] == 1);
    assert(source.pixels[3 * SIDE + 3] == 1);

    Image output = {.target = 1, .source_id = 99, .pixels = {0.25}};
    Image before = output;
    source.pixels[0] = 1;
    assert(!translate_pixels(source.pixels, output.pixels, -1, 0));
    assert(memcmp(&output, &before, sizeof output) == 0);
    source.pixels[0] = 0;
    for (size_t column = 2; column < 6; ++column) {
        source.pixels[3 * SIDE + column] = 1;
    }
    assert(augment_image(&source, &moved, AUGMENT_BAD_ROTATION, 1));
    assert(moved.target == source.target && moved.source_id == source.source_id);
    for (size_t row = 2; row < 6; ++row) {
        assert(moved.pixels[row * SIDE + 4] == 1);
    }
    Image noisy = {0};
    Image noisy_again = {0};
    assert(augment_image(&source, &noisy, AUGMENT_NOISE, 2025));
    assert(augment_image(&source, &noisy_again, AUGMENT_NOISE, 2025));
    assert(noisy.target == source.target && noisy.source_id == source.source_id);
    assert(memcmp(&noisy, &noisy_again, sizeof noisy) == 0);
    for (size_t i = 0; i < PIXELS; ++i) {
        assert(noisy.pixels[i] >= 0 && noisy.pixels[i] <= 1);
    }
    assert(!augment_image(&source, &output, (Augmentation)99, 1));
    assert(memcmp(&output, &before, sizeof output) == 0);
    source.pixels[0] = NAN;
    assert(!augment_image(&source, &output, AUGMENT_NONE, 1));
    assert(memcmp(&output, &before, sizeof output) == 0);
}

static void test_gradient(void)
{
    Model model = {0};
    Image image = {0};
    image.target = 0;
    image.pixels[9] = 1;
    for (size_t i = 0; i < PARAMETERS; ++i) {
        double value = 0.03 * (double)((int)(i % 7) - 3);
        if (i < WEIGHTS) {
            model.weights[i] = value;
        } else {
            model.biases[i - WEIGHTS] = value;
        }
    }
    double gradients[PARAMETERS];
    double loss = 0;
    assert(model_gradient(&model, &image, gradients, &loss));
    const double step = 1e-6;
    for (size_t parameter = 0; parameter < PARAMETERS; ++parameter) {
        Model plus = model;
        Model minus = model;
        if (parameter < WEIGHTS) {
            plus.weights[parameter] += step;
            minus.weights[parameter] -= step;
        } else {
            size_t bias = parameter - WEIGHTS;
            plus.biases[bias] += step;
            minus.biases[bias] -= step;
        }
        Forward high;
        Forward low;
        assert(model_forward(&plus, &image, &high));
        assert(model_forward(&minus, &image, &low));
        double numerical = (high.loss - low.loss) / (2 * step);
        assert(fabs(numerical - gradients[parameter]) < 2e-7);
    }
    assert(isfinite(loss));
}

static void test_splits_and_repeatability(void)
{
    Dataset training = {0};
    Dataset validation = {0};
    Dataset testing = {0};
    assert(dataset_generate(&training, TRAIN_COUNT, 2025, 1));
    assert(dataset_generate(&validation, EVALUATION_COUNT, 2026, 2));
    assert(dataset_generate(&testing, EVALUATION_COUNT, 2027, 3));
    Dataset preserved = training;
    assert(!dataset_generate(&training, TRAIN_COUNT + 1, 2025, 1));
    assert(memcmp(&training, &preserved, sizeof training) == 0);
    for (size_t i = 0; i < training.count; ++i) {
        assert(training.samples[i].source_id / 1024 != validation.samples[0].source_id / 1024);
        assert(training.samples[i].source_id / 1024 != testing.samples[0].source_id / 1024);
    }
    for (size_t i = 0; i < training.count; ++i) {
        assert(training.samples[i].target == i % CLASSES);
    }
    RunResult first = run_one(&training, &validation, &testing, AUGMENT_TRANSLATION, 42);
    RunResult second = run_one(&training, &validation, &testing, AUGMENT_TRANSLATION, 42);
    assert(first.validation.loss == second.validation.loss);
    assert(first.validation.accuracy == second.validation.accuracy);
    for (size_t view = 0; view < VIEW_COUNT; ++view) {
        assert(first.test[view].loss == second.test[view].loss);
        assert(first.test[view].accuracy == second.test[view].accuracy);
    }
    Image transformed;
    assert(augment_image(&training.samples[0], &transformed, AUGMENT_COMBINED, 42));
    assert(training.samples[0].source_id == transformed.source_id);
    assert(memcmp(&training.samples[0], &validation.samples[0], sizeof(Image)) != 0);
}

static int run_tests(void)
{
    test_transformations();
    test_gradient();
    test_splits_and_repeatability();
    puts("D02 C 检查通过：标签变换、失败原子性、195 个参数梯度、分组划分和固定种子复现。");
    return EXIT_SUCCESS;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        return run_tests();
    }
    if (argc == 2 && strcmp(argv[1], "--hand") == 0) {
        print_hand_example();
        return EXIT_SUCCESS;
    }
    if (argc != 1 && !(argc == 2 && strcmp(argv[1], "--csv") == 0)) {
        fprintf(stderr, "用法：%s [--hand | --test | --csv]\n", argv[0]);
        return EXIT_FAILURE;
    }
    static Dataset training;
    static Dataset validation;
    static Dataset testing;
    if (!dataset_generate(&training, TRAIN_COUNT, 2025, 1) ||
        !dataset_generate(&validation, EVALUATION_COUNT, 2026, 2) ||
        !dataset_generate(&testing, EVALUATION_COUNT, 2027, 3)) {
        fputs("D02 数据集尺寸无效\n", stderr);
        return EXIT_FAILURE;
    }
    run_experiment(&training, &validation, &testing);
    if (argc == 2) {
        print_csv();
    } else {
        print_summary();
    }
    return EXIT_SUCCESS;
}
