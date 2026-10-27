/**
 * @file
 * @brief M01：用二维点与基向量观察矩阵变换及复合顺序。
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double x;
    double y;
} Vec2;

/* 行优先存储：m[行][列]；列向量 e1、e2 的像分别是矩阵的两列。 */
typedef struct {
    double m[2][2];
} Mat2;

typedef struct {
    const char *name;
    Mat2 matrix;
} Scene;

static const Vec2 POINT = {1.0, 2.0};
static const Scene SCENES[] = {
    {"identity", {{{1.0, 0.0}, {0.0, 1.0}}}},
    {"scale", {{{2.0, 0.0}, {0.0, 0.5}}}},
    {"rotate_ccw_90", {{{0.0, -1.0}, {1.0, 0.0}}}},
    {"shear_x", {{{1.0, 1.0}, {0.0, 1.0}}}},
    {"project_x", {{{1.0, 0.0}, {0.0, 0.0}}}},
    {"rotate_after_scale", {{{0.0, -0.5}, {2.0, 0.0}}}},
    {"scale_after_rotate", {{{0.0, -2.0}, {0.5, 0.0}}}},
};

/**
 * @brief 计算列向量的像；只在全部结果有限时写入。
 * @param matrix 行优先矩阵，各元素须有限。
 * @param point 有限输入点或向量。
 * @param output 可写结果，不转移所有权；失败时保持原值。
 * @return 输入、输出均有效时为 true；否则为 false。
 */
static bool mat2_apply(Mat2 matrix, Vec2 point, Vec2 *output)
{
    if (output == nullptr || !isfinite(point.x) || !isfinite(point.y)) {
        return false;
    }
    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 2; ++column) {
            if (!isfinite(matrix.m[row][column])) {
                return false;
            }
        }
    }

    Vec2 result = {
        matrix.m[0][0] * point.x + matrix.m[0][1] * point.y,
        matrix.m[1][0] * point.x + matrix.m[1][1] * point.y,
    };
    if (!isfinite(result.x) || !isfinite(result.y)) {
        return false;
    }
    *output = result;
    return true;
}

/**
 * @brief 计算 left × right，表示对列向量先做 right 再做 left。
 * @param left 有限矩阵，最后作用。
 * @param right 有限矩阵，最先作用。
 * @param output 可写结果，不转移所有权；失败时保持原值。
 * @return 输入或结果无非有限数时为 true；否则为 false。
 */
static bool mat2_multiply(Mat2 left, Mat2 right, Mat2 *output)
{
    if (output == nullptr) {
        return false;
    }
    Mat2 result = {0};
    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 2; ++column) {
            if (!isfinite(left.m[row][column]) || !isfinite(right.m[row][column])) {
                return false;
            }
            for (size_t inner = 0; inner < 2; ++inner) {
                if (!isfinite(left.m[row][inner]) || !isfinite(right.m[inner][column])) {
                    return false;
                }
            }
            result.m[row][column] = left.m[row][0] * right.m[0][column] +
                                    left.m[row][1] * right.m[1][column];
            if (!isfinite(result.m[row][column])) {
                return false;
            }
        }
    }
    *output = result;
    return true;
}

/**
 * @brief 计算二维行列式，供观察定向面积缩放。
 * @param matrix 有限且运算不溢出的矩阵。
 * @return 行列式；输入前提由调用者保证。
 */
static double mat2_determinant(Mat2 matrix)
{
    return matrix.m[0][0] * matrix.m[1][1] - matrix.m[0][1] * matrix.m[1][0];
}

/** @brief 验证手算、线性、复合顺序与失败后输出不变。 */
static void test_transforms(void)
{
    const Mat2 identity = SCENES[0].matrix;
    const Mat2 scale = SCENES[1].matrix;
    const Mat2 rotation = SCENES[2].matrix;
    const Mat2 shear = SCENES[3].matrix;
    const Mat2 projection = SCENES[4].matrix;
    Vec2 result = {99.0, 98.0};

    assert(mat2_apply(identity, POINT, &result));
    assert(result.x == 1.0 && result.y == 2.0);
    assert(mat2_apply(scale, POINT, &result));
    assert(result.x == 2.0 && result.y == 1.0);
    assert(mat2_apply(rotation, POINT, &result));
    assert(result.x == -2.0 && result.y == 1.0);
    assert(result.x * result.x + result.y * result.y == 5.0);
    assert(mat2_apply(shear, POINT, &result));
    assert(result.x == 3.0 && result.y == 2.0);
    assert(mat2_apply(projection, POINT, &result));
    assert(result.x == 1.0 && result.y == 0.0);
    assert(mat2_determinant(projection) == 0.0);
    assert(mat2_determinant(rotation) == 1.0);

    Mat2 rotate_after_scale = {0};
    Mat2 scale_after_rotate = {0};
    assert(mat2_multiply(rotation, scale, &rotate_after_scale));
    assert(mat2_multiply(scale, rotation, &scale_after_rotate));
    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 2; ++column) {
            assert(rotate_after_scale.m[row][column] == SCENES[5].matrix.m[row][column]);
            assert(scale_after_rotate.m[row][column] == SCENES[6].matrix.m[row][column]);
        }
    }
    Vec2 first = {0};
    Vec2 second = {0};
    assert(mat2_apply(rotate_after_scale, POINT, &first));
    assert(mat2_apply(scale_after_rotate, POINT, &second));
    assert(first.x == -1.0 && first.y == 2.0);
    assert(second.x == -4.0 && second.y == 0.5);
    assert(first.x != second.x);

    /* 线性性：对输入的和做变换，等于分别变换后求和。 */
    Vec2 e1 = {1.0, 0.0};
    Vec2 e2 = {0.0, 1.0};
    Vec2 image_e1 = {0};
    Vec2 image_e2 = {0};
    assert(mat2_apply(shear, e1, &image_e1));
    assert(mat2_apply(shear, e2, &image_e2));
    assert(image_e1.x + 2.0 * image_e2.x == 3.0);
    assert(image_e1.y + 2.0 * image_e2.y == 2.0);

    result = (Vec2){99.0, 98.0};
    assert(!mat2_apply(scale, (Vec2){NAN, 1.0}, &result));
    assert(!mat2_apply((Mat2){{{INFINITY, 0.0}, {0.0, 1.0}}}, POINT, &result));
    assert(!mat2_apply((Mat2){{{1e308, 1e308}, {0.0, 1.0}}}, POINT, &result));
    assert(!mat2_apply(scale, POINT, nullptr));
    assert(result.x == 99.0 && result.y == 98.0);

    Mat2 sentinel = identity;
    assert(!mat2_multiply((Mat2){{{NAN, 0.0}, {0.0, 1.0}}}, scale, &sentinel));
    assert(!mat2_multiply((Mat2){{{1e308, 0.0}, {0.0, 1.0}}}, scale, &sentinel));
    assert(!mat2_multiply(scale, rotation, nullptr));
    assert(sentinel.m[0][0] == 1.0 && sentinel.m[1][1] == 1.0);
}

/**
 * @brief 打印原点基、两个变换基和教学点的坐标。
 * @param csv 为 true 时输出便于与 Python 程序比较的 CSV。
 * @return 数据有效则返回成功；否则返回失败。
 */
static int print_scenes(bool csv)
{
    const Vec2 points[] = {{1.0, 0.0}, {0.0, 1.0}, POINT};
    const char *labels[] = {"e1", "e2", "p"};
    if (csv) {
        puts("scene,point,x,y,determinant");
    }
    for (size_t scene = 0; scene < sizeof(SCENES) / sizeof(SCENES[0]); ++scene) {
        double determinant = mat2_determinant(SCENES[scene].matrix);
        if (!isfinite(determinant)) {
            return EXIT_FAILURE;
        }
        for (size_t point = 0; point < sizeof(points) / sizeof(points[0]); ++point) {
            Vec2 image = {0};
            if (!mat2_apply(SCENES[scene].matrix, points[point], &image)) {
                return EXIT_FAILURE;
            }
            if (csv) {
                printf("%s,%s,%.17g,%.17g,%.17g\n", SCENES[scene].name, labels[point],
                       image.x, image.y, determinant);
            } else {
                printf("%-20s %-2s -> (%5.1f, %5.1f) det=%4.1f\n", SCENES[scene].name,
                       labels[point], image.x, image.y, determinant);
            }
        }
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行确定性演示或自检。
 * @param argc 命令行参数数量，允许零个或一个开关。
 * @param argv 命令行参数数组；--csv 输出对照表，--test 执行断言。
 * @return 正常执行为 EXIT_SUCCESS，非法参数或数值失败为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_transforms();
        puts("二维矩阵变换测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1) {
        return print_scenes(false);
    }
    if (argc == 2 && strcmp(argv[1], "--csv") == 0) {
        return print_scenes(true);
    }
    fputs("用法：30_matrix_transforms [--csv | --test]\n", stderr);
    return EXIT_FAILURE;
}
