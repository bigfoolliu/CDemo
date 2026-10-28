/**
 * @file
 * @brief M07：小型 QR 最小二乘、正规方程与二维奇异值分解。
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_ROWS = 4, COLUMNS = 2 };

typedef struct {
    double q[MAX_ROWS][COLUMNS];
    double r[COLUMNS][COLUMNS];
    double x[COLUMNS];
    double residual;
} QrResult;

typedef struct {
    double u[2][2];
    double v[2][2];
    double sigma[2];
    double reconstruction_error;
} SvdResult;

/**
 * @brief 以两遍 Gram-Schmidt 分解满列秩矩阵，再解最小二乘。
 * @param a 行优先的两列矩阵，至少有 rows 行；所有元素须有限。
 * @param b 长度至少为 rows 的有限目标向量。
 * @param rows 行数，范围为 2..MAX_ROWS。
 * @param output 可写结果，不转移所有权；失败时保持原值。
 * @return 成功为 true；形状、秩或数值无效时为 false。
 */
static bool qr_solve(const double a[][COLUMNS], const double *b,
                     size_t rows, QrResult *output)
{
    if (a == nullptr || b == nullptr || output == nullptr ||
        rows < COLUMNS || rows > MAX_ROWS) {
        return false;
    }
    QrResult next = {0};
    double norm0 = 0;
    double norm1 = 0;
    for (size_t i = 0; i < rows; ++i) {
        if (!isfinite(a[i][0]) || !isfinite(a[i][1]) || !isfinite(b[i])) {
            return false;
        }
        norm0 = hypot(norm0, a[i][0]);
        norm1 = hypot(norm1, a[i][1]);
    }
    if (!isfinite(norm0) || !isfinite(norm1) || norm0 == 0 || norm1 == 0) {
        return false;
    }
    next.r[0][0] = norm0;
    for (size_t i = 0; i < rows; ++i) {
        next.q[i][0] = a[i][0] / norm0;
        next.r[0][1] += next.q[i][0] * a[i][1];
    }
    for (size_t i = 0; i < rows; ++i) {
        next.q[i][1] = a[i][1] - next.r[0][1] * next.q[i][0];
    }
    double correction = 0;
    for (size_t i = 0; i < rows; ++i) {
        correction += next.q[i][0] * next.q[i][1];
    }
    next.r[0][1] += correction;
    for (size_t i = 0; i < rows; ++i) {
        next.q[i][1] -= correction * next.q[i][0];
        next.r[1][1] = hypot(next.r[1][1], next.q[i][1]);
    }
    if (!isfinite(next.r[1][1]) || next.r[1][1] <= 1e-12 * norm1) {
        return false;
    }
    double projected[2] = {0};
    for (size_t i = 0; i < rows; ++i) {
        next.q[i][1] /= next.r[1][1];
        projected[0] += next.q[i][0] * b[i];
        projected[1] += next.q[i][1] * b[i];
    }
    next.x[1] = projected[1] / next.r[1][1];
    next.x[0] = (projected[0] - next.r[0][1] * next.x[1]) / norm0;
    for (size_t i = 0; i < rows; ++i) {
        double error = b[i] - a[i][0] * next.x[0] - a[i][1] * next.x[1];
        next.residual = hypot(next.residual, error);
    }
    if (!isfinite(next.x[0]) || !isfinite(next.x[1]) || !isfinite(next.residual)) {
        return false;
    }
    *output = next;
    return true;
}

/**
 * @brief 用正规方程作为病态数据对照，非通用线性求解器。
 * @param a 行优先两列矩阵，至少 rows 行且元素有限。
 * @param b 长度至少 rows 的有限向量。
 * @param rows 行数，范围为 2..MAX_ROWS。
 * @param x 长度为 2 的可写输出；失败时保持原值。
 * @return 行列式为正且结果有限时为 true，否则为 false。
 */
static bool normal_solve(const double a[][COLUMNS], const double *b,
                         size_t rows, double x[2])
{
    if (a == nullptr || b == nullptr || x == nullptr ||
        rows < COLUMNS || rows > MAX_ROWS) {
        return false;
    }
    double s00 = 0;
    double s01 = 0;
    double s11 = 0;
    double t0 = 0;
    double t1 = 0;
    for (size_t i = 0; i < rows; ++i) {
        if (!isfinite(a[i][0]) || !isfinite(a[i][1]) || !isfinite(b[i])) {
            return false;
        }
        s00 += a[i][0] * a[i][0];
        s01 += a[i][0] * a[i][1];
        s11 += a[i][1] * a[i][1];
        t0 += a[i][0] * b[i];
        t1 += a[i][1] * b[i];
    }
    double determinant = s00 * s11 - s01 * s01;
    if (!isfinite(determinant) || determinant <= 0) {
        return false;
    }
    double next[2] = {
        (t0 * s11 - t1 * s01) / determinant,
        (s00 * t1 - s01 * t0) / determinant
    };
    if (!isfinite(next[0]) || !isfinite(next[1])) {
        return false;
    }
    memcpy(x, next, sizeof(next));
    return true;
}

/**
 * @brief 从 A 的右奇异方向构造二维满秩 SVD，并检查重建。
 * @param a 有限的 2×2 矩阵；教学算法要求两个非零奇异值。
 * @param output 可写分解结果，失败时不修改。
 * @return 满秩且正交、重建检查通过时为 true，否则为 false。
 */
static bool svd2(const double a[2][2], SvdResult *output)
{
    if (a == nullptr || output == nullptr) {
        return false;
    }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            if (!isfinite(a[i][j])) {
                return false;
            }
        }
    }
    double g00 = a[0][0] * a[0][0] + a[1][0] * a[1][0];
    double g01 = a[0][0] * a[0][1] + a[1][0] * a[1][1];
    double g11 = a[0][1] * a[0][1] + a[1][1] * a[1][1];
    double angle = 0.5 * atan2(2 * g01, g00 - g11);
    SvdResult next = {0};
    next.v[0][0] = cos(angle);
    next.v[1][0] = sin(angle);
    next.v[0][1] = -sin(angle);
    next.v[1][1] = cos(angle);
    for (size_t k = 0; k < 2; ++k) {
        double y0 = a[0][0] * next.v[0][k] + a[0][1] * next.v[1][k];
        double y1 = a[1][0] * next.v[0][k] + a[1][1] * next.v[1][k];
        next.sigma[k] = hypot(y0, y1);
        if (!isfinite(next.sigma[k]) || next.sigma[k] == 0) {
            return false;
        }
        next.u[0][k] = y0 / next.sigma[k];
        next.u[1][k] = y1 / next.sigma[k];
    }
    if (next.sigma[1] <= 1e-12 * next.sigma[0]) {
        return false;
    }
    double dot = next.u[0][0] * next.u[0][1] + next.u[1][0] * next.u[1][1];
    if (!isfinite(dot) || fabs(dot) > 1e-10 || next.sigma[0] < next.sigma[1]) {
        return false;
    }
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            double restored = 0;
            for (size_t k = 0; k < 2; ++k) {
                restored += next.u[i][k] * next.sigma[k] * next.v[j][k];
            }
            next.reconstruction_error = hypot(next.reconstruction_error, restored - a[i][j]);
        }
    }
    if (!isfinite(next.reconstruction_error) || next.reconstruction_error > 1e-9) {
        return false;
    }
    *output = next;
    return true;
}

/** @brief 检查手算、正交性、病态误差、秩亏与失败保留。 */
static void test_decomposition(void)
{
    const double a[3][2] = {{1, 0}, {1, 1}, {1, 2}};
    const double b[3] = {1, 2, 2};
    QrResult qr = {0};
    assert(qr_solve(a, b, 3, &qr));
    assert(fabs(qr.x[0] - 7.0 / 6) < 1e-12);
    assert(fabs(qr.x[1] - 0.5) < 1e-12);
    assert(fabs(qr.residual - sqrt(1.0 / 6)) < 1e-12);
    double qdot = 0;
    for (size_t i = 0; i < 3; ++i) {
        qdot += qr.q[i][0] * qr.q[i][1];
    }
    assert(fabs(qdot) < 1e-12);
    for (size_t column = 0; column < 2; ++column) {
        double length = 0;
        for (size_t i = 0; i < 3; ++i) {
            length += qr.q[i][column] * qr.q[i][column];
            double restored = qr.q[i][0] * qr.r[0][column] +
                              qr.q[i][1] * qr.r[1][column];
            assert(fabs(restored - a[i][column]) < 1e-12);
        }
        assert(fabs(length - 1) < 1e-12);
    }
    const double deficient[3][2] = {{1, 2}, {2, 4}, {3, 6}};
    QrResult untouched = {.residual = 99};
    assert(!qr_solve(deficient, b, 3, &untouched));
    assert(untouched.residual == 99);
    assert(!qr_solve(a, b, 1, &untouched));
    const double invalid[3][2] = {{1, 0}, {1, NAN}, {1, 2}};
    assert(!qr_solve(invalid, b, 3, &untouched));
    assert(untouched.residual == 99);
    double normal_saved[2] = {99, 98};
    assert(!normal_solve(deficient, b, 3, normal_saved));
    assert(normal_saved[0] == 99 && normal_saved[1] == 98);

    const double image[2][2] = {{2, 1}, {1, 2}};
    SvdResult svd = {0};
    assert(svd2(image, &svd));
    assert(fabs(svd.sigma[0] - 3) < 1e-12);
    assert(fabs(svd.sigma[1] - 1) < 1e-12);
    assert(svd.reconstruction_error < 1e-12);
    for (size_t k = 0; k < 2; ++k) {
        double u_length = svd.u[0][k] * svd.u[0][k] + svd.u[1][k] * svd.u[1][k];
        double v_length = svd.v[0][k] * svd.v[0][k] + svd.v[1][k] * svd.v[1][k];
        assert(fabs(u_length - 1) < 1e-12);
        assert(fabs(v_length - 1) < 1e-12);
    }
    for (size_t row = 0; row < 2; ++row) {
        for (size_t column = 0; column < 2; ++column) {
            double value = svd.u[row][0] * svd.sigma[0] * svd.v[column][0] +
                           svd.u[row][1] * svd.sigma[1] * svd.v[column][1];
            assert(fabs(value - image[row][column]) < 1e-12);
        }
    }
    const double singular[2][2] = {{1, 2}, {2, 4}};
    SvdResult saved = {.reconstruction_error = 99};
    assert(!svd2(singular, &saved));
    assert(saved.reconstruction_error == 99);

    const double h = 1e-6;
    const double difficult[3][2] = {{1, 1}, {1, 1 + h}, {1, 1 - h}};
    const double targets[3] = {2, 2 + h, 2 - h};
    double normal[2] = {0};
    assert(qr_solve(difficult, targets, 3, &qr));
    assert(normal_solve(difficult, targets, 3, normal));
    double qr_error = fmax(fabs(qr.x[0] - 1), fabs(qr.x[1] - 1));
    double normal_error = fmax(fabs(normal[0] - 1), fabs(normal[1] - 1));
    assert(qr_error < 1e-8);
    assert(normal_error > qr_error);
}

/**
 * @brief 输出固定实验字段，供 C/Python 对照。
 * @param csv 为 true 时输出 field,value。
 * @return 成功为 EXIT_SUCCESS，失败为 EXIT_FAILURE。
 */
static int print_demo(bool csv)
{
    const double a[3][2] = {{1, 0}, {1, 1}, {1, 2}};
    const double b[3] = {1, 2, 2};
    const double image[2][2] = {{2, 1}, {1, 2}};
    const double h = 1e-6;
    const double difficult[3][2] = {{1, 1}, {1, 1 + h}, {1, 1 - h}};
    const double targets[3] = {2, 2 + h, 2 - h};
    QrResult qr = {0};
    SvdResult svd = {0};
    double normal[2] = {0};
    if (!qr_solve(a, b, 3, &qr) || !svd2(image, &svd)) {
        return EXIT_FAILURE;
    }
    double x0 = qr.x[0];
    double x1 = qr.x[1];
    double residual = qr.residual;
    if (!qr_solve(difficult, targets, 3, &qr) ||
        !normal_solve(difficult, targets, 3, normal)) {
        return EXIT_FAILURE;
    }
    double qr_error = fmax(fabs(qr.x[0] - 1), fabs(qr.x[1] - 1));
    double normal_error = fmax(fabs(normal[0] - 1), fabs(normal[1] - 1));
    if (csv) {
        puts("field,value");
        printf("x0,%.17g\nx1,%.17g\nresidual,%.17g\n", x0, x1, residual);
        printf("sigma0,%.17g\nsigma1,%.17g\nreconstruction,%.17g\n",
               svd.sigma[0], svd.sigma[1], svd.reconstruction_error);
        printf("qr_error,%.17g\nnormal_error,%.17g\n", qr_error, normal_error);
    } else {
        printf("QR 最小二乘：x=(%.8f, %.8f)，残差范数=%.8f\n", x0, x1, residual);
        printf("二维 SVD：奇异值=(%.8f, %.8f)，重建误差=%.3g\n",
               svd.sigma[0], svd.sigma[1], svd.reconstruction_error);
        printf("近共线输入：QR 参数误差=%.6g，正规方程参数误差=%.6g\n",
               qr_error, normal_error);
    }
    return EXIT_SUCCESS;
}

/**
 * @brief 运行 M07 固定实验。
 * @param argc 参数个数，可使用 --test 或 --csv。
 * @param argv 参数数组。
 * @return 成功为 EXIT_SUCCESS，否则为 EXIT_FAILURE。
 */
int main(int argc, char *argv[])
{
    if (argc == 2 && strcmp(argv[1], "--test") == 0) {
        test_decomposition();
        puts("投影与矩阵分解测试通过");
        return EXIT_SUCCESS;
    }
    if (argc == 1 || (argc == 2 && strcmp(argv[1], "--csv") == 0)) {
        return print_demo(argc == 2);
    }
    fputs("用法：36_projection_decomposition [--test | --csv]\n", stderr);
    return EXIT_FAILURE;
}
