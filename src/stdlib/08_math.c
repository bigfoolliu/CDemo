/** @file
 * @brief 学习常用数学函数、误差容限和非有限值检查。
 *
 * 学习范围：<math.h> 实数数学库；接口数量较多，下面是分类与代表性接口。
 * 接口概览（不是完整函数清单）：
 * - 三角与双曲：sin/cos/tan、asin/acos/atan/atan2、sinh/cosh/tanh、asinh/acosh/atanh。
 * - 指数与对数：exp/exp2/expm1、log/log2/log10/log1p；幂与距离：pow/sqrt/cbrt/hypot。
 * - 取整：ceil/floor/trunc/round、lround/llround、rint/nearbyint、lrint/llrint。
 * - 余数与组合：fmod/remainder/remquo、fma、fabs、fmin/fmax、fdim、copysign。
 * - 浮点分解与相邻值：frexp/ldexp、modf、scalbn/scalbln、ilogb/logb、nextafter/nexttoward。
 * - 特殊函数：erf/erfc、tgamma/lgamma、nan；多数函数有 f 和 l 后缀版本。
 * - 分类/比较宏：fpclassify、isfinite、isinf、isnan、isnormal、signbit、isgreater 等。
 * - 常量与错误机制：INFINITY、NAN、HUGE_VAL 等、math_errhandling、MATH_ERRNO/MATH_ERREXCEPT。
 * C23 还扩充了数学接口，新增及扩展浮点支持需单独查阅，不由本例保证。
 * 本文件已演示：sqrt、hypot、exp、log、pow、fabs、floor、ceil、round、
 * isfinite、isnan，以及 INFINITY/NAN；封装函数在调用 sqrt 前检查定义域。
 * 本文件未演示：其他数学分类、float/long double 版本、errno/浮点异常错误处理。
 * 观察重点：定义域、有限值和误差容限；链接数学库通常需要 -lm。
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>

/**
 * @brief 在调用 sqrt 前检查定义域和输入是否有限。
 * @param input 待开平方的数。
 * @param output 有效的结果输出地址，仅成功时修改。
 * @return 有限且非负的输入返回 true；否则返回 false。
 */
static bool finite_sqrt(double input, double *output)
{
    if (!isfinite(input) || input < 0) {
        return false;
    }
    *output = sqrt(input);
    return true;
}

/** @brief 验证数学函数与定义域边界。 @return 验证完成返回 0。 */
int main(void)
{
    double root = 0;
    bool ok = finite_sqrt(9, &root);
    assert(ok && fabs(root - 3) < 1e-12);
    ok = finite_sqrt(-1, &root);
    assert(!ok && root == 3);
    ok = finite_sqrt(INFINITY, &root);
    assert(!ok && root == 3);
    ok = finite_sqrt(0, &root);
    assert(ok && root == 0);
    assert(fabs(hypot(3, 4) - 5) < 1e-12);
    assert(fabs(log(exp(1)) - 1) < 1e-12);
    assert(fabs(pow(2, 3) - 8) < 1e-12);
    assert(floor(-1.2) == -2 && ceil(-1.2) == -1 && round(-1.5) == -2);
    assert(isnan(NAN) && !isfinite(INFINITY));
    puts("math：平方根、指数/对数、取整、误差容限与定义域检查验证通过");
    return 0;
}
