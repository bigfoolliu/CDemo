"""用固定小数据演示硬币 MLE/Beta 后验和一维高斯方差。"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CoinEstimate:
    """保存硬币参数的三个点估计及后验形状。"""

    mle: float
    posterior_mean: float
    map: float
    posterior_alpha: float
    posterior_beta: float


@dataclass(frozen=True)
class GaussianEstimate:
    """保存高斯均值、MLE 方差及可选无偏样本方差。"""

    mean: float
    mle_variance: float
    unbiased_variance: float | None


def estimate_coin(heads: int, tails: int, alpha: float, beta: float) -> CoinEstimate:
    """从硬币计数求 MLE、Beta 后验均值与内点 MAP。

    :param heads: 正面次数，非负。
    :param tails: 反面次数，非负；总次数必须为正。
    :param alpha: Beta 先验第一个形状参数，有限且大于 1。
    :param beta: Beta 先验第二个形状参数，有限且大于 1。
    :return: 三个估计及更新后的 Beta 形状参数。
    :raises ValueError: 计数或先验无效时抛出。
    """
    if (not isinstance(heads, int) or isinstance(heads, bool) or heads < 0 or
            not isinstance(tails, int) or isinstance(tails, bool) or tails < 0 or
            heads + tails == 0 or not math.isfinite(alpha) or not math.isfinite(beta) or
            alpha <= 1 or beta <= 1):
        raise ValueError("计数必须非负且非空，先验形状必须有限且大于 1")
    count = heads + tails
    posterior_alpha = alpha + heads
    posterior_beta = beta + tails
    posterior_sum = posterior_alpha + posterior_beta
    mle = heads / count
    posterior_mean = posterior_alpha / posterior_sum
    map_value = (posterior_alpha - 1) / (posterior_sum - 2)
    if not all(map(math.isfinite, (mle, posterior_mean, map_value, posterior_sum))):
        raise ValueError("估计结果非有限")
    return CoinEstimate(mle, posterior_mean, map_value, posterior_alpha, posterior_beta)


def coin_mean_nll(heads: int, tails: int, probability: float) -> float:
    """计算有序独立硬币观测的平均负对数似然。

    :param heads: 正面次数，非负。
    :param tails: 反面次数，非负；总次数必须为正。
    :param probability: 正面概率，范围为闭区间 0 到 1。
    :return: 每次观测的平均负对数似然。
    :raises ValueError: 非法输入或模型把已出现事件赋予零概率时抛出。
    """
    if (not isinstance(heads, int) or isinstance(heads, bool) or heads < 0 or
            not isinstance(tails, int) or isinstance(tails, bool) or tails < 0 or
            heads + tails == 0 or not math.isfinite(probability) or
            not 0 <= probability <= 1 or
            (heads > 0 and probability == 0) or
            (tails > 0 and probability == 1)):
        raise ValueError("无效概率或观测")
    total = 0.0
    if heads:
        total -= heads * math.log(probability)
    if tails:
        total -= tails * math.log1p(-probability)
    result = total / (heads + tails)
    if not math.isfinite(result):
        raise ValueError("评分结果非有限")
    return result


def estimate_gaussian(values: tuple[float, ...]) -> GaussianEstimate:
    """两遍扫描求一维高斯 MLE 和无偏样本方差。

    :param values: 至少一个有限实数；不修改输入。
    :return: 均值、MLE 方差，以及样本数大于 1 时的无偏方差。
    :raises ValueError: 空输入、非有限值或溢出时抛出。
    """
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("需要至少一个有限值")
    count = len(values)
    mean = sum(values) / count
    squared = sum((value - mean) ** 2 for value in values)
    mle_variance = squared / count
    unbiased = squared / (count - 1) if count > 1 else None
    if not math.isfinite(mean) or not math.isfinite(mle_variance) or (
            unbiased is not None and not math.isfinite(unbiased)):
        raise ValueError("统计量非有限")
    return GaussianEstimate(mean, mle_variance, unbiased)


def gaussian_density(estimate: GaussianEstimate, x: float) -> float:
    """在非退化高斯模型下计算一点的概率密度。

    :param estimate: 有有限均值和严格正方差的估计结果。
    :param x: 有限评估点。
    :return: 非负且有限的概率密度。
    :raises ValueError: 输入无效或方差为零时抛出。
    """
    variance = estimate.mle_variance
    if (not math.isfinite(x) or not math.isfinite(estimate.mean) or
            not math.isfinite(variance) or variance <= 0):
        raise ValueError("概率密度需要有限输入及严格正的方差")
    delta = x - estimate.mean
    result = math.exp(-(delta * delta) / (2 * variance)) / math.sqrt(2 * math.pi * variance)
    if not math.isfinite(result):
        raise ValueError("概率密度非有限")
    return result


def results() -> dict[str, float]:
    """运行固定训练数据与独立测试数据。

    :return: 与 C 版本同名的数值字段。
    """
    coin = estimate_coin(7, 3, 2, 2)
    gaussian = estimate_gaussian((1.0, 2.0, 3.0, 4.0))
    assert gaussian.unbiased_variance is not None
    return {
        "coin_mle": coin.mle,
        "coin_posterior_mean": coin.posterior_mean,
        "coin_map": coin.map,
        "coin_test_nll_mle": coin_mean_nll(4, 2, coin.mle),
        "coin_test_nll_mean": coin_mean_nll(4, 2, coin.posterior_mean),
        "coin_test_nll_map": coin_mean_nll(4, 2, coin.map),
        "coin_test_nll_baseline": coin_mean_nll(4, 2, 0.5),
        "gaussian_mean": gaussian.mean,
        "gaussian_mle_variance": gaussian.mle_variance,
        "gaussian_unbiased_variance": gaussian.unbiased_variance,
        "gaussian_density_at_mean": gaussian_density(gaussian, gaussian.mean),
    }


def main() -> None:
    """打印教学摘要或供 C/Python 对照的 CSV。

    :return: 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", action="store_true", help="输出 field,value")
    args = parser.parse_args()
    values = results()
    if args.csv:
        print("field,value")
        for field, value in values.items():
            print(f"{field},{value:.17g}")
        return
    print("硬币：MLE {coin_mle:.6f}，后验均值 {coin_posterior_mean:.6f}，"
          "MAP {coin_map:.6f}".format(**values))
    print("独立测试平均负对数似然：MLE {coin_test_nll_mle:.6f}，"
          "后验均值 {coin_test_nll_mean:.6f}，MAP {coin_test_nll_map:.6f}，"
          "公平硬币基线 {coin_test_nll_baseline:.6f}".format(**values))
    print("高斯：均值 {gaussian_mean:.6f}，MLE 方差 {gaussian_mle_variance:.6f}，"
          "无偏样本方差 {gaussian_unbiased_variance:.6f}，"
          "均值处密度 {gaussian_density_at_mean:.6f}".format(**values))


if __name__ == "__main__":
    main()
