"""用 Python 标准库演示二维高斯混合模型与期望最大化。"""

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import List, Tuple

Point = Tuple[float, float]
Covariance = Tuple[float, float, float]

SAMPLE_COUNT = 6
COMPONENT_COUNT = 2
MAX_ITERATIONS = 20
COVARIANCE_REGULARIZATION = 1e-6
EFFECTIVE_COUNT_MINIMUM = 1e-12
CONVERGENCE_TOLERANCE = 1e-10

POINTS: List[Point] = [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 0.0),
    (8.0, 8.0),
    (8.0, 9.0),
    (9.0, 8.0),
]


@dataclass
class GaussianMixture:
    """保存两个高斯成分的权重、均值和协方差。"""

    weights: List[float]
    means: List[Point]
    covariances: List[Covariance]


def covariance_is_valid(covariance: Covariance) -> bool:
    """检查二维协方差矩阵是否有限且正定。

    :param covariance: 按 (xx, xy, yy) 排列的对称协方差矩阵。
    :return: 对角元素和行列式满足正定条件时返回 True。
    """
    xx, xy, yy = covariance
    determinant = xx * yy - xy * xy
    return (
        isfinite(xx)
        and isfinite(xy)
        and isfinite(yy)
        and isfinite(determinant)
        and xx > 0.0
        and determinant > 0.0
    )


def model_is_valid(model: GaussianMixture) -> bool:
    """检查 GMM 参数是否满足基本概率约束。

    :param model: 待检查的混合模型。
    :return: 权重为正且和为 1、均值有限、协方差正定时返回 True。
    """
    if len(model.weights) != COMPONENT_COUNT:
        return False
    if len(model.means) != COMPONENT_COUNT:
        return False
    if len(model.covariances) != COMPONENT_COUNT:
        return False

    if any(not isfinite(weight) or weight <= 0.0 for weight in model.weights):
        return False
    if abs(sum(model.weights) - 1.0) > 1e-9:
        return False

    for mean, covariance in zip(model.means, model.covariances):
        if not all(isfinite(value) for value in mean):
            return False
        if not covariance_is_valid(covariance):
            return False

    return True


def log_gaussian_2d(point: Point, mean: Point, covariance: Covariance) -> float:
    """计算二维高斯分布的对数概率密度。

    :param point: 当前二维观测点。
    :param mean: 高斯成分的均值向量。
    :param covariance: 按 (xx, xy, yy) 排列的正定协方差矩阵。
    :return: 当前点的对数概率密度。
    :raises ValueError: 输入坐标、均值或协方差无效时抛出。
    """
    if not all(isfinite(value) for value in point + mean):
        raise ValueError("样本和均值必须是有限数")
    if not covariance_is_valid(covariance):
        raise ValueError("协方差矩阵必须有限且正定")

    xx, xy, yy = covariance
    determinant = xx * yy - xy * xy
    dx = point[0] - mean[0]
    dy = point[1] - mean[1]
    # 计算残差转置、协方差逆矩阵和残差组成的马氏距离平方。
    quadratic = (yy * dx * dx - 2.0 * xy * dx * dy + xx * dy * dy) / determinant
    # 对数密度避免先计算很小的高斯密度而下溢。
    result = -log(2.0 * 3.141592653589793) - 0.5 * log(determinant) - 0.5 * quadratic

    if not isfinite(result):
        raise ValueError("对数概率密度不是有限数")

    return result


def expectation_step(
    model: GaussianMixture, points: List[Point]
) -> Tuple[List[List[float]], float]:
    """执行 E 步并计算当前模型的观测对数似然。

    :param model: 当前有效 GMM 参数。
    :param points: 固定的二维观测点列表。
    :return: 每个样本的责任度矩阵及全部样本的对数似然。
    :raises ValueError: 模型无效或数值计算失败时抛出。
    """
    if not model_is_valid(model):
        raise ValueError("GMM 参数无效")
    if not points:
        raise ValueError("样本列表不能为空")

    responsibilities: List[List[float]] = []
    total_log_likelihood = 0.0

    for point in points:
        log_scores = []
        for component in range(COMPONENT_COUNT):
            # 对数分数 = 对数混合权重 + 当前成分的对数密度。
            score = log(model.weights[component]) + log_gaussian_2d(
                point,
                model.means[component],
                model.covariances[component],
            )
            log_scores.append(score)

        maximum_score = max(log_scores)
        # 减去最大分数，稳定地计算各成分分数之和。
        log_normalizer = maximum_score + log(
            sum(exp(score - maximum_score) for score in log_scores)
        )
        total_log_likelihood += log_normalizer
        # 归一化后的概率就是样本对各成分的责任度。
        responsibilities.append(
            [exp(score - log_normalizer) for score in log_scores]
        )

    if not isfinite(total_log_likelihood):
        raise ValueError("观测对数似然不是有限数")

    return responsibilities, total_log_likelihood


def maximization_step(
    points: List[Point], responsibilities: List[List[float]]
) -> GaussianMixture:
    """执行 M 步，返回按责任度更新的新模型。

    :param points: 参与本轮训练的二维观测点。
    :param responsibilities: 上一轮 E 步计算出的责任度矩阵。
    :return: 更新后的高斯混合模型。
    :raises ValueError: 责任度形状无效、成分为空或协方差退化时抛出。
    """
    if len(points) != len(responsibilities) or not points:
        raise ValueError("样本数必须与责任度行数相等且非零")
    if any(len(row) != COMPONENT_COUNT for row in responsibilities):
        raise ValueError("每个样本都必须对应两个成分责任度")

    # 每个样本按责任度向各成分贡献小数份额。
    effective_counts = [0.0] * COMPONENT_COUNT
    for row in responsibilities:
        if any(not isfinite(value) or value < 0.0 or value > 1.0 for value in row):
            raise ValueError("责任度必须是 [0, 1] 内的有限数")
        if abs(sum(row) - 1.0) > 1e-9:
            raise ValueError("每个样本的责任度之和必须为 1")
        for component in range(COMPONENT_COUNT):
            effective_counts[component] += row[component]

    if any(count <= EFFECTIVE_COUNT_MINIMUM for count in effective_counts):
        raise ValueError("存在有效样本数接近零的空成分")

    weights = [count / len(points) for count in effective_counts]
    means: List[Point] = []
    for component in range(COMPONENT_COUNT):
        # 分别计算两个坐标的责任度加权平均。
        mean_x = sum(
            responsibilities[sample][component] * points[sample][0]
            for sample in range(len(points))
        ) / effective_counts[component]
        mean_y = sum(
            responsibilities[sample][component] * points[sample][1]
            for sample in range(len(points))
        ) / effective_counts[component]
        means.append((mean_x, mean_y))

    covariances: List[Covariance] = []
    for component in range(COMPONENT_COUNT):
        variance_x = 0.0
        covariance_xy = 0.0
        variance_y = 0.0
        for sample, point in enumerate(points):
            responsibility = responsibilities[sample][component]
            dx = point[0] - means[component][0]
            dy = point[1] - means[component][1]
            variance_x += responsibility * dx * dx
            covariance_xy += responsibility * dx * dy
            variance_y += responsibility * dy * dy

        # 累计加权外积，并给两个方差对角线加入微小正则量。
        covariance = (
            variance_x / effective_counts[component] + COVARIANCE_REGULARIZATION,
            covariance_xy / effective_counts[component],
            variance_y / effective_counts[component] + COVARIANCE_REGULARIZATION,
        )
        if not covariance_is_valid(covariance):
            raise ValueError("更新后的协方差矩阵不是正定矩阵")
        covariances.append(covariance)

    updated_model = GaussianMixture(weights, means, covariances)
    if not model_is_valid(updated_model):
        raise ValueError("M 步产生了无效模型")

    return updated_model


def print_model(
    iteration: int,
    model: GaussianMixture,
    log_likelihood: float,
    change: float,
) -> None:
    """输出一轮 EM 的似然和全部成分参数。

    :param iteration: 当前迭代编号，初始模型使用 0。
    :param model: 当前 GMM 参数。
    :param log_likelihood: 当前观测对数似然。
    :param change: 与上一轮相比的对数似然变化。
    :return: 无。
    """
    print(f"迭代 {iteration}：对数似然={log_likelihood: .9f}，变化={change:+.3e}")
    for component in range(COMPONENT_COUNT):
        xx, xy, yy = model.covariances[component]
        mean_x, mean_y = model.means[component]
        print(
            f"  成分 {component}：权重={model.weights[component]:.6f}，"
            f"均值=({mean_x:.6f}, {mean_y:.6f})，"
            f"协方差=[[{xx:.6f}, {xy:.6f}], [{xy:.6f}, {yy:.6f}]]"
        )


def print_responsibilities(responsibilities: List[List[float]]) -> None:
    """输出每个样本属于两个成分的后验概率。

    :param responsibilities: E 步生成的责任度矩阵。
    :return: 无。
    """
    for sample, point in enumerate(POINTS):
        first, second = responsibilities[sample]
        print(
            f"  样本 {sample} ({point[0]:g}, {point[1]:g})："
            f"责任度=({first:.9f}, {second:.9f})"
        )


def main() -> None:
    """使用固定二维数据运行 EM 并逐轮显示参数。

    :return: 无。
    """
    model = GaussianMixture(
        weights=[0.5, 0.5],
        means=[(0.0, 0.0), (8.0, 8.0)],
        covariances=[(1.0, 0.0, 1.0), (1.0, 0.0, 1.0)],
    )
    responsibilities, log_likelihood = expectation_step(model, POINTS)
    print_model(0, model, log_likelihood, 0.0)
    print_responsibilities(responsibilities)
    converged = False

    # 每轮先用当前责任度更新参数，再计算新模型的似然和下一轮责任度。
    for iteration in range(1, MAX_ITERATIONS + 1):
        previous_log_likelihood = log_likelihood
        model = maximization_step(POINTS, responsibilities)
        responsibilities, log_likelihood = expectation_step(model, POINTS)
        change = log_likelihood - previous_log_likelihood
        print_model(iteration, model, log_likelihood, change)
        print_responsibilities(responsibilities)

        if abs(change) < CONVERGENCE_TOLERANCE:
            converged = True
            break

    assert model_is_valid(model)
    status = "达到似然变化阈值" if converged else "达到迭代上限"
    print(f"训练结束：{status}。")


if __name__ == "__main__":
    main()
