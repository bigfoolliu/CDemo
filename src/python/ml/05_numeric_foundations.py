"""演示矩阵基础运算、线性模型梯度与有限差分检查。"""

from typing import List

Matrix = List[List[float]]


def multiply(left: Matrix, right: Matrix) -> Matrix:
    """计算两个兼容矩阵的乘积。

    :param left: 形状为 m×n 的左矩阵。
    :param right: 形状为 n×p 的右矩阵。
    :return: 形状为 m×p 的结果矩阵。
    :raises ValueError: 两矩阵内维数不相等时抛出。
    """
    if not left or not right or len(left[0]) != len(right):
        raise ValueError("矩阵形状不兼容")
    return [[sum(a * right[k][column] for k, a in enumerate(row))
             for column in range(len(right[0]))] for row in left]


def main() -> None:
    """打印矩阵乘法和线性回归单步梯度。

    :return: 无。
    """
    left = [[1.0, 2.0], [3.0, 4.0]]
    right = [[2.0, 0.0], [1.0, 2.0]]
    print("矩阵乘积：{}".format(multiply(left, right)))

    x = [1.0, 2.0]
    y = [3.0, 5.0]
    weight = [0.5, -0.25]
    bias = 0.1
    errors = [sum(w * value for w, value in zip(weight, row)) + bias - target
              for row, target in zip(([1.0, 0.0], [0.0, 1.0]), y)]
    analytic = [sum(2.0 * error * row[index] for error, row in zip(errors, ([1.0, 0.0], [0.0, 1.0]))) / 2
                for index in range(2)]
    print("线性模型误差={}，权重梯度={}，示例输入 x={}".format(errors, analytic, x))

    def objective(first_weight: float) -> float:
        """计算固定微型数据上的均方误差。

        :param first_weight: 第一个权重候选值。
        :return: 均方误差。
        """
        predictions = [first_weight * 1.0 + weight[1] * 0.0 + bias,
                       first_weight * 0.0 + weight[1] * 1.0 + bias]
        return sum((prediction - target) ** 2 for prediction, target in zip(predictions, y)) / 2

    epsilon = 1e-5
    numeric = (objective(weight[0] + epsilon) - objective(weight[0] - epsilon)) / (2 * epsilon)
    print("梯度检查：解析={:.6f}，有限差分={:.6f}".format(analytic[0], numeric))


if __name__ == "__main__":
    main()
