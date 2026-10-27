"""展示本项目 Python 示例的函数、类型注解、列表与异常处理。"""

from typing import List


def calculate_mean(values: List[float]) -> float:
    """计算非空数值列表的平均值。

    :param values: 至少包含一个数值的列表。
    :return: 列表中所有数值的算术平均值。
    :raises ValueError: values 为空时抛出。
    """
    if not values:
        raise ValueError("values 不能为空")

    return sum(values) / len(values)


def main() -> None:
    """运行小型示例并展示错误输入的处理方式。

    :return: 无。
    """
    scores = [78.0, 86.0, 92.0]
    mean_score = calculate_mean(scores)
    print("样本数：{}".format(len(scores)))
    print("平均分：{:.1f}".format(mean_score))

    try:
        calculate_mean([])
    except ValueError as error:
        print("空列表处理：{}".format(error))


if __name__ == "__main__":
    main()
