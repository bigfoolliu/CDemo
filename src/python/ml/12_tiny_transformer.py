"""展示 Transformer 的缩放点积因果自注意力权重。"""

from math import exp, sqrt


def softmax(values: list[float]) -> list[float]:
    """将一组分数转换为概率。

    :param values: 任意有限实数分数。
    :return: 总和为 1 的概率列表。
    """
    maximum = max(values)
    exponents = [exp(value - maximum) for value in values]
    total = sum(exponents)
    return [value / total for value in exponents]


def main() -> None:
    """计算三个 token 的因果注意力，并打印每行的关注分布。

    :return: 无。
    """
    tokens = ("我", "喜欢", "学习")
    vectors = ((1.0, 0.0), (0.8, 0.6), (0.0, 1.0))
    scale = sqrt(len(vectors[0]))
    for query_index, query in enumerate(vectors):
        scores = [sum(a * b for a, b in zip(query, key)) / scale
                  for key in vectors[:query_index + 1]]
        weights = softmax(scores)
        visible = ["{}:{:.2f}".format(tokens[i], weight)
                   for i, weight in enumerate(weights)]
        print("{} 关注 {}".format(tokens[query_index], " ".join(visible)))
    print("因果掩码禁止读取未来 token；真实 Transformer 还包含多头、残差和前馈层。")


if __name__ == "__main__":
    main()
