"""用 bootstrap 抽样训练多棵随机决策树桩并集成投票。"""

from random import Random


def main() -> None:
    """固定随机种子构建浅层树桩森林并报告投票结果。

    :return: 无。
    """
    random = Random(17)
    points = [(x, y) for x, y in ((0, 0), (0, 1), (1, 0), (1, 1),
                                  (2, 1), (3, 2), (3, 3), (4, 3))]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    query = (3.5, 2.5)
    votes = []
    tree_count = 9
    for _ in range(tree_count):
        bootstrap = [random.randrange(len(points)) for _ in points]
        feature = random.randrange(2)
        candidates = sorted({points[index][feature] for index in bootstrap})
        threshold = candidates[len(candidates) // 2]
        low = [labels[index] for index in bootstrap if points[index][feature] <= threshold]
        high = [labels[index] for index in bootstrap if points[index][feature] > threshold]
        low_vote = int(sum(low) >= len(low) / 2) if low else 0
        high_vote = int(sum(high) >= len(high) / 2) if high else low_vote
        prediction = high_vote if query[feature] > threshold else low_vote
        votes.append(int(prediction))
    print("树数={}，类别 1 投票={} / {}".format(tree_count, sum(votes), tree_count))
    print("森林预测：{}；bootstrap 样本和随机特征让树之间产生差异。".format(
        int(sum(votes) > tree_count / 2)))


if __name__ == "__main__":
    main()
