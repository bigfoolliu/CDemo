"""以词袋计数和拉普拉斯平滑实现多项式朴素贝叶斯。"""

from collections import Counter
from math import log


def main() -> None:
    """训练垃圾短信分类器并给新句子计算类别分数。

    :return: 无。
    """
    texts = ["cheap offer now", "win a prize", "cheap prize today",
             "meeting today", "project meeting now", "review the project"]
    labels = [1, 1, 1, 0, 0, 0]
    vocabulary = sorted({word for text in texts for word in text.split()})
    class_docs = Counter(labels)
    word_counts = {label: Counter() for label in (0, 1)}
    totals = Counter()
    for text, label in zip(texts, labels):
        word_counts[label].update(text.split())
        totals[label] += len(text.split())
    query = "cheap project offer".split()
    scores = {}
    for label in (0, 1):
        score = log(class_docs[label] / len(labels))
        score += sum(log((word_counts[label][word] + 1) /
                         (totals[label] + len(vocabulary))) for word in query)
        scores[label] = score
    print("词表大小：{}".format(len(vocabulary)))
    print("待分类文本：{}".format(" ".join(query)))
    print("对数后验分数：{} -> {}".format(scores, "垃圾" if scores[1] > scores[0] else "正常"))


if __name__ == "__main__":
    main()
