"""在一维高斯数据上手写 GAN 的判别器与生成器交替梯度更新。"""

from math import exp, sqrt
from random import Random


def main() -> None:
    """训练线性生成器和逻辑判别器拟合一维真实高斯分布。

    :return: 无。
    """
    random = Random(14)
    generator_weight = 0.4
    generator_bias = -0.5
    discriminator_weight = 0.1
    discriminator_bias = 0.0
    rate = 0.03
    batch_size = 32
    for _ in range(2500):
        latent = [random.gauss(0.0, 1.0) for _ in range(batch_size)]
        real = [random.gauss(1.5, 0.4) for _ in range(batch_size)]
        fake = [generator_weight * value + generator_bias for value in latent]

        grad_weight = 0.0
        grad_bias = 0.0
        for value in real:
            probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0,
                discriminator_weight * value + discriminator_bias))))
            grad_weight += (probability - 1.0) * value / (2 * batch_size)
            grad_bias += (probability - 1.0) / (2 * batch_size)
        for value in fake:
            probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0,
                discriminator_weight * value + discriminator_bias))))
            grad_weight += probability * value / (2 * batch_size)
            grad_bias += probability / (2 * batch_size)
        discriminator_weight -= rate * grad_weight
        discriminator_bias -= rate * grad_bias

        grad_generator_weight = 0.0
        grad_generator_bias = 0.0
        for noise, value in zip(latent, fake):
            probability = 1.0 / (1.0 + exp(-max(-30.0, min(30.0,
                discriminator_weight * value + discriminator_bias))))
            derivative = (probability - 1.0) * discriminator_weight
            grad_generator_weight += derivative * noise / batch_size
            grad_generator_bias += derivative / batch_size
        generator_weight -= rate * grad_generator_weight
        generator_bias -= rate * grad_generator_bias

    generated = [generator_weight * random.gauss(0.0, 1.0) + generator_bias
                 for _ in range(500)]
    mean = sum(generated) / len(generated)
    deviation = sqrt(sum((value - mean) ** 2 for value in generated) / len(generated))
    print("真实分布：均值=1.50 标准差=0.40")
    print("生成分布：均值={:.3f} 标准差={:.3f}".format(mean, deviation))
    print("G(x)=a·z+b；D(x)=sigmoid(w·x+c)。D 与 G 交替更新，G 使用非饱和损失。")
    print("这是低维缩小版；C 案例使用二维网络并展示模式覆盖和决策边界。")


if __name__ == "__main__":
    main()
