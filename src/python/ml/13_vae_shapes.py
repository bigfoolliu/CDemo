"""用一个二维潜变量样本演示 VAE 的重参数化与 KL 项。"""

from math import exp, log


def main() -> None:
    """计算二维高斯潜变量样本、重建误差和 KL 散度。

    :return: 无。
    """
    mean = (0.4, -0.2)
    log_variance = (log(0.25), log(1.0))
    noise = (1.0, -0.5)
    latent = tuple(mu + exp(0.5 * lv) * epsilon
                   for mu, lv, epsilon in zip(mean, log_variance, noise))
    target = (1.0, 0.0, 1.0, 0.0)
    reconstruction = (0.8, 0.3, 0.7, 0.2)
    bce = -sum(y * log(p) + (1 - y) * log(1 - p)
               for y, p in zip(target, reconstruction))
    kl = 0.5 * sum(mu * mu + exp(lv) - 1.0 - lv
                   for mu, lv in zip(mean, log_variance))
    print("重参数化 z = μ + σ·ε：z={}".format(tuple(round(v, 3) for v in latent)))
    print("重建 BCE={:.4f}  KL={:.4f}  ELBO 损失={:.4f}".format(bce, kl, bce + kl))
    print("这里展示单次前向计算；C 案例还训练网络并导出潜空间图像。")


if __name__ == "__main__":
    main()
