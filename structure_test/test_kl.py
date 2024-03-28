import numpy
import numpy as np


def kl_divergence(p, q):
    """
    计算两个概率分布之间的KL散度
    :param p: 第一个概率分布
    :param q: 第二个概率分布
    :return: KL散度值
    """
    p, q = np.array(p, dtype=np.float_), np.array(q, dtype=np.float_)

    # 确保概率分布的元素之和为1
    p /= np.sum(p)
    q /= np.sum(q)

    # 避免出现log(0)的情况，使用numpy的clip函数将概率限制在一个小的正数范围内
    p, q = np.clip(p, 1e-10, 1.0), np.clip(q, 1e-10, 1.0)

    # 计算KL散度
    return np.sum(p * np.log(p / q))


if __name__ == '__main__':
    p = [0.2, 0.4, 0.6, 0.9]
    q = [0.9, 0.6, 0.4, 0.2]
    print(kl_divergence(p, q))
