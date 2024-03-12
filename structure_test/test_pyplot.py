import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib

if __name__ == '__main__':
    x = np.arange(-10, 10, 0.1)
    y = []
    for i in x:
        y_1 = 1 / (1 + math.exp(-i))
        y.append(y_1)
    plt.plot(x, y, label="sigmoid")
    plt.ylim(0, 1)
    plt.legend()

    # 先save再show，不然会save个空白
    plt.savefig("test.png")
    plt.show()
