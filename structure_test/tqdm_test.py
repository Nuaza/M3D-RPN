from tqdm import tqdm
from time import sleep


def test(name):
    print(name)


if __name__ == '__main__':
    for item in tqdm([i for i in range(0, 1000)], desc="Testing"):
        sleep(1)

