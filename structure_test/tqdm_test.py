from tqdm import tqdm
from time import sleep


def test(name):
    print(name)


if __name__ == '__main__':
    max_iter = 50000
    display = 250
    j = 1
    for item in tqdm([i for i in range(0, 100)], desc="Testing " + str(j) + "/" + str(int(max_iter/display))):
        j += 1
        sleep(0.01)

