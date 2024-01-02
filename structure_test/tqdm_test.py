from tqdm import tqdm
import time

if __name__ == '__main__':
    list = [i for i in range(0, 10)]
    for char in tqdm(list, desc="测试用例"):
        time.sleep(0.1)

