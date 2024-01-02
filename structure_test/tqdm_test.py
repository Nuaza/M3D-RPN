from tqdm import tqdm
from time import sleep

if __name__ == '__main__':
    for item in tqdm([i for i in range(0, 1000)], desc="正在保存设置"):
        sleep(0.005)

