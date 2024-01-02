from tqdm import tqdm
from time import sleep

if __name__ == '__main__':
    # for item in tqdm([i for i in range(0, 100)], desc="正在保存设置"):
    #     sleep(0.05)
    bar = tqdm(total=250)
    for i in range(0, 250):
        bar.update(1)
    bar = tqdm(total=150)
    for i in range(0, 150):
        bar.update(1)

