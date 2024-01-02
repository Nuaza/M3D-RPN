from tqdm import tqdm
import time

if __name__ == '__main__':
    for item in tqdm([i for i in range(0, 5)], desc="正在保存设置"):
        time.sleep(1)

