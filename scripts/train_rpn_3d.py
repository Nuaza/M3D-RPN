# -----------------------------------------
# python模块
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os

# 禁用pyc缓存
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# 自定义模块
# -----------------------------------------
from time import sleep
from tqdm import tqdm
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *


# tqdm做的伪进度条，显示的内容不会被写入log中
def load_bar(desc, max=100, sleep_time=0.05):
    for item in tqdm([i for i in range(0, max)], desc=desc):
        sleep(sleep_time)


def main(argv):

    # -----------------------------------------
    # 传入参数
    # -----------------------------------------
    opts, args = getopt(argv, '', ['config=', 'restore='])

    # 默认
    conf_name = None
    restore = None

    # 读取参数
    for opt, arg in opts:

        if opt in ('--config'): conf_name = arg
        if opt in ('--restore'): restore = int(arg)

    # 必须要传入的参数 --config
    if conf_name is None:
        raise ValueError('请提供参数文件名，例如： --config=<参数名>')

    # -----------------------------------------
    # 基础设置
    # -----------------------------------------

    # 获取到配置文件的Config()函数，其返回一个包含训练所需的各项配置变量的字典
    conf = init_config(conf_name)
    # 初始化训练的路径
    paths = init_training_paths(conf_name)

    # 初始化torch，传随机种子
    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    # 不用visdom
    # vis = init_visdom(conf_name, conf.visdom_port)

    start_iter = 0
    # 追踪目标统计信息的tracker
    tracker = edict()
    iterator = None
    # has_visdom = vis is not None
    has_visdom = None

    # 加载数据集
    dataset = Dataset(conf, paths.data, paths.output)
    sleep(3)
    # 生成锚
    generate_anchors(conf, dataset.imdb, paths.output)
    sleep(3)
    # 计算边界框(bbox)的回归参数(均值和标准差)
    compute_bbox_stats(conf, dataset.imdb, paths.output)
    sleep(3)

    # -----------------------------------------
    # 存储设置
    # -----------------------------------------

    # 保存设置
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)
    load_bar("正在保存设置")
    logging.info('设置保存完成')
    sleep(3)

    # 显示设置
    logging.info('训练配置一览')
    sleep(2)
    pretty = pretty_print('训练配置', conf)
    logging.info(pretty)

    # -----------------------------------------
    # 网络与损失
    # -----------------------------------------

    # 加载网络模型和优化器
    rpn_net, optimizer = init_training_model(conf, paths.output)

    # 打印网络结构
    load_bar("正在装载神经网络")
    logging.info('装载完成')
    sleep(3)

    # 显示网络结构
    logging.info('神经网络结构一览')
    sleep(2)
    logging.info(rpn_net)

    # 设置损失
    criterion_det = RPN_3D_loss(conf)

    # 自定义预训练网络
    if 'pretrained' in conf:

        load_weights(rpn_net, conf.pretrained)

    # 继续训练
    if restore:
        start_iter = (restore - 1)
        resume_checkpoint(optimizer, rpn_net, paths.weights, restore)

    freeze_blacklist = None if 'freeze_blacklist' not in conf else conf.freeze_blacklist
    freeze_whitelist = None if 'freeze_whitelist' not in conf else conf.freeze_whitelist

    freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)

    optimizer.zero_grad()

    start_time = time()

    # -----------------------------------------
    # 训练
    # -----------------------------------------
    logging.info('开始训练')

    training_bar = tqdm(total=conf.display)
    for iteration in range(start_iter, conf.max_iter):

        # 迭代
        iterator, images, imobjs = next_iteration(dataset.loader, iterator)

        # 动态调整学习率
        adjust_lr(conf, optimizer, iteration)

        # 前向传播
        cls, prob, bbox_2d, bbox_3d, feat_size = rpn_net(images)

        # 计算损失
        det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

        total_loss = det_loss
        stats = det_stats

        # 后向传播
        if total_loss > 0:

            total_loss.backward()

            # 通过跳过梯度步骤来模拟更大的批次
            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                optimizer.step()
                optimizer.zero_grad()

        # 使用tracker追踪目标信息
        compute_stats(tracker, stats)

        training_bar.update(1)
        # -----------------------------------------
        # 显示
        # -----------------------------------------
        if (iteration + 1) % conf.display == 0 and iteration > start_iter:

            # 将结果记录到日志中
            log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

            # 展示结果
            # if has_visdom:
            #     display_stats(vis, tracker, iteration, start_time, start_iter, conf.max_iter, conf_name, pretty)

            # 重设tracker
            tracker = edict()

            # 重设进度条
            training_bar.close()
            training_bar = tqdm(total=conf.display)

        # -----------------------------------------
        # 测试网络
        # -----------------------------------------
        if (iteration + 1) % conf.snapshot_iter == 0 and iteration > start_iter:

            # 存储检查点
            save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))

            if conf.do_test:

                # 验证模式
                rpn_net.eval()

                # 必要的路径
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # 测试kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'kitti':

                    # 删除原结果并重做
                    results_path = os.path.join(results_path, 'data')
                    mkdir_if_missing(results_path, delete_if_exist=True)

                    test_kitti_3d(conf.dataset_test, rpn_net, conf, results_path, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # 回到训练模式
                rpn_net.train()

                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)


# 从命令行运行
if __name__ == "__main__":
    main(sys.argv[1:])
