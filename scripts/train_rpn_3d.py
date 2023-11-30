# -----------------------------------------
# python模块
# -----------------------------------------
from easydict import EasyDict as edict
from getopt import getopt
import numpy as np
import sys
import os

# stop python from writing so much bytecode
sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

# -----------------------------------------
# 自定义模块
# -----------------------------------------
from lib.core import *
from lib.imdb_util import *
from lib.loss.rpn_3d import *


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
        raise ValueError('Please provide a configuration file name, e.g., --config=<config_name>')

    # -----------------------------------------
    # 基础设置
    # -----------------------------------------

    conf = init_config(conf_name)
    paths = init_training_paths(conf_name)

    init_torch(conf.rng_seed, conf.cuda_seed)
    init_log_file(paths.logs)

    # 不用visdom
    # vis = init_visdom(conf_name, conf.visdom_port)

    # 默认
    start_iter = 0
    tracker = edict()
    iterator = None
    # has_visdom = vis is not None
    has_visdom = None

    dataset = Dataset(conf, paths.data, paths.output)

    generate_anchors(conf, dataset.imdb, paths.output)
    compute_bbox_stats(conf, dataset.imdb, paths.output)


    # -----------------------------------------
    # 存储设置
    # -----------------------------------------

    # 保存设置
    pickle_write(os.path.join(paths.output, 'conf.pkl'), conf)

    # 显示设置
    pretty = pretty_print('conf', conf)
    logging.info(pretty)


    # -----------------------------------------
    # 网络与损失
    # -----------------------------------------

    # 训练网络
    rpn_net, optimizer = init_training_model(conf, paths.output)
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

    for iteration in range(start_iter, conf.max_iter):

        # 下一个迭代
        iterator, images, imobjs = next_iteration(dataset.loader, iterator)

        # 动态学习率
        adjust_lr(conf, optimizer, iteration)

        # 前向传播
        cls, prob, bbox_2d, bbox_3d, feat_size = rpn_net(images)

        # 损失
        det_loss, det_stats = criterion_det(cls, prob, bbox_2d, bbox_3d, imobjs, feat_size)

        total_loss = det_loss
        stats = det_stats

        # 后向传播
        if total_loss > 0:

            total_loss.backward()

            # batch skip, simulates larger batches by skipping gradient step
            if (not 'batch_skip' in conf) or ((iteration + 1) % conf.batch_skip) == 0:
                optimizer.step()
                optimizer.zero_grad()

        # keep track of stats
        compute_stats(tracker, stats)

        # -----------------------------------------
        # display
        # -----------------------------------------
        if (iteration + 1) % conf.display == 0 and iteration > start_iter:

            # log results
            log_stats(tracker, iteration, start_time, start_iter, conf.max_iter)

            # display results
            # if has_visdom:
            #     display_stats(vis, tracker, iteration, start_time, start_iter, conf.max_iter, conf_name, pretty)

            # reset tracker
            tracker = edict()

        # -----------------------------------------
        # test network
        # -----------------------------------------
        if (iteration + 1) % conf.snapshot_iter == 0 and iteration > start_iter:

            # store checkpoint
            save_checkpoint(optimizer, rpn_net, paths.weights, (iteration + 1))

            if conf.do_test:

                # eval mode
                rpn_net.eval()

                # necessary paths
                results_path = os.path.join(paths.results, 'results_{}'.format((iteration + 1)))

                # -----------------------------------------
                # test kitti
                # -----------------------------------------
                if conf.test_protocol.lower() == 'kitti':

                    # delete and re-make
                    results_path = os.path.join(results_path, 'data')
                    mkdir_if_missing(results_path, delete_if_exist=True)

                    test_kitti_3d(conf.dataset_test, rpn_net, conf, results_path, paths.data)

                else:
                    logging.warning('Testing protocol {} not understood.'.format(conf.test_protocol))

                # train mode
                rpn_net.train()

                freeze_layers(rpn_net, freeze_blacklist, freeze_whitelist)


# run from command line
if __name__ == "__main__":
    main(sys.argv[1:])
