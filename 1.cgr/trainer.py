#
import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import json

def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)

def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    log_dir = args["log_dir"]
    
    # 保持原有的目录结构
    logs_name = "{}/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'], args["log_name"])
    logs_name = os.path.join(log_dir, logs_name)

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    # --- 修改：构建包含超参数的后缀 ---
    # 检查 args 中是否有这些 key，如果有则加入到文件名中
    hyper_suffix = ""
    if "beta_cvae" in args:
        hyper_suffix += "_beta{}".format(args["beta_cvae"])
    if "lambda_mmd_base" in args:
        hyper_suffix += "_mmd{}".format(args["lambda_mmd_base"])
    if "D_rff" in args:
        hyper_suffix += "_rff{}".format(args["D_rff"])

    # 拼接完整文件名
    logfilename = os.path.join(
        log_dir,
        "{}/{}/{}/{}/{}/{}_{}_{}{}".format( # 注意末尾增加了一个 {}
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args['log_name'],
            args["prefix"],
            args["seed"],
            args["convnet_type"],
            hyper_suffix  # 填入超参数后缀
        )
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True # 强制重新配置 logging，防止循环中 handler 重复累积
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": [], "top5": []}

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info("Trainable params: {}".format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)

        # === eval
        cnn_accy, nme_accy = model.eval_task()
        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        if nme_accy is not None:
            logging.info("NME: {}".format(nme_accy["grouped"]))
        else:
            logging.info("NME: None")

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])
        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))

        # 旧写法
        cnn_accy_new, nme_accy_new = model.eval_task(only_new=True)
        cnn_accy_old, nme_accy_old = model.eval_task(only_old=True)
        logging.info("Eval only_new => CNN top1: {}, NME: {}".format(
            cnn_accy_new["top1"], "N/A" if nme_accy_new is None else nme_accy_new["top1"]))
        logging.info("Eval only_old => CNN top1: {}, NME: {}".format(
            cnn_accy_old["top1"], "N/A" if nme_accy_old is None else nme_accy_old["top1"]))

        model.after_task()
        if args["is_task0"]:
            break

def _set_device(args):
    device_type = args["device"]
    gpus = []
    for dev in device_type:
        if dev == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(dev))
        gpus.append(device)
    args["device"] = gpus

def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))