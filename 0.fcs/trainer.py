import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import json
import numpy as np

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
    logs_name = "{}/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'], args["log_name"])
    logs_name = os.path.join(log_dir, logs_name)

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = os.path.join(
        log_dir,
        "{}/{}/{}/{}/{}/{}_{}_{}".format(
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args['log_name'],
            args["prefix"],
            args["seed"],
            args["convnet_type"],
        )
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
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
    
    # === NEW: Initialize Accuracy Matrix ===
    # Shape: (Total_Tasks, Total_Tasks)
    # Row i: Results after training Task i
    # Col j: Accuracy on Task j
    num_tasks = data_manager.nb_tasks
    acc_matrix = np.zeros((num_tasks, num_tasks))
    # =======================================

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

        # === NEW: Fill Matrix & Log Detailed Metrics ===
        # cnn_accy["task_acc"] contains [acc_task_0, acc_task_1, ..., acc_task_current]
        if "task_acc" in cnn_accy:
            task_accs = cnn_accy["task_acc"]
            for t_id, acc in enumerate(task_accs):
                if t_id < num_tasks: # Safety check
                    acc_matrix[task, t_id] = acc
        
        # Check if this is the last task (and we are not forced to stop early by is_task0)
        is_last_task = (task == data_manager.nb_tasks - 1)
        if is_last_task:
            logging.info("=== Final Incremental Learning Metrics ===")
            
            # 1. Acc of each stage (Final Accuracy per Task)
            final_task_accs = acc_matrix[task, :task+1]
            logging.info("1. Final Acc of each stage (Task 0 to {}): {}".format(task, final_task_accs.tolist()))
            
            # 2. Average Acc (Average of final task accuracies)
            avg_acc = np.mean(final_task_accs)
            logging.info("2. Average Acc: {:.2f}".format(avg_acc))
            
            # 3. Forgetting Rate
            # Forgetting_j = max(Acc_prev_tasks, j) - Acc_final, j
            # We compute average forgetting over all OLD tasks (0 to T-1)
            if task > 0:
                forgetting = []
                for j in range(task): # Exclude current task
                    # Best accuracy for task j in previous steps
                    max_acc = np.max(acc_matrix[:task, j]) 
                    curr_acc = acc_matrix[task, j]
                    forgetting.append(max_acc - curr_acc)
                
                avg_forgetting = np.mean(forgetting)
                logging.info("3. Average Forgetting Rate: {:.2f}".format(avg_forgetting))
            else:
                logging.info("3. Forgetting Rate: 0.00 (First Task Only)")
            logging.info("========================================")
        # ===============================================

        # (保留原本的 only_new/only_old 评估)
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