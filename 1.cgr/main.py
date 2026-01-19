#
import json
import argparse
from trainer import train


def main():
    # 1. 解析命令行参数
    args_namespace = setup_parser().parse_args()
    args_cli = vars(args_namespace)  # 转为字典

    # 2. 加载配置文件参数
    param = load_json(args_namespace.config)

    # 3. 合并参数：CLI参数 > JSON参数 > 默认值
    # 先使用 JSON 参数作为基础
    final_args = param
    # 将 CLI 中非 None 的参数更新进去 (覆盖 JSON 中的同名参数)
    for key, value in args_cli.items():
        if value is not None:
            final_args[key] = value

    train(final_args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param



def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/finetune.json',
                        help='Json file of settings.')
    
    # --- 新增 Grid Search 超参数 ---
    parser.add_argument('--beta_cvae', type=float, help='weight for CVAE loss')
    parser.add_argument('--lambda_mmd_base', type=float, help='base weight for MMD loss')
    parser.add_argument('--D_rff', type=int, help='dimension for RFF')
    parser.add_argument('--seed', type=int, nargs='+', help='random seeds') 

    # --- 【必须新增】用于路径重定向的参数 ---
    parser.add_argument('--ckpt_path', type=str, help='Path to load checkpoint (overrides json)')
    parser.add_argument('--log_name', type=str, help='Name of the log folder (overrides json)')

    return parser


if __name__ == '__main__':
    main()