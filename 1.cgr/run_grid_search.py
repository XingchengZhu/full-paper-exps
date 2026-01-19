import os

seeds = [2025, 2026, 2027]
betas = [0.1, 0.5, 1.0]           # beta_cvae
lambdas = [10.0, 50.0, 100.0]     # lambda_mmd_base
rff_dims = [256, 512, 1024]       # D_rff

# 固定配置
config_path = "./exps/cgr/cifar100/5/second_stage.json" 

# 遍历
for seed in seeds:
    for beta in betas:
        for lam in lambdas:
            for rff in rff_dims:
                print(f"Running: Seed={seed}, Beta={beta}, Lambda={lam}, RFF={rff}")
                
                # 构造命令
                cmd = (
                    f"python main.py "
                    f"--config={config_path} "
                    f"--seed {seed} "
                    f"--beta_cvae {beta} "
                    f"--lambda_mmd_base {lam} "
                    f"--D_rff {rff}"
                )
                
                # 执行命令
                os.system(cmd)