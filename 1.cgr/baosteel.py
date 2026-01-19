import os
import json

# ================= 配置区域 =================
DATASET = "baosteel"
JSON_DIR = f"./exps/cgr/{DATASET}/5"
FIRST_STAGE_CONFIG = os.path.join(JSON_DIR, "first_stage.json")
SECOND_STAGE_CONFIG = os.path.join(JSON_DIR, "second_stage.json")

SEEDS = [2025, 2026, 2027]
BETAS = [0.1, 0.5, 1.0]           # beta_cvae
LAMBDAS = [10.0, 50.0, 100.0]     # lambda_mmd_base
RFF_DIMS = [256, 512, 1024]       # D_rff
# ===========================================

def load_json_params(path):
    with open(path, 'r') as f:
        return json.load(f)

def run():
    # 1. 读取 First Stage 配置以获取路径结构 (init_cls, increment)
    params1 = load_json_params(FIRST_STAGE_CONFIG)
    model_name = params1.get("model_name", "cgr")
    init_cls = params1["init_cls"]
    increment = params1["increment"]
    base_log_dir = params1.get("log_dir", "logs")

    for seed in SEEDS:
        print(f"\n{'='*20} Processing Seed: {seed} {'='*20}")
        
        # -------------------------------------------------
        # Step 1: 运行 First Stage (每个 Seed 只运行一次)
        # -------------------------------------------------
        # 为 First Stage 创建一个带 seed 的唯一 log_name，防止被其他 seed 覆盖
        fs_log_name = f"first_stage_seed{seed}"
        
        # 构造 Checkpoint 路径: logs/cgr/baosteel/10/2/first_stage_seed2025
        # 注意：trainer.py 的路径拼接逻辑是 {model}/{dataset}/{init}/{inc}/{log_name}
        fs_ckpt_path = os.path.join(base_log_dir, model_name, DATASET, str(init_cls), str(increment), fs_log_name)

        print(f"--- [Stage 1] Training Base Model for Seed {seed} ---")
        # 检查是否已经跑过 (可选)，这里默认覆盖运行
        cmd_stage1 = (
            f"python main.py "
            f"--config {FIRST_STAGE_CONFIG} "
            f"--seed {seed} "
            f"--log_name {fs_log_name}" 
        )
        os.system(cmd_stage1)

        # -------------------------------------------------
        # Step 2: 运行 Grid Search (共享 Stage 1 权重)
        # -------------------------------------------------
        for beta in BETAS:
            for lam in LAMBDAS:
                for rff in RFF_DIMS:
                    print(f"--- [Stage 2] Grid: Beta={beta}, Lam={lam}, RFF={rff} ---")
                    
                    # 构造 Stage 2 的 log_name，包含所有超参
                    ss_log_name = f"stage2_s{seed}_b{beta}_l{lam}_r{rff}"
                    
                    cmd_stage2 = (
                        f"python main.py "
                        f"--config {SECOND_STAGE_CONFIG} "
                        f"--seed {seed} "
                        f"--beta_cvae {beta} "
                        f"--lambda_mmd_base {lam} "
                        f"--D_rff {rff} "
                        f"--ckpt_path {fs_ckpt_path} "  # <--- 关键：强制读取刚才跑完的 Stage 1
                        f"--log_name {ss_log_name}"     # 保存到独立日志
                    )
                    os.system(cmd_stage2)

if __name__ == "__main__":
    run()