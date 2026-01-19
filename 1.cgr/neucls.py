import os
import json

# ================= 配置区域 =================
DATASET = "neucls"
JSON_DIR = f"./exps/cgr/{DATASET}/5"
FIRST_STAGE_CONFIG = os.path.join(JSON_DIR, "first_stage.json")
SECOND_STAGE_CONFIG = os.path.join(JSON_DIR, "second_stage.json")

SEEDS = [1991, 1993, 2001, 2000, 2025, 2026, 2027]
BETAS = [0.1, 0.5, 1.0]           
LAMBDAS = [10.0, 50.0, 100.0]     
RFF_DIMS = [256, 512, 1024]       
# ===========================================

def load_json_params(path):
    with open(path, 'r') as f:
        return json.load(f)

def run():
    params1 = load_json_params(FIRST_STAGE_CONFIG)
    model_name = params1.get("model_name", "cgr")
    init_cls = params1["init_cls"]
    increment = params1["increment"]
    base_log_dir = params1.get("log_dir", "logs")

    for seed in SEEDS:
        print(f"\n{'='*20} Processing Seed: {seed} {'='*20}")
        
        # Step 1: First Stage
        fs_log_name = f"first_stage_seed{seed}"
        fs_ckpt_path = os.path.join(base_log_dir, model_name, DATASET, str(init_cls), str(increment), fs_log_name)

        print(f"--- [Stage 1] Training Base Model for Seed {seed} ---")
        cmd_stage1 = (
            f"python main.py --config {FIRST_STAGE_CONFIG} --seed {seed} --log_name {fs_log_name}" 
        )
        os.system(cmd_stage1)

        # Step 2: Grid Search
        for beta in BETAS:
            for lam in LAMBDAS:
                for rff in RFF_DIMS:
                    print(f"--- [Stage 2] Grid: Beta={beta}, Lam={lam}, RFF={rff} ---")
                    ss_log_name = f"stage2_s{seed}_b{beta}_l{lam}_r{rff}"
                    
                    cmd_stage2 = (
                        f"python main.py "
                        f"--config {SECOND_STAGE_CONFIG} "
                        f"--seed {seed} "
                        f"--beta_cvae {beta} "
                        f"--lambda_mmd_base {lam} "
                        f"--D_rff {rff} "
                        f"--ckpt_path {fs_ckpt_path} " 
                        f"--log_name {ss_log_name}"
                    )
                    os.system(cmd_stage2)

if __name__ == "__main__":
    run()