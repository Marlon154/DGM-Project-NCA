import argparse
import yaml
import wandb
from train import main

def hyperparameter_tuning(config, run_count):
    sweep_id = wandb.sweep(config, project="dgm-nca-hyperparameters")
    wandb.agent(sweep_id, main, count=run_count)




def hyperparameter_tuning_all_filter_loss_combs(config, run_count):
    filter = ["gaussian", "sobel", "laplacian", "identity"]
    loss = ["hinge", "manhattan", "mse", "ssim", "combined_ssim_l1"]
    for f in filter:
        for l in loss:
            config["parameters"]["loss_function"]["value"] = l
            config["parameters"]["filter_name"]["value"] = f
            sweep_id = wandb.sweep(config, project=f"dgm-nca-hyperparameters_loss_{l}_filter_{f}")
            wandb.agent(sweep_id, main,count=run_count)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("-c", "--config", type=str, default="conf/sweep_config.yaml", help="Path to sweep config.")
    parser.add_argument("-r", "--run_count", type=int, default=20, help="Determines how many times the sweep agent runs the train function")
    args = parser.parse_args()

    sweep_config = yaml.safe_load(open(args.config, "r"))
    run_count = args.run_count
    hyperparameter_tuning_all_filter_loss_combs(sweep_config, run_count)

