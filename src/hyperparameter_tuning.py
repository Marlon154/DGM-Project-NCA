import argparse
import os

import yaml
import wandb
from train import main


def hyperparameter_tuning(config, run_count, user_name, path, project_name="dgm-nca-hyperparameters"):
    sweep_id = wandb.sweep(config, project=project_name)
    wandb.agent(sweep_id, main, count=run_count)
    save_best_configuration_for_sweep(user_name, project_name, sweep_id, path)


def save_best_configuration_for_sweep(user_name, project_name, sweep_id, path):
    # Get best run parameters
    api = wandb.Api()
    sweep = api.sweep(f"{user_name}/{project_name}/sweeps/{sweep_id}")
    best_run = sweep.best_run(order='validation/accuracy')
    best_parameters = best_run.config

    # Save best Model + corresponding parameters
    path = f"{path}"
    print(f"Saving best run to {path}")
    os.makedirs(path, exist_ok=True)
    best_parameters["model_path"] = f"{path}/{sweep_id}_best_nca_model.pth"
    yaml.dump(best_parameters, open(f"{path}/{sweep_id}_best_configuration.yaml", 'w'))
    main(best_parameters)


def hyperparameter_tuning_all_filter_loss_combs(config, run_count, user_name, path, s_filter=None):
    filter = ["gaussian", "sobel", "laplacian", "identity", "sobel_identity"]
    filter = [s_filter] if s_filter is not None else filter
    loss = ["hinge", "manhattan", "mse", "ssim", "combined_ssim_l1"]
    for f in filter:
        for l in loss:
            config["parameters"]["loss_function"]["value"] = l
            config["parameters"]["filter_name"]["value"] = f
            config["sweep_name"] = f"s_{l}_{f}"
            config["model_path"] = f"models/28-_{l}_{f}.pth"
            project_name = f"dgm-nca"
            hyperparameter_tuning(config, run_count, user_name, path, project_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter Sweep")
    parser.add_argument("-c", "--config", type=str, default="conf/sweep_config.yaml", help="Path to sweep config.")
    parser.add_argument("-r", "--run_count", type=int, default=50, help="Determines how many times the sweep agent runs the train function")
    parser.add_argument("-u", "--user_name", type=str, default="mayma-lab", help="Weights and Biases username to retrieve best config")
    parser.add_argument("-p", "--best_model_path", type=str, default="models/best_configs", help="Path where best configuration for sweep is saved")
    parser.add_argument("--filter", type=str, default=None, help="Specify filter to run hyperparameter tuning for.")
    args = parser.parse_args()

    sweep_config = yaml.safe_load(open(args.config, "r"))
    run_count = args.run_count
    user_name = args.user_name
    path = args.best_model_path
    filter = args.filter
    # hyperparameter_tuning(sweep_config,run_count, user_name, path)
    hyperparameter_tuning_all_filter_loss_combs(sweep_config, run_count, user_name, path, filter)
