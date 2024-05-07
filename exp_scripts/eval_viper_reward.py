"""Evalutate the viper reward (likelihood) given a dataset and a model"""

import logging as log
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

log.basicConfig(level=log.INFO)

from aimev2.data import SequenceDataset
from aimev2.models.ssm import SSM, ssm_classes
from aimev2.utils import DATA_PATH, MODEL_PATH, parse_world_model_config, symlog


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_names", type=str, nargs="+")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--use_symlog", action="store_true")
    args = parser.parse_args()

    config = OmegaConf.load(os.path.join(MODEL_PATH, args.model_name, "config.yaml"))
    dataset_folders = [
        os.path.join(DATA_PATH, dataset_name) for dataset_name in args.dataset_names
    ]
    datasets = [
        SequenceDataset(dataset_folder, config["horizon"], overlap=False)
        for dataset_folder in dataset_folders
    ]
    data = datasets[0][0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sensor_layout = config["env"]["sensors"]
    world_model_config = parse_world_model_config(
        config, sensor_layout, data, predict_reward=False, use_probe=False
    )
    world_model_name = world_model_config.pop("name")
    world_model_config["action_dim"] = 0
    world_model: SSM = ssm_classes[world_model_name](**world_model_config)
    world_model.load_state_dict(
        torch.load(
            os.path.join(MODEL_PATH, args.model_name, "model.pt"), map_location="cpu"
        ),
        strict=False,
    )
    world_model.requires_grad_(False)
    world_model.eval()
    world_model.to(device)

    rewards = []
    viper_rewards = []
    samples = 1
    plt.figure()
    with torch.inference_mode():
        for dataset_name, dataset in zip(args.dataset_names, datasets):
            rewards = []
            viper_rewards = []
            for i in tqdm(range(len(dataset.trajectories))):
                data = dataset.get_trajectory(i)
                rewards.append(torch.sum(data["reward"]).item())
                data.vmap_(lambda tensor: tensor.unsqueeze(dim=1))
                data.vmap_(
                    lambda tensor: torch.repeat_interleave(tensor, samples, dim=1)
                )
                data = data.to(device)
                action = data["pre_action"][..., :0]
                if args.use_symlog:
                    total_likelihood = 0
                    state = world_model.reset(samples)
                    for j in range(len(data)):
                        likelihood, state = world_model.likelihood_step(
                            data[j], action[j], state
                        )
                        total_likelihood += symlog(torch.tensor(likelihood)).item()
                    viper_rewards.append(total_likelihood)
                else:
                    _, _, _, metrics = world_model(data, action)
                    viper_rewards.append(metrics["elbo"])

            plt.scatter(rewards, viper_rewards, label=dataset_name)

    plt.xlabel("real reward")
    plt.ylabel("viper reward")
    plt.title(f"datasets : {args.dataset_names}\nmodel : {args.model_name}")
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    plt.legend()
    plt.tight_layout()
    dataset_names = [d.replace("/", ".") for d in args.dataset_names]
    plt.savefig(
        f"datasets-{dataset_names}-model-{args.model_name}-samples-{samples}-viper-evaluation-symlog={args.use_symlog}.png"
    )


if __name__ == "__main__":
    main()
