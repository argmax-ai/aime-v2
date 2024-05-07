"""Exam the average rewards and length of a given dataset"""

import os
from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm

from aimev2.data import SequenceDataset
from aimev2.utils import DATA_PATH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    model_folder = os.path.join(DATA_PATH, args.dataset_name)
    dataset = SequenceDataset(
        model_folder, 1000000000, False, selected_keys=["reward", "success"]
    )

    rewards = []
    successes = []
    lengths = []
    for traj in tqdm(dataset.trajectories):
        traj = traj.get_trajectory()
        rewards.append(traj["reward"].sum().item())
        lengths.append(len(traj["reward"]))
        if "success" in traj.keys():
            successes.append(traj["success"][-1].sum().item())

    print(f"Information of dataset {args.dataset_name}:")
    print(f"Length of the trajectories: {np.mean(lengths)} +- {np.std(lengths)}.")
    print(f"Rewards of the trajectories: {np.mean(rewards)} +- {np.std(rewards)}.")
    if len(successes) > 0:
        print(f"Success rate of the trajectories: {np.mean(successes)}.")
