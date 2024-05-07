import os
import pickle
from argparse import ArgumentParser

import cv2
import numpy as np

from aimev2.utils import savehdf5

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str)
    parser.add_argument("--output_folder", "-o", type=str)
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(args.input_file, "rb") as f:
        data = pickle.load(f)

    # the data is in [N, T, ...] format
    images, states, actions, rewards = data

    images = images[:, :, -3:]
    images = np.transpose(images, (0, 1, 3, 4, 2))
    reshaped_images = np.zeros((*images.shape[:2], 64, 64, 3))
    for i in range(reshaped_images.shape[0]):
        for j in range(reshaped_images.shape[1]):
            reshaped_images[i, j] = cv2.resize(images[i, j], (64, 64))
    pre_actions = np.concatenate(
        [np.zeros_like(actions[:, :1]), actions[:, :-1]], axis=1
    )
    rewards = rewards[..., None]
    rewards = np.concatenate([np.zeros_like(rewards[:, :1]), rewards[:, :-1]], axis=1)
    is_firsts = np.zeros_like(rewards)
    is_firsts[:, 0] = 1.0
    is_terminals = np.zeros_like(rewards)
    is_last = np.zeros_like(rewards)
    is_last[:, -1] = 1.0

    for i in range(images.shape[0]):
        traj_data = {
            "image": reshaped_images[i],
            "pre_action": pre_actions[i],
            "state": states[i],
            "reward": rewards[i],
            "is_first": is_firsts[i],
            "is_terminal": is_terminals[i],
            "is_last": is_last[i],
        }
        savehdf5(traj_data, os.path.join(args.output_folder, f"{i}.hdf5"))
