import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from aimev2.data import ArrayDict
from aimev2.env import MetaWorld, SaveTrajectories
from aimev2.utils import DATA_PATH, setup_seed

TASK_SET = {
    "mt30": [
        # 19 original dmcontrol tasks
        "walker-stand",
        "walker-walk",
        "walker-run",
        "cheetah-run",
        "reacher-easy",
        "reacher-hard",
        "acrobot-swingup",
        "pendulum-swingup",
        "cartpole-balance",
        "cartpole-balance-sparse",
        "cartpole-swingup",
        "cartpole-swingup-sparse",
        "cup-catch",
        "finger-spin",
        "finger-turn-easy",
        "finger-turn-hard",
        "fish-swim",
        "hopper-stand",
        "hopper-hop",
        # 11 custom dmcontrol tasks
        "walker-walk-backwards",
        "walker-run-backwards",
        "cheetah-run-backwards",
        "cheetah-run-front",
        "cheetah-run-back",
        "cheetah-jump",
        "hopper-hop-backwards",
        "reacher-three-easy",
        "reacher-three-hard",
        "cup-spin",
        "pendulum-spin",
    ],
    "mt80": [
        # 19 original dmcontrol tasks
        "walker-stand",
        "walker-walk",
        "walker-run",
        "cheetah-run",
        "reacher-easy",
        "reacher-hard",
        "acrobot-swingup",
        "pendulum-swingup",
        "cartpole-balance",
        "cartpole-balance-sparse",
        "cartpole-swingup",
        "cartpole-swingup-sparse",
        "cup-catch",
        "finger-spin",
        "finger-turn-easy",
        "finger-turn-hard",
        "fish-swim",
        "hopper-stand",
        "hopper-hop",
        # 11 custom dmcontrol tasks
        "walker-walk-backwards",
        "walker-run-backwards",
        "cheetah-run-backwards",
        "cheetah-run-front",
        "cheetah-run-back",
        "cheetah-jump",
        "hopper-hop-backwards",
        "reacher-three-easy",
        "reacher-three-hard",
        "cup-spin",
        "pendulum-spin",
        # meta-world mt50
        "mw-assembly",
        "mw-basketball",
        "mw-button-press-topdown",
        "mw-button-press-topdown-wall",
        "mw-button-press",
        "mw-button-press-wall",
        "mw-coffee-button",
        "mw-coffee-pull",
        "mw-coffee-push",
        "mw-dial-turn",
        "mw-disassemble",
        "mw-door-open",
        "mw-door-close",
        "mw-drawer-close",
        "mw-drawer-open",
        "mw-faucet-open",
        "mw-faucet-close",
        "mw-hammer",
        "mw-handle-press-side",
        "mw-handle-press",
        "mw-handle-pull-side",
        "mw-handle-pull",
        "mw-lever-pull",
        "mw-peg-insert-side",
        "mw-peg-unplug-side",
        "mw-pick-out-of-hole",
        "mw-pick-place",
        "mw-pick-place-wall",
        "mw-plate-slide",
        "mw-plate-slide-side",
        "mw-plate-slide-back",
        "mw-plate-slide-back-side",
        "mw-push-back",
        "mw-push",
        "mw-push-wall",
        "mw-reach",
        "mw-reach-wall",
        "mw-shelf-place",
        "mw-soccer",
        "mw-stick-push",
        "mw-stick-pull",
        "mw-sweep-into",
        "mw-sweep",
        "mw-window-open",
        "mw-window-close",
        "mw-bin-picking",
        "mw-box-close",
        "mw-door-lock",
        "mw-door-unlock",
        "mw-hand-insert",
    ],
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", "-i", type=str)
    args = parser.parse_args()

    setup_seed(42)

    ROOT = args.root
    FILES = ["chunk_11.pt", "chunk_12.pt", "chunk_13.pt"]
    sample_from = 200
    num_trajectories_per_task = 200
    image_size = 64
    OUTPUT_ROOT = os.path.join(DATA_PATH, "tdmpc2-metaworld-mt50")

    for file in FILES:
        print(f"loading file {file} ...")
        data = torch.load(os.path.join(ROOT, file), map_location="cpu")
        data = data[data["task"][:, 0] >= 30]
        tasks = set(data["task"][:, 0].numpy().tolist())
        print(f"file have {data.shape[0]} trajectories and {len(tasks)} tasks.")

        for task_id in tasks:
            task_name = TASK_SET["mt80"][task_id][3:]
            print(f"running task {task_name}")
            output_dir = os.path.join(OUTPUT_ROOT, f"metaworld-{task_name}")
            env = MetaWorld(
                f"metaworld-{task_name}",
                action_repeat=2,
                size=(image_size, image_size),
                seed=1,
            )
            env = SaveTrajectories(env, output_dir)
            task_data = data[data["task"][:, 0] == task_id]

            for index in tqdm(range(num_trajectories_per_task)):
                traj = task_data[index * (sample_from // num_trajectories_per_task)]

                obs_index = (
                    1
                    if task_name
                    in ["button-press-topdown", "button-press-topdown-wall"]
                    else 0
                )
                obs = env.set_state_from_obs(traj["obs"][obs_index].numpy())
                env.trajectory_data.append(ArrayDict(obs))

                for i in range(100):
                    obs = env.step(traj["action"][i + 1][:4].numpy())
