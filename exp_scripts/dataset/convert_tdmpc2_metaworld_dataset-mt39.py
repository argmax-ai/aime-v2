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

metaworld_difficulties = {
    "easy": [
        "button-press",
        "button-press-topdown",
        "button-press-topdown-wall",
        "button-press-wall",
        "coffee-button",
        "dial-turn",
        "door-close",
        "door-lock",
        "door-open",
        "door-unlock",
        "drawer-close",
        "drawer-open",
        "faucet-close",
        "faucet-open",
        "handle-press",
        "handle-press-side",
        "handle-pull",
        "handle-pull-side",
        "lever-pull",
        "plate-slide",
        "plate-slide-back",
        "plate-slide-back-side",
        "plate-slide-side",
        "reach",
        "reach-wall",
        "window-close",
        "window-open",
        "peg-unplug-side",
    ],
    "medium": [
        "basketball",
        "bin-picking",
        "box-close",
        "coffee-pull",
        "coffee-push",
        "hammer",
        "peg-insert-side",
        "push-wall",
        "soccer",
        "sweep",
        "sweep-into",
    ],
    "hard": [
        "assembly",
        "hand-insert",
        "pick-out-of-hole",
        "pick-place",
        "push",
        "push-back",
    ],
    "very hard": [
        "shelf-place",
        "disassemble",
        "stick-pull",
        "stick-push",
        "pick-place-wall",
    ],
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root", "-i", type=str)
    args = parser.parse_args()

    setup_seed(42)

    ROOT = args.root
    FILES = ["chunk_11.pt", "chunk_12.pt", "chunk_13.pt"]
    image_size = 64
    OUTPUT_ROOT = os.path.join(DATA_PATH, "tdmpc2-metaworld-mt39")
    selected_tasks = metaworld_difficulties["easy"] + metaworld_difficulties["medium"]

    for file in FILES:
        print(f"loading file {file} ...")
        data = torch.load(os.path.join(ROOT, file), map_location="cpu")
        data = data[data["task"][:, 0] >= 30]
        tasks = set(data["task"][:, 0].numpy().tolist())
        print(f"file have {data.shape[0]} trajectories and {len(tasks)} tasks.")

        for task_id in tasks:
            task_name = TASK_SET["mt80"][task_id][3:]
            if task_name not in selected_tasks:
                continue
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

            for index in tqdm(range(250)):
                traj = task_data[index * 40]

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
