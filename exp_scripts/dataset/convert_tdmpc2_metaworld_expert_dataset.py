import os
from argparse import ArgumentParser

import numpy as np

from aimev2.data import ArrayDict
from aimev2.utils import DATA_PATH, deepdown, savehdf5, setup_seed

setup_seed(42)

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

    ROOT = args.root

    for task in TASK_SET["mt80"][30:]:
        task_name = task[3:]
        print(f"running task {task_name}")
        output_dir = os.path.join(DATA_PATH, f"tdmpc2-metaworld-{task_name}-expert")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        input_dir = deepdown(os.path.join(ROOT, task))
        files = [f for f in os.listdir(input_dir) if "npz" in f]
        files = sorted(files, key=lambda x: int(x.split(".")[0]))

        index = 0
        for file in files:
            data = ArrayDict(dict(np.load(os.path.join(input_dir, file))))
            if not data["success"][-1]:
                continue
            data.expand_dim_equal_()

            data["state"] = data.pop("obs")
            data["pre_action"] = data.pop("action")
            data["is_first"] = np.zeros_like(data["reward"])
            data["is_first"][0] = 1
            data["is_last"] = np.zeros_like(data["reward"])
            data["is_last"][-1] = 1
            data["is_terminal"] = np.zeros_like(data["reward"])

            savehdf5(data, os.path.join(output_dir, f"{index}.hdf5"))
            index += 1

            if index >= 10:
                break
