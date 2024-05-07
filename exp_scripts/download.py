"""Download datasets and pretrained models from AIME-v2 paper."""

import os
from argparse import ArgumentParser

from aimev2.utils import DATA_PATH, MODEL_PATH

supported_name = [
    "cartpole-plan2explore-buffer",
    "walker-plan2explore-buffer",
    "hopper-plan2explore-buffer",
    "cheetah-plan2explore-buffer",
    "finger-plan2explore-buffer",
    "quadruped-plan2explore-buffer",
    "dmc_models",
    "metaworld_models",
    "metaworld_expert_datasets",
]

parts = {
    "walker-plan2explore-buffer": 2,
    "hopper-plan2explore-buffer": 2,
    "cheetah-plan2explore-buffer": 2,
    "quadruped-plan2explore-buffer": 4,
}

remote_url = "https://github.com/argmax-ai/aime-v2/releases/download/arxiv/"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", "-n", type=str, required=True)
    parser.add_argument("--keep_files", "-k", action="store_true")
    args = parser.parse_args()

    assert (
        args.name in supported_name
    ), f"please selected one of the following names: {supported_name}."

    output_folder = MODEL_PATH if "model" in args.name else DATA_PATH

    print("Downloading files ...")
    files = (
        [f"{args.name}.zip"]
        if args.name not in parts.keys()
        else [f"{args.name}-part{i}.zip" for i in range(parts[args.name])]
    )
    for file in files:
        os.system(f"wget -P {output_folder} {remote_url+file}")

    print("Extracting files ...")
    extract_folder = (
        output_folder
        if "buffer" not in args.name
        else os.path.join(output_folder, args.name)
    )
    for file in files:
        os.system(f"unzip {os.path.join(output_folder, file)} -d {extract_folder}")

    if not args.keep_files:
        print("Deleting downloaded files ...")
        for file in files:
            os.system(f"rm {os.path.join(output_folder, file)}")
