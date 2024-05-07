"""Export the trained models with progress from the log folder"""

import os
from argparse import ArgumentParser

from aimev2.utils import MODEL_PATH

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--log_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--max_epoch", type=int, required=True)
    args = parser.parse_args()

    log_folder = args.log_folder
    assert os.path.exists(log_folder)

    for file in sorted(
        [file for file in os.listdir(log_folder) if "pt" in file and "-e" in file]
    ):
        ep = int(file.split(".")[0].split("-e")[-1])
        if ep > args.max_epoch:
            continue
        model_folder = os.path.join(MODEL_PATH, f"{args.model_name}-ep{ep}")
        if not os.path.exists(model_folder):
            os.makedirs(model_folder)

        os.system(
            f'cp {os.path.join(log_folder, file)} {os.path.join(model_folder, "model.pt")}'
        )
        os.system(f'cp {os.path.join(log_folder, "config.yaml")} {model_folder}')
        # make a document of the source
        with open(os.path.join(model_folder, "source.txt"), "w") as f:
            f.write(log_folder)
