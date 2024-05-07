import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger("main")
from einops import rearrange

from aimev2.actor import StackPolicyActor
from aimev2.data import SequenceDataset, get_epoch_loader
from aimev2.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aimev2.logger import get_default_logger
from aimev2.models.base import MLP, MultimodalEncoder
from aimev2.runtimes import runtime_classes
from aimev2.utils import *


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bc")
def main(config: DictConfig):
    runtime = runtime_classes[config["runtime"]["name"]](config)

    setup_seed(config["seed"])

    log.info("Using the following config:\n" + OmegaConf.to_yaml(config))

    stack = 1 if config["environment_setup"] == "mdp" else config["stack"]

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    demonstration_dataset_folder = os.path.join(
        DATA_PATH, config["demonstration_dataset_name"]
    )
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    log.info("Creating environment ...")
    env_config = dict(config["env"])
    env_config["seed"] = config["seed"] * 2
    env_class_name = env_config.pop("class")
    test_env = env_classes[env_class_name](**env_config)
    test_env.enable_render(True)
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    sensor_layout = env_config["sensors"]
    encoder_configs = config["encoders"]
    sensor_shapes = get_sensor_shapes(test_env.observation_space)
    input_sensors, _, _ = get_inputs_outputs(sensor_layout, config["environment_setup"])
    selected_keys = input_sensors + ["pre_action"]
    multimodal_encoder_config = [
        (k, sensor_shapes[k], dict(encoder_configs[sensor_layout[k]["modility"]]))
        for k in input_sensors
    ]
    image_sensors = [k for k, v in sensor_layout.items() if v["modility"] == "visual"]

    log.info("Loading datasets ...")
    demonstration_dataset = SequenceDataset(
        demonstration_dataset_folder,
        stack + 1,
        overlap=True,
        max_capacity=config["num_expert_trajectories"],
        selected_keys=selected_keys,
        **config["data"]["dataset"],
    )
    eval_dataset = SequenceDataset(eval_folder, stack + 1, overlap=False)

    log.info("Creating models ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    policy_encoder = MultimodalEncoder(multimodal_encoder_config)
    policy_encoder = policy_encoder.to(device)

    policy_config = config["policy"]
    policy = MLP(
        policy_encoder.output_dim * stack,
        sensor_shapes["pre_action"],
        output_activation="tanh",
        **policy_config,
    )
    policy = policy.to(device)

    loss_fn = torch.nn.MSELoss()

    logger = get_default_logger(output_folder)

    policy_optim = torch.optim.Adam(
        [*policy.parameters(), *policy_encoder.parameters()], lr=config["policy_lr"]
    )

    log.info("Training Policy ...")
    train_size = int(
        len(demonstration_dataset) * config["train_validation_split_ratio"]
    )
    val_size = len(demonstration_dataset) - train_size
    demonstration_dataset_train, demonstration_dataset_val = (
        torch.utils.data.random_split(demonstration_dataset, [train_size, val_size])
    )
    train_loader = get_epoch_loader(
        demonstration_dataset_train,
        config["batch_size"],
        shuffle=True,
        **config["data"]["dataloader"],
    )
    val_loader = get_epoch_loader(
        demonstration_dataset_val,
        config["batch_size"],
        shuffle=False,
        **config["data"]["dataloader"],
    )
    e = 0
    best_val_loss = float("inf")
    convergence_count = 0
    while True:
        log.info(f"Starting epcoh {e}")

        metrics = {}
        train_metric_tracker = AverageMeter()
        for data in tqdm(iter(train_loader), disable=runtime.disable_tqdm):
            data = data.to(device)
            emb_policy = policy_encoder(data[:-1])
            emb_policy = rearrange(emb_policy, "t b f -> b (t f)")
            policy_action = policy(emb_policy)
            loss = loss_fn(data[-1]["pre_action"], policy_action)

            policy_optim.zero_grad()
            loss.backward()
            policy_optim.step()

            metric = {
                "train/policy_loss": loss.item(),
            }

            train_metric_tracker.add(metric)

        metrics.update(train_metric_tracker.get())

        val_metric_tracker = AverageMeter()
        with torch.no_grad():
            for data in tqdm(iter(val_loader), disable=runtime.disable_tqdm):
                data = data.to(device)
                emb_policy = policy_encoder(data[:-1])
                emb_policy = rearrange(emb_policy, "t b f -> b (t f)")
                policy_action = policy(emb_policy)
                loss = loss_fn(data[-1]["pre_action"], policy_action)

                metric = {
                    "val/policy_loss": loss.item(),
                }

                val_metric_tracker.add(metric)

        metrics.update(val_metric_tracker.get())

        if test_env.interactive:
            log.info("Evaluating the model ...")
            with torch.no_grad():
                actor = StackPolicyActor(policy_encoder, policy, stack)
                metrics.update(eval_actor_on_env(test_env, actor, image_sensors))
            eval_dataset.update()
            for image_key in image_sensors:
                metrics[f"eval_video_{image_key}"] = (
                    eval_dataset.get_trajectory(-1)[image_key]
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    * 255
                )

        logger(metrics, e)

        e += 1

        if metrics["val/policy_loss"] < best_val_loss:
            best_val_loss = metrics["val/policy_loss"]
            torch.save(
                policy_encoder.state_dict(),
                os.path.join(output_folder, "policy_encoder.pt"),
            )
            torch.save(policy.state_dict(), os.path.join(output_folder, "policy.pt"))
            convergence_count = 0
        else:
            convergence_count += 1
            if (
                convergence_count >= config["patience"]
                and e >= config["min_policy_epoch"]
            ):
                break

    log.info(f"Policy training finished in {e} epoches!")

    if test_env.interactive:
        # restore the best policy for a final test
        policy_encoder.load_state_dict(
            torch.load(os.path.join(output_folder, "policy_encoder.pt"))
        )
        policy.load_state_dict(torch.load(os.path.join(output_folder, "policy.pt")))

        metrics = {}
        with torch.no_grad():
            actor = StackPolicyActor(policy_encoder, policy, stack)
            metrics.update(
                eval_actor_on_env(
                    test_env, actor, image_sensors, config["num_test_trajectories"]
                )
            )
        eval_dataset.update()
        for image_key in image_sensors:
            metrics[f"eval_video_{image_key}"] = (
                eval_dataset.get_trajectory(-1)[image_key]
                .permute(0, 2, 3, 1)
                .contiguous()
                * 255
            )

        logger(metrics, e)

    runtime.finish(output_folder)


if __name__ == "__main__":
    main()
