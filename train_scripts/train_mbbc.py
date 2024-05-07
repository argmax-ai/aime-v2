import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger("main")

from aimev2.actor import PolicyActor
from aimev2.data import SequenceDataset, get_epoch_loader
from aimev2.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aimev2.logger import get_default_logger
from aimev2.models.policy import TanhGaussianPolicy
from aimev2.models.ssm import ssm_classes
from aimev2.runtimes import runtime_classes
from aimev2.utils import *


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="mbbc")
def main(config: DictConfig):
    runtime = runtime_classes[config["runtime"]["name"]](config)

    setup_seed(config["seed"])

    log.info("Using the following config:\n" + OmegaConf.to_yaml(config))

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    model_folder = os.path.join(MODEL_PATH, config["pretrained_model_name"])
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
    world_model_config = parse_world_model_config(
        config, sensor_layout, test_env.observation_space, False
    )
    selected_keys = get_seleted_keys_from_world_model_config(world_model_config)
    world_model_name = world_model_config.pop("name")
    image_sensors, used_image_sensors = get_image_sensors(
        world_model_config, sensor_layout
    )

    log.info("Loading datasets ...")
    demonstration_dataset = SequenceDataset(
        demonstration_dataset_folder,
        config["horizon"],
        overlap=True,
        max_capacity=config["num_expert_trajectories"],
        selected_keys=selected_keys,
        **config["data"]["dataset"],
    )
    eval_dataset = SequenceDataset(eval_folder, config["horizon"], overlap=False)

    log.info("Creating models ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    model = ssm_classes[world_model_name](**world_model_config)
    model.load_state_dict(
        torch.load(os.path.join(model_folder, "model.pt"), map_location="cpu"),
        strict=False,
    )
    model = model.to(device)
    if "reward" in model.decoders.keys():
        model.decoders.pop("reward")
    if config["freeze_model"]:
        log.info("freeze the model weights")
        model.requires_grad_(False)

    policy_config = config["policy"]
    policy = TanhGaussianPolicy(
        model.state_feature_dim, world_model_config["action_dim"], **policy_config
    )
    policy = policy.to(device)

    loss_fn = torch.nn.MSELoss()

    logger = get_default_logger(output_folder)

    policy_optim = torch.optim.Adam(policy.parameters(), lr=config["policy_lr"])

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
            state_seq, _ = model.filter(data, data["pre_action"])
            state_features = torch.stack(
                [model.get_state_feature(state) for state in state_seq[:-1]]
            )

            if config["use_log_prob"]:
                policy_dist = policy(state_features)
                loss = -policy_dist.log_prob(data["pre_action"][1:]).sum(dim=-1).mean()
                mse = loss_fn(policy_dist.mode, data["pre_action"][1:])

                policy_optim.zero_grad()
                loss.backward()
                policy_optim.step()

                metric = {
                    "train/policy_loss": loss.item(),
                    "train/mse": mse.item(),
                }
            else:
                policy_action = policy(state_features).mode
                loss = loss_fn(data["pre_action"][1:], policy_action)

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
                state_seq, _ = model.filter(data, data["pre_action"])
                state_features = torch.stack(
                    [model.get_state_feature(state) for state in state_seq[:-1]]
                )

                if config["use_log_prob"]:
                    policy_dist = policy(state_features)
                    loss = (
                        -policy_dist.log_prob(data["pre_action"][1:]).sum(dim=-1).mean()
                    )
                    mse = loss_fn(policy_dist.mode, data["pre_action"][1:])

                    metric = {"val/policy_loss": loss.item(), "val/mse": mse.item()}
                else:
                    policy_action = policy(state_features).mode
                    loss = loss_fn(data["pre_action"][1:], policy_action)

                    metric = {
                        "val/policy_loss": loss.item(),
                    }

                val_metric_tracker.add(metric)

        metrics.update(val_metric_tracker.get())

        if config["eval_every_epoch"] and test_env.interactive:
            log.info("Evaluating the model ...")
            with torch.no_grad():
                actor = PolicyActor(model, policy, eval=True)
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
            torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))
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
        model.load_state_dict(torch.load(os.path.join(output_folder, "model.pt")))
        policy.load_state_dict(torch.load(os.path.join(output_folder, "policy.pt")))

        metrics = {}
        with torch.no_grad():
            actor = PolicyActor(model, policy, eval=True)
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
