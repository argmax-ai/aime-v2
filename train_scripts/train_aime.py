import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger("main")

from aimev2.actor import PolicyActor
from aimev2.data import SequenceDataset, get_sample_loader
from aimev2.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aimev2.logger import get_default_logger
from aimev2.models.policy import TanhGaussianPolicy
from aimev2.models.ssm import ssm_classes
from aimev2.runtimes import runtime_classes
from aimev2.utils import *


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="aime")
def main(config: DictConfig):
    runtime = runtime_classes[config["runtime"]["name"]](config)

    setup_seed(config["seed"])

    log.info("Using the following config:\n" + OmegaConf.to_yaml(config))

    log_name = config["log_name"]
    output_folder = os.path.join(OUTPUT_PATH, log_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log.info(f"Log will be stored at {output_folder}")

    OmegaConf.save(config, os.path.join(output_folder, "config.yaml"))
    model_folder = os.path.join(MODEL_PATH, config["pretrained_model_name"])
    demonstration_dataset_folder = os.path.join(
        DATA_PATH, config["demonstration_dataset_name"]
    )
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    env_config = dict(config["env"])
    env_config["seed"] = config["seed"] * 2
    env_class_name = env_config.pop("class")
    render = env_config["render"] or need_render(config["environment_setup"])
    test_env = env_classes[env_class_name](**env_config)
    test_env.enable_render(render)
    test_env = SaveTrajectories(test_env, eval_folder)
    test_env = TerminalSummaryWrapper(test_env)

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(
        config,
        sensor_layout,
        test_env.observation_space,
        predict_terminal=config["use_terminal"],
        predict_reward=config["use_reward"],
        use_probe=config["use_probe"],
    )
    selected_keys = get_seleted_keys_from_world_model_config(world_model_config)
    world_model_name = world_model_config.pop("name")
    image_sensors, used_image_sensors = get_image_sensors(
        world_model_config, sensor_layout
    )

    demonstration_dataset = SequenceDataset(
        demonstration_dataset_folder,
        config["horizon"],
        overlap=True,
        max_capacity=config["num_expert_trajectories"],
        selected_keys=selected_keys,
        **config["data"]["dataset"],
    )
    log.info(
        f"Training on {len(demonstration_dataset.trajectories)} expert trajectories!"
    )
    eval_dataset = SequenceDataset(eval_folder, config["horizon"], overlap=False)

    if config["embodiment_dataset_name"] is not None:
        embodiment_dataset_folder = os.path.join(
            DATA_PATH, config["embodiment_dataset_name"]
        )
        embodiment_dataset = make_static_dataset(
            embodiment_dataset_folder,
            config["horizon"],
            overlap=True,
            selected_keys=selected_keys,
            **config["data"]["dataset"],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    # load the pretrained policy
    policy_config = config["policy"]
    policy_file = os.path.join(model_folder, "policy.pt")
    if os.path.exists(policy_file):
        policy = TanhGaussianPolicy(
            model.state_feature_dim, world_model_config["action_dim"], **policy_config
        )
        policy.load_state_dict(
            torch.load(os.path.join(model_folder, "policy.pt"), map_location="cpu")
        )
        policy = policy.to(device)

        # directly test this model and policy on the new task
        if test_env.interactive:
            log.info("Evaluating the pretrained model and policy ...")
            with torch.no_grad():
                actor = PolicyActor(model, policy)
                interact_with_environment(test_env, actor, image_sensors)
            eval_dataset.update()

    # reinitialize the policy to random policy
    if config["random_policy"]:
        policy = TanhGaussianPolicy(
            model.state_feature_dim, world_model_config["action_dim"], **policy_config
        )
        policy = policy.to(device)
        if test_env.interactive:
            log.info("Evaluating the pretrained model and random policy ...")
            with torch.no_grad():
                actor = PolicyActor(model, policy)
                interact_with_environment(test_env, actor, image_sensors)
            eval_dataset.update()

    if config["use_idm"]:
        idm = model.idm
        model.idm = None  # remove the idm from the model, so that it won't be count twice in optimizor.
        # idm.requires_grad_(True)
    else:
        idm = None

    logger = get_default_logger(output_folder)

    parameters = [*policy.parameters(), *model.parameters()]
    if idm is not None:
        parameters = parameters + [*idm.parameters()]
    policy_optim = torch.optim.Adam(parameters, lr=config["policy_lr"])
    policy_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])

    if config["embodiment_dataset_name"] is not None:
        model_optim = model.get_optimizor(dict(world_model_config["optimizor"]))
        model_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])

    for e in range(config["epoch"]):
        log.info(f"Starting epcoh {e}")

        metrics = {}

        if config["embodiment_dataset_name"] is not None:
            loader = get_sample_loader(
                embodiment_dataset,
                config["batch_size"],
                config["batch_per_epoch"],
                **config["data"]["dataloader"],
            )
            model.requires_grad_(True)

            log.info("Training Model with Embodiment dataset ...")
            train_metric_tracker = AverageMeter()
            training_start_time = time.time()
            for data in tqdm(iter(loader), disable=runtime.disable_tqdm):
                data = data.to(device)
                with torch.autocast(
                    device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
                ):
                    action_seq = data["pre_action"]
                    _, _, loss, metric = model(data, action_seq)

                model_optim.zero_grad(set_to_none=True)
                model_scaler.scale(loss).backward()
                model_scaler.unscale_(model_optim)
                grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), config["grad_clip"]
                )
                model_scaler.step(model_optim)
                model_scaler.update()
                metric["model_grad_norm"] = grad_norm.item()

                train_metric_tracker.add(metric)

            metrics.update(
                {f"model/{k}": v for k, v in train_metric_tracker.get().items()}
            )
            log.info(f"Training last for {time.time() - training_start_time:.3f} s")

            if config["freeze_model"]:
                log.info("freeze the model weights")
                model.requires_grad_(False)

        loader = get_sample_loader(
            demonstration_dataset,
            config["batch_size"],
            config["batch_per_epoch"],
            **config["data"]["dataloader"],
        )

        log.info("Training Policy with AIME ...")
        train_metric_tracker = AverageMeter()
        training_start_time = time.time()
        for data in tqdm(iter(loader), disable=runtime.disable_tqdm):
            data = data.to(device)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                _, _, action_seq, loss, metric = model.filter_with_policy(
                    data,
                    policy,
                    idm,
                    idm_mode=config["idm_mode"],
                    kl_only=config["kl_only"],
                )
                # you should not be able to compute this metric in the real setting, we compute here only for analysis
                metric["action_mse"] = model.metric_func(
                    data["pre_action"], action_seq
                ).item()

            policy_optim.zero_grad(set_to_none=True)
            policy_scaler.scale(loss).backward()
            policy_scaler.unscale_(policy_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                policy.parameters(), config["grad_clip"]
            )
            policy_scaler.step(policy_optim)
            policy_scaler.update()
            metric["policy_grad_norm"] = grad_norm.item()

            train_metric_tracker.add(metric)

        metrics.update(
            {f"policy/{k}": v for k, v in train_metric_tracker.get().items()}
        )
        log.info(f"Training last for {time.time() - training_start_time:.3f} s")

        if len(used_image_sensors) > 0 or test_env.set_state_from_obs_support:
            log.info("Generating prediction videos ...")
            metrics.update(
                generate_prediction_videos(
                    model,
                    data,
                    test_env,
                    image_sensors,
                    used_image_sensors,
                    None,
                    6,
                    action_seq,
                )
            )

        if e % config["test_period"] == 0 and test_env.interactive:
            log.info("Evaluating the model ...")
            with torch.no_grad():
                actor = PolicyActor(model, policy)
                metrics.update(
                    eval_actor_on_env(
                        test_env, actor, image_sensors, config["num_test_trajectories"]
                    )
                )
            if render:
                eval_dataset.update()
                for image_key in image_sensors:
                    metrics[f"eval_video_{image_key}"] = (
                        eval_dataset.get_trajectory(-1)[image_key]
                        .permute(0, 2, 3, 1)
                        .contiguous()
                        * 255
                    )

        logger(metrics, e)
        torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))
        torch.save(policy.state_dict(), os.path.join(output_folder, "policy.pt"))

        runtime.upload(e, output_folder)

    if test_env.interactive:
        log.info("Evaluating the final model ...")
        metrics = {}
        with torch.no_grad():
            actor = PolicyActor(model, policy)
            metrics.update(
                eval_actor_on_env(
                    test_env,
                    actor,
                    image_sensors,
                    config["final_num_test_trajectories"],
                )
            )
        if render:
            eval_dataset.update()
            for image_key in image_sensors:
                metrics[f"eval_video_{image_key}"] = (
                    eval_dataset.get_trajectory(-1)[image_key]
                    .permute(0, 2, 3, 1)
                    .contiguous()
                    * 255
                )
        logger(metrics, e + 1)
    torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))
    torch.save(policy.state_dict(), os.path.join(output_folder, "policy.pt"))

    runtime.finish(output_folder)


if __name__ == "__main__":
    main()
