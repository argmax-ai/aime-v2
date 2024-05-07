import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger("main")

from aimev2.data import get_sample_loader
from aimev2.env import env_classes
from aimev2.logger import get_default_logger
from aimev2.models.ssm import ssm_classes
from aimev2.runtimes import runtime_classes
from aimev2.utils import *


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="model-only")
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
    dataset_folder = os.path.join(DATA_PATH, config["embodiment_dataset_name"])

    env_config = config["env"]
    env_class_name = env_config["class"]
    try:
        env = env_classes[env_class_name](**env_config)
        env.enable_render(need_render(config["environment_setup"]))
    except Exception as e:
        log.info(f"The environment is not instanceable due to {e}.")
        env = None

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(
        config,
        sensor_layout,
        env.observation_space,
        predict_reward=config["use_reward"],
        use_probe=config["use_probe"],
    )
    selected_keys = get_seleted_keys_from_world_model_config(world_model_config)
    world_model_name = world_model_config.pop("name")
    all_image_sensors, used_image_sensors = get_image_sensors(
        world_model_config, sensor_layout
    )
    if config["action_free"]:
        world_model_config["action_dim"] = 0

    dataset = make_static_dataset(
        dataset_folder,
        config["horizon"],
        overlap=True,
        max_capacity=config["max_num_trajectories"],
        selected_keys=selected_keys,
        **config["data"]["dataset"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    model = ssm_classes[world_model_name](**world_model_config)
    if config["pretrained_model_name"] is not None:
        model.load_state_dict(
            torch.load(
                os.path.join(MODEL_PATH, config["pretrained_model_name"], "model.pt"),
                map_location="cpu",
            ),
            strict=False,
        )
    model = model.to(device)

    logger = get_default_logger(output_folder)

    model_optim = model.get_optimizor(dict(world_model_config["optimizor"]))
    model_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])

    for e in range(config["epoch"]):
        log.info(f"Starting epcoh {e}")

        loader = get_sample_loader(
            dataset,
            config["batch_size"],
            config["batch_per_epoch"],
            **config["data"]["dataloader"],
        )

        log.info("Training Model ...")
        train_metric_tracker = AverageMeter()
        training_start_time = time.time()
        for data in tqdm(iter(loader), disable=runtime.disable_tqdm):
            data = data.to(device)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                action_seq = data["pre_action"]
                if config["action_free"]:
                    action_seq = torch.zeros(*action_seq.shape[:2], 0).to(action_seq)
                _, _, loss, metrics = model(data, action_seq)

            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )
            model_scaler.step(model_optim)
            model_scaler.update()
            metrics["model_grad_norm"] = grad_norm.item()

            train_metric_tracker.add(metrics)

        metrics = train_metric_tracker.get()
        log.info(f"Training last for {time.time() - training_start_time:.3f} s")

        with torch.no_grad():
            if len(used_image_sensors) > 0 or (
                env is not None and env.set_state_from_obs_support
            ):
                log.info("Generating prediction videos ...")
                metrics.update(
                    generate_prediction_videos(
                        model,
                        data,
                        env,
                        all_image_sensors,
                        used_image_sensors,
                        10,
                        6,
                        custom_action_seq=action_seq,
                    )
                )
            metrics.update(
                eval_prediction(model, data, 10, custom_action_seq=action_seq)
            )

        log.info("Saving the models ...")
        torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))

        if (
            config["checkpoint_period"] is not None
            and (e + 1) % config["checkpoint_period"] == 0
        ):
            torch.save(
                model.state_dict(), os.path.join(output_folder, f"model-e{e + 1}.pt")
            )

        logger(metrics, e)

    runtime.finish(output_folder)


if __name__ == "__main__":
    main()
