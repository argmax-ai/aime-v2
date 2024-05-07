import logging
import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger("main")

from aimev2.actor import GuassianNoiseActorWrapper, PolicyActor, RandomActor
from aimev2.data import SequenceDataset, get_sample_loader
from aimev2.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aimev2.logger import get_default_logger
from aimev2.models.policy import TanhGaussianPolicy
from aimev2.models.ssm import ssm_classes
from aimev2.models.value import VNetDict
from aimev2.runtimes import runtime_classes
from aimev2.utils import *


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="plan2explore")
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
    dataset_folder = os.path.join(output_folder, "train_trajectories")
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    env_config = dict(config["env"])
    env_config["seed"] = config["seed"]
    env_class_name = env_config.pop("class")
    env = env_classes[env_class_name](**env_config)
    env.enable_render(need_render(config["environment_setup"]))
    env = SaveTrajectories(env, dataset_folder)
    env = TerminalSummaryWrapper(env)
    env.action_space.seed(config["seed"])
    if env.multi_instancable:
        env_config = dict(env_config)
        env_config["seed"] *= 2
        test_env = env_classes[env_class_name](**env_config)
        test_env.enable_render(True)
        test_env = SaveTrajectories(test_env, eval_folder)
        test_env = TerminalSummaryWrapper(test_env)
    else:
        # NOTE: This is creating a bug that the evalutation trajectories will also go into the reply buffer.
        test_env = SaveTrajectories(env, eval_folder)

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(
        config, sensor_layout, env.observation_space, predict_reward=False
    )
    selected_keys = get_seleted_keys_from_world_model_config(world_model_config)
    world_model_name = world_model_config.pop("name")
    image_sensors, used_image_sensors = get_image_sensors(
        world_model_config, sensor_layout
    )

    # collect initial dataset
    for _ in range(config["prefill"]):
        actor = RandomActor(env.action_space)
        interact_with_environment(env, actor, [])

    dataset = SequenceDataset(
        dataset_folder,
        config["horizon"],
        overlap=True,
        selected_keys=selected_keys,
        **config["data"]["dataset"],
    )
    eval_dataset = SequenceDataset(eval_folder, config["horizon"], overlap=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    model = ssm_classes[world_model_name](**world_model_config)
    if config["pretrained_model_name"] is not None:
        pretrained_paramters = torch.load(
            os.path.join(MODEL_PATH, config["pretrained_model_name"], "model.pt"),
            map_location="cpu",
        )
        model.load_state_dict(pretrained_paramters, strict=False)
        if config["freeze_pretrained_parameters"]:
            for name, parameter in model.named_parameters():
                if name in pretrained_paramters.keys():
                    parameter.requires_grad_(False)
    model = model.to(device)

    policy_config = config["policy"]
    policy = TanhGaussianPolicy(
        model.state_feature_dim, world_model_config["action_dim"], **policy_config
    )
    if config["pretrained_model_name"] is not None and config["load_policy"]:
        policy.load_state_dict(
            torch.load(
                os.path.join(MODEL_PATH, config["pretrained_model_name"], "policy.pt"),
                map_location="cpu",
            ),
            strict=False,
        )
    policy = policy.to(device)

    vnet_config = config["vnet"]
    reward_keys = ["intrinsic_reward"]
    vnet = VNetDict(model.state_feature_dim, reward_keys, **vnet_config)
    if config["pretrained_model_name"] is not None and config["load_vnet"]:
        vnet.load_state_dict(
            torch.load(
                os.path.join(MODEL_PATH, config["pretrained_model_name"], "vnet.pt"),
                map_location="cpu",
            ),
            strict=False,
        )
    vnet = vnet.to(device)

    logger = get_default_logger(output_folder)

    model_optim = model.get_optimizor(dict(world_model_config["optimizor"]))
    model_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])
    policy_optim = torch.optim.Adam(policy.parameters(), lr=config["policy_lr"])
    policy_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])
    vnet_optim = torch.optim.Adam(vnet.parameters(), lr=config["vnet_lr"])
    vnet_scaler = torch.cuda.amp.GradScaler(enabled=config["use_fp16"])

    if config["pretraining_iterations"] > 0:
        log.info(
            f'pretrain the model for {config["pretraining_iterations"]} iterations.'
        )
        loader = get_sample_loader(
            dataset,
            config["batch_size"],
            config["pretraining_iterations"],
            **config["data"]["dataloader"],
        )
        for data in tqdm(iter(loader), disable=runtime.disable_tqdm):
            data = data.to(device)
            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                _, state_seq, loss, metrics = model(data, data["pre_action"])

            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )
            model_scaler.step(model_optim)
            model_scaler.update()

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
                _, state_seq, loss, metrics = model(data, data["pre_action"])

            metrics = {f"model/{k}": v for k, v in metrics.items()}
            model_optim.zero_grad(set_to_none=True)
            model_scaler.scale(loss).backward()
            model_scaler.unscale_(model_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                model.parameters(), config["grad_clip"]
            )
            model_scaler.step(model_optim)
            model_scaler.update()
            metrics["model/model_grad_norm"] = grad_norm.item()

            # rollout for longer
            states = model.flatten_states(state_seq)
            states.vmap_(lambda v: v.detach())

            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                state_seq, _, outputs = model.rollout_with_policy(
                    states,
                    policy,
                    config["imagine_horizon"],
                    names=["reward", "is_terminal"],
                    state_detach=True,
                    action_sample=True,
                )

                state_features = torch.stack(
                    [model.get_state_feature(state) for state in state_seq]
                )
                target_value_dict = vnet.compute_target(state_features)

                policy_loss = 0
                target_return_dict = {}
                discount = config["gamma"] * (1 - outputs["is_terminal"])
                cum_discount = torch.cumprod(
                    torch.cat([torch.ones_like(discount[:1]), discount[:-1]], dim=0),
                    dim=0,
                )

                for reward_key in reward_keys:
                    reward = outputs[reward_key]
                    value = target_value_dict[reward_key]

                    target_return_dict[reward_key] = lambda_return(
                        reward[:-1],
                        value[:-1],
                        discount[:-1],
                        value[-1],
                        config["lambda"],
                    )
                    if env.action_type == "continuous":
                        _policy_loss = -torch.mean(
                            cum_discount[:-2] * target_return_dict[reward_key][1:]
                        )
                    elif env.action_type == "discrete":
                        advantage = (
                            target_return_dict[reward_key][1:] - value[:-2]
                        ).detach()
                        _policy_loss = -torch.mean(
                            cum_discount[:-2] * outputs["action_logp"][:-1] * advantage
                        )
                    policy_loss = policy_loss + _policy_loss
                    metrics[f"policy/policy_loss_{reward_key}"] = _policy_loss.item()

                metrics["policy/policy_loss"] = policy_loss.item()
                policy_entropy_loss = -config["policy_entropy_scale"] * torch.mean(
                    outputs["action_entropy"].sum(dim=-1)
                )
                metrics["policy/policy_entropy_loss"] = policy_entropy_loss.item()
                policy_loss = policy_loss + policy_entropy_loss

            policy_optim.zero_grad(set_to_none=True)
            policy_scaler.scale(policy_loss).backward()
            policy_scaler.unscale_(policy_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                policy.parameters(), config["grad_clip"]
            )
            policy_scaler.step(policy_optim)
            policy_scaler.update()
            metrics["policy/policy_grad_norm"] = grad_norm.item()

            with torch.autocast(
                device_type=device, dtype=torch.float16, enabled=config["use_fp16"]
            ):
                value_dict = vnet(state_features[:-1].detach())
                value_loss = 0
                for reward_key in reward_keys:
                    _value_loss = 0.5 * torch.mean(
                        (
                            target_return_dict[reward_key].detach()
                            - value_dict[reward_key]
                        )
                        ** 2
                        * cum_discount[:-1].detach()
                    )
                    value_loss = value_loss + _value_loss
                    metrics[f"value/value_{reward_key}"] = (
                        value_dict[reward_key].mean().item()
                    )
                    metrics[f"value/value_loss_{reward_key}"] = _value_loss.item()
                metrics["value/value_loss"] = value_loss.item()

            vnet_optim.zero_grad(set_to_none=True)
            vnet_scaler.scale(value_loss).backward()
            vnet_scaler.unscale_(vnet_optim)
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
                vnet.parameters(), config["grad_clip"]
            )
            vnet_scaler.step(vnet_optim)
            vnet_scaler.update()
            metrics["value/vnet_grad_norm"] = grad_norm.item()
            vnet.update_target()

            train_metric_tracker.add(metrics)

        metrics = train_metric_tracker.get()
        log.info(f"Training last for {time.time() - training_start_time:.3f} s")

        log.info("Collecting new data ...")
        with torch.no_grad():
            actor = PolicyActor(model, policy)
            actor = GuassianNoiseActorWrapper(
                actor, config["epsilon"], env.action_space
            )
            result = interact_with_environment(env, actor, image_sensors)
            metrics.update({f"train_{k}": v for k, v in result.items()})

        dataset.update()

        if e % 10 == 0:
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

            if len(used_image_sensors) > 0 or test_env.set_state_from_obs_support:
                log.info("Generating prediction videos ...")
                metrics.update(
                    generate_prediction_videos(
                        model, data, test_env, image_sensors, used_image_sensors, 10, 6
                    )
                )

            log.info("Saving the models ...")
            torch.save(model.state_dict(), os.path.join(output_folder, "model.pt"))
            torch.save(policy.state_dict(), os.path.join(output_folder, "policy.pt"))
            torch.save(vnet.state_dict(), os.path.join(output_folder, "vnet.pt"))

        logger(metrics, e)

    runtime.finish(output_folder)


if __name__ == "__main__":
    main()
