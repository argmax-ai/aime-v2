import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger("main")

from aimev2.actor import StackPolicyActor
from aimev2.data import MultiDataset, SequenceDataset, get_epoch_loader
from aimev2.env import SaveTrajectories, TerminalSummaryWrapper, env_classes
from aimev2.logger import get_default_logger
from aimev2.models.base import MLP, ConcatMerge, FlareMerge, MultimodalEncoder
from aimev2.runtimes import runtime_classes
from aimev2.utils import *


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="bco-alpha")
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
    embodiment_dataset_folder = os.path.join(
        DATA_PATH, config["embodiment_dataset_name"]
    )
    demonstration_dataset_folder = os.path.join(
        DATA_PATH, config["demonstration_dataset_name"]
    )
    eval_folder = os.path.join(output_folder, "eval_trajectories")

    log.info("Creating environment ...")
    env_config = dict(config["env"])
    env_config["seed"] = config["seed"]
    env_class_name = env_config.pop("class")
    render = env_config["render"] or need_render(config["environment_setup"])
    env = env_classes[env_class_name](**env_config)
    env.enable_render(render)
    env = TerminalSummaryWrapper(env)
    env_config["seed"] = config["seed"] * 2
    test_env = env_classes[env_class_name](**env_config)
    test_env.enable_render(render)
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
    embodiment_dataset = make_static_dataset(
        embodiment_dataset_folder,
        stack + 1,
        overlap=True,
        selected_keys=selected_keys,
        **config["data"]["dataset"],
    )
    demonstration_dataset = SequenceDataset(
        demonstration_dataset_folder,
        stack + 1,
        overlap=True,
        selected_keys=selected_keys,
        max_capacity=config["num_expert_trajectories"],
        **config["data"]["dataset"],
    )
    eval_dataset = SequenceDataset(eval_folder, stack + 1, overlap=False)

    log.info("Creating models ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"using device {device}")
    idm_encoder = MultimodalEncoder(multimodal_encoder_config)
    idm_encoder = idm_encoder.to(device)
    idm_merger = (
        FlareMerge(idm_encoder.output_dim, stack + 1)
        if config["merger_type"] == "flare"
        else ConcatMerge(idm_encoder.output_dim, stack + 1)
    )
    idm_merger = idm_merger.to(device)
    idm_config = config["idm"]
    idm = MLP(
        idm_merger.output_dim,
        sensor_shapes["pre_action"],
        output_activation="tanh",
        **idm_config,
    )
    idm = idm.to(device)

    # not sure whether we should share the encoder weights
    policy_encoder = MultimodalEncoder(multimodal_encoder_config)
    policy_encoder = policy_encoder.to(device)
    policy_merger = (
        FlareMerge(idm_encoder.output_dim, stack)
        if config["merger_type"] == "flare" and stack > 1
        else ConcatMerge(idm_encoder.output_dim, stack)
    )
    policy_merger = policy_merger.to(device)
    policy_config = config["policy"]
    policy = MLP(
        policy_merger.output_dim,
        sensor_shapes["pre_action"],
        output_activation="tanh",
        **policy_config,
    )
    policy = policy.to(device)

    loss_fn = torch.nn.MSELoss()

    logger = get_default_logger(output_folder)

    idm_optim = torch.optim.Adam(
        [*idm.parameters(), *idm_merger.parameters(), *idm_encoder.parameters()],
        lr=config["idm_lr"],
    )
    policy_optim = torch.optim.Adam(
        [
            *policy.parameters(),
            *policy_merger.parameters(),
            *policy_encoder.parameters(),
        ],
        lr=config["policy_lr"],
    )

    def train_idm(
        embodiment_dataset_train,
        embodiment_dataset_val,
        e_shift=0,
        disable_patience=False,
    ):
        train_loader = get_epoch_loader(
            embodiment_dataset_train,
            config["batch_size"],
            shuffle=True,
            **config["data"]["dataloader"],
        )
        val_loader = get_epoch_loader(
            embodiment_dataset_val,
            config["batch_size"],
            shuffle=False,
            **config["data"]["dataloader"],
        )
        e = 0
        s = 0
        best_val_loss = float("inf")
        convergence_count = 0
        while True:
            log.info(f"Starting epcoh {e}")
            metrics = {}
            train_metric_tracker = AverageMeter()
            for data in tqdm(iter(train_loader), disable=runtime.disable_tqdm):
                data = data.to(device)
                emb = idm_encoder(data)
                emb = idm_merger(emb)
                predict_action = idm(emb)
                loss = loss_fn(predict_action, data[-1]["pre_action"])

                idm_optim.zero_grad()
                loss.backward()
                idm_optim.step()
                s += 1

                train_metric_tracker.add({"train/idm_loss": loss.item()})

            metrics.update(train_metric_tracker.get())

            val_metric_tracker = AverageMeter()
            with torch.no_grad():
                for data in tqdm(iter(val_loader), disable=runtime.disable_tqdm):
                    data = data.to(device)
                    emb = idm_encoder(data)
                    emb = idm_merger(emb)
                    predict_action = idm(emb)
                    loss = loss_fn(predict_action, data[-1]["pre_action"])

                    val_metric_tracker.add({"val/idm_loss": loss.item()})

            metrics.update(val_metric_tracker.get())

            logger(metrics, e + e_shift)

            e += 1

            if metrics["val/idm_loss"] < best_val_loss:
                best_val_loss = metrics["val/idm_loss"]
                torch.save(
                    idm_encoder.state_dict(),
                    os.path.join(output_folder, "idm_encoder.pt"),
                )
                torch.save(
                    idm_merger.state_dict(),
                    os.path.join(output_folder, "idm_merger.pt"),
                )
                torch.save(idm.state_dict(), os.path.join(output_folder, "idm.pt"))
                convergence_count = 0
            else:
                convergence_count += 1
                if disable_patience or (
                    convergence_count >= config["patience"]
                    and e >= config["min_idm_epoch"]
                    and s >= config["min_idm_steps"]
                ):
                    break

        log.info(f"IDM training finished in {e} epoches!")
        return e + e_shift

    def train_policy(
        demonstration_dataset_train,
        demonstration_dataset_val,
        e_shift=0,
        disable_patience=False,
    ):
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
        s = 0
        best_val_loss = float("inf")
        convergence_count = 0
        while True:
            log.info(f"Starting epcoh {e}")

            metrics = {}
            train_metric_tracker = AverageMeter()
            for data in tqdm(iter(train_loader), disable=runtime.disable_tqdm):
                data = data.to(device)
                emb_idm = idm_encoder(data)
                emb_idm = idm_merger(emb_idm)
                idm_action = idm(emb_idm)
                emb_policy = policy_encoder(data[:-1])
                emb_policy = policy_merger(emb_policy)
                policy_action = policy(emb_policy)
                loss = loss_fn(idm_action, policy_action)
                # NOTE: these two loss below is not possible to compute for the real setting,
                #       we only compute them for debugging.
                idm_to_real_loss = loss_fn(idm_action, data[-1]["pre_action"])
                policy_to_real_loss = loss_fn(policy_action, data[-1]["pre_action"])

                policy_optim.zero_grad()
                loss.backward()
                policy_optim.step()
                s += 1

                metric = {
                    "train/policy_loss": loss.item(),
                    "train/idm_to_real_loss": idm_to_real_loss.item(),
                    "train/policy_to_real_loss": policy_to_real_loss.item(),
                }

                train_metric_tracker.add(metric)

            metrics.update(train_metric_tracker.get())

            val_metric_tracker = AverageMeter()
            with torch.no_grad():
                for data in tqdm(iter(val_loader), disable=runtime.disable_tqdm):
                    data = data.to(device)
                    emb_idm = idm_encoder(data)
                    emb_idm = idm_merger(emb_idm)
                    idm_action = idm(emb_idm)
                    emb_policy = policy_encoder(data[:-1])
                    emb_policy = policy_merger(emb_policy)
                    policy_action = policy(emb_policy)
                    loss = loss_fn(idm_action, policy_action)
                    # NOTE: these two loss below is not possible to compute for the real setting,
                    #       we only compute them for debugging.
                    idm_to_real_loss = loss_fn(idm_action, data[-1]["pre_action"])
                    policy_to_real_loss = loss_fn(policy_action, data[-1]["pre_action"])

                    metric = {
                        "val/policy_loss": loss.item(),
                        "val/idm_to_real_loss": idm_to_real_loss.item(),
                        "val/policy_to_real_loss": policy_to_real_loss.item(),
                    }

                    val_metric_tracker.add(metric)

            metrics.update(val_metric_tracker.get())

            logger(metrics, e + e_shift)

            e += 1

            if metrics["val/policy_loss"] < best_val_loss:
                best_val_loss = metrics["val/policy_loss"]
                torch.save(
                    policy_encoder.state_dict(),
                    os.path.join(output_folder, "policy_encoder.pt"),
                )
                torch.save(
                    policy_merger.state_dict(),
                    os.path.join(output_folder, "policy_merger.pt"),
                )
                torch.save(
                    policy.state_dict(), os.path.join(output_folder, "policy.pt")
                )
                convergence_count = 0
            else:
                convergence_count += 1
                if disable_patience or (
                    convergence_count >= config["patience"]
                    and e >= config["min_policy_epoch"]
                    and s >= config["min_policy_steps"]
                ):
                    break

        log.info(f"Policy training finished in {e} epoches!")
        return e + e_shift

    def eval(epoch):
        # restore the best policy for a final test
        policy_encoder.load_state_dict(
            torch.load(os.path.join(output_folder, "policy_encoder.pt"))
        )
        policy_merger.load_state_dict(
            torch.load(os.path.join(output_folder, "policy_merger.pt"))
        )
        policy.load_state_dict(torch.load(os.path.join(output_folder, "policy.pt")))

        metrics = {}
        with torch.no_grad():
            actor = StackPolicyActor(policy_encoder, policy_merger, policy, stack)
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

        logger(metrics, epoch)

    log.info("Training IDM ...")
    embodiment_dataset_train, embodiment_dataset_val = (
        embodiment_dataset.split_train_and_val(config["train_validation_split_ratio"])
    )
    idm_epoch = train_idm(
        embodiment_dataset_train, embodiment_dataset_val, disable_patience=False
    )

    # restore the best idm
    idm_encoder.load_state_dict(
        torch.load(os.path.join(output_folder, "idm_encoder.pt"))
    )
    idm_merger.load_state_dict(torch.load(os.path.join(output_folder, "idm_merger.pt")))
    idm.load_state_dict(torch.load(os.path.join(output_folder, "idm.pt")))
    idm_encoder.requires_grad_(False)
    idm_merger.requires_grad_(False)
    idm.requires_grad_(False)

    # I think reusing the weight is a good thing to go
    if config["init_with_idm_encoder"]:
        policy_encoder.load_state_dict(idm_encoder.state_dict())

    log.info("Training Policy ...")
    demonstration_dataset_train, demonstration_dataset_val = (
        demonstration_dataset.split_train_and_val(
            config["train_validation_split_ratio"]
        )
    )
    policy_epoch = train_policy(
        demonstration_dataset_train, demonstration_dataset_val, disable_patience=False
    )
    idm_encoder.requires_grad_(True)
    idm_merger.requires_grad_(True)
    idm.requires_grad_(True)

    global_epoch = 0
    eval(global_epoch)

    while global_epoch < config["epoch"]:
        global_epoch += config["num_collecting_trajectories"]

        log.info("Collecting new environment interactions ...")
        replay_buffer_folder = os.path.join(
            output_folder, f"train_trajectories_{global_epoch}"
        )
        env = SaveTrajectories(env, replay_buffer_folder)
        with torch.no_grad():
            actor = StackPolicyActor(policy_encoder, policy_merger, policy, stack)
            [
                interact_with_environment(env, actor, image_sensors)
                for _ in range(config["num_collecting_trajectories"])
            ]
        env = env.env
        replay_dataset = SequenceDataset(
            replay_buffer_folder,
            stack + 1,
            overlap=True,
            selected_keys=selected_keys,
            **config["data"]["dataset"],
        )
        replay_dataset_train, replay_dataset_val = replay_dataset.split_train_and_val(
            config["train_validation_split_ratio"]
        )

        embodiment_dataset_train = MultiDataset(
            [embodiment_dataset_train, replay_dataset_train]
        )
        embodiment_dataset_val = MultiDataset(
            [embodiment_dataset_val, replay_dataset_val]
        )

        log.info("Training IDM ...")
        idm_epoch = train_idm(
            embodiment_dataset_train,
            embodiment_dataset_val,
            idm_epoch,
            disable_patience=True,
        )
        # restore the best idm
        idm_encoder.load_state_dict(
            torch.load(os.path.join(output_folder, "idm_encoder.pt"))
        )
        idm_merger.load_state_dict(
            torch.load(os.path.join(output_folder, "idm_merger.pt"))
        )
        idm.load_state_dict(torch.load(os.path.join(output_folder, "idm.pt")))
        idm_encoder.requires_grad_(False)
        idm_merger.requires_grad_(False)
        idm.requires_grad_(False)

        log.info("Training Policy ...")
        policy_epoch = train_policy(
            demonstration_dataset_train,
            demonstration_dataset_val,
            policy_epoch,
            disable_patience=True,
        )
        idm_encoder.requires_grad_(True)
        idm_merger.requires_grad_(True)
        idm.requires_grad_(True)

        eval(global_epoch)

    runtime.finish(output_folder)


if __name__ == "__main__":
    main()
