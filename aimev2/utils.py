import json
import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger("utils")
from einops import rearrange
from h5py import File
from omegaconf import OmegaConf

from aimev2.data import ArrayDict


def setup_seed(seed=42):
    """Fix the common random source in deep learning programs"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    log.info(f"global seed is set to {seed}")


class AverageMeter:
    """Class to collect and average a sequence of metrics"""

    def __init__(self) -> None:
        self.storage = None

    def add(self, metrics):
        if self.storage is None:
            self.storage = {k: [v] for k, v in metrics.items()}
        else:
            for k in metrics.keys():
                self.storage[k].append(metrics[k])

    def get(
        self,
    ):
        if self.storage is None:
            return {}
        return {k: np.mean(v) for k, v in self.storage.items()}


class MovingAverage:
    def __init__(self, momentum=0.99) -> None:
        self.momentum = momentum
        self.value = None

    def update(self, value):
        if self.value is None:
            self.value = value
        else:
            self.value = self.momentum * self.value + (1 - self.momentum) * value

    def get(self):
        return self.value


def soft_update(local_model, target_model, tau):
    """
    update the parameters of the target model with the parameters of the local model.
    `tau` is value between 0 and 1, indicates how much parameters are copied. 1.0 means completely overwrite.
    """
    for target_param, local_param in zip(
        target_model.parameters(), local_model.parameters()
    ):
        target_param.data.copy_(
            tau * local_param.data + (1.0 - tau) * target_param.data
        )


def get_sensor_shapes(example_data):
    shapes = {}
    for k, v in example_data.items():
        shape = v.shape
        if len(shape) == 1 or len(shape) == 2:
            shapes[k] = shape[-1]
        elif len(shape) == 3 or len(shape) == 4:
            # TODO: Need to check this further!
            if isinstance(v, torch.Tensor):
                shapes[k] = shape[-2:]
            else:  # observation space
                shapes[k] = shape[-3:-1]
    return shapes


def get_inputs_outputs(
    sensor_layout,
    environment_setup,
    predict_emb=False,
    visual_encoders=None,
    as_extra_decoder=False,
):
    assert environment_setup in [
        "lpomdp",
        "pomdp",
        "mdp",
        "exp",
        "visual",
        "full",
        "real",
    ]
    if environment_setup == "mdp":
        input_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "tabular"
        ]
        output_sensors = input_sensors.copy()
        probe_sensors = []
    elif environment_setup == "pomdp" or environment_setup == "lpomdp":
        input_sensors = [
            k
            for k, v in sensor_layout.items()
            if v["modility"] == "tabular" and v["order"] == "first"
        ]
        output_sensors = input_sensors.copy()
        probe_sensors = [
            k
            for k, v in sensor_layout.items()
            if v["modility"] == "tabular" and v["order"] == "second"
        ]
    elif environment_setup == "exp":
        input_sensors = [
            k
            for k, v in sensor_layout.items()
            if v["modility"] == "tabular" and v["order"] == "first"
        ]
        output_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "tabular"
        ]
        probe_sensors = []
    elif environment_setup == "visual":
        input_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "visual"
        ]
        output_sensors = input_sensors.copy()
        probe_sensors = [
            k for k, v in sensor_layout.items() if v["modility"] == "tabular"
        ]
    elif environment_setup == "full":
        input_sensors = [k for k, v in sensor_layout.items()]
        output_sensors = input_sensors.copy()
        probe_sensors = []
    elif environment_setup == "real":
        input_sensors = [
            k
            for k, v in sensor_layout.items()
            if (v["modility"] == "visual") or (v["type"] == "internal")
        ]
        output_sensors = input_sensors.copy()
        probe_sensors = [k for k, v in sensor_layout.items() if k not in input_sensors]

    if visual_encoders is not None and not as_extra_decoder:
        for index, sensor_name in enumerate(input_sensors):
            if sensor_name in visual_encoders.keys():
                input_sensors[index] = f"{sensor_name}_emb"
        for index, sensor_name in enumerate(output_sensors):
            if sensor_name in visual_encoders.keys():
                output_sensors[index] = f"{sensor_name}_emb"
        for index, sensor_name in enumerate(probe_sensors):
            if sensor_name in visual_encoders.keys():
                probe_sensors[index] = f"{sensor_name}_emb"

    if as_extra_decoder:
        for sensor_name in visual_encoders.keys():
            output_sensors.append(f"{sensor_name}_emb")

    if predict_emb:
        probe_sensors += output_sensors
        output_sensors = ["emb"]

    return input_sensors, output_sensors, probe_sensors


def parse_world_model_config(
    config,
    sensor_layout,
    example_data,
    predict_terminal=True,
    predict_reward=False,
    detach_reward=False,
    visual_encoders=None,
    as_extra_decoder=False,
    use_probe=False,
):
    world_model_config = dict(config["world_model"])
    predict_emb = world_model_config.pop("predict_emb")
    input_sensors, output_sensors, probe_sensors = get_inputs_outputs(
        sensor_layout,
        config["environment_setup"],
        predict_emb,
        visual_encoders,
        as_extra_decoder,
    )
    sensor_shapes = get_sensor_shapes(example_data)
    sensor_layout = dict(sensor_layout)
    sensor_layout["emb"] = {"modility": "tabular"}
    sensor_shapes["emb"] = None
    if visual_encoders is not None:
        for sensor_name, model in visual_encoders.items():
            sensor_layout[f"{sensor_name}_emb"] = {"modility": "tabular"}
            sensor_shapes[f"{sensor_name}_emb"] = model.output_dim
    encoder_configs = world_model_config.pop("encoders")
    decoder_configs = world_model_config.pop("decoders")
    probe_configs = world_model_config.pop("probes")
    world_model_config["input_config"] = [
        (k, sensor_shapes[k], dict(encoder_configs[sensor_layout[k]["modility"]]))
        for k in input_sensors
    ]
    world_model_config["output_config"] = [
        (k, sensor_shapes[k], dict(decoder_configs[sensor_layout[k]["modility"]]))
        for k in output_sensors
    ]
    if use_probe:
        world_model_config["probe_config"] = []
        for k in probe_sensors:
            if k in sensor_shapes.keys():
                world_model_config["probe_config"].append(
                    (
                        k,
                        sensor_shapes[k],
                        dict(probe_configs[sensor_layout[k]["modility"]]),
                    )
                )
            else:
                log.warning(
                    f"Try to probe {k}, but it is not exist in the data, skip for now. Please make sure it is the intended behaviour!"
                )
    else:
        world_model_config["probe_config"] = []
    if predict_reward:
        if not detach_reward:
            world_model_config["output_config"] = world_model_config[
                "output_config"
            ] + [("reward", 1, dict(decoder_configs["tabular"]))]
        else:
            world_model_config["probe_config"] = world_model_config["probe_config"] + [
                ("reward", 1, dict(probe_configs["tabular"]))
            ]
    if predict_terminal:
        world_model_config["output_config"] = world_model_config["output_config"] + [
            ("is_terminal", 1, dict(decoder_configs["binary"]))
        ]
    world_model_config["action_dim"] = sensor_shapes["pre_action"]
    return world_model_config


def get_seleted_keys_from_world_model_config(world_model_config):
    seleted_keys = [c[0] for c in world_model_config["input_config"]]
    seleted_keys = seleted_keys + [c[0] for c in world_model_config["output_config"]]
    seleted_keys = seleted_keys + [c[0] for c in world_model_config["probe_config"]]
    seleted_keys = seleted_keys + ["pre_action", "is_first", "is_terminal", "is_last"]
    return list(set(seleted_keys))


def get_image_sensors(world_model_config, sensor_layout):
    image_sensors = [k for k, v in sensor_layout.items() if v["modility"] == "visual"]
    used_sensors = [config[0] for config in world_model_config["output_config"]]
    used_image_sensors = [
        image_sensor for image_sensor in image_sensors if image_sensor in used_sensors
    ]
    return image_sensors, used_image_sensors


def get_used_keys(world_model_config):
    keys = ["is_first", "is_last", "is_terminal"]
    keys = keys + [config[0] for config in world_model_config["input_config"]]
    keys = keys + [config[0] for config in world_model_config["output_config"]]
    keys = keys + [config[0] for config in world_model_config["probe_config"]]
    return list(set(keys))


def load_pretrained_model(model_root: str, strict: bool = False):
    """load the pretrained world model"""

    from aimev2 import __version__
    from aimev2.env import env_classes
    from aimev2.models.ssm import ssm_classes

    config = OmegaConf.load(os.path.join(model_root, "config.yaml"))

    # version check
    version_info = config.get("version", {"version": "0.1"})
    if not __version__ == version_info["version"]:
        log.warning(
            f"Trying to load a model trained by version {version_info['version']}, but the current version is {__version__}, maybe not compatiale!"
        )

    env_config = config["env"]
    env_class_name = env_config["class"]
    env = env_classes[env_class_name](
        env_config["name"],
        action_repeat=env_config["action_repeat"],
        seed=config["seed"],
        render=True,
    )

    sensor_layout = env_config["sensors"]
    world_model_config = parse_world_model_config(
        config,
        sensor_layout,
        env.observation_space,
        predict_terminal=config.get("use_terminal", True),
        predict_reward=config.get("use_reward", False),
        use_probe=config.get("use_probe", True),
    )
    action_free = config.get("action_free", False)
    if action_free:
        world_model_config["action_dim"] = 0
    world_model_name = world_model_config.pop("name")
    model = ssm_classes[world_model_name](**world_model_config)
    incompatible_keys = model.load_state_dict(
        torch.load(os.path.join(model_root, "model.pt"), map_location="cpu"),
        strict=strict,
    )

    if not strict:
        log.warning(
            f"Found imcompatible keys {incompatible_keys}. Please make sure that is the desired behaviour!"
        )

    return model


def need_render(environment_setup: str):
    """determine whether the render is a must during training"""
    return environment_setup in ["visual", "full", "real"]


def interact_with_environment(env, actor, image_sensors) -> float:
    """interact a environment with an actor for one trajectory"""
    obs = env.reset()
    has_success = "success" in obs.keys()
    actor.reset()
    reward = 0
    any_success = False
    while not obs.get("is_last", False) and not obs.get("is_terminal", False):
        for image_key in image_sensors:
            if image_key in obs.keys():
                obs[image_key] = rearrange(obs[image_key], "h w c -> c h w") / 255.0
        action = actor(obs)
        obs = env.step(action)
        reward += obs["reward"]
        if has_success:
            any_success = any_success or obs["success"]

    if has_success:
        result = {
            "reward": reward,
            "any_success": any_success,
            "success": obs["success"],
        }
    else:
        result = {"reward": reward}

    return result


def eval_actor_on_env(env, actor, image_sensors, num_test_trajectories=1, suffix=None):
    metrics = {}
    if num_test_trajectories == 1:
        result = interact_with_environment(env, actor, image_sensors)
        metrics = {f"eval_{k}": v for k, v in result.items()}
    else:
        results = [
            interact_with_environment(env, actor, image_sensors)
            for _ in range(num_test_trajectories)
        ]
        rewards = [r["reward"] for r in results]
        reward_key = "eval_reward" if suffix is None else f"eval_reward_{suffix}"
        metrics[f"{reward_key}_raw"] = rewards
        metrics[reward_key] = np.mean(rewards)
        metrics[f"{reward_key}_std"] = np.std(rewards)
        metrics[f"{reward_key}_max"] = np.max(rewards)
        metrics[f"{reward_key}_min"] = np.min(rewards)
        if "success" in results[0].keys():
            any_successes = [r["any_success"] for r in results]
            successes = [r["success"] for r in results]
            any_success_key = (
                "eval_any_success_rate"
                if suffix is None
                else f"eval_any_success_rate_{suffix}"
            )
            metrics[f"{any_success_key}_raw"] = any_successes
            metrics[any_success_key] = np.mean(any_successes)
            success_key = (
                "eval_success_rate" if suffix is None else f"eval_success_rate_{suffix}"
            )
            metrics[f"{success_key}_raw"] = successes
            metrics[success_key] = np.mean(successes)
    return metrics


@torch.no_grad()
def generate_prediction_videos(
    model,
    data,
    env,
    all_image_sensors,
    used_image_sensors,
    filter_step: int = 10,
    samples: int = 6,
    custom_action_seq=None,
):
    videos = {}
    data = data[:, :samples]
    data.vmap_(lambda x: x.contiguous())
    pre_action_seq = (
        data["pre_action"]
        if custom_action_seq is None
        else custom_action_seq[:, :samples]
    )
    predicted_obs_seq, _, _, _ = model(data, pre_action_seq, filter_step=filter_step)
    if len(used_image_sensors) == 0:
        # one must render the scene from other signals
        some_key = list(predicted_obs_seq.keys())[0]
        some_value = predicted_obs_seq[some_key][..., 0]
        t, b = predicted_obs_seq[some_key].shape[:2]
        predicted_obs_seq.to_numpy()
        image_obs_seq = []
        for i in range(t):
            _image_obs_seq = []
            for j in range(b):
                obs = predicted_obs_seq[i, j]
                env.set_state_from_obs(obs)
                _image_obs_seq.append(ArrayDict(env.render()))
            image_obs_seq.append(ArrayDict.stack(_image_obs_seq, dim=0))
        image_obs_seq = ArrayDict.stack(image_obs_seq, dim=0)
        image_obs_seq.to_torch()
        for image_key in image_obs_seq.keys():
            image_obs_seq[image_key] = (
                rearrange(image_obs_seq[image_key], "t b h w c -> t b c h w") / 255.0
            )
        predicted_obs_seq.to_torch()
        predicted_obs_seq.update(image_obs_seq)
        predicted_obs_seq.to(some_value)

        data.to_numpy()
        image_obs_seq = []
        for i in range(t):
            _image_obs_seq = []
            for j in range(b):
                obs = data[i, j]
                env.set_state_from_obs(obs)
                _image_obs_seq.append(ArrayDict(env.render()))
            image_obs_seq.append(ArrayDict.stack(_image_obs_seq, dim=0))
        image_obs_seq = ArrayDict.stack(image_obs_seq, dim=0)
        image_obs_seq.to_torch()
        for image_key in image_obs_seq.keys():
            image_obs_seq[image_key] = (
                rearrange(image_obs_seq[image_key], "t b h w c -> t b c h w") / 255.0
            )
        data.to_torch()
        data.update(image_obs_seq)
        data.to(some_value)

    for image_key in all_image_sensors:
        if image_key not in predicted_obs_seq.keys():
            continue
        gt_video = data[image_key]
        pred_video = predicted_obs_seq[image_key]
        diff_video = (gt_video - pred_video) / 2 + 0.5
        log_video = torch.cat([gt_video, pred_video, diff_video], dim=1)
        log_video = rearrange(log_video, "t (m b) c h w -> t (m h) (b w) c", m=3) * 255
        videos[f"rollout_video_{image_key}"] = log_video

    return videos


@torch.no_grad()
def eval_prediction(
    model,
    data,
    filter_step: int = 10,
    custom_action_seq=None,
):
    metrics = {}
    pre_action_seq = (
        custom_action_seq if custom_action_seq is not None else data["pre_action"]
    )
    predicted_obs_seq, _, _, _ = model(data, pre_action_seq, filter_step=filter_step)

    for name in model.decoders.keys():
        metrics[f"prediction_{name}_mse"] = torch.nn.MSELoss()(
            predicted_obs_seq[name][filter_step:], data[name][filter_step:]
        ).item()

    return metrics


@torch.jit.script
def lambda_return(reward, value, discount, bootstrap, lambda_: float):
    """
    Modify from https://github.com/danijar/dreamer/blob/master/tools.py,
    Setting lambda=1 gives a discounted Monte Carlo return.
    Setting lambda=0 gives a fixed 1-step return.
    """
    next_values = torch.cat([value[1:], bootstrap[None]], dim=0)
    inputs = reward + discount * next_values * (1 - lambda_)
    returns = []
    curr_value = bootstrap
    for t in reversed(torch.arange(len(value))):
        curr_value = inputs[t] + lambda_ * discount[t] * curr_value
        returns.append(curr_value)
    returns = torch.stack(returns)
    returns = torch.flip(returns, dims=[0])
    return returns


def symlog(tensor):
    return torch.sign(tensor) * torch.log(torch.abs(tensor) + 1)


def symexp(tensor):
    return torch.sign(tensor) * (torch.exp(torch.abs(tensor)) - 1)


def get_dataset_format(root):
    files = os.listdir(root)
    return files[0].split(".")[-1]


def make_static_dataset(root, *args, **kwargs):
    """make a dataset from the root"""
    from aimev2.data import MultiFolderDataset, SequenceDataset

    inside = os.listdir(root)[0]
    if os.path.isdir(os.path.join(root, inside)):
        return MultiFolderDataset(root, *args, **kwargs)
    else:
        return SequenceDataset(root, *args, **kwargs)


def npz2hdf5(npz_file, hdf5_file, compression_config=dict(compression="lzf")):
    data = dict(np.load(npz_file))
    savehdf5(data, hdf5_file, compression_config)


def savenpz(data, npz_file):
    np.savez_compressed(npz_file, **data)


def savehdf5(data, hdf5_file, compression_config=dict(compression="lzf")):
    with File(hdf5_file, mode="w") as f:
        for k, v in data.items():
            # use the heuristic to decide the chunk size
            data_size = np.prod(v.shape[1:])
            chunk_size = 8192 // data_size
            chunk_size = max(chunk_size, 1)
            chunk_size = min(chunk_size, v.shape[0])
            chunks = (chunk_size, *v.shape[1:])
            # create the dataset
            # TODO: find the best compression for images
            f.create_dataset(k, v.shape, v.dtype, chunks=chunks, **compression_config)
            f[k][:] = v


def gymoutput2modelinput(data):
    """convert the output dict from gym interface to the format that can used by pytorch models"""
    data = ArrayDict(data)
    data = ArrayDict.stack(
        [data], dim=0
    )  # this line will add the batch axis and covert scale to ndarray
    data.expand_dim_equal_()
    data.to_torch()
    data.to_float_torch()
    for k, v in data.items():
        if len(v.shape) >= 3:  # image
            data[k] = rearrange(v, "... h w c -> ... c h w") / 255
    return data


def deepdown(path):
    while len(os.listdir(path)) == 1:
        new_path = os.path.join(path, list(os.listdir(path))[0])
        if os.path.isdir(new_path):
            path = new_path
        else:
            break
    return path


def load_jsonl(file_name):
    with open(file_name, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def merge_result(result, padding_mode="mean", smoothing_momentum=0.0):
    result = list(result.values())
    result = [smoothing(r, smoothing_momentum) for r in result]
    try:
        merged_result = np.stack(result)
    except:
        if padding_mode == "zero":
            max_length = max([len(r) for r in result])
            merged_result = np.zeros((len(result), max_length))
            for i, r in enumerate(result):
                merged_result[i][: len(r)] = np.array(r)
        elif padding_mode == "mean":
            result = sorted(result, key=lambda x: len(x), reverse=True)
            result = [np.array(r) for r in result]
            merged_result = result[0][None]
            for r in result[1:]:
                l = len(r)
                r = np.concatenate([r, np.mean(merged_result[:, l:], axis=0)], axis=-1)
                merged_result = np.concatenate([merged_result, r[None]], axis=0)
    return merged_result


def smoothing(result, momentum=0.0):
    value = result[0]
    new_result = [value]
    for i in range(1, len(result)):
        value = momentum * value + (1 - momentum) * result[i]
        new_result.append(value)
    return new_result


def save_gif(filename, video, fps=25):
    from moviepy import editor as mpy

    video = video.permute(0, 2, 3, 1).contiguous() * 255
    video = video.numpy()

    clip = mpy.ImageSequenceClip(list(video), fps=fps)

    try:  # newer version of moviepy use logger instead of progress_bar argument.
        clip.write_gif(filename, verbose=False, logger=None)
    except TypeError:
        try:  # older version of moviepy does not support progress_bar argument.
            clip.write_gif(filename, verbose=False, progress_bar=False)
        except TypeError:
            clip.write_gif(filename, verbose=False)


CONFIG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "configs")

OUTPUT_PATH = "logs"
DATA_PATH = "datasets"
MODEL_PATH = "pretrained-models"
