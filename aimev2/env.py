import logging
import os
import re
import time

import gym
import gymnasium
import numpy as np
import torch

log = logging.getLogger("env")

from aimev2.data import ArrayDict
from aimev2.utils import gymoutput2modelinput, savehdf5, savenpz, symlog


def cheetah_obs_to_state_fn(obs):
    x_pos = np.zeros(1)
    pos = obs["position"]
    vel = obs.get("velocity", np.zeros(9))
    return np.concatenate([x_pos, pos, vel])


obs_to_state_fns = {"cheetah": cheetah_obs_to_state_fn}


def get_cheetah_custom_reward_fn(flip: bool, backward: bool):
    max_run_velocity = 10
    max_rotation_velocity = 5

    def reward_fn(obs, env, parameter):
        if flip:
            speed = env.physics.named.data.subtree_angmom["torso"][1]
            max_speed = max_rotation_velocity
        else:  # run
            speed = env.physics.named.data.sensordata["torso_subtreelinvel"][0]
            max_speed = max_run_velocity

        sign = -1 if backward else 1
        return max(0, min(sign * speed / max_speed, 1))

    return reward_fn


def get_walker_custom_reward_fn(task):
    max_run_velocity = 8
    max_walker_velocity = 1
    max_jump_velocity = 2
    max_rotation_velocity = 5

    def reward_fn(obs, env, parameter):
        if "flip" in task:
            speed = env.physics.named.data.subtree_angmom["torso"][1]
            max_speed = max_rotation_velocity
            sign = -1 if "backward" in task else 1
            reward = max(0, min(sign * speed / max_speed, 1))
        elif "jump" in task:
            stand_reward = obs["reward"]
            jump_speed = env.physics.named.data.sensordata["torso_subtreelinvel"][2]
            max_speed = max_jump_velocity
            reward = stand_reward * max(0, min(jump_speed / max_speed, 1))
        elif "move" in task:
            stand_reward = obs["reward"]
            move_speed = env.physics.named.data.sensordata["torso_subtreelinvel"][0]
            target_speed = parameter * 2 * max_run_velocity - max_run_velocity
            move_reward = max(
                1 - np.abs(move_speed - target_speed) / max_run_velocity, 0
            )
            reward = stand_reward * (5 * move_reward + 1) / 6
        else:  # walk and run
            stand_reward = obs["reward"]
            max_speed = max_run_velocity if "run" in task else max_walker_velocity
            move_speed = env.physics.named.data.sensordata["torso_subtreelinvel"][0]
            sign = -1 if "backward" in task else 1
            reward = (
                stand_reward
                * (5 * max(0, min(sign * move_speed / max_speed, 1)) + 1)
                / 6
            )

        return reward

    return reward_fn


def get_manipulator_dense_reward_fn():
    distance_threshold = (
        0.1  # within this range the hand and object is considered close
    )

    def reward_fn(obs, env, parameter):
        hand_pos = obs["hand_pos"][:2]
        object_pos = obs["object_pos"][:2]
        target_pos = obs["target_pos"][:2]

        object2target_distance = np.linalg.norm(object_pos - target_pos)
        object2target_reward = 1 / (1 + 25 * object2target_distance)
        hand2object_distance = np.linalg.norm(object_pos - hand_pos)
        hand2object_reward = 1 / (
            1 + 25 * max(hand2object_distance - distance_threshold, 0)
        )
        reward = (hand2object_reward + object2target_reward) / 2

        return reward

    return reward_fn


def get_walker_color_move_reset_fn(mode="discrete"):
    def reset_fn(env):
        if mode == "discrete":
            color = np.random.choice([0, 0.4375, 0.5, 0.5625, 1.0])
        else:  # continuous
            color = np.random.rand()
        env.physics.named.model.mat_rgba["self"][:-1] = color
        return color

    return reset_fn


class DMC(gym.Env):
    interactive = True
    multi_instancable = True
    action_type = "continuous"

    """gym environment for dm_control, adapted from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/envs.py"""

    def __init__(
        self,
        name,
        embodiment=None,
        task=None,
        action_repeat=1,
        size=(64, 64),
        camera=None,
        render=True,
        seed=None,
        *args,
        **kwargs,
    ):
        _embodiment, _task = name.split("-", 1)
        embodiment = embodiment or _embodiment
        task = task or _task
        self._reward_fn = None
        self._reset_fn = None
        self._additional_parameters = None
        self._obs_to_state_fn = obs_to_state_fns.get(embodiment, None)
        if embodiment == "manip":
            from dm_control import manipulation

            # manipulation.ALL can list all the tasks
            self._env = manipulation.load(task + "_vision", seed=seed)
        elif embodiment == "locom":
            from dm_control.locomotion.examples import basic_rodent_2020

            self._env = getattr(basic_rodent_2020, task)(np.random.RandomState(seed))
        else:
            from dm_control import suite

            if embodiment == "cheetah":
                self._env = suite.load("cheetah", "run", task_kwargs={"random": seed})
                if task == "run":
                    self._reward_fn = None
                elif task == "runbackward":
                    self._reward_fn = get_cheetah_custom_reward_fn(False, True)
                elif task == "flip":
                    self._reward_fn = get_cheetah_custom_reward_fn(True, False)
                elif task == "flipbackward":
                    self._reward_fn = get_cheetah_custom_reward_fn(True, True)
                else:
                    raise NotImplementedError(
                        f"Task {task} is not defined for cheetah!"
                    )
            elif embodiment == "walker":
                if task in ["stand", "walk", "run"]:
                    self._env = suite.load(
                        embodiment, task, task_kwargs={"random": seed}
                    )
                    self._reward_fn = None
                elif task in [
                    "walkbackward",
                    "runbackward",
                    "jump",
                    "flip",
                    "flipbackward",
                ]:
                    self._env = suite.load(
                        "walker", "stand", task_kwargs={"random": seed}
                    )
                    self._reward_fn = get_walker_custom_reward_fn(task)
                elif task in ["color_move_discrete", "color_move_continuous"]:
                    self._env = suite.load(
                        "walker", "stand", task_kwargs={"random": seed}
                    )
                    self._reward_fn = get_walker_custom_reward_fn(task)
                    self._reset_fn = get_walker_color_move_reset_fn(task.split("_")[-1])
                else:
                    raise NotImplementedError(f"Task {task} is not defined for walker!")
            elif embodiment == "manipulator":
                if task.endswith("dense"):
                    task = task[:-6]
                    self._env = suite.load(
                        embodiment, task, task_kwargs={"random": seed}
                    )
                    self._reward_fn = get_manipulator_dense_reward_fn()
                else:
                    self._env = suite.load(
                        embodiment, task, task_kwargs={"random": seed}
                    )
                    self._reward_fn = None
            else:
                self._env = suite.load(embodiment, task, task_kwargs={"random": seed})
        self._action_repeat = action_repeat
        self._size = size
        self._render = render
        if camera in (-1, None):
            camera = {
                "quadruped-walk": 2,
                "quadruped-run": 2,
                "quadruped-escape": 2,
                "quadruped-fetch": 2,
                "locom_rodent-maze_forage": 1,
                "locom_rodent-two_touch": 1,
            }.get(name, 0)
        self._camera = camera
        self._ignored_keys = []
        for key, value in self._env.observation_spec().items():
            if value.shape == (0,):
                print(f"Ignoring empty observation key '{key}'.")
                self._ignored_keys.append(key)

        # setup observation and action space
        spec = self._env.action_spec()
        self.act_space = gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

        spaces = {
            "reward": gym.spaces.Box(0, self._action_repeat, (1,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=bool),
        }
        if self._render:
            spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        for key, value in self._env.observation_spec().items():
            if key in self._ignored_keys:
                continue
            if value.dtype == np.float64:
                spaces[key] = gym.spaces.Box(
                    -np.inf, np.inf, (int(np.prod(value.shape)),), np.float32
                )
            elif value.dtype == np.uint8:
                spaces[key] = gym.spaces.Box(
                    0, 255, (int(np.prod(value.shape)),), np.uint8
                )
            else:
                raise NotImplementedError(value.dtype)
        spaces["pre_action"] = self.act_space
        self.obs_space = spaces

    @property
    def fps(self):
        return 1 / self._env.control_timestep() / self._action_repeat

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    def enable_render(self, enable=True):
        self._render = enable
        if enable:
            self.obs_space["image"] = gym.spaces.Box(
                0, 255, self._size + (3,), dtype=np.uint8
            )
        else:
            if "image" in self.obs_space:
                self.obs_space.pop("image")

    def step(self, action):
        assert np.isfinite(action).all(), action
        action = np.clip(action, self.act_space.low, self.act_space.high)
        reward = 0.0
        for _ in range(self._action_repeat):
            time_step = self._env.step(action)
            _obs = time_step.observation.copy()
            _obs["reward"] = time_step.reward or 0.0
            if self._reward_fn is not None:
                _obs["reward"] = self._reward_fn(
                    _obs, self._env, self._additional_parameters
                )
            reward += _obs["reward"]
            if time_step.last():
                break
        assert time_step.discount in (0, 1)
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": time_step.last(),
            "is_terminal": time_step.discount == 0,
            "pre_action": action,
        }
        if self._render:
            obs.update(self.render())
        obs.update(
            {
                k: v.reshape((-1))
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        return obs

    def reset(self):
        if self._reset_fn is not None:
            self._additional_parameters = self._reset_fn(self._env)
        time_step = self._env.reset()
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        if self._render:
            obs.update(self.render())
        obs.update(
            {
                k: v.reshape((-1))
                for k, v in dict(time_step.observation).items()
                if k not in self._ignored_keys
            }
        )
        obs["pre_action"] = np.zeros(self.act_space.sample().shape)
        return obs

    def render(self, *args, **kwargs):
        size = kwargs.get("size", self._size)
        return {"image": self._env.physics.render(*size, camera_id=self._camera)}

    @property
    def set_state_from_obs_support(self):
        return self._obs_to_state_fn is not None

    def set_state_from_obs(self, obs):
        """
        Set the state of the robot to the one defined by an observation. Mainly for rendering.
        NOTE: This is not support for all environments! Please check `set_state_from_obs_support` before use.
        """
        assert (
            self.set_state_from_obs_support
        ), "`set_state_from_obs` is not supported for this environment!"
        state = self._obs_to_state_fn(obs)
        self._env.physics.set_state(state)
        self._env.physics.after_reset()


class OneHotDiscrete(gymnasium.spaces.Space):
    def __init__(self, n, dtype, seed=None):
        self.n = n
        self._shape = (n,)
        self.dtype = None if dtype is None else np.dtype(dtype)
        self._np_random = None
        if seed is not None:
            if isinstance(seed, np.random.Generator):
                self._np_random = seed
            else:
                self.seed(seed)

    @property
    def low(self):
        return np.zeros(self.n)

    @property
    def high(self):
        return np.ones(self.n)

    @property
    def is_np_flattenable(self) -> bool:
        """Checks whether this space can be flattened to a :class:`gymnasium.spaces.Box`."""
        raise True

    def sample(self, mask=None):
        s = np.zeros(self._shape, dtype=self.dtype)
        i = self.np_random.integers(self.n)
        s[i] = 1.0
        return s

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        return (
            x.shape[0] == self.n
            and np.sum(x == 0) + np.sum(x == 1) == self.n
            and np.sum(x == 1) <= 1
        )


class MetaWorld(gym.Env):
    interactive = True
    multi_instancable = True
    action_type = "continuous"
    set_state_from_obs_supported_envs = [
        "faucet-open",
        "faucet-close",  # chunk 11 790000
        "hammer",
        "stick-pull",
        "soccer",
        "shelf-place",  # chunk 12 -90000
        "reach",  # chunk 12 -130000
        "reach-wall",  # chunk 12 -170000
        "push",  # chunk 12 -210000
        "push-wall",  # chunk 12 -250000
        "push-back",  # chunk 12 -290000
        "plate-slide",  # chunk 12 -330000
        "plate-slide-side",  # chunk 12 -370000
        "plate-slide-back",  # chunk 12 -410000
        "plate-slide-back-side",  # chunk 12 -450000
        "pick-place",  # chunk 12 -500000
        "pick-place-wall",  # chunk 12 -530000,
        "pick-out-of-hole",  # chunk 12 -570000
        "peg-unplug-side",  # chunk 12 -610000
        "peg-insert-side",  # chunk 12 -650000
        "lever-pull",  # chunk 12 -690000
        "hand-insert",  # chunk 12 110000
        "handle-press-side",  # chunk 12 150000
        "handle-press",  # chunk 12 190000
        "handle-pull-side",  # chunk 12 230000
        "handle-pull",  # chunk 12 270000
        "assembly",  # chunk 11 30000 # need to check again
        "basketball",  # chunk 11 70000 search-based reset
        "bin-picking",  # chunk 11 110000
        "box-close",  # chunk 11 150000 search-based reset
        "button-press-topdown-wall",  # chunk 11 190000
        "button-press-topdown",  # chunk 11 230000
        "button-press-wall",  # chunk 11 270000
        "button-press",  # chunk 11 310000
        "coffee-button",  # chunk 11 350000
        "coffee-pull",  # chunk 11 390000
        "coffee-push",  # chunk 11 430000
        "dial-turn",  # chunk 11 470000
        "disassemble",  # chunk 11 510000
        "door-close",  # chunk 11 550000
        "door-lock",  # chunk 11 590000
        "door-open",  # chunk 11 630000
        "door-unlock",  # chunk 11 670000
        "drawer-close",  # chunk 11 710000
        "drawer-open",  # chunk 11 750000
        "stick-push",  # chunk 13 30000
        "sweep-into",  # chunk 13 70000
        "sweep",  # chunk 13 110000
        "window-close",  # chunk 13 150000
        "window-open",  # chunk 13 190000
    ]

    # the difficulties split is borrowed from Seo et al., Masked World Models for Visual Control, CoRL 2022
    difficulties = {
        "easy": [
            "button-press",
            "button-press-topdown",
            "button-press-topdown-wall",
            "button-press-wall",
            "coffee-button",
            "dial-turn",
            "door-close",
            "door-lock",
            "door-open",
            "door-unlock",
            "drawer-close",
            "drawer-open",
            "faucet-close",
            "faucet-open",
            "handle-press",
            "handle-press-side",
            "handle-pull",
            "handle-pull-side",
            "lever-pull",
            "plate-slide",
            "plate-slide-back",
            "plate-slide-back-side",
            "plate-slide-side",
            "reach",
            "reach-wall",
            "window-close",
            "window-open",
            "peg-unplug-side",
        ],
        "medium": [
            "basketball",
            "bin-picking",
            "box-close",
            "coffee-pull",
            "coffee-push",
            "hammer",
            "peg-insert-side",
            "push-wall",
            "soccer",
            "sweep",
            "sweep-into",
        ],
        "hard": [
            "assembly",
            "hand-insert",
            "pick-out-of-hole",
            "pick-place",
            "push",
            "push-back",
        ],
        "very hard": [
            "shelf-place",
            "disassemble",
            "stick-pull",
            "stick-push",
            "pick-place-wall",
        ],
    }

    """gym environment for [metaworld](https://github.com/Farama-Foundation/Metaworld)"""

    def __init__(
        self,
        name,
        action_repeat=2,
        size=(64, 64),
        camera="corner2",
        render=True,
        seed=None,
        max_steps=200,
        *args,
        **kwargs,
    ):
        import metaworld

        domain, task = name.split("-", 1)
        assert domain == "metaworld"
        self.task = task
        task = task + "-v2-goal-observable"
        assert (
            task in metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()
        ), f"metaworld does not support task {task}."
        self.task_id = list(
            metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()
        ).index(task)
        env_class = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task]
        self._env = env_class(seed=seed)
        self._env._freeze_rand_vec = False
        self._camera = camera
        self._render = render
        self._size = size
        self._action_repeat = action_repeat
        self._max_steps = max_steps

        self.act_space = self._env.action_space

        spaces = {
            "reward": gym.spaces.Box(0, self._action_repeat, (1,), dtype=np.float32),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "success": gym.spaces.Box(0, 1, (1,), dtype=bool),
            "task_id": gym.spaces.Discrete(50),
        }
        if self._render:
            spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        spaces["state"] = self._env.observation_space
        spaces["pre_action"] = self.act_space
        self.obs_space = spaces

        self.current_step = 0

    def enable_render(self, enable=True):
        self._render = enable
        if enable:
            self.obs_space["image"] = gym.spaces.Box(
                0, 255, self._size + (3,), dtype=np.uint8
            )
        else:
            if "image" in self.obs_space:
                self.obs_space.pop("image")

    @property
    def fps(self):
        return 1 / self._env.dt / self._action_repeat

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    def step(self, action):
        assert np.isfinite(action).all(), action
        action = np.clip(action, self.act_space.low, self.act_space.high)
        reward = 0.0
        success = False
        for _ in range(self._action_repeat):
            s, r, d, info = self._env.step(action)
            reward += r
            success = success or info["success"] == 1.0
            self.current_step += 1
            if self.current_step >= self._max_steps:
                break
        obs = {
            "reward": reward,
            "is_first": False,
            "is_last": self.current_step >= self._max_steps,
            "is_terminal": d,
            "success": success,
            "pre_action": action,
            "task_id": self.task_id,
        }
        if self._render:
            obs.update(self.render())
        obs["state"] = s
        return obs

    def reset(self, force_disable_render=False):
        # trick borrow from https://github.com/younggyoseo/MWM/blob/master/mwm/common/envs.py#L252-L253
        if self._camera == "corner2":
            self._env.model.cam_pos[2][:] = [0.75, 0.075, 0.7]
        s = self._env.reset()
        self.current_step = 0
        obs = {
            "reward": 0.0,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
            "success": False,
            "task_id": self.task_id,
        }
        if self._render and not force_disable_render:
            obs.update(self.render())
        obs["state"] = s
        obs["pre_action"] = np.zeros(self.act_space.sample().shape)
        return obs

    def render(self, *args, **kwargs):
        size = kwargs.get("size", self._size)
        return {
            "image": self._env.render(
                offscreen=True, camera_name=self._camera, resolution=size
            )
        }

    @property
    def set_state_from_obs_support(self):
        return self.task in self.set_state_from_obs_supported_envs

    def set_state_from_obs(self, obs):
        """
        Set the state of the robot to the one defined by an observation. Mainly for rendering.
        NOTE: This is not support for all environments! Please check `set_state_from_obs_support` before use.
        NOTE: Please only use this for a fresh environment before running any reset!
        """
        assert (
            self.set_state_from_obs_support
        ), "`set_state_from_obs` is not supported for this environment!"
        self._env.random_init = False
        if self.task == "faucet-open":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0.175, 0, 0.125]
            )

        elif self.task == "faucet-close":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [-0.175, 0, 0.125]
            )

        elif self.task == "hammer":
            self._env.init_config["hammer_init_pos"] = obs[4:7]

        elif self.task in ["stick-pull", "stick-push"]:
            # the first frame still have problem of incorrect goal position
            self._env.init_config["stick_init_pos"] = obs[4:7]
            _obs = self.reset(force_disable_render=True)
            self._env._target_pos = obs[-3:]
            _obs["state"][-3:] = obs[-3:]
            _obs.update(self.render())
            return _obs

        elif self.task == "soccer":
            self._env.goal = obs[-3:]
            self._env.obj_init_pos = obs[4:7]
            self._env.sim.model.body_pos[self._env.model.body_name2id("goal_whole")] = (
                self._env.goal
            )

        elif self.task in [
            "reach",
            "push",
            "pick-out-of-hole",
            "hand-insert",
            "bin-picking",
            "coffee-pull",
            "coffee-push",
            "sweep-into",
            "sweep",
        ]:
            # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[4:7]

        elif self.task == "reach-wall":
            # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.obj_init_pos = obs[4:7]

        elif self.task in ["push-wall", "push-back"]:
            # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[4:7]
            _obs = self.reset()
            self._env.obj_init_pos = obs[4:7]
            return _obs

        elif self.task in ["plate-slide", "plate-slide-side"]:
            # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[4:7]
            self._env.init_tcp = self._env.tcp_center

        elif self.task in ["pick-place", "pick-place-wall"]:
            # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[4:7]
            self._env.init_tcp = self._env.tcp_center
            self._env.init_left_pad = self._env.get_body_com("leftpad")
            self._env.init_right_pad = self._env.get_body_com("rightpad")
            _obs = self.reset()
            self._env.obj_init_pos = obs[4:7]
            return _obs

        elif self.task == "peg-unplug-side":
            # the first frame still have problem of incorrect goal position
            self._env.goal = (
                obs[-3:] - np.array([0.15, 0.0, 0.0]) - np.array([0.044, 0.0, 0.131])
            )

        elif self.task == "peg-insert-side":
            # the first frame still have problem of incorrect goal position
            # NOTE: maybe some bugs
            self._env.goal = obs[-3:] - np.array([0.03, 0.0, 0.13])
            self._env.obj_init_pos = obs[4:7]

        elif self.task == "lever-pull":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0.12, 0.0, 0.25 + self._env.LEVER_RADIUS]
            )

        elif self.task in ["button-press-topdown", "button-press-topdown-wall"]:
            # NOTE: need to be reset from the second observation! The simulator has one time step of delay for some reason.
            pos = obs[4:7] - np.array([0.0, 0.0, 0.193])
            # pos = obs[-3:] - np.array([0, 0, 0.1])
            pos[-1] = 0.115
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = pos

        elif self.task in ["button-press", "button-press-wall"]:
            # the button position has some error
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[4:7] - np.array(
                [0.0, -0.193, 0.0]
            )

        elif self.task == "coffee-button":
            self._env.init_config["obj_init_pos"] = (
                obs[-3:]
                - np.array([0.0, -0.22, 0.3])
                - np.array([0.0, self._env.max_dist, 0.0])
            )

        elif self.task == "dial-turn":
            # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array([0, 0.03, 0.03])

        elif self.task == "assembly":
            # the first frame still have problem of incorrect goal position
            # not always working
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[4:7] - np.array([0.13, 0, 0])

        elif self.task == "disassemble":
            # the first frame still have problem of incorrect goal position
            # not always working
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array([0, 0, 0.15])

        elif self.task == "door-close":
            # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0.2, -0.2, 0.0]
            )

        elif self.task == "door-open":
            # the first frame still have problem of incorrect goal position
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [-0.3, -0.45, 0.0]
            )

        elif self.task == "door-lock":
            # the first frame still have problem of incorrect goal position
            self._env.init_config["obj_init_pos"] = obs[4:7] - np.array(
                [0.09, -np.pi / 20, 0.08]
            )

        elif self.task == "door-unlock":
            # the first frame still have problem of incorrect goal position
            self._env.init_config["obj_init_pos"] = obs[4:7] - np.array(
                [-0.01, -np.pi / 20, -0.03]
            )

        elif self.task == "drawer-close":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0.0, -0.16, 0.09]
            )

        elif self.task == "drawer-open":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0.0, -0.16 - self._env.maxDist, 0.09]
            )

        elif self.task == "window-close":
            self._env.init_config["obj_init_pos"] = obs[-3:]

        elif self.task == "window-open":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array([0.2, 0.0, 0.0])

        elif self.task in ["plate-slide-back-side", "plate-slide-back"]:
            # # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]

        elif self.task == "shelf-place":
            # # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]
            self._env.init_config["obj_init_pos"] = obs[4:7] + np.array([0, 0, 0.3])

        elif self.task == "basketball":
            # # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:] - np.array([0, -0.083, 0.25])
            # final_pos = obs[4:7].copy()

            # search for the best reset pos, could use a better search algorithm
            budget = 1000

            # # random search
            # pos_candidates = np.stack([self._env._get_state_rand_vec() for _ in range(budget)], axis=0)
            # pos_candidates[:, 2] = 0.03
            # pos_results = []
            # for i in range(budget):
            #     self._env.init_config['obj_init_pos'] = pos_candidates[i][:3]
            #     pos_results.append(self._env.reset())
            # pos_results = np.stack(pos_results, axis=0)
            # dist = np.linalg.norm(pos_results[:, :8] - obs[:8], axis=1, ord=2)
            # index = np.argmin(dist)
            # final_pos = pos_candidates[index][:3]

            # zoopt search
            from zoopt import Dimension, Objective, Opt, Parameter

            def eval_fn(solution):
                pos = solution.get_x()
                self._env.init_config["obj_init_pos"] = pos
                reset_obs = self._env.reset()
                return np.linalg.norm(reset_obs[:8] - obs[:8], ord=2)

            dim = Dimension(3, [[-0.1, 0.1], [0.6, 0.7], [0.03, 0.03]], [True] * 3)
            obj = Objective(eval_fn, dim)
            solution = Opt.min(obj, Parameter(budget=budget))
            final_pos = np.array(solution.get_x())

            self._env.init_config["obj_init_pos"] = final_pos

        elif self.task == "box-close":
            # # the first frame still have problem of incorrect goal position
            self._env.goal = obs[-3:]

            # search for the best reset pos
            budget = 1000

            # # random search
            # pos_candidates = np.stack([self._env._get_state_rand_vec() for _ in range(budget)], axis=0)
            # pos_candidates[:, 2] = 0.02
            # pos_results = []
            # for i in range(budget):
            #     self._env.init_config['obj_init_pos'] = pos_candidates[i][:3]
            #     pos_results.append(self._env.reset())
            # pos_results = np.stack(pos_results, axis=0)
            # dist = np.linalg.norm(pos_results[:, 4:8] - obs[4:8], axis=1, ord=1)
            # index = np.argmin(dist)
            # final_pos = pos_candidates[index][:3]

            # zoopt search
            from zoopt import Dimension, Objective, Opt, Parameter

            def eval_fn(solution):
                pos = solution.get_x()
                self._env.init_config["obj_init_pos"] = pos
                reset_obs = self._env.reset()
                return np.linalg.norm(reset_obs[4:8] - obs[4:8], ord=2)

            dim = Dimension(3, [[-0.05, 0.05], [0.5, 0.55], [0.02, 0.02]], [True] * 3)
            obj = Objective(eval_fn, dim)
            solution = Opt.min(obj, Parameter(budget=budget))
            final_pos = np.array(solution.get_x())

            # self._env.init_config['obj_init_pos'] = obs[4:7] - np.array([0, 0, 0.075])
            self._env.init_config["obj_init_pos"] = final_pos

        elif self.task == "handle-press-side":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0.215, 0, 0.075]
            )

        elif self.task == "handle-press":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0, -0.215, 0.075]
            )

        elif self.task == "handle-pull-side":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0.215, 0, 0.172]
            )

        elif self.task == "handle-pull":
            self._env.init_config["obj_init_pos"] = obs[-3:] - np.array(
                [0, -0.215, 0.172]
            )

        return self.reset()


class SaveTrajectories(gym.Wrapper):
    def __init__(self, env: gym.Env, root: str, mode: str = "a", format: str = "hdf5"):
        super().__init__(env)
        self.root = root
        self.format = format
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        if mode == "w":
            self.trajectory_count = 0
        elif mode == "a":
            files = os.listdir(self.root)
            if len(files) == 0:
                self.trajectory_count = 0
            else:
                self.trajectory_count = 0
                for file_name in files:
                    match = re.search(r"^(\d+)", file_name)
                    self.trajectory_count = max(
                        self.trajectory_count, int(match.group(1))
                    )
                self.trajectory_count += 1
        self.trajectory_data = []

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.trajectory_data.append(ArrayDict(obs))
        return obs

    def step(self, action):
        obs = super().step(action)
        self.trajectory_data.append(ArrayDict(obs))
        if obs.get("is_last", False) or obs.get("is_terminal", False):
            if len(self.trajectory_data) > 0:
                data = ArrayDict.stack(self.trajectory_data, dim=0)
                data.expand_dim_equal_()
                if self.format == "npz":
                    savenpz(
                        data, os.path.join(self.root, f"{self.trajectory_count}.npz")
                    )
                elif self.format == "hdf5":
                    savehdf5(
                        data, os.path.join(self.root, f"{self.trajectory_count}.hdf5")
                    )
                self.trajectory_count += 1
                self.trajectory_data = []
        return obs


class TerminalSummaryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.rewards = {}
        self.success = False
        self.step_count = 0
        self.start_time = time.time()
        return super().reset(**kwargs)

    def step(self, action):
        obs = super().step(action)
        self.step_count += 1
        for k, v in obs.items():
            if "reward" in k:
                self.rewards[k] = self.rewards.get(k, 0) + obs[k]
            if k == "success":
                self.success = self.success or v
        if obs.get("is_last", False) or obs.get("is_terminal", False):
            message = (
                f"Trajectory finished in {self.step_count} steps ({time.time() - self.start_time:.3f} s), with "
                + " and ".join([f"total {k} {v}" for k, v in self.rewards.items()])
            )
            if "success" in obs.keys():
                message = (
                    message
                    + f', and with any success {self.success} and final success {obs["success"]}'
                )
            log.info(message)
        return obs


class ViperRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, likelihood_model, use_symlog=True):
        super().__init__(env)
        self.likelihood_model = likelihood_model
        self.use_symlog = use_symlog

    def reset(self, **kwargs):
        self.state = self.likelihood_model.reset(1)
        obs = super().reset(**kwargs)
        pytorch_obs = gymoutput2modelinput(obs)
        state_feature = self.likelihood_model.get_state_feature(self.state)
        pytorch_obs = pytorch_obs.to(state_feature)
        _, self.state = self.likelihood_model.likelihood_step(
            pytorch_obs, torch.zeros(1, 0).to(state_feature), self.state
        )
        obs["viper_reward"] = 0
        return obs

    def step(self, action):
        obs = super().step(action)
        pytorch_obs = gymoutput2modelinput(obs)
        state_feature = self.likelihood_model.get_state_feature(self.state)
        pytorch_obs = pytorch_obs.to(state_feature)
        likelihood, self.state = self.likelihood_model.likelihood_step(
            pytorch_obs, torch.zeros(1, 0).to(state_feature), self.state
        )
        if self.use_symlog:
            likelihood = symlog(torch.tensor(likelihood)).item()
        obs["viper_reward"] = likelihood
        return obs


class MaxStepsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steps: int):
        super().__init__(env)
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs = super().step(action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            obs["is_last"] = True
        return obs


env_classes = {
    "dmc": DMC,
    "metaworld": MetaWorld,
}
