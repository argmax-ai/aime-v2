from copy import deepcopy
from typing import Optional

import numpy as np
import torch

from aimev2.data import ArrayDict


class RandomActor:
    """Actor that random samples from the action space"""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def __call__(self, obs):
        return self.action_space.sample()

    def reset(self):
        pass


class CEMActor:
    """MPC based on CEM optimizor"""

    def __init__(
        self, ssm, action_space, horizon=12, iteration=10, candidate=1000, elite=100
    ) -> None:
        """
        NOTE: default hyper-parameters are borrowed from PlaNet
        ssm          : a state space model
        action_space : the action space of the environment
        horizon      : the rollout length when planning
        iteration    : the number of iteration for the optimization
        candidate    : the number of action sequence we evaluate every iteration
        elite        : the number of top return action sequence we use to fit the prior
        """
        self.horizon = horizon
        self.iteration = iteration
        self.candidate = candidate
        self.elite = elite
        self.ssm = ssm
        self.action_space = action_space

    def reset(self):
        self.state = self.ssm.reset(1)
        self.model_parameter = list(self.ssm.parameters())[0]

    def __call__(self, obs):
        obs = ArrayDict(deepcopy(obs))
        obs.to_torch()
        obs.expand_dim_equal_()
        obs.to(self.model_parameter)
        obs.vmap_(lambda v: v.unsqueeze(dim=0))

        self.state, _ = self.ssm.posterior_step(obs, obs["pre_action"], self.state)

        gaussian_parameters = ArrayDict(
            mean=(self.action_space.high + self.action_space.low) / 2,
            stddev=(self.action_space.high - self.action_space.low) / 2,
        )
        action_dim = gaussian_parameters["mean"].shape[-1]

        gaussian_parameters.to_torch()
        gaussian_parameters.to(self.model_parameter)
        gaussian_parameters.vmap_(
            lambda v: torch.repeat_interleave(v.unsqueeze(dim=0), self.horizon, dim=0)
        )

        for _ in range(self.iteration):
            action = gaussian_parameters["mean"] + gaussian_parameters[
                "stddev"
            ] * torch.randn(self.candidate, self.horizon, action_dim).to(
                gaussian_parameters["mean"]
            )
            action = action.permute(1, 0, 2)
            action = torch.clamp(
                action,
                torch.tensor(self.action_space.low).to(action),
                torch.tensor(self.action_space.high).to(action),
            )
            state = deepcopy(self.state)
            state.vmap_(lambda v: torch.repeat_interleave(v, self.candidate, dim=0))

            states, outputs = self.ssm.generate(
                state, action, ["reward", "is_terminal"]
            )

            rewards = outputs["reward"]  # reward must be one of the output
            discount = 1 - outputs["is_terminal"]
            discount = torch.cumprod(discount, dim=0)
            rewards = (
                (rewards[1:] * discount[:-1]).sum(dim=0).squeeze()
            )  # sum over the rollout

            index = torch.argsort(-rewards)[: self.elite]

            action = action.permute(1, 0, 2)
            action = action[index]

            gaussian_parameters = ArrayDict(
                mean=torch.mean(action, dim=0), stddev=torch.std(action, dim=0)
            )

        gaussian_parameters.to_numpy()
        return gaussian_parameters["mean"][0]


class MixCEMActor:
    """
    MPC based on CEM optimizor, mixed with policy and value function as inspired by the TD-MPC paper.
    Reference: https://github.com/nicklashansen/tdmpc/blob/main/src/algorithm/tdmpc.py#L92
    """

    def __init__(
        self,
        ssm,
        policy,
        vnet,
        action_space,
        horizon=5,
        iteration=6,
        candidate=512,
        elite=64,
        policy_ratio=0.05,
        min_std=0.3,
        eval=True,
    ) -> None:
        """
        NOTE: default hyper-parameters are borrowed from TD-MPC code
        TODO: implement the softmax trick for elite selection and refit
        TODO: implement the momentum of planning
        ssm          : a state space model
        action_space : the action space of the environment
        horizon      : the rollout length when planning
        iteration    : the number of iteration for the optimization
        candidate    : the number of action sequence we evaluate every iteration
        elite        : the number of top return action sequence we use to fit the prior
        policy_ratio : the ratio of trajectories that are generated by the policy
        eval         : flag for evaluation mode
        """
        self.horizon = horizon
        self.iteration = iteration
        self.candidate = candidate
        self.elite = elite
        self.ssm = ssm
        self.policy = policy
        self.vnet = vnet
        self.action_space = action_space
        self.policy_ratio = policy_ratio
        self.num_policy_candidates = int(self.candidate * self.policy_ratio)
        self.num_plan_candidates = self.candidate - self.num_policy_candidates
        self.min_std = min_std
        self.eval_mode = eval

    def reset(self):
        self.state = self.ssm.reset(1)
        self.model_parameter = list(self.ssm.parameters())[0]
        self.action_space_low = torch.tensor(self.action_space.low).to(
            self.model_parameter
        )
        self.action_space_high = torch.tensor(self.action_space.high).to(
            self.model_parameter
        )
        self.last_plan = None

    def __call__(self, obs):
        obs = ArrayDict(deepcopy(obs))
        obs.to_torch()
        obs.expand_dim_equal_()
        obs.to(self.model_parameter)
        obs.vmap_(lambda v: v.unsqueeze(dim=0))

        self.state, _ = self.ssm.posterior_step(obs, obs["pre_action"], self.state)
        state = deepcopy(self.state)
        state.vmap_(
            lambda v: torch.repeat_interleave(v, self.num_policy_candidates, dim=0)
        )
        states_policy, action_policy, outputs_policy = self.ssm.rollout_with_policy(
            state,
            self.policy,
            self.horizon,
            names=["reward", "is_terminal"],
            state_detach=True,
            action_sample=True,
        )

        gaussian_parameters = ArrayDict(
            mean=(self.action_space_high + self.action_space_low) / 2,
            stddev=(self.action_space_high - self.action_space_low) / 2,
        )
        action_dim = gaussian_parameters["mean"].shape[-1]

        gaussian_parameters.vmap_(
            lambda v: torch.repeat_interleave(v.unsqueeze(dim=0), self.horizon, dim=0)
        )
        if self.last_plan is not None:
            gaussian_parameters["mean"][:-1] = self.last_plan["mean"][1:]

        for _ in range(self.iteration):
            action_plan = gaussian_parameters["mean"] + gaussian_parameters[
                "stddev"
            ] * torch.randn(self.num_plan_candidates, self.horizon, action_dim).to(
                gaussian_parameters["mean"]
            )
            action_plan = action_plan.permute(1, 0, 2)
            action_plan = torch.clamp(
                action_plan, self.action_space_low, self.action_space_high
            )

            state = deepcopy(self.state)
            state.vmap_(
                lambda v: torch.repeat_interleave(v, self.num_plan_candidates, dim=0)
            )
            states_plan, outputs_plan = self.ssm.generate(
                state, action_plan, ["reward", "is_terminal"]
            )

            states = ArrayDict.cat(
                [
                    ArrayDict.stack(states_plan, dim=0),
                    ArrayDict.stack(states_policy, dim=0),
                ],
                dim=1,
            )
            state_features = self.ssm.get_state_feature(states)
            actions = torch.cat([action_plan, action_policy], dim=1)

            rewards = torch.cat(
                [outputs_plan["reward"], outputs_policy["reward"]], dim=1
            )
            discount = 1 - torch.cat(
                [outputs_plan["is_terminal"], outputs_policy["is_terminal"]], dim=1
            )
            discount = torch.cumprod(discount, dim=0)
            rewards = (
                (rewards[1:] * discount[:-1]).sum(dim=0).squeeze()
            )  # sum over the rollout
            last_value = (
                self.vnet(state_features[-1])["total_reward"].squeeze()
                * discount[-1].squeeze()
            )
            rewards = rewards + last_value

            index = torch.argsort(-rewards)[: self.elite]

            actions = actions.permute(1, 0, 2)
            actions = actions[index]

            gaussian_parameters = ArrayDict(
                mean=torch.mean(actions, dim=0),
                stddev=torch.std(actions, dim=0),
            )
            gaussian_parameters["stddev"] = torch.clamp(
                gaussian_parameters["stddev"],
                torch.ones_like(gaussian_parameters["stddev"]) * self.min_std,
                (self.action_space_high - self.action_space_low) / 2,
            )

        self.last_plan = gaussian_parameters

        action = gaussian_parameters["mean"][0]
        if not self.eval_mode:
            action + gaussian_parameters["stddev"][0] * torch.randn_like(
                gaussian_parameters["stddev"][0]
            )

        return action.detach().cpu().numpy()


class PolicyActor:
    """Model-based policy for taking actions. Optionally, with additional conditions as input"""

    def __init__(
        self, ssm, policy, condition: Optional[torch.Tensor] = None, eval=True
    ) -> None:
        """
        ssm          : a state space model
        policy       : a policy take a hidden state and output the distribution of actions
        """
        self.ssm = ssm
        self.policy = policy
        self.condition = condition
        if self.condition is not None:
            self.condition = self.condition.unsqueeze(dim=0)
        self.eval = eval

    def reset(self):
        self.state = self.ssm.reset(1)
        self.model_parameter = list(self.ssm.parameters())[0]

    def __call__(self, obs):
        obs = ArrayDict(deepcopy(obs))
        obs.to_torch()
        obs.expand_dim_equal_()
        obs.to(self.model_parameter)
        obs.vmap_(lambda v: v.unsqueeze(dim=0))

        self.state, _ = self.ssm.posterior_step(obs, obs["pre_action"], self.state)
        state_feature = self.ssm.get_state_feature(self.state)
        action_dist = self.policy(state_feature, self.condition)
        action = action_dist.mode if self.eval else action_dist.sample()
        action = action.detach().cpu().numpy()[0]

        return action


class StackPolicyActor:
    """Actor for the BCO policy, who needs a stack of observation to operate"""

    def __init__(self, encoder, merger, policy, stack: int) -> None:
        self.encoder = encoder
        self.merger = merger
        self.policy = policy
        self.stack = stack
        self.embs = []

    def reset(self):
        self.embs = []
        self.model_parameter = list(self.policy.parameters())[0]

    def __call__(self, obs):
        obs = ArrayDict(deepcopy(obs))
        obs.to_torch()
        obs.expand_dim_equal_()
        obs.to(self.model_parameter)
        obs.vmap_(lambda v: v.unsqueeze(dim=0))
        emb = self.encoder(obs)

        if len(self.embs) == 0:
            for _ in range(self.stack):
                self.embs.append(emb)
        else:
            self.embs.pop(0)
            self.embs.append(emb)

        emb = torch.stack(self.embs)
        emb = self.merger(emb)

        action = self.policy(emb)
        action = action.detach().cpu().numpy()[0]

        return action


class GuassianNoiseActorWrapper:
    def __init__(self, actor, noise_level, action_space) -> None:
        self._actor = actor
        self.noise_level = noise_level
        self.action_space = action_space

    def reset(self):
        return self._actor.reset()

    def __call__(self, obs):
        action = self._actor(obs)
        action = action + self.noise_level * np.random.randn(*action.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return action
