from typing import Dict, Optional

import numpy as np
import torch
from einops import rearrange

from aimev2.data import ArrayDict
from aimev2.dist import MultipleOneHot, Normal, kl_divergence
from aimev2.utils import soft_update

from .base import MIN_STD, MLP, EnsembleMLP, LNGRUCell, decoder_classes, encoder_classes
from .policy import TanhGaussianPolicy


class SSM(torch.nn.Module):
    """
    State-Space Model BaseClass.
    NOTE: in some literature, this type of model is also called sequential auto-encoder or stochastic rnn.
    NOTE: in the current version, SSM also contain encoders, decoders and probes for the
          sack of simplicity for model training. But this may not be the optimal modularization.
    """

    def __init__(
        self,
        input_config,
        output_config,
        action_dim,
        state_dim,
        probe_config=None,
        intrinsic_reward_config=None,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        kl_scale=1.0,
        free_nats=0.0,
        kl_rebalance=None,
        nll_reweight="dim_wise",
        idm_mode="none",
        momentum_parameter=0.01,
        state_distribution="continuous",
        min_std=None,
        *args,
        **kwargs,
    ) -> None:
        """
        input_config            : a list of tuple(name, dim, encoder_config)
        output_config           : a list of tuple(name, dim, decoder_config)
        action_dim              : int
        state_dim               : config for dims of the latent state
        probe_config            : a list of tuple(name, dim, decoder_config)
        intrinsic_reward_config : an optional config to enable the intrinsic reward
        hidden_size             : width of the neural networks
        hidden_layers           : depth of the neural networks
        norm                    : what type of normalization layer used in the network, default is `None`.
        kl_scale                : scale for the kl term in the loss function
        free_nats               : free information per dim in latent space that is not penalized
        kl_rebalance            : rebalance the kl term with the linear combination of the two detached version, default is `None` meaning disable, enable with a float value between 0 and 1
        nll_reweight            : reweight method for the likelihood term (also the kl accordingly), choose from `none`, `modility_wise`, `dim_wise`.
        idm_mode                : mode for idm, choose from `none`, `end2end` and `detach`
        momentum_parameter      : float [0, 1], the update coefficient of the momentum_encoders, value means how much the parameter is updated for each step.
        state_distribution      : the distribution type of the stochastic latent state, choose from `continuous` and `discrete`. Default: continuous. Note: not supported by every ssm.
        min_std                 : the minimal std for all the learnable distributions for numerical stablity, set to None will follow the global default.

        NOTE: For output and probe configs, their can be a special name `emb` which indicate to predict the detached embedding from the encoders.
              For that use case, the `dim` in that config tuple will be overwrite.
        """
        super().__init__()
        self.input_config = input_config
        self.output_config = output_config
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.probe_config = probe_config
        self.intrinsic_reward_config = intrinsic_reward_config
        self.intrinsic_reward_disable = (
            self.intrinsic_reward_config is None
            or len(self.intrinsic_reward_config) == 0
        )
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.norm = norm
        self.kl_scale = kl_scale
        self.free_nats = free_nats
        self.kl_rebalance = kl_rebalance
        self.nll_reweight = nll_reweight
        assert self.nll_reweight in ("none", "modility_wise", "dim_wise")
        self.idm_mode = idm_mode
        assert self.idm_mode in (
            "none",
            "end2end",
            "detach",
        ), f"recieved unknown idm_mode `{self.idm_mode}`."
        self.min_std = min_std if min_std is not None else MIN_STD
        self.use_emb = "emb" in [
            name for name, dim, decoder_config in self.output_config
        ]
        self.momentum_parameter = momentum_parameter
        self.state_distribution = state_distribution

        self.input_keys = [config[0] for config in self.input_config]
        self.output_keys = [config[0] for config in self.output_config]
        self.probe_keys = (
            [config[0] for config in self.probe_config]
            if self.probe_config is not None
            else []
        )

        self.create_network()
        self.metric_func = torch.nn.MSELoss()

    def create_network(self):
        self._create_encoders()
        self._create_decoders()
        self._create_probes()
        self._create_idm()
        self._create_intrinsic_reward_networks()
        self._create_transition()

    def _create_encoders(self):
        self.encoders = torch.nn.ModuleDict()
        for name, dim, encoder_config in self.input_config:
            encoder_config = encoder_config.copy()
            encoder_type = encoder_config.pop("name")
            self.encoders[name] = encoder_classes[encoder_type](dim, **encoder_config)

        if self.use_emb and self.momentum_parameter < 1.0:
            self.momentum_encoders = torch.nn.ModuleDict()
            for name, dim, encoder_config in self.input_config:
                encoder_config = encoder_config.copy()
                encoder_type = encoder_config.pop("name")
                self.momentum_encoders[name] = encoder_classes[encoder_type](
                    dim, **encoder_config
                )
                self.momentum_encoders[name].requires_grad_(False)
                soft_update(self.encoders[name], self.momentum_encoders[name], 1.0)

        self.emb_dim = sum(
            [encoder.output_dim for name, encoder in self.encoders.items()]
        )

    def _create_decoders(self):
        self.decoders = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.output_config:
            if name == "emb":
                dim = self.emb_dim
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            feature_dim = (
                self.state_decoder_feature_dim
                if name in self.input_keys
                else self.state_feature_dim
            )
            self.decoders[name] = decoder_classes[decoder_type](
                feature_dim, dim, **decoder_config
            )

    def _create_probes(self):
        self.probes = torch.nn.ModuleDict()
        if self.probe_config is not None:
            for name, dim, decoder_config in self.probe_config:
                if name == "emb":
                    dim = self.emb_dim
                decoder_config = decoder_config.copy()
                decoder_type = decoder_config.pop("name")
                self.probes[name] = decoder_classes[decoder_type](
                    self.state_feature_dim, dim, **decoder_config
                )

        self.baseline_probes = torch.nn.ModuleDict()
        if self.probe_config is not None:
            for name, dim, decoder_config in self.probe_config:
                if name == "emb":
                    dim = self.emb_dim
                decoder_config = decoder_config.copy()
                decoder_type = decoder_config.pop("name")
                self.baseline_probes[name] = decoder_classes[decoder_type](
                    self.emb_dim, dim, **decoder_config
                )

    def _create_idm(self):
        if not (self.idm_mode == "none"):
            # Inverse Dynamic Model (IDM) is a non-casual policy
            self.idm = TanhGaussianPolicy(
                self.state_feature_dim + self.emb_dim,
                self.action_dim,
                hidden_size=self.hidden_size,
                hidden_layers=self.hidden_layers,
            )

    def _create_intrinsic_reward_networks(self):
        if not self.intrinsic_reward_disable:
            self.intrinsic_reward_algo = self.intrinsic_reward_config.get(
                "algo", "disagreement"
            )
            if self.intrinsic_reward_algo == "disagreement":
                self.emb_prediction_heads = EnsembleMLP(
                    self.state_feature_dim + self.action_dim,
                    self.emb_dim,
                    self.intrinsic_reward_config["hidden_size"],
                    self.intrinsic_reward_config["hidden_layers"],
                    self.intrinsic_reward_config["num_ensembles"],
                )

                if self.intrinsic_reward_config.get("use_probe", False):
                    self.intrinsic_reward_probe_head = MLP(
                        self.state_feature_dim,
                        1,
                        self.intrinsic_reward_config["hidden_size"],
                        self.intrinsic_reward_config["hidden_layers"],
                    )
            elif self.intrinsic_reward_algo == "lbs":
                # Mazzaglia et al., Curiosity-driven exploration via latent bayesian surprise, AAAI 2022
                # NOTE: not working for unknown reason...
                self.intrinsic_reward_probe_head = MLP(
                    self.state_feature_dim,
                    1,
                    self.intrinsic_reward_config["hidden_size"],
                    self.intrinsic_reward_config["hidden_layers"],
                )

    def _create_transition(self):
        raise NotImplementedError

    def get_optimizor(self, optimizor_config):
        optimizor_name = optimizor_config.pop("name")
        optimizor_class = getattr(torch.optim, optimizor_name)
        return optimizor_class(self.parameters(), **optimizor_config)

    def reset(self, batch_size):
        """reset the hidden state of the SSM"""
        raise NotImplementedError

    @property
    def state_feature_dim(self):
        return self.state_dim

    @property
    def state_decoder_feature_dim(self):
        return self.state_feature_dim

    def stack_states(self, states):
        return ArrayDict.stack(states, dim=0)

    def flatten_states(self, states):
        # flatten the sequence of states as the starting state of rollout
        if isinstance(states, list):
            states = self.stack_states(states)
        states.vmap_(lambda v: rearrange(v, "t b f -> (t b) f"))
        return states

    def get_state_feature(self, state):
        return state

    def get_state_decoder_feature(self, state):
        return self.get_state_feature(state)

    def get_emb(self, obs, encoders=None):
        if encoders is None:
            encoders = self.encoders
        return torch.cat([model(obs[name]) for name, model in encoders.items()], dim=-1)

    def get_output_dists(self, state_feature, state_decoder_feature, names=None):
        if names is None:
            names = self.decoders.keys()
        feature_selector = (
            lambda name: state_decoder_feature
            if name in self.input_keys
            else state_feature
        )
        return {
            name: self.decoders[name](feature_selector(name))
            for name in names
            if name in self.decoders.keys()
        }

    def get_outputs(self, state_feature, names=None):
        if names is None:
            names = self.decoders.keys()
        return {
            name: self.decoders[name](state_feature).mode
            for name in names
            if name in self.decoders.keys()
        }

    def get_probe_dists(self, state_feature, names=None):
        if names is None:
            names = self.probes.keys()
        return {
            name: self.probes[name](state_feature)
            for name in names
            if name in self.probes.keys()
        }

    def get_baseline_probe_dists(self, emb, names=None):
        if names is None:
            names = self.baseline_probes.keys()
        return {
            name: self.baseline_probes[name](emb)
            for name in names
            if name in self.baseline_probes.keys()
        }

    def get_probes(self, state_feature, names=None):
        if names is None:
            names = self.probes.keys()
        return {
            name: self.probes[name](state_feature).mode
            for name in names
            if name in self.probes.keys()
        }

    def main_parameters(self):
        """return parameters in the model but excluding the probes, for which applying gradient clip may lead to information leak."""
        return [v for k, v in self.named_parameters() if "probes" not in k]

    def compute_kl(self, posterior, prior, kl_rebalance_disable=False):
        # when kl_rebalance is provided to this function, it will overwrite the default value
        if self.kl_rebalance is None or kl_rebalance_disable:
            return kl_divergence(posterior, prior)
        else:
            return self.kl_rebalance * kl_divergence(posterior.detach(), prior) + (
                1 - self.kl_rebalance
            ) * kl_divergence(posterior, prior.detach())

    def _get_mask(self, is_terminal):
        index = torch.argmax(is_terminal, dim=0).squeeze()
        mask = torch.clone(is_terminal)
        mask[index, torch.arange(mask.shape[1])] = 0.0
        return mask

    def compute_per_step_loss(self, obs_seq, kls, output_dists):
        reconstruction_loss = 0
        kl_loss = 0
        metrics = {}

        if self.nll_reweight == "none":
            for name, dist in output_dists.items():
                if name not in obs_seq.keys():
                    continue
                _r_loss = -torch.flatten(dist.log_prob(obs_seq[name]), 2)
                _r_loss = _r_loss * obs_seq.get(f"{name}_mask", 1)
                _r_loss = _r_loss.sum(dim=-1, keepdim=True)
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_reconstruction_loss"] = (
                    _r_loss.sum(dim=0).mean().item()
                )
            kl_loss = kls.sum(dim=-1, keepdim=True)
        # Reference: Seitzer et. al., On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks, ICLR 2022
        # NOTE: the original version only reweight the log_prob, but here I think if the likelihood is reweighted, the kl should be reweighted accordingly.
        elif self.nll_reweight == "modility_wise":
            for name, dist in output_dists.items():
                if name not in obs_seq.keys():
                    continue
                _r_loss = -torch.flatten(
                    dist.log_prob(obs_seq[name]) * dist.stddev.detach(), 2
                )
                _r_loss = _r_loss * obs_seq.get(f"{name}_mask", 1)
                _r_loss = _r_loss.sum(dim=-1, keepdim=True)
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_reconstruction_loss"] = (
                    _r_loss.sum(dim=0).mean().item()
                )
                kl_loss = kl_loss + kls.sum(dim=-1, keepdim=True) * torch.flatten(
                    dist.stddev[: kls.shape[0]], 2
                ).detach().mean(dim=-1, keepdim=True)
            kl_loss = kl_loss / len(output_dists)
        elif self.nll_reweight == "dim_wise":
            total_dims = 0
            for name, dist in output_dists.items():
                if name not in obs_seq.keys():
                    continue
                _r_loss = -torch.flatten(
                    dist.log_prob(obs_seq[name]) * dist.stddev.detach(), 2
                )
                _r_loss = _r_loss * obs_seq.get(f"{name}_mask", 1)
                _r_loss = _r_loss.sum(dim=-1, keepdim=True)
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_reconstruction_loss"] = (
                    _r_loss.sum(dim=0).mean().item()
                )
                kl_loss = kl_loss + (
                    kls.sum(dim=-1, keepdim=True)
                    * torch.flatten(dist.stddev[: kls.shape[0]], 2).detach()
                ).sum(dim=-1, keepdim=True)
                total_dims = total_dims + np.prod(dist.stddev.shape[2:])
            kl_loss = kl_loss / total_dims

        return kl_loss, reconstruction_loss, metrics

    def forward(self, obs_seq, pre_action_seq, filter_step=None, initial_state=None):
        """the call for training the model"""
        if filter_step is None:
            filter_step = len(obs_seq)
        state = (
            self.reset(obs_seq[self.input_config[0][0]].shape[1])
            if initial_state is None
            else initial_state
        )
        emb_seq = self.get_emb(obs_seq)

        if self.use_emb:
            if self.momentum_parameter < 1.0:
                for name in self.encoders.keys():
                    soft_update(
                        self.encoders[name],
                        self.momentum_encoders[name],
                        self.momentum_parameter,
                    )
                obs_seq["emb"] = self.get_emb(obs_seq, self.momentum_encoders)
            else:
                obs_seq["emb"] = emb_seq.detach()

        states, kls = self.filter(
            obs_seq[:filter_step],
            pre_action_seq[:filter_step],
            emb_seq[:filter_step],
            state,
        )
        states, kls = self.smoother(states, kls)  # use smoother when available
        states = states + self.rollout(states[-1], pre_action_seq[filter_step:])

        # clamp the kls with free nats, but keep the real value at log
        clamp_kls = (
            torch.clamp_min(torch.sum(kls, dim=-1, keepdim=True), self.free_nats)
            / kls.shape[-1]
        )
        kls = clamp_kls + (kls - clamp_kls).detach()

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )
        state_decoder_features = torch.stack(
            [self.get_state_decoder_feature(state) for state in states]
        )
        output_dists = self.get_output_dists(state_features, state_decoder_features)

        metrics = {}
        rec_term = 0
        for name, dist in output_dists.items():
            if name not in obs_seq.keys():
                continue
            _r_term = torch.flatten(dist.log_prob(obs_seq[name]), 2)
            _r_term = _r_term * obs_seq.get(f"{name}_mask", 1)
            _r_term = torch.mean(_r_term.sum(dim=-1).sum(dim=0))
            rec_term = rec_term + _r_term
            metrics[f"{name}_mse"] = self.metric_func(dist.mean, obs_seq[name]).item()
            metrics[f"{name}_rec_term"] = _r_term.item()
        kl_term = -torch.mean(kls.sum(dim=-1).sum(dim=0))
        elbo = rec_term + kl_term
        metrics.update(
            {
                "rec_term": rec_term.item(),
                "kl_term": kl_term.item(),
                "elbo": elbo.item(),
            }
        )

        reconstruction_loss = 0
        kl_loss = 0

        kl_loss, reconstruction_loss, _metrics = self.compute_per_step_loss(
            obs_seq, kls, output_dists
        )
        mask = self._get_mask(is_terminal=obs_seq["is_terminal"])
        kl_loss = kl_loss * (1 - mask[:filter_step])
        reconstruction_loss = reconstruction_loss * (1 - mask)
        metrics.update(_metrics)
        kl_loss = kl_loss.sum(dim=0).mean()
        reconstruction_loss = reconstruction_loss.sum(dim=0).mean()

        if self.probe_config is not None:
            # NOTE: ad hoc skip the first state because there is no initial state estimator
            probe_state_features = state_features.detach()
            probe_dists = self.get_probe_dists(probe_state_features)
            for name, dist in probe_dists.items():
                if name not in obs_seq.keys():
                    continue
                _r_loss = -dist.log_prob(obs_seq[name])
                _r_loss = _r_loss * obs_seq.get(f"{name}_mask", 1)
                _r_loss = torch.mean(_r_loss[1:].sum(dim=-1).sum(dim=0))
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_probe_mse"] = self.metric_func(
                    dist.mean[1:], obs_seq[name][1:]
                ).item()
                metrics[f"{name}_probe_reconstruction_loss"] = _r_loss.item()

            probe_state_features = emb_seq.detach()
            probe_dists = self.get_baseline_probe_dists(probe_state_features)
            for name, dist in probe_dists.items():
                if name not in obs_seq.keys():
                    continue
                _r_loss = -dist.log_prob(obs_seq[name])
                _r_loss = _r_loss * obs_seq.get(f"{name}_mask", 1)
                _r_loss = torch.mean(_r_loss[1:].sum(dim=-1).sum(dim=0))
                reconstruction_loss = reconstruction_loss + _r_loss
                metrics[f"{name}_baseline_probe_mse"] = self.metric_func(
                    dist.mean[1:], obs_seq[name][1:]
                ).item()
                metrics[f"{name}_baseline_probe_reconstruction_loss"] = _r_loss.item()

        loss = reconstruction_loss + self.kl_scale * kl_loss

        if not (self.idm_mode == "none" or self.idm is None):
            idm_inputs = torch.cat([state_features[:-1], emb_seq[1:]], dim=-1)
            if self.idm_mode == "detach":
                idm_inputs = idm_inputs.detach()
            action_dist = self.idm(idm_inputs)
            idm_loss = (
                -action_dist.log_prob(pre_action_seq[1:]).sum(dim=-1).mean(dim=0).sum()
            )
            idm_mse = self.metric_func(action_dist.mode, pre_action_seq[1:])

            metrics.update({"idm_loss": idm_loss.item(), "idm_mse": idm_mse.item()})

            loss = loss + idm_loss

        if not self.intrinsic_reward_disable:
            if self.intrinsic_reward_algo == "disagreement":
                intrinsic_rewards_state_features = state_features[:-1].detach()
                inputs = torch.cat(
                    [intrinsic_rewards_state_features, pre_action_seq[1:]], dim=-1
                )
                outputs = self.emb_prediction_heads(inputs)
                target = emb_seq[1:].detach()
                emb_prediction_loss = self.metric_func(
                    outputs, target.unsqueeze(dim=-2)
                )
                loss = loss + emb_prediction_loss
                metrics["emb_prediction_loss"] = emb_prediction_loss.item()

                if self.intrinsic_reward_config.get("use_probe", False):
                    disagreements = torch.std(outputs, dim=-2)
                    intrinsic_rewards = torch.mean(
                        disagreements.log(), dim=-1, keepdim=True
                    ).detach()
                    intrinsic_rewards_probe = self.intrinsic_reward_probe_head(
                        state_features[1:].detach()
                    )
                    intrinsic_rewards_probe_loss = self.metric_func(
                        intrinsic_rewards, intrinsic_rewards_probe
                    )
                    loss = loss + intrinsic_rewards_probe_loss
                    metrics["intrinsic_rewards_probe_loss"] = (
                        intrinsic_rewards_probe_loss.item()
                    )

            elif self.intrinsic_reward_algo == "lbs":
                intrinsic_rewards = kls.detach().sum(dim=-1, keepdim=True)
                intrinsic_rewards_probe = self.intrinsic_reward_probe_head(
                    state_features.detach()
                )
                intrinsic_rewards_probe_loss = self.metric_func(
                    intrinsic_rewards,
                    intrinsic_rewards_probe[: intrinsic_rewards.shape[0]],
                )
                loss = loss + intrinsic_rewards_probe_loss
                metrics["intrinsic_rewards_probe_loss"] = (
                    intrinsic_rewards_probe_loss.item()
                )

        metrics.update(
            {
                "total_loss": loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "kl_loss": kl_loss.item(),
            }
        )

        outputs = ArrayDict({name: dist.mean for name, dist in output_dists.items()})
        if self.probe_config is not None:
            for name, dist in probe_dists.items():
                if name not in outputs.keys():
                    outputs[name] = dist.mean

        return outputs, states, loss, metrics

    def smoother(self, states, kls, kl_rebalance_disable=False):
        """Refined the posterior based on future, only used during training and available for a few types of model. Do nothing by default."""
        return states, kls

    def filter(
        self,
        obs_seq,
        pre_action_seq,
        emb_seq=None,
        initial_state=None,
        kl_rebalance_disable=False,
    ):
        if initial_state is None:
            initial_state = self.reset(obs_seq[self.input_config[0][0]].shape[1])
        if emb_seq is None:
            emb_seq = self.get_emb(obs_seq)
        assert pre_action_seq.shape[0] == emb_seq.shape[0]
        states = []
        kls = []
        state = initial_state
        for t in range(emb_seq.shape[0]):
            state, kl = self.posterior_step(
                obs_seq[t],
                pre_action_seq[t],
                state,
                emb_seq[t],
                kl_rebalance_disable=kl_rebalance_disable,
            )
            states.append(state)
            kls.append(kl)
        kls = torch.stack(kls, dim=0)
        return states, kls

    def rollout(self, initial_state, action_seq):
        state = initial_state
        states = []
        for t in range(action_seq.shape[0]):
            state = self.prior_step(action_seq[t], state)
            states.append(state)
        return states

    def likelihood_step(
        self,
        obs,
        pre_action,
        state,
        emb=None,
        determinastic=True,
        kl_rebalance_disable=False,
    ):
        """return the likelihood of the current step"""
        new_state, kl = self.posterior_step(
            obs,
            pre_action,
            state,
            emb,
            determinastic,
            kl_rebalance_disable=kl_rebalance_disable,
        )
        state_feature = self.get_state_feature(new_state)
        state_decoder_feature = self.get_state_decoder_feature(new_state)
        output_dists = self.get_output_dists(state_feature, state_decoder_feature)
        rec_term = 0
        for name, dist in output_dists.items():
            _r_term = torch.mean(torch.flatten(dist.log_prob(obs[name]), 1).sum(dim=-1))
            rec_term = rec_term + _r_term
        kl_term = -torch.mean(kl.sum(dim=-1))
        elbo = rec_term + kl_term
        return elbo.item(), new_state

    def _prepare_posterior_state(self, obs, state):
        """mainly use to handle the first state of the sequence"""
        if "is_first" in obs.keys():
            reset = 1 - obs["is_first"]
            state.vmap_(lambda v: v * reset)
        return state

    def posterior_step(
        self,
        obs,
        pre_action,
        state,
        emb=None,
        determinastic=False,
        kl_rebalance_disable=False,
    ):
        raise NotImplementedError

    def prior_step(self, pre_action, state, determinastic=False):
        raise NotImplementedError

    def generate(self, initial_state, action_seq, names=None):
        states = [initial_state, *self.rollout(initial_state, action_seq)]

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )

        outputs = self.get_probes(state_features, names)
        outputs.update(self.get_outputs(state_features, names))

        return states, ArrayDict(outputs)

    def filter_with_policy(
        self,
        obs_seq,
        policy,
        idm=None,
        idm_mode="posterior",
        filter_step=None,
        kl_only=False,
        free_nats_disable=True,
        kl_rebalance_disable=True,
    ):
        """
        Filter the states with a policy that generate actions.
        This is used for the AIME algorithm for now.
        Free nats is not used here and kl_rebalance is disable by default.
        """
        if filter_step is None:
            filter_step = len(obs_seq)
        state = self.reset(obs_seq[self.input_config[0][0]].shape[1])
        emb_seq = self.get_emb(obs_seq)
        conditions = obs_seq.get("condition", [None] * len(emb_seq))

        states = []
        kls = []
        actions_kls = []
        actions = []
        action_entropys = []

        # compute the aime/aime-idm loss first
        for t in range(filter_step):
            if idm is None:
                action_dist = policy(self.get_state_feature(state), conditions[t])
                action = action_dist.rsample()
                action_entropys.append(action_dist.entropy())
            elif idm_mode == "posterior" or idm_mode == "regularizer":
                state_feature = self.get_state_feature(state)
                action_posterior_dist = idm(
                    torch.cat([state_feature, emb_seq[t]], dim=-1)
                )
                action_prior_dist = policy(state_feature, conditions[t])
                actions_kl = torch.distributions.kl_divergence(
                    action_posterior_dist, action_prior_dist
                )
                actions_kls.append(actions_kl)
                action_entropys.append(action_prior_dist.entropy())
                action = action_posterior_dist.rsample()
            elif idm_mode == "prior":
                state_feature = self.get_state_feature(state)
                action_prior_dist = idm(torch.cat([state_feature, emb_seq[t]], dim=-1))
                action_posterior_dist = policy(state_feature, conditions[t])
                actions_kl = torch.distributions.kl_divergence(
                    action_posterior_dist, action_prior_dist
                )
                actions_kls.append(actions_kl)
                action_entropys.append(action_posterior_dist.entropy())
                action = action_posterior_dist.rsample()
            else:
                raise NotImplementedError
            state, kl = self.posterior_step(
                obs_seq[t],
                action,
                state,
                emb_seq[t],
                kl_rebalance_disable=kl_rebalance_disable,
            )
            states.append(state)
            kls.append(kl)
            actions.append(action)

        for t in range(filter_step, len(obs_seq)):
            action_dist = policy(self.get_state_feature(state), conditions[t])
            action = action_dist.rsample()
            state = self.prior_step(action, state)
            states.append(state)
            actions.append(action)
            action_entropys.append(action_dist.entropy())

        kls = torch.stack(kls)

        if not free_nats_disable:
            # clamp the kls with free nats, but keep the real value at log
            clamp_kls = (
                torch.clamp_min(torch.sum(kls, dim=-1, keepdim=True), self.free_nats)
                / kls.shape[-1]
            )
            kls = clamp_kls + (kls - clamp_kls).detach()

        actions = torch.stack(actions)
        action_entropys = torch.stack(action_entropys, dim=0)

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )
        state_decoder_features = torch.stack(
            [self.get_state_decoder_feature(state) for state in states]
        )
        output_dists = self.get_output_dists(state_features, state_decoder_features)

        metrics = {}
        rec_term = 0
        for name, dist in output_dists.items():
            if name not in obs_seq.keys():
                continue
            _r_term = torch.flatten(dist.log_prob(obs_seq[name]), 2)
            _r_term = _r_term * obs_seq.get(f"{name}_mask", 1)
            _r_term = torch.mean(_r_term.sum(dim=-1).sum(dim=0))
            rec_term = rec_term + _r_term
            metrics[f"{name}_mse"] = self.metric_func(dist.mean, obs_seq[name]).item()
            metrics[f"{name}_rec_term"] = _r_term.item()
        kl_term = -torch.mean(kls.sum(dim=-1).sum(dim=0))
        elbo = rec_term + kl_term
        metrics.update(
            {
                "rec_term": rec_term.item(),
                "kl_term": kl_term.item(),
                "elbo": elbo.item(),
            }
        )

        if idm is not None:
            action_kls = torch.stack(actions_kls)
            kls = torch.cat([kls, action_kls], dim=-1)
            action_kl_loss = action_kls.sum(dim=-1).sum(dim=0).mean()
            # loss = loss + self.kl_scale * action_kl_loss
            metrics["action_kl_loss"] = action_kl_loss.item()

        kl_loss, reconstruction_loss, _metrics = self.compute_per_step_loss(
            obs_seq, kls, output_dists
        )
        mask = self._get_mask(obs_seq["is_terminal"])
        kl_loss = kl_loss * (1 - mask[:filter_step])
        reconstruction_loss = reconstruction_loss * (1 - mask)
        metrics.update(_metrics)

        if not kl_only:
            per_step_loss = reconstruction_loss + self.kl_scale * kl_loss
        else:
            per_step_loss = self.kl_scale * kl_loss

        # when condition meets, the above loss will be regularizer, we compute the aime loss here again
        if idm is not None and idm_mode == "regularizer":
            states = []
            kls = []
            actions = []
            action_entropys = []

            # compute the global loss first
            for t in range(filter_step):
                action_dist = policy(self.get_state_feature(state), conditions[t])
                action = action_dist.rsample()
                action_entropys.append(action_dist.entropy())
                state, kl = self.posterior_step(
                    obs_seq[t],
                    action,
                    state,
                    emb_seq[t],
                    kl_rebalance_disable=kl_rebalance_disable,
                )
                states.append(state)
                kls.append(kl)
                actions.append(action)

            for t in range(filter_step, len(obs_seq)):
                action_dist = policy(self.get_state_feature(state), conditions[t])
                action = action_dist.rsample()
                state = self.prior_step(action, state)
                states.append(state)
                actions.append(action)
                action_entropys.append(action_dist.entropy())

            kls = torch.stack(kls)

            if not free_nats_disable:
                # clamp the kls with free nats, but keep the real value at log
                clamp_kls = (
                    torch.clamp_min(
                        torch.sum(kls, dim=-1, keepdim=True), self.free_nats
                    )
                    / kls.shape[-1]
                )
                kls = clamp_kls + (kls - clamp_kls).detach()

            actions = torch.stack(actions)
            action_entropys = torch.stack(action_entropys, dim=0)

            state_features = torch.stack(
                [self.get_state_feature(state) for state in states]
            )
            state_decoder_features = torch.stack(
                [self.get_state_decoder_feature(state) for state in states]
            )
            output_dists = self.get_output_dists(state_features, state_decoder_features)

            metrics = {}
            rec_term = 0
            for name, dist in output_dists.items():
                _r_term = torch.flatten(dist.log_prob(obs_seq[name]), 2)
                _r_term = _r_term * obs_seq.get(f"{name}_mask", 1)
                _r_term = torch.mean(_r_term.sum(dim=-1).sum(dim=0))
                rec_term = rec_term + _r_term
                metrics[f"{name}_mse"] = self.metric_func(
                    dist.mean, obs_seq[name]
                ).item()
                metrics[f"{name}_rec_term"] = _r_term.item()
            kl_term = -torch.mean(kls.sum(dim=-1).sum(dim=0))
            elbo = rec_term + kl_term
            metrics.update(
                {
                    "rec_term": rec_term.item(),
                    "kl_term": kl_term.item(),
                    "elbo": elbo.item(),
                }
            )

            kl_loss, reconstruction_loss, _metrics = self.compute_per_step_loss(
                obs_seq, kls, output_dists
            )
            mask = self._get_mask(obs_seq["is_terminal"])
            kl_loss = kl_loss * (1 - mask[:filter_step])
            reconstruction_loss = reconstruction_loss * (1 - mask)
            metrics.update(_metrics)

            if not kl_only:
                per_step_loss = (
                    per_step_loss + reconstruction_loss + self.kl_scale * kl_loss
                )
            else:
                per_step_loss = per_step_loss + self.kl_scale * kl_loss

        loss = per_step_loss.sum(dim=0).mean()

        if self.probe_config is not None:
            # ad hoc skip the first state because there is no initial state estimator
            probe_state_features = state_features.detach()
            probe_dists = self.get_probe_dists(probe_state_features)
            for name, dist in probe_dists.items():
                _r_loss = -dist.log_prob(obs_seq[name])
                _r_loss = _r_loss * obs_seq.get(f"{name}_mask", 1)
                _r_loss = torch.mean(_r_loss[1:].sum(dim=-1).sum(dim=0))
                loss = loss + _r_loss
                metrics[f"{name}_probe_mse"] = self.metric_func(
                    dist.mean[1:], obs_seq[name][1:]
                ).item()
                metrics[f"{name}_probe_reconstruction_loss"] = _r_loss.item()

            probe_state_features = emb_seq.detach()
            probe_dists = self.get_baseline_probe_dists(probe_state_features)
            for name, dist in probe_dists.items():
                _r_loss = -dist.log_prob(obs_seq[name])
                _r_loss = _r_loss * obs_seq.get(f"{name}_mask", 1)
                _r_loss = torch.mean(_r_loss[1:].sum(dim=-1).sum(dim=0))
                loss = loss + _r_loss
                metrics[f"{name}_baseline_probe_mse"] = self.metric_func(
                    dist.mean[1:], obs_seq[name][1:]
                ).item()
                metrics[f"{name}_baseline_probe_reconstruction_loss"] = _r_loss.item()

        metrics.update(
            {
                "total_loss": loss.item(),
                "reconstruction_loss": reconstruction_loss.sum(dim=0).mean().item(),
                "kl_loss": kl_loss.sum(dim=0).mean().item(),
            }
        )

        outputs = ArrayDict({name: dist.mean for name, dist in output_dists.items()})
        if self.probe_config is not None:
            for name, dist in probe_dists.items():
                if name not in outputs.keys():
                    outputs[name] = dist.mean
        outputs["action_entropy"] = action_entropys
        outputs["per_step_elbo"] = -per_step_loss

        return outputs, states, actions, loss, metrics

    def rollout_with_policy(
        self,
        initial_state,
        policy,
        horizon,
        condition=None,
        names=None,
        state_detach=False,
        action_sample=True,
    ):
        """
        Rollout the world model in imagination with a policy.
        In this way, the world model serves as a virtual environment for the agent.
        This is used for Dyna-style algorithm like Dreamer.
        """
        state = initial_state
        states = [initial_state]
        actions = []
        action_entropys = []
        action_logps = []
        for t in range(horizon):
            state_feature = self.get_state_feature(state)
            if state_detach:
                state_feature = state_feature.detach()
            action_dist = policy(state_feature, condition)
            action = action_dist.rsample() if action_sample else action_dist.mode
            actions.append(action)
            action_entropys.append(action_dist.entropy())
            action_logps.append(action_dist.log_prob(action.detach()))
            state = self.prior_step(action, state)
            states.append(state)

        state_features = torch.stack(
            [self.get_state_feature(state) for state in states]
        )
        actions = torch.stack(actions, dim=0)
        action_entropys = torch.stack(action_entropys, dim=0)
        action_logps = torch.stack(action_logps, dim=0)

        outputs = self.get_probes(state_features, names)
        outputs.update(self.get_outputs(state_features, names))
        outputs = ArrayDict(outputs)

        if not self.intrinsic_reward_disable:
            if self.intrinsic_reward_algo == "disagreement":
                if self.intrinsic_reward_config.get("use_probe", False):
                    outputs["intrinsic_reward"] = (
                        self.intrinsic_reward_probe_head(state_features)
                        * self.intrinsic_reward_config["scale"]
                    )
                else:
                    # NOTE: there is a problem for computing the IR for the initial state, since it depends on the previous state and the action.
                    #       Here we copy the value from the first rollout state, but it is incorrect.
                    inputs = torch.cat([state_features[:-1], actions], dim=-1)
                    emb_predictions = self.emb_prediction_heads(inputs)
                    disagreements = torch.std(emb_predictions, dim=-2)
                    intrinsic_rewards = torch.mean(
                        disagreements.log(), dim=-1, keepdim=True
                    )
                    intrinsic_rewards = torch.cat(
                        [intrinsic_rewards[:1], intrinsic_rewards], dim=0
                    )  # Wrong!
                    # NOTE: make a detach of the intrinsic reward can improve the speed.
                    outputs["intrinsic_reward"] = (
                        intrinsic_rewards * self.intrinsic_reward_config["scale"]
                    )
            elif self.intrinsic_reward_algo == "lbs":
                outputs["intrinsic_reward"] = (
                    self.intrinsic_reward_probe_head(state_features)
                    * self.intrinsic_reward_config["scale"]
                )

        outputs["action_entropy"] = action_entropys
        outputs["action_logp"] = action_logps

        return states, actions, outputs


class RSSM(SSM):
    """
    The one from PlaNet and Dreamer, name is short for Recurrent State-Space Model.
    Interpretation of the latent variables:
        h: history of all the states before (exclude the current time step)
        s: information about the current time step (ambiguity reduced by the history)
    """

    def _create_transition(self):
        memory_dim, state_dim = self.state_dim

        self.prior_mean = MLP(
            memory_dim, state_dim, self.hidden_size, self.hidden_layers, self.norm
        )
        self.prior_std = MLP(
            memory_dim,
            state_dim,
            self.hidden_size,
            self.hidden_layers,
            self.norm,
            output_activation="softplus",
        )
        self.posterior_mean = MLP(
            self.emb_dim + memory_dim,
            state_dim,
            self.hidden_size,
            self.hidden_layers,
            self.norm,
        )
        self.posterior_std = MLP(
            self.emb_dim + memory_dim,
            state_dim,
            self.hidden_size,
            self.hidden_layers,
            self.norm,
            output_activation="softplus",
        )
        self.memory_cell = torch.nn.GRUCell(state_dim + self.action_dim, memory_dim)

        self.register_buffer("initial_memory", torch.zeros(1, memory_dim))
        self.register_buffer("initial_state", torch.zeros(1, state_dim))

    @property
    def state_feature_dim(self) -> int:
        if self.state_distribution == "continuous":
            return sum(self.state_dim)
        elif self.state_distribution == "discrete":
            return self.state_dim[0] + self.state_dim[1] * self.state_dim[2]

    def get_state_feature(self, state: ArrayDict) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def reset(self, batch_size: int) -> ArrayDict:
        return ArrayDict(
            deter=torch.repeat_interleave(self.initial_memory, batch_size, dim=0),
            stoch=torch.repeat_interleave(self.initial_state, batch_size, dim=0),
        )

    def posterior_step(
        self,
        obs: ArrayDict,
        pre_action: torch.Tensor,
        state: ArrayDict,
        emb: Optional[torch.Tensor] = None,
        determinastic=False,
        kl_rebalance_disable=False,
    ):
        state = self._prepare_posterior_state(obs, state)
        h, s = state["deter"], state["stoch"]

        if emb is None:
            emb = self.get_emb(obs)

        # 1. update the determinastic part
        h = self.memory_cell(torch.cat([s, pre_action], dim=-1), h)

        # 2. compute the prior
        prior = Normal(self.prior_mean(h), self.prior_std(h) + self.min_std)

        # 3. compute the posterior
        info = torch.cat([h, emb], dim=-1)
        posterior = Normal(
            self.posterior_mean(info), self.posterior_std(info) + self.min_std
        )

        # 4. determine the state
        s = posterior.rsample() if not determinastic else posterior.mean

        # 5. compute kl for loss
        kl = self.compute_kl(
            posterior, prior, kl_rebalance_disable=kl_rebalance_disable
        )

        return ArrayDict(deter=h, stoch=s), kl

    def prior_step(self, pre_action, state, determinastic=False):
        h, s = state["deter"], state["stoch"]

        # 1. update the determinastic part
        h = self.memory_cell(torch.cat([s, pre_action], dim=-1), h)

        # 2. compute the prior
        prior = Normal(self.prior_mean(h), self.prior_std(h) + self.min_std)

        # 3. update the stochastic part
        s = prior.rsample() if not determinastic else prior.mean

        return ArrayDict(deter=h, stoch=s)


class RSSMO(RSSM):
    """
    Implement everything in the same way as the original repo.
    """

    def _create_encoders(self):
        # NOTE: this temprary implementation only consider visual and tabular modilities
        self.encoders = torch.nn.ModuleDict()
        self.modilities_and_keys = {"visual": [], "tabular": []}
        tabular_dims = 0
        for name, dim, encoder_config in self.input_config:
            encoder_config = encoder_config.copy()
            encoder_type = encoder_config.pop("name")
            if encoder_type == "mlp":
                self.modilities_and_keys["tabular"].append(name)
                tabular_encoder_config = encoder_config
                tabular_dims += dim
            else:
                self.modilities_and_keys["visual"].append(name)
                self.encoders[name] = encoder_classes[encoder_type](
                    dim, **encoder_config
                )

        if tabular_dims > 0:
            self.encoders["tabular"] = encoder_classes["mlp"](
                tabular_dims, **tabular_encoder_config
            )

        if self.use_emb and self.momentum_parameter < 1.0:
            self.momentum_encoders = torch.nn.ModuleDict()
            self.modilities_and_keys = {"visual": [], "tabular": []}
            tabular_dims = 0
            for name, dim, encoder_config in self.input_config:
                encoder_config = encoder_config.copy()
                encoder_type = encoder_config.pop("name")
                if encoder_type == "mlp":
                    self.modilities_and_keys["tabular"].append(name)
                    tabular_encoder_config = encoder_config
                    tabular_dims += dim
                else:
                    self.modilities_and_keys["visual"].append(name)
                    self.momentum_encoders[name] = encoder_classes[encoder_type](
                        dim, **encoder_config
                    )

            if tabular_dims > 0:
                self.momentum_encoders["tabular"] = encoder_classes["mlp"](
                    tabular_dims, **tabular_encoder_config
                )

            for name in self.encoders.keys():
                soft_update(self.encoders[name], self.momentum_encoders[name], 1.0)

        self.emb_dim = sum(
            [encoder.output_dim for name, encoder in self.encoders.items()]
        )

    def _create_transition(self):
        if self.state_distribution == "continuous":
            memory_dim, state_dim = self.state_dim
        elif self.state_distribution == "discrete":
            memory_dim, num_categories, category_size = self.state_dim
            state_dim = num_categories * category_size

        self.pre_memory = MLP(
            state_dim + self.action_dim,
            None,
            self.hidden_size,
            1,
            self.norm,
            have_head=False,
            hidden_activation="swish",
        )
        self.memory_cell = LNGRUCell(self.hidden_size, memory_dim)

        if self.state_distribution == "continuous":
            self.prior_net = MLP(
                memory_dim,
                2 * state_dim,
                self.hidden_size,
                1,
                self.norm,
                hidden_activation="swish",
            )
            self.posterior_net = MLP(
                self.emb_dim + memory_dim,
                2 * state_dim,
                self.hidden_size,
                1,
                self.norm,
                hidden_activation="swish",
            )
        elif self.state_distribution == "discrete":
            self.prior_net = MLP(
                memory_dim,
                state_dim,
                self.hidden_size,
                1,
                self.norm,
                hidden_activation="swish",
            )
            self.posterior_net = MLP(
                self.emb_dim + memory_dim,
                state_dim,
                self.hidden_size,
                1,
                self.norm,
                hidden_activation="swish",
            )

        self.register_buffer("initial_memory", torch.zeros(1, memory_dim))
        self.register_buffer("initial_state", torch.zeros(1, state_dim))

    def get_emb(self, obs, encoders=None):
        if encoders is None:
            encoders = self.encoders
        outputs = []
        for name, model in encoders.items():
            if name == "tabular":
                inputs = torch.cat(
                    [obs[name] for name in self.modilities_and_keys["tabular"]], dim=-1
                )
            else:
                inputs = obs[name]
            outputs.append(model(inputs))
        return torch.cat(outputs, dim=-1)

    def posterior_step(
        self,
        obs: ArrayDict,
        pre_action: torch.Tensor,
        state: ArrayDict,
        emb: Optional[torch.Tensor] = None,
        determinastic=False,
        kl_rebalance_disable=False,
    ):
        state = self._prepare_posterior_state(obs, state)
        h, s = state["deter"], state["stoch"]

        if emb is None:
            emb = self.get_emb(obs)

        # 1. update the determinastic part
        info = self.pre_memory(torch.cat([s, pre_action], dim=-1))
        h = self.memory_cell(info, h)

        # 2. compute the prior
        if self.state_distribution == "continuous":
            prior_mean, prior_std = torch.chunk(self.prior_net(h), 2, dim=-1)
            prior_std = self._sigmoid2(prior_std) + self.min_std
            prior = Normal(prior_mean, prior_std)
        elif self.state_distribution == "discrete":
            prior = MultipleOneHot(self.prior_net(h), num=self.state_dim[1])
        else:
            raise NotImplementedError

        # 3. compute the posterior
        info = torch.cat([h, emb], dim=-1)
        if self.state_distribution == "continuous":
            posterior_mean, posterior_std = torch.chunk(
                self.posterior_net(info), 2, dim=-1
            )
            posterior_std = self._sigmoid2(posterior_std) + self.min_std
            posterior = Normal(posterior_mean, posterior_std)
        elif self.state_distribution == "discrete":
            posterior = MultipleOneHot(self.posterior_net(info), num=self.state_dim[1])
        else:
            raise NotImplementedError

        # 4. determine the state
        s = posterior.rsample() if not determinastic else posterior.mode

        # 5. compute kl for loss
        kl = self.compute_kl(
            posterior, prior, kl_rebalance_disable=kl_rebalance_disable
        )

        return ArrayDict(deter=h, stoch=s), kl

    def prior_step(self, pre_action, state, determinastic=False):
        h, s = state["deter"], state["stoch"]

        # 1. update the determinastic part
        info = self.pre_memory(torch.cat([s, pre_action], dim=-1))
        h = self.memory_cell(info, h)

        # 2. compute the prior
        if self.state_distribution == "continuous":
            prior_mean, prior_std = torch.chunk(self.prior_net(h), 2, dim=-1)
            prior_std = self._sigmoid2(prior_std) + self.min_std
            prior = Normal(prior_mean, prior_std)
        elif self.state_distribution == "discrete":
            prior = MultipleOneHot(self.prior_net(h), num=self.state_dim[1])
        else:
            raise NotImplementedError

        # 3. update the stochastic part
        s = prior.rsample() if not determinastic else prior.mean

        return ArrayDict(deter=h, stoch=s)

    def _sigmoid2(self, x):
        # from dreamerv2
        return 2 * torch.sigmoid(x / 2)


ssm_classes: Dict[str, SSM] = {
    "rssm": RSSM,
    "rssmo": RSSMO,
}
