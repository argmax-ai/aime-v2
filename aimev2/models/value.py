import torch

from aimev2.utils import soft_update

from .base import MLP, EnsembleMLP


class Updater:
    def __init__(self, period, tau) -> None:
        self.period = period
        self.tau = tau
        self.count = 0

    def __call__(self, source, target):
        self.count += 1
        if self.count % self.period == 0:
            self.count = 0
            soft_update(source, target, self.tau)


class VNet(torch.nn.Module):
    def __init__(
        self,
        feature_dim,
        hidden_size,
        hidden_layers,
        norm=None,
        target_config=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.target_config = target_config

        self.vnet = MLP(feature_dim, 1, hidden_size, hidden_layers, norm=norm)

        if target_config is not None:
            self.vnet_target = MLP(
                feature_dim, 1, hidden_size, hidden_layers, norm=norm
            )
            soft_update(self.vnet, self.vnet_target, 1.0)
            self.vnet_target.requires_grad_(False)
            self.updater = Updater(**self.target_config)
        else:
            self.vnet_target = self.vnet
            self.updater = lambda x, y: None

    def forward(self, feature):
        return self.vnet(feature)

    def compute_target(self, feature):
        return self.vnet_target(feature)

    def update_target(self):
        """update the target network if should"""
        self.updater(self.vnet, self.vnet_target)


class EnsembleVNet(torch.nn.Module):
    """
    Ensemble version of the Value function, inspired from the REDQ paper.
    Chen et al., Randomized Ensembled Double Q-Learning: Learning Fast Without a Model, ICLR 2021
    """

    def __init__(
        self,
        feature_dim,
        hidden_size,
        hidden_layers,
        num_ensembles,
        num_target_selections=2,
        norm=None,
        target_config=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.num_ensembles = num_ensembles
        self.num_target_selections = num_target_selections
        self.target_config = target_config

        self.vnet = EnsembleMLP(
            feature_dim,
            1,
            hidden_size,
            hidden_layers,
            num_ensembles=num_ensembles,
            norm=norm,
        )

        if target_config is not None:
            self.vnet_target = EnsembleMLP(
                feature_dim,
                1,
                hidden_size,
                hidden_layers,
                num_ensembles=num_ensembles,
                norm=norm,
            )
            soft_update(self.vnet, self.vnet_target, 1.0)
            self.vnet_target.requires_grad_(False)
            self.updater = Updater(**self.target_config)
        else:
            self.vnet_target = self.vnet
            self.updater = lambda x, y: None

    def forward(self, feature):
        return self.vnet(feature).mean(dim=-2)

    def compute_target(self, feature):
        values = self.vnet_target(feature)
        head_shape = values.shape[:-2]
        indexes = torch.randint(
            0, self.num_ensembles, size=(self.num_target_selections,)
        )
        values = values[..., indexes, :]
        values = torch.min(values, dim=-2)[0]
        return values

    def update_target(self):
        """update the target network if should"""
        self.updater(self.vnet, self.vnet_target)


class VNetDict(torch.nn.Module):
    def __init__(
        self,
        feature_dim,
        reward_keys,
        hidden_size,
        hidden_layers,
        norm=None,
        target_config=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.reward_keys = reward_keys
        self.vnets = torch.nn.ModuleDict(
            {
                k: VNet(
                    feature_dim,
                    hidden_size,
                    hidden_layers,
                    norm,
                    target_config,
                    *args,
                    **kwargs,
                )
                for k in self.reward_keys
            }
        )

    def forward(self, feature):
        values = {k: model(feature) for k, model in self.vnets.items()}
        values["total_reward"] = sum([values[k] for k in self.reward_keys])
        return values

    def compute_target(self, feature):
        values = {k: model.compute_target(feature) for k, model in self.vnets.items()}
        values["total_reward"] = sum([values[k] for k in self.reward_keys])
        return values

    def update_target(self):
        """update the target network if should"""
        for k, model in self.vnets.items():
            model.update_target()
