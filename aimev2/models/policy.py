import torch

from aimev2.dist import Normal, OneHot, TanhNormal

from .base import MIN_STD, MLP


class TanhGaussianPolicy(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        condition_dim=None,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        min_std=None,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.min_std = min_std if min_std is not None else MIN_STD
        input_dim = (
            state_dim if self.condition_dim is None else state_dim + self.condition_dim
        )
        self.mean_net = MLP(
            input_dim, action_dim, hidden_size, hidden_layers, norm=norm
        )
        self.std_net = MLP(
            input_dim,
            action_dim,
            hidden_size,
            hidden_layers,
            norm=norm,
            output_activation="softplus",
        )

    def forward(self, state, condition=None):
        if condition is not None:
            state = torch.cat([state, condition], dim=-1)
        mean = self.mean_net(state)
        std = self.std_net(state) + self.min_std
        return TanhNormal(mean, std)


class GaussianPolicy(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        condition_dim=None,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        min_std=None,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.min_std = min_std if min_std is not None else MIN_STD
        input_dim = (
            state_dim if self.condition_dim is None else state_dim + self.condition_dim
        )
        self.mean_net = MLP(
            input_dim, action_dim, hidden_size, hidden_layers, norm=norm
        )
        self.std_net = MLP(
            input_dim,
            action_dim,
            hidden_size,
            hidden_layers,
            norm=norm,
            output_activation="softplus",
        )

    def forward(self, state, condition=None):
        if condition is not None:
            state = torch.cat([state, condition], dim=-1)
        mean = self.mean_net(state)
        std = self.std_net(state) + self.min_std
        return Normal(mean, std)


class OneHotPolicy(torch.nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        condition_dim=None,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        input_dim = (
            state_dim if self.condition_dim is None else state_dim + self.condition_dim
        )
        self.logits_net = MLP(
            input_dim, action_dim, hidden_size, hidden_layers, norm=norm, zero_init=True
        )

    def forward(self, state, condition=None):
        if condition is not None:
            state = torch.cat([state, condition], dim=-1)
        logits = self.logits_net(state)
        return OneHot(probs=torch.softmax(logits, dim=-1))
