import math

import torch
import torchvision
from einops import rearrange
from torch import nn
from torch.functional import F

from aimev2.dist import Bernoulli, Normal

MIN_STD = 1e-6


class GEGLU(nn.Module):
    def __init__(self, dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gate = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU("tanh")

    def forward(self, x):
        return self.activation(self.gate(x)) * self.linear(x)


class EnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_ensembles: int):
        super().__init__()
        self.out_features = out_features
        self.num_ensembles = num_ensembles
        self.in_features = in_features
        self.weight = nn.Parameter(
            torch.empty((num_ensembles, out_features, in_features))
        )

        self.bias = nn.Parameter(torch.empty(num_ensembles, out_features))
        self.init_parameters()

    def init_parameters(self):
        for i in range(self.num_ensembles):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, add_ensemble_dim=False):
        if add_ensemble_dim:
            input = torch.unsqueeze(input, dim=-2)
            input = torch.repeat_interleave(input, self.num_ensembles, dim=-2)
        output = torch.einsum("...ni,nji->...nj", input, self.weight)
        output = output + self.bias
        return output

    def extra_repr(self):
        return "in_features={}, out_features={}, num_ensembles={}".format(
            self.in_features,
            self.out_features,
            self.num_ensembles,
        )


class EnsembleLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        num_ensembles: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.num_ensembles = num_ensembles
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.num_ensembles, self.normalized_shape)
            )
            self.bias = nn.Parameter(
                torch.empty(self.num_ensembles, self.normalized_shape)
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input, add_ensemble_dim=False):
        output = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        if add_ensemble_dim:
            output = torch.unsqueeze(output, dim=-2)
            output = torch.repeat_interleave(output, self.num_ensembles, dim=-2)
        if self.elementwise_affine:
            output = output * self.weight + self.bias
        return output

    def extra_repr(self) -> str:
        return "shape={normalized_shape}, num_ensemble={num_ensembles}, eps={eps}, elementwise_affine={elementwise_affine}".format(
            **self.__dict__
        )


class MLP(nn.Module):
    r"""
    Multi-layer Perceptron
    Inputs:
        in_features : int, features numbers of the input
        out_features : int, features numbers of the output
        hidden_size : int, features numbers of the hidden layers
        hidden_layers : int, numbers of the hidden layers
        norm : str, normalization method between hidden layers, default : None
        hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu'
        output_activation : str, activation function used in output layer, default : 'identity'
    """

    ACTIVATION_CREATORS = {
        "relu": lambda dim: nn.ReLU(inplace=True),
        "elu": lambda dim: nn.ELU(),
        "leakyrelu": lambda dim: nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "tanh": lambda dim: nn.Tanh(),
        "sigmoid": lambda dim: nn.Sigmoid(),
        "identity": lambda dim: nn.Identity(),
        "gelu": lambda dim: nn.GELU(approximate="tanh"),
        "geglu": lambda dim: GEGLU(dim),
        "swish": lambda dim: nn.SiLU(inplace=True),
        "softplus": lambda dim: nn.Softplus(),
    }

    NORMALIZATION_CREATORS = {
        "ln": lambda dim: nn.LayerNorm(dim),
        "bn": lambda dim: nn.BatchNorm1d(dim),
    }

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        hidden_size: int = 32,
        hidden_layers: int = 2,
        norm: str = None,
        have_head: bool = True,
        dropout: float = 0.0,
        hidden_activation: str = "elu",
        output_activation: str = "identity",
        zero_init: bool = False,
    ):
        super(MLP, self).__init__()

        if out_features is None:
            out_features = hidden_size
        self.output_dim = out_features

        hidden_activation_creator = self.ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = self.ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            assert have_head, "you have to have a head when there is no hidden layers!"
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                output_activation_creator(out_features),
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(
                    nn.Linear(in_features if i == 0 else hidden_size, hidden_size)
                )
                if norm:
                    assert (
                        norm in self.NORMALIZATION_CREATORS.keys()
                    ), f"{norm} does not supported!"
                    norm_creator = self.NORMALIZATION_CREATORS[norm]
                    net.append(norm_creator(hidden_size))
                net.append(hidden_activation_creator(hidden_size))
                if dropout > 0:
                    net.append(nn.Dropout(dropout))
            if have_head:
                net.append(nn.Linear(hidden_size, out_features))
                if zero_init:
                    with torch.no_grad():
                        net[-1].weight.fill_(0)
                        net[-1].bias.fill_(0)
                net.append(output_activation_creator(out_features))
            self.net = nn.Sequential(*net)

    def forward(self, x):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        head_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])
        out = self.net(x)
        out = out.view(*head_shape, out.shape[-1])
        return out


class EnsembleMLP(nn.Module):
    r"""
    Multi-layer Perceptron with Ensemble
    Inputs:
        in_features : int, features numbers of the input
        out_features : int, features numbers of the output
        hidden_size : int, features numbers of the hidden layers
        hidden_layers : int, numbers of the hidden layers
        num_ensembles : int, numbers of the ensemble models
        norm : str, normalization method between hidden layers, default : None
        hidden_activation : str, activation function used in hidden layers, default : 'leakyrelu'
        output_activation : str, activation function used in output layer, default : 'identity'
    """

    ACTIVATION_CREATORS = MLP.ACTIVATION_CREATORS

    NORMALIZATION_CREATORS = {
        "ln": lambda dim, num_ensembles: EnsembleLayerNorm(dim, num_ensembles),
    }

    def __init__(
        self,
        in_features: int,
        out_features: int = None,
        hidden_size: int = 32,
        hidden_layers: int = 2,
        num_ensembles: int = 1,
        norm: str = None,
        have_head: bool = True,
        dropout: float = 0.0,
        hidden_activation: str = "elu",
        output_activation: str = "identity",
        zero_init: bool = False,
    ):
        super(EnsembleMLP, self).__init__()
        self.num_ensembles = num_ensembles

        if out_features is None:
            out_features = hidden_size
        self.output_dim = out_features

        hidden_activation_creator = self.ACTIVATION_CREATORS[hidden_activation]
        output_activation_creator = self.ACTIVATION_CREATORS[output_activation]

        if hidden_layers == 0:
            assert have_head, "you have to have a head when there is no hidden layers!"
            self.net = nn.Sequential(
                EnsembleLinear(in_features, out_features, num_ensembles),
                output_activation_creator(out_features),
            )
        else:
            net = []
            for i in range(hidden_layers):
                net.append(
                    EnsembleLinear(
                        in_features if i == 0 else hidden_size,
                        hidden_size,
                        num_ensembles,
                    )
                )
                if norm:
                    assert (
                        norm in self.NORMALIZATION_CREATORS.keys()
                    ), f"{norm} does not supported!"
                    norm_creator = self.NORMALIZATION_CREATORS[norm]
                    net.append(norm_creator(hidden_size, num_ensembles))
                net.append(hidden_activation_creator(hidden_size))
                if dropout > 0:
                    net.append(nn.Dropout(dropout))
            if have_head:
                net.append(EnsembleLinear(hidden_size, out_features, num_ensembles))
                if zero_init:
                    with torch.no_grad():
                        net[-1].weight.fill_(0)
                        net[-1].bias.fill_(0)
                net.append(output_activation_creator(out_features))
            self.net = nn.Sequential(*net)

    def forward(self, x, add_ensemble_dim=True):
        r"""forward method of MLP only assume the last dim of x matches `in_features`"""
        if add_ensemble_dim:
            x = torch.unsqueeze(x, dim=-2)
            x = torch.repeat_interleave(x, self.num_ensembles, dim=-2)
        out = self.net(x)
        return out


class CNNEncoderHa(nn.Module):
    """
    The structure is introduced in Ha and Schmidhuber, World Model.
    NOTE: The structure only works for 64 x 64 image.
    """

    def __init__(self, image_size, width=32, *args, **kwargs) -> None:
        super().__init__()

        self.resize = torchvision.transforms.Resize(64)
        self.net = nn.Sequential(
            nn.Conv2d(3, width, 4, 2),
            nn.ReLU(True),  # This relu is problematic
            nn.Conv2d(width, width * 2, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(width * 2, width * 4, 4, 2),
            nn.ReLU(True),
            nn.Conv2d(width * 4, width * 8, 4, 2),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )

        self.output_dim = 4 * width * 8

    def forward(self, image):
        """forward process an image, the return feature is 1024 dims"""
        head_dims = image.shape[:-3]
        image = image.view(-1, *image.shape[-3:])
        image = self.resize(image)
        output = self.net(image)
        return output.view(*head_dims, self.output_dim)


class DreamerResnetBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.InstanceNorm2d(dim, affine=True),
            nn.SiLU(True),
            nn.Conv2d(dim, dim, 3, 1, padding=1),
            nn.InstanceNorm2d(dim, affine=True),
            nn.SiLU(True),
            nn.Conv2d(dim, dim, 3, 1, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class DreamerResnetEncoder(torch.nn.Module):
    """
    The encoder structure used in dreamerv3 repo.
    NOTE: the layer norm in original jax repo is a normalization on the last C dimensions,
          which is essentially a InstanceNorm.
    """

    def __init__(self, image_size, width, blocks=0, minres=4, *args, **kwargs):
        super().__init__()

        assert image_size[0] == image_size[1], "we only support square size for now"

        self._width = width
        self._blocks = blocks
        self._minres = minres

        self.net = nn.ModuleList()

        stages = int(math.log2(image_size[0]) - math.log2(self._minres))
        width = self._width
        current_size = image_size[0]
        for i in range(stages):
            self.net.append(
                nn.Conv2d(3 if i == 0 else width // 2, width, 4, 2, padding=1)
            )
            current_size = current_size // 2
            self.net.append(nn.InstanceNorm2d(width, affine=True))
            self.net.append(nn.SiLU(True))
            for j in range(self._blocks):
                self.net.append(DreamerResnetBlock(width))
            width *= 2

        if self._blocks:
            self.net.append(nn.SiLU(True))

        self.net = nn.Sequential(*self.net)

        self.output_dim = self._minres**2 * (width // 2)

    def __call__(self, image):
        image = image - 0.5
        head_dims = image.shape[:-3]
        image = image.view(-1, *image.shape[-3:])
        output = self.net(image)
        return output.view(*head_dims, self.output_dim)


class IndentityEncoder(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = self.input_dim

    def forward(self, x):
        return x


encoder_classes = {
    "mlp": MLP,
    "identity": IndentityEncoder,
    "cnn_ha": CNNEncoderHa,
    "dreamer_resnet": DreamerResnetEncoder,
}


class MultimodalEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.encoders = torch.nn.ModuleDict()
        for name, dim, encoder_config in self.config:
            encoder_config = encoder_config.copy()
            encoder_type = encoder_config.pop("name")
            self.encoders[name] = encoder_classes[encoder_type](dim, **encoder_config)

        self.output_dim = sum(
            [encoder.output_dim for name, encoder in self.encoders.items()]
        )

    def forward(self, obs):
        return torch.cat(
            [model(obs[name]) for name, model in self.encoders.items()], dim=-1
        )


class MLPDeterministicDecoder(torch.nn.Module):
    """
    determinasticly decode the states to outputs.
    For consistent API, it output a Guassian with \sigma=1,
    so that the gradient is the same as L2 loss.
    """

    def __init__(
        self,
        state_dim,
        obs_dim,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        hidden_activation="elu",
    ) -> None:
        super().__init__()
        self.net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
        )

    def forward(self, states):
        obs = self.net(states)
        return Normal(obs, torch.ones_like(obs))


class MLPStochasticDecoder(torch.nn.Module):
    """
    decode the states to Gaussian distributions of outputs.
    """

    def __init__(
        self,
        state_dim,
        obs_dim,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        min_std=None,
        hidden_activation="elu",
    ) -> None:
        super().__init__()
        self.min_std = min_std if min_std is not None else MIN_STD
        self.mu_net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
        )
        self.std_net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
            output_activation="softplus",
        )

    def forward(self, states):
        obs_dist = Normal(self.mu_net(states), self.std_net(states) + self.min_std)
        return obs_dist


class MLPStaticStochasticDecoder(torch.nn.Module):
    """
    decode the states to Gaussian distributions of outputs with the standard deviation is a learnable global variable.
    """

    def __init__(
        self,
        state_dim,
        obs_dim,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        min_std=None,
        hidden_activation="elu",
    ) -> None:
        super().__init__()
        self.min_std = min_std if min_std is not None else MIN_STD
        self.mu_net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
        )
        self.log_std = nn.Parameter(torch.zeros(obs_dim))

    def forward(self, states):
        obs_dist = Normal(self.mu_net(states), torch.exp(self.log_std) + self.min_std)
        return obs_dist


class MLPBinaryDecoder(torch.nn.Module):
    """
    decode the states to Bernoulli distributions of outputs.
    """

    def __init__(
        self,
        state_dim,
        obs_dim,
        hidden_size=32,
        hidden_layers=2,
        norm=None,
        hidden_activation="elu",
    ) -> None:
        super().__init__()
        self.prob_net = MLP(
            state_dim,
            obs_dim,
            hidden_size,
            hidden_layers,
            norm,
            hidden_activation=hidden_activation,
            output_activation="sigmoid",
        )

    def forward(self, states):
        prob = self.prob_net(states)
        return Bernoulli(probs=prob)


class CNNDecoderHa(nn.Module):
    """
    The structure is introduced in Ha and Schmidhuber, World Model.
    NOTE: The structure only works for 64 x 64 image, pixel range [0, 1].
    """

    def __init__(
        self, state_dim, output_size=64, width=32, sigmoid=True, *args, **kwargs
    ) -> None:
        # Here the `sigmoid` is setted to `True` by default to be compatible with old models, for new models we suggest to set to `False`.
        super().__init__()
        self.latent_dim = state_dim
        self.output_size = output_size
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim, 32 * width),
            nn.Unflatten(-1, (32 * width, 1, 1)),
            nn.ConvTranspose2d(32 * width, 4 * width, 5, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4 * width, 2 * width, 5, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(2 * width, width, 6, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(width, 3, 6, 2),
        )

        self.output_activation = nn.Sigmoid() if sigmoid else lambda x: x + 0.5

    def forward(self, state):
        head_dims = state.shape[:-1]
        state = state.view(-1, self.latent_dim)
        output = self.output_activation(self.net(state))
        output = F.interpolate(output, self.output_size)
        return Normal(output.view(*head_dims, *output.shape[-3:]), 1)


class DreamerResnetDecoder(nn.Module):
    """
    The decoder structure used in dreamerv3 repo.
    """

    def __init__(
        self,
        state_dim,
        output_size,
        width,
        blocks=0,
        minres=4,
        sigmoid=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        assert output_size[0] == output_size[1], "we only support square size for now"

        self.latent_dim = state_dim
        self.output_size = output_size
        self._width = width
        self._blocks = blocks
        self._minres = minres
        self._sigmoid = sigmoid

        stages = int(math.log2(output_size[0]) - math.log2(self._minres))
        width = self._width * 2 ** (stages - 1)

        self.net = nn.ModuleList()
        self.net.append(nn.Linear(self.latent_dim, self._minres**2 * width))
        self.net.append(nn.Unflatten(-1, (width, self._minres, self._minres)))

        for i in range(stages):
            for j in range(self._blocks):
                self.net.append(DreamerResnetBlock(width))

            if i == stages - 1:
                self.net.append(nn.ConvTranspose2d(width, 3, 4, 2, padding=1))
            else:
                self.net.append(nn.ConvTranspose2d(width, width // 2, 4, 2, padding=1))
                width = width // 2
                self.net.append(nn.InstanceNorm2d(width))
                self.net.append(nn.SiLU(True))

        self.net = nn.Sequential(*self.net)

    def forward(self, state):
        head_dims = state.shape[:-1]
        state = state.view(-1, self.latent_dim)
        output = self.net(state)
        output = torch.sigmoid(output) if self._sigmoid else output + 0.5
        return Normal(output.view(*head_dims, *output.shape[-3:]), 1)


decoder_classes = {
    "dmlp": MLPDeterministicDecoder,
    "smlp": MLPStochasticDecoder,
    "ssmlp": MLPStaticStochasticDecoder,
    "binary": MLPBinaryDecoder,
    "cnn_ha": CNNDecoderHa,
    "dreamer_resnet": DreamerResnetDecoder,
}


class MultimodalDecoder(nn.Module):
    def __init__(self, emb_dim, config) -> None:
        super().__init__()
        self.config = config
        self.decoders = torch.nn.ModuleDict()
        for name, dim, decoder_config in self.config:
            decoder_config = decoder_config.copy()
            decoder_type = decoder_config.pop("name")
            self.decoders[name] = decoder_classes[decoder_type](
                emb_dim, dim, **decoder_config
            )

    def forward(self, emb):
        return {name: decoder(emb).mean for name, decoder in self.decoders.items()}


class LNGRUCell(nn.Module):
    """GRU Cell with LayerNorm for its inputs"""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_layer = torch.nn.Linear(input_dim + hidden_dim, 3 * hidden_dim)
        self.norm = torch.nn.LayerNorm(3 * hidden_dim)

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=-1)
        z, r, o = torch.chunk(self.norm(self.input_layer(inputs)), 3, dim=-1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)
        o = torch.tanh(r * o)
        h = (1 - z) * o + z * h
        return h


class FlareMerge(nn.Module):
    "The merge method proposed in Shang et al. Reinforcement Learning with Latent Flow, NeurIPS 2021"

    def __init__(self, dim, stack) -> None:
        super().__init__()
        self.dim = dim
        self.stack = stack
        self.output_dim = dim
        self.fc = nn.Linear(dim * 2 * (stack - 1), dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, latents):
        assert latents.shape[0] == self.stack
        flow = latents[1:] - latents[:-1].detach()
        flow = torch.cat([flow, latents[1:]], dim=0)
        flow = rearrange(flow, "t b f -> b (t f)")
        return self.norm(self.fc(flow))


class ConcatMerge(nn.Module):
    def __init__(self, dim, stack) -> None:
        super().__init__()
        self.dim = dim
        self.stack = stack
        self.output_dim = dim * stack

    def forward(self, latents):
        assert latents.shape[0] == self.stack
        return rearrange(latents, "t b f -> b (t f)")


if __name__ == "__main__":
    image = torch.randn(2, 3, 128, 128)
    encoder = DreamerResnetEncoder((128, 128), 32, 1)
    decoder = DreamerResnetDecoder(encoder.output_dim, (128, 128), 32, 1)

    latent = encoder(image)
    print(latent.shape)
    image_dist = decoder(latent)
    print(image_dist.scale.shape)
    print(image_dist.log_prob(image))
