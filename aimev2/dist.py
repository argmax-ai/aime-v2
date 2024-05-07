import math

import torch
from einops import rearrange
from torch.distributions import TransformedDistribution
from torch.distributions.kl import kl_divergence, register_kl
from torch.distributions.transforms import TanhTransform

TANH_CLIP = 0.999
NUM_KL_APPROXIMATE_SAMPLES = 1024


class Normal(torch.distributions.Normal):
    constant = 0.5 * math.log(2 * math.pi)

    def __init__(self, loc, scale, remove_constant=True, validate_args=False):
        super().__init__(loc, scale, validate_args)
        self.remove_constant = remove_constant

    def log_prob(self, value):
        log_prob = super().log_prob(value)
        if self.remove_constant:
            log_prob = log_prob + self.constant
        return log_prob

    @property  # make pytorch < 1.12 compatible with the mode api
    def mode(self):
        return self.mean

    def detach(self):
        """return a new distribution with the same parameters but the gradients are detached"""
        return Normal(self.mean.detach(), self.scale.detach())


class Delta(Normal):
    def __init__(self, loc, remove_constant=True, validate_args=False):
        super().__init__(loc, 1, remove_constant, validate_args)

    def rsample(self, sample_shape=torch.Size()):
        mode = self.mode
        return mode.repeat(*sample_shape, *([1] * len(mode.shape)))

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape).detach()

    @property
    def stddev(self):
        return torch.zeros_like(self.loc)


class TanhNormal(torch.distributions.Distribution):
    def __init__(self, mean, std, validate_args=False):
        self.base = Normal(mean, std, validate_args)
        super().__init__(self.base.batch_shape, self.base.event_shape, validate_args)
        self.dist = TransformedDistribution(self.base, TanhTransform(), validate_args)

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def rsample(self, sample_shape=torch.Size()):
        return self.dist.rsample(sample_shape)

    def log_prob(self, value):
        value = torch.clamp(
            value, -TANH_CLIP, TANH_CLIP
        )  # NOTE: normally, we don't need gradient from here
        return self.base.log_prob(torch.atanh(value)) - torch.log(1 - value**2)

    @property
    def mode(self):
        """NOTE: this is not really the mode, just a easy computation"""
        return torch.tanh(self.base.mode)

    def detach(self):
        """return a new distribution with the same parameters but the gradients are detached"""
        return TanhNormal(self.base.mean.detach(), self.base.scale.detach())

    def entropy(self):
        # the empirical maximima entropy tanhNormal is (0, 0.8742) in which p(0.5) = 0.5
        return -kl_divergence(self, TanhNormal(0, 0.8742))


class OneHot(torch.distributions.OneHotCategoricalStraightThrough):
    def detach(self):
        return OneHot(probs=self.probs.detach())

    def log_prob(self, value):
        logp = super().log_prob(value)
        return logp.unsqueeze(dim=-1)


class OneHotReinMax(OneHot):
    """
    implement ReinMax rsample from Liu et al., Bridging Discrete and Backpropagation: Straight-Through and Beyond, NeurIPS 2023
    """

    def rsample(self, sample_shape=torch.Size()):
        sample = self.sample(sample_shape)
        p = (self.probs + sample) / 2
        p = torch.softmax((torch.log(p) - self.logits).detach() + self.logits, dim=-1)
        p = 2 * p - self.probs / 2
        sample = sample + p - p.detach()
        return sample


class MultipleOneHot(torch.distributions.Distribution):
    def __init__(self, logits, num, onehot_cls=OneHot, validate_args=False):
        self.num = num
        self.onehot_cls = onehot_cls
        self._original_logits = logits
        self.logits = rearrange(logits, "... (n d) -> ... n d", n=self.num)
        self.probes = torch.softmax(self.logits, dim=-1)
        self.dist = self.onehot_cls(probs=self.probes, validate_args=validate_args)
        self._batch_shape = logits.shape[:-1]
        self._event_shape = logits.shape[-1:]

    def __getattr__(self, name):
        return getattr(self.dist, name)

    def detach(self):
        return MultipleOneHot(self._original_logits.detach(), self.num, self.onehot_cls)

    def log_prob(self, value):
        value = rearrange(value, "... (n d) -> ... n d", n=self.num)
        log_prob = self.dist.log_prob(value)
        log_prob = log_prob.squeeze(dim=-1)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        sample = self.dist.sample(sample_shape)
        sample = rearrange(sample, "... n d -> ... (n d)")
        return sample

    def rsample(self, sample_shape=torch.Size()):
        sample = self.dist.rsample(sample_shape)
        sample = rearrange(sample, "... n d -> ... (n d)")
        return sample

    @property
    def mode(self):
        mode = self.dist.mode
        mode = rearrange(mode, "... n d -> ... (n d)")
        return mode

    def entropy(self):
        entropy = self.dist.entropy()
        return entropy


class Bernoulli(torch.distributions.Bernoulli):
    def detach(self):
        return Bernoulli(probs=self.probs.detach())

    @property
    def mode(self):
        return self.mean


@register_kl(TanhNormal, TanhNormal)
def _kl_tanhnormal_tanhnormal(p: TanhNormal, q: TanhNormal):
    # NOTE: kl between two distribution transformed with the same transformation,
    #       is equal to the kl between the two distribution before transformation.
    return kl_divergence(p.base, q.base)


@register_kl(Normal, TanhNormal)
def _kl_normal_tanhnormal(p: Normal, q: TanhNormal):
    # NOTE: This quantity should be infinity in theory due to the fact that
    #       Noraml cover space that is not covered by TanhNormal.
    #       Here the quantity is fakely computed just to fit in the equation.
    samples = p.sample((NUM_KL_APPROXIMATE_SAMPLES,))
    logp = p.entropy()
    logq = q.log_prob(samples)
    return logp - logq.mean(dim=0)


@register_kl(TanhNormal, Normal)
def _kl_tanhnormal_normal(p: TanhNormal, q: Normal):
    samples = p.sample((NUM_KL_APPROXIMATE_SAMPLES,))
    logp = p.log_prob(samples)
    logq = q.log_prob(samples)
    return (logp - logq).mean(dim=0)


@register_kl(MultipleOneHot, MultipleOneHot)
def _kl_multipleonehot_multipleonehot(p: MultipleOneHot, q: MultipleOneHot):
    return kl_divergence(p.dist, q.dist)


if __name__ == "__main__":
    mean = torch.randn(10)
    std = torch.rand(10)
    target_dist = TanhNormal(mean, std)
    learnable_mean = torch.randn(10)
    learnable_logstd = torch.randn(10)
    learnable_mean.requires_grad_(True)
    learnable_logstd.requires_grad_(True)
    optim = torch.optim.Adam((learnable_mean, learnable_logstd), lr=1e-3)

    for _ in range(10000):
        samples = target_dist.sample((1024,))
        dist = TanhNormal(learnable_mean, torch.exp(learnable_logstd))
        log_prob = dist.log_prob(samples)
        loss = -log_prob.sum(dim=-1).mean()

        # dist = TanhNormal(learnable_mean, torch.exp(learnable_logstd))
        # loss = kl_divergence(target_dist, dist).sum()

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(loss)

    print(mean)
    print(learnable_mean)
    print(std)
    print(learnable_logstd.exp())
