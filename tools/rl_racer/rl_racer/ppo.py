"""Separate Gaussian actor/value networks and clipped PPO updates."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


def mlp(input_dim, hidden, output_dim, slope):
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.LeakyReLU(slope)))
        input_dim = width
    return nn.Sequential(*layers, nn.Linear(input_dim, output_dim))


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, config):
        super().__init__()
        self.actor = mlp(obs_dim, config['actor_hidden'], 2, config['negative_slope'])
        self.critic = mlp(obs_dim, config['critic_hidden'], 1, config['negative_slope'])
        self.log_std = nn.Parameter(torch.full((2,), float(config['init_log_std'])))

    def distribution(self, obs):
        return Normal(self.actor(obs), self.log_std.clamp(-5, 2).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def sample(self, obs):
        dist = self.distribution(obs)
        latent = dist.sample()
        # Store pre-tanh samples: the Jacobian cancels in PPO likelihood ratios.
        return torch.tanh(latent), latent, dist.log_prob(latent).sum(-1), self.value(obs)


def advantages(rewards, values, next_values, terminated, truncated, gamma, lam):
    """Bootstrap truncations, but never propagate GAE across either reset type."""
    adv = np.zeros_like(rewards)
    carry = np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_values[t] * ~terminated[t] - values[t]
        carry = delta + gamma * lam * ~(terminated[t] | truncated[t]) * carry
        adv[t] = carry
    return adv, adv + values


def update(model, optimizer, batch, config):
    obs, latent, old_logp, adv, returns = batch
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    metrics = []
    for _ in range(config['epochs']):
        order = torch.randperm(len(obs), device=obs.device)
        for ids in order.split(config['minibatch_size']):
            dist = model.distribution(obs[ids])
            logp = dist.log_prob(latent[ids]).sum(-1)
            ratio = (logp - old_logp[ids]).exp()
            policy = -torch.minimum(ratio * adv[ids], ratio.clamp(1 - config['clip'], 1 + config['clip']) * adv[ids]).mean()
            value = .5 * (model.value(obs[ids]) - returns[ids]).square().mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy + config['value_coef'] * value - config['entropy_coef'] * entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            if not torch.isfinite(norm):
                raise FloatingPointError('non-finite PPO gradient')
            optimizer.step()
            metrics.append([loss.item(), policy.item(), value.item(), entropy.item()])
    return dict(zip(('loss', 'policy_loss', 'value_loss', 'entropy'), np.mean(metrics, axis=0).tolist()))
