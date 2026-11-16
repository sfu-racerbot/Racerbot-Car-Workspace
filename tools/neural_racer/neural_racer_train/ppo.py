"""PPO pieces: generalized advantage estimation, the clipped losses, and the
update every PPO stage runs (plan A8, item 1.1).

Kept free of any environment so each part can be tested against a hand
computation (tests/test_ppo.py).
"""
from dataclasses import dataclass, replace

import math

import torch


@dataclass(frozen=True)
class PPOConfig:
    gamma: float = 0.998        # teacher.py's since teacher-v3: a 12.5 s horizon
    lam: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    epochs: int = 4
    minibatches: int = 4
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    aux_path: float = 0.5
    aux_opp: float = 0.5
    entropy_coef: float = 0.005
    entropy_decay_frac: float = 0.25    # linear to zero over this fraction of the run
    desired_kl: float = 0.01            # adaptive actor learning rate (Rudin et al. 2021); 0 = fixed
    lr_min: float = 1e-5
    lr_max: float = 1e-3


def gae(rewards, values, next_values, terminal, done, gamma: float, lam: float):
    """Advantages and returns over a (T, N) rollout.

    next_values[t]: the value of the state step t reached, taken before any
    reset -- so a step cut by the time limit bootstraps from where the car
    really was, not from the next episode's start (Pardo et al. 2018).
    terminal: the episode really ended (no bootstrap). done: any episode end,
    terminal or truncated (the advantage trace stops there).
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_values[t] * (~terminal[t]).float() - values[t]
        last = delta + gamma * lam * (~done[t]).float() * last
        adv[t] = last
    return adv, adv + values


class ValueNorm:
    """Running mean and variance of the returns: the critic regresses
    normalised targets, so a -40 crash and a +0.1 tick of progress train it
    at one scale (PopArt's statistics, without the output rescaling)."""

    def __init__(self, device='cpu'):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = torch.tensor(1e-4, device=device)

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1).float()
        n = x.numel()
        b_mean, b_var = x.mean(), x.var(unbiased=False)
        delta, total = b_mean - self.mean, self.count + n
        self.var = (self.var * self.count + b_var * n + delta ** 2 * self.count * n / total) / total
        self.mean = self.mean + delta * n / total
        self.count = total

    def std(self):
        return torch.sqrt(self.var).clamp(min=1e-3)

    def norm(self, x):
        return (x - self.mean) / self.std()

    def denorm(self, y):
        return y * self.std() + self.mean

    def state(self) -> dict:
        return {'mean': float(self.mean), 'var': float(self.var), 'count': float(self.count)}

    def load(self, st: dict) -> None:
        for k in ('mean', 'var', 'count'):
            setattr(self, k, torch.tensor(st[k], device=self.mean.device))


def gaussian_kl(mean_old, mean_new, log_std_old, log_std_new):
    """KL(old || new) of diagonal Gaussians, summed over action dims, averaged over the batch."""
    var_old, var_new = torch.exp(2 * log_std_old), torch.exp(2 * log_std_new)
    kl = log_std_new - log_std_old + (var_old + (mean_old - mean_new) ** 2) / (2 * var_new) - 0.5
    return kl.sum(-1).mean()


def adapt_lr(lr: float, kl: float, cfg: PPOConfig) -> float:
    """legged_gym's rule: shrink the step when the policy moved too far, grow it when it barely moved."""
    if cfg.desired_kl <= 0:
        return lr
    if kl > 2.0 * cfg.desired_kl:
        lr /= 1.5
    elif 0.0 < kl < 0.5 * cfg.desired_kl:
        lr *= 1.5
    return min(max(lr, cfg.lr_min), cfg.lr_max)


def policy_loss(logp_new, logp_old, adv, clip: float):
    ratio = torch.exp(logp_new - logp_old)
    return -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv).mean()


def value_loss(v_new, returns):
    return 0.5 * ((v_new - returns) ** 2).mean()


def gaussian_entropy(log_std):
    """Entropy of the diagonal Gaussian policy (its std is state-independent)."""
    return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()


def entropy_coef(cfg: PPOConfig, steps: float, total: float) -> float:
    return cfg.entropy_coef * max(0.0, 1.0 - steps / (cfg.entropy_decay_frac * total))


def normalize(adv):
    return (adv - adv.mean()) / (adv.std() + 1e-8)


def kl_config(cfg: PPOConfig, steps: float, total: float, frac: float, final: float) -> PPOConfig:
    """The KL target, shrinking linearly to `final` x over the last `frac` of the run (plan item 1.6)."""
    left = (total - steps) / max(frac * total, 1.0)
    return cfg if frac <= 0 or left >= 1.0 else replace(cfg, desired_kl=cfg.desired_kl * (final + (1 - final) * max(left, 0.0)))


def accumulate(loss_fn, mb, micro: int, scale: bool = True):
    """Backward of loss_fn(p, mb) over the minibatch `mb` in pieces p of at
    most `micro` units: the same gradient as one pass over mb, in bounded
    memory. scale: loss_fn returns its piece's mean, weighted here by the
    piece's share of mb; else loss_fn already normalises by mb (a masked
    mean over sequences). Returns mb's loss."""
    total = 0.0
    for p in mb.split(micro):
        loss = loss_fn(p, mb)
        if scale:
            loss = loss * (p.numel() / mb.numel())
        loss.backward()
        total = total + loss.detach()
    return total


class Update:
    """The PPO update the teacher and the league share (plan item 1.1):
    separate optimisers and gradient clips for actor and critic (one joint
    norm let the critic's gradient shrink the actor's step: to 20-51% on the
    teacher), the critic stepped first on every minibatch, the actor only once
    the critic has warmed up, and a KL-adaptive actor step measured from each
    minibatch's own forward pass -- the KL so far, before its step, as
    legged_gym measures it, so no third forward pass (item 1.5). The first
    minibatch's KL is zero by construction and does not move the step."""

    def __init__(self, cfg: PPOConfig, actor_params, critic_params, actor_lr: float, critic_lr: float,
                 device='cpu', micro: int = 131072):
        fused = {'fused': True} if str(device).startswith('cuda') else {}
        self.cfg, self.micro, self.device = cfg, micro, device
        self.actor_params, self.critic_params = list(actor_params), list(critic_params)
        self.actor_opt = torch.optim.Adam(self.actor_params, lr=actor_lr, **fused)
        self.critic_opt = torch.optim.Adam(self.critic_params, lr=critic_lr, **fused)
        self.lr = actor_lr

    def state(self) -> dict:
        return {'actor_opt': self.actor_opt.state_dict(), 'critic_opt': self.critic_opt.state_dict(), 'lr': self.lr}

    def load(self, st: dict) -> None:
        self.actor_opt.load_state_dict(st['actor_opt'])
        self.critic_opt.load_state_dict(st['critic_opt'])
        self.lr = st['lr']
        for g in self.actor_opt.param_groups:
            g['lr'] = self.lr

    def run(self, n: int, critic_loss, actor_loss, train_actor: bool, kl_cfg: PPOConfig | None = None,
            actor_extra=None, scale: bool = True, micro: int | None = None) -> dict:
        """cfg.epochs passes over n units (samples, or whole env sequences) in
        cfg.minibatches. critic_loss(p, mb) -> loss; actor_loss(p, mb) ->
        (loss, kl), kl the piece's KL(rollout || now) on the same scale as
        its loss; actor_extra() -> a loss added once per actor step (the
        entropy bonus). Returns {'kl': mean, 'kls': [...], 'value_loss'}."""
        cfg, kl_cfg, micro = self.cfg, kl_cfg or self.cfg, micro or self.micro
        kls, v_loss, first = [], torch.zeros(()), True
        for _ in range(cfg.epochs if n > 1 else 0):
            for mb in torch.randperm(n, device=self.device).chunk(cfg.minibatches):
                self.critic_opt.zero_grad()
                v_loss = accumulate(critic_loss, mb, micro, scale)
                torch.nn.utils.clip_grad_norm_(self.critic_params, cfg.max_grad_norm)
                self.critic_opt.step()
                if not train_actor:
                    continue
                self.actor_opt.zero_grad()
                parts = []

                def piece(p, whole):
                    loss, kl = actor_loss(p, whole)
                    parts.append(kl.detach() * (p.numel() / whole.numel() if scale else 1.0))
                    return loss
                accumulate(piece, mb, micro, scale)
                if not first:
                    kl = float(sum(parts))
                    kls.append(kl)
                    self.lr = adapt_lr(self.lr, kl, kl_cfg)
                    for g in self.actor_opt.param_groups:
                        g['lr'] = self.lr
                first = False
                if actor_extra is not None:
                    actor_extra().backward()
                torch.nn.utils.clip_grad_norm_(self.actor_params, cfg.max_grad_norm)
                self.actor_opt.step()
        return {'kl': sum(kls) / len(kls) if kls else 0.0, 'kls': kls, 'value_loss': float(v_loss)}
