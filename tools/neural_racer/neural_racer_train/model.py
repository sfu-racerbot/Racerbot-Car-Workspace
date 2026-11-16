"""The networks (plan A4): the deployable student policy, the privileged
teacher, and the privileged critic.

PolicyNet's parameter names are exactly the keys src/neural_racer's
policy_numpy.py reads, so export.py copies the state_dict straight into
weights.npz. tests/test_export_golden.py holds the two forward passes to 1e-5.

Actions: the networks output pre-squash values a; the car applies
steering = 0.33 tanh(a0), speed = v_cap sigmoid(a1). PPO samples and stores
the pre-squash value, so the squash's Jacobian cancels in every ratio. From
teacher v4 the teacher's speed squash also holds the wrapper's curvature cap
(capped_squash); the student keeps the plain one, and the car's wrapper caps it.
"""
import math

import torch
from torch import nn

from neural_racer import observation as obs
from neural_racer.safety_wrapper import WrapperConfig
from neural_racer_train.gpu_sim.params import CONTROL_DT
from neural_racer_train.gpu_sim.safety_torch import curvature_speed
from neural_racer_train.gpu_sim.sim import PRIV_A_LAT, PRIV_BASE, PRIV_DIM, PRIV_OTHERS, PRIV_PLACE, PRIV_VIS

FULL = ((32, 64, 64, 64), 256)
HALF = ((16, 32, 32, 32), 128)
WIDE = ((48, 96, 96, 96), 384)        # 1.5x FULL: plan item 2.6's upper bracket
WIDTHS = {'full': FULL, 'half': HALF, 'wide': WIDE}
GRU = 128


class PolicyNet(nn.Module):
    def __init__(self, width=FULL, recurrent: bool = True):
        super().__init__()
        (c0, c1, c2, c3), fc = width
        self.width, self.recurrent = width, recurrent
        self.scan = nn.ModuleDict({
            'c0': nn.Conv1d(4, c0, 7, 2, 3), 'c1': nn.Conv1d(c0, c1, 5, 2, 2),
            'c2': nn.Conv1d(c1, c2, 5, 2, 2), 'c3': nn.Conv1d(c2, c3, 3, 2, 1),
            'fc': nn.Linear(c3 * 17, fc)})
        self.ego = nn.ModuleDict({'fc0': nn.Linear(obs.EGO_DIM, 64), 'fc1': nn.Linear(64, 64)})
        self.prev = nn.ModuleDict({'c0': nn.Conv1d(4, 32, 5, 2, 2), 'c1': nn.Conv1d(32, 32, 3, 2, 1),
                                   'fc': nn.Linear(256, 64)})
        self.race = nn.ModuleDict({'fc': nn.Linear(obs.RACE_DIM, 32)})
        ln = fc + 64 + 64 + 32
        self.fuse = nn.ModuleDict({'ln': nn.LayerNorm(ln), 'fc': nn.Linear(ln + obs.CAMERA_DIM, 256)})
        # the MLP ablation (main1_mlp) keeps the same names: a GRUCell whose
        # hidden input is always zero is a feed-forward layer with gates
        self.gru = nn.GRUCell(256, GRU)
        self.head = nn.ModuleDict({'fc0': nn.Linear(GRU, 128), 'act': nn.Linear(128, 2)})
        self.aux = nn.ModuleDict({'path': nn.Linear(GRU, 2 * obs.AUX_PATH_POINTS),
                                  'opp': nn.Linear(GRU, obs.OPP_SLOTS * (1 + 2 * obs.AUX_PATH_POINTS))})
        self.log_std = nn.Parameter(torch.full((2,), -1.0))     # training only; never exported
        with torch.no_grad():
            # v1 contract (A4/A5): everything the camera reaches starts at
            # exactly zero, so enabling a camera later changes nothing until
            # fine-tuning moves it.
            self.fuse['fc'].weight[:, ln:] = 0.0
            self.ego['fc0'].weight[:, obs.EGO['camera_present']:obs.EGO['camera_age'] + 1] = 0.0

    def forward(self, o: dict, h):
        """o: blocks with a leading batch dim. Returns (a (N, 2), h', path (N, 6, 2) m, opp (N, 4, 13))."""
        h = self.recur(self.encode(o), h)
        a, path, opp = self.heads(h)
        return a, h, path, opp

    def encode(self, o: dict):
        """Everything before the memory: the scan, ego, preview and race
        encoders and the fusion layer, (N, 256). No state, so a training
        update runs it over every timestep of a rollout at once (item 1.4)."""
        act = nn.functional.silu
        ego = o['ego']
        mode = ego[:, obs.EGO['mode']:obs.EGO['mode'] + 1]
        cam = ego[:, obs.EGO['camera_present']:obs.EGO['camera_present'] + 1]
        x = o['scan']
        for k in ('c0', 'c1', 'c2', 'c3'):
            x = act(self.scan[k](x))
        z_scan = act(self.scan['fc'](x.flatten(1)))
        z_ego = act(self.ego['fc1'](act(self.ego['fc0'](ego))))
        y = act(self.prev['c1'](act(self.prev['c0'](o['preview']))))
        z_prev = act(self.prev['fc'](y.flatten(1))) * mode
        z_race = act(self.race['fc'](o['race'])) * mode
        z = self.fuse['ln'](torch.cat([z_scan, z_ego, z_prev, z_race], dim=1))
        return act(self.fuse['fc'](torch.cat([z, o['camera'] * cam], dim=1)))

    def recur(self, u, h):
        """One memory step (the MLP ablation's hidden input is always zero)."""
        if not self.recurrent:
            h = torch.zeros_like(h)
        return gru_cell(u, h, self.gru)

    def heads(self, h):
        """(a (N, 2), path (N, 6, 2) m, opp (N, 4, 13)) from the memory's output."""
        a = self.head['act'](nn.functional.silu(self.head['fc0'](h)))
        path = self.aux['path'](h).view(-1, obs.AUX_PATH_POINTS, 2) * obs.AUX_POS_SCALE
        opp = self.aux['opp'](h).view(-1, obs.OPP_SLOTS, 1 + 2 * obs.AUX_PATH_POINTS)
        return a, path, opp

    def unroll(self, o: dict, h0, start):
        """A whole sequence: o blocks (T, N, ...), h0 (N, 128), start (T, N)
        bool (a new episode began at that step: the memory restarts from zero).
        The encoders run once over all T x N inputs; only the GRU steps in
        order. Returns (a (T, N, 2), path, opp, h (T, N, 128)), equal to T
        calls of forward()."""
        T, N = start.shape
        u = self.encode({k: v.reshape(T * N, *v.shape[2:]) for k, v in o.items()}).view(T, N, -1)
        h, hs = h0, []
        for t in range(T):
            h = self.recur(u[t], h * (~start[t]).to(h.dtype).unsqueeze(1))
            hs.append(h)
        H = torch.stack(hs)
        a, path, opp = self.heads(H.reshape(T * N, -1))
        return a.view(T, N, 2), path.view(T, N, *path.shape[1:]), opp.view(T, N, *opp.shape[1:]), H


class RunningNorm(nn.Module):
    """Running mean and variance (Chan et al.'s parallel update), kept as
    buffers so checkpoints carry them. update() is called by the trainer on
    rollout data only; forward() never changes the statistics."""

    def __init__(self, dim: int, clip: float = 10.0, min_std: float = 0.05):
        super().__init__()
        self.clip, self.min_std = clip, min_std
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, self.mean.shape[0]).float()
        n = x.shape[0]
        b_mean, b_var = x.mean(0), x.var(0, unbiased=False)
        delta, total = b_mean - self.mean, self.count + n
        m2 = self.var * self.count + b_var * n + delta ** 2 * self.count * n / total
        self.mean.add_(delta * n / total)
        self.var.copy_(m2 / total)
        self.count.fill_(float(total))

    @torch.no_grad()
    def set_columns(self, cols, x):
        """Statistics of the columns `cols` straight from the sample x (N, dim):
        for inputs a checkpoint never saw (load_expanded)."""
        x = x.reshape(-1, self.mean.shape[0]).float()
        self.mean[cols] = x[:, cols].mean(0)
        self.var[cols] = x[:, cols].var(0, unbiased=False)

    def std(self):
        return torch.sqrt(self.var).clamp(min=self.min_std)

    def forward(self, x):
        return ((x - self.mean) / self.std()).clamp(-self.clip, self.clip)

    def denorm(self, y):
        return y * self.std() + self.mean


TRACK_POINTS = 64       # sim.privileged(): 64 centreline points, 0.25 m apart, as x | y | w_left | w_right
TIME_LEFT = PRIV_BASE - 1   # sim.privileged()'s share of the episode limit still to run


class PrivEncoder(nn.Module):
    """The privileged state, encoded: the track preview by a 1-D convolution
    along the lap (a bend reads the same wherever it is in the window), with
    each point's known flag as a fifth channel; the rest -- car state,
    parameters, the commands in the delay lines, the cornering cap -- by an
    MLP on running-normalised inputs. The preview is already in metres/16
    and so passes un-normalised: per-point statistics would undo the
    convolution's sharing. The time left is masked out: the episode limit is
    a training artefact (bootstrapped, Pardo et al. 2018), and a 3-lap race
    has none."""

    def __init__(self, out: int = 256):
        super().__init__()
        rest = PRIV_VIS - 4 * TRACK_POINTS
        keep = torch.ones(rest)
        keep[TIME_LEFT - 4 * TRACK_POINTS] = 0.0
        self.register_buffer('keep', keep)
        self.norm = RunningNorm(rest)
        act = nn.SiLU
        self.track = nn.Sequential(nn.Conv1d(5, 32, 5, padding=2), act(),
                                   nn.Conv1d(32, 64, 5, stride=2, padding=2), act(),     # 32 points
                                   nn.Conv1d(64, 64, 5, stride=2, padding=2), act())     # 16 points
        self.track_fc = nn.Linear(64 * TRACK_POINTS // 4, 256)
        self.state = nn.Sequential(nn.Linear(rest, 128), act(), nn.Linear(128, 128), act())
        self.fuse = nn.Sequential(nn.LayerNorm(256 + 128), nn.Linear(256 + 128, out), act())

    def update_norm(self, priv):
        self.norm.update(priv[..., 4 * TRACK_POINTS:PRIV_VIS] * self.keep)

    def forward(self, priv):
        t = torch.cat([priv[:, :4 * TRACK_POINTS].reshape(-1, 4, TRACK_POINTS),
                       priv[:, PRIV_VIS:PRIV_VIS + TRACK_POINTS].unsqueeze(1)], dim=1)
        z_t = nn.functional.silu(self.track_fc(self.track(t).flatten(1)))
        z_s = self.state(self.norm(priv[:, 4 * TRACK_POINTS:PRIV_VIS] * self.keep))
        return self.fuse(torch.cat([z_t, z_s], dim=1))


class TeacherNet(nn.Module):
    """Stage 1: privileged state in, the same pre-squash actions out. No memory:
    its input carries the actuator delay lines, so it is Markov. The last
    layer starts 100x small (Andrychowicz et al. 2021): the first actions are
    near the centre of the squash, wherever the inputs are."""

    def __init__(self):
        super().__init__()
        self.enc = PrivEncoder(256)
        self.head = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2))
        with torch.no_grad():
            self.head[-1].weight.mul_(0.01)
            self.head[-1].bias.zero_()
        self.log_std = nn.Parameter(torch.full((2,), math.log(0.3)))

    def update_norm(self, priv):
        self.enc.update_norm(priv)

    def forward(self, priv):
        return self.head(self.enc(priv))


class Critic(nn.Module):
    """The privileged value function (asymmetric actor-critic). Never ships.
    The trainer may regress it on normalised returns (ppo.ValueNorm)."""

    def __init__(self):
        super().__init__()
        self.enc = PrivEncoder(512)
        self.head = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 1))

    def update_norm(self, priv):
        self.enc.update_norm(priv)

    def forward(self, priv):
        return self.head(self.enc(priv)).squeeze(-1)


def gru_cell(x, h, cell: nn.GRUCell):
    """nn.GRUCell's equations from its own parameters, written as plain matrix
    products: identical results, but vmap can batch it (it has no batching
    rule for aten::gru_cell and falls back to a per-model loop)."""
    gi = nn.functional.linear(x, cell.weight_ih, cell.bias_ih)
    gh = nn.functional.linear(h, cell.weight_hh, cell.bias_hh)
    i_r, i_z, i_n = gi.chunk(3, dim=-1)
    h_r, h_z, h_n = gh.chunk(3, dim=-1)
    r = torch.sigmoid(i_r + h_r)
    z = torch.sigmoid(i_z + h_z)
    n = torch.tanh(i_n + r * h_n)
    return (1 - z) * n + z * h


def squash(a, v_cap):
    """Pre-squash actions -> (steering rad, speed m/s), as the car applies them."""
    return torch.stack([obs.STEER_LIMIT * torch.tanh(a[..., 0]), v_cap * torch.sigmoid(a[..., 1])], dim=-1)


def capped_squash(a, v_cap, priv, cfg: WrapperConfig = WrapperConfig()):
    """The teacher's squash from v4 (plan item 2.1): the speed spans 0 to the
    lower of v_cap and the wrapper's curvature cap at the steering the car
    will actually apply this tick -- the command after the clip and the slew
    limit from the last applied steering (sim.privileged()'s newest delay-line
    entry). So the cap never has to cut what the teacher asks for. The
    race's cornering cap is read from the privileged state too."""
    steer = obs.STEER_LIMIT * torch.tanh(a[..., 0])
    prev = priv[..., PRIV_BASE] * obs.STEER_LIMIT
    step = cfg.max_steering_rate * min(CONTROL_DT, cfg.command_slew_max_dt)
    applied = torch.minimum(torch.maximum(steer.clamp(-cfg.max_steering_angle, cfg.max_steering_angle),
                                          prev - step), prev + step)
    top = torch.minimum(v_cap, curvature_speed(applied, cfg.wheelbase, priv[..., PRIV_A_LAT] * 5))
    return torch.stack([steer, top * torch.sigmoid(a[..., 1])], dim=-1)


SQUASHES = {'plain': squash, 'capped': lambda a, v_cap, priv: capped_squash(a, v_cap, priv)}


def teacher_squash(mode: str):
    """(a, v_cap, priv) -> commands, for a teacher checkpoint's 'squash' entry
    (absent before v4: plain)."""
    if mode == 'plain':
        return lambda a, v_cap, priv: squash(a, v_cap)
    return SQUASHES[mode]


def load_expanded(module: nn.Module, state: dict) -> list:
    """Load a state dict saved when the privileged state was narrower (before
    the cornering cap and the track flags were appended): the inputs it lacks
    -- weight columns, or a convolution's input channel -- get zero weights,
    so the network computes exactly what it did, whatever the new inputs hold;
    their normaliser statistics are left at the module's own (set them from
    data with RunningNorm.set_columns). Only an input dimension may grow: a
    linear or conv weight's dim 1, a 1-D buffer's dim 0. Returns the keys widened."""
    own = module.state_dict()
    out, widened = {}, []
    for k, v in state.items():
        mine = own[k]
        if v.shape == mine.shape:
            out[k] = v
            continue
        grown = [d for d in range(mine.dim()) if v.dim() == mine.dim() and v.shape[d] != mine.shape[d]]
        ok = v.dim() == mine.dim() and len(grown) == 1 and v.shape[grown[0]] < mine.shape[grown[0]] \
            and grown[0] == (0 if v.dim() == 1 else 1)
        if not ok:
            raise ValueError(f'{k}: {tuple(v.shape)} does not widen to {tuple(mine.shape)}')
        w = torch.zeros_like(mine) if k.endswith('.weight') else mine.clone()   # buffers: the module's own beyond
        w[tuple(slice(0, n) for n in v.shape)] = v
        out[k] = w
        widened.append(k)
    module.load_state_dict(out)
    return widened


def mask_track(priv, vis, masked):
    """The teacher's input as a student could know the track on a lap it has
    not mapped (plan item 2.3): for the rows in `masked` (N,), the track
    points the LiDAR cannot see (vis (N, 64) False) are zeroed, flag included."""
    hide = (~vis) & masked.unsqueeze(1)
    keep = (~hide).to(priv.dtype)
    out = priv.clone()
    for ch in range(4):
        out[:, ch * TRACK_POINTS:(ch + 1) * TRACK_POINTS] *= keep
    out[:, PRIV_VIS:PRIV_VIS + TRACK_POINTS] = keep
    return out


def solo_view(priv):
    """The privileged state as a teacher trained alone (teacher.py races one
    car) saw it: no other cars, first of one. In a sim with seats for more
    cars, privileged() reports the empty seats' cars (parked at the origin)
    and any real opponents, inputs that teacher's normaliser only ever saw at
    zero and so clips: alone in a 4-seat sim it stalled in 35 of 40 races,
    and in 4 of 40 with this view (2026-10-30 audit)."""
    out = priv.clone()
    out[..., PRIV_OTHERS] = 0.0
    out[..., PRIV_PLACE] = 0.25
    return out


def fresh_opponent_inputs(enc: PrivEncoder, priv, max_count: float = 1e7) -> None:
    """For a privileged encoder trained alone (the teacher's critic) that will
    now see other cars: its normaliser statistics for the opponent block and
    the place, from the sample priv (N, PRIV_DIM), and zero first-layer
    weights for them -- they only ever saw a constant, so their weights never
    trained, and the encoder starts computing exactly what it did. The
    sample count is capped so the statistics keep adapting (a teacher's is
    ~3e9 samples: they would otherwise never move)."""
    base = 4 * TRACK_POINTS
    cols = list(range(PRIV_OTHERS.start - base, PRIV_OTHERS.stop - base)) + [PRIV_PLACE - base]
    with torch.no_grad():
        enc.norm.set_columns(cols, priv[..., base:PRIV_VIS] * enc.keep)
        enc.norm.count.clamp_(max=max_count)
        enc.state[0].weight[:, cols] = 0.0


def gaussian_logp(a, mean, log_std):
    var = torch.exp(2 * log_std)
    return (-(a - mean) ** 2 / (2 * var) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)


def flat_obs(o: dict, mask=None):
    """(E, C, ...) blocks -> (N, ...) for the cars in mask (default: all)."""
    if mask is None:
        return {k: v.reshape(-1, *v.shape[2:]) for k, v in o.items()}
    return {k: v[mask] for k, v in o.items()}
