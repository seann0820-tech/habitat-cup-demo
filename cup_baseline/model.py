"""Learned (or optional residual) dynamics ensemble + fixed-budget CEM/MPC.

This is an interpretable model-based RL baseline, not a PETS reproduction.
MLPs learn transitions from real Habitat data. The optional residual control
adds known position-control kinematics. No supervised optimal-action labels.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch
from torch import nn
from .geometry import detour, clearance

torch.set_num_threads(1)


def wrap(x):
    return (x+np.pi) % (2*np.pi)-np.pi


def body_to_world(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([c*v[..., 0]+s*v[..., 2], v[..., 1], -s*v[..., 0]+c*v[..., 2]], -1)


def world_to_body(v, theta):
    return body_to_world(v, -theta)


def prior(x, a):
    """Free-space position-servo prior. Does not query the simulator or navmesh."""
    y = x.copy()
    theta = wrap(x[..., 2]+a[..., 1]*0.35)
    dx = a[..., 0]*0.18*np.cos(theta)
    dz = -a[..., 0]*0.18*np.sin(theta)
    rel = x[..., 3:6]-np.stack([x[..., 0], np.zeros_like(theta), x[..., 1]], -1)
    rel = body_to_world(rel, a[..., 1]*0.35)
    y[..., 0] += dx; y[..., 1] += dz; y[..., 2] = theta
    y[..., 3:6] = rel + np.stack([y[..., 0], np.zeros_like(theta), y[..., 1]], -1)
    y[..., 3:6] += body_to_world(a[..., 2:]*0.055, theta)
    return y


def features(x, a, joints, rays, phase, context=None):
    b = np.stack([x[..., 0], np.zeros_like(x[..., 0]), x[..., 1]], -1)
    rel = world_to_body(x[..., 3:6]-b, x[..., 2])/2
    if context is None:context=np.zeros((*x.shape[:-1],8),dtype=np.float32)
    return np.concatenate([rel, joints/np.pi, rays/3, phase, a, context], -1).astype(np.float32)


def task_context(obs):
    extra=np.zeros(8,dtype=np.float32);extra[int(obs.get('task_id',0))]=1
    boxes=np.asarray(obs.get('arm_boxes',[])).reshape(-1,2,3)
    if len(boxes):
        centers=boxes.mean(1);i=np.argmin(np.linalg.norm(centers-obs['x'][3:6],axis=1))
        extra[2:5]=world_to_body(centers[i]-obs['x'][3:6],obs['x'][2])/3
        extra[5:]=(boxes[i,1]-boxes[i,0])/3
    return extra


@dataclass
class Compute:
    nn_forward_calls: int = 0
    nn_sample_forwards: int = 0
    nn_flops_est: int = 0
    planner_calls: int = 0
    planner_seconds: float = 0.0
    meta_nn_forward_calls: int = 0
    meta_nn_sample_forwards: int = 0
    meta_nn_flops_est: int = 0


class Ensemble:
    def __init__(self, seed=0, members=3, hidden=64, replay_limit=4096, mode='residual'):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.members, self.hidden, self.replay_limit = members, hidden, replay_limit
        if mode not in ('learned', 'residual'):
            raise ValueError('Unknown dynamics mode')
        self.mode = mode
        self.collect_with_prior = False
        self.monitor = None
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(38, hidden), nn.ReLU(),
                                               nn.Linear(hidden, hidden), nn.ReLU(),
                                               nn.Linear(hidden, 6)) for _ in range(members)])
        # Zero prediction before data: zero-shot performance is always reported.
        for net in self.nets:
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)
        self.optimizers = [torch.optim.Adam(net.parameters(), lr=1e-3) for net in self.nets]
        self.inputs, self.targets = [], []
        self.count = Compute()
        self.train_sample_forwards = self.train_flops_est = self.train_updates = 0
        self.scale = np.array([0.18, 0.18, 0.35, 0.10, 0.10, 0.10] if mode == 'residual'
                              else [0.18, 0.18, 0.35, 0.30, 0.15, 0.30], np.float32)
        self.flops_one = sum(2*m.in_features*m.out_features+m.out_features
                             for m in self.nets[0] if isinstance(m, nn.Linear))+2*hidden

    def add(self, obs, a, nxt):
        x = obs['x'][None]
        f = features(x, np.asarray(a)[None], obs['joints'][None], obs['rays'][None], np.eye(3)[[obs['phase']]],task_context(obs)[None])[0]
        p = prior(x, np.asarray(a)[None])[0] if self.mode == 'residual' else x[0].copy()
        delta = nxt['x']-p
        delta[2] = wrap(delta[2])
        base_delta = world_to_body(np.array([delta[0], 0, delta[1]]), obs['x'][2])
        residual = np.r_[base_delta[[0, 2]], delta[2], world_to_body(delta[3:6], obs['x'][2])]
        self.inputs.append(f); self.targets.append((residual/self.scale).astype(np.float32))
        if len(self.inputs) > self.replay_limit:
            del self.inputs[:len(self.inputs)-self.replay_limit]
            del self.targets[:len(self.targets)-self.replay_limit]

    def fit(self, updates=40, batch=128):
        if len(self.inputs) < 32:
            return None
        x = torch.from_numpy(np.asarray(self.inputs))
        y = torch.from_numpy(np.asarray(self.targets))
        losses = []
        for net, opt in zip(self.nets, self.optimizers):
            net.train()
            # Each member has its own fixed bootstrap sample for this fitting pass.
            boot = self.rng.integers(len(x), size=len(x))
            for _ in range(updates):
                ix = boot[self.rng.integers(len(boot), size=batch)]
                pred = net(x[ix])
                loss = nn.functional.smooth_l1_loss(pred, y[ix])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                losses.append(float(loss.detach()))
                self.train_sample_forwards += batch
                # Approximate forward+backward MLP cost; Adam and data movement excluded.
                self.train_flops_est += int(3*batch*self.flops_one)
                self.train_updates += 1
            net.eval()
        if self.monitor is not None:
            self.monitor.fit()
        return float(np.mean(losses))

    @torch.no_grad()
    def predict(self, x, a, obs):
        n = len(x)
        if self.collect_with_prior:
            # Explicit model-based warm-up controller; its real interactions count
            # toward sample budgets. Never enabled during validation or test.
            return prior(x, a), np.zeros(n, dtype=np.float32)
        f = features(x, a, np.broadcast_to(obs['joints'], (n, 7)),
                     np.broadcast_to(obs['rays'], (n, 12)), np.broadcast_to(np.eye(3)[obs['phase']], (n, 3)),np.broadcast_to(task_context(obs),(n,8)))
        inp = torch.from_numpy(f)
        residuals = np.stack([net(inp).numpy() for net in self.nets])
        self.count.nn_forward_calls += self.members
        self.count.nn_sample_forwards += self.members*n
        self.count.nn_flops_est += self.members*n*self.flops_one
        # Bounded residual prevents unsupported long-horizon extrapolation.
        bound = 1.0 if self.mode == 'residual' else 2.0
        r = np.clip(residuals.mean(0), -bound, bound)*self.scale
        y = prior(x, a) if self.mode == 'residual' else x.copy()
        b = body_to_world(np.stack([r[:, 0], np.zeros(n), r[:, 1]], -1), x[:, 2])
        y[:, 0] += b[:, 0]; y[:, 1] += b[:, 2]
        y[:, 2] = wrap(y[:, 2]+r[:, 2])
        y[:, 3:6] += body_to_world(r[:, 3:6], x[:, 2])
        uncertainty = (residuals.var(0)*self.scale**2).sum(-1)
        return y, uncertainty

    def snapshot(self):
        return dict(weights=self.nets.state_dict(), optimizers=[o.state_dict() for o in self.optimizers],
                    inputs=self.inputs, targets=self.targets, rng=self.rng.bit_generator.state,
                    train_sample_forwards=self.train_sample_forwards, train_flops_est=self.train_flops_est,
                    train_updates=self.train_updates, members=self.members, hidden=self.hidden,
                    replay_limit=self.replay_limit, mode=self.mode,
                    metacognition=self.monitor.snapshot() if self.monitor is not None else None)

    def restore(self, state):
        if (state.get('metacognition') is not None) != (self.monitor is not None):
            raise ValueError('Agent/checkpoint mismatch: baseline and metacognitive agents cannot resume each other')
        if self.monitor is not None:
            self.monitor.restore(state['metacognition'])
        if state.get('mode', 'residual') != self.mode:
            raise ValueError('Dynamics mode does not match checkpoint')
        if state['weights']['0.0.weight'].shape[1]!=38:
            raise ValueError('Old single-task checkpoint is incompatible; train home_two_tasks_v2 in a new output directory')
        self.replay_limit=state['replay_limit']
        self.nets.load_state_dict(state['weights'])
        for opt, s in zip(self.optimizers, state['optimizers']):
            opt.load_state_dict(s)
        self.inputs, self.targets = state['inputs'], state['targets']
        self.rng.bit_generator.state = state['rng']
        for name in ('train_sample_forwards', 'train_flops_est', 'train_updates'):
            setattr(self, name, state[name])


class MPC:
    def __init__(self, model, horizon=8, candidates=64, iterations=2, elite=8, risk=0.2):
        self.model = model
        self.horizon, self.candidates, self.iterations, self.elite = horizon, candidates, iterations, elite
        self.risk = risk
        self.reset(0)

    def reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.mean = np.zeros((self.horizon, 5), dtype=np.float32)
        self.prev_phase = -1

    def guide(self, x, obs):
        a = np.zeros(5, dtype=np.float32)
        if obs['phase'] == 0:
            d = obs['nav_goal']-x[:2]
            dist = np.linalg.norm(d)
            temporary=obs.get('temporary_nav_goal',False)
            goal = obs['cup'][[0, 2]]-x[:2] if dist < 0.20 and not temporary else d
            theta = np.arctan2(-goal[1], goal[0])
            angle = wrap(theta-x[2])
            a[1] = np.clip(angle/0.35, -1, 1)
            a[0] = min(1.0, dist/0.18)*max(0, np.cos(angle)) if dist > (.025 if temporary else .13) else 0
        else:
            d = world_to_body(obs['ee_goal']-x[3:6], x[2])
            a[2:] = np.clip(d/0.055, -1, 1)
        return a

    def act(self, obs, exploration=0.0):
        t0 = time.perf_counter()
        obs=obs.copy()
        boxes=np.asarray(obs.get('arm_boxes',[])).reshape(-1,2,3)
        if len(boxes):
            if obs['phase']==0:
                low=boxes[boxes[:,0,1]<1.4]
                waypoint=detour(obs['x'][:2],obs['nav_goal'],low[:,:,[0,2]],.32)
                obs['temporary_nav_goal']=bool(np.linalg.norm(waypoint-obs['nav_goal'])>.001)
                obs['nav_goal']=waypoint
            else:
                obs['ee_goal']=detour(obs['x'][3:6],obs['ee_goal'],boxes,.06,vertical=True)
        phase = obs['phase']
        guided = []
        xg = obs['x'][None].copy()
        for _ in range(self.horizon):
            a = self.guide(xg[0], obs)
            guided.append(a); xg = prior(xg, a[None])
        guided = np.asarray(guided)
        if self.prev_phase != phase:
            self.mean = guided.copy()
        else:
            self.mean = 0.5*self.mean+0.5*guided
        std = np.full_like(self.mean, 0.65)
        best = guided
        for _ in range(self.iterations):
            seq = np.clip(self.rng.normal(self.mean, std, (self.candidates, self.horizon, 5)), -1, 1).astype(np.float32)
            seq[0] = guided
            if phase == 0:
                seq[:, :, 2:] = 0
            else:
                seq[:, :, :2] = 0
            x = np.broadcast_to(obs['x'], (self.candidates, 6)).copy()
            cost = np.zeros(self.candidates)
            for h in range(self.horizon):
                x, unc = self.model.predict(x, seq[:, h], obs)
                if phase == 0:
                    dist = np.linalg.norm(x[:, :2]-obs['nav_goal'], axis=1)
                    face = (obs['nav_goal'] if obs.get('temporary_nav_goal',False) else obs['cup'][[0, 2]])-x[:, :2]
                    angle = np.abs(wrap(np.arctan2(-face[:, 1], face[:, 0])-x[:, 2]))
                    c = 5*dist + 0.8*np.exp(-dist*3)*angle
                    if len(obs['obstacles']):
                        near = np.linalg.norm(x[:, None, :2]-obs['obstacles'][None], axis=2).min(1)
                        c += 60*np.maximum(0, 0.45-near)**2
                else:
                    c = 15*np.linalg.norm(x[:, 3:6]-obs['ee_goal'], axis=1)
                    c += 200*np.maximum(0,.08-clearance(x[:,3:6],boxes,.04))**2
                c += self.risk*unc + 0.02*np.sum(seq[:, h]**2, axis=1)
                cost += (0.98**h)*c
            elite = np.argsort(cost)[:self.elite]
            best = seq[elite[0]]
            self.mean = seq[elite].mean(0)
            std = np.maximum(seq[elite].std(0), 0.12)
        self.mean = np.concatenate([best[1:], best[-1:]], axis=0)
        self.prev_phase = phase
        action = best[0].copy()
        if exploration:
            action += self.rng.normal(0, exploration, 5)
        if phase == 0:
            action[2:] = 0
        else:
            action[:2] = 0
        self.model.count.planner_calls += 1
        self.model.count.planner_seconds += time.perf_counter()-t0
        return np.clip(action, -1, 1)
