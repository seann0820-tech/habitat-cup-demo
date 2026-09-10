"""Learn process reliability and the value of an additional MPC computation.

This module needs only NumPy. A shared MLP has three diagnostic outputs
(phase completion, next-action collision, and normalized prediction error) and
four action-value outputs. An action value is a bootstrapped estimate of future
REAL task reward minus explicitly counted computation and execution costs.
Unselected actions never receive invented outcome labels.

Every fifth collected episode is reserved for probability calibration. Its
examples never update the shared MLP or action values. Calibration is a small,
regularized affine transformation of each diagnostic logit, and is used only
when enough independent episodes and both outcome classes have been observed.
Evaluation may call predict/choose, but must not call add_episode or fit.
"""
import copy
import numpy as np

ACTIONS = ('execute', 'expand', 'restart', 'baseline')
FEATURES = ('task_cup', 'task_drawer', 'phase_nav', 'phase_reach', 'phase_finish',
    'nav_distance', 'ee_distance', 'time_used', 'replay_fraction',
    'error_ema', 'error_trend', 'last_error', 'disagreement', 'search_gain',
    'cost_gap', 'last_action_change', 'stalled_steps', 'collision_last',
    'goal_progress', 'phase_changed', 'current_best_cost', 'learning_age',
    'search_fraction', 'execution_surprise', 'revisit', 'waypoint_progress',
    'actual_displacement', 'remaining_steps')
META_SCHEMA = 'meta_process_v2'


def sigmoid(x):
    """Stable logistic link; this transformation alone does not ensure calibration."""
    return 1. / (1. + np.exp(-np.clip(x, -30, 30)))


def softplus(x):
    return np.maximum(x, 0) + np.log1p(np.exp(-np.abs(x)))


def binary_metrics(probabilities, outcomes):
    """Descriptive probability quality, not an intrinsic metacognitive parameter."""
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if not len(p):
        return dict(n=0, brier=None, ece=None, discrimination_auc=None, bins=[])
    bins = []; ece = 0.
    for i in range(5):
        selected = (p >= i / 5) & ((p < (i + 1) / 5) if i < 4 else (p <= 1))
        n = int(selected.sum())
        if n:
            confidence = float(p[selected].mean()); frequency = float(y[selected].mean())
            ece += n / len(p) * abs(confidence - frequency)
            bins.append(dict(low=i/5, high=(i+1)/5, n=n,
                             confidence=confidence, success_rate=frequency))
    # Average ranks handle ties without a quadratic pairwise probability matrix.
    positive_n, negative_n = int((y == 1).sum()), int((y == 0).sum())
    auc = None
    if positive_n and negative_n:
        order = np.argsort(p, kind='stable')
        _, starts, counts = np.unique(p[order], return_index=True, return_counts=True)
        ranks = np.repeat(starts + (counts + 1)/2, counts)
        positive_rank_sum = ranks[y[order] == 1].sum()
        auc = float((positive_rank_sum-positive_n*(positive_n+1)/2)/(positive_n*negative_n))
    high = p >= .85
    return dict(n=len(p), brier=float(np.mean((p-y)**2)), ece=float(ece),
        discrimination_auc=auc, bins=bins, high_confidence_n=int(high.sum()),
        high_confidence_failures=int(((y == 0) & high).sum()))


class MetaMonitor:
    """Small process monitor with separate calibration and metacontrol training.

    The last layer is ordered as phase logit, collision logit, error preactivation,
    and the four action values in ACTIONS order. Parameters are trained with
    clipped SGD. The target network uses a Polyak update after each SGD step.
    There is no stochastic inference or mutation during predict/choose.
    """
    def __init__(self, seed=0, hidden=32, capacity=512, compute_weight=.10,
                 step_weight=.05, calibration_capacity=128):
        if (hidden < 1 or capacity < 8 or calibration_capacity < 8 or
                not np.all(np.isfinite([compute_weight, step_weight])) or
                min(compute_weight, step_weight) < 0):
            raise ValueError('Invalid monitor dimensions, capacity, or cost weights')
        self.rng = np.random.default_rng(seed + 9127)
        self.hidden, self.capacity = int(hidden), int(capacity)
        self.calibration_capacity = int(calibration_capacity)
        self.compute_weight, self.step_weight = float(compute_weight), float(step_weight)
        sizes = (len(FEATURES), hidden, hidden, 3 + len(ACTIONS))
        self.weights = [self.rng.normal(0, np.sqrt(2./a), (a, b)).astype(np.float32)
                        for a, b in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.weights[-1] *= .05
        self.biases[-1][2] = -2.  # A small positive initial normalized-error prediction.
        self.target_weights = copy.deepcopy(self.weights)
        self.target_biases = copy.deepcopy(self.biases)
        self.flops_one = int(sum(2*a*b+b for a, b in zip(sizes[:-1], sizes[1:])) + 2*hidden)
        self.samples, self.calibration_samples = [], []
        self.training_episodes = self.shared_training_episodes = 0
        self.train_flops_est = self.train_sample_forwards = self.train_updates = 0
        self.target_sample_forwards = self.calibration_sample_forwards = 0
        self.calibration_flops_est = self.calibration_updates = 0
        self.action_counts = np.zeros(len(ACTIONS), dtype=np.int64)
        self.last_loss = None
        self.calibration = {
            key: dict(log_scale=0., bias=0., ready=False, n=0, episodes=0,
                      raw_brier=None, calibrated_brier=None)
            for key in ('phase', 'collision')}

    @property
    def ready(self):
        """Training coverage gate only; it is not a claim of accurate confidence."""
        return (self.shared_training_episodes >= 4 and self.train_updates > 0
                and bool(np.all(self.action_counts >= 4)))

    def forward(self, features, target=False):
        x = np.asarray(features, np.float32).reshape(-1, len(FEATURES))
        if not np.all(np.isfinite(x)):
            raise ValueError('Monitor features must be finite')
        weights = self.target_weights if target else self.weights
        biases = self.target_biases if target else self.biases
        h1 = np.maximum(0, x @ weights[0] + biases[0])
        h2 = np.maximum(0, h1 @ weights[1] + biases[1])
        z = h2 @ weights[2] + biases[2]
        return x, h1, h2, z

    def _probability(self, logit, head):
        c = self.calibration[head]
        if c['ready']:
            logit = np.exp(c['log_scale']) * logit + c['bias']
        return float(sigmoid(logit))

    def predict(self, features):
        """Read-only forecast. The planner counts this one dense network forward."""
        z = self.forward(features)[-1][0]
        return dict(raw_phase_probability=float(sigmoid(z[0])),
            phase_probability=self._probability(z[0], 'phase'),
            raw_collision_probability=float(sigmoid(z[1])),
            collision_probability=self._probability(z[1], 'collision'),
            predicted_error=float(softplus(z[2])), action_values=z[3:].astype(float).tolist(),
            monitor_ready=bool(self.ready),
            phase_calibration_ready=bool(self.calibration['phase']['ready']),
            collision_calibration_ready=bool(self.calibration['collision']['ready']))

    def choose(self, features, rng, learning, allowed=None):
        """Select a legal computation; exploration uses the episode-local RNG.

        Actions are uniformly explored until the coverage gate is met, then with
        epsilon=0.15 in training. Evaluation is deterministic and performs no RNG
        draws. The Q values already contain costs through their training rewards.
        """
        mask = np.ones(len(ACTIONS), bool) if allowed is None else np.asarray(allowed, bool)
        if mask.shape != (len(ACTIONS),) or not mask.any():
            raise ValueError('At least one of the four metacontrol actions must be allowed')
        forecast = self.predict(features)
        legal = np.flatnonzero(mask)
        values = np.asarray(forecast['action_values'])
        greedy = int(legal[np.argmax(values[legal])])
        epsilon = (1. if not self.ready else .15) if learning else 0.
        action = greedy
        if epsilon and rng.random() < epsilon:
            action = int(rng.choice(legal))
        forecast.update(propensity=float(epsilon / len(legal) + (1-epsilon) * (action == greedy)),
                        allowed=mask.tolist(), exploration_probability=float(epsilon))
        return action, forecast

    def add_episode(self, trace, success, steps):
        """Store actual selected actions and their chronological TD successors.

        Each trace entry contains features, choice, reward, discount, allowed,
        step, phase and diagnostic_valid. Physical execution entries additionally
        contain phase_success, collision and prediction_error labels. Labels can
        be at the top level or in a 'diagnostics' dictionary. The final entry is
        terminal. Linkage is constructed BEFORE storing at most 16 records, so a
        downsampled record never accidentally bootstraps from a distant timestep.
        The success argument is episode metadata, not the phase diagnostic label.
        """
        if not trace:
            return
        episode_id = self.training_episodes + 1
        held_out = episode_id % 5 == 0
        records = []
        for i, entry in enumerate(trace):
            action = int(entry['choice'])
            if not 0 <= action < len(ACTIONS):
                raise ValueError('Unknown metacontrol action')
            features = np.asarray(entry['features'], np.float32).reshape(len(FEATURES)).copy()
            allowed = np.asarray(entry.get('allowed', [True]*len(ACTIONS)), bool)
            if allowed.shape != (len(ACTIONS),) or not allowed[action]:
                raise ValueError('A recorded metacontrol action was not allowed')
            done = i == len(trace)-1
            following = None if done else trace[i+1]
            labels = entry.get('diagnostics', entry)
            valid = bool(entry.get('diagnostic_valid', False))
            diagnostic = np.zeros(3, np.float32)
            if valid:
                diagnostic = np.array([labels['phase_success'], labels['collision'],
                                       labels['prediction_error']], np.float32)
                if (not np.all(np.isfinite(diagnostic)) or
                        np.any((diagnostic[:2] != 0) & (diagnostic[:2] != 1)) or diagnostic[2] < 0):
                    raise ValueError('Invalid physical-execution diagnostic labels')
            reward, discount = float(entry['reward']), float(entry['discount'])
            if not np.isfinite(reward) or not 0 <= discount <= 1:
                raise ValueError('Invalid metacontrol reward or discount')
            next_features = features.copy() if done else np.asarray(following['features'], np.float32).reshape(len(FEATURES)).copy()
            next_allowed = np.zeros(len(ACTIONS), bool) if done else np.asarray(following.get('allowed', [True]*len(ACTIONS)), bool).copy()
            if next_allowed.shape != (len(ACTIONS),) or (not done and not next_allowed.any()):
                raise ValueError('Nonterminal TD successor needs a legal action')
            if not np.all(np.isfinite(features)) or not np.all(np.isfinite(next_features)):
                raise ValueError('Non-finite metacontrol feature')
            records.append(dict(features=features, action=action, reward=reward,
                discount=discount, next_features=next_features, next_allowed=next_allowed,
                done=done, diagnostic_valid=valid, diagnostic=diagnostic,
                episode_id=episode_id, step=int(entry['step']), phase=int(entry['phase']),
                episode_success=bool(success), episode_steps=int(steps)))
        # Select calibration examples from physical execution only. Entire held-out
        # episodes are excluded from the shared network, including their Q records.
        eligible = [r for r in records if r['diagnostic_valid']] if held_out else records
        if eligible:
            indices = np.unique(np.linspace(0, len(eligible)-1, min(16, len(eligible)), dtype=int))
            selected = [eligible[i] for i in indices]
            if held_out:
                self.calibration_samples = (self.calibration_samples + selected)[-self.calibration_capacity:]
            else:
                self.samples = (self.samples + selected)[-self.capacity:]
        self.training_episodes = episode_id
        if not held_out:
            self.shared_training_episodes += 1
        self.action_counts = np.bincount([r['action'] for r in self.samples], minlength=len(ACTIONS))

    def _td_targets(self, rows):
        """Target-network bootstrap on legal actions; terminal rewards stand alone."""
        targets = np.asarray([r['reward'] for r in rows], np.float32)
        indices = [i for i, r in enumerate(rows) if not r['done']]
        if indices:
            future = self.forward(np.stack([rows[i]['next_features'] for i in indices]), target=True)[-1][:, 3:]
            masks = np.stack([rows[i]['next_allowed'] for i in indices])
            future = np.where(masks, future, -np.inf).max(axis=1)
            targets[indices] += np.asarray([rows[i]['discount'] for i in indices]) * future
            self.target_sample_forwards += len(indices)
            self.train_sample_forwards += len(indices)
            self.train_flops_est += len(indices) * self.flops_one
        return targets

    def fit(self, updates=24, batch=64, lr=.01):
        """Update from training replay, then fit scalar calibrators on held-out data."""
        if updates < 0 or batch < 1 or not np.isfinite(lr) or lr <= 0:
            raise ValueError('Invalid monitor optimizer settings')
        losses = []
        if len(self.samples) >= 16:
            for _ in range(updates):
                rows = [self.samples[i] for i in self.rng.integers(len(self.samples), size=batch)]
                targets = self._td_targets(rows)
                x, h1, h2, z = self.forward(np.stack([r['features'] for r in rows]))
                actions = np.asarray([r['action'] for r in rows])
                qerror = z[np.arange(batch), 3+actions] - targets
                qloss = np.where(np.abs(qerror) < 1, .5*qerror**2, np.abs(qerror)-.5).mean()
                grad = np.zeros_like(z)
                grad[np.arange(batch), 3+actions] = np.clip(qerror, -1, 1) / batch
                valid = np.asarray([r['diagnostic_valid'] for r in rows], bool)
                diagnostic_loss = 0.
                if valid.any():
                    labels = np.stack([r['diagnostic'] for r in rows])[valid]
                    p = sigmoid(z[valid, :2]); count = int(valid.sum())
                    bce = -(labels[:, :2]*np.log(p+1e-7) + (1-labels[:, :2])*np.log(1-p+1e-7))
                    error = softplus(z[valid, 2]) - labels[:, 2]
                    huber = np.where(np.abs(error) < 1, .5*error**2, np.abs(error)-.5)
                    diagnostic_loss = float((bce.sum(axis=1) + .5*huber).mean())
                    grad[valid, :2] = (p-labels[:, :2]) / count
                    grad[valid, 2] = .5*np.clip(error, -1, 1)*sigmoid(z[valid, 2]) / count
                g2 = grad
                gw2, gb2 = h2.T@g2, g2.sum(axis=0)
                g1 = (g2@self.weights[2].T)*(h2 > 0)
                gw1, gb1 = h1.T@g1, g1.sum(axis=0)
                g0 = (g1@self.weights[1].T)*(h1 > 0)
                gw0, gb0 = x.T@g0, g0.sum(axis=0)
                gw, gb = [gw0, gw1, gw2], [gb0, gb1, gb2]
                norm = np.sqrt(sum(float(np.sum(g*g)) for g in gw+gb))
                scale = min(1., 5./max(norm, 1e-8))
                for j in range(3):
                    self.weights[j] -= lr*scale*gw[j]
                    self.biases[j] -= lr*scale*gb[j]
                    self.target_weights[j] = .98*self.target_weights[j] + .02*self.weights[j]
                    self.target_biases[j] = .98*self.target_biases[j] + .02*self.biases[j]
                self.train_sample_forwards += batch
                self.train_flops_est += 3*batch*self.flops_one
                self.train_updates += 1
                losses.append(float(qloss + diagnostic_loss))
        if losses:
            self.last_loss = float(np.mean(losses))
        self._fit_calibration()
        return self.last_loss if losses else None

    def _fit_calibration(self):
        """Regularized monotonic logit calibration; never fit shared parameters here.

        Minimum support is 32 records from at least three held-out episodes and
        both outcome classes for the particular head. Reported fitting-set Brier
        scores are descriptive; independent evaluation remains necessary.
        """
        rows = self.calibration_samples
        episodes = len({r['episode_id'] for r in rows})
        for c in self.calibration.values():
            c.update(n=len(rows), episodes=episodes, ready=False)
        if len(rows) < 32 or episodes < 3:
            return
        logits = self.forward(np.stack([r['features'] for r in rows]))[-1][:, :2]
        self.train_sample_forwards += len(rows)
        self.calibration_sample_forwards += len(rows)
        counted = len(rows)*self.flops_one
        self.calibration_flops_est += counted; self.train_flops_est += counted
        labels = np.stack([r['diagnostic'] for r in rows])[:, :2]
        # Equal episode weight prevents long trajectories dominating calibration.
        ids = np.asarray([r['episode_id'] for r in rows])
        sample_weight = np.asarray([1./np.sum(ids == i) for i in ids], float)
        sample_weight /= sample_weight.sum()
        for j, head in enumerate(('phase', 'collision')):
            c = self.calibration[head]; y = labels[:, j]
            if len(np.unique(y)) < 2:
                continue
            raw = logits[:, j].astype(float)
            log_scale, bias = float(c['log_scale']), float(c['bias'])
            for _ in range(80):
                scale = np.exp(log_scale); p = sigmoid(scale*raw + bias)
                error = (p-y)*sample_weight
                gs = np.sum(error*scale*raw) + .02*log_scale
                gb = np.sum(error) + .02*bias
                log_scale = float(np.clip(log_scale - .05*gs, -2, 2))
                bias = float(np.clip(bias - .05*gb, -3, 3))
            c.update(log_scale=log_scale, bias=bias, ready=True,
                raw_brier=float(np.sum(sample_weight*(sigmoid(raw)-y)**2)),
                calibrated_brier=float(np.sum(sample_weight*(sigmoid(np.exp(log_scale)*raw+bias)-y)**2)))
            # Scalar elementwise work is approximate, separately itemized; all
            # network forward/backward FLOPs remain included in train_flops_est.
            scalar_flops = 80*(20*len(rows)+12)
            self.calibration_flops_est += scalar_flops; self.train_flops_est += scalar_flops
            self.calibration_updates += 80

    def snapshot(self):
        """Complete learning state; checkpoint restoration reproduces future updates."""
        state = {k: v for k, v in self.__dict__.items() if k != 'rng'}
        state.update(schema=META_SCHEMA, features=FEATURES, actions=ACTIONS,
                     rng=self.rng.bit_generator.state)
        return copy.deepcopy(state)

    def restore(self, state):
        if (state.get('schema') != META_SCHEMA or tuple(state.get('features', ())) != FEATURES
                or tuple(state.get('actions', ())) != ACTIONS or state.get('hidden') != self.hidden):
            raise ValueError('Incompatible process-monitor checkpoint; start a new v2 experiment')
        for key, value in state.items():
            if key not in ('schema', 'features', 'actions', 'rng'):
                setattr(self, key, copy.deepcopy(value))
        self.rng.bit_generator.state = copy.deepcopy(state['rng'])

    def summary(self):
        arrays = lambda records: sum(v.nbytes for r in records for v in r.values() if isinstance(v, np.ndarray))
        return dict(schema=META_SCHEMA, ready=bool(self.ready),
            readiness_definition='At least four shared-training episodes, four stored choices per action, and one update; coverage only',
            training_episodes=self.training_episodes, shared_training_episodes=self.shared_training_episodes,
            stored_records=len(self.samples), record_capacity=self.capacity,
            calibration_records=len(self.calibration_samples), calibration_capacity=self.calibration_capacity,
            record_array_bytes=int(arrays(self.samples)),
            calibration_array_bytes=int(arrays(self.calibration_samples)),
            parameter_bytes=int(sum(x.nbytes for x in self.weights+self.biases)),
            target_parameter_bytes=int(sum(x.nbytes for x in self.target_weights+self.target_biases)),
            trained_action_counts=self.action_counts.tolist(), train_flops_est=int(self.train_flops_est),
            train_sample_forwards=int(self.train_sample_forwards), train_updates=self.train_updates,
            target_sample_forwards=self.target_sample_forwards,
            calibration_sample_forwards=self.calibration_sample_forwards,
            calibration_flops_est=int(self.calibration_flops_est),
            calibration=copy.deepcopy(self.calibration),
            confidence_definition='P(current phase eventually completes within the episode | process evidence, recent behavior policy)',
            collision_definition='P(next executed physical action collides | evidence at execution decision)',
            control='Legal-action Q maximization with one-step target-network TD; recorded rewards include known computation and execution costs',
            calibration_definition='Every fifth collected training episode held out from shared-network/Q updates; scalar logit calibration only',
            caveat='Diagnostic confidence is policy dependent, not global task confidence or a causal counterfactual. Error variance is not intrinsic human metacognitive noise. Calibration fitting scores need independent evaluation.')


def monitoring_metrics(rows):
    """One ready physical-execution PHASE forecast per episode; never final success.

    The planner supplies the eventual outcome of that same phase in the same
    episode. This differs from the final task-success probability in version 1.
    """
    selected = [r for r in rows if r.get('meta_phase_confidence') is not None
                and r.get('meta_phase_success') is not None]
    result = binary_metrics([r['meta_phase_confidence'] for r in selected],
                            [r['meta_phase_success'] for r in selected])
    result.update(target='current_phase_completion',
        note='First ready physical-execution phase forecast per episode, scored against that phase outcome. 0.85 is a reporting bin, not a control gate. Episode-level descriptive metrics; no intrinsic-noise interpretation.')
    return result


def diagnostic_metrics(records):
    """Summarize frozen forecasts and their actual execution labels.

    Records are physical trace entries with diagnostic_valid=True and the fields
    returned by predict plus eventual phase_success/collision/prediction_error.
    Multiple records within an episode are correlated; no independent-record
    confidence interval is implied by these descriptive summaries.
    """
    rows = [r for r in records if r.get('diagnostic_valid') and r.get('monitor_ready')]
    result = dict(n=len(rows), note='Physical-execution forecasts only; within-episode correlated observations. Descriptive metrics, not a parameter estimate of intrinsic metacognitive noise.')
    for head, label in (('phase', 'phase_success'), ('collision', 'collision')):
        for kind in ('raw_', ''):
            key = kind + head + '_probability'
            selected = [r for r in rows if key in r and label in r]
            result[kind + head] = binary_metrics([r[key] for r in selected], [r[label] for r in selected])
    errors = [r['predicted_error']-r['prediction_error'] for r in rows
              if 'predicted_error' in r and 'prediction_error' in r]
    result['prediction_error_mae'] = float(np.mean(np.abs(errors))) if errors else None
    result['prediction_error_rmse'] = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    return result
