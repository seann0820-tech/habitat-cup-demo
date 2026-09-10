"""Incremental MPC controlled by learned metacognitive action values.

The fixed MPC's world model, action space, objective and geometry guide are
shared. A paid search prefix is followed by EXECUTE, another small EXPAND batch,
RESTART of the search distribution, or the original BASELINE planner. Search
has a finite batch budget. A baseline fallback additionally pays every already
completed search and monitor call; it is never advertised as free computation.

The monitor's phase/collision forecasts are diagnostics. They are distinct from
its action-value estimates, which are trained on actual computation costs and
subsequent environment rewards. No imagined branch receives a success label.
"""
import time
import numpy as np
from .model import MPC, prior, wrap
from .geometry import detour, clearance
from .metacognition import ACTIONS, FEATURES, diagnostic_metrics


class MetaMPC(MPC):
    """One metacontroller; fixed dynamics learning and replay remain external."""

    def reset(self, seed):
        super().reset(seed)
        self.choice_rng = np.random.default_rng(seed + 3721)
        self.learning = False
        self.max_steps = 240
        self.trace = []
        self.step_index = 0
        self.error_ema = self.error_trend = self.last_error = 0.
        self.stalled = self.collision_last = self.progress = self.phase_changed = 0.
        self.execution_surprise = self.revisit = self.waypoint_progress = 0.
        self.actual_displacement = 0.
        self.previous_collisions = 0
        self.last_action = np.zeros(5, np.float32)
        self.action_change = 0.
        self.expected = None
        self.last_predicted_error = .25
        self.active_record = None
        self.active_waypoint = None
        self.position_history = []
        self.completed_phases = set()
        self.completion_events = []
        self.recovery_pending = False
        self.recovery_cooldown = 0
        self.recovery_resets = 0
        self.search_batches = 0
        self.fallback_steps = 0
        self.last_info = {}

    def start_episode(self, learning, max_steps):
        self.learning = bool(learning)
        self.max_steps = int(max_steps)

    @property
    def monitor(self):
        return self.model.monitor

    def total_flops(self):
        return self.model.count.nn_flops_est + self.model.count.meta_nn_flops_est

    @property
    def reference_episode_flops(self):
        """Fixed planner's full max-length episode, used only to scale cost."""
        return max(1, self.max_steps * self.horizon * self.candidates *
                   self.iterations * self.model.members * self.model.flops_one)

    def _monitor_cost(self):
        self.model.count.meta_nn_forward_calls += 1
        self.model.count.meta_nn_sample_forwards += 1
        self.model.count.meta_nn_flops_est += self.monitor.flops_one

    def _cost(self, obs, seq):
        n = len(seq)
        x = np.broadcast_to(obs['x'], (n, 6)).copy()
        cost = np.zeros(n)
        uncs = np.zeros(n)
        boxes = np.asarray(obs.get('arm_boxes', [])).reshape(-1, 2, 3)
        for h in range(self.horizon):
            x, unc = self.model.predict(x, seq[:, h], obs)
            uncs += unc / self.horizon
            if obs['phase'] == 0:
                dist = np.linalg.norm(x[:, :2] - obs['nav_goal'], axis=1)
                face = (obs['nav_goal'] if obs.get('temporary_nav_goal', False)
                        else obs['cup'][[0, 2]]) - x[:, :2]
                angle = np.abs(wrap(np.arctan2(-face[:, 1], face[:, 0]) - x[:, 2]))
                c = 5 * dist + .8 * np.exp(-dist * 3) * angle
                if len(obs['obstacles']):
                    near = np.linalg.norm(x[:, None, :2] - obs['obstacles'][None], axis=2).min(1)
                    c += 60 * np.maximum(0, .45 - near) ** 2
            else:
                c = 15 * np.linalg.norm(x[:, 3:6] - obs['ee_goal'], axis=1)
                c += 200 * np.maximum(0, .08 - clearance(x[:, 3:6], boxes, .04)) ** 2
            cost += (.98 ** h) * (c + self.risk * unc + .02 * np.sum(seq[:, h] ** 2, axis=1))
        return cost, uncs

    def _search(self, obs, guided, mean, std, n, incumbent=None):
        """One CEM batch. Incumbent is re-evaluated and charged with the batch."""
        seq = np.clip(self.rng.normal(mean, std, (n, self.horizon, 5)), -1, 1).astype(np.float32)
        seq[0] = guided
        if incumbent is not None and n > 1:
            seq[1] = incumbent
        if obs['phase'] == 0:
            seq[:, :, 2:] = 0
        else:
            seq[:, :, :2] = 0
        cost, unc = self._cost(obs, seq)
        order = np.argsort(cost)
        k = int(order[0])
        best = seq[k].copy()
        elite = seq[order[:self.elite]]
        mean = elite.mean(0)
        std = np.maximum(elite.std(0), .12)
        reference = float(cost[1] if incumbent is not None and n > 1 else cost[0])
        gap = float(cost[order[1]] - cost[k]) if n > 1 else 0.
        self.search_batches += 1
        return best, mean, std, dict(best_cost=float(cost[k]), unc=float(unc[k]),
            gain=max(0., reference - float(cost[k])) / (abs(reference) + 1),
            gap=gap / (abs(float(cost[k])) + 1))

    def _prepare(self, original):
        obs = original.copy()
        boxes = np.asarray(obs.get('arm_boxes', [])).reshape(-1, 2, 3)
        if len(boxes):
            if obs['phase'] == 0:
                low = boxes[boxes[:, 0, 1] < 1.4]
                point = detour(obs['x'][:2], obs['nav_goal'], low[:, :, [0, 2]], .32)
                obs['temporary_nav_goal'] = bool(np.linalg.norm(point - obs['nav_goal']) > .001)
                obs['nav_goal'] = point
            else:
                obs['ee_goal'] = detour(obs['x'][3:6], obs['ee_goal'], boxes, .06, vertical=True)
        self.active_waypoint = np.asarray(obs['nav_goal'] if obs['phase'] == 0 else obs['ee_goal']).copy()
        guided = []
        x = obs['x'][None].copy()
        for _ in range(self.horizon):
            action = self.guide(x[0], obs)
            guided.append(action)
            x = prior(x, action[None])
        guided = np.asarray(guided)
        reset_history = self.recovery_pending and self.recovery_cooldown == 0
        if reset_history:
            self.recovery_resets += 1
            self.recovery_pending = False
            self.recovery_cooldown = 4
        fresh = self.prev_phase != obs['phase'] or reset_history
        mean = guided.copy() if fresh else .5 * self.mean + .5 * guided
        std = np.full_like(mean, .9 if reset_history else .65)
        return obs, guided, mean, std

    def _features(self, obs, search, used):
        values = dict(
            task_cup=float(obs.get('task_id', 0) == 0),
            task_drawer=float(obs.get('task_id', 0) == 1),
            phase_nav=float(obs['phase'] == 0), phase_reach=float(obs['phase'] == 1),
            phase_finish=float(obs['phase'] == 2),
            nav_distance=min(1., np.linalg.norm(obs['x'][:2] - obs['nav_goal']) / 5),
            ee_distance=min(1., np.linalg.norm(obs['x'][3:] - obs['ee_goal']) / 2),
            time_used=self.step_index / self.max_steps,
            replay_fraction=len(self.model.inputs) / self.model.replay_limit,
            error_ema=self.error_ema, error_trend=self.error_trend, last_error=self.last_error,
            disagreement=np.sqrt(max(0., search['unc'])) / .1,
            search_gain=search['gain'], cost_gap=search['gap'],
            last_action_change=self.action_change, stalled_steps=self.stalled / 10,
            collision_last=self.collision_last, goal_progress=self.progress,
            phase_changed=self.phase_changed,
            current_best_cost=search['best_cost'] / max(1, 20 * self.horizon),
            learning_age=min(1., self.monitor.training_episodes / 60),
            search_fraction=used / max(1, self.candidates * self.iterations),
            execution_surprise=self.execution_surprise, revisit=self.revisit,
            waypoint_progress=self.waypoint_progress,
            actual_displacement=self.actual_displacement,
            remaining_steps=max(0., 1 - self.step_index / self.max_steps))
        return np.clip(np.nan_to_num([values[key] for key in FEATURES], nan=0., posinf=5., neginf=-5.),
                       -5., 5.).astype(np.float32)

    def _record(self, features, choice, forecast, allowed, cost_start):
        forecast = {k:v for k,v in forecast.items() if k not in ('allowed','choice','action','phase','step','reward','discount','features','diagnostic_valid')}
        record = dict(features=features.tolist(), choice=int(choice), action=ACTIONS[choice],
            phase=int(np.argmax(features[2:5])), step=self.step_index,
            allowed=list(map(bool, allowed)), diagnostic_valid=False,
            reward=-self.monitor.compute_weight * (self.total_flops() - cost_start) / self.reference_episode_flops,
            discount=1., **forecast)
        self.trace.append(record)
        return record

    def _prediction_probe(self, obs, action):
        """Predict the exact noisy action that will be executed, before feedback."""
        self.expected = self.model.predict(obs['x'][None], action[None], obs)[0][0].copy()
        self.action_change = float(np.linalg.norm(action - self.last_action) / np.sqrt(5))
        self.last_action = action.copy()

    def act(self, original, exploration=0.):
        self.active_record = None
        if self.model.collect_with_prior or (not self.learning and not self.monitor.ready):
            # No preliminary search or monitoring is secretly charged in this path.
            self.expected = None
            self.active_waypoint = None
            self.fallback_steps += 1
            return super().act(original, exploration=exploration)
        started = time.perf_counter()
        cost_start = self.total_flops()
        obs, guided, mean, std = self._prepare(original)
        limit = self.candidates * self.iterations
        chunk = max(self.elite, self.candidates // 2)
        chunk = min(chunk, limit)
        best, mean, std, search = self._search(obs, guided, mean, std, chunk)
        used = chunk
        while True:
            features = self._features(original, search, used)
            can_search = used < limit
            allowed = [True, can_search, can_search, can_search]
            choice, forecast = self.monitor.choose(features, self.choice_rng, self.learning, allowed=allowed)
            self._monitor_cost()
            if choice in (1, 2):
                if choice == 2:
                    mean = guided.copy()
                    std = np.full_like(mean, 1.)
                n = min(chunk, limit - used)
                best, mean, std, search = self._search(obs, guided, mean, std, n, incumbent=best)
                used += n
                self._record(features, choice, forecast, allowed, cost_start)
                cost_start = self.total_flops()
                continue
            if choice == 3:
                # Execute the original fixed MPC, preserving its full search code.
                # Its internal time/call counters are accounted once by the wrapper.
                calls_before = self.model.count.planner_calls
                seconds_before = self.model.count.planner_seconds
                action = super().act(original, exploration=exploration)
                self.model.count.planner_calls = calls_before
                self.model.count.planner_seconds = seconds_before
                self.fallback_steps += 1
            else:
                self.mean = np.concatenate([best[1:], best[-1:]], axis=0)
                self.prev_phase = original['phase']
                action = best[0].copy()
                if exploration:
                    action += self.rng.normal(0, exploration, 5)
                if original['phase'] == 0:
                    action[2:] = 0
                else:
                    action[:2] = 0
                action = np.clip(action, -1, 1)
            self._prediction_probe(original, action)
            record = self._record(features, choice, forecast, allowed, cost_start)
            record['diagnostic_valid'] = True
            record['discount'] = .99
            self.active_record = record
            self.last_predicted_error = float(forecast['predicted_error'])
            break
        self.model.count.planner_calls += 1
        self.model.count.planner_seconds += time.perf_counter() - started
        return action

    def shadow_step(self, original, action):
        """Read-only observer of an externally selected fixed-policy action.

        Prefix search, monitor inference and next-state probe are all charged to
        an observer ledger by the caller. The observer never changes the action.
        """
        cost_start = self.total_flops()
        obs, guided, mean, std = self._prepare(original)
        chunk = max(self.elite, self.candidates // 2)
        best, mean, std, search = self._search(obs, guided, mean, std, chunk)
        features = self._features(original, search, chunk)
        forecast = self.monitor.predict(features)
        forecast['propensity'] = 1.
        self._monitor_cost()
        self._prediction_probe(original, np.asarray(action))
        record = self._record(features, 3, forecast, [False, False, False, True], cost_start)
        record.update(diagnostic_valid=True, discount=.99, shadow=True)
        self.active_record = record
        self.last_predicted_error = float(forecast['predicted_error'])
        self.mean = np.concatenate([best[1:], best[-1:]], axis=0)
        self.prev_phase = original['phase']

    def observe_transition(self, obs, action, nxt, info):
        old_phase = int(obs['phase'])
        if self.expected is not None:
            error = nxt['x'] - self.expected
            error[2] = wrap(error[2])
            scale = np.array([.18, .18, .35, .10, .10, .10])
            self.last_error = float(min(5., np.sqrt(np.mean((error / scale) ** 2))))
            old = self.error_ema
            self.error_ema = .8 * old + .2 * self.last_error
            self.error_trend = self.error_ema - old
            unexpected = np.clip(self.last_error / max(.25, self.last_predicted_error) - 1., 0., 3.)
            self.execution_surprise = float(.8 * self.execution_surprise + .2 * unexpected)
        key = 'nav_goal' if old_phase == 0 else 'ee_goal'
        before = obs['x'][:2] if old_phase == 0 else obs['x'][3:]
        after = nxt['x'][:2] if old_phase == 0 else nxt['x'][3:]
        self.progress = float(np.linalg.norm(before - obs[key]) - np.linalg.norm(after - obs[key]))
        self.actual_displacement = float(np.linalg.norm(after - before))
        waypoint = self.active_waypoint if self.active_waypoint is not None else obs[key]
        self.waypoint_progress = float(np.linalg.norm(before - waypoint) - np.linalg.norm(after - waypoint))
        self.phase_changed = int(nxt['phase'] != old_phase)
        forward_completion = (old_phase, int(nxt['phase'])) in ((0, 1), (1, 2))
        new_milestone = forward_completion and old_phase not in self.completed_phases
        if forward_completion:
            self.completed_phases.add(old_phase)
            self.completion_events.append((self.step_index, old_phase))
        if self.phase_changed:
            self.position_history = []
        # Distance from the final goal is NOT a stall detector: a detour may
        # require moving away. Rotations and deliberate hold actions are allowed.
        turn = abs(float(wrap(nxt['x'][2] - obs['x'][2]))) if old_phase == 0 else 0.
        command = np.linalg.norm(action[:2] if old_phase == 0 else action[2:])
        no_motion = self.actual_displacement < .003 and turn < .02 and command > .15
        self.stalled = min(50, self.stalled + 1) if no_motion and not self.phase_changed else 0
        signature = np.r_[after, nxt['x'][2]] if old_phase == 0 else after.copy()
        older = self.position_history[:-3]
        self.revisit = float(sum(np.linalg.norm(signature - p) < .025 for p in older) >= 2)
        self.position_history = (self.position_history + [signature.copy()])[-12:]
        collisions = info.get('collision_steps', 0) + info.get('manipulation_collision_steps', 0)
        self.collision_last = float(collisions > self.previous_collisions)
        self.previous_collisions = collisions
        if self.recovery_cooldown > 0:
            self.recovery_cooldown -= 1
        if self.execution_surprise >= 1.2 or self.stalled >= 3 or (self.revisit and command > .15):
            self.recovery_pending = True
        if self.phase_changed:
            self.recovery_pending = False
            self.execution_surprise = 0.
        if self.active_record is not None:
            self.active_record.update(collision=int(self.collision_last), prediction_error=self.last_error)
            # Detaching a drawer handle can regress 2→1. Regression and repeated
            # reattachment do not earn another milestone reward.
            self.active_record['reward'] += .1 * new_milestone - self.monitor.step_weight / self.max_steps
        self.last_info = dict(info)
        self.step_index += 1

    def finish_episode(self, success, steps):
        """Labels are assigned after actual phase transitions; no future leakage."""
        if success:
            self.completion_events.append((max(0, self.step_index-1), 2))
        physical = [r for r in self.trace if r.get('diagnostic_valid')]
        for record in self.trace:
            # A forecast must precede its qualifying completion. A failed later
            # retry must not inherit a successful attachment from an earlier step.
            record['phase_success'] = int(any(phase == record['phase'] and step >= record['step']
                for step, phase in self.completion_events))
        if self.trace:
            self.trace[-1]['reward'] += 1. if success else -1.
            self.trace[-1]['discount'] = 0.
        if self.learning and self.trace:
            self.monitor.add_episode(self.trace, success, steps)
        ready = [r for r in physical if r.get('monitor_ready')]
        result = dict(meta_decisions=len(self.trace),
            meta_execute=sum(r['choice'] == 0 for r in self.trace),
            meta_expand=sum(r['choice'] == 1 for r in self.trace),
            meta_restart=sum(r['choice'] == 2 for r in self.trace),
            meta_baseline=sum(r['choice'] == 3 for r in self.trace),
            meta_fallback_steps=self.fallback_steps, meta_search_batches=self.search_batches,
            meta_recovery_resets=self.recovery_resets,
            meta_ready_decisions=sum(bool(r.get('monitor_ready')) for r in self.trace),
            meta_phase_confidence=ready[0]['phase_probability'] if ready else None,
            meta_phase_success=ready[0]['phase_success'] if ready else None,
            meta_confidence=ready[0]['phase_probability'] if ready else None,
            meta_mean_prediction_error=float(self.error_ema),
            local_monitoring_records=[{k:r[k] for k in (
                'diagnostic_valid','monitor_ready',
                'phase','raw_phase_probability','phase_probability','phase_success',
                'raw_collision_probability','collision_probability','collision','predicted_error','prediction_error')}
                for r in physical])
        return result
