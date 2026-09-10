"""Actual NumPy monitor and MPC regressions with explicit analytical fixtures.

These tests verify computation accounting, supervision and frozen evaluation.
They do not load Habitat and their synthetic outcomes are never benchmark data.
"""
import ast
import copy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import pickle
import sys
import time
import types
import unittest
import numpy as np
from cup_baseline.metacognition import MetaMonitor, FEATURES, ACTIONS, monitoring_metrics
from cup_baseline.geometry import detour, clearance


def controller_classes():
    # Colab uses the normal PyTorch import. Without PyTorch, execute the exact
    # controller AST against AnalyticalDynamics; no learned model is imitated.
    if importlib.util.find_spec('torch'):
        from cup_baseline.model import MPC, Compute, prior
        from cup_baseline.meta_planner import MetaMPC
        return MPC, MetaMPC, Compute, prior
    root = Path(__file__).parents[1] / 'cup_baseline'
    tree = ast.parse((root / 'model.py').read_text())
    selected = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in ('wrap', 'body_to_world', 'world_to_body', 'prior', 'MPC', 'Compute')]
    mod = types.ModuleType('cup_baseline.model')
    mod.__dict__.update(np=np, time=time, dataclass=dataclass, detour=detour, clearance=clearance)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'actual_model_ast', 'exec'), mod.__dict__)
    before = sys.modules.get('cup_baseline.model')
    sys.modules['cup_baseline.model'] = mod
    try:
        spec = importlib.util.spec_from_file_location('cup_baseline._tested_meta_planner', root / 'meta_planner.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if before is None:
            sys.modules.pop('cup_baseline.model', None)
        else:
            sys.modules['cup_baseline.model'] = before
    return mod.MPC, module.MetaMPC, mod.Compute, mod.prior


MPC, MetaMPC, Compute, prior = controller_classes()


def observation(phase=0):
    return dict(x=np.array([0, 0, 0, .5, .9, 0], np.float32), joints=np.zeros(7, np.float32),
        rays=np.full(12, 3, np.float32), phase=phase, nav_goal=np.array([2, 0]),
        cup=np.array([2.7, .8, 0]), ee_goal=np.array([2.7, .8, 0]),
        obstacles=np.empty((0, 2)), arm_boxes=np.empty((0, 2, 3)), task_id=0)


def training_record(choice=0, step=0, feature=0., phase_success=1, reward=.5):
    """An explicit synthetic observed computation, accepted by the public API."""
    x = np.zeros(len(FEATURES), np.float32)
    x[FEATURES.index('phase_nav')] = 1
    x[FEATURES.index('nav_distance')] = feature
    return dict(features=x.tolist(), choice=choice, step=step, phase=0,
                reward=reward, discount=.99, allowed=[True] * len(ACTIONS),
                diagnostic_valid=True, phase_success=phase_success,
                collision=1-phase_success, prediction_error=.1 if phase_success else .8)


class AnalyticalDynamics:
    """Explicit test double; no learned dynamics or Habitat benchmark claims."""
    def __init__(self, monitor):
        self.monitor = monitor
        self.count = Compute()
        self.members = 3
        self.flops_one = 100
        self.inputs = []
        self.replay_limit = 4096
        self.collect_with_prior = False

    def predict(self, x, a, obs):
        self.count.nn_forward_calls += 3
        self.count.nn_sample_forwards += 3 * len(x)
        self.count.nn_flops_est += 3 * len(x) * self.flops_one
        return prior(x, a), np.zeros(len(x))


class MetaTests(unittest.TestCase):
    def forced_monitor(self, choice):
        """Fixed Q preferences exercise the real search and legal-action mask."""
        monitor = MetaMonitor(1)
        monitor.training_episodes = monitor.shared_training_episodes = 4
        monitor.train_updates = 1
        monitor.action_counts = np.full(len(ACTIONS), 4)
        for weight in monitor.weights:
            weight.fill(0)
        monitor.biases[-1][3:] = -8
        monitor.biases[-1][3+choice] = 8
        return monitor

    def planner(self, choice=0):
        model = AnalyticalDynamics(self.forced_monitor(choice))
        planner = MetaMPC(model, horizon=3, candidates=12, iterations=2, elite=3)
        planner.reset(8)
        planner.start_episode(False, 20)
        return planner, model

    def test_optimizer_learns_selected_terminal_values_and_phase_diagnostics(self):
        monitor = MetaMonitor(8)
        rng = np.random.default_rng(44)
        for i in range(600):
            feature = float(rng.choice([-1., 1.]))
            choice = i % len(ACTIONS)
            success = int(feature < 0 or choice == 1)
            record = training_record(choice, feature=feature, phase_success=int(feature < 0),
                                     reward=1. if success else -1.)
            # One-record episodes have terminal rewards, so this checks actual
            # selected-action supervision without a target-network confound.
            monitor.add_episode([record], success, 1)
        probe = np.zeros(len(FEATURES), np.float32)
        probe[FEATURES.index('phase_nav')] = 1
        probe[FEATURES.index('nav_distance')] = 1
        before = monitor.predict(probe)
        monitor.fit(updates=700, lr=.03)
        after = monitor.predict(probe)
        self.assertGreater(after['action_values'][1], .8)
        self.assertLess(after['action_values'][0], -.8)
        self.assertEqual(int(np.argmax(after['action_values'])), 1)
        self.assertLess(after['raw_phase_probability'], .1)
        probe[FEATURES.index('nav_distance')] = -1
        self.assertGreater(monitor.predict(probe)['raw_phase_probability'], .9)
        self.assertGreater(monitor.train_flops_est, 0)
        self.assertGreater(abs(after['action_values'][0] - before['action_values'][0]), .5)

    def test_snapshot_restores_rng_targets_buffers_and_calibration(self):
        original = MetaMonitor(7)
        for episode in range(30):
            y = episode % 2
            records = [training_record(i % 4, i, feature=float(y), phase_success=y) for i in range(8)]
            original.add_episode(records, y, 8)
        original.fit(updates=2)
        self.assertTrue(original.calibration['phase']['ready'])
        restored = MetaMonitor(100)
        restored.restore(original.snapshot())
        self.assertEqual(pickle.dumps(original.snapshot()), pickle.dumps(restored.snapshot()))
        original.fit(updates=4)
        restored.fit(updates=4)
        self.assertEqual(pickle.dumps(original.snapshot()), pickle.dumps(restored.snapshot()))
        bad = original.snapshot()
        bad['schema'] = 'old'
        with self.assertRaises(ValueError):
            restored.restore(bad)

    def test_incremental_search_and_paid_baseline_fallback_exact_accounting(self):
        for choice in range(len(ACTIONS)):
            with self.subTest(choice=ACTIONS[choice]):
                planner, model = self.planner(choice)
                action = planner.act(observation())
                if choice == 0:
                    candidate_total, monitor_calls, batches = 6, 1, 1
                elif choice in (1, 2):
                    candidate_total, monitor_calls, batches = 24, 4, 4
                else:
                    # The six prefix candidates remain paid when the original
                    # fixed planner then evaluates its full 12*2 candidates.
                    candidate_total, monitor_calls, batches = 6 + 24, 1, 1
                expected_samples = 3 * (3 * candidate_total + 1)  # Includes next-state probe.
                self.assertEqual(planner.trace[0]['choice'], choice)
                self.assertEqual(model.count.nn_sample_forwards, expected_samples)
                self.assertEqual(model.count.nn_flops_est, expected_samples * model.flops_one)
                self.assertEqual(model.count.meta_nn_forward_calls, monitor_calls)
                self.assertEqual(model.count.meta_nn_sample_forwards, monitor_calls)
                self.assertEqual(model.count.meta_nn_flops_est, monitor_calls * model.monitor.flops_one)
                self.assertEqual(model.count.planner_calls, 1)
                self.assertEqual(planner.search_batches, batches)
                self.assertTrue(np.all(action[2:] == 0))
                self.assertTrue(np.all(abs(action) <= 1))
                if choice in (1, 2):
                    self.assertEqual(planner.trace[-1]['choice'], 0)
                    self.assertEqual(planner.trace[-1]['allowed'], [True, False, False, False])
                expected_cost = -model.monitor.compute_weight * planner.total_flops() / planner.reference_episode_flops
                self.assertAlmostEqual(sum(r['reward'] for r in planner.trace), expected_cost)

    def test_eval_freezes_all_persistent_learning_state_but_updates_local_history(self):
        planner, model = self.planner(0)
        before = pickle.dumps(model.monitor.snapshot())
        obs = observation()
        action = planner.act(obs)
        nxt = copy.deepcopy(obs)
        nxt['x'][0] -= .2
        planner.observe_transition(obs, action, nxt, dict(collision_steps=1))
        result = planner.finish_episode(0, 1)
        self.assertGreater(planner.last_error, 0)
        self.assertGreater(planner.error_ema, 0)
        self.assertEqual(planner.step_index, 1)
        self.assertEqual(before, pickle.dumps(model.monitor.snapshot()))
        self.assertEqual(result['meta_decisions'], 1)
        self.assertEqual(len(model.inputs), 0)

    def test_bounded_memory_preserves_phase_label_after_later_task_failure(self):
        monitor = MetaMonitor(2, capacity=12)
        trace = [training_record(i % 4, i, phase_success=1) for i in range(20)]
        for _ in range(3):
            monitor.add_episode(trace, False, 20)
        self.assertEqual(len(monitor.samples), 12)
        self.assertEqual(monitor.training_episodes, 3)
        self.assertTrue(all(r['diagnostic'][0] == 1 for r in monitor.samples))
        self.assertTrue(all(not r['episode_success'] for r in monitor.samples))

    def test_successful_navigation_not_labeled_failure_when_reach_fails(self):
        planner, model = self.planner(0)
        planner.start_episode(True, 20)
        obs = observation(0)
        # Disable training exploration in this fixture while preserving learning.
        model.monitor.choose = lambda features, rng, learning, allowed=None: (
            0, dict(model.monitor.predict(features), propensity=1.))
        action = planner.act(obs)
        reached = copy.deepcopy(obs)
        reached['phase'] = 1
        planner.observe_transition(obs, action, reached, dict(nav_success=1))
        action = planner.act(reached)
        planner.observe_transition(reached, action, reached, dict(nav_success=1))
        result = planner.finish_episode(False, 2)
        physical = [r for r in planner.trace if r['diagnostic_valid']]
        self.assertEqual([r['phase_success'] for r in physical], [1, 0])
        self.assertEqual(result['meta_phase_success'], 1)
        stored = model.monitor.samples + model.monitor.calibration_samples
        self.assertEqual([r['diagnostic'][0] for r in stored], [1, 0])
        score = monitoring_metrics([dict(success=0, **result)])
        self.assertAlmostEqual(score['brier'], (result['meta_phase_confidence'] - 1)**2)

    def test_drawer_detach_does_not_reward_failure_or_label_later_reach_as_success(self):
        planner, model = self.planner(0)
        obs = observation(0)
        obs['task_id'] = 1
        bonuses = []
        # First reach succeeds, handle detaches, reattachment succeeds, then a
        # second detach is followed by an unsuccessful final reach attempt.
        for next_phase in (1, 2, 1, 2, 1, 1):
            action = planner.act(obs)
            record = planner.active_record
            before = record['reward']
            nxt = copy.deepcopy(obs)
            nxt['phase'] = next_phase
            planner.observe_transition(obs, action, nxt, dict(nav_success=1))
            bonuses.append(record['reward'] - before + model.monitor.step_weight/planner.max_steps)
            obs = nxt
        result = planner.finish_episode(False, 6)
        physical = [r for r in planner.trace if r['diagnostic_valid']]
        self.assertEqual([r['phase'] for r in physical], [0, 1, 2, 1, 2, 1])
        self.assertEqual([r['phase_success'] for r in physical], [1, 1, 0, 1, 0, 0])
        np.testing.assert_allclose(bonuses, [.1, .1, 0, 0, 0, 0], atol=1e-12)
        self.assertNotIn(2, planner.completed_phases)
        self.assertEqual(result['local_monitoring_records'][-1]['phase_success'], 0)

    def test_stall_detector_allows_detours_turns_and_holds_but_detects_rejected_motion(self):
        planner, _ = self.planner(0)
        obs = observation()
        detoured = copy.deepcopy(obs)
        detoured['x'][0] = -.1  # Further from final goal, yet valid physical progress.
        planner.active_waypoint = np.array([-1., 0.])
        planner.observe_transition(obs, np.array([-.5, 0, 0, 0, 0]), detoured, {})
        self.assertLess(planner.progress, 0)
        self.assertGreater(planner.waypoint_progress, 0)
        self.assertEqual(planner.stalled, 0)
        turned = copy.deepcopy(obs)
        turned['x'][2] = .2
        planner.observe_transition(obs, np.array([0, .6, 0, 0, 0]), turned, {})
        self.assertEqual(planner.stalled, 0)
        planner.observe_transition(obs, np.zeros(5), obs, {})
        self.assertEqual(planner.stalled, 0)
        for _ in range(3):
            planner.observe_transition(obs, np.array([.5, 0, 0, 0, 0]), obs, {})
        self.assertEqual(planner.stalled, 3)
        self.assertTrue(planner.recovery_pending)

    def test_recovery_resets_are_bounded_by_cooldown_and_reset_between_episodes(self):
        planner, _ = self.planner(0)
        obs = observation()
        planner.recovery_pending = True
        planner._prepare(obs)
        self.assertEqual(planner.recovery_resets, 1)
        self.assertEqual(planner.recovery_cooldown, 4)
        planner.recovery_pending = True
        planner._prepare(obs)
        self.assertEqual(planner.recovery_resets, 1)
        for _ in range(4):
            planner.observe_transition(obs, np.zeros(5), obs, {})
        planner._prepare(obs)
        self.assertEqual(planner.recovery_resets, 2)
        planner.previous_collisions = 10
        planner.reset(99)
        self.assertEqual(planner.previous_collisions, 0)
        self.assertFalse(planner.recovery_pending)
        self.assertEqual(planner.recovery_cooldown, 0)
        self.assertEqual(planner.position_history, [])

    def test_cold_eval_is_exact_fixed_baseline_without_hidden_probe(self):
        a, b = AnalyticalDynamics(MetaMonitor(1)), AnalyticalDynamics(None)
        planner = MetaMPC(a, horizon=3, candidates=12, iterations=2, elite=3)
        baseline = MPC(b, horizon=3, candidates=12, iterations=2, elite=3)
        planner.reset(9)
        baseline.reset(9)
        np.testing.assert_array_equal(planner.act(observation()), baseline.act(observation()))
        self.assertEqual(a.count.nn_flops_est, b.count.nn_flops_est)
        self.assertEqual(a.count.meta_nn_flops_est, 0)
        self.assertIsNone(planner.finish_episode(0, 1)['meta_phase_confidence'])
        self.assertEqual(len(planner.trace), 0)

    def test_manipulation_action_masks_apply_to_every_search_option(self):
        for choice in range(len(ACTIONS)):
            for phase in (1, 2):
                with self.subTest(choice=choice, phase=phase):
                    planner, _ = self.planner(choice)
                    self.assertTrue(np.all(planner.act(observation(phase))[:2] == 0))

    def test_shadow_observer_counts_compute_without_changing_action_or_learning(self):
        planner, model = self.planner(1)
        action = np.array([.3, -.2, 0, 0, 0], np.float32)
        original_action = action.copy()
        state = pickle.dumps(model.monitor.snapshot())
        obs = observation()
        planner.shadow_step(obs, action)
        nxt = copy.deepcopy(obs)
        nxt['x'] = prior(obs['x'][None], action[None])[0]
        planner.observe_transition(obs, action, nxt, {})
        result = planner.finish_episode(False, 1)
        np.testing.assert_array_equal(action, original_action)
        self.assertEqual(pickle.dumps(model.monitor.snapshot()), state)
        self.assertEqual(model.count.nn_sample_forwards, 3 * (3 * 6 + 1))
        self.assertEqual(model.count.meta_nn_forward_calls, 1)
        self.assertEqual(planner.trace[0]['choice'], ACTIONS.index('baseline'))
        self.assertTrue(planner.trace[0]['shadow'])
        self.assertEqual(len(result['local_monitoring_records']), 1)

    def test_monitoring_is_per_episode_scored_against_phase_outcome(self):
        self.assertEqual(monitoring_metrics([{'success': 1}])['n'], 0)
        rows = [dict(success=0, meta_phase_confidence=.9, meta_phase_success=1),
                dict(success=1, meta_phase_confidence=.9, meta_phase_success=0)]
        result = monitoring_metrics(rows)
        self.assertEqual(result['n'], 2)
        self.assertAlmostEqual(result['brier'], .41)
        self.assertEqual(result['high_confidence_failures'], 1)


if __name__ == '__main__':
    unittest.main()
