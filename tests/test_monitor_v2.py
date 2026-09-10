"""Real NumPy learning tests on fixtures; these are not Habitat performance scores."""
import copy
import pickle
import unittest
import numpy as np
from cup_baseline.metacognition import (ACTIONS, FEATURES, META_SCHEMA, MetaMonitor,
                                        diagnostic_metrics, monitoring_metrics)


def record(i=0, action=0, value=1., reward=0., diagnostic=True):
    x = np.zeros(len(FEATURES), np.float32); x[0] = value
    return dict(features=x.tolist(), choice=action, step=i, phase=0,
        reward=reward, discount=.99, allowed=[True]*4,
        diagnostic_valid=diagnostic, phase_success=int(value > 0),
        collision=int(value < 0), prediction_error=.15 if value > 0 else .75)


class MonitorV2Tests(unittest.TestCase):
    def test_selected_action_optimizer_learns_diagnostics_and_q(self):
        model = MetaMonitor(seed=8)
        for ep in range(40):
            trace = []
            for i in range(16):
                value = 1. if (i//4+ep) % 2 else -1.
                action = i % 4
                target_action = 1 if value > 0 else 3
                r = record(i, action, value, reward=1. if action == target_action else -.3)
                r['discount'] = 0.  # Isolated immediate-return fixture.
                trace.append(r)
            model.add_episode(trace, success=True, steps=16)
        self.assertFalse(model.ready)  # Coverage is not a fitted controller.
        model.fit(updates=800, batch=64, lr=.025)
        self.assertTrue(model.ready)
        for value, correct in [(1., 1), (-1., 3)]:
            p = model.predict(record(value=value)['features'])
            self.assertEqual(int(np.argmax(p['action_values'])), correct)
            if value > 0:
                self.assertGreater(p['raw_phase_probability'], .9)
                self.assertLess(p['raw_collision_probability'], .1)
            else:
                self.assertLess(p['raw_phase_probability'], .1)
                self.assertGreater(p['raw_collision_probability'], .9)
            self.assertAlmostEqual(p['predicted_error'], .15 if value > 0 else .75, delta=.15)
        self.assertGreater(model.target_sample_forwards, 0)
        self.assertGreater(model.train_flops_est, model.train_sample_forwards*model.flops_one)
        self.assertGreater(model.calibration_flops_est, 0)

    def test_full_trace_successor_before_subsampling_and_action_mask(self):
        model = MetaMonitor(1)
        trace = [record(i, action=i % 4, value=float(i), reward=.25, diagnostic=False)
                 for i in range(33)]
        model.add_episode(trace, False, 33)
        self.assertEqual(len(model.samples), 16)
        for r in model.samples:
            if not r['done']:
                self.assertEqual(r['next_features'][0], r['features'][0]+1)
        for w in model.target_weights:
            w.fill(0)
        model.target_biases[-1][3:] = [2, 1000, 3, 7]
        r = copy.deepcopy(model.samples[0]); r['next_allowed'] = np.array([True, False, True, False])
        r['reward'] = .25; r['discount'] = .5
        self.assertAlmostEqual(float(model._td_targets([r])[0]), 1.75)
        r['done'] = True; r['next_allowed'][:] = False
        self.assertAlmostEqual(float(model._td_targets([r])[0]), .25)

    def test_episode_holdout_has_no_shared_training_or_q_leakage(self):
        model = MetaMonitor(2, capacity=40, calibration_capacity=40)
        for ep in range(15):
            model.add_episode([record(i, action=i % 4, value=(-1.)**i) for i in range(16)], True, 16)
        self.assertEqual(model.training_episodes, 15)
        self.assertEqual(model.shared_training_episodes, 12)
        self.assertEqual(len(model.samples), 40)
        self.assertEqual(len(model.calibration_samples), 40)
        self.assertTrue(all(r['episode_id'] % 5 for r in model.samples))
        self.assertTrue(all(r['episode_id'] % 5 == 0 for r in model.calibration_samples))
        self.assertTrue(all(r['diagnostic_valid'] for r in model.calibration_samples))
        before = pickle.dumps((model.weights, model.biases, model.target_weights, model.target_biases, model.rng.bit_generator.state))
        model.fit(updates=0)
        after = pickle.dumps((model.weights, model.biases, model.target_weights, model.target_biases, model.rng.bit_generator.state))
        self.assertEqual(before, after)
        self.assertTrue(model.calibration['phase']['ready'])
        self.assertTrue(model.calibration['collision']['ready'])

    def test_calibration_requires_episodes_and_classes_then_reduces_known_bias(self):
        model = MetaMonitor(7)
        for w in model.weights:
            w.fill(0)
        model.biases[-1][:2] = 2.
        for _ in range(10):
            model.add_episode([record(i, value=(-1.)**i) for i in range(16)], True, 16)
        model.fit(updates=0)
        self.assertFalse(model.calibration['phase']['ready'])
        for _ in range(5):
            model.add_episode([record(i, value=(-1.)**i) for i in range(16)], True, 16)
        model.fit(updates=0)
        self.assertTrue(model.calibration['phase']['ready'])
        self.assertLess(model.calibration['phase']['calibrated_brier'], model.calibration['phase']['raw_brier'])
        for r in model.calibration_samples:
            r['diagnostic'][1] = 0
        model.fit(updates=0)
        self.assertFalse(model.calibration['collision']['ready'])
        self.assertEqual(model.predict(record()['features'])['raw_collision_probability'],
                         model.predict(record()['features'])['collision_probability'])

    def test_restore_and_frozen_evaluation_are_bitwise_reproducible(self):
        model = MetaMonitor(5)
        for ep in range(20):
            model.add_episode([record(i, action=i % 4, value=(-1.)**i) for i in range(16)], True, 16)
        model.fit(updates=3)
        restored = MetaMonitor(99); restored.restore(model.snapshot())
        model.fit(updates=4); restored.fit(updates=4)
        self.assertEqual(pickle.dumps(model.snapshot()), pickle.dumps(restored.snapshot()))
        state = pickle.dumps(model.snapshot()); rng = np.random.default_rng(10)
        rng_state = pickle.dumps(rng.bit_generator.state)
        for _ in range(3):
            action, p = model.choose(record()['features'], rng, learning=False, allowed=[False, False, True, False])
            self.assertEqual(action, 2); self.assertEqual(p['propensity'], 1.)
        self.assertEqual(state, pickle.dumps(model.snapshot()))
        self.assertEqual(rng_state, pickle.dumps(rng.bit_generator.state))
        broken = model.snapshot(); broken['schema'] = 'meta_mc_v1'
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            restored.restore(broken)
        self.assertEqual(model.summary()['schema'], META_SCHEMA)

    def test_phase_metrics_never_substitute_final_task_success(self):
        self.assertEqual(monitoring_metrics([dict(success=0, meta_confidence=.9)])['n'], 0)
        rows = [dict(success=0, meta_phase_confidence=.9, meta_phase_success=1),
                dict(success=1, meta_phase_confidence=.2, meta_phase_success=0)]
        result = monitoring_metrics(rows)
        self.assertAlmostEqual(result['brier'], .025)
        self.assertEqual(result['discrimination_auc'], 1.)
        physical = dict(record(), monitor_ready=True, raw_phase_probability=.8, phase_probability=.9,
                        raw_collision_probability=.2, collision_probability=.1, predicted_error=.25)
        d = diagnostic_metrics([physical, dict(physical, diagnostic_valid=False)])
        self.assertEqual(d['n'], 1); self.assertAlmostEqual(d['prediction_error_mae'], .1)

    def test_invalid_labels_and_illegal_recorded_actions_fail_explicitly(self):
        model = MetaMonitor()
        r = record(); r['allowed'][0] = False
        with self.assertRaises(ValueError):
            model.add_episode([r], False, 1)
        self.assertEqual(model.training_episodes, 0)
        r = record(); r['prediction_error'] = float('nan')
        with self.assertRaises(ValueError):
            model.add_episode([r], False, 1)
        with self.assertRaises(ValueError):
            model.choose(record()['features'], np.random.default_rng(2), False, [False]*4)


if __name__ == '__main__':
    unittest.main()
