"""Actual experiment wiring with explicitly synthetic dynamics and physics.

The NumPy monitor, calibration, MPC, experiment functions and comparison code run
unchanged. Temporary fixture results test software contracts only; they must
never be reported as Habitat measurements or shipped as example scores.
"""
import ast
from contextlib import redirect_stdout
from dataclasses import asdict
import csv
import io
import json
from pathlib import Path
import pickle
import platform
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
import numpy as np
from test_meta import MPC, MetaMPC, Compute, prior, AnalyticalDynamics, observation
from cup_baseline.metacognition import MetaMonitor, monitoring_metrics, diagnostic_metrics
from cup_baseline.protocol import (TASKS, DOMAINS, task_summary, schedule, threshold_report,
                                   adaptation_summary, cost_summary)
from cup_baseline.metrics import continual_metrics
from cup_baseline.generation import GENERATION_VERSION
from cup_baseline.compare import compare


class TestWorld(AnalyticalDynamics):
    """Predictable state transitions expose pipeline, not learning performance."""
    def __init__(self, monitor):
        super().__init__(monitor)
        self.train_flops_est = 0

    def add(self, obs, action, nxt):
        self.inputs.append(1)
        self.inputs = self.inputs[-self.replay_limit:]

    def fit(self, updates):
        self.train_flops_est += 100 * updates
        if self.monitor:
            self.monitor.fit(updates=2)
        return .1

    def snapshot(self):
        return dict(inputs=list(self.inputs), train_flops=self.train_flops_est,
                    meta=self.monitor.snapshot() if self.monitor else None)

    def restore(self, state):
        self.inputs = list(state['inputs'])
        self.train_flops_est = state['train_flops']
        if self.monitor:
            self.monitor.restore(state['meta'])


class TestEnvironment:
    """Eight-step fixture with real phase transitions and mixed phase outcomes."""
    max_steps = 8

    def reset(self, scene, seed, shift=0, task=TASKS[0]):
        self.k = 0
        self.o = observation()
        self.o['task_id'] = TASKS.index(task)
        self.task = task
        self.manifest = dict(scene=scene, seed=seed, task=task, domain=DOMAINS[shift])
        return self.o

    def step(self, action):
        self.k += 1
        phase = 0 if self.k < 2 else 1 if self.k < 4 else 2
        self.o = dict(self.o, x=prior(self.o['x'][None], action[None])[0], phase=phase)
        success = int(self.task == TASKS[0] and self.k == self.max_steps)
        info = dict(task=self.task, success=success, nav_success=int(self.k >= 2),
            pick_success=success, drawer_success=0, steps=self.k,
            physics_substeps=self.k*10, nav_spl=.5, raycasts=12*self.k, ik_fk_evals=0,
            collision_steps=0, manipulation_collision_steps=0)
        return self.o, self.k == self.max_steps, info

    def close(self):
        pass


def actual_experiment_functions(created):
    """Load actual run.py functions without requiring native simulator imports."""
    tree = ast.parse((Path(__file__).parents[1] / 'cup_baseline/run.py').read_text())
    excluded = {'create', 'main', 'plot_curve', 'record', 'smoke', 'check_domains'}
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name not in excluded]
    torch = SimpleNamespace(__version__='EXPLICIT_TEST_DOUBLE',
        save=lambda obj, path: Path(path).write_bytes(pickle.dumps(obj)),
        load=lambda path, **kwargs: pickle.loads(Path(path).read_bytes()))
    config = dict(episodes=24, evaluate_every=4, validation_n=1, test_n=1, updates=1,
                  max_steps=8, horizon=2, candidates=8, iterations=1, elite=2)

    def create(args, cfg):
        monitor = MetaMonitor(args.seed) if cfg.get('agent') == 'metacognitive' else None
        model = TestWorld(monitor)
        cls = MetaMPC if monitor else MPC
        planner = cls(model, **{k: cfg[k] for k in ('horizon', 'candidates', 'iterations', 'elite')})
        created.append(model)
        return TestEnvironment(), model, planner

    namespace = dict(np=np, Path=Path, time=time, json=json, csv=csv, asdict=asdict, Compute=Compute,
        TASKS=TASKS, DOMAINS=DOMAINS, GENERATION_VERSION=GENERATION_VERSION,
        SCHEMA='home_two_tasks_v2', TRAIN_SCENES=['EXPLICIT_TEST_SCENE'],
        UNSEEN_SCENES=['EXPLICIT_TEST_UNSEEN'], PRESETS={'fixture': config}, create=create,
        torch=torch, platform=platform, task_summary=task_summary, schedule=schedule,
        threshold_report=threshold_report, cost_summary=cost_summary,
        monitoring_metrics=monitoring_metrics, diagnostic_metrics=diagnostic_metrics,
        adaptation_summary=adaptation_summary, continual_metrics=continual_metrics,
        plot_curve=lambda *args: None)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'actual_run_protocol', 'exec'), namespace)
    return namespace


class MetaProtocolTests(unittest.TestCase):
    def assert_learning_state_equal(self, left, right):
        # Checkpoint deserialization may change object aliasing and pickle byte
        # layout without changing any parameter, buffer, counter or RNG value.
        if isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_learning_state_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_learning_state_equal(a, b)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            self.assertEqual(left, right)

    def test_actual_training_resume_calibration_continual_and_paired_report(self):
        created = []
        functions = actual_experiment_functions(created)
        old = sys.modules.get('habitat_sim')
        sys.modules['habitat_sim'] = SimpleNamespace(__version__='EXPLICIT_TEST_DOUBLE')
        try:
            with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
                root = Path(temp)
                for agent in ('baseline', 'metacognitive'):
                    args = SimpleNamespace(preset='fixture', dynamics='learned', replay_limit=4096,
                        episodes=None, agent=agent, meta_compute_weight=.1, meta_step_weight=.05,
                        out=root/agent, resume=False, seed=0, data='NO_NATIVE_ASSETS')
                    functions['train'](args)
                    summary = json.loads((args.out / 'summary.json').read_text())
                    self.assertEqual(summary['train_episodes'], 24)
                    self.assertEqual(summary['algorithm_version'], 'meta_comparison_v2')
                    if agent == 'metacognitive':
                        self.assertEqual(summary['metacognition']['training_episodes'], 20)
                        self.assertGreater(summary['training_cost']['meta_update_nn_flops'], 0)
                        self.assertTrue(summary['metacognition']['calibration']['phase']['ready'])
                        self.assertGreater(summary['seen']['by_task'][TASKS[0]]['local_monitoring']['n'], 0)
                    before = created[-1].snapshot()
                    args.resume = True
                    functions['train'](args)  # Re-evaluate a completed checkpoint; no fitting.
                    self.assert_learning_state_equal(before, created[-1].snapshot())
                    args.episodes = 26
                    functions['train'](args)  # Resume a genuinely extended training budget.
                    extended = json.loads((args.out / 'summary.json').read_text())
                    self.assertEqual(extended['train_episodes'], 26)
                    if agent == 'metacognitive':
                        self.assertEqual(extended['metacognition']['training_episodes'], 22)
                        self.assertEqual(extended['metacognition']['calibration_records'], 32)
                    before = created[-1].snapshot()
                    env, model, planner = functions['create'](args, extended['config'])
                    model.restore(before)
                    functions['evaluate'](env, model, planner, ['EXPLICIT_TEST_SCENE'], 1,
                        950000, 'freeze_regression', args.out, 26)
                    self.assert_learning_state_equal(before, model.snapshot())
                    continual = SimpleNamespace(checkpoint=args.out/'checkpoint.pt',
                        out=root/(agent+'_cl'), data=args.data,
                        phase_episodes=2, eval_n=1, validation_n=1, check_every=2)
                    functions['continual'](continual)
                    result = json.loads((continual.out/'continual_summary.json').read_text())
                    self.assertEqual(result['agent'], agent)
                    self.assertEqual(len(result['costs']), 3)
                    if agent == 'metacognitive':
                        self.assertEqual(result['metacognition']['training_episodes'], 28)
                        self.assertGreater(result['metacognition']['calibration_records'], 32)
                        for cost in result['costs']:
                            self.assertGreater(cost['meta_update_nn_flops'], 0)
                    saved_result = (continual.out/'continual_summary.json').read_bytes()
                    functions['continual'](continual)  # Completed protocol reuse preserves results.
                    self.assertEqual(saved_result, (continual.out/'continual_summary.json').read_bytes())
                report = compare(root/'baseline', root/'metacognitive', root/'report',
                                 root/'baseline_cl', root/'metacognitive_cl')
                self.assertEqual(report['status'], 'complete')
                self.assertEqual(report['training_cost']['baseline']['meta_update_nn_flops'], 0)
                self.assertGreater(report['training_cost']['metacognitive']['meta_update_nn_flops'], 0)
                self.assertTrue((root/'report'/'comparison.csv').exists())
                self.assertFalse(list(root.rglob('*.html')))
        finally:
            if old is None:
                sys.modules.pop('habitat_sim', None)
            else:
                sys.modules['habitat_sim'] = old


if __name__ == '__main__':
    unittest.main()
