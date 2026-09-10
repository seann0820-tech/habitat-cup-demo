"""Read-only shadow protocol regression with explicit analytical fixtures.

The actual run.shadow, NumPy monitor and MPC implementations execute here. Only
native physics, learned dynamics and checkpoint transport are replaced. All
outputs are temporary synthetic software-test records, never Habitat scores.
"""
import ast
import copy
import csv
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import test_meta as controllers
from test_meta_protocol import TestEnvironment
from cup_baseline.metacognition import MetaMonitor, diagnostic_metrics
from cup_baseline.protocol import TASKS, DOMAINS
from cup_baseline.generation import GENERATION_VERSION


class FrozenMonitor(MetaMonitor):
    def add_episode(self, *args, **kwargs):
        raise AssertionError('Shadow observation must never write monitor replay')

    def fit(self, *args, **kwargs):
        raise AssertionError('Shadow observation must never fit a monitor')


class FrozenWorld(controllers.AnalyticalDynamics):
    """Analytical prediction with checkpoint state and forbidden learning calls."""
    def __init__(self, seed=0, mode='learned', replay_limit=4096):
        super().__init__(None)
        self.replay_limit=replay_limit
        self.parameters=np.array([.1,.2,.3],np.float32)
        self.train_flops_est=123

    def restore(self, state):
        self.inputs=copy.deepcopy(state['inputs'])
        self.parameters=state['parameters'].copy()
        self.train_flops_est=state['train_flops']

    def snapshot(self):
        return dict(inputs=copy.deepcopy(self.inputs),parameters=self.parameters.copy(),
            train_flops=self.train_flops_est,metacognition=self.monitor.snapshot() if self.monitor else None)

    def add(self, *args, **kwargs):
        raise AssertionError('Shadow observation must never write world replay')

    def fit(self, *args, **kwargs):
        raise AssertionError('Shadow observation must never fit dynamics')


class RecordingEnvironment(TestEnvironment):
    def __init__(self):
        self.episodes=[]
        self.closed=False

    def reset(self, scene, seed, shift=0, task=TASKS[0]):
        obs=super().reset(scene,seed,shift=shift,task=task)
        self.episodes.append(dict(scene=scene,seed=seed,shift=shift,task=task,actions=[]))
        return obs

    def step(self, action):
        self.episodes[-1]['actions'].append(action.copy())
        return super().step(action)

    def close(self):
        self.closed=True


def actual_shadow(worlds,environments):
    """Load the real protocol and inject only the unavailable native services."""
    def create(args,cfg):
        model=FrozenWorld(args.seed,mode=cfg['dynamics'],replay_limit=cfg['replay_limit'])
        world_args={k:cfg[k] for k in ('horizon','candidates','iterations','elite')}
        policy=controllers.MPC(model,**world_args)
        env=RecordingEnvironment();worlds.append(model);environments.append(env)
        return env,model,policy

    def ensemble(*args,**kwargs):
        model=FrozenWorld(*args,**kwargs);worlds.append(model);return model

    source=Path(__file__).parents[1]/'cup_baseline/run.py'
    selected=[node for node in ast.parse(source.read_text()).body
              if isinstance(node,ast.FunctionDef) and node.name in ('shadow','dump','save_csv')]
    namespace=dict(__name__='cup_baseline._shadow_test',__package__='cup_baseline',
        Path=Path,np=np,json=json,csv=csv,time=time,create=create,Ensemble=ensemble,
        MetaMonitor=FrozenMonitor,Compute=controllers.Compute,diagnostic_metrics=diagnostic_metrics,
        load_checkpoint=lambda path,**kwargs:pickle.loads(Path(path).read_bytes()),
        TASKS=TASKS,DOMAINS=DOMAINS,TRAIN_SCENES=['EXPLICIT_TEST_SCENE'],
        SCHEMA='EXPLICIT_TEST_SCHEMA',GENERATION_VERSION=GENERATION_VERSION)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'actual_run_shadow','exec'),namespace)
    return namespace['shadow']


class ShadowTests(unittest.TestCase):
    def checkpoint_pair(self,root):
        config=dict(agent='baseline',algorithm_version='meta_comparison_v2',dynamics='learned',
            replay_limit=4096,max_steps=8,horizon=2,candidates=8,iterations=1,elite=2)
        world=FrozenWorld();world.inputs=[11,22]
        base=dict(seed=0,config=config,model=world.snapshot())
        monitor=controllers.MetaTests().forced_monitor(1)
        # A known calibration transform changes probabilities, never labels.
        monitor.calibration['phase'].update(ready=True,bias=.7,log_scale=0.)
        monitor.calibration['collision'].update(ready=True,bias=-.4,log_scale=0.)
        meta=copy.deepcopy(base);meta['config']['agent']='metacognitive'
        meta['model']['metacognition']=monitor.snapshot()
        paths=root/'baseline.pt',root/'metacognitive.pt'
        for path,record in zip(paths,(base,meta)):path.write_bytes(pickle.dumps(record))
        return paths,base,meta

    def run_protocol(self,shadow,args):
        module=ModuleType('cup_baseline.meta_planner')
        module.MetaMPC=controllers.MetaMPC  # Actual class, not a controller double.
        with patch.dict(sys.modules,{'cup_baseline.meta_planner':module}):
            return shadow(args)

    def test_fixed_actions_frozen_state_same_labels_and_separate_costs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths,base,meta=self.checkpoint_pair(root)
            worlds=[];environments=[];shadow=actual_shadow(worlds,environments)
            args=SimpleNamespace(checkpoint=paths[0],monitor_checkpoint=paths[1],out=root/'shadow',n=1)
            self.run_protocol(shadow,args)
            self.assertEqual(len(worlds),2);self.assertEqual(len(environments),1)
            self.assertTrue(environments[0].closed)
            self.assertEqual(pickle.dumps(worlds[0].snapshot()),pickle.dumps(base['model']))
            # Compare monitor and world separately: pickle memoization can differ
            # when otherwise equal NumPy dtype objects are shared between them.
            self.assertEqual(pickle.dumps(worlds[1].monitor.snapshot()),pickle.dumps(meta['model']['metacognition']))
            observer_world=worlds[1].snapshot();observer_world['metacognition']=None
            self.assertEqual(pickle.dumps(observer_world),pickle.dumps(base['model']))

            # Independently execute the same fixed baseline without the observer.
            standalone=FrozenWorld();standalone.restore(base['model'])
            policy=controllers.MPC(standalone,**{k:base['config'][k] for k in
                ('horizon','candidates','iterations','elite')})
            env=RecordingEnvironment()
            for episode in environments[0].episodes:
                obs=env.reset(episode['scene'],episode['seed'],shift=episode['shift'],task=episode['task'])
                policy.reset(episode['seed']+17)
                for observed_action in episode['actions']:
                    action=policy.act(obs)
                    np.testing.assert_array_equal(action,observed_action)
                    obs,done,_=env.step(action)
                self.assertTrue(done)
            report=json.loads((args.out/'shadow_summary.json').read_text())
            costs=report['costs']
            self.assertEqual(costs['episodes'],6)
            self.assertEqual(costs['environment_steps'],48)
            self.assertGreater(costs['policy_nn_flops'],0)
            self.assertGreater(costs['observer_nn_flops'],0)
            self.assertNotEqual(costs['policy_nn_flops'],costs['observer_nn_flops'])
            self.assertEqual(costs['total_nn_flops'],costs['policy_nn_flops']+costs['observer_nn_flops'])
            self.assertEqual(costs['policy_nn_flops'],standalone.count.nn_flops_est)
            for group in report['by_task_domain']:
                domain=DOMAINS.index(group['domain'])
                event=json.loads((args.out/f'{group["task"]}_domain{domain}_episode000.json').read_text())
                records=event['metrics']['local_monitoring_records']
                self.assertEqual(len(records),8)
                for head,label in [('phase','phase_success'),('collision','collision')]:
                    truth=np.array([r[label] for r in records])
                    for kind in ('raw_',''):
                        predicted=np.array([r[kind+head+'_probability'] for r in records])
                        score=group['diagnostics'][kind+head]
                        self.assertEqual(score['n'],len(truth))
                        self.assertAlmostEqual(score['brier'],float(np.mean((predicted-truth)**2)))
                    self.assertNotEqual(records[0]['raw_'+head+'_probability'],records[0][head+'_probability'])

            # A completed identical request reuses outputs without more episodes.
            before={p.name:p.read_bytes() for p in args.out.iterdir()}
            self.run_protocol(shadow,args)
            self.assertEqual(len(worlds),2)
            self.assertEqual(before,{p.name:p.read_bytes() for p in args.out.iterdir()})

    def test_mismatched_checkpoint_or_cached_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths,base,meta=self.checkpoint_pair(root)
            worlds=[];environments=[];shadow=actual_shadow(worlds,environments)
            args=SimpleNamespace(checkpoint=paths[0],monitor_checkpoint=paths[1],out=root/'shadow',n=1)
            wrong=copy.deepcopy(meta);wrong['seed']=3
            paths[1].write_bytes(pickle.dumps(wrong))
            with self.assertRaisesRegex(ValueError,'matching seeds'):self.run_protocol(shadow,args)
            self.assertEqual(worlds,[])
            paths[1].write_bytes(pickle.dumps(meta));self.run_protocol(shadow,args)
            changed=copy.deepcopy(meta);changed['model']['metacognition']['biases'][-1][0]+=.1
            paths[1].write_bytes(pickle.dumps(changed))
            with self.assertRaisesRegex(RuntimeError,'different checkpoints/protocol'):self.run_protocol(shadow,args)
            self.assertEqual(len(worlds),2)


if __name__=='__main__':unittest.main()
