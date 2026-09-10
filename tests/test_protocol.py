"""Simulator-independent regression tests; no test values are experiment scores."""
import ast
import csv
from dataclasses import dataclass,asdict
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
import numpy as np
from cup_baseline.geometry import segment_hits_box,clearance,detour
from cup_baseline.protocol import TASKS,DOMAINS,schedule,task_summary,threshold_report,adaptation_summary,cost_summary
from cup_baseline.metrics import continual_metrics
from cup_baseline.generation import GENERATION_VERSION


def row(task,success):
    r=dict(task=task,success=success,nav_success=1,pick_success=int(task==TASKS[0] and success),
        drawer_success=int(task==TASKS[1] and success),steps=2,physics_substeps=20,
        nn_forward_calls=3,nn_sample_forwards=12,nn_flops_est=100,
        planner_seconds=.01,wall_seconds=.02,nav_spl=.8,raycasts=24,ik_fk_evals=0)
    return r


class ProtocolTests(unittest.TestCase):
    def test_drawer_requires_pull_and_hold_not_just_reaching_handle(self):
        tree=ast.parse((Path(__file__).parents[1]/'cup_baseline/tasks.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HomeEnv')
        import math
        ns=dict(CupEnv=object,np=np,math=math,DOMAINS=DOMAINS,TASKS=TASKS,
            wrap=lambda x:(x+np.pi)%(2*np.pi)-np.pi,
            node_bounds=lambda n:(np.array([-.2,0,-.2]),np.array([.2,1,.2])))
        exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual_HomeEnv','exec'),ns)
        env=ns['HomeEnv'].__new__(ns['HomeEnv']);ee=np.array([0.,.8,0.])
        env.task=TASKS[1];env.phase=1;env.held=False;env.nav_success=True
        env.steps=env.physics_steps=env.collisions=env.manipulation_collisions=env.collision_checks=0
        env.raycast_count=env.ik_fk_evals=env.hold_ticks=0;env.path_length=0.;env.shortest_distance=1.
        env.shift=0;env.pick_success=env.drawer_success=False;env.boxes=[];env.max_steps=100
        env.arm_step=.055;env.grasp_distance=.09;env.nav_distance=.22
        env.drawer=SimpleNamespace(joint_positions=np.zeros(1),get_link_scene_node=lambda lid:None)
        env.drawer_ix=0;env.drawer_link=0;env.drawer_closed=np.zeros(1);env.joint_scale=1.
        env.pull_axis=np.array([1.,0,0]);env.nav_goal=np.array([.75,0,0])
        env.robot=SimpleNamespace(arm_joint_pos=np.zeros(7),update=lambda:None,open_gripper=lambda:None,close_gripper=lambda:None)
        env.sim=SimpleNamespace(step_physics=lambda dt:None)
        env.state=lambda:np.r_[.75,0.,0.,ee]
        env.target_position=lambda:np.array([env.drawer.joint_positions[0],.8,0.])
        env.handle_position=env.target_position;env.observe=lambda:{}
        env._ik_delta=lambda delta:ee.__iadd__(delta)
        _,done,_=env.step(np.zeros(5))
        self.assertTrue(env.held);self.assertEqual(env.opening(),0.);self.assertFalse(done)
        for _ in range(3):env.step(np.zeros(5))
        self.assertFalse(env.drawer_success)
        for _ in range(4):env.step([0,0,1,0,0])
        self.assertGreaterEqual(env.opening(),.18);self.assertFalse(env.drawer_success)
        for _ in range(4):_,done,info=env.step(np.zeros(5))
        self.assertTrue(done);self.assertEqual(info['drawer_success'],1);self.assertEqual(info['pick_success'],0)

    def test_collision_is_swept_not_endpoint_only(self):
        self.assertTrue(segment_hits_box([-2,0,0],[2,0,0],[-.1,-1,-1],[.1,1,1]))
        self.assertFalse(segment_hits_box([-2,2,0],[2,2,0],[-.1,-1,-1],[.1,1,1]))
        self.assertTrue(segment_hits_box([0,0,0],[0,0,0],[-1]*3,[1]*3))

    def test_clearance_and_detour(self):
        boxes=np.array([[[-.2,-.2],[.2,.2]]])
        p=detour(np.array([-1.,0]),np.array([1.,0]),boxes,.1)
        self.assertFalse(segment_hits_box([-1,0],p,boxes[0,0],boxes[0,1],.1))
        self.assertGreater(np.linalg.norm(p-[1,0]),.1)
        np.testing.assert_allclose(clearance([[0,0],[1,0]],boxes),[0,.8])

    def test_task_scene_balance(self):
        scenes=['a','b','c','d']
        samples=[schedule(i,scenes) for i in range(8)]
        self.assertEqual(set(samples),{(t,s) for t in TASKS for s in scenes})

    def test_conditional_task_rates_and_macro(self):
        s=task_summary([row(TASKS[0],1)]*3+[row(TASKS[1],0)])
        self.assertEqual(s['success_rate'],.5)
        self.assertEqual(s['pick_success_rate'],1.)
        self.assertEqual(s['drawer_success_rate'],0.)

    def test_n85_does_not_hide_one_failed_task(self):
        curve=[]
        for ep in (0,2,4):
            scores=task_summary([row(TASKS[0],1),row(TASKS[1],0)])
            curve.append(dict(train_episodes=ep,train_steps=ep*2,**scores))
        result=threshold_report(curve,4)
        self.assertTrue(result[TASKS[0]]['already_at_threshold'])
        self.assertFalse(result['both_tasks']['reached'])
        self.assertFalse(result[TASKS[1]]['reached'])

    def test_adaptation_and_cost(self):
        curve=[dict(train_episodes=ep,train_steps=ep*2,**task_summary([row(t,ok) for t in TASKS]))
               for ep,ok in [(0,0),(2,1),(4,1)]]
        a=adaptation_summary(curve,4)
        self.assertEqual(a['n85']['both_tasks']['episodes'],2)
        self.assertEqual(a['learning_curve_auc']['macro'],.75)
        costs=cost_summary([row(t,1) for t in TASKS],600)
        self.assertEqual(costs['total_nn_flops'],800)
        self.assertEqual(costs['environment_steps'],4)

    def test_real_episode_and_evaluate_do_not_learn_during_tests(self):
        # Execute actual orchestration functions with a tiny explicit test environment.
        tree=ast.parse((Path(__file__).parents[1]/'cup_baseline/run.py').read_text())
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
            ('episode','evaluate','dump','save_csv','load_checkpoint','checkpoint','continual')]
        @dataclass
        class Compute:
            nn_forward_calls:int=0
            nn_sample_forwards:int=0
            nn_flops_est:int=0
        class Model:
            collect_with_prior=False;replay_limit=4
            def __init__(self):self.inputs=[];self.train_flops_est=0;self.updates=0
            def add(self,*args):self.inputs.append(1);self.inputs=self.inputs[-self.replay_limit:]
            def fit(self,n):self.updates+=n;self.train_flops_est+=100*n
            def snapshot(self):return dict(inputs=list(self.inputs),flops=self.train_flops_est)
            def restore(self,s):self.inputs=list(s['inputs']);self.train_flops_est=s['flops']
        class Env:
            max_steps=2
            def reset(self,scene,seed,shift=0,task=TASKS[0]):
                self.task=task;self.k=0;self.manifest=dict(task=task,seed=seed,domain=DOMAINS[shift]);return {}
            def step(self,a):
                self.k+=1;info=row(self.task,int(self.task==TASKS[0]))
                for key in ('nn_forward_calls','nn_sample_forwards','nn_flops_est','wall_seconds'):info.pop(key)
                return {},self.k==2,info
            def close(self):pass
        class Planner:
            def reset(self,*args):pass
            def act(self,*args,**kwargs):return np.zeros(5)
        model=Model();env=Env();planner=Planner()
        ns=dict(np=np,Path=Path,time=time,json=json,csv=csv,asdict=asdict,Compute=Compute,
            TASKS=TASKS,DOMAINS=DOMAINS,GENERATION_VERSION=GENERATION_VERSION,SCHEMA='test_schema',TRAIN_SCENES=['test_scene'],
            task_summary=task_summary,schedule=schedule,cost_summary=cost_summary,
            adaptation_summary=adaptation_summary,continual_metrics=continual_metrics)
        exec(compile(ast.Module(body=selected,type_ignores=[]),'actual_run_functions','exec'),ns)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            ns['evaluate'](env,model,planner,['s'],2,750000,'test',out,0)
            self.assertEqual(model.inputs,[])
            ns['episode'](env,model,planner,'s',1,learn=True,task=TASKS[1])
            self.assertEqual(len(model.inputs),2)
            # Native dependency-free end-to-end protocol check, including required report.
            cp=out/'checkpoint.pt';cp.write_bytes(b'EXPLICIT TEST CHECKPOINT')
            (out/'summary.json').write_text('{"test_fixture":true}')
            saved=dict(schema='test_schema',seed=0,config={'updates':1,'algorithm_version':'meta_comparison_v2'},model=model.snapshot())
            ns['torch']=SimpleNamespace(load=lambda *a,**k:saved,
                save=lambda obj,p:Path(p).write_text(json.dumps(obj)))
            ns['create']=lambda *a:(env,model,planner)
            args=SimpleNamespace(phase_episodes=2,check_every=2,eval_n=1,validation_n=1,
                checkpoint=cp,out=out/'continual',data='test_data')
            ns['continual'](args)
            result=json.loads((args.out/'continual_summary.json').read_text())
            self.assertEqual(model.updates,6)
            self.assertEqual(len(model.inputs),4)
            self.assertEqual(result['matrix'][TASKS[1]],[[0.,0.,0.]]*3)
            self.assertTrue((out/'mandatory_report.json').exists())
            for c in result['costs']:self.assertEqual(c['update_nn_flops'],200)
            saved['schema']='old_schema'
            with self.assertRaises(ValueError):ns['load_checkpoint'](cp)


if __name__=='__main__':unittest.main()
