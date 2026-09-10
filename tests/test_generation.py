"""Generation regression tests use explicit geometry responses, not Habitat scores."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
import numpy as np
from cup_baseline.generation import GENERATION_VERSION, EpisodeGenerationError, passage_candidates
from cup_baseline.geometry import segment_hits_box
from cup_baseline.compare import paired_geometry
from cup_baseline.protocol import TASKS, DOMAINS, schedule


def fixture(accept_after=0, require_new_start=False, reject_all=False):
    source=Path(__file__).parents[1]/'cup_baseline/tasks.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HomeEnv')
    ns=dict(CupEnv=object,np=np,time=time,GENERATION_VERSION=GENERATION_VERSION,
        EpisodeGenerationError=EpisodeGenerationError,passage_candidates=passage_candidates,
        segment_hits_box=segment_hits_box,mn=SimpleNamespace(Vector3=np.asarray),
        habitat_sim=SimpleNamespace(ShortestPath=SimpleNamespace))
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'actual_HomeEnv','exec'),ns)
    env=ns['HomeEnv'].__new__(ns['HomeEnv'])
    env.start=np.array([-1.,0,0]);env.nav_goal=np.array([2.,0,0])
    env.robot=SimpleNamespace(base_pos=env.start.copy(),base_rot=.7,update=lambda:None)
    env.boxes=[];env.proposals=[];env.clear_count=0
    def box(center,half,label):
        env.proposals.append(center.copy());env.boxes=[np.array([center-half,center+half])]
    def clear():env.boxes=[];env.clear_count+=1
    def geodesic(start,end):
        d=float(np.linalg.norm(start-end))
        if not env.boxes:return d
        if reject_all or len(env.proposals)<=accept_after:return float('inf')
        if require_new_start and start[0]<0:return float('inf')
        return d+.10
    def find_path(path):
        path.points=[path.requested_start.copy(),path.requested_end.copy()]
        path.geodesic_distance=float(np.linalg.norm(path.requested_start-path.requested_end))
        return True
    pf=SimpleNamespace(find_path=find_path,snap_point=lambda p:p.copy(),seed=lambda seed:None,
        get_random_navigable_point=lambda:np.array([0.,0,1.]))
    env.sim=SimpleNamespace(pathfinder=pf);env.geodesic=geodesic
    env._box=box;env._clear_added=clear;env._rebuild_navmesh=lambda:None
    return env


class GenerationTests(unittest.TestCase):
    def test_candidate_extension_keeps_box_size_out_of_search(self):
        points=np.array([[0.,0,0],[3.,0,0]])
        legacy=passage_candidates(points,np.random.default_rng(9),legacy=True)
        new=passage_candidates(points,np.random.default_rng(9))
        self.assertEqual(len(legacy),8);self.assertEqual(len(new),32)
        self.assertTrue(all(offset==0 for _,_,offset in legacy))
        self.assertTrue(any(offset!=0 for _,_,offset in new))

    def test_failed_legacy_candidates_do_not_abort_search(self):
        env=fixture(accept_after=8)
        env._block_passage(np.random.default_rng(7))
        self.assertGreater(env.generation_diagnostics['box_attempts'],8)
        self.assertFalse(env.generation_diagnostics['start_resampled'])
        self.assertEqual(len(env.boxes),1)
        np.testing.assert_allclose(env.boxes[0][1]-env.boxes[0][0],[.32,.54,.32])
        self.assertGreater(env.generation_diagnostics['detour_extra_m'],.03)

    def test_geometry_only_start_fallback_is_deterministic(self):
        a,b=fixture(require_new_start=True),fixture(require_new_start=True)
        for env in (a,b):env._block_passage(np.random.default_rng(7))
        self.assertTrue(a.generation_diagnostics['start_resampled'])
        self.assertEqual(a.generation_diagnostics['start_attempts'],2)
        self.assertEqual(a.robot.base_rot,.7)
        np.testing.assert_array_equal(a.start,b.start)
        np.testing.assert_array_equal(a.boxes,b.boxes)
        self.assertEqual(a.generation_diagnostics['attempts'],b.generation_diagnostics['attempts'])

    def test_exhaustion_is_bounded_cleaned_and_not_a_scored_failure(self):
        env=fixture(reject_all=True)
        with self.assertRaises(EpisodeGenerationError):env._block_passage(np.random.default_rng(7))
        self.assertLessEqual(env.generation_diagnostics['box_attempts'],264)
        self.assertLessEqual(env.generation_diagnostics['start_attempts'],8)
        self.assertEqual(env.boxes,[])
        np.testing.assert_array_equal(env.start,[-1.,0,0])
        json.dumps(env.generation_diagnostics,allow_nan=False)

    def test_equal_seeds_with_different_geometry_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            a,b=Path(temp)/'a.json',Path(temp)/'b.json'
            row=dict(scene='fixture',seed=1,task=TASKS[0],domain=DOMAINS[1],start=[0,0,0])
            a.write_text(json.dumps([row]));b.write_text(json.dumps([row]))
            paired_geometry(a,b)
            row['start']=[1,0,0];b.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):paired_geometry(a,b)

    def test_preflight_covers_protocol_without_actions_and_caches_success(self):
        tree=ast.parse((Path(__file__).parents[1]/'cup_baseline/run.py').read_text())
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('dump','check_domains')]
        resets=[]
        class Env:
            def __init__(self,*a,**k):pass
            def reset(self,scene,seed,shift,task):
                resets.append((scene,seed,task,shift));self.manifest=dict(scene=scene,seed=seed,task=task,domain=DOMAINS[shift])
            def close(self):pass
        ns=dict(Path=Path,json=json,time=time,CupEnv=Env,TASKS=TASKS,DOMAINS=DOMAINS,
            TRAIN_SCENES=['a','b'],GENERATION_VERSION=GENERATION_VERSION,schedule=schedule,
            load_checkpoint=lambda *a,**k:dict(config={'max_steps':240}))
        exec(compile(ast.Module(body=selected,type_ignores=[]),'actual_generation_check','exec'),ns)
        with tempfile.TemporaryDirectory() as temp:
            args=SimpleNamespace(data='fixture_data',out=temp,checkpoint='fixture.pt',
                phase_episodes=2,eval_n=2,validation_n=1)
            ns['check_domains'](args)
            self.assertEqual(len(resets),24)
            self.assertEqual(len(set(resets)),24)
            self.assertIn(('a',560001,TASKS[0],1),resets)
            ns['check_domains'](args);self.assertEqual(len(resets),24)


if __name__=='__main__':unittest.main()
