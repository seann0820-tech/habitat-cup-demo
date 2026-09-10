"""Report/paired-protocol regression checks using explicitly fabricated fixtures.
All files live in a temporary directory and are never baseline results.
"""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from cup_baseline.compare import compare
from cup_baseline.protocol import TASKS,task_summary,threshold_report,cost_summary
from cup_baseline.metrics import continual_metrics


class ComparisonTests(unittest.TestCase):
    def fixture(self,root,agent,training_seed=0):
        root.mkdir();rows=[]
        for task in TASKS:
            for seed in (1,2):
                rows.append(dict(task=task,scene='EXPLICIT_TEST_FIXTURE',seed=seed,shift=0,
                    success=int(seed==1),nav_success=1,pick_success=int(task==TASKS[0] and seed==1),
                    drawer_success=int(task==TASKS[1] and seed==1),steps=2,physics_substeps=20,
                    nn_forward_calls=3,nn_sample_forwards=10,nn_flops_est=100,
                    meta_nn_flops_est=10 if agent=='metacognitive' else 0,
                    total_nn_flops_est=110 if agent=='metacognitive' else 100,
                    total_nn_forward_calls=4 if agent=='metacognitive' else 3,
                    planner_seconds=.01,wall_seconds=.02,nav_spl=1,raycasts=24,ik_fk_evals=0,
                    meta_confidence=.8 if agent=='metacognitive' else None))
        summary=task_summary(rows)
        curve=[dict(train_episodes=e,train_steps=2*e,**summary) for e in (0,2)]
        cfg=dict(replay_limit=4096,horizon=3,candidates=12,iterations=2,elite=3,updates=1,
            max_steps=2,warmup_episodes=0,evaluate_every=2,validation_n=2,test_n=2)
        result=dict(agent=agent,seed=training_seed,train_episodes=2,dynamics='learned',algorithm_version='fixture',
            config=cfg,seen=summary,unseen=summary,n85=threshold_report(curve,2),
            generalization_gap_pp={t:0 for t in TASKS},training_cost=cost_summary(rows,100,20 if agent=='metacognitive' else 0),
            metacognition=None)
        (root/'summary.json').write_text(json.dumps(result));(root/'validation_curve.json').write_text(json.dumps(curve))
        for split in ('seen','unseen'):
            with (root/f'{split}_ep0002.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return root

    def continual_fixture(self,root,run,agent):
        """Small internally consistent retention files, all explicitly synthetic."""
        root.mkdir();checkpoint=b'EXPLICIT_TEST_CHECKPOINT_'+agent.encode()
        (run/'checkpoint.pt').write_bytes(checkpoint)
        source=json.loads((run/'summary.json').read_text())
        with (run/'seen_ep0002.csv').open() as f:raw=list(csv.DictReader(f))
        counts=[[1,1,0],[1,1,2],[1,1,1]]
        for phase in range(3):
            for domain in range(3):
                rows=[dict(r,shift=domain,success=int(int(r['seed'])<=counts[phase][domain])) for r in raw]
                with (root/f'after{phase}_domain{domain}_ep0002.csv').open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        matrix={task:[[n/2 for n in row] for row in counts] for task in TASKS}
        result=dict(agent=agent,checkpoint_sha256=hashlib.sha256(checkpoint).hexdigest(),
            protocol=dict(phase_episodes=2,eval_n=2,check_every=2,validation_n=2),
            matrix=matrix,frozen_reference=[source['seen'] for _ in range(3)],
            retention={task:continual_metrics(matrix[task]) for task in TASKS},
            adaptation=[dict(n85=source['n85'],learning_curve_auc={task:.5 for task in TASKS}) for _ in range(3)],
            costs=[source['training_cost'] for _ in range(3)])
        (root/'continual_summary.json').write_text(json.dumps(result))
        return root

    def test_report_requires_matching_episodes_and_keeps_cl_pending(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);b=self.fixture(root/'b','baseline');m=self.fixture(root/'m','metacognitive')
            r=compare(b,m,root/'report')
            self.assertEqual(r['status'],'basic_complete_continual_pending')
            self.assertIsNone(r['continual']);self.assertEqual(len(r['metrics']),8)
            self.assertEqual(r['paired'][0]['success_difference_pp'],0)
            self.assertTrue((root/'report/comparison.png').exists())
            self.assertEqual(r['schema'],'paired_comparison_v2')
            self.assertEqual(r['metrics'][0]['steps_success_only'],2)
            self.assertEqual(r['n85'][0]['reached'],False)
            self.assertEqual({p.suffix for p in (root/'report').iterdir()},{'.json','.csv','.png'})
            path=m/'seen_ep0002.csv';path.write_text(path.read_text().replace('EXPLICIT_TEST_FIXTURE','different'))
            with self.assertRaises(ValueError):compare(b,m,root/'bad')

    def test_total_cost_includes_meta_update_and_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root=self.fixture(Path(temp)/'m','metacognitive')
            r=json.loads((root/'summary.json').read_text())
            self.assertEqual(r['training_cost']['total_nn_flops'],4*110+100+20)
            self.assertEqual(r['seen']['by_task'][TASKS[0]]['mean_total_nn_flops_est'],110)

    def test_local_phase_diagnostics_preserve_missing_values_and_observation_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);b=self.fixture(root/'b','baseline');m=self.fixture(root/'m','metacognitive')
            path=m/'summary.json';data=json.loads(path.read_text())
            data['seen']['by_task']['pick_cup']['local_monitoring']={
                'phase_calibrated':{'n':2,'brier':.2},'collision':{'ece':None},'unit':'physical action'}
            path.write_text(json.dumps(data))
            result=compare(b,m,root/'comparison')
            rows={r['diagnostic']:r['value'] for r in result['local_monitoring_table']}
            self.assertEqual(rows['phase_calibrated.brier'],.2)
            self.assertEqual(rows['phase_calibrated.n'],2)
            self.assertIsNone(rows['collision.ece'])
            self.assertNotIn('unit',rows)

    def test_paired_manifest_geometry_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);b=self.fixture(root/'b','baseline');m=self.fixture(root/'m','metacognitive')
            # Same seeds can conceal different generated obstacle/start geometry.
            stem='seen_ep0002_episodes.json'
            record=[dict(scene='EXPLICIT_TEST_FIXTURE',seed=1,task=TASKS[0],start=[0.,0.,0.],
                         added_boxes=[dict(center=[1.,0.,0.])])]
            (b/stem).write_text(json.dumps(record));(m/stem).write_text(json.dumps(record))
            compare(b,m,root/'good')
            record[0]['added_boxes'][0]['center'][0]=2.
            (m/stem).write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'geometry differs'):compare(b,m,root/'bad')

    def test_continual_phase_start_is_not_frozen_initial_and_checkpoint_is_guarded(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);b=self.fixture(root/'b','baseline');m=self.fixture(root/'m','metacognitive')
            bc=self.continual_fixture(root/'bc',b,'baseline');mc=self.continual_fixture(root/'mc',m,'metacognitive')
            result=compare(b,m,root/'comparison',bc,mc)
            row=next(r for r in result['continual_adaptation'] if r['agent']=='baseline'
                     and r['task']=='pick_cup' and r['domain']=='C_workspace_clutter')
            self.assertEqual(row['frozen_initial_sr'],.5)
            self.assertEqual(row['phase_start_sr'],1.)
            self.assertEqual(row['phase_after_sr'],.5)
            self.assertEqual(row['phase_gain_pp'],-50.)
            self.assertEqual(row['total_gain_from_frozen_pp'],0.)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(len(result['continual_paired']),18)
            (b/'checkpoint.pt').write_bytes(b'DIFFERENT_TEST_CHECKPOINT')
            with self.assertRaisesRegex(ValueError,'different checkpoint'):compare(b,m,root/'bad',bc,mc)


if __name__=='__main__':unittest.main()
