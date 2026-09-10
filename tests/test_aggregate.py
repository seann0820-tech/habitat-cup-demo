"""Seed aggregation tests use fabricated temporary records, never demo scores."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from cup_baseline.compare import compare
from cup_baseline.aggregate import aggregate,threshold_summary
import test_comparison as fixtures


class AggregateTests(unittest.TestCase):
    def build(self,root,seed):
        root.mkdir()
        fixture=fixtures.ComparisonTests().fixture
        b=fixture(root/'baseline','baseline',seed);m=fixture(root/'metacognitive','metacognitive',seed)
        with patch('cup_baseline.compare.plot'):
            compare(b,m,root/'comparison')
        return root

    def test_seed_means_pairing_and_missing_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths=[self.build(root/str(s),s) for s in (0,1)]
            # Create a single reached seed; the other remains right-censored.
            path=paths[0]/'comparison/comparison.json';record=json.loads(path.read_text())
            item=record['n85'][0]
            item.update(reached=True,episodes=0,transitions=0,confirmed_at_episode=2)
            path.write_text(json.dumps(record))
            result=aggregate(paths,root/'aggregate')
            self.assertEqual(result['n_seeds'],2)
            sr=next(r for r in result['metrics'] if r['agent']=='baseline' and r['task']=='pick_cup'
                    and r['split']=='seen' and r['metric']=='success_pct')
            self.assertEqual(sr['mean'],50);self.assertEqual(sr['std'],0);self.assertEqual(sr['n_defined_seeds'],2)
            threshold=next(r for r in result['n85'] if r['agent']=='baseline' and r['task']=='pick_cup')
            self.assertEqual(threshold['reached_seeds'],1);self.assertEqual(threshold['censored_seeds'],1)
            self.assertEqual(threshold['mean_reached_episodes'],0)
            self.assertIsNone(threshold['mean_all_seeds_episodes'])
            self.assertEqual(threshold['censored_seed_ids'],[1])
            self.assertTrue((root/'aggregate/aggregate.png').exists())
            self.assertEqual({p.suffix for p in (root/'aggregate').iterdir()},{'.json','.csv','.png'})

    def test_duplicate_seed_and_changed_protocol_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);a=self.build(root/'a',0);b=self.build(root/'b',1)
            with self.assertRaisesRegex(ValueError,'exactly once'):aggregate([a,a],root/'bad')
            path=b/'comparison/comparison.json';record=json.loads(path.read_text());record['common_config']['updates']=999
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'across seeds'):aggregate([a,b],root/'bad')

    def test_all_unreached_and_single_reached_statistics_are_explicit(self):
        base=dict(agent='baseline',task='pick_cup',scope='continual',domain='B_passage_blocked',
                  reached=False,episodes=None,transitions=None,budget_episodes=20)
        row=threshold_summary([dict(base,seed=0),dict(base,seed=1)])[0]
        self.assertEqual(row['reached_fraction'],0);self.assertIsNone(row['mean_reached_episodes'])
        self.assertEqual(row['censored_seeds'],2)
        row=threshold_summary([dict(base,seed=0,reached=True,episodes=8,transitions=100)])[0]
        self.assertEqual(row['mean_all_seeds_episodes'],8);self.assertIsNone(row['std_reached_episodes'])

    def test_complete_seed_coverage_and_continual_protocol_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths=[self.build(root/str(s),s) for s in (0,1)]
            fixture=fixtures.ComparisonTests().continual_fixture
            for path in paths:
                bc=fixture(path/'baseline_cl',path/'baseline','baseline')
                mc=fixture(path/'metacognitive_cl',path/'metacognitive','metacognitive')
                with patch('cup_baseline.compare.plot'):
                    compare(path/'baseline',path/'metacognitive',path/'comparison',bc,mc)
            with patch('cup_baseline.aggregate.plot_aggregate'):
                result=aggregate(paths,root/'aggregate')
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['continual_complete_seed_ids'],[0,1])
            row=next(r for r in result['continual'] if r['table']=='adaptation' and r['agent']=='baseline'
                     and r['task']=='pick_cup' and r['domain']=='C_workspace_clutter' and r['metric']=='phase_gain_pp')
            self.assertEqual(row['mean'],-50.);self.assertEqual(row['n_seeds'],2)
            report=paths[1]/'comparison/comparison.json';data=json.loads(report.read_text())
            data['continual']['metacognitive']['protocol']['eval_n']=1
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Continual protocols'):aggregate(paths,root/'bad')


if __name__=='__main__':unittest.main()
