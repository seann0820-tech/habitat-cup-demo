"""Summarize paired experiments across independent training seeds.

Run after ``cup_baseline.compare`` has produced comparison/comparison.json in
all experiment directories. Each training seed contributes one observation per
agent/task/split. Episode counts are never treated as independent training seeds.
CSV tables use long form: identifiers, metric, mean, sample standard deviation,
and the number of seeds with a defined value. Standard deviation is undefined
for a single seed. N85 has a dedicated table because an unreached threshold is
right-censored; it must not be replaced by zero or silently dropped from a mean.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .compare import read, write_csv, TASKS


def stats(values):
    """Missing conditional metrics (e.g. no successful episodes) stay missing."""
    data=np.asarray([v for v in values if v is not None],dtype=float)
    if not np.all(np.isfinite(data)):raise ValueError('Non-finite summary values')
    return dict(mean=float(data.mean()) if len(data) else None,
                std=float(data.std(ddof=1)) if len(data)>1 else None,
                n_defined_seeds=len(data))


def summarize(records,keys,metrics):
    grouped=defaultdict(list)
    for row in records:grouped[tuple(row[k] for k in keys)].append(row)
    result=[]
    for key,rows in sorted(grouped.items(),key=lambda item:str(item[0])):
        seeds=[r['seed'] for r in rows]
        if len(set(seeds))!=len(seeds):raise ValueError('Duplicate seed within aggregate group')
        for metric in metrics:
            result.append(dict(zip(keys,key),metric=metric,n_seeds=len(seeds),
                **stats([r.get(metric) for r in rows])))
    return result


def threshold_summary(records):
    """Report threshold attainment and reached-only means together.

    ``mean_reached_episodes`` is conditional on attainment. It is not a valid
    overall speed ranking when reach rates differ. ``mean_all_seeds_episodes``
    is defined only when every seed reached the threshold within its budget.
    """
    grouped=defaultdict(list)
    for row in records:
        grouped[(row['agent'],row['task'],row['scope'],row.get('domain'))].append(row)
    result=[]
    for key,rows in sorted(grouped.items(),key=lambda item:str(item[0])):
        if len({r['seed'] for r in rows})!=len(rows):raise ValueError('Duplicate threshold seed')
        budgets={r['budget_episodes'] for r in rows}
        if len(budgets)!=1:raise ValueError('Threshold budgets differ across seeds')
        reached=[r for r in rows if r['reached']]
        censored=[r for r in rows if not r['reached']]
        for r in rows:
            if r['reached'] and (r['episodes'] is None or r['episodes']<0):
                raise ValueError('Reached N85 needs a non-negative episode count')
            if not r['reached'] and r['episodes'] is not None:
                raise ValueError('Censored N85 must have episodes=null')
        ep=stats([r['episodes'] for r in reached]);steps=stats([r.get('transitions') for r in reached])
        result.append(dict(zip(('agent','task','scope','domain'),key),n_seeds=len(rows),
            reached_seeds=len(reached),censored_seeds=len(censored),reached_fraction=len(reached)/len(rows),
            already_at_threshold_seeds=sum(r['episodes']==0 for r in reached),
            budget_episodes=next(iter(budgets)),
            mean_reached_episodes=ep['mean'],std_reached_episodes=ep['std'],
            mean_all_seeds_episodes=ep['mean'] if not censored else None,
            mean_reached_train_steps=steps['mean'],std_reached_train_steps=steps['std'],
            reached_seed_ids=[r['seed'] for r in reached],censored_seed_ids=[r['seed'] for r in censored],
            note='Unreached N85 is > budget, not zero. Reached-only means are conditional; compare reach rates first.'))
    return result


def _numeric_columns(rows,exclude):
    return sorted({key for row in rows for key,value in row.items()
                   if key not in exclude and (value is None or isinstance(value,(int,float)))})


def aggregate(experiments,out):
    """Read one comparison per directory and reject incompatible protocols."""
    roots=[Path(p) for p in experiments]
    if not roots:raise ValueError('Provide at least one experiment directory')
    reports=[read(path/'comparison'/'comparison.json') for path in roots]
    seeds=[r['seed'] for r in reports]
    if len(set(seeds))!=len(seeds):raise ValueError('Each training seed must appear exactly once')
    for r in reports:
        if r.get('schema')!='paired_comparison_v2':raise ValueError('Regenerate comparisons with the current code before aggregation')
        if r['status'] not in ('complete','basic_complete_continual_pending'):
            raise ValueError('A comparison is unfinished')
        for key in ('algorithm_version','dynamics','train_episodes','common_config','agent_config'):
            if r[key]!=reports[0][key]:raise ValueError('Unequal experiment setting across seeds: '+key)
        expected={(a,t,s) for a in ('baseline','metacognitive') for t in TASKS for s in ('seen','unseen')}
        found=[(x['agent'],x['task'],x['split']) for x in r['metrics']]
        if len(found)!=len(expected) or set(found)!=expected:
            raise ValueError('Each seed must contain both agents, both tasks and both evaluation splits')
    complete=[r for r in reports if r['status']=='complete']
    if complete:
        protocol=complete[0]['continual']['baseline']['protocol']
        for r in complete:
            for agent in ('baseline','metacognitive'):
                if r['continual'][agent]['protocol']!=protocol:
                    raise ValueError('Continual protocols differ across training seeds')
    def records(field,subset=reports):
        return [dict(row,seed=r['seed']) for r in subset for row in r.get(field,[])]
    rows=records('metrics')
    metric_columns=_numeric_columns(rows,{'seed','n','n85_train_steps','n85_total_episodes'})
    metric_summary=summarize(rows,('agent','task','split'),metric_columns)
    paired_summary=summarize(records('paired'),('task','split'),
        ('success_difference_pp','steps_difference','nn_gflops_episode_difference'))
    n85=threshold_summary(records('n85'))
    adaptation=summarize(records('continual_adaptation',complete),('agent','task','domain'),
        ('auc','frozen_initial_sr','phase_start_sr','phase_after_sr','phase_gain_pp','total_gain_from_frozen_pp'))
    retention=summarize(records('continual_retention',complete),('agent','task'),
        ('final_average_success','backward_transfer','average_forgetting'))
    for row in adaptation:row['table']='adaptation'
    for row in retention:row['table']='retention'
    continual=adaptation+retention
    continual_pairs=summarize(records('continual_paired',complete),('task','after_phase','domain'),
        ('success_difference_pp','steps_difference','nn_gflops_episode_difference'))
    costs=[]
    for r in reports:
        for agent,c in r['training_cost'].items():
            costs.append(dict(seed=r['seed'],agent=agent,scope='initial_training',domain=None,
                episodes=c['training_episodes'],steps=c['environment_steps'],
                total_gflops=c['total_nn_flops']/1e9,
                meta_inference_gflops=c['meta_inference_nn_flops']/1e9,
                meta_update_gflops=c['meta_update_nn_flops']/1e9,
                world_update_gflops=c['update_nn_flops']/1e9))
        for agent,c in r['evaluation_cost'].items():
            costs.append(dict(seed=r['seed'],agent=agent,scope='evaluation_completed_calls',domain=None,
                episodes=c['episodes'],steps=c['steps'],total_gflops=c['nn_flops']/1e9))
        for c in r.get('continual_costs',[]):
            costs.append(dict(c,seed=r['seed'],scope='continual',episodes=c['training_episodes']))
        if r['status']=='complete':
            for agent,initial in r['training_cost'].items():
                phases=[c for c in r['continual_costs'] if c['agent']==agent]
                total=dict(seed=r['seed'],agent=agent,scope='continual_total',domain=None,
                    episodes=sum(c['training_episodes'] for c in phases),steps=sum(c['steps'] for c in phases))
                for key in ('total_gflops','meta_inference_gflops','meta_update_gflops','world_update_gflops'):
                    total[key]=sum(c[key] for c in phases)
                costs.append(total)
                all_training=dict(total,scope='all_training',episodes=total['episodes']+initial['training_episodes'],
                                  steps=total['steps']+initial['environment_steps'])
                for output,source in [('total_gflops','total_nn_flops'),('meta_inference_gflops','meta_inference_nn_flops'),
                                      ('meta_update_gflops','meta_update_nn_flops'),('world_update_gflops','update_nn_flops')]:
                    all_training[output]+=initial[source]/1e9
                costs.append(all_training)
    cost_summary=summarize(costs,('agent','scope','domain'),
        ('episodes','steps','total_gflops','meta_inference_gflops','meta_update_gflops','world_update_gflops'))
    calibration_rows=records('calibration')
    calibration=summarize(calibration_rows,('task','split'),_numeric_columns(calibration_rows,{'seed'}))
    local_monitoring=summarize(records('local_monitoring_table'),('agent','task','split','diagnostic'),('value',))
    result=dict(schema='seed_aggregate_v2',status='complete' if len(complete)==len(reports) else 'continual_pending_for_some_seeds',
        seed_ids=seeds,n_seeds=len(seeds),continual_complete_seed_ids=[r['seed'] for r in complete],
        algorithm_version=reports[0]['algorithm_version'],dynamics=reports[0]['dynamics'],
        train_episodes=reports[0]['train_episodes'],common_config=reports[0]['common_config'],
        agent_config=reports[0]['agent_config'],metrics=metric_summary,paired=paired_summary,n85=n85,
        continual=continual,continual_paired=continual_pairs,costs=cost_summary,calibration=calibration,
        local_monitoring=local_monitoring,
        notes=[
            'One independently trained seed contributes one observation; no pooling episodes as training replicates.',
            'Std is the sample standard deviation across seeds (ddof=1); null when fewer than two defined seeds.',
            'Paired differences are metacognitive minus baseline within each training seed before aggregation.',
            'Episode-bootstrap intervals are not averaged. This table makes no significance claim.',
            'Local monitoring summaries are averaged across training seeds; repeated actions are correlated within episodes.',
            'N85 counts total training episodes across both tasks; zero means already above threshold at the initial confirmed checkpoint.',
            'Unreached N85 is right-censored. Reach counts and conditional reached-only means must be read together.',
            'Continual rows use only explicitly listed completed seeds; finish every seed for the primary comparison.',
            'Evaluation cost includes completed evaluation calls and reruns recorded in each ledger; do not mix it into N85.',
            'FLOPs estimate neural-network arithmetic, not electricity. Simulation, geometry and other CPU work are excluded.',
        ])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'aggregate.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for name in ('metrics','paired','n85','continual','continual_paired','costs','calibration','local_monitoring'):
        write_csv(out/('aggregate_'+name+'.csv'),result[name])
    plot_aggregate(result,out)
    print('多 seed 汇总：',out/'aggregate.json','训练 seed 数：',len(seeds),flush=True)
    return result


def plot_aggregate(result,out):
    """Error bars describe seed standard deviation, not a confidence interval."""
    import os
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    keys=[(t,s) for t in TASKS for s in ('seen','unseen')];x=np.arange(len(keys))
    for ax,metric,label in zip(axes,('success_pct','nn_gflops_episode'),('Success (%)','NN GFLOPs / episode')):
        for j,agent in enumerate(('baseline','metacognitive')):
            index={(r['task'],r['split']):r for r in result['metrics'] if r['agent']==agent and r['metric']==metric}
            values=[index[k]['mean'] for k in keys];errors=[index[k]['std'] or 0 for k in keys]
            ax.bar(x+(j-.5)*.36,values,.36,yerr=errors,capsize=3,label=agent)
        ax.set_xticks(x,[t+'\n'+s for t,s in keys]);ax.tick_params(axis='x',labelsize=8)
        ax.set_ylabel(label);ax.legend(fontsize=8)
    fig.suptitle('Mean across training seeds; error bars = sample SD (N='+str(result['n_seeds'])+')')
    fig.tight_layout();fig.savefig(Path(out)/'aggregate.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments',nargs='+',required=True,help='Experiment directories containing comparison/comparison.json')
    parser.add_argument('--out',required=True,help='Directory for aggregate JSON, CSV tables and PNG')
    args=parser.parse_args();aggregate(args.experiments,args.out)


if __name__=='__main__':main()
