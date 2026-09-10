"""Create paired CSV/JSON tables and a scientific plot from completed run files.

One invocation compares a baseline and a metacognitive model trained with the
same seed. Evaluation episodes and continual geometry must match. Continual
results remain explicitly pending until both mandatory A/B/C protocols finish.
Use ``python -m cup_baseline.aggregate`` to summarize independent training seeds.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

TASKS=('pick_cup','open_drawer')
LABELS={'pick_cup':'取杯','open_drawer':'开抽屉'}
DOMAINS=('A_clear','B_passage_blocked','C_workspace_clutter')
COMMON_SETTINGS=('replay_limit','horizon','candidates','iterations','elite','updates',
                 'max_steps','warmup_episodes','evaluate_every','validation_n','test_n')


def read(path):
    return json.loads(Path(path).read_text())


def threshold(r):
    return str(r['episodes']) if r['reached'] else r['display']


def write_csv(path,rows):
    """Use a union of columns so optional diagnostics remain visible."""
    if not rows:return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def numeric_diagnostics(value,prefix=''):
    """Flatten named local scores without conflating their observation units."""
    if isinstance(value,dict):
        for key,item in value.items():
            yield from numeric_diagnostics(item,prefix+'.'+key if prefix else key)
    elif isinstance(value,(int,float)) or value is None:
        if prefix:yield prefix,value


def paired_rows(a,b,task):
    def load(path):
        with Path(path).open() as f:rows=[r for r in csv.DictReader(f) if r['task']==task]
        keyed={(r['scene'],r['seed'],r['task'],r['shift']):r for r in rows}
        if len(keyed)!=len(rows):raise ValueError('Duplicate evaluation episode identities')
        return keyed
    x,y=load(a),load(b)
    if not x or x.keys()!=y.keys():raise ValueError('Evaluation episodes do not match; comparison refused')
    d=np.array([float(y[k]['success'])-float(x[k]['success']) for k in sorted(x)])
    rng=np.random.default_rng(811)
    boot=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)*100
    # Means include failed attempts. These differences share the same pairing;
    # only success gets an episode-bootstrap interval in this table.
    steps=np.array([float(y[k]['steps'])-float(x[k]['steps']) for k in sorted(x)])
    costs=np.array([float(y[k]['total_nn_flops_est'])-float(x[k]['total_nn_flops_est']) for k in sorted(x)])/1e9
    return dict(n=len(d),success_difference_pp=float(d.mean()*100),
        baseline_success_pct=float(np.mean([float(x[k]['success']) for k in sorted(x)])*100),
        metacognitive_success_pct=float(np.mean([float(y[k]['success']) for k in sorted(y)])*100),
        steps_difference=float(steps.mean()),nn_gflops_episode_difference=float(costs.mean()),
        episode_bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
        note='Paired episode bootstrap conditional on this trained seed and these layouts; not seed-to-seed uncertainty.')


def paired_geometry(a,b):
    """Seeds alone are insufficient when a generator may resample a start."""
    fields=('scene','seed','task','domain','generation_version','target','target_object',
            'start','start_yaw','nav_goal','shortest_distance','added_boxes','drawer_link')
    def normalized(path):
        rows=read(path)
        if isinstance(rows,dict):rows=[rows]
        def clean(value):
            if isinstance(value,float):return round(value,5)
            if isinstance(value,list):return [clean(x) for x in value]
            if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
            return value
        return sorted(json.dumps({k:clean(r.get(k)) for k in fields},sort_keys=True) for r in rows)
    if normalized(a)!=normalized(b):
        raise ValueError('Paired episode geometry differs: '+str(a)+' / '+str(b))


def compare(baseline,metacognitive,out,baseline_cl=None,metacognitive_cl=None):
    baseline,metacognitive,out=map(Path,(baseline,metacognitive,out))
    sources={'baseline':read(baseline/'summary.json'),'metacognitive':read(metacognitive/'summary.json')}
    b,m=sources.values()
    if b.get('agent')!='baseline' or m.get('agent')!='metacognitive':
        raise ValueError('Expected a baseline run and a metacognitive run')
    for key in ('seed','dynamics','train_episodes','algorithm_version'):
        if b[key]!=m[key]:raise ValueError('Unequal comparison setting: '+key)
    for key in COMMON_SETTINGS:
        if b['config'][key]!=m['config'][key]:raise ValueError('Unequal comparison budget/protocol: '+key)
    out.mkdir(parents=True,exist_ok=True);metrics=[];pairs=[];calibration=[];thresholds=[]
    for agent,r in sources.items():
        for task in TASKS:
            thresholds.append(dict(agent=agent,task=task,scope='initial_training',domain=None,**r['n85'][task]))
    for split in ('seen','unseen'):
        manifest=f'{split}_ep{b["train_episodes"]:04d}_episodes.json'
        if (baseline/manifest).exists() or (metacognitive/manifest).exists():
            if not (baseline/manifest).exists() or not (metacognitive/manifest).exists():
                raise ValueError('A paired final-evaluation manifest is missing')
            paired_geometry(baseline/manifest,metacognitive/manifest)
        for task in TASKS:
            pair=paired_rows(baseline/f'{split}_ep{b["train_episodes"]:04d}.csv',
                             metacognitive/f'{split}_ep{m["train_episodes"]:04d}.csv',task)
            pairs.append(dict(split=split,task=task,**pair))
            for agent,r in sources.items():
                s=r[split]['by_task'][task]
                if s['n']!=pair['n'] or not np.isclose(100*s['success_rate'],pair[agent+'_success_pct']):
                    raise ValueError('Summary and evaluation CSV disagree: '+agent+' '+task+' '+split)
                metrics.append(dict(agent=agent,task=task,split=split,n=s['n'],
                    success_pct=100*s['success_rate'],steps=s['mean_steps'],
                    steps_success_only=s.get('mean_steps_success_only'),
                    nav_success_pct=100*s['nav_success_rate'],
                    collision_steps=s.get('mean_collision_steps'),
                    manipulation_collision_steps=s.get('mean_manipulation_collision_steps'),
                    nn_gflops_episode=s['mean_total_nn_flops_est']/1e9,
                    nn_gflops_per_success=(s['total_nn_flops_per_success_est']/1e9 if s['total_nn_flops_per_success_est'] is not None else None),
                    nn_batch_forwards=s['mean_total_nn_forward_calls'],
                    meta_gflops_episode=s['mean_meta_nn_flops_est']/1e9,
                    n85_total_episodes=threshold(r['n85'][task]),
                    n85_train_steps=r['n85'][task]['transitions'],
                    generalization_gap_pp=r['generalization_gap_pp'][task]))
                if agent=='metacognitive':
                    decisions=s['mean_meta_decisions']
                    calibration.append(dict(task=task,split=split,**s['monitoring'],
                        execute_fraction=s['mean_meta_execute']/decisions if decisions else None,
                        expand_fraction=s['mean_meta_expand']/decisions if decisions else None,
                        restart_fraction=s['mean_meta_restart']/decisions if decisions else None,
                        baseline_fraction=s.get('mean_meta_baseline',0)/decisions if decisions else None))
    continual=None
    if bool(baseline_cl)!=bool(metacognitive_cl):raise ValueError('Supply both continual result directories')
    if baseline_cl:
        continual={}
        for agent,path,run in [('baseline',baseline_cl,baseline),('metacognitive',metacognitive_cl,metacognitive)]:
            c=read(Path(path)/'continual_summary.json')
            digest=hashlib.sha256((run/'checkpoint.pt').read_bytes()).hexdigest()
            if c['checkpoint_sha256']!=digest:raise ValueError('Continual result belongs to a different checkpoint')
            if c.get('agent')!=agent:raise ValueError('Wrong agent in continual result')
            continual[agent]=c
        if continual['baseline']['protocol']!=continual['metacognitive']['protocol']:
            raise ValueError('Continual protocols must match')
        if continual['baseline']['protocol'].get('generation_version'):
            # Includes training, adaptation, frozen reference and retention.
            left,right=Path(baseline_cl),Path(metacognitive_cl)
            names={p.name for p in left.glob('*_episodes.json')} | {p.name for p in left.glob('phase*_train_episode*.json')}
            other={p.name for p in right.glob('*_episodes.json')} | {p.name for p in right.glob('phase*_train_episode*.json')}
            if not names or names!=other:
                raise ValueError('Continual episode manifests are incomplete or unequal')
            for name in sorted(names):paired_geometry(left/name,right/name)
    # Readiness is not a calibration guarantee. The first eligible phase
    # forecast and action-local phase/error diagnostics have different units
    # of observation and remain separate.
    local_monitoring={agent:{split:{task:r[split]['by_task'][task].get('local_monitoring')
        for task in TASKS} for split in ('seen','unseen')} for agent,r in sources.items()}
    local_rows=[]
    for agent,splits in local_monitoring.items():
        for split,tasks in splits.items():
            for task,diagnostics in tasks.items():
                for key,value in numeric_diagnostics(diagnostics):
                    local_rows.append(dict(agent=agent,task=task,split=split,diagnostic=key,value=value))
    adaptation=[];retention=[];continual_costs=[];continual_pairs=[]
    if continual:
        for agent,c in continual.items():
            for task in TASKS:
                retention.append(dict(agent=agent,task=task,**c['retention'][task]))
                for i,d in enumerate(c['adaptation']):
                    frozen=c['frozen_reference'][i]['by_task'][task]['success_rate']
                    # Compare phase updates on the same retention test episodes.
                    # Frozen means before *all* A/B/C updates. Phase-start means
                    # after previous phases; conflating them inflates adaptation.
                    phase_start=frozen if i==0 else c['matrix'][task][i-1][i]
                    after=c['matrix'][task][i][i]
                    adaptation.append(dict(agent=agent,task=task,domain=DOMAINS[i],
                        n85=threshold(d['n85'][task]),auc=d['learning_curve_auc'][task],
                        frozen_initial_sr=frozen,phase_start_sr=phase_start,phase_after_sr=after,
                        phase_gain_pp=100*(after-phase_start),total_gain_from_frozen_pp=100*(after-frozen)))
                    thresholds.append(dict(agent=agent,task=task,scope='continual',domain=DOMAINS[i],**d['n85'][task]))
            for i,d in enumerate(c['costs']):
                continual_costs.append(dict(agent=agent,domain=DOMAINS[i],
                    training_episodes=d['training_episodes'],steps=d['environment_steps'],
                    total_gflops=d['total_nn_flops']/1e9,
                    meta_inference_gflops=d['meta_inference_nn_flops']/1e9,
                    meta_update_gflops=d['meta_update_nn_flops']/1e9,
                    world_update_gflops=d['update_nn_flops']/1e9))
        for phase in range(3):
            for domain in range(3):
                for task in TASKS:
                    stem=f'after{phase}_domain{domain}_ep{continual["baseline"]["protocol"]["phase_episodes"]:04d}.csv'
                    continual_pairs.append(dict(after_phase=DOMAINS[phase],domain=DOMAINS[domain],task=task,
                        **paired_rows(Path(baseline_cl)/stem,Path(metacognitive_cl)/stem,task)))
    ledgers={}
    for agent,run,cl in [('baseline',baseline,baseline_cl),('metacognitive',metacognitive,metacognitive_cl)]:
        entries=[]
        for folder in [run]+([Path(cl)] if cl else []):
            path=folder/'evaluation_ledger.jsonl'
            if path.exists():entries += [json.loads(s) for s in path.read_text().splitlines() if s]
        ledgers[agent]={k:sum(e[k] for e in entries) for k in ('episodes','steps','nn_flops')}
    result=dict(schema='paired_comparison_v2',status='complete' if continual else 'basic_complete_continual_pending',seed=b['seed'],
        algorithm_version=b['algorithm_version'],dynamics=b['dynamics'],train_episodes=b['train_episodes'],
        common_config={key:b['config'][key] for key in COMMON_SETTINGS},
        agent_config={agent:r['config'] for agent,r in sources.items()},
        metrics=metrics,paired=pairs,calibration=calibration,continual=continual,
        n85=thresholds,continual_adaptation=adaptation,continual_retention=retention,
        continual_costs=continual_costs,continual_paired=continual_pairs,local_monitoring=local_monitoring,
        local_monitoring_table=local_rows,
        training_cost={a:r['training_cost'] for a,r in sources.items()},evaluation_cost=ledgers,
        monitor=sources['metacognitive']['metacognition'],
        limits=['Native Habitat results only; no generated/example results.',
            'One seed is a pilot; paired episode intervals do not measure training-seed variability.',
            'FLOPs are NN estimates, not joules; geometry, simulation, CEM, IK and CPU overhead are excluded.',
            'Meta controller learns planning/execution regulation only; dynamics updates and FIFO remain fixed.',
            'Monitor replay and calibration memory are included in the monitor summary; they are additional memory.',
            'Confidence prediction quality and control benefit are separate questions; neither is assumed.',
            'Phase-completion forecasts use phase outcomes, not final episode outcomes.',
            'Local action records within an episode are correlated; do not treat them as independent replicates.',
            'All actual model calls count, including initial search, extensions, restarts and monitor inference.',
            'N85 uses validation checkpoints with confirmation; final tests do not determine N85.',
            'Successful-only steps describe a selected subset; all-attempt costs include failed episodes.',
            'A/B/C are geometric changes; strong frozen performance can leave little learning improvement.'])
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name,rows in [('comparison',metrics),('paired',pairs),('calibration',calibration),('n85',thresholds),
                      ('local_monitoring',local_rows),
                      ('continual_adaptation',adaptation),('continual_retention',retention),
                      ('continual_costs',continual_costs),('continual_paired',continual_pairs)]:
        write_csv(out/(name+'.csv'),rows)
    plot(sources,baseline,metacognitive,out)
    print('对比数据：',out/'comparison.json','状态：',result['status'],flush=True)
    return result


def plot(sources,baseline,meta,out):
    import os
    os.environ['MPLBACKEND']='Agg'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for agent,path in [('baseline',baseline),('metacognitive',meta)]:
        curve=read(path/'validation_curve.json')
        for j,t in enumerate(TASKS):
            axes[0,j].plot([x['train_episodes'] for x in curve],
                          [100*x['by_task'][t]['success_rate'] for x in curve],label=agent)
    for j,t in enumerate(TASKS):
        axes[0,j].axhline(85,color='gray',ls='--');axes[0,j].set(title=t,xlabel='Training episodes (both tasks)',ylabel='Validation success %',ylim=(0,105));axes[0,j].legend()
    labels=[f'{t}\n{s}' for t in TASKS for s in ('seen','unseen')];x=np.arange(4)
    for j,(agent,r) in enumerate(sources.items()):
        sr=[100*r[s]['by_task'][t]['success_rate'] for t in TASKS for s in ('seen','unseen')]
        cost=[r[s]['by_task'][t]['mean_total_nn_flops_est']/1e9 for t in TASKS for s in ('seen','unseen')]
        axes[1,0].bar(x+(j-.5)*.36,sr,.36,label=agent)
        axes[1,1].bar(x+(j-.5)*.36,cost,.36,label=agent)
    for ax in axes[1]:ax.set_xticks(x,labels);ax.tick_params(axis='x',labelsize=8);ax.legend()
    axes[1,0].set(ylabel='Final success %',ylim=(0,105));axes[1,1].set(ylabel='Total NN GFLOPs / episode')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    for name in ('baseline','metacognitive','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--baseline-cl');p.add_argument('--metacognitive-cl')
    a=p.parse_args();compare(a.baseline,a.metacognitive,a.out,a.baseline_cl,a.metacognitive_cl)


if __name__=='__main__':main()
