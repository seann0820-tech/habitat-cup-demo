"""Experiment entry points for fixed MPC and process-monitoring metacontrol.

train: learn on A layouts with matched seeds and environment budgets.
check_domains: validate every distinct A/B/C initial condition geometrically.
continual: continue each checkpoint through A, B, C and measure adaptation,
retention and all training NN costs. Evaluation never updates model or buffers.
shadow: score a frozen monitor on one fixed baseline policy's trajectories.
record: optional native rendering; it is separate from scored experiments.

Results are CSV/JSON and scientific PNG figures. Website design is intentionally
outside the training pipeline. This module never creates HTML or publishes data.
"""
import argparse
import csv
import json
import os
os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path
import platform
import time
from dataclasses import asdict

import numpy as np
import torch
from .tasks import HomeEnv as CupEnv, SCHEMA
from .generation import GENERATION_VERSION
from .protocol import TASKS, DOMAINS, schedule, task_summary, threshold_report, adaptation_summary, cost_summary
from .model import Ensemble, MPC, Compute
from .metrics import summarize, n85, continual_metrics
from .metacognition import MetaMonitor, monitoring_metrics, diagnostic_metrics

TRAIN_SCENES = ['v3_sc0_staging_00', 'v3_sc0_staging_01', 'v3_sc1_staging_00', 'v3_sc1_staging_01']
UNSEEN_SCENES = ['v3_sc2_staging_00', 'v3_sc3_staging_00']
PRESETS = {
    'smoke': dict(episodes=4, evaluate_every=2, validation_n=2, test_n=2, updates=4, max_steps=140, horizon=5, candidates=24, iterations=1, elite=4),
    'pilot': dict(episodes=60, evaluate_every=10, validation_n=10, test_n=20, updates=40, max_steps=240, horizon=8, candidates=64, iterations=2, elite=8),
    'research': dict(episodes=300, evaluate_every=20, validation_n=100, test_n=100, updates=80, max_steps=240, horizon=12, candidates=128, iterations=3, elite=16),
}


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='') as f:
        # Detailed trace arrays live in JSON; CSV keeps scalar columns readable.
        fields=list(dict.fromkeys(k for row in rows for k,v in row.items()
                                  if not isinstance(v,(list,dict))))
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader(); w.writerows(rows)


def episode(env, model, planner, scene, seed, learn=False, shift=0, video=None, task="pick_cup"):
    started = time.perf_counter()
    try:
        obs = env.reset(scene, seed, shift=shift, task=task)
    except RuntimeError as error:
        raise RuntimeError(f'Episode reset failed: scene={scene}, seed={seed}, task={task}, domain={DOMAINS[shift]}; {error}') from error
    planner.reset(seed+17)
    model.count = Compute()
    if hasattr(planner,'start_episode'):
        planner.start_episode(learn,env.max_steps)
    manifest = env.manifest.copy()
    frames = []
    for _ in range(env.max_steps):
        if video is not None:
            frames.append(env.render())
        action = planner.act(obs, exploration=0.10 if learn else 0.)
        nxt, done, info = env.step(action)
        if hasattr(planner,'observe_transition'):
            planner.observe_transition(obs,action,nxt,info)
        if learn:
            model.add(obs, action, nxt)
        obs = nxt
        if done:
            break
    if video is not None:
        import imageio.v2 as imageio
        frames.append(env.render())
        imageio.mimsave(str(video), frames, fps=6)
    row = dict(scene=scene, seed=int(seed), shift=shift, **info, **asdict(model.count),
               wall_seconds=time.perf_counter()-started)
    row['total_nn_flops_est']=row['nn_flops_est']+row.get('meta_nn_flops_est',0)
    row['total_nn_forward_calls']=row['nn_forward_calls']+row.get('meta_nn_forward_calls',0)
    if hasattr(planner,'finish_episode'):
        row.update(planner.finish_episode(info['success'],info['steps']))
        manifest['metacognition_trace']=planner.trace
    return row, manifest


def evaluate(env,model,planner,scenes,n,seed_offset,split,out,train_episodes,shift=0):
    if model.collect_with_prior:raise RuntimeError('Disable warm-up during evaluation')
    rows=[];manifests=[]
    for task in TASKS:
        for i in range(n):
            row,manifest=episode(env,model,planner,scenes[i%len(scenes)],seed_offset+i,task=task,shift=shift)
            rows.append(row);manifests.append(manifest)
        print(f'{split}/{task}: {n} episodes, SR={np.mean([r["success"] for r in rows if r["task"]==task]):.3f}',flush=True)
    stem=f'{split}_ep{train_episodes:04d}'
    save_csv(out/(stem+'.csv'),rows);dump(out/(stem+'_episodes.json'),manifests)
    with (out/'evaluation_ledger.jsonl').open('a') as ledger:
        ledger.write(json.dumps(dict(split=split,train_episodes=train_episodes,episodes=len(rows),
            steps=sum(r['steps'] for r in rows),
            nn_flops=sum(r['total_nn_flops_est'] for r in rows)))+'\n')
    return task_summary(rows)

def create(args, config):
    env = CupEnv(args.data, render=getattr(args, 'render', False), max_steps=config['max_steps'])
    model = Ensemble(args.seed, mode=config.get('dynamics', 'residual'), replay_limit=config.get('replay_limit',4096))
    planner_class=MPC
    if config.get('agent','baseline')=='metacognitive':
        from .meta_planner import MetaMPC
        model.monitor=MetaMonitor(args.seed,compute_weight=config.get('meta_compute_weight',.10),
                                  step_weight=config.get('meta_step_weight',.05))
        planner_class=MetaMPC
    planner = planner_class(model, **{k: config[k] for k in ('horizon', 'candidates', 'iterations', 'elite')})
    return env, model, planner


def load_checkpoint(path, **kwargs):
    saved=torch.load(path, **kwargs)
    if saved.get('schema')!=SCHEMA:
        raise ValueError('Checkpoint task/state schema is incompatible; use the current notebook to train in a new output directory')
    return saved


def checkpoint(path, model, **state):
    temp = Path(str(path)+'.tmp')
    torch.save(dict(schema=SCHEMA, model=model.snapshot(), **state), temp)
    temp.replace(path)


def plot_curve(curve,result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,3.8))
    for task in TASKS:
        ax[0].plot([r['train_episodes'] for r in curve],[100*r['by_task'][task]['success_rate'] for r in curve],label=task)
    ax[0].axhline(85,color='gray',ls='--');ax[0].legend();ax[0].set(xlabel='Total training episodes',ylabel='Validation success (%)',ylim=(0,105))
    x=np.arange(2)
    for k,split in enumerate(('seen','unseen')):
        ax[1].bar(x+k*.35,[100*result[split]['by_task'][t]['success_rate'] for t in TASKS],width=.35,label=split)
    ax[1].set_xticks(x+.175,TASKS);ax[1].set(ylabel='Test success (%)',ylim=(0,105));ax[1].legend()
    fig.tight_layout();fig.savefig(out/'learning_curve.png',dpi=150);plt.close(fig)

def train(args):
    cfg = PRESETS[args.preset].copy()
    cfg['dynamics'] = args.dynamics
    cfg['replay_limit'] = args.replay_limit
    cfg['schema'] = SCHEMA
    cfg['agent']=args.agent
    cfg['meta_compute_weight']=args.meta_compute_weight
    cfg['meta_step_weight']=args.meta_step_weight
    cfg['algorithm_version']='meta_comparison_v2'
    cfg['warmup_episodes'] = 4 if args.dynamics == 'learned' else 0
    if args.episodes is not None:
        cfg['episodes'] = args.episodes
    if cfg['episodes'] < 2 or cfg['episodes'] % 2 or cfg['replay_limit'] < 128:
        raise ValueError('episodes must be positive even counts; replay-limit >=128')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    state_path = out/'checkpoint.pt'
    if state_path.exists() and not args.resume:
        raise RuntimeError('Output has checkpoint.pt; use --resume or choose a new --out directory')
    env, model, planner = create(args, cfg)
    curve, training = [], []
    first = 1; train_steps = 0
    if args.resume:
        if not state_path.exists():
            raise FileNotFoundError('No checkpoint to resume')
        saved = load_checkpoint(state_path, map_location='cpu', weights_only=False)
        if saved['config'].get('algorithm_version')!=cfg['algorithm_version']:
            raise ValueError('旧实验缺少本版计费/元模型协议；请保留旧结果，在新目录开始两组比较')
        saved['config'].setdefault('dynamics', 'residual')
        saved['config'].setdefault('warmup_episodes', 0)
        for key,value in dict(agent='baseline',meta_compute_weight=.10,meta_step_weight=.05,
                              algorithm_version='meta_comparison_v2').items():
            saved['config'].setdefault(key,value)
        if cfg['episodes'] < saved['episode']:
            raise ValueError('Total episode budget cannot be below the completed checkpoint episode')
        for key in cfg:
            if key != 'episodes' and cfg[key] != saved['config'][key]:
                raise ValueError('Cannot resume with changed '+key)
        if args.seed != saved['seed']:
            raise ValueError('Resume seed must match checkpoint seed')
        model.restore(saved['model']); curve = saved['curve']; training = saved['training']
        for row in training:
            row.setdefault('warmup', False)
        first = saved['episode']+1; train_steps = saved['train_steps']
    import habitat_sim
    run_meta = dict(schema=SCHEMA,tasks=TASKS,domains=DOMAINS,config=cfg, preset=args.preset, seed=args.seed, python=platform.python_version(),
                    habitat_sim=habitat_sim.__version__, torch=torch.__version__, numpy=np.__version__,
                    observation='privileged state, task ID, twelve rays and added obstacle boxes',
                    grasp='assisted cup/handle attachment; native drawer joint driven by EE motion; not force grasping', model=cfg['dynamics']+'-ensemble-CEM-MPC',
                    train_scenes=TRAIN_SCENES, unseen_scenes=UNSEEN_SCENES)
    dump(out/'run_config.json', run_meta)
    try:
        if not curve:
            s = evaluate(env, model, planner, TRAIN_SCENES, cfg['validation_n'], 150000,
                         'validation', out, 0)
            curve.append(dict(train_episodes=0, train_steps=0, **s))
        for ep in range(first, cfg['episodes']+1):
            model.collect_with_prior = ep <= cfg['warmup_episodes']
            row, manifest = episode(env, model, planner, TRAIN_SCENES[((ep-1)//2)%len(TRAIN_SCENES)],
                                     args.seed*10000+ep, learn=True, task=TASKS[(ep-1)%2])
            row['warmup'] = model.collect_with_prior
            model.collect_with_prior = False
            train_steps += row['steps']
            t0 = time.perf_counter()
            fit_loss = model.fit(cfg['updates'])
            row.update(episode=ep, fit_loss=fit_loss, fit_seconds=time.perf_counter()-t0,
                       train_steps_total=train_steps, training_flops_total_est=model.train_flops_est)
            row['meta_training_flops_total_est']=model.monitor.train_flops_est if model.monitor else 0
            training.append(row)
            dump(out/f'train_episode_{ep:04d}.json', manifest)
            print(f'train {ep}/{cfg["episodes"]}: task={row["task"]}, success={row["success"]}, steps={row["steps"]}, loss={fit_loss}', flush=True)
            if ep % cfg['evaluate_every'] == 0 or ep == cfg['episodes']:
                s = evaluate(env, model, planner, TRAIN_SCENES, cfg['validation_n'], 150000,
                             'validation', out, ep)
                curve.append(dict(train_episodes=ep, train_steps=train_steps, **s))
                dump(out/'validation_curve.json', curve)
            save_csv(out/'training.csv', training)
            checkpoint(state_path, model, config=cfg, seed=args.seed, episode=ep, train_steps=train_steps,
                       curve=curve, training=training)
        seen = evaluate(env, model, planner, TRAIN_SCENES, cfg['test_n'], 250000, 'seen', out, cfg['episodes'])
        unseen = evaluate(env, model, planner, UNSEEN_SCENES, cfg['test_n'], 350000, 'unseen', out, cfg['episodes'])
        result = dict(schema=SCHEMA,preset=args.preset, agent=cfg['agent'],algorithm_version=cfg['algorithm_version'],
                      config=cfg,dynamics=cfg['dynamics'], warmup_episodes=cfg['warmup_episodes'],
                      seed=args.seed, train_episodes=cfg['episodes'], train_steps=train_steps,
                      n85=threshold_report(curve, cfg['episodes']), seen=seen, unseen=unseen,
                      generalization_gap_pp={t:100*(seen['by_task'][t]['success_rate']-unseen['by_task'][t]['success_rate']) for t in TASKS},
                      train_mlp_flops_total_est=model.train_flops_est,
                      train_collection_mlp_flops_total_est=sum(r['nn_flops_est'] for r in training),
                      training_cost=cost_summary(training,model.train_flops_est,
                          model.monitor.train_flops_est if model.monitor else 0),
                      metacognition=model.monitor.summary() if model.monitor else None,
                      continual_learning='required_pending',
                      caveat='NN FLOPs include meta monitoring and training, but exclude simulator, rendering, IK, geometry, CEM arithmetic and optimizer arithmetic. Not measured energy. Meta TD memory and episode-disjoint calibration memory are reported separately; raw dynamics replay capacity is matched.')
        dump(out/'summary.json', result)
        dump(out/'validation_curve.json', curve)
        save_csv(out/'metrics.csv', [dict(split=name,task=task,**{k:v for k,v in summary['by_task'][task].items() if not isinstance(v,(list,dict))})
                                    for name,summary in [('seen',seen),('unseen',unseen)] for task in TASKS])
        plot_curve(curve, result, out)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    finally:
        env.close()


def smoke(args):
    cfg=PRESETS['smoke'].copy();cfg.update(dynamics='residual',replay_limit=4096,max_steps=240)
    env,model,planner=create(args,cfg);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    try:
        for j in range(3):
            for task in TASKS:
                row,manifest=episode(env,model,planner,TRAIN_SCENES[0],42,task=task,shift=j)
                rows.append(row);dump(out/f'{task}_{j}.json',dict(metrics=row,episode=manifest))
                if j==0 and not row['success']:
                    raise RuntimeError(f'原始环境的参考控制器未完成 {task}；请提供 environment_check/{task}_0.json，先排查可达性再训练。')
                if j and not manifest['added_boxes']:raise AssertionError('B/C must add physical boxes')
                if task=='open_drawer':
                    q=np.asarray(env.drawer.joint_positions).copy();p=env.handle_position().copy()
                    q[env.drawer_ix]+=.01;env.drawer.joint_positions=q
                    assert np.linalg.norm(env.handle_position()-p)>.001,'Drawer joint did not move handle'
                print('CHECK',task,DOMAINS[j],row['success'],row['steps'],flush=True)
        dump(out/'environment_check.json',dict(rows=rows,native_scene_checks='passed',note='Reference-controller smoke checks, not trained scores; failed tasks remain visible.'))
    finally:env.close()

def record(args):
    saved = load_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)
    args.seed = saved['seed']; args.render = True
    env, model, planner = create(args, saved['config'])
    model.restore(saved['model'])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    try:
        row, manifest = episode(env, model, planner, TRAIN_SCENES[0], args.episode_seed, task=args.task, shift=args.domain, video=out/'baseline_demo.mp4')
        dump(out/'video_episode.json', dict(metrics=row, episode=manifest))
    finally:
        env.close()


def shadow(args):
    """Read-only raw/calibrated monitoring on a fixed baseline trajectory.

    The observer uses a separate frozen copy of the BASELINE dynamics plus the
    metacognitive checkpoint's monitor. Prefix search cannot change policy RNG
    or actions. This deliberately tests transfer to baseline process evidence;
    it does not estimate a counterfactual advantage of a metacontrol action.
    All extra observer computation is reported separately from policy costs.
    """
    import hashlib
    from .meta_planner import MetaMPC
    if args.n < 1:
        raise ValueError('shadow --n must be positive (episodes per task/domain)')
    base = load_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)
    meta = load_checkpoint(args.monitor_checkpoint, map_location='cpu', weights_only=False)
    if (base['config'].get('agent') != 'baseline' or
            meta['config'].get('agent') != 'metacognitive'):
        raise ValueError('shadow needs a baseline policy checkpoint and a metacognitive monitor checkpoint')
    for saved in (base, meta):
        if saved['config'].get('algorithm_version') != 'meta_comparison_v2':
            raise ValueError('shadow requires meta_comparison_v2 checkpoints')
    keys = ('dynamics', 'max_steps', 'horizon', 'candidates', 'iterations', 'elite')
    if base['seed'] != meta['seed'] or any(base['config'][k] != meta['config'][k] for k in keys):
        raise ValueError('shadow checkpoints must have matching seeds and world/planner settings')
    identity = dict(schema=SCHEMA, algorithm_version='meta_comparison_v2',
        generation_version=GENERATION_VERSION, seed=base['seed'], n=args.n,
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        monitor_checkpoint_sha256=hashlib.sha256(Path(args.monitor_checkpoint).read_bytes()).hexdigest(),
        observer_world='frozen_baseline_copy', seed_offset=750000)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    target = out/'shadow_summary.json'
    if target.exists():
        if json.loads(target.read_text()).get('identity') != identity:
            raise RuntimeError('Existing shadow results use different checkpoints/protocol; choose another output directory')
        print('Reuse completed frozen shadow diagnostics:', target, flush=True)
        return
    args.seed = base['seed']; args.render = False
    env, policy_model, policy = create(args, base['config'])
    rows = []; started = time.perf_counter()
    try:
        policy_model.restore(base['model'])
        cfg = base['config']
        observer_model = Ensemble(args.seed, mode=cfg['dynamics'], replay_limit=cfg['replay_limit'])
        observer_model.restore(base['model'])
        observer_model.monitor = MetaMonitor(args.seed)
        observer_model.monitor.restore(meta['model']['metacognition'])
        observer = MetaMPC(observer_model, **{k: cfg[k] for k in
            ('horizon', 'candidates', 'iterations', 'elite')})
        for domain, name in enumerate(DOMAINS):
            for task in TASKS:
                for i in range(args.n):
                    scene = TRAIN_SCENES[i % len(TRAIN_SCENES)]; seed = 750000+i
                    obs = env.reset(scene, seed, shift=domain, task=task)
                    policy.reset(seed+17); observer.reset(seed+17)
                    observer.start_episode(False, env.max_steps)
                    policy_model.count = Compute(); observer_model.count = Compute()
                    for _ in range(env.max_steps):
                        action = policy.act(obs)
                        observer.shadow_step(obs, action)
                        nxt, done, info = env.step(action)
                        observer.observe_transition(obs, action, nxt, info)
                        obs = nxt
                        if done:
                            break
                    diagnostics = observer.finish_episode(info['success'], info['steps'])
                    row = dict(task=task, domain=name, scene=scene, seed=seed,
                        success=int(info['success']), steps=int(info['steps']),
                        policy_nn_flops=int(policy_model.count.nn_flops_est),
                        observer_nn_flops=int(observer.total_flops()),
                        policy_nn_forward_calls=int(policy_model.count.nn_forward_calls),
                        observer_nn_forward_calls=int(observer_model.count.nn_forward_calls+
                            observer_model.count.meta_nn_forward_calls), **diagnostics)
                    rows.append(row)
                    dump(out/f'{task}_domain{domain}_episode{i:03d}.json',
                        dict(metrics=row, episode=env.manifest, monitoring_trace=observer.trace))
                print(f'SHADOW {name}/{task}: {args.n} frozen episodes', flush=True)
        groups = []; flat = []
        for domain in DOMAINS:
            for task in TASKS:
                selected = [r for r in rows if r['task'] == task and r['domain'] == domain]
                metrics = diagnostic_metrics([x for r in selected for x in r['local_monitoring_records']])
                sr = float(np.mean([r['success'] for r in selected]))
                groups.append(dict(task=task, domain=domain, success_rate=sr, diagnostics=metrics))
                record = dict(task=task, domain=domain, episodes=len(selected), success_rate=sr)
                # Flatten scalar diagnostic values for a first-time CSV reader.
                def flatten(value, prefix=''):
                    for key, item in value.items():
                        label = prefix+key
                        if isinstance(item, dict):
                            flatten(item, label+'_')
                        elif isinstance(item, (int, float)) or item is None:
                            record[label] = item
                flatten(metrics); flat.append(record)
        costs = dict(episodes=len(rows), environment_steps=sum(r['steps'] for r in rows),
            policy_nn_flops=sum(r['policy_nn_flops'] for r in rows),
            observer_nn_flops=sum(r['observer_nn_flops'] for r in rows),
            total_nn_flops=sum(r['policy_nn_flops']+r['observer_nn_flops'] for r in rows),
            wall_seconds=time.perf_counter()-started)
        save_csv(out/'shadow_episodes.csv', rows); save_csv(out/'shadow_metrics.csv', flat)
        dump(target, dict(identity=identity, costs=costs, by_task_domain=groups,
            monitor_ready=bool(observer_model.monitor.ready),
            note='No learning or fitting. Raw/calibrated forecasts share identical physical trajectories and labels. Baseline-world evidence may differ from monitor training evidence. Step records within episodes are correlated. Costs are additional diagnostic costs, not training samples or measured electrical energy.'))
    except Exception as error:
        dump(out/'shadow_failure.json', dict(error=str(error),
            context=getattr(env, 'generation_context', {}), completed_episodes=len(rows)))
        raise
    finally:
        env.close()


def continual(args):
    if args.phase_episodes<2 or args.phase_episodes%2 or args.eval_n<1 or args.validation_n<1 or args.check_every<2 or args.check_every%2:
        raise ValueError('phase-episodes/check-every: positive even counts; evaluation n >=1')
    saved=load_checkpoint(args.checkpoint,map_location='cpu',weights_only=False)
    if saved['config'].get('algorithm_version')!='meta_comparison_v2':
        raise ValueError('Use a meta_comparison_v2 checkpoint; earlier metacognitive supervision is incompatible')
    args.seed=saved['seed'];args.render=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    import hashlib
    digest=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    protocol=dict(phase_episodes=args.phase_episodes,eval_n=args.eval_n,check_every=args.check_every,validation_n=args.validation_n,
        generation_version=GENERATION_VERSION)
    if (out/'continual_summary.json').exists():
        previous=json.loads((out/'continual_summary.json').read_text())
        if previous.get('checkpoint_sha256')!=digest or previous.get('protocol')!=protocol:
            raise RuntimeError('Existing continual results use different checkpoint/budget; choose another output directory')
        dump(Path(args.checkpoint).parent/'mandatory_report.json',dict(baseline=json.loads((Path(args.checkpoint).parent/'summary.json').read_text()),continual=previous,status='complete'))
        print('复用已完成的必测持续学习结果：',out/'continual_summary.json',flush=True);return
    marker=out/'continual_protocol.json'
    identity=dict(checkpoint_sha256=digest,protocol=protocol)
    if marker.exists() and json.loads(marker.read_text())!=identity:
        raise RuntimeError('Continual output has a different generation version/checkpoint/budget; use a new directory')
    if not marker.exists() and (list(out.glob('frozen_*.csv')) or list(out.glob('phase*.pt'))):
        raise RuntimeError('Unversioned partial continual results; preserve them and use a new output directory')
    dump(marker,identity)
    env,model,planner=create(args,saved['config']);model.restore(saved['model'])
    matrix={t:[] for t in ('macro',)+TASKS};costs=[];adaptation=[];evaluations=[]
    try:
        frozen=[evaluate(env,model,planner,TRAIN_SCENES,args.eval_n,750000,f'frozen_{j}',out,0,shift=j) for j in range(3)]
        for phase,domain in enumerate(DOMAINS):
            start_flops=model.train_flops_est;rows=[];curve=[];phase_steps=0
            monitor=getattr(model,'monitor',None)
            start_meta_flops=monitor.train_flops_est if monitor else 0
            for ep in range(args.phase_episodes+1):
                if ep:
                    task,scene=schedule(ep-1,TRAIN_SCENES)
                    row,manifest=episode(env,model,planner,scene,550000+phase*10000+ep,task=task,learn=True,shift=phase)
                    row['phase_episode']=ep;rows.append(row);phase_steps+=row['steps'];model.fit(saved['config']['updates'])
                    dump(out/f'phase{phase}_train_episode{ep:04d}.json',manifest)
                    save_csv(out/f'phase{phase}_training.csv',rows)
                if ep==0 or ep%args.check_every==0 or ep==args.phase_episodes:
                    score=evaluate(env,model,planner,TRAIN_SCENES,args.validation_n,650000,f'phase{phase}_adapt',out,ep,shift=phase)
                    curve.append(dict(train_episodes=ep,train_steps=phase_steps,**score))
                    dump(out/f'phase{phase}_adaptation_curve.json',curve)
            scores=[evaluate(env,model,planner,TRAIN_SCENES,args.eval_n,750000,f'after{phase}_domain{j}',out,args.phase_episodes,shift=j) for j in range(3)]
            evaluations.append(scores)
            for task in matrix:
                matrix[task].append([r['success_rate'] if task=='macro' else r['by_task'][task]['success_rate'] for r in scores])
            costs.append(dict(domain=domain,**cost_summary(rows,model.train_flops_est-start_flops,
                monitor.train_flops_est-start_meta_flops if monitor else 0),replay_size=len(model.inputs),
                meta_records=len(monitor.samples) if monitor else 0,
                meta_calibration_records=len(getattr(monitor,'calibration_samples',[])) if monitor else 0))
            adaptation.append(dict(domain=domain,**adaptation_summary(curve,args.phase_episodes)))
            checkpoint(out/f'phase{phase}.pt',model,config=saved['config'],seed=args.seed)
            dump(out/'partial_results.json',dict(matrix=matrix,costs=costs,adaptation=adaptation))
        result=dict(schema=SCHEMA,agent=saved['config'].get('agent','baseline'),domains=DOMAINS,tasks=TASKS,matrix=matrix,adaptation=adaptation,costs=costs,
            retention={t:continual_metrics(m) for t,m in matrix.items()},
            frozen_reference=frozen,after_phase_evaluation=evaluations,replay_capacity=model.replay_limit,
            checkpoint_sha256=digest,protocol=protocol,
            metacognition=monitor.summary() if monitor else None,
            evaluation_note='Validation/test freeze dynamics AND metacognition weights/buffers. Episode-local error history may update. Domain IDs excluded from policy inputs. Learning/replay schedules identical across agents.')
        dump(out/'continual_summary.json',result)
        save_csv(out/'continual_matrix.csv',[dict(task=t,after=i,A=r[0],B=r[1],C=r[2]) for t,m in matrix.items() for i,r in enumerate(m)])
        dump(Path(args.checkpoint).parent/'mandatory_report.json',dict(baseline=json.loads((Path(args.checkpoint).parent/'summary.json').read_text()),continual=result,status='complete'))
        print('必测 A/B/C 完成：',out/'continual_summary.json',flush=True)
    except Exception as error:
        dump(out/'generation_failure.json',dict(error=str(error),
            context=getattr(env,'generation_context',{}),
            geometry=getattr(env,'generation_diagnostics',{}),
            note='Inspect traceback: this file records context, not a scored policy failure.'))
        raise
    finally:env.close()


def check_domains(args):
    """Reset every distinct continual-learning episode without model actions."""
    if args.phase_episodes<2 or args.phase_episodes%2 or args.eval_n<1 or args.validation_n<1:
        raise ValueError('Need a positive even phase budget and positive evaluation counts')
    saved=load_checkpoint(args.checkpoint,map_location='cpu',weights_only=False)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    identity=dict(generation_version=GENERATION_VERSION,data=str(Path(args.data).resolve()),
        scenes=TRAIN_SCENES,phase_episodes=args.phase_episodes,eval_n=args.eval_n,
        validation_n=args.validation_n,max_steps=saved['config']['max_steps'])
    target=out/'generation_check.json'
    if target.exists():
        previous=json.loads(target.read_text())
        if previous.get('identity')==identity and previous.get('status')=='complete':
            print('复用几何预检（不是 agent 成绩）:',target,flush=True);return
    cases=[]
    for domain in range(3):
        for offset,n in ((750000,args.eval_n),(650000,args.validation_n)):
            for task in TASKS:
                cases.extend((TRAIN_SCENES[i%len(TRAIN_SCENES)],offset+i,task,domain) for i in range(n))
        for ep in range(1,args.phase_episodes+1):
            task,scene=schedule(ep-1,TRAIN_SCENES)
            cases.append((scene,550000+domain*10000+ep,task,domain))
    env=CupEnv(args.data,render=False,max_steps=saved['config']['max_steps'])
    rows=[];started=time.perf_counter()
    try:
        for i,(scene,seed,task,domain) in enumerate(cases):
            print(f'GEOMETRY {i+1}/{len(cases)} {task} {DOMAINS[domain]} {scene} seed={seed}',flush=True)
            env.reset(scene,seed,shift=domain,task=task)
            rows.append(env.manifest.copy())
            dump(target,dict(status='running',identity=identity,episodes=rows))
        dump(target,dict(status='complete',identity=identity,episodes=rows,
            seconds=time.perf_counter()-started,
            note='Geometry resets only: no policy actions, learning, replay writes or N85 samples. Not an arm reachability proof.'))
        print('持续学习几何预检通过:',len(rows),'个独立 episode；没有执行 agent。',flush=True)
    except Exception as error:
        dump(out/'generation_failure.json',dict(error=str(error),
            context=getattr(env,'generation_context',{}),geometry=getattr(env,'generation_diagnostics',{}),
            completed_resets=len(rows),identity=identity))
        raise
    finally:env.close()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    for name in ['train','smoke','record','continual','check_domains','shadow']:
        q = sub.add_parser(name)
        q.add_argument('--data', required=True)
        q.add_argument('--out', required=True)
        q.add_argument('--seed', type=int, default=0)
        if name == 'train':
            q.add_argument('--agent',choices=['baseline','metacognitive'],default='baseline')
            q.add_argument('--meta-compute-weight',type=float,default=.10)
            q.add_argument('--meta-step-weight',type=float,default=.05)
            q.add_argument('--preset', choices=PRESETS, default='pilot')
            q.add_argument('--dynamics', choices=['learned','residual'], default='learned')
            q.add_argument('--episodes', type=int)
            q.add_argument('--resume', action='store_true')
            q.add_argument('--replay-limit', type=int, default=4096)
        if name == 'smoke':
            q.add_argument('--render', action='store_true')
        if name in ['record','continual','check_domains','shadow']:
            q.add_argument('--checkpoint', required=True)
        if name == 'shadow':
            q.add_argument('--monitor-checkpoint', required=True)
            q.add_argument('--n', type=int, default=5)
        if name == 'record':
            q.add_argument('--episode-seed', type=int, default=250000)
            q.add_argument('--task', choices=TASKS, default=TASKS[0])
            q.add_argument('--domain', type=int, choices=range(3), default=0)
        if name in ['continual','check_domains']:
            q.add_argument('--phase-episodes', type=int, default=20)
            q.add_argument('--eval-n', type=int, default=20)
            q.add_argument('--validation-n', type=int, default=10)
            q.add_argument('--check-every', type=int, default=4)
    a = p.parse_args()
    globals()[a.command](a)


if __name__ == '__main__':
    main()
