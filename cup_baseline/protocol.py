"""Pure two-task split, cost and threshold logic; no simulator dependencies."""
import numpy as np
from .metrics import summarize, n85
from .metacognition import monitoring_metrics, diagnostic_metrics
TASKS=('pick_cup','open_drawer')
DOMAINS=('A_clear','B_passage_blocked','C_workspace_clutter')

def schedule(index,scenes):
    return TASKS[index%2], scenes[(index//2)%len(scenes)]

def task_summary(rows):
    by_task={t:summarize([r for r in rows if r['task']==t]) for t in TASKS}
    for t in TASKS:
        selected=[r for r in rows if r['task']==t]
        by_task[t]['monitoring']=monitoring_metrics(selected)
        by_task[t]['local_monitoring']=diagnostic_metrics(
            [record for r in selected for record in r.get('local_monitoring_records',[])])
    result=summarize(rows);result['by_task']=by_task
    result['success_rate']=float(np.mean([by_task[t]['success_rate'] for t in TASKS]))
    result['minimum_task_success_rate']=min(by_task[t]['success_rate'] for t in TASKS)
    result['pick_success_rate']=by_task['pick_cup']['pick_success_rate']
    result['drawer_success_rate']=by_task['open_drawer']['drawer_success_rate']
    return result

def threshold_report(curve,budget):
    reports={}
    for task in ('macro','both_tasks')+TASKS:
        values=[]
        for row in curve:
            sr=(row['success_rate'] if task=='macro' else row['minimum_task_success_rate']
                if task=='both_tasks' else row['by_task'][task]['success_rate'])
            values.append(dict(train_episodes=row['train_episodes'],train_steps=row['train_steps'],success_rate=sr))
        result=n85(values,budget);result['episode_axis']='total training episodes across both tasks'
        if result['reached']:
            result['already_at_threshold']=result['episodes']==0
            if task in TASKS:result['task_episodes']=(result['episodes']+1)//2 if task==TASKS[0] else result['episodes']//2
        reports[task]=result
    return reports

def adaptation_summary(curve,budget):
    result={'n85':threshold_report(curve,budget),'learning_curve_auc':{}}
    x=np.asarray([r['train_episodes'] for r in curve],dtype=float)
    for task in ('macro',)+TASKS:
        y=np.asarray([r['success_rate'] if task=='macro' else r['by_task'][task]['success_rate'] for r in curve])
        result['learning_curve_auc'][task]=float(np.sum(np.diff(x)*(y[:-1]+y[1:])/2)/budget)
    return result

def cost_summary(rows,update_flops,meta_update_flops=0):
    return dict(training_episodes=len(rows),environment_steps=sum(r['steps'] for r in rows),
        planning_nn_flops=sum(r['nn_flops_est'] for r in rows),
        planning_nn_forward_calls=sum(r['nn_forward_calls'] for r in rows),
        update_nn_flops=int(update_flops),
        meta_inference_nn_flops=sum(r.get('meta_nn_flops_est',0) for r in rows),
        meta_update_nn_flops=int(meta_update_flops),
        total_nn_flops=sum(r['nn_flops_est']+r.get('meta_nn_flops_est',0) for r in rows)+int(update_flops)+int(meta_update_flops),
        collection_by_task={t:dict(episodes=sum(r['task']==t for r in rows),
            environment_steps=sum(r['steps'] for r in rows if r['task']==t),
            planning_nn_flops=sum(r['nn_flops_est'] for r in rows if r['task']==t)) for t in TASKS},
        update_cost_scope='shared model with mixed-task replay; do not attribute all update cost to the most recent task',
        note='NN FLOPs only; excludes simulation, IK, geometry and CEM arithmetic; evaluation costs separate')
