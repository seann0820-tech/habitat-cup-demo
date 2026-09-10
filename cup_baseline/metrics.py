"""Pure metric functions; failure/right-censoring and denominators are explicit."""
import math
import numpy as np


def wilson(successes, n, z=1.96):
    if n == 0:
        return [None, None]
    p = successes/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def summarize(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    successes = sum(r['success'] for r in rows)
    nav = sum(r['nav_success'] for r in rows)
    out = dict(n=n, success_rate=successes/n,
               success_ci95=wilson(successes, n),
               nav_success_rate=nav/n,
               pick_success_rate=sum(r['pick_success'] for r in rows)/n,
               drawer_success_rate=sum(r.get('drawer_success',0) for r in rows)/n,
               pick_given_nav=successes/nav if nav else None)
    for key in ('steps', 'physics_substeps', 'nn_forward_calls', 'nn_sample_forwards',
                'nn_flops_est', 'planner_seconds', 'wall_seconds', 'nav_spl', 'raycasts', 'ik_fk_evals'):
        out['mean_'+key] = float(np.mean([r[key] for r in rows]))
    out['nn_flops_per_success_est'] = sum(r['nn_flops_est'] for r in rows)/successes if successes else None
    out['mean_steps_success_only'] = float(np.mean([r['steps'] for r in rows if r['success']])) if successes else None
    for key in ('manipulation_collision_steps','collision_checks','collision_steps','drawer_open_m'):
        out['mean_'+key]=float(np.mean([r.get(key,0) for r in rows]))
    for key in ('meta_nn_forward_calls','meta_nn_sample_forwards','meta_nn_flops_est',
                'meta_decisions','meta_execute','meta_expand','meta_restart','meta_ready_decisions',
                'meta_baseline','meta_fallback_steps','meta_search_batches','meta_recovery_resets'):
        out['mean_'+key]=float(np.mean([r.get(key,0) for r in rows]))
    total=sum(r.get('total_nn_flops_est',r['nn_flops_est']) for r in rows)
    out['mean_total_nn_flops_est']=total/n
    out['mean_total_nn_forward_calls']=float(np.mean([r.get('total_nn_forward_calls',r['nn_forward_calls']) for r in rows]))
    out['total_nn_flops_per_success_est']=total/successes if successes else None
    return out


def n85(curve, budget):
    """First fixed-validation checkpoint >=85%, confirmed at the next checkpoint."""
    for a, b in zip(curve, curve[1:]):
        if a['success_rate'] >= .85 and b['success_rate'] >= .85:
            return dict(episodes=a['train_episodes'], transitions=a['train_steps'],
                        confirmed_at_episode=b['train_episodes'], reached=True, budget_episodes=budget)
    return dict(episodes=None, transitions=None, confirmed_at_episode=None,
                reached=False, budget_episodes=budget, display='>'+str(budget))


def continual_metrics(matrix):
    """Rows: after learning A/B/C; columns: held-out probes in domains A/B/C."""
    r = np.asarray(matrix, dtype=float)
    t = len(r)
    if r.shape != (t, t) or t < 2:
        raise ValueError('Need square matrix of >=2 phases')
    bwt = np.mean([r[-1, j]-r[j, j] for j in range(t-1)])
    forgetting = np.mean([max(0., r[j:, j].max()-r[-1, j]) for j in range(t-1)])
    return dict(final_average_success=float(r[-1].mean()), backward_transfer=float(bwt),
                average_forgetting=float(forgetting))
