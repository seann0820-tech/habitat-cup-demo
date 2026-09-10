"""Conservative geometry checks for the added boxes (metres, world frame)."""
import numpy as np


def segment_hits_box(start, end, lo, hi, margin=0.):
    start, end = np.asarray(start), np.asarray(end)
    lo, hi = np.asarray(lo)-margin, np.asarray(hi)+margin
    d = end-start
    enter, leave = 0., 1.
    for k in range(len(start)):
        if abs(d[k]) < 1e-9:
            if start[k] < lo[k] or start[k] > hi[k]: return False
        else:
            a, b = sorted(((lo[k]-start[k])/d[k], (hi[k]-start[k])/d[k]))
            enter, leave = max(enter,a), min(leave,b)
            if enter > leave: return False
    return True


def clearance(points, boxes, margin=0.):
    points = np.asarray(points)
    if not len(boxes): return np.full(points.shape[:-1], 10.)
    boxes = np.asarray(boxes)
    lo, hi = boxes[:,0]-margin, boxes[:,1]+margin
    distances = np.maximum(np.maximum(lo-points[...,None,:], points[...,None,:]-hi),0)
    return np.linalg.norm(distances,axis=-1).min(-1)


def detour(start, goal, boxes, margin, vertical=False):
    """One-box waypoint from observed bounds only; no Habitat path query."""
    start, goal = np.asarray(start), np.asarray(goal)
    for lo,hi in np.asarray(boxes).reshape(-1,2,len(start)):
        if not segment_hits_box(start,goal,lo,hi,margin): continue
        lo, hi = lo-margin-.04, hi+margin+.04
        if vertical:
            candidates = [np.array([x,hi[1],z]) for x in (lo[0],hi[0]) for z in (lo[2],hi[2])]
        else:
            candidates = [np.array([x,z]) for x in (lo[0],hi[0]) for z in (lo[1],hi[1])]
        candidates = [p for p in candidates if not segment_hits_box(start,p,lo+.015,hi-.015)]
        if candidates:
            return min(candidates,key=lambda p: np.linalg.norm(p-start)+np.linalg.norm(p-goal))
    return goal
