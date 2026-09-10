"""Two tasks in ReplicaCAD with a shared Fetch model-based controller.

Cup grasp and drawer handle attachment are proximity-gated kinematic skills.
Drawer motion uses the native prismatic joint, driven by actual EE displacement;
this is not a force-controlled grip/contact benchmark. Added clutter is checked
against swept arm-link capsules, gripper, carried cup and moving drawer bounds.
"""
import math
import time
import numpy as np
import magnum as mn
import habitat_sim
from habitat_sim.physics import MotionType, JointType
from .env import CupEnv, world_bb, wrap
from .geometry import segment_hits_box, clearance
from .generation import GENERATION_VERSION, EpisodeGenerationError, passage_candidates

TASKS = ('pick_cup','open_drawer')
DOMAINS = ('A_clear','B_passage_blocked','C_workspace_clutter')
SCHEMA = 'home_two_tasks_v2'


def node_bounds(node):
    bb, tr = node.cumulative_bb, node.absolute_transformation()
    pts = np.asarray([tr.transform_point(mn.Vector3(x,y,z))
        for x in (bb.min.x,bb.max.x) for y in (bb.min.y,bb.max.y) for z in (bb.min.z,bb.max.z)])
    return pts.min(0), pts.max(0)


class HomeEnv(CupEnv):
    drawer_open_m = .18

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.added = []; self.boxes = []; self.preparing = True

    def _rebuild_navmesh(self):
        ns = habitat_sim.NavMeshSettings(); ns.set_defaults()
        ns.agent_radius=.30; ns.agent_height=1.4; ns.include_static_objects=True
        furniture=list(self.sim.get_articulated_object_manager().get_objects_by_handle_substring('kitchen_counter').values())
        modes=[o.motion_type for o in furniture]
        try:
            for obj in furniture:obj.motion_type=MotionType.STATIC
            if not self.sim.recompute_navmesh(self.sim.pathfinder,ns):
                raise RuntimeError('Navigation mesh rebuild failed')
        finally:
            for obj,mode in zip(furniture,modes):obj.motion_type=mode

    def _clear_added(self):
        if self.sim is not None:
            for obj in self.added:
                if obj.is_alive: self.sim.get_rigid_object_manager().remove_object_by_id(obj.object_id)
        self.added=[]; self.boxes=[]

    def _box(self,center,half,label):
        tm=self.sim.get_object_template_manager()
        handles=tm.get_template_handles('cubeSolid')
        if not handles: raise RuntimeError('Habitat cubeSolid primitive missing')
        template=tm.get_template_by_handle(handles[0])
        template.scale=mn.Vector3(np.asarray(half,dtype=float))
        handle=f'home_box_{label}_{len(self.added)}'
        template.margin=0.0
        tm.register_template(template,handle)
        obj=self.sim.get_rigid_object_manager().add_object_by_template_handle(handle)
        obj.translation=mn.Vector3(center); obj.motion_type=MotionType.STATIC
        self.added.append(obj); self.boxes.append(np.stack(world_bb(obj)))
        return obj

    def _select_drawer(self,rng):
        objects=self.sim.get_articulated_object_manager().get_objects_by_handle_substring('kitchen_counter')
        options=[]
        for obj in objects.values():
            obj.motion_type=MotionType.KINEMATIC
            for lid in obj.get_link_ids():
                if obj.get_link_joint_type(lid)!=JointType.Prismatic: continue
                if 'top' not in obj.get_link_name(lid): continue
                ix=obj.get_link_joint_pos_offset(lid)
                q=np.asarray(obj.joint_positions).copy(); q[ix]=0.; obj.joint_positions=q
                node=obj.get_link_scene_node(lid)
                lo,hi=node_bounds(node); p=(lo+hi)/2
                origin=np.asarray(node.absolute_transformation().translation).copy()
                q[ix]=.02;obj.joint_positions=q
                d=np.asarray(node.absolute_transformation().translation)-origin
                q[ix]=0.;obj.joint_positions=q
                scale=np.linalg.norm(d)/.02
                if scale<.01: continue
                axis=d/np.linalg.norm(d)
                k=int(np.argmax(np.abs(axis)))
                p[k]=(hi[k] if axis[k]>0 else lo[k])+axis[k]*.015
                if not .50<p[1]<1.15: continue
                for dist in (.75,.85,.65):
                    candidate=p+axis*dist; candidate[1]=0.
                    nav=np.asarray(self.sim.pathfinder.snap_point(candidate))
                    if np.isfinite(nav).all() and np.linalg.norm(nav[[0,2]]-candidate[[0,2]])<.14:
                        options.append((obj,lid,ix,p,axis,scale,nav));break
        if not options: raise RuntimeError('No reachable native top drawer in this scene')
        self.drawer,self.drawer_link,self.drawer_ix,p,self.pull_axis,self.joint_scale,self.nav_goal=options[int(rng.integers(len(options)))]
        self.drawer_local=self.drawer.get_link_scene_node(self.drawer_link).absolute_transformation().inverted().transform_point(mn.Vector3(p))
        self.handle_rest=p.copy(); self.drawer_closed=np.asarray(self.drawer.joint_positions).copy()
        self.table_handle=self.drawer.handle

    def handle_position(self):
        return np.asarray(self.drawer.get_link_scene_node(self.drawer_link).absolute_transformation().transform_point(self.drawer_local))

    def opening(self):
        return float((self.drawer.joint_positions[self.drawer_ix]-self.drawer_closed[self.drawer_ix])*self.joint_scale)

    def _sample_start(self,rng,start_range):
        for _ in range(1000):
            p=np.asarray(self.sim.pathfinder.get_random_navigable_point())
            d=self.geodesic(p,self.nav_goal)
            if start_range[0]<=d<=start_range[1]:
                self.start=p;self.shortest_distance=d;break
        else: raise RuntimeError('No connected episode start')
        self.robot.base_pos=mn.Vector3(self.start)
        self.robot.base_rot=float(rng.uniform(-np.pi,np.pi));self.robot.update()

    def _block_passage(self,rng,start_range=(1.5,4.)):
        # Only geometry is consulted. Never filter candidates by policy success.
        started=time.perf_counter()
        original_start=self.start.copy();original_yaw=float(self.robot.base_rot)
        report=dict(version=GENERATION_VERSION,original_start=original_start.tolist(),
            start_resampled=False,start_attempts=0,box_attempts=0,
            rejected=dict(endpoint_clearance=0,disconnected=0,no_detour=0,off_route=0),
            attempts=[])
        self.generation_diagnostics=report
        half=np.array([.16,.27,.16])
        for start_attempt in range(8):
            report['start_attempts']=start_attempt+1
            path=habitat_sim.ShortestPath();path.requested_start=self.start;path.requested_end=self.nav_goal
            if not self.sim.pathfinder.find_path(path):
                raise EpisodeGenerationError('Original path disconnected before B placement')
            points=np.asarray(path.points);original=float(path.geodesic_distance)
            proposals=passage_candidates(points,rng,legacy=True) if start_attempt==0 else []
            proposals+=passage_candidates(points,rng)
            for p,fraction,offset in proposals:
                if min(np.linalg.norm(p[[0,2]]-self.start[[0,2]]),
                       np.linalg.norm(p[[0,2]]-self.nav_goal[[0,2]]))<.65:
                    report['rejected']['endpoint_clearance']+=1;continue
                p=p.copy();p[1]+=half[1]+.01
                if not any(segment_hits_box(a[[0,2]],b[[0,2]],
                    (p-half)[[0,2]],(p+half)[[0,2]],.30) for a,b in zip(points[:-1],points[1:])):
                    report['rejected']['off_route']+=1;continue
                self._box(p,half,'parcel');self._rebuild_navmesh()
                report['box_attempts']+=1
                # find_path may snap endpoints; reject an obstacle that covers one.
                endpoints_clear=True
                for endpoint in (self.start,self.nav_goal):
                    snapped=np.asarray(self.sim.pathfinder.snap_point(endpoint))
                    if not np.isfinite(snapped).all() or np.linalg.norm(snapped-endpoint)>.05:
                        endpoints_clear=False;break
                distance=self.geodesic(self.start,self.nav_goal) if endpoints_clear else float('inf')
                accepted=np.isfinite(distance) and distance>original+.03
                reason='accepted' if accepted else ('endpoint_clearance' if not endpoints_clear else
                    'disconnected' if not np.isfinite(distance) else 'no_detour')
                report['attempts'].append(dict(start_attempt=start_attempt,fraction=float(fraction),
                    lateral_offset_m=float(offset),original_distance_m=original,
                    blocked_distance_m=float(distance) if np.isfinite(distance) else None,result=reason))
                if accepted:
                    self.shortest_distance=distance
                    report.update(start_resampled=bool(start_attempt),selected_start=self.start.tolist(),
                        original_distance_m=original,blocked_distance_m=float(distance),
                        detour_extra_m=float(distance-original),seconds=time.perf_counter()-started)
                    return
                report['rejected'][reason]+=1
                self._clear_added();self._rebuild_navmesh()
            if start_attempt==7:break
            # Keep scene, target, box size, distance range and yaw. Resample only
            # the start when that route admits none of the bounded proposals.
            self.sim.pathfinder.seed(int(rng.integers(1,2**31-1)))
            for _ in range(256):
                candidate=np.asarray(self.sim.pathfinder.get_random_navigable_point())
                d=self.geodesic(candidate,self.nav_goal)
                if start_range[0]<=d<=start_range[1] and np.linalg.norm(candidate-original_start)>.40:
                    self.start=candidate.copy();self.shortest_distance=d
                    self.robot.base_pos=mn.Vector3(candidate);self.robot.update();break
            else:break
        self._clear_added();self._rebuild_navmesh()
        self.start=original_start;self.shortest_distance=self.geodesic(self.start,self.nav_goal)
        self.robot.base_pos=mn.Vector3(self.start);self.robot.base_rot=original_yaw;self.robot.update()
        report['seconds']=time.perf_counter()-started
        raise EpisodeGenerationError('B geometry search exhausted: '
            f"{report['start_attempts']} starts, {report['box_attempts']} boxes; "
            f"rejections={report['rejected']}. No episode was scored; see generation_failure.json")

    def _clutter_workspace(self,rng):
        target=self.target_position()
        outward=np.r_[self.nav_goal[0]-target[0],0.,self.nav_goal[2]-target[2]]
        outward/=max(np.linalg.norm(outward),1e-8)
        sideways=np.array([-outward[2],0,outward[0]])
        if self.task=='pick_cup':
            # Small fixed parcel on the same table, in front/side of the cup.
            table=self.sim.get_rigid_object_manager().get_object_by_handle(self.table_handle)
            lo,hi=world_bb(table)
            center=None
            for offset in (outward*.18,sideways*.18,-sideways*.18,-outward*.18):
                candidate=target+offset
                candidate[[0,2]]=np.clip(candidate[[0,2]],lo[[0,2]]+.065,hi[[0,2]]-.065)
                hit=self._ray([candidate[0],hi[1]+.2,candidate[2]],[0,-1,0],.4)
                if np.linalg.norm(candidate[[0,2]]-target[[0,2]])>=.12 and hit is not None and hit.object_id==table.object_id:
                    candidate[1]=float(hit.point[1])+.075
                    self._box(candidate,[.045,.075,.055],'table_parcel')
                    if self._dock_arm_clear():center=candidate;break
                    self._clear_added()
            if center is None:raise RuntimeError('No supported clutter placement around cup')
        else:
            # Narrow floor box beside the handle approach, outside drawer sweep.
            half_y=max(.12,(target[1]+.10-self.start[1])/2)
            placed=False
            for side in rng.permutation([-.10,.10,-.18,.18]):
                center=target+outward*.32+sideways*side;center[1]=self.start[1]+half_y
                self._box(center,[.065,half_y,.065],'handle_side_parcel')
                if self._dock_arm_clear():placed=True;break
                self._clear_added()
            if not placed:raise RuntimeError('C drawer clutter intersects initial docked arm')
            q=np.asarray(self.drawer.joint_positions).copy()
            for opening in np.linspace(0,self.drawer_open_m+.02,6):
                trial=q.copy();trial[self.drawer_ix]=opening/self.joint_scale;self.drawer.joint_positions=trial
                lo,hi=node_bounds(self.drawer.get_link_scene_node(self.drawer_link))
                if any(np.all(hi>b[0]) and np.all(lo<b[1]) for b in self.boxes):
                    self.drawer.joint_positions=q
                    raise RuntimeError('C clutter blocks the required drawer opening; episode rejected')
            self.drawer.joint_positions=q
        self._rebuild_navmesh()
        distance=self.geodesic(self.start,self.nav_goal)
        if not np.isfinite(distance): raise RuntimeError('C clutter disconnected navigation')
        self.shortest_distance=distance
        # The initial target itself must remain accessible to the gripper.
        if clearance(target[None],self.boxes,.045)[0]<=0:
            raise RuntimeError('C placement overlaps grasp target')

    def _dock_arm_clear(self):
        """Generation check only, never provides actions/waypoints to the policy."""
        base=np.asarray(self.robot.base_pos).copy();yaw=self.robot.base_rot
        joints=np.asarray(self.robot.arm_joint_pos).copy();target=self.target_position()
        try:
            self.robot.base_pos=mn.Vector3(self.nav_goal)
            self.robot.base_rot=float(math.atan2(-(target[2]-self.nav_goal[2]),target[0]-self.nav_goal[0]))
            self.robot.arm_joint_pos=self.initial_q;self.robot.update()
            return not self._arm_hits()
        finally:
            self.robot.base_pos=mn.Vector3(base);self.robot.base_rot=yaw
            self.robot.arm_joint_pos=joints;self.robot.update()

    def reset(self,scene,seed,start_range=(1.5,4.),shift=0,task='pick_cup'):
        if task not in TASKS or shift not in range(3): raise ValueError('Unknown task/domain')
        self.generation_context=dict(scene=scene,seed=int(seed),task=task,domain=DOMAINS[shift])
        self.generation_diagnostics={}
        self.preparing=True;self._clear_added()
        if self.sim is not None:
            for obj in self.sim.get_articulated_object_manager().get_objects_by_handle_substring('kitchen_counter').values():
                obj.joint_positions=np.zeros_like(obj.joint_positions);obj.motion_type=MotionType.STATIC
            self._rebuild_navmesh()
        self.task=task;self.drawer=None;self.manipulation_collisions=0;self.collision_checks=0
        super().reset(scene,seed,start_range=start_range,shift=0)
        self.shift=shift; rng=np.random.default_rng(seed+7919)
        if task=='open_drawer':
            self.sim.get_rigid_object_manager().remove_object_by_id(self.cup.object_id);self.cup=None
            self._select_drawer(rng);self._sample_start(rng,start_range)
        self.drawer_success=False; self.preparing=False
        if shift==1:self._block_passage(rng,start_range)
        if shift==2:self._clutter_workspace(rng)
        self.manifest=dict(schema=SCHEMA,task=task,domain=DOMAINS[shift],scene=scene,seed=int(seed),
            target=self.target_position().tolist(),target_object=self.table_handle,
            start=self.start.tolist(),start_yaw=float(self.robot.base_rot),nav_goal=self.nav_goal.tolist(),
            shortest_distance=float(self.shortest_distance),added_boxes=[b.tolist() for b in self.boxes],
            drawer_link=self.drawer_link if task=='open_drawer' else None,
            generation_collision_checks=self.collision_checks,
            generation_version=GENERATION_VERSION,generation=self.generation_diagnostics)
        self.collision_checks=0
        return self.observe()

    def target_position(self):
        return np.asarray(self.cup.translation) if self.task=='pick_cup' else self.handle_position()

    def observe(self):
        if self.preparing:return super().observe()
        # Reuse the twelve rays; they and box bounds are privileged geometry observations.
        x=self.state(); ranges=[];points=[]
        for a in x[2]+np.arange(self.rays_n)*2*np.pi/self.rays_n:
            direction=np.array([np.cos(a),0.,-np.sin(a)])
            hit=self._ray([x[0],self.start[1]+.30,x[1]],direction,self.ray_range)
            self.raycast_count+=1;d=self.ray_range if hit is None else min(self.ray_range,float(hit.ray_distance))
            ranges.append(d)
            if d<self.ray_range-.01:points.append(x[:2]+d*direction[[0,2]])
        target=self.target_position()
        goal=target.copy()
        if self.held:
            goal=self.cup_rest+[0,.23,0] if self.task=='pick_cup' else self.handle_rest+self.pull_axis*(self.drawer_open_m+.035)
        boxes=np.asarray(self.boxes,dtype=np.float32).reshape(-1,2,3)
        return dict(x=x,joints=np.asarray(self.robot.arm_joint_pos,dtype=np.float32),rays=np.asarray(ranges,dtype=np.float32),
            obstacles=np.asarray(points,dtype=np.float32).reshape(-1,2),arm_boxes=boxes,
            nav_goal=self.nav_goal[[0,2]].astype(np.float32),cup=target.astype(np.float32),
            ee_goal=np.asarray(goal,dtype=np.float32),phase=self.phase,held=self.held,
            task_id=TASKS.index(self.task),task=self.task)

    def _arm_points(self):
        pts=[np.asarray(self.robot.sim_obj.get_link_scene_node(lid).absolute_transformation().translation).copy()
             for lid in self.robot.params.arm_joints]
        pts.append(np.asarray(self.robot.ee_transform().translation).copy())
        return np.asarray(pts)

    def _arm_hits(self):
        pts=self._arm_points()
        for lo,hi in self.boxes:
            for a,b in zip(pts[:-1],pts[1:]):
                self.collision_checks+=1
                if segment_hits_box(a,b,lo,hi,.04):return True
            if self.task=='pick_cup' and self.held:
                center=np.asarray((self.robot.ee_transform()@self.grasp_transform).translation)
                if clearance(center[None],[[lo,hi]],.065)[0]<=0:return True
        return False

    def _ik_delta(self,delta):
        old=np.asarray(self.robot.arm_joint_pos).copy();super()._ik_delta(delta)
        new=np.asarray(self.robot.arm_joint_pos).copy()
        for fraction in np.linspace(.2,1.,5):
            self.robot.arm_joint_pos=old+(new-old)*fraction
            if self._arm_hits():
                self.robot.arm_joint_pos=old;self.manipulation_collisions+=1;return
        self.robot.arm_joint_pos=new

    def step(self,action):
        a=np.clip(np.asarray(action,dtype=float),-1,1)
        if a.shape!=(5,) or not np.isfinite(a).all():raise ValueError('Expected finite action[5]')
        before=self.state();old_q=np.asarray(self.robot.arm_joint_pos).copy()
        if self.phase==0:
            self.robot.base_rot=float(wrap(before[2]+a[1]*self.yaw_step))
            b=np.asarray(self.robot.base_pos);theta=self.robot.base_rot
            desired=b+a[0]*self.base_step*np.array([np.cos(theta),0,-np.sin(theta)])
            actual=np.asarray(self.sim.pathfinder.try_step(b,desired))
            if not np.isfinite(actual).all():actual=b
            for lo,hi in self.boxes:
                if lo[1]<self.start[1]+1.4 and segment_hits_box(b[[0,2]],actual[[0,2]],lo[[0,2]],hi[[0,2]],.30):actual=b
            self.collisions+=int(np.linalg.norm(actual-desired)>.01)
            self.robot.base_pos=mn.Vector3(actual);self.path_length+=float(np.linalg.norm(actual-b))
        else:
            theta=before[2];local=a[2:]*self.arm_step
            self._ik_delta(np.array([np.cos(theta)*local[0]+np.sin(theta)*local[2],local[1],-np.sin(theta)*local[0]+np.cos(theta)*local[2]]))
        self.robot.update()
        if self.held and self.task=='open_drawer':
            q=np.asarray(self.drawer.joint_positions).copy();old_drawer=q.copy()
            movement=float(np.dot(self.state()[3:]-before[3:],self.pull_axis))
            q[self.drawer_ix]=np.clip(q[self.drawer_ix]+movement/self.joint_scale,0.,.35/self.joint_scale)
            self.drawer.joint_positions=q
            lo,hi=node_bounds(self.drawer.get_link_scene_node(self.drawer_link))
            blocked=any(np.all(hi>b[0]) and np.all(lo<b[1]) for b in self.boxes)
            if blocked:
                self.drawer.joint_positions=old_drawer;self.robot.arm_joint_pos=old_q;self.manipulation_collisions+=1
            if np.linalg.norm(self.state()[3:]-self.handle_position())>.16:
                self.held=False;self.phase=1;self.robot.open_gripper()
        if not self.held and self.phase==1 and np.linalg.norm(self.state()[3:]-self.target_position())<=self.grasp_distance:
            self.held=True;self.phase=2;self.robot.close_gripper()
            if self.task=='pick_cup':
                self.grasp_transform=self.robot.ee_transform().inverted()@self.cup.transformation
                self.cup.motion_type=MotionType.KINEMATIC
        for _ in range(10):
            if self.held and self.task=='pick_cup':self.cup.transformation=self.robot.ee_transform()@self.grasp_transform
            self.sim.step_physics(1/60);self.physics_steps+=1
        self.robot.update();self.steps+=1;now=self.state();target=self.target_position()
        yaw=math.atan2(-(target[2]-now[1]),target[0]-now[0])
        if self.phase==0 and np.linalg.norm(now[:2]-self.nav_goal[[0,2]])<=self.nav_distance and abs(wrap(yaw-now[2]))<.35:
            self.nav_success=True;self.phase=1
        complete=(target[1]>=self.cup_rest[1]+self.lift_height if self.task=='pick_cup' else self.opening()>=self.drawer_open_m)
        self.hold_ticks=self.hold_ticks+1 if self.held and complete else 0
        self.pick_success=self.task=='pick_cup' and self.hold_ticks>=5
        self.drawer_success=self.task=='open_drawer' and self.hold_ticks>=5
        failed=self.task=='pick_cup' and target[1]<self.cup_rest[1]-.20
        done=self.pick_success or self.drawer_success or failed or self.steps>=self.max_steps
        info=self.metrics();info['dropped']=bool(failed)
        return self.observe(),bool(done),info

    def metrics(self):
        return dict(task=self.task,domain=DOMAINS[self.shift],nav_success=int(self.nav_success),
            pick_success=int(self.pick_success),drawer_success=int(self.drawer_success),
            success=int(self.nav_success and (self.pick_success or self.drawer_success)),steps=self.steps,
            drawer_open_m=self.opening() if self.task=='open_drawer' else 0.,
            physics_substeps=self.physics_steps,collision_steps=self.collisions,
            manipulation_collision_steps=self.manipulation_collisions,collision_checks=self.collision_checks,
            path_length_m=self.path_length,shortest_path_m=self.shortest_distance,
            nav_spl=float(self.nav_success)*self.shortest_distance/max(self.shortest_distance,self.path_length,1e-6),
            raycasts=self.raycast_count,ik_fk_evals=self.ik_fk_evals)
