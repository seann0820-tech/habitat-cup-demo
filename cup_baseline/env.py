"""ReplicaCAD + Fetch; native Habitat-Sim physics and Habitat-Lab robot adapter.

State-observation benchmark, kinematic position control, proximity-gated assisted
grasp. No oracle path is passed to the policy. Not the official Habitat challenge.
"""
import math
import os
import json
from pathlib import Path
from types import SimpleNamespace

import magnum as mn
import numpy as np
import habitat_sim
from habitat_sim.physics import MotionType
from habitat.articulated_agents.robots.fetch_robot import FetchRobot


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def world_bb(obj):
    bb = obj.root_scene_node.cumulative_bb
    corners = [obj.transformation.transform_point(mn.Vector3(x, y, z))
               for x in (bb.min.x, bb.max.x) for y in (bb.min.y, bb.max.y)
               for z in (bb.min.z, bb.max.z)]
    a = np.asarray(corners)
    return a.min(0), a.max(0)


class CupEnv:
    """Actions: forward, yaw, EE-forward, EE-up, EE-side; all in [-1, 1].

    The shared task automaton switches navigation -> reach -> lift. Closing the
    assisted gripper is automatic within 9 cm; picking still requires reaching
    the actual cup and lifting it by 12 cm for five successive control steps.
    One control step advances 10 physics substeps (1/6 simulated second).
    """
    base_step = 0.18
    yaw_step = 0.35
    arm_step = 0.055
    grasp_distance = 0.09
    nav_distance = 0.22
    lift_height = 0.12
    rays_n = 12
    ray_range = 3.0

    def __init__(self, data, render=False, max_steps=200):
        self.data = Path(data).resolve()
        self.render_enabled = render
        self.max_steps = max_steps
        self.sim = None
        self.scene = None
        self.cup = None
        self.last_frame = None

    def load(self, scene):
        if scene == self.scene:
            return
        self.close()
        dataset_dir = self.data / 'replica_cad'
        urdf = self.data / 'robots/hab_fetch/robots/hab_fetch.urdf'
        if not urdf.exists() or not (dataset_dir/'configs/scenes'/f'{scene}.scene_instance.json').exists():
            raise FileNotFoundError('Run download_assets.py to completion before creating the environment')
        # The public v1.6 metadata also references unavailable sc4 navmeshes and an
        # obsolete robot path. Use a local derived config, retaining source assets.
        source = json.loads((dataset_dir/'replicaCAD.scene_dataset_config.json').read_text())
        source['navmesh_instances'] = {k:v for k,v in source['navmesh_instances'].items()
                                       if (dataset_dir/v).is_file()}
        source['articulated_objects']['paths']['.urdf'] = ['urdf/*/']
        clean_config = dataset_dir/'cup_baseline.scene_dataset_config.json'
        if not clean_config.exists():
            clean_config.write_text(json.dumps(source, indent=2))
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_dataset_config_file = str(clean_config)
        cfg.scene_id = scene
        cfg.enable_physics = True
        cfg.create_renderer = self.render_enabled
        cfg.requires_textures = self.render_enabled
        cfg.gpu_device_id = int(os.environ.get("HABITAT_RENDER_GPU_ID", "0")) if self.render_enabled else 0
        agent = habitat_sim.agent.AgentConfiguration()
        agent.sensor_specifications = []
        if self.render_enabled:
            sensor = habitat_sim.CameraSensorSpec()
            sensor.uuid = 'third_rgb'
            sensor.sensor_type = habitat_sim.SensorType.COLOR
            sensor.resolution = [384, 512]
            sensor.position = [0.0, 1.5, 0.0]
            agent.sensor_specifications = [sensor]
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [agent]))
        self.scene = scene
        # Furniture is fixed for this benchmark; doors are not task variables.
        for obj in self.sim.get_rigid_object_manager().get_objects_by_handle_substring().values():
            obj.motion_type = MotionType.STATIC
        for obj in self.sim.get_articulated_object_manager().get_objects_by_handle_substring().values():
            obj.motion_type = MotionType.STATIC
        ns = habitat_sim.NavMeshSettings()
        ns.set_defaults()
        ns.agent_radius = 0.30
        ns.agent_height = 1.4
        ns.include_static_objects = True
        if not self.sim.recompute_navmesh(self.sim.pathfinder, ns):
            raise RuntimeError('Could not build a navigation mesh for ' + scene)
        robot_cfg = SimpleNamespace(articulated_agent_urdf=str(urdf))
        self.robot = FetchRobot(robot_cfg, self.sim)
        self.robot.reconfigure()
        self.robot.sim_obj.motion_type = MotionType.KINEMATIC
        self.robot_id = self.robot.sim_obj.object_id
        self.initial_q = np.asarray(self.robot.params.arm_init_params).copy()
        self.cup = None

    def _ray(self, origin, direction, limit, ignore_robot=True):
        result = self.sim.cast_ray(habitat_sim.geo.Ray(mn.Vector3(origin), mn.Vector3(direction)), limit)
        for hit in result.hits:
            if ignore_robot and hit.object_id == self.robot_id:
                continue
            return hit
        return None

    def _table_goal(self, rng):
        rom = self.sim.get_rigid_object_manager()
        tables = list(rom.get_objects_by_handle_substring('table').values())
        rng.shuffle(tables)
        candidates = []
        for table in tables:
            lo, hi = world_bb(table)
            if not 0.55 < hi[1] < 1.15:
                continue
            center = (hi + lo) / 2
            for _ in range(15):
                xz = center[[0, 2]] + rng.uniform(-0.20, 0.20, 2) * (hi-lo)[[0, 2]]
                hit = self._ray([xz[0], hi[1]+0.3, xz[1]], [0, -1, 0], 1.0)
                if hit is None or hit.object_id != table.object_id:
                    continue
                top = np.asarray(hit.point)
                for angle in rng.permutation(np.linspace(-np.pi, np.pi, 16, endpoint=False)):
                    expected = top + np.array([np.cos(angle)*0.75, -top[1], np.sin(angle)*0.75])
                    nav = np.asarray(self.sim.pathfinder.snap_point(expected))
                    if not np.isfinite(nav).all() or np.linalg.norm(nav[[0, 2]]-expected[[0, 2]]) > 0.18:
                        continue
                    dist = np.linalg.norm(nav[[0, 2]]-top[[0, 2]])
                    if 0.58 < dist < 0.88:
                        candidates.append((table.handle, top.copy(), nav.copy()))
                if candidates:
                    return candidates[int(rng.integers(len(candidates)))]
        raise RuntimeError('No valid reachable table surface in scene ' + self.scene)

    def geodesic(self, start, end):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        return float(path.geodesic_distance) if self.sim.pathfinder.find_path(path) else float('inf')

    def reset(self, scene, seed, start_range=(1.5, 4.0), shift=0):
        self.load(scene)
        rng = np.random.default_rng(seed)
        self.sim.seed(seed)
        self.sim.pathfinder.seed(seed)
        if self.cup is not None:
            self.sim.get_rigid_object_manager().remove_object_by_id(self.cup.object_id)
        self.robot.base_pos = mn.Vector3(50, 0, 50)  # out of placement rays
        self.robot.base_rot = 0.0
        self.robot.arm_joint_pos = self.initial_q
        self.robot.update()
        self.table_handle, surface, self.nav_goal = self._table_goal(rng)
        tm = self.sim.get_object_template_manager()
        cup_handles = sorted(tm.get_template_handles('frl_apartment_cup_01'))
        if not cup_handles:
            raise RuntimeError('ReplicaCAD cup template is missing')
        self.cup = self.sim.get_rigid_object_manager().add_object_by_template_handle(cup_handles[0])
        self.cup.motion_type = MotionType.DYNAMIC
        lo, hi = world_bb(self.cup)
        self.cup.translation = mn.Vector3(surface + np.array([0.0, 0.004-lo[1], 0.0]))
        for _ in range(60):
            self.sim.step_physics(1/60)
        self.cup_rest = np.asarray(self.cup.translation).copy()
        if abs(world_bb(self.cup)[0][1]-surface[1]) > 0.06:
            raise RuntimeError('Cup failed to settle on table; episode not generated')
        self.start = None
        for _ in range(1000):
            p = np.asarray(self.sim.pathfinder.get_random_navigable_point())
            d = self.geodesic(p, self.nav_goal)
            if start_range[0] <= d <= start_range[1]:
                self.start, self.shortest_distance = p, d
                break
        if self.start is None:
            raise RuntimeError('Could not sample connected start in requested distance range')
        self.robot.base_pos = mn.Vector3(self.start)
        self.robot.base_rot = float(rng.uniform(-np.pi, np.pi))
        self.robot.open_gripper()
        self.robot.update()
        self.steps = self.physics_steps = self.raycast_count = self.ik_fk_evals = 0
        self.phase = 0
        self.nav_success = self.pick_success = self.held = False
        self.hold_ticks = 0
        self.path_length = self.collisions = 0
        self.shift = shift  # 0 nominal, 1 slower base, 2 slower arm; continual stream
        self.grasp_transform = None
        self.manifest = dict(scene=scene, seed=int(seed), table=self.table_handle,
                             cup_start=self.cup_rest.tolist(), start=self.start.tolist(),
                             start_yaw=float(self.robot.base_rot), nav_goal=self.nav_goal.tolist(),
                             shortest_distance=self.shortest_distance, shift=shift)
        return self.observe()

    def state(self):
        b = np.asarray(self.robot.base_pos)
        ee = np.asarray(self.robot.ee_transform().translation)
        return np.array([b[0], b[2], self.robot.base_rot, *ee], dtype=np.float32)

    def observe(self):
        x = self.state()
        angles = x[2] + np.arange(self.rays_n)*2*np.pi/self.rays_n
        ranges, points = [], []
        for a in angles:
            direction = [np.cos(a), 0., -np.sin(a)]
            hit = self._ray([x[0], self.start[1]+0.30, x[1]], direction, self.ray_range)
            self.raycast_count += 1
            d = self.ray_range if hit is None else min(self.ray_range, float(hit.ray_distance))
            ranges.append(d)
            if d < self.ray_range-0.01:
                points.append([x[0]+d*np.cos(a), x[1]-d*np.sin(a)])
        cup = np.asarray(self.cup.translation, dtype=np.float32)
        # Goal is the cup center until attached, then a height above the rest pose.
        ee_goal = cup.copy() if not self.held else self.cup_rest + [0, 0.23, 0]
        return dict(x=x, joints=np.asarray(self.robot.arm_joint_pos, dtype=np.float32),
                    rays=np.asarray(ranges, dtype=np.float32), obstacles=np.asarray(points, dtype=np.float32).reshape(-1, 2),
                    nav_goal=self.nav_goal[[0, 2]].astype(np.float32), cup=cup,
                    ee_goal=np.asarray(ee_goal, dtype=np.float32), phase=self.phase, held=self.held)

    def _ik_delta(self, delta):
        # Damped least-squares IK through the actual Habitat Fetch forward kinematics.
        # Position servo is shared by every algorithm; FK calls are counted separately.
        q = np.asarray(self.robot.arm_joint_pos).copy()
        target = np.asarray(self.robot.ee_transform().translation) + delta
        for _ in range(3):
            self.robot.arm_joint_pos = q
            p = np.asarray(self.robot.ee_transform().translation).copy()
            self.ik_fk_evals += 1
            jac = np.empty((3, len(q)))
            for j in range(len(q)):
                probe = q.copy(); probe[j] += 0.002
                self.robot.arm_joint_pos = probe
                jac[:, j] = (np.asarray(self.robot.ee_transform().translation)-p)/0.002
                self.ik_fk_evals += 1
            dq = jac.T @ np.linalg.solve(jac@jac.T + 0.015*np.eye(3), target-p)
            q += np.clip(dq, -0.15, 0.15)
            indices = [self.robot.joint_pos_indices[j] for j in self.robot.params.arm_joints]
            q = np.clip(q, np.asarray(self.robot.joint_limits[0])[indices], np.asarray(self.robot.joint_limits[1])[indices])
        self.robot.arm_joint_pos = q

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float), -1, 1)
        if a.shape != (5,) or not np.isfinite(a).all():
            raise ValueError('Action must be a finite 5-vector')
        before = self.state()
        if self.phase == 0:
            a[2:] = 0
            theta = float(wrap(before[2] + a[1]*self.yaw_step))
            self.robot.base_rot = theta
            b = np.asarray(self.robot.base_pos)
            displacement = a[0]*self.base_step*(0.65 if self.shift == 1 else 1.0)
            desired = b + displacement*np.array([np.cos(theta), 0, -np.sin(theta)])
            actual = np.asarray(self.sim.pathfinder.try_step(b, desired))
            if not np.isfinite(actual).all():
                actual = b
            self.collisions += int(np.linalg.norm(actual-desired) > 0.01)
            self.robot.base_pos = mn.Vector3(actual)
            self.path_length += float(np.linalg.norm(actual-b))
        else:
            a[:2] = 0
            theta = before[2]
            local = a[2:]*self.arm_step*(0.65 if self.shift == 2 else 1.0)
            delta = np.array([np.cos(theta)*local[0]+np.sin(theta)*local[2],
                              local[1], -np.sin(theta)*local[0]+np.cos(theta)*local[2]])
            self._ik_delta(delta)
        self.robot.update()
        if not self.held and self.phase == 1:
            distance = np.linalg.norm(np.asarray(self.robot.ee_transform().translation)-np.asarray(self.cup.translation))
            if distance <= self.grasp_distance:
                self.held = True
                self.phase = 2
                self.robot.close_gripper()
                self.grasp_transform = self.robot.ee_transform().inverted() @ self.cup.transformation
                self.cup.motion_type = MotionType.KINEMATIC
        for _ in range(10):
            if self.held:
                self.cup.transformation = self.robot.ee_transform() @ self.grasp_transform
            self.sim.step_physics(1/60)
            self.physics_steps += 1
        self.robot.update()
        self.steps += 1
        now = self.state()
        cup = np.asarray(self.cup.translation)
        desired_yaw = math.atan2(-(cup[2]-now[1]), cup[0]-now[0])
        near = np.linalg.norm(now[:2]-self.nav_goal[[0, 2]]) <= self.nav_distance
        aligned = abs(wrap(desired_yaw-now[2])) < 0.35
        if self.phase == 0 and near and aligned:
            self.nav_success = True
            self.phase = 1
        if self.held and cup[1] >= self.cup_rest[1]+self.lift_height:
            self.hold_ticks += 1
        else:
            self.hold_ticks = 0
        self.pick_success = self.hold_ticks >= 5
        failed = cup[1] < self.cup_rest[1]-0.20
        done = self.pick_success or failed or self.steps >= self.max_steps
        obs = self.observe()
        info = self.metrics()
        info['dropped'] = bool(failed)
        return obs, done, info

    def metrics(self):
        return dict(nav_success=int(self.nav_success), pick_success=int(self.pick_success),
                    success=int(self.nav_success and self.pick_success), steps=self.steps,
                    physics_substeps=self.physics_steps, collision_steps=self.collisions,
                    path_length_m=self.path_length, shortest_path_m=self.shortest_distance,
                    nav_spl=float(self.nav_success)*self.shortest_distance/max(self.shortest_distance, self.path_length, 1e-6),
                    raycasts=self.raycast_count, ik_fk_evals=self.ik_fk_evals)

    def render(self):
        if not self.render_enabled:
            raise RuntimeError('Create CupEnv(render=True) to record RGB video')
        # Follow camera placement supplied by Habitat-Lab FetchRobot.update().
        frame = self.sim.get_sensor_observations()['third_rgb'][..., :3]
        self.last_frame = frame
        return frame

    def close(self):
        if self.sim is not None:
            self.sim.close()
        self.sim = None
        self.scene = None
        self.cup = None
