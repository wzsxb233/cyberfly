"""Real articulated fly tasks with explicit engineered rewards and visible goals."""
from __future__ import annotations

from copy import deepcopy
import mujoco
import numpy as np
from gymnasium import spaces
from rl.env import FlyNavigationEnv

from .fly_task_specs import TASK_NAMES, PARAMETERS, CURRICULUM_LEVELS, task_spec

def _coordinates(value, width, minimum, maximum, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != width or not minimum <= len(array) <= maximum or not np.isfinite(array).all() or np.abs(array[:, :2]).max() > 90:
        raise ValueError(f"{name} needs {minimum}..{maximum} rows of {width} finite values, XY within ±90 mm.")
    if width == 3 and (np.any(array[:, 2] <= 0) or np.any(array[:, 2] > 10)):
        raise ValueError("Danger radii must be within (0,10] mm.")
    return array.tolist()


def validate_task_config(config):
    result = deepcopy(config)
    if result["task_type"] not in TASK_NAMES:
        raise ValueError(f"task_type must be one of {list(TASK_NAMES)}.")
    for name, lo, hi in [("difficulty",0,3),("max_episode_steps",1,1000),("food_count",1,8),
                          ("waypoint_count",2,8),("follow_steps",1,1000),("hazard_count",1,4)]:
        if type(result[name]) is not int or not lo <= result[name] <= hi:
            raise ValueError(f"{name} must be an integer within {lo}..{hi}.")
    for name in ("randomization", "hazard_terminates"):
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be boolean.")
    speed = result["target_speed_mm_s"]
    if isinstance(speed, bool) or not np.isfinite(speed) or not 0 <= speed <= 10:
        raise ValueError("target_speed_mm_s must be within 0..10.")
    if result.get("curriculum_level") is not None:
        level = result["curriculum_level"]
        if type(level) is not int or not 0 <= level <= 3:
            raise ValueError("curriculum_level must be an integer within 0..3.")
        result["difficulty"] = level
    if result.get("goal_mm") is not None:
        result["goal_mm"] = _coordinates([result["goal_mm"]],2,1,1,"goal_mm")[0]
    if result.get("waypoints_mm") is not None:
        result["waypoints_mm"] = _coordinates(result["waypoints_mm"],2,2,8,"waypoints_mm")
    if result.get("hazards_mm") is not None:
        result["hazards_mm"] = _coordinates(result["hazards_mm"],3,1,4,"hazards_mm")
    return result


class FlyTaskScenario(FlyNavigationEnv):
    """36-state task family sharing real FlyGym dynamics, never steering scripts."""
    def __init__(self, *, scenario_name=None, task_type="food", difficulty=0, randomization=True,
                 curriculum_level=None, food_count=1, waypoint_count=3, target_speed_mm_s=0.8,
                 follow_steps=40, hazard_count=1, hazard_terminates=True, goal_mm=None,
                 waypoints_mm=None, hazards_mm=None, max_episode_steps=300, **body_kwargs):
        self._task_config = validate_task_config(dict(task_type=task_type, difficulty=difficulty,
            randomization=randomization, curriculum_level=curriculum_level, food_count=food_count,
            waypoint_count=waypoint_count, target_speed_mm_s=target_speed_mm_s, follow_steps=follow_steps,
            hazard_count=hazard_count, hazard_terminates=hazard_terminates, goal_mm=goal_mm,
            waypoints_mm=waypoints_mm, hazards_mm=hazards_mm, max_episode_steps=max_episode_steps))
        self._pending_config = None
        self.scenario_spec = task_spec(scenario_name or f"fly_{task_type}", task_type)
        self._started = False
        self._done = False
        self._last_info = {}
        self._target_velocity = np.zeros(2)
        self._points = np.zeros((1,2))
        self._point_index = 0
        self._hazards = np.empty((0,3))
        self._hold_steps = 0
        self._collected = 0
        super().__init__(max_episode_steps=max_episode_steps, **body_kwargs)
        self.observation_space = spaces.Box(-10,10,(36,),dtype=np.float32)
        self._task_geoms = {name: mujoco.mj_name2id(self.sim.mj_model,mujoco.mjtObj.mjOBJ_GEOM,name)
                           for name in [*[f"task_waypoint_{i}" for i in range(8)], *[f"task_hazard_{i}" for i in range(4)]]}

    def _decorate_world(self, world):
        for i in range(8):
            world.mjcf_root.worldbody.add_geom(name=f"task_waypoint_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[0,0,.15], size=[.16,0,0], rgba=[1,.65,.1,0], contype=0, conaffinity=0)
        for i in range(4):
            world.mjcf_root.worldbody.add_geom(name=f"task_hazard_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0,0,.012], size=[.5,.01,0], rgba=[1,.12,.08,0], contype=0, conaffinity=0)

    def _random_point(self, origin, *, index=0):
        preset = CURRICULUM_LEVELS[self._task_config["difficulty"]]
        if self._task_config["randomization"]:
            distance = self.np_random.uniform(*preset["goal_distance_mm"])
            angle = self.np_random.uniform(-preset["goal_angle_rad"],preset["goal_angle_rad"])
        else:
            distance = float(np.mean(preset["goal_distance_mm"]))
            angle = (0 if index==0 else (-1 if index%2 else 1)*preset["goal_angle_rad"]*.5)
        return np.asarray(origin) + distance*np.array([np.cos(angle),np.sin(angle)])

    def _update_markers(self):
        model = self.sim.mj_model
        for geom in self._task_geoms.values(): model.geom_rgba[geom,3] = 0
        if self._task_config["task_type"] == "waypoints":
            for i, point in enumerate(self._points):
                geom = self._task_geoms[f"task_waypoint_{i}"]
                model.geom_pos[geom,:2] = point
                model.geom_rgba[geom] = [.4,.45,.45,.35] if i<self._point_index else [1,.7,.15,.8]
        for i,(x,y,radius) in enumerate(self._hazards):
            geom=self._task_geoms[f"task_hazard_{i}"]
            model.geom_pos[geom,:2]=[x,y];model.geom_size[geom,0]=radius;model.geom_rgba[geom,3]=.65
        model.geom_rgba[self.goal_geom_id] = [1,.64,.12,.95] if self._task_config["task_type"]=="food" else [.3,1,.2,.8]
        mujoco.mj_forward(model,self.sim.mj_data)

    def reset(self, *, seed=None, options=None):
        if options:
            options=dict(options)
            if "goal_xy" in options: options["goal_mm"]=options.pop("goal_xy")
            self.apply_task(options)
        if self._pending_config is not None:
            self._task_config=self._pending_config;self._pending_config=None
        self.max_episode_steps=self._task_config["max_episode_steps"]
        # The original reset supplies deterministic CPG initialisation and warms
        # up the articulated fly. Only task markers are changed afterwards.
        super().reset(seed=seed)
        config=self._task_config;origin=self._pose()[0][:2]
        task=config["task_type"]
        self.scenario_spec=task_spec(self.scenario_spec["name"],task)
        self._point_index=self._collected=self._hold_steps=0
        self._target_velocity=np.zeros(2)
        if task=="waypoints":
            if config["waypoints_mm"] is not None: self._points=np.asarray(config["waypoints_mm"],dtype=float)
            else:
                points=[];last=origin
                for i in range(config["waypoint_count"]): last=self._random_point(last,index=i);points.append(last)
                self._points=np.array(points)
            goal=self._points[0]
        else:
            goal=np.asarray(config["goal_mm"]) if config["goal_mm"] is not None else self._random_point(origin)
            self._points=np.asarray([goal])
        self.set_goal(goal)
        self._follow_origin=self.goal.copy()
        self._follow_clock=0.0
        if config["hazards_mm"] is not None:
            self._hazards=np.asarray(config["hazards_mm"],dtype=float)
        elif task=="hazard":
            direction=self.goal-origin;normal=np.array([-direction[1],direction[0]])/(np.linalg.norm(direction)+1e-9)
            radius=.25+.12*config["difficulty"]
            self._hazards=np.array([[*(origin+direction*(i+1)/(config["hazard_count"]+1)+normal*radius*.8*(-1 if i%2 else 1)),radius]
                                     for i in range(config["hazard_count"])])
        else: self._hazards=np.empty((0,3))
        self._update_markers()
        self._started=True;self._done=False
        self._last_info=self._task_info()
        return self.observe(),deepcopy(self._last_info)

    def _task_info(self, *, success=False, fallen=False, hazard_hit=False, terminated=False, truncated=False, events=None):
        config=self._task_config
        return {**super()._info(success=success,fallen=fallen),"scenario":self.scenario_spec["name"],
            "task_type":config["task_type"],"difficulty":config["difficulty"],"randomization":config["randomization"],
            "step_limit":self.max_episode_steps,"food_collected":self._collected,"food_target":config["food_count"],
            "waypoint_index":self._point_index,"waypoint_count":len(self._points),"waypoints_mm":self._points.tolist(),
            "target_velocity_mm_s":self._target_velocity.tolist(),"follow_hold_steps":self._hold_steps,
            "follow_required_steps":config["follow_steps"],"follow_radius_mm":CURRICULUM_LEVELS[config["difficulty"]]["follow_radius_mm"],
            "hazards_mm":self._hazards.tolist(),"hazard_hit":bool(hazard_hit),"hazard_kind":"non-contact geometric danger zone",
            "is_success":bool(success),"terminated":bool(terminated),"truncated":bool(truncated),
            "failure_reason":"fall" if fallen else "danger_zone" if hazard_hit and config["hazard_terminates"] else None,
            "task_events":list(events or []),"task_config":deepcopy(config),"pending_task":deepcopy(self._pending_config)}

    @staticmethod
    def _crosses_circle(start,end,zone):
        delta=end-start
        t=np.clip(np.dot(zone[:2]-start,delta)/(np.dot(delta,delta)+1e-12),0,1)
        return np.linalg.norm(start+t*delta-zone[:2]) < zone[2]

    def step(self, action):
        if not self._started or self._done: raise RuntimeError("Call reset() before stepping a new or finished episode.")
        config=self._task_config;task=config["task_type"]
        old_pos=self._pose()[0][:2].copy()
        if task=="follow":
            self._follow_clock+=self.action_dt
            speed=config["target_speed_mm_s"]*(1+.25*config["difficulty"])
            amplitude=.15*(1+config["difficulty"])
            t=self._follow_clock
            target=self._follow_origin+np.array([8*np.sin(speed*t/8),amplitude*np.sin(2*t)])
            self._target_velocity=np.array([speed*np.cos(speed*t/8),2*amplitude*np.cos(2*t)])
            self.set_goal(target)  # Refresh previous_distance so marker motion earns no progress reward.
        action,pos,upright=self._advance_physics(action)
        distance=float(np.linalg.norm(self.goal-pos[:2]))
        rc=self.reward_config
        fallen=bool(upright<.2 or pos[2]<.15)
        hazard_hit=any(self._crosses_circle(old_pos,pos[:2],zone) for zone in self._hazards)
        failed=fallen or (hazard_hit and config["hazard_terminates"])
        reward=rc["progress"]*(self._previous_distance-distance)-rc["time"]-rc["smoothness"]*float(np.square(action-self._last_action).sum())
        reward-=rc["fall"]*fallen+8*hazard_hit
        success=False;events=[]
        if not failed:
            if task=="follow":
                inside=distance<CURRICULUM_LEVELS[config["difficulty"]]["follow_radius_mm"]
                self._hold_steps=self._hold_steps+1 if inside else 0
                reward+=.05 if inside else -.02*min(distance,10)
                success=self._hold_steps>=config["follow_steps"]
            elif distance<rc["success_radius_mm"]:
                if task=="food":
                    self._collected+=1;reward+=3;events.append("food_collected")
                    self.show_feedback("food",.9,800)
                    success=self._collected>=config["food_count"]
                    if not success: self.set_goal(self._random_point(pos[:2],index=self._collected))
                elif task=="waypoints":
                    self._point_index+=1;reward+=3;events.append("waypoint_reached")
                    success=self._point_index>=len(self._points)
                    if not success: self.set_goal(self._points[self._point_index])
                    self._update_markers()
                else: success=True
        reward+=rc["success"]*success
        self._last_action=action.copy()
        self._previous_distance=float(np.linalg.norm(self.goal-pos[:2]))
        self._step_count+=1;self._episode_reward+=reward
        terminated=bool(success or failed);truncated=bool(self._step_count>=self.max_episode_steps and not terminated)
        self._done=terminated or truncated
        self._last_info=self._task_info(success=success,fallen=fallen,hazard_hit=hazard_hit,terminated=terminated,truncated=truncated,events=events)
        return self.observe(),float(reward),terminated,truncated,deepcopy(self._last_info)

    def observe(self):
        base=super().observe()
        config=self._task_config;task=config["task_type"]
        pos,yaw,_=self._pose();c,s=np.cos(yaw),np.sin(yaw);rotation=np.array([[c,s],[-s,c]])
        if len(self._hazards):
            distances=np.linalg.norm(self._hazards[:,:2]-pos[:2],axis=1)-self._hazards[:,2]
            index=int(np.argmin(distances));hazard=np.r_[rotation@(self._hazards[index,:2]-pos[:2])/10,distances[index]/10]
        else: hazard=np.array([0,0,1])
        progress=self._collected/config["food_count"] if task=="food" else self._point_index/max(1,len(self._points)) if task=="waypoints" else self._hold_steps/config["follow_steps"] if task=="follow" else 0
        extra=np.r_[rotation@self._target_velocity/10,[float(task==name) for name in TASK_NAMES],progress,hazard,
                    self._hold_steps/config["follow_steps"],max(0,1-self._step_count/self.max_episode_steps),config["difficulty"]/3,config["target_speed_mm_s"]/10]
        return np.clip(np.r_[base,extra],-10,10).astype(np.float32)

    def apply_task(self, task):
        if not isinstance(task,dict) or not task: raise ValueError("Task must be a nonempty object.")
        if "task" in task and (set(task)!={"task"} or not isinstance(task["task"],dict) or not task["task"]):
            raise ValueError("Nested task must contain exactly one nonempty task object.")
        task=task.get("task",task)
        unknown=set(task)-set(PARAMETERS)
        if unknown: raise ValueError(f"Unsupported task settings: {sorted(unknown)}")
        config={**deepcopy(self._pending_config or self._task_config),**deepcopy(task)}
        # An explicit difficulty update supersedes an earlier curriculum override.
        if "difficulty" in task and "curriculum_level" not in task: config["curriculum_level"]=None
        self._pending_config=validate_task_config(config)
        return {"applied":deepcopy(self._pending_config),"requires_reset":True,"effect":"Task changes begin next episode; body state is unchanged"}

    def task_state(self):
        return {**deepcopy(self._last_info),"scenario":self.scenario_spec["name"],
                "phase":"finished" if self._done else "running" if self._started else "not_reset", "pending_task":deepcopy(self._pending_config),
                "feedback_visual":self.feedback_visual_state()}
