"""Measured navigation, hover and landing objectives over one unchanged FlyBody."""
from copy import deepcopy

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .flybody_specs import flybody_spec, BODY_OBSERVATION_DIM

MODES = ('walking', 'flight', 'landing')


class FlyBodyTaskScenario(gym.Wrapper):
    """Nine real actuator-pattern commands; reward never changes physical state."""

    def __init__(self, *, mode='flight', target_altitude_mm=3., goal_mm=None,
                 max_episode_steps=150, control_dt=.01, render_mode='rgb_array',
                 flight_reward_version='v1', **kwargs):
        from .flybody_physics import FlyBodyPhysics
        if flight_reward_version not in ('v1', 'airborne_v2'):
            raise ValueError('flight_reward_version must be v1 or airborne_v2.')
        self.flight_reward_version = flight_reward_version
        altitude, goal = self._targets(target_altitude_mm, [5., 0.] if goal_mm is None else goal_mm)
        super().__init__(FlyBodyPhysics(mode=mode, control_dt=control_dt,
            max_episode_steps=max_episode_steps, render_mode=render_mode, **kwargs))
        import mujoco
        self._flight_up_site = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_SITE, 'hover_up_dir')
        if self._flight_up_site < 0:
            self.env.close()
            raise ValueError('Complete FlyBody is missing its actual hover reference site.')
        from .flybody_feedback import FlyBodyFeedback
        self._feedback = FlyBodyFeedback()
        self.env.feedback_renderer = self._feedback.draw
        self.scenario_spec = flybody_spec()
        self.observation_space = spaces.Box(-np.inf, np.inf, (BODY_OBSERVATION_DIM,), dtype=np.float32)
        self.target_altitude_mm, self.goal_mm = altitude, goal
        self._started, self._done = False, False
        self._stable_s, self._inverted_s = 0., 0.
        self._last_info = {}
        self._last_reward = 0.
        self._episode_reward = 0.
        self._previous_error = None
        self._qpos_scales = None

    @staticmethod
    def _targets(altitude, goal):
        if type(altitude) not in (float, int) or not np.isfinite(altitude) or not 1 <= altitude <= 50:
            raise ValueError('target_altitude_mm must be finite within [1,50].')
        goal = np.asarray(goal, dtype=np.float64)
        if goal.shape != (2,) or not np.isfinite(goal).all() or np.any(np.abs(goal) > 100):
            raise ValueError('goal_mm must contain two finite XY coordinates within ±100 mm.')
        return float(altitude), goal.copy()

    @property
    def mode(self):
        return self.env.mode

    def _measure(self):
        packet = self.env.body_sensory_packet()
        position = np.asarray(packet['thorax']['position_world_mm'], dtype=np.float64)
        velocity = np.asarray(packet['thorax']['velocity_world_rot_rad_s_then_lin_mm_s'], dtype=np.float64)
        ground = sum(not c['exclude'] and 'cyberfly_ground' in c['geom_names'] for c in packet['contacts'])
        upright = float(self.env._pose()[2])
        flight_upright = float(self.env.mj_data.site_xmat[self._flight_up_site].reshape(3, 3)[2, 2])
        error = (float(np.linalg.norm(position[:2] - self.goal_mm)) if self.mode == 'walking'
                 else abs(float(position[2]) - self.target_altitude_mm) if self.mode == 'flight'
                 else max(0., float(position[2]) - .6))
        return {'position_world_mm': position.tolist(), 'linear_velocity_mm_s': velocity[3:].tolist(),
                'angular_velocity_rad_s': velocity[:3].tolist(), 'ground_contact_count': int(ground),
                'upright_cosine': upright, 'flight_upright_cosine': flight_upright, 'goal_error_mm': error}

    def reset(self, *, seed=None, options=None):
        self._feedback.clear()
        self.env.reset(seed=seed, options=options)
        packet = self.env.body_sensory_packet()
        self._qpos_scales = np.array([10. if unit == 'mm' else np.pi if unit == 'rad' else 1.
                                     for unit in packet['layout']['qpos_units']], np.float32)
        self._started, self._done = True, False
        self._stable_s, self._inverted_s = 0., 0.
        self._last_reward = 0.
        self._episode_reward = 0.
        self._last_info = self._measure()
        self._previous_error = self._last_info['goal_error_mm']
        self._last_info.update(success=False, failure_reason=None, reward_components={})
        return self.observe(), self.task_state()

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError('mode must be walking, flight or landing.')
        before = float(self.env.mj_data.time)
        model, data = self.env.mj_model, self.env.mj_data
        qpos, qvel = data.qpos.copy(), data.qvel.copy()
        result = self.env.set_mode(mode)
        if self.env.mj_model is not model or self.env.mj_data is not data or float(data.time) != before or not np.array_equal(qpos, data.qpos) or not np.array_equal(qvel, data.qvel):
            raise RuntimeError('A body mode switch changed the physical state.')
        self._stable_s, self._inverted_s = 0., 0.
        if self._started:
            self._last_info.update(self._measure(), success=False, failure_reason=None)
            self._previous_error = self._last_info['goal_error_mm']
        return result

    def apply_task(self, task):
        if not isinstance(task, dict) or not task or set(task) - {'mode', 'target_altitude_mm', 'goal_mm'}:
            raise ValueError('Unified-body tasks support mode, target_altitude_mm and goal_mm.')
        altitude, goal = self._targets(task.get('target_altitude_mm', self.target_altitude_mm), task.get('goal_mm', self.goal_mm))
        if task.get('mode', self.mode) not in MODES:
            raise ValueError('Unknown body mode.')
        self.target_altitude_mm, self.goal_mm = altitude, goal
        if 'mode' in task:
            self.set_mode(task['mode'])
        elif self._started:
            self._last_info.update(self._measure(), success=False)
            self._previous_error = self._last_info['goal_error_mm']
            self._stable_s = 0.
        return {'applied': {'mode': self.mode, 'target_altitude_mm': altitude, 'goal_mm': goal.tolist()},
                'physical_state_reset': False, 'body_mode': self.env.task_state()['last_mode_change']}

    def step(self, action):
        if not self._started or self._done:
            raise RuntimeError('Reset before stepping a finished unified-body episode.')
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError('Unified body requires nine finite actions in [-1,1].')
        _, _, terminated, truncated, _ = self.env.step(action)
        measured = self._measure()
        position = np.asarray(measured['position_world_mm'])
        velocity = np.asarray(measured['linear_velocity_mm_s'])
        angular = np.asarray(measured['angular_velocity_rad_s'])
        upright = measured['flight_upright_cosine'] if self.mode == 'flight' else measured['upright_cosine']
        dt = self.env.action_dt
        self._inverted_s = self._inverted_s + dt if upright < -.25 else 0.
        failure = 'physical_boundary' if terminated else 'inverted_for_100ms' if self._inverted_s >= .1 else None
        airborne_ground_failure = (self.mode == 'flight' and self.flight_reward_version == 'airborne_v2'
                                   and (self.env._ground_contact_substeps > 0 or measured['ground_contact_count'] > 0))
        if airborne_ground_failure:
            failure = 'ground_contact_during_airborne_hover'
        if self.mode == 'walking':
            stable = measured['goal_error_mm'] <= .6 and upright > .25
        elif self.mode == 'flight':
            stable = (measured['goal_error_mm'] <= .5 and upright > .25 and not measured['ground_contact_count']
                      and not airborne_ground_failure and np.linalg.norm(velocity) <= 20.)
        else:
            stable = (measured['ground_contact_count'] > 0 and position[2] <= 1.2
                      and upright > .25 and np.linalg.norm(velocity) <= 20.)
        self._stable_s = self._stable_s + dt if stable else 0.
        hold = .05 if self.mode == 'walking' else .1 if self.mode == 'landing' else .5
        success = self._stable_s + 1e-9 >= hold and failure is None
        distance_progress = float(self._previous_error - measured['goal_error_mm'])
        target_score = float(np.exp(-measured['goal_error_mm'] / max(1., self.target_altitude_mm if self.mode == 'flight' else 3.)))
        components = {'progress': .1 * distance_progress, 'target_proximity': .04 * target_score,
                      'upright': .02 * max(-1., upright), 'control_cost': -.005 * float(np.mean(action**2)),
                      'angular_speed_cost': -.01 * min(2., float(np.linalg.norm(angular)) / 200.),
                      'horizontal_speed_cost': 0. if self.mode == 'walking' else -.01 * min(2., float(np.linalg.norm(velocity[:2])) / 50.),
                      'success_bonus': 2. if success else 0., 'failure_penalty': -2. if failure else 0.}
        if airborne_ground_failure:
            # This optional task starts in the air. Ground support cannot earn
            # positive shaping; negative progress/attitude costs still apply.
            # This changes the reward and termination only, never the physics.
            for name in ('progress', 'target_proximity', 'upright'):
                components[name] = min(0., components[name])
        self._last_reward = float(sum(components.values()))
        self._episode_reward += self._last_reward
        self._last_info = {**measured, 'success': success, 'failure_reason': failure,
                           'stable_duration_s': self._stable_s, 'required_stable_duration_s': hold,
                           'reward_components': components}
        self._previous_error = measured['goal_error_mm']
        terminated = bool(terminated or failure or success)
        self._done = bool(terminated or truncated)
        self._last_info.update(terminated=terminated, truncated=bool(truncated))
        return self.observe(), self._last_reward, terminated, bool(truncated), self.task_state()

    def observe(self):
        if not self._started:
            raise RuntimeError('Reset before observing this unified body.')
        raw = self.env.observe().astype(np.float32, copy=True)
        nq, nv = self.env.mj_model.nq, self.env.mj_model.nv
        if raw.shape != (nq + nv + 3,) or nq + nv + 3 != 220:
            raise ValueError('The complete physical body observation layout changed.')
        raw[:nq] /= self._qpos_scales
        raw[nq:nq + nv] /= 50.
        goal = np.r_[self.goal_mm / 10., self.target_altitude_mm / 10., self._stable_s].astype(np.float32)
        return np.r_[raw, goal].astype(np.float32)

    def task_state(self):
        return {**self.env.task_state(), **deepcopy(self._last_info),
                'scenario': self.scenario_spec['name'],
                'phase': 'finished' if self._done else 'running' if self._started else 'not_reset',
                'target_altitude_mm': self.target_altitude_mm, 'goal_mm': self.goal_mm.tolist(),
                'reward': self._last_reward, 'episode_reward': self._episode_reward,
                'is_success': bool(self._last_info.get('success', False)),
                'reward_version': ('flybody_airborne_v2' if self.mode == 'flight' and self.flight_reward_version == 'airborne_v2'
                                   else 'flybody_objectives_v1'),
                'flight_reward_version': self.flight_reward_version,
                'flight_objective': ('Airborne stabilization; any actual ground contact fails the episode; not a ground takeoff task'
                                     if self.flight_reward_version == 'airborne_v2'
                                     else 'Original hover shaping; ground contact excludes success but does not itself terminate'),
                'reward_design': 'Measured navigation/hover/landing shaping; privileged simulator state, not a neural reward claim',
                'flight_attitude_reference': 'Actual hover_up_dir site world-up cosine; respects official -47.5-degree thorax flight pitch',
                'task_observation': '109 qpos normalized by unit, 108 qvel/50, 3 mode flags, goalXY/10, target height/10, stable seconds',
                'flight_success_claimed': bool(self._last_info.get('success') and self.mode == 'flight'),
                'feedback_visual': self._feedback.state()}

    def show_feedback(self, kind='aversive', intensity=1., duration_ms=800):
        return self._feedback.show(kind, intensity, duration_ms)

    def render(self):
        return self.env.render()
