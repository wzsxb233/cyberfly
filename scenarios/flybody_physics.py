"""One official FlyBody physical model with continuously available legs/wings.

The complete upstream MJCF remains in CGS units. No mode change rebuilds the
plant, teleports it, changes gravity, or applies artificial upward forces.
The gait/wing pattern controllers are disclosed engineering baselines, not
trained locomotion policies. Numerical body packets convert geometry/forces
to the existing millimetre interface and carry a new, exact model layout hash.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

os.environ.setdefault('MUJOCO_GL', 'egl')
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'vendor/flybody/flybody/fruitfly/assets'
SOURCE_XML = ASSETS / 'fruitfly.xml'
sys.path.insert(0, str(ROOT / 'vendor/flybody'))
from flybody.tasks.pattern_generators import WingBeatPatternGenerator
MODES = ('walking', 'flight', 'landing')
WING_NAMES = tuple(f'wing_{axis}_{side}' for side in ('left', 'right') for axis in ('yaw', 'roll', 'pitch'))


def _quat_mul(a, b):
    result = np.zeros(4)
    mujoco.mju_mulQuat(result, np.asarray(a, float), np.asarray(b, float))
    return result


def _inv(q):
    value = np.asarray(q, float).copy()
    value /= np.linalg.norm(value)
    value[1:] *= -1
    return value


def _rotate(v, q):
    result = np.zeros(3)
    mujoco.mju_rotVecQuat(result, np.asarray(v, float), np.asarray(q, float))
    return result


def _numbers(text, default):
    return np.asarray(default, float) if text is None else np.fromstring(text, sep=' ')


def _text(values):
    return ' '.join(format(float(value), '.17g') for value in values)


def _physical_xml(source_bytes=None):
    """Apply the official Flying wing parameters, preserving all original DOFs.

    Wing-frame transformation mirrors fruitfly.change_body_frame: render/fluid
    geoms remain in their original location while active wing axes use the
    original flight stroke frame. Both legs and wings remain in the same model.
    """
    tree = ET.parse(SOURCE_XML) if source_bytes is None else ET.ElementTree(ET.fromstring(source_bytes))
    root = tree.getroot()
    root.find('compiler').set('meshdir', str(ASSETS))
    root.find('option').set('timestep', '0.00005')
    root.find('option').set('integrator', 'implicitfast')
    root.find('size').set('nconmax', '400')
    root.find('size').set('njmax', '2000')
    wing = root.find(".//default[@class='wing']")
    wing.find('joint').set('stiffness', '0.01')
    wing.find('joint').set('damping', '0.007769230')
    for axis in ('yaw', 'roll', 'pitch'):
        wing.find(f"default[@class='{axis}']/general").set('gainprm', '18')
    root.find(".//default[@class='wing-fluid']/geom").set('fluidshape', 'ellipsoid')
    root.find(".//default[@class='wing-fluid']/geom").set('fluidcoef', '1 0.5 1.5 1.7 1')
    # Same 47.5 degree body/stroke reference as the official Flying task.
    up_dir = np.array([np.cos(np.deg2rad(47.5) / 2), 0, np.sin(np.deg2rad(47.5) / 2), 0])
    root.find(".//site[@name='hover_up_dir']").set('quat', _text(up_dir))
    for side, side_q in (('left', [0, 0, 0, 1]), ('right', [0, -1, 0, 0])):
        body = root.find(f".//body[@name='wing_{side}']")
        old = _numbers(body.get('quat'), [1, 0, 0, 0])
        old /= np.linalg.norm(old)
        new = _quat_mul(side_q, _inv(up_dir))
        change = _quat_mul(_inv(new), old)
        body.set('quat', _text(new))
        for child in body:
            # Hinge axes keep their specified flight-frame axes, like upstream.
            if child.tag not in ('geom', 'site', 'body', 'camera', 'inertial', 'joint'):
                continue
            pos = _numbers(child.get('pos'), [0, 0, 0])
            child.set('pos', _text(_rotate(pos, change)))
            if child.tag != 'joint':
                if any(key in child.attrib for key in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
                    raise ValueError('Unexpected non-quaternion wing child orientation in pinned official model')
                child.set('quat', _text(_quat_mul(change, _numbers(child.get('quat'), [1, 0, 0, 0]))))
    # Match official Flying exclusions without removing legs or floor contacts.
    contacts = root.find('contact')
    for body in root.findall('.//body'):
        name = body.get('name', '')
        if any(part in name for part in ('coxa', 'femur', 'tibia', 'tarsus', 'claw')):
            for side in ('left', 'right'):
                ET.SubElement(contacts, 'exclude', body1=name, body2='wing_' + side)
    asset = root.find('asset')
    ET.SubElement(asset, 'texture', name='cyberfly_grid', type='2d', builtin='checker',
                  rgb1='.08 .13 .18', rgb2='.13 .2 .24', width='256', height='256')
    ET.SubElement(asset, 'material', name='cyberfly_floor', texture='cyberfly_grid', texrepeat='4 4', texuniform='true')
    world = root.find('worldbody')
    ET.SubElement(world, 'geom', name='cyberfly_ground', type='plane', size='10 10 .1',
                  material='cyberfly_floor', pos='0 0 0', contype='1', conaffinity='1', solref='.0002 1')
    visual = root.find('visual')
    if visual is None:
        visual = ET.SubElement(root, 'visual')
    scale = visual.find('scale')
    if scale is None:
        scale = ET.SubElement(visual, 'scale')
    scale.set('forcewidth', '.005')
    scale.set('contactwidth', '.003')
    map_element = visual.find('map')
    if map_element is None:
        map_element = ET.SubElement(visual, 'map')
    map_element.set('znear', '.001')
    return ET.tostring(root, encoding='unicode')


class FlyBodyPhysics(gym.Env):
    """True flight-capable plant; baseline controllers have not learned to fly.

    Action[9]: left/right leg gait drive, 6 wing-angle residuals, wingbeat
    frequency delta. Wings use the official documented approximate pattern
    when no measured wing-pattern asset is provided. Actions never set qpos.
    The wrapper deliberately returns reward=0; a task defines training rewards.
    """
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 25}

    def __init__(self, *, mode='walking', control_dt=.002, max_episode_steps=1000,
                 render_mode='rgb_array', render_width=640, render_height=480,
                 wing_pattern_path=None):
        super().__init__()
        if mode not in MODES or render_mode not in (None, 'rgb_array'):
            raise ValueError('Unsupported body/control or rendering mode')
        if not np.isfinite(control_dt) or not .0002 <= control_dt <= .05 or abs(control_dt / .0002 - round(control_dt / .0002)) > 1e-8:
            raise ValueError('control_dt must be a multiple of 0.0002 s within [0.0002,0.05]')
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError('A positive finite episode step limit is required')
        self.mode = mode
        self.action_dt = float(control_dt)
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.width, self.height = int(render_width), int(render_height)
        # Identify the exact source bytes used to construct this model once.
        # Fast control telemetry must not reopen and hash the same file 5,000
        # times per simulated second, or describe a later on-disk edit as the
        # source of an already compiled body.
        source_bytes = SOURCE_XML.read_bytes()
        self._source_xml_sha256 = hashlib.sha256(source_bytes).hexdigest()
        self.xml = _physical_xml(source_bytes)
        self._compiled_xml_sha256 = hashlib.sha256(self.xml.encode()).hexdigest()
        self.mj_model = mujoco.MjModel.from_xml_string(self.xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.sim = SimpleNamespace(mj_model=self.mj_model, mj_data=self.mj_data, timestep=float(self.mj_model.opt.timestep))
        self.thorax_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, 'thorax')
        self._ground_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, 'cyberfly_ground')
        if self._ground_geom_id < 0:
            raise ValueError('Unified body is missing its actual ground geometry')
        self.action_space = spaces.Box(-1., 1., shape=(9,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.mj_model.nq + self.mj_model.nv + 3,), dtype=np.float32)
        self.actuator_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self.mj_model.nu)]
        self.wing_ids = np.asarray([self.actuator_names.index(name) for name in WING_NAMES])
        wing_joints = [mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in WING_NAMES]
        self.wing_qpos = self.mj_model.jnt_qposadr[wing_joints].copy()
        self.wing_qvel = self.mj_model.jnt_dofadr[wing_joints].copy()
        self._leg_qpos = np.asarray([self.mj_model.jnt_qposadr[i] for i in range(self.mj_model.njnt)
            if any(part in (mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i) or '')
                   for part in ('coxa', 'femur', 'tibia', 'tarsus'))], dtype=int)
        if wing_pattern_path is not None:
            wing_pattern_path = Path(wing_pattern_path).expanduser().resolve(strict=True)
            if wing_pattern_path.suffix != '.npy' or wing_pattern_path.stat().st_size > 10_000_000:
                raise ValueError('Wing-pattern input must be a bounded numeric NPY file')
            pattern = np.load(wing_pattern_path, allow_pickle=False)
            if pattern.ndim != 2 or pattern.shape[1] != 3 or not 3 <= len(pattern) <= 100000 or not np.isfinite(pattern).all():
                raise ValueError('Wing-pattern input must contain finite [T,3] joint angles')
        self._wing_pattern_source = str(wing_pattern_path) if wing_pattern_path else 'official generated approximation; no measured wing data loaded'
        self._wing_pattern_sha = hashlib.sha256(wing_pattern_path.read_bytes()).hexdigest() if wing_pattern_path else None
        self._wbpg = WingBeatPatternGenerator(base_pattern_path=wing_pattern_path)
        self._leg_ids = [i for i, name in enumerate(self.actuator_names) if any(part in name for part in ('coxa', 'femur', 'tibia', 'tarsus')) and 'adhere' not in name]
        self._adhesion_ids = [i for i, name in enumerate(self.actuator_names) if 'adhere' in name]
        self._renderer = None
        self.feedback_renderer = None
        self._frame = None
        self._step_count = 0
        self._phase = 0.
        self._leg_phase = 0.
        self._done = False
        self._last_action = np.zeros(9, np.float32)
        self._last_mode_change = None
        self._initialization = None
        self._control_limit = self.mj_model.actuator_ctrlrange.copy()
        self._wing_fold = self.mj_model.qpos_spring[self.wing_qpos].copy()
        self._rest_ctrl = np.zeros(self.mj_model.nu)
        self._reset_done = False
        self._clear_step_contacts()

    def _clear_step_contacts(self):
        self._ground_contact_substeps = 0
        self._ground_contact_peak_count = 0
        self._first_ground_contact_time = None

    def _record_step_contacts(self):
        # Read the contacts produced by each actual 50-us solver step. Counting
        # never applies forces, changes state or stops physical integration.
        contacts = self.mj_data.contact
        count = sum(not contacts[i].exclude and
                    (contacts[i].geom1 == self._ground_geom_id or contacts[i].geom2 == self._ground_geom_id)
                    for i in range(self.mj_data.ncon))
        if count:
            self._ground_contact_substeps += 1
            self._ground_contact_peak_count = max(self._ground_contact_peak_count, int(count))
            if self._first_ground_contact_time is None:
                self._first_ground_contact_time = float(self.mj_data.time)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {'height_mm', 'yaw_rad', 'mode', 'pitch_deg', 'wing_phase', 'initialize_wing_motion', 'retract_legs'}:
            raise ValueError('Unsupported physical reset option')
        mode = options.get('mode', self.mode)
        if mode not in MODES:
            raise ValueError('Unknown body control mode')
        airborne = mode in ('flight', 'landing')
        height = float(options.get('height_mm', 8. if airborne else 1.5))
        yaw = float(options.get('yaw_rad', 0.))
        pitch = float(options.get('pitch_deg', -47.5 if airborne else 0.))
        phase = float(options.get('wing_phase', self.np_random.uniform()))
        initialize_wings = options.get('initialize_wing_motion', airborne)
        retract_legs = options.get('retract_legs', airborne)
        if not np.isfinite(height) or not .2 <= height <= 100 or not np.isfinite(yaw) or not np.isfinite(pitch) or not -90 <= pitch <= 90:
            raise ValueError('Invalid explicit initial body pose')
        if not np.isfinite(phase) or not 0 <= phase < 1 or type(initialize_wings) is not bool or type(retract_legs) is not bool:
            raise ValueError('Wing phase must be in [0,1); initialization switches must be booleans')
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:3] = [0, 0, height / 10]
        self.mj_data.qpos[3:7] = _quat_mul([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
            [np.cos(np.deg2rad(pitch) / 2), 0, np.sin(np.deg2rad(pitch) / 2), 0])
        self.mj_data.qpos[self.wing_qpos] = self._wing_fold
        initial_qpos, initial_qvel = self._wbpg.reset(initial_phase=phase, return_qvel=True)
        if initialize_wings:
            self.mj_data.qpos[self.wing_qpos] = initial_qpos
            self.mj_data.qvel[self.wing_qvel] = initial_qvel
        if retract_legs:
            self.mj_data.qpos[self._leg_qpos] = self.mj_model.qpos_spring[self._leg_qpos]
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._rest_ctrl = np.clip(self.mj_data.actuator_length.copy(), self._control_limit[:, 0], self._control_limit[:, 1])
        self._rest_ctrl[self._adhesion_ids] = 0
        self.mode = mode
        self._step_count = 0
        self._phase = phase * 2 * np.pi
        self._leg_phase = float(self.np_random.uniform(0, 2 * np.pi))
        self._last_action[:] = 0
        self._done = False
        self._reset_done = True
        self._frame = None
        self._clear_step_contacts()
        self._initialization = {'height_mm': height, 'yaw_rad': yaw, 'pitch_deg': pitch,
            'wing_phase': phase, 'initialize_wing_motion': initialize_wings, 'retract_legs': retract_legs,
            'applied_at_episode_reset_only': True, 'body_linear_velocity_mm_s': [0., 0., 0.]}
        return self.observe(), self.task_state()

    def set_mode(self, mode):
        if mode not in MODES:
            raise ValueError('Mode must be walking, flight or landing')
        previous = self.mode
        self.mode = mode
        self._last_mode_change = {'from': previous, 'to': mode, 'simulation_time_s': float(self.mj_data.time),
            'same_model_and_data': True, 'physical_state_reset': False}
        return dict(self._last_mode_change)

    def apply_task(self, task):
        if not isinstance(task, dict) or set(task) - {'mode'}:
            raise ValueError('Physical adapter task accepts mode only; training goals belong to the task wrapper')
        if 'mode' in task:
            self.set_mode(task['mode'])
        return self.task_state()

    def _parse_action(self, action):
        if isinstance(action, dict):
            if set(action) - {'legs', 'wings', 'wing_frequency'}:
                raise ValueError('Action keys are legs[2], wings[6], wing_frequency scalar')
            action = [*action.get('legs', [0, 0]), *action.get('wings', [0] * 6), action.get('wing_frequency', 0)]
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action) or not np.isfinite(action).all():
            raise ValueError('Use nine finite control values in [-1,1]')
        return action

    def _control(self, action):
        control = self._rest_ctrl.copy()
        if self.mode == 'walking':
            self._leg_phase += 2 * np.pi * 12 * .0002
            for index in self._leg_ids:
                name = self.actuator_names[index]
                side = 0 if 'left' in name else 1
                leg = next((number for number in (1, 2, 3) if f'T{number}' in name), 1)
                phase = self._leg_phase + (np.pi if ((leg == 2) ^ (side == 1)) else 0)
                amplitude = .5 * (float(action[side]) + 1)
                if name.startswith('coxa_') and 'twist' not in name and 'abduct' not in name:
                    control[index] += amplitude * .22 * np.sin(phase)
                elif name.startswith('femur_') and 'twist' not in name:
                    control[index] += amplitude * .18 * max(0, np.cos(phase))
                elif name.startswith('tibia_'):
                    control[index] -= amplitude * .2 * max(0, np.cos(phase))
            target = self._wing_fold
        else:
            # The unmodified upstream NumPy generator handles cycle seams and
            # frequency filtering; optional measured data are explicit inputs.
            target = self._wbpg.step(ctrl_freq=218 * (1 + .05 * float(action[8]))) + .25 * action[2:8]
        # General force actuators with official gain18 and physical damping;
        # the position error is an input torque command, never a qpos write.
        control[self.wing_ids] = target - self.mj_data.qpos[self.wing_qpos]
        control = np.clip(control, self._control_limit[:, 0], self._control_limit[:, 1])
        self.mj_data.ctrl[:] = control

    def advance(self, action):
        """Advance the identical physical loop without constructing unused output.

        Fast controllers provide their own observations and rewards. The public
        Gym-style step below still returns the full original body interface.
        """
        if not self._reset_done or self._done:
            raise RuntimeError('Reset this physical episode before stepping')
        action = self._parse_action(action)
        self._clear_step_contacts()
        before_time = float(self.mj_data.time)
        for _ in range(round(self.action_dt / .0002)):
            self._control(action)
            for _ in range(4):
                mujoco.mj_step(self.mj_model, self.mj_data)
                self._record_step_contacts()
            if not np.isfinite(self.mj_data.qpos).all() or not np.isfinite(self.mj_data.qvel).all():
                raise RuntimeError('Physical state became nonfinite')
            if float(self.mj_data.time) < before_time:
                raise RuntimeError('MuJoCo reset time after an unstable step')
        self._last_action = action.copy()
        self._step_count += 1
        self._frame = None
        position, _, _ = self._pose()
        terminated = bool(np.linalg.norm(position[:2]) > 100 or position[2] < -.5 or position[2] > 100)
        truncated = self._step_count >= self.max_episode_steps
        self._done = terminated or truncated
        return terminated, truncated

    def step(self, action):
        terminated, truncated = self.advance(action)
        info = self.task_state()
        info['reward_design'] = 'No task reward in physical adapter; a separate training task must supply one'
        return self.observe(), 0., terminated, truncated, info

    def _pose(self):
        position = self.mj_data.xpos[self.thorax_body_id].copy() * 10
        rotation = self.mj_data.xmat[self.thorax_body_id].reshape(3, 3)
        return position, float(np.arctan2(rotation[1, 0], rotation[0, 0])), float(rotation[2, 2])

    def observe(self):
        pos = self.mj_data.qpos.copy()
        velocity = self.mj_data.qvel.copy()
        for index in range(self.mj_model.njnt):
            kind = int(self.mj_model.jnt_type[index])
            p, v = int(self.mj_model.jnt_qposadr[index]), int(self.mj_model.jnt_dofadr[index])
            if kind == 0:
                pos[p:p + 3] *= 10
                velocity[v:v + 3] *= 10
            elif kind == 2:
                pos[p] *= 10
                velocity[v] *= 10
        return np.r_[pos, velocity, [float(self.mode == name) for name in MODES]].astype(np.float32)

    def task_state(self):
        return {'body_model': 'FlyBody complete official body; active legs and wings', 'mode': self.mode,
            'phase': 'finished' if self._done else 'ready', 'step': self._step_count,
            'simulation_time_s': float(self.mj_data.time), 'body_interval_ms': self.action_dt * 1000,
            'position_world_mm': self._pose()[0].tolist(), 'contact_count': int(self.mj_data.ncon),
            'ground_contact_during_step': self._ground_contact_substeps > 0,
            'ground_contact_substeps': self._ground_contact_substeps,
            'ground_contact_peak_count': self._ground_contact_peak_count,
            'first_ground_contact_substep_end_time_s': self._first_ground_contact_time,
            'ground_contact_trace_scope': 'Active solver contacts involving cyberfly_ground at every physical substep; cleared before each outer step and on reset',
            'wing_actuator_force_cgs': self.mj_data.actuator_force[self.wing_ids].copy().tolist(),
            'last_action': self._last_action.tolist(), 'last_mode_change': self._last_mode_change,
            'reset_initialization': self._initialization, 'controller_version': 'unified_wbpg_v2',
            'wing_pattern_source': self._wing_pattern_source, 'wing_pattern_sha256': self._wing_pattern_sha,
            'controller_trained': False, 'flight_success_claimed': False,
            'controller': 'Engineering leg gait and original upstream wing-pattern generator; no trained flight policy loaded',
            'source_xml_sha256': self._source_xml_sha256,
            'compiled_xml_sha256': self._compiled_xml_sha256,
            'native_units': {'length': 'cm', 'mass': 'g', 'time': 's', 'force': 'dyn', 'torque': 'dyn*cm'},
            'ui_length_unit': 'mm', 'observation_size': int(self.observation_space.shape[0]),
            'physical_dimensions': {'nq': int(self.mj_model.nq), 'nv': int(self.mj_model.nv), 'nu': int(self.mj_model.nu)},
            'action_contract': 'legs2 + wing angle residuals6 + wing-frequency delta1, all [-1,1]',
            'actuator_catalog': self.actuator_names}

    def body_sensory_packet(self, *, rgb=None):
        from shared_io.body_senses import sample_body_senses, json_sha
        packet = sample_body_senses(self, rgb=rgb)
        # The shared sampler reads native model values; convert all named mm
        # fields before publishing a packet. Sensors remain explicit native CGS.
        m = self.mj_model
        for index in range(m.njnt):
            kind = int(m.jnt_type[index]); p = int(m.jnt_qposadr[index]); v = int(m.jnt_dofadr[index])
            width = 3 if kind == 0 else 1 if kind == 2 else 0
            for offset in range(width):
                packet['qpos'][p + offset] *= 10
                packet['qvel'][v + offset] *= 10
                packet['qacc'][v + offset] *= 10
        for index, unit in enumerate(packet['layout']['qvel_units']):
            packet['qfrc_actuator'][index] *= 10 if unit == 'mm/s' else 100
        for index, transmission in enumerate(m.actuator_trntype):
            rotational = int(transmission) == 0 and int(m.jnt_type[m.actuator_trnid[index, 0]]) in (1, 3)
            packet['actuator_force'][index] *= 100 if rotational else 10
            packet['layout']['actuators'][index]['force_units'] = 'g*mm^2/s^2' if rotational else 'g*mm/s^2'
        packet['body_contact_force_world'] = (np.asarray(packet['body_contact_force_world']) * 10).tolist()
        packet['body_contact_torque_at_com_world'] = (np.asarray(packet['body_contact_torque_at_com_world']) * 100).tolist()
        for contact in packet['contacts']:
            contact['position_world_mm'] = (np.asarray(contact['position_world_mm']) * 10).tolist()
            contact['distance_mm'] *= 10
            contact['world_force_on_geom2'] = (np.asarray(contact['world_force_on_geom2']) * 10).tolist()
            contact['world_torque_at_contact_on_geom2'] = (np.asarray(contact['world_torque_at_contact_on_geom2']) * 100).tolist()
            contact['wrench_contact_force_then_torque'] = (np.asarray(contact['wrench_contact_force_then_torque']) * [10, 10, 10, 100, 100, 100]).tolist()
        packet['thorax']['position_world_mm'] = (np.asarray(packet['thorax']['position_world_mm']) * 10).tolist()
        velocity = packet['thorax']['velocity_world_rot_rad_s_then_lin_mm_s']
        velocity[3:] = (np.asarray(velocity[3:]) * 10).tolist()
        layout = packet['layout']
        layout['units'].update(mass='g (official CGS model)', force='g*mm/s^2', torque='g*mm^2/s^2',
            sensordata='Native CGS MuJoCo sensor values by sensor type; not rescaled to mm')
        layout['body_source'] = {'name': 'FlyBody complete legs+wings', 'xml_sha256': self._compiled_xml_sha256,
            'physical_units': 'CGS', 'packet_geometric_units': 'mm', 'position_scale': 10, 'force_scale': 10, 'torque_scale': 100}
        packet['source'] = 'real MuJoCo FlyBody complete walking/flying body'
        packet['model_layout_sha256'] = json_sha(layout)
        packet['interpretation'] += '; complete FlyBody layout differs from NeuroMechFly: existing 758-feature trained weights are incompatible'
        packet.pop('packet_sha256', None)
        packet['packet_sha256'] = json_sha(packet)
        return packet

    def render(self, camera='track3'):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, height=self.height, width=self.width)
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[0:2] = 1
        self._renderer.update_scene(self.mj_data, camera=camera, scene_option=option)
        if self.feedback_renderer is not None:
            self.feedback_renderer(self._renderer.scene, self)
        self._frame = self._renderer.render().copy()
        return self._frame.copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
