"""Explicit CPU prototype: published 104-input/12-output teacher, complete body.

This module is intentionally separate from the registered nine-action scenes.
Import it directly from scenarios/ in the independent TF CPU environment to
avoid importing unrelated application backends. No default physics is changed.
"""
from __future__ import annotations
from collections import deque
import hashlib
import json
from pathlib import Path
import numpy as np
import mujoco

if __package__:
    from .flybody_physics import FlyBodyPhysics, ROOT, WING_NAMES
else:
    from flybody_physics import FlyBodyPhysics, ROOT, WING_NAMES
from flybody.quaternions import get_dquat_local
from flybody.tasks.task_utils import com2root

ASSETS = ROOT / 'vendor_data/flybody/figshare-25309105-v4'
PATTERN = ASSETS / 'datasets_flight-imitation/wing_pattern_fmech.npy'
DATASET = ASSETS / 'datasets_flight-imitation/flight-dataset_saccade-evasion_augmented.hdf5'
TEACHER = ASSETS / 'trained-fly-policies/flight'
DT = .0002
ACTION_NAMES = ('head_abduct', 'head_twist', 'head', *WING_NAMES, 'abdomen_abduct', 'abdomen', 'user_0')
JOINT_NAMES = ('head_abduct', 'head_twist', 'head', *WING_NAMES, 'abdomen_abduct', 'abdomen',
               *(name for i in range(2, 8) for name in (f'abdomen_abduct_{i}', f'abdomen_{i}')),
               'haltere_left', 'haltere_right')
OBS_SHAPES = {'walker/accelerometer': (3,), 'walker/actuator_activation': (0,),
              'walker/gyro': (3,), 'walker/joints_pos': (25,), 'walker/joints_vel': (25,),
              'walker/velocimeter': (3,), 'walker/world_zaxis': (3,),
              'walker/ref_displacement': (6, 3), 'walker/ref_root_quat': (6, 4)}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays_sha(arrays):
    return {name: hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
            for name, value in arrays.items()}


def verify_asset(path):
    path = Path(path).resolve(strict=True)
    manifest = json.loads((ASSETS / 'manifest.json').read_text())
    expected = {str((Path(a['extracted_directory']) / m['name']).resolve()): m['sha256']
                for a in manifest['files'] for m in a['members']}
    actual = file_sha(path)
    if expected.get(str(path)) != actual:
        raise ValueError(f'Expected an unchanged published asset: {path}')
    return actual


class OfficialFlightTeacher:
    """Original SavedModel mean, CPU only; no optimizer or graph modification."""
    def __init__(self):
        import tensorflow as tf
        import tensorflow_probability as tfp
        from tensorflow.python.framework import type_spec_registry
        if tf.config.list_physical_devices('GPU'):
            raise RuntimeError('Run the teacher prototype with CUDA_VISIBLE_DEVICES empty')
        self.tf = tf
        self.asset_sha = {str(p.relative_to(ASSETS)): verify_asset(p)
                          for p in sorted(TEACHER.rglob('*')) if p.is_file()}
        prototype = tfp.distributions.Independent(tfp.distributions.Normal(0., 1.), 0)
        old_name = 'tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec'
        try:
            type_spec_registry.lookup(old_name)
        except ValueError:
            @type_spec_registry.register(old_name)
            class LegacyIndependentSpec(type(prototype._type_spec)):
                pass
        self.model = tf.saved_model.load(str(TEACHER))
        signature = self.model.__call__.concrete_functions[0].structured_input_signature[0][0]
        if {k: tuple(v.shape[1:]) for k, v in signature.items()} != OBS_SHAPES:
            raise ValueError('Published teacher observation signature changed')
        self.calls = 0

    def __call__(self, observation):
        values = {name: np.asarray(observation[name], np.float32) for name in OBS_SHAPES}
        if any(value.shape != OBS_SHAPES[name] or not np.isfinite(value).all() for name, value in values.items()):
            raise ValueError('Need 104 finite, actually sampled teacher observation values')
        batch = {name: self.tf.convert_to_tensor(value[None]) for name, value in values.items()}
        distribution = self.model(batch)
        output = distribution.mean()[0].numpy()
        np.testing.assert_array_equal(output, distribution.distribution.loc[0].numpy())
        if output.shape != (12,) or not np.isfinite(output).all():
            raise ValueError('Teacher did not produce 12 finite mean controls')
        self.calls += 1
        return output


class FlightReference:
    """Published reference targets, not fabricated sensory observations."""
    def __init__(self, trajectory_index=0):
        import h5py
        self.source_sha256 = verify_asset(DATASET)
        with h5py.File(DATASET, 'r') as data:
            if float(data['timestep_seconds'][()]) != DT:
                raise ValueError('Reference timestep differs from original teacher')
            group = data['trajectories'][f'{int(trajectory_index):03d}']
            com = group['com_qpos'][()].copy()
            velocity = group['com_qvel'][()].copy()
        com[:, :2] -= com[0, :2]
        self.com_qpos, self.qvel = com, velocity
        self.root_qpos = np.concatenate((com2root(com[:, :3], com[:, 3:]), com[:, 3:]), axis=1)
        self.identity = {'kind': 'official measured COM reference transformed with official fixed COM-to-root offset',
                         'trajectory_index': int(trajectory_index), 'source_sha256': self.source_sha256,
                         'array_sha256': arrays_sha({'root_qpos': self.root_qpos, 'qvel': velocity}),
                         'timestep_s': DT, 'future_steps': 5, 'units': 'cm, quaternion wxyz, cm/s and rad/s'}


class TeacherObservationReader:
    """Same named joints, original CGS sensors and four-substep mean semantics."""
    def __init__(self, model, data, *, prefix='', root_qpos_address=0):
        self.model, self.data = model, data
        self.root_address = root_qpos_address
        self.thorax_id = self._id(mujoco.mjtObj.mjOBJ_BODY, prefix+'thorax')
        joint_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, prefix+name) for name in JOINT_NAMES]
        self.qpos_indices = model.jnt_qposadr[joint_ids].copy()
        self.qvel_indices = model.jnt_dofadr[joint_ids].copy()
        self.slices = {}
        for name in ('accelerometer', 'gyro', 'velocimeter'):
            index = self._id(mujoco.mjtObj.mjOBJ_SENSOR, prefix+name)
            if model.sensor_dim[index] != 3:
                raise ValueError('Unexpected original sensor dimension')
            self.slices[name] = slice(int(model.sensor_adr[index]), int(model.sensor_adr[index])+3)
        self.samples = deque(maxlen=4)

    def _id(self, kind, name):
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f'Actual model lacks required original teacher element: {name}')
        return index

    def reset(self):
        self.samples.clear()
        # Composer initializes a four-slot observable buffer with zeros, then
        # inserts one real reset sample. Repeating the sample would incorrectly
        # multiply the teacher's first sensor inputs by four.
        for _ in range(3):
            self.samples.append({name: np.zeros(3) for name in self.slices})
        self.sample_sensors()

    def sample_sensors(self):
        self.samples.append({name: self.data.sensordata[indices].copy() for name, indices in self.slices.items()})

    def read(self, reference_root_qpos, step):
        target = reference_root_qpos[step:step+6]
        if target.shape != (6, 7):
            raise ValueError('Reference ended; never pad missing future targets silently')
        root = self.data.qpos[self.root_address:self.root_address+7]
        result = {f'walker/{name}': np.mean([sample[name] for sample in self.samples], axis=0)
                  for name in self.slices}
        result.update({
            'walker/actuator_activation': np.zeros(0),
            'walker/joints_pos': self.data.qpos[self.qpos_indices].copy(),
            'walker/joints_vel': self.data.qvel[self.qvel_indices].copy(),
            'walker/world_zaxis': self.data.xmat[self.thorax_id].reshape(3, 3)[2].copy(),
            'walker/ref_displacement': (target[:, :3]-root[:3]) @ self.data.xmat[self.thorax_id].reshape(3, 3),
            'walker/ref_root_quat': get_dquat_local(root[3:], target[:, 3:]),
        })
        return {name: np.asarray(value, np.float32) for name, value in result.items()}


class _TeacherBody(FlyBodyPhysics):
    """Subclass-local callbacks leave the main nine-action implementation intact."""
    def __init__(self, **kwargs):
        self.fast_controller = None
        self.sensor_reader = None
        self._needs_observable_forward = False
        super().__init__(**kwargs)

    def _control(self, action):
        if self.fast_controller is not None and self.mode == 'flight':
            self.fast_controller()
            self._needs_observable_forward = True
        else:
            super()._control(action)

    def _record_step_contacts(self):
        super()._record_step_contacts()
        if self.sensor_reader is not None:
            # Like dm_control's original Physics.step, refresh derived position
            # and velocity fields without writing/integrating qpos or qvel.
            if self._needs_observable_forward:
                # The published task moves its reference ghost before every
                # control. Its dirty MJCF sensor binding invokes forward at the
                # first observable read, recomputing acceleration at this state.
                # Reproduce that evaluation stage, without creating a ghost or
                # changing actual body pose/velocity.
                mujoco.mj_forward(self.mj_model, self.mj_data)
                self._needs_observable_forward = False
            else:
                mujoco.mj_step1(self.mj_model, self.mj_data)
            self.sensor_reader.sample_sensors()


class FlightTeacherAdapter:
    """Unregistered 12-control prototype; retains a single complete physical body."""
    def __init__(self, teacher, *, controller='teacher', reference=None, integrator='unified',
                 control_dt=.002, max_episode_steps=1000):
        if controller not in ('teacher', 'zero', 'hold_initial'):
            raise ValueError('controller must be teacher, zero or hold_initial')
        if integrator not in ('unified', 'official_euler'):
            raise ValueError('Explicit integrator must be unified or official_euler')
        self.teacher, self.controller = teacher, controller
        self.reference = reference or FlightReference()
        self.body = _TeacherBody(mode='flight', control_dt=control_dt, max_episode_steps=max_episode_steps,
                                 render_mode='rgb_array', wing_pattern_path=PATTERN)
        if integrator == 'official_euler':
            # Explicit prototype variant before any reset; never edits the main
            # body's default integrator or an already-running physical instance.
            self.body.mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
        m = self.body.mj_model
        self.indices = np.asarray([self.body.actuator_names.index(name) for name in ACTION_NAMES[:-1]])
        self.minimum = np.r_[m.actuator_ctrlrange[self.indices, 0], -1].astype(np.float32)
        self.maximum = np.r_[m.actuator_ctrlrange[self.indices, 1], 1].astype(np.float32)
        self.reader = TeacherObservationReader(m, self.body.mj_data)
        self.body.sensor_reader = self.reader
        self.body.fast_controller = self._control
        self.records = []
        self._held = None
        self.reference_step = 0
        self.identity = {'schema': 1, 'controller_id': 'official_teacher_full_body_12_fast_v1',
                         'controller_condition': controller, 'control_dt_s': DT, 'physics_dt_s': float(m.opt.timestep),
                         'observation_shapes': {k: list(v) for k, v in OBS_SHAPES.items()},
                         'observation_dimension': 104, 'action_dimension': 12, 'action_names': list(ACTION_NAMES),
                         'actuator_indices': self.indices.tolist(), 'joint_names': list(JOINT_NAMES),
                         'physical_dimensions': {'nq': m.nq, 'nv': m.nv, 'nu': m.nu},
                         'integrator_variant': integrator, 'integrator_enum': int(m.opt.integrator),
                         'compiled_xml_sha256': hashlib.sha256(self.body.xml.encode()).hexdigest(),
                         'published_teacher_files': teacher.asset_sha,
                         'pattern_sha256': verify_asset(PATTERN), 'reference': self.reference.identity,
                         'sensor_time_semantics': 'Reset: three zero slots plus one actual sample. Each control first substep uses mj_forward, matching the original ghost-dirty sensor binding; remaining substeps use mj_step1. Actual four samples averaged; qpos/qvel never changed by sensing.',
                         'contact_time_semantics': 'Active contacts recorded after mj_step, before mj_step1 refresh; timestamp is the substep end and may lag the contact computation by one 50-us step',
                         'leg_control': 'Same existing retracted-leg position controls; legs, tendons, actuators, contacts all retained',
                         'output_semantics': '12 canonical means clipped, scaled to original head/abdomen ranges; wing residuals added unscaled to WBPG angle error, gain18/gear1',
                         'experimental_limitations': ['External pretrained teacher, zero new optimization',
                            'Original teacher trained with leg DOFs disabled; this full plant retains them',
                            'Reference COM-to-root offset is the original nominal constant, not a new measured full-body COM model',
                            'Prototype does not use or train the connectome at physical microsteps']}
        self.identity_sha256 = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()

    def reset(self, *, seed=42, initialize_from_reference=True):
        self.body.reset(seed=seed)
        if initialize_from_reference:
            # Only episode initialization may set pose/velocity; steps below
            # exclusively actuate the existing MuJoCo motors.
            self.body.mj_data.qpos[:7] = self.reference.root_qpos[0]
            self.body.mj_data.qvel[:3] = self.reference.qvel[0, :3]
            self.body.mj_data.qvel[3:6] = 0  # Matches published initialization.
            mujoco.mj_forward(self.body.mj_model, self.body.mj_data)
        self.reader.reset()
        self.records.clear(); self._held = None; self.reference_step = 0
        self.initial_arrays = {name: np.asarray(getattr(self.body.mj_data, name)).copy()
                               for name in ('qpos', 'qvel', 'act', 'ctrl')}
        self.initial_phase = self.body._initialization['wing_phase']
        self.initialization = {'source': 'original reference root pose/linear velocity' if initialize_from_reference else 'unchanged complete-body reset',
                               'seed': seed, 'phase': self.initial_phase,
                               'actual_state_sha256': arrays_sha(self.initial_arrays)}
        return self.state()

    def _control(self):
        b, step = self.body, self.reference_step
        before = (float(b.mj_data.time), b.mj_data.qpos.copy(), b.mj_data.qvel.copy())
        observation = self.reader.read(self.reference.root_qpos, step)
        if self.controller == 'zero':
            mean = np.zeros(12, np.float32)
        elif self.controller == 'hold_initial' and self._held is not None:
            mean = self._held.copy()
        else:
            mean = self.teacher(observation)
            if self.controller == 'hold_initial':
                self._held = mean.copy()
        canonical = np.clip(mean, -1, 1)
        action = .5 * (canonical + 1.)
        action *= self.maximum-self.minimum
        action += self.minimum
        target = b._wbpg.step(ctrl_freq=218*(1+.05*float(action[11])))
        requested = action.copy()
        requested[3:9] += target-b.mj_data.qpos[b.wing_qpos]
        ctrl = b._rest_ctrl.copy()
        ctrl[self.indices] = requested[:11]
        b.mj_data.ctrl[:] = ctrl  # Original MuJoCo ctrllimited actuator behavior.
        assert float(b.mj_data.time) == before[0]
        np.testing.assert_array_equal(before[1], b.mj_data.qpos)
        np.testing.assert_array_equal(before[2], b.mj_data.qvel)
        self.records.append({'simulation_time_s': before[0], 'reference_step': step,
            'observation': {name: value.tolist() for name, value in observation.items()},
            'mean_unclipped': mean.tolist(), 'canonical_action': canonical.tolist(),
            'task_action_before_wbpg': action.tolist(), 'wbpg_target_rad': target.tolist(),
            'mujoco_ctrl_requested': ctrl.tolist(),
            'sensor_samples': len(self.reader.samples)})
        self.reference_step += 1

    def advance(self):
        """Keep every physical/sensory update; omit unconsumed state formatting."""
        before = float(self.body.mj_data.time)
        model, data = self.body.mj_model, self.body.mj_data
        self.body.advance(np.zeros(9, np.float32))
        if self.body.mj_model is not model or self.body.mj_data is not data or self.body.mj_data.time <= before:
            raise RuntimeError('Teacher control replaced or reset its physical body')
        if np.any(self.body.mj_data.xfrc_applied):
            raise RuntimeError('Unexpected external body force in teacher prototype')

    def step(self):
        self.advance()
        return self.state()

    def state(self):
        b, d = self.body, self.body.mj_data
        step = min(self.reference_step, len(self.reference.root_qpos)-1)
        thorax = d.xpos[b.thorax_body_id].copy()
        com = d.subtree_com[b.thorax_body_id].copy()
        up = mujoco.mj_name2id(b.mj_model, mujoco.mjtObj.mjOBJ_SITE, 'hover_up_dir')
        return {'simulation_time_s': float(d.time), 'controller_id': self.identity_sha256,
                'controller_condition': self.controller, 'control_calls': self.reference_step,
                'thorax_position_mm': (thorax*10).tolist(), 'actual_com_mm': (com*10).tolist(),
                'root_reference_error_mm': float(np.linalg.norm(d.qpos[:3]-self.reference.root_qpos[step, :3])*10),
                'actual_com_reference_error_mm': float(np.linalg.norm(com-self.reference.com_qpos[step, :3])*10),
                'hover_up_cosine': float(d.site_xmat[up].reshape(3, 3)[2, 2]),
                'ground_contact_during_step': b._ground_contact_substeps > 0,
                'ground_contact_substeps': b._ground_contact_substeps,
                'first_ground_contact_substep_end_time_s': b._first_ground_contact_time,
                'wing_actuator_force_cgs': d.actuator_force[b.wing_ids].tolist(),
                'force_applied_max': float(np.abs(d.xfrc_applied).max()),
                'qpos': d.qpos.tolist(), 'qvel': d.qvel.tolist()}

    def close(self):
        self.body.close()
