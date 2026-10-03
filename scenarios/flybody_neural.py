"""Full MaleCNS inputs to the same nine-command FlyBody in all body modes."""
from copy import deepcopy

from gymnasium import spaces
import numpy as np

from connectome_adapter.ports import validate_neural_drive
from .neural_inputs import validate_general_stimulation
from .neural_link import FlyNeuralLink
from .flybody_specs import flybody_spec, BODY_ACTION_LABELS


class FlyBodyNeuralLink(FlyNeuralLink):
    SCENARIO_SPEC = flybody_spec(neural=True)

    def __init__(self, **kwargs):
        if kwargs.get('motor_readout_checkpoint') is not None:
            raise ValueError('Unified FlyBody cannot load an old two-output motor checkpoint; select its separate body policy or the disclosed rule BCI.')
        super().__init__(**kwargs)
        # The new physical layout is independent of all old 33-value policies.
        self.observation_space = spaces.Box(-np.inf, np.inf,
            tuple(self.scenario_spec['observation_shape']), dtype=np.float32)

    def _make_body(self, max_episode_steps, render_mode, kwargs):
        from .flybody_tasks import FlyBodyTaskScenario
        return FlyBodyTaskScenario(max_episode_steps=max_episode_steps, render_mode=render_mode, **kwargs)

    def _zero_body_action(self):
        return np.zeros(9, np.float32)

    def _decode_brain_to_body(self, summary):
        native = summary['native_action']
        turn = float(np.clip(float(native['turn']) / 6., -1., 1.))
        forward = float(np.clip(float(native['forward']) / 10. - 1., -1., 1.))
        legs = self.decode_body_action(native)
        # Small engineering residuals around the existing low-level wingbeat
        # pattern. No semantic/model current is routed directly to this action.
        wing = np.array([.15 * turn, .10 * forward, .10 * forward,
                         -.15 * turn, .10 * forward, .10 * forward, .20 * forward], np.float32)
        action = np.r_[legs, wing].astype(np.float32)
        self._motor_trace = {
            'mode': 'unified_rule_bci', 'trainable': False, 'optimizer_updates': 0,
            'body_mode': self.body.mode, 'body_action': action.tolist(),
            'action_labels': deepcopy(BODY_ACTION_LABELS),
            'source': 'Actual completed full-brain native_action readouts only',
            'native_action': deepcopy(native),
            'neural_simulation_ms': summary['neural']['simulation_ms'],
            'input_counts_sha256': summary['neural']['counts_sha256'],
            'normalized_forward': forward, 'normalized_turn': turn,
            'normalization': 'turn=clip(native.turn/6); forward=clip(native.forward/10-1)',
            'leg_mapping': '[clip(forward-turn), clip(forward+turn)]',
            'wing_mapping': '[.15*turn,.10*forward,.10*forward,-.15*turn,.10*forward,.10*forward,.20*forward]',
            'residual_angle_scale_rad': .25, 'frequency_base_hz': 218., 'frequency_relative_scale': .05,
            'maximum_bci_wing_angle_residual_rad': .0375, 'maximum_bci_frequency_delta_hz': 2.18,
            'physical_control': 'Actual joint torque controllers; no body pose, velocity or lift force is set',
            'limitation': 'Untrained engineering output mapping and fixed low-level WBPG; anatomical muscle mapping is not established',
        }
        summary['motor_readout'] = deepcopy(self._motor_trace)
        return action

    def set_motor_readout(self, checkpoint):
        if checkpoint is not None:
            raise ValueError('Unified FlyBody requires its own nine-output readout; old two-command checkpoints cannot drive these wings. Train a dedicated body policy first.')
        self.motor_readout, self._motor_trace = None, None
        return {'mode': 'unified_rule_bci', 'trainable': False, 'body_action_shape': [9]}

    def set_mode(self, mode):
        change = self.body.set_mode(mode)
        self._body_info = self.body.task_state()
        return {**change, 'brain_reset': False, 'brain_learning': self.learning}

    def apply_task(self, task):
        allowed = {'neural_drive', 'general_stimulation', 'mode', 'target_altitude_mm', 'goal_mm'}
        if not isinstance(task, dict) or not task or set(task) - allowed:
            raise ValueError('Unified neural task supports sensory inputs, mode, altitude and walking goal.')
        # Validate every request before mutating either the body or neural input.
        context = validate_neural_drive(task['neural_drive']) if 'neural_drive' in task else self.semantic_drive
        general = validate_general_stimulation(task['general_stimulation']) if 'general_stimulation' in task else self.general_stimulation
        body_task = {key: value for key, value in task.items() if key in {'mode', 'target_altitude_mm', 'goal_mm'}}
        body_result = self.body.apply_task(body_task) if body_task else None
        self.semantic_drive, self.general_stimulation = deepcopy(context), deepcopy(general)
        self._body_info = self.body.task_state()
        return {'applied': deepcopy(task), 'body': body_result, 'brain_reset': False,
                'brain_learning': self.learning, 'physical_state_reset': False,
                'sensory_effect': 'Original retinal vision and other inputs remain; changed inputs apply at the next real neural interval'}

    def task_state(self):
        value = super().task_state()
        value.update(body_mode=self.body.mode,
            motor_readout=deepcopy(self._motor_trace) if self._motor_trace else {'mode': 'unified_rule_bci', 'trainable': False, 'body_action_shape': [9]},
            decoder_note='Actual neural turn/forward modulate legs, wing residuals and wingbeat frequency through the disclosed untrained engineering BCI',
            controller_timing={'brain_interval_ms': 28.6, 'body_outer_interval_ms': self.body.unwrapped.action_dt * 1000,
                'wing_control_interval_ms': .2, 'physics_interval_ms': .05,
                'full_brain_calls_per_body_outer_step': 1,
                'strict_neural_body_clock_synchronization_claimed': False})
        return value
