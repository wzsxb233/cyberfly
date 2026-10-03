"""Serializable contracts for the new unified FlyBody; imports start no simulator."""
from copy import deepcopy

BODY_ACTION_LABELS = [
    'leg_left', 'leg_right', 'wing_left_yaw_residual', 'wing_left_roll_residual',
    'wing_left_pitch_residual', 'wing_right_yaw_residual', 'wing_right_roll_residual',
    'wing_right_pitch_residual', 'wing_frequency_delta',
]
BODY_OBSERVATION_DIM = 224


def flybody_spec(*, neural=False):
    name = 'flybody_neural_link' if neural else 'flybody_unified'
    return {
        'name': name,
        'title': '统一飞行果蝇 · 全脑直连' if neural else '统一飞行果蝇 · 身体训练',
        'observation_kind': 'vector',
        'observation_shape': [BODY_OBSERVATION_DIM + (11 if neural else 0)],
        'policy_type': 'MlpPolicy', 'action_kind': 'continuous',
        'action_shape': [3 if neural else 9],
        'action_labels': ['retina_left', 'retina_right', 'sugar'] if neural else deepcopy(BODY_ACTION_LABELS),
        'body_action_shape': [9], 'body_action_labels': deepcopy(BODY_ACTION_LABELS),
        'render': 'rgb_array', 'render_shape': [480, 640, 3],
        'neural_input': neural, 'embodiment': 'physical_3d',
        'control_modes': ['manual', 'policy', 'llm_brain'] if neural else ['manual', 'policy'],
        'engine': ('Real full MaleCNS → explicit engineering BCI → ' if neural else '') + 'complete FlyBody MuJoCo legs/wings',
        'supports_demonstration': True, 'supports_3d_feedback': True,
        'body_modes': ['walking', 'flight', 'landing'],
        'task_updates': {'mode': 'walking, flight, landing; same body and brain, no reset',
                         'target_altitude_mm': 'Hover target height, 1..50 mm',
                         'goal_mm': 'Walking goal XY, each within ±100 mm',
                         **({'neural_drive': 'retina_left, retina_right and sugar: normalized [0,1] context',
                             'general_stimulation': 'Persistent annotated-group/neuron-ID currents; one-interval model files are appended without replacing original senses'} if neural else {})},
        'parameter_schema': {
            'mode': {'type': 'enum', 'values': ['walking', 'flight', 'landing'], 'default': 'flight'},
            'target_altitude_mm': {'type': 'number', 'minimum': 1, 'maximum': 50, 'default': 3},
            'max_episode_steps': {'type': 'integer', 'minimum': 1, 'maximum': 100000, 'default': 150},
        },
        'limitations': [
            'Leg CPG, wingbeat pattern generator, sensory encoding and BCI mapping are engineering interfaces',
            'Full original body has one model/data instance across modes; wings do not imply a trained flying policy',
            'Separate body layout and policy shape: old NeuroMechFly checkpoints cannot be reused',
            'Hover/navigation/landing rewards use privileged actual physical state, not biological reward measurements',
        ] + (['PPO trains neural inputs; full-brain plasticity follows the explicit learning flag',
              'Default rule BCI is untrained; old two-output learned motor readout is rejected'] if neural else []),
    }
