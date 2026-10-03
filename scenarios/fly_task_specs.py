"""Dependency-light task catalog and parameter schemas."""
from copy import deepcopy

TASK_NAMES = {"food": "果蝇寻食", "follow": "追随移动目标", "waypoints": "连续路标", "hazard": "绕开危险区"}
PARAMETERS = {
    "task_type": {"type": "string", "enum": list(TASK_NAMES), "default": "food", "description": "任务类型"},
    "difficulty": {"type": "integer", "minimum": 0, "maximum": 3, "default": 0, "description": "难度等级"},
    "curriculum_level": {"type": "integer", "minimum": 0, "maximum": 3, "description": "课程等级；提供时覆盖 difficulty"},
    "randomization": {"type": "boolean", "default": True, "description": "每回合随机目标与路线，受 reset(seed) 控制"},
    "max_episode_steps": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 300},
    "food_count": {"type": "integer", "minimum": 1, "maximum": 8, "default": 1},
    "waypoint_count": {"type": "integer", "minimum": 2, "maximum": 8, "default": 3},
    "target_speed_mm_s": {"type": "number", "minimum": 0, "maximum": 10, "default": 0.8},
    "follow_steps": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 40},
    "hazard_count": {"type": "integer", "minimum": 1, "maximum": 4, "default": 1},
    "hazard_terminates": {"type": "boolean", "default": True},
    "goal_mm": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "明确目标坐标，毫米"},
    "waypoints_mm": {"type": "array", "minItems": 2, "maxItems": 8, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}},
    "hazards_mm": {"type": "array", "minItems": 1, "maxItems": 4, "description": "每项 [x_mm,y_mm,radius_mm]", "items": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}},
}
CURRICULUM_LEVELS = [
    {"level": 0, "title": "入门", "goal_distance_mm": [0.9, 1.5], "goal_angle_rad": 0.12, "follow_radius_mm": 1.6},
    {"level": 1, "title": "基础", "goal_distance_mm": [2, 4], "goal_angle_rad": 0.45, "follow_radius_mm": 1.3},
    {"level": 2, "title": "进阶", "goal_distance_mm": [3, 6], "goal_angle_rad": 0.9, "follow_radius_mm": 1.0},
    {"level": 3, "title": "挑战", "goal_distance_mm": [5, 9], "goal_angle_rad": 1.4, "follow_radius_mm": 0.8},
]

def task_spec(name, task_type, *, neural=False):
    return {"name": name, "title": ("神经直连 · " if neural else "") + TASK_NAMES[task_type],
        "observation_kind": "vector", "observation_shape": [47 if neural else 36], "policy_type": "MlpPolicy",
        "action_kind": "continuous", "action_shape": [3 if neural else 2],
        "action_labels": ["retina_left", "retina_right", "sugar"] if neural else ["left_descending", "right_descending"],
        "render": "rgb_array", "render_shape": [480, 640, 3], "embodiment": "physical_3d",
        "neural_input": neural, "supports_demonstration": True,
        "supports_3d_feedback": True,
        "control_modes": ["manual", "policy", "llm_brain"] if neural else ["manual", "policy", "connectome"],
        "engine": "Real full connectome → engineered decoder → MuJoCo" if neural else "MuJoCo / official NeuroMechFly",
        "task_type": task_type, "parameters": {**deepcopy(PARAMETERS), "task_type": {**PARAMETERS["task_type"], "default": task_type}},
        "curriculum_levels": deepcopy(CURRICULUM_LEVELS),
        "task_updates": {"task": "Task parameters take effect on next reset", **({"neural_drive": "Normalized retina_left/right/sugar context"} if neural else {})},
        "limitations": ["Goals, target velocities and hazard geometry are privileged simulator state",
            "Food collection is proximity-based; no digestive or feeding biomechanics are simulated",
            "Danger zones are visible non-contact geometry with explicit crossing penalties; they are not collision obstacles",
            "Moving target is a scripted 3D marker; the fly always moves through MuJoCo contacts",
            "Rewards and the optional neural sensory/motor interfaces are engineered task definitions"],
        "reward_definition": {"progress": "5 × distance reduction excluding target self-motion", "time": -0.01,
            "food_or_waypoint": 3, "completion": 10, "fall": -5, "danger_entry": -8,
            "following": "0.05 when within following radius, -0.02 × capped distance otherwise"}}
