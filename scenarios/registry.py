from __future__ import annotations

from copy import deepcopy
from importlib.metadata import entry_points
import re
from typing import Callable

import gymnasium as gym


FLY_SPEC = {
    "name": "fly_navigation", "title": "果蝇三维导航",
    "observation_kind": "vector", "observation_shape": [22], "policy_type": "MlpPolicy",
    "action_kind": "continuous", "action_shape": [2],
    "action_labels": ["left_descending", "right_descending"],
    "render": "rgb_array", "render_shape": [480, 640, 3],
    "engine": "MuJoCo / official NeuroMechFly", "supports_demonstration": True,
    "neural_input": False, "embodiment": "physical_3d", "control_modes": ["manual", "policy", "connectome"],
    "supports_3d_feedback": True,
    "task_updates": {"goal_mm": "two coordinates in [-100,100] mm"},
    "limitations": ["Uses simulator goal coordinates, not retinal vision", "Official CPG is fixed; PPO learns descending control", "Not connectome learning"],
}
DOOM_SPEC = {
    "name": "doom_basic", "title": "Doom 三维瞄准射击",
    "observation_kind": "image", "observation_shape": [3, 84, 84], "policy_type": "CnnPolicy",
    "action_kind": "discrete", "action_count": 3,
    "action_labels": ["move_left", "move_right", "attack"],
    "render": "rgb_array", "render_shape": [240, 320, 3],
    "engine": "ViZDoom / official basic.cfg and basic.wad", "supports_demonstration": True,
    "neural_input": False, "embodiment": "game", "control_modes": ["manual", "policy", "connectome"],
    "task_updates": {"step_limit": "integer from 1 to 1000", "target_kills": "only 1 (the built-in map has one target)"},
    "limitations": ["A single-target basic scenario, not the full Doom game", "Policy sees RGB pixels; planner telemetry is privileged engine state"],
}


def _fly_factory(kwargs):
    from .fly_navigation import FlyNavigationScenario
    return FlyNavigationScenario(**kwargs)


def _doom_factory(kwargs):
    from .doom_basic import DoomBasicScenario
    return DoomBasicScenario(**kwargs)


def _neural_fly_factory(kwargs):
    from .neural_link import FlyNeuralLink
    return FlyNeuralLink(**kwargs)


def _brain_sandbox_factory(kwargs):
    from .brain_sandbox import BrainSandboxScenario
    return BrainSandboxScenario(**kwargs)


def _neural_doom_factory(kwargs):
    from .doom_neural_link import DoomNeuralLink
    return DoomNeuralLink(**kwargs)


def _full_brain_factory(kwargs):
    from .full_brain_sandbox import FullBrainSandboxScenario
    return FullBrainSandboxScenario(**kwargs)


def _unified_flybody_factory(kwargs, *, neural):
    if neural:
        from .flybody_neural import FlyBodyNeuralLink
        return FlyBodyNeuralLink(**kwargs)
    from .flybody_tasks import FlyBodyTaskScenario
    return FlyBodyTaskScenario(**kwargs)


_REGISTRY = {
    "fly_navigation": (_fly_factory, FLY_SPEC),
    "doom_basic": (_doom_factory, DOOM_SPEC),
    "fly_neural_link": (_neural_fly_factory, {
        "name": "fly_neural_link", "title": "神经直连 · 三维果蝇",
        "observation_kind": "vector", "observation_shape": [33], "policy_type": "MlpPolicy",
        "action_kind": "continuous", "action_shape": [3], "action_labels": ["retina_left", "retina_right", "sugar"],
        "render": "rgb_array", "render_shape": [480, 640, 3], "neural_input": True,
        "embodiment": "physical_3d", "control_modes": ["manual", "policy", "llm_brain"],
        "engine": "Real full MaleCNS graph → engineered decoder → MuJoCo fly", "supports_demonstration": True,
        "task_updates": {"neural_drive": "retina_left/right and sugar, normalized 0..1"},
        "limitations": ["PPO trains neural-input adapter; graph plasticity is a separate mode", "Sensory and motor interfaces are engineering approximations"],
    }),
    "brain_sandbox": (_brain_sandbox_factory, {
        "name": "brain_sandbox", "title": "全脑沙盒 · 无身体",
        "observation_kind": "vector", "observation_shape": [12], "policy_type": "MlpPolicy",
        "action_kind": "continuous", "action_shape": [3], "action_labels": ["retina_left", "retina_right", "sugar"],
        "render": "rgb_array", "render_shape": [480, 640, 3], "neural_input": True,
        "embodiment": "none", "control_modes": ["manual", "policy", "llm_brain"],
        "engine": "Real full MaleCNS neural graph; no body/game", "supports_demonstration": True,
        "task_updates": {"neural_drive": "retina_left/right and sugar, normalized 0..1", "target_rate_hz": "engineered target firing activity"},
        "limitations": ["Activity-tracking reward is an engineering objective", "Rendered activity chart is not a simulated 3D body", "No biological equivalence claim"],
    }),
    "doom_neural_link": (_neural_doom_factory, {
        "name": "doom_neural_link", "title": "神经直连 · Doom",
        "observation_kind": "vector", "observation_shape": [81], "policy_type": "MlpPolicy",
        "action_kind": "continuous", "action_shape": [3], "action_labels": ["retina_left", "retina_right", "sugar"],
        "render": "rgb_array", "render_shape": [240, 320, 3], "neural_input": True,
        "embodiment": "game", "control_modes": ["manual", "policy", "llm_brain"],
        "engine": "Real full MaleCNS graph → engineered decoder → ViZDoom basic", "supports_demonstration": True,
        "task_updates": {"neural_drive": "retina_left/right and sugar, normalized 0..1"},
        "limitations": ["PPO trains neural-input adapter; graph plasticity is a separate mode", "Adapter sees 8x8 real grayscale plus game telemetry; brain receives full real RGB"],
    }),
}
_LOADED_PLUGINS = set()


def _fly_task_factory(kwargs, *, task_type, neural, name):
    if neural:
        from .fly_tasks_neural import FlyTaskNeuralScenario
        cls = FlyTaskNeuralScenario
    else:
        from .fly_tasks import FlyTaskScenario
        cls = FlyTaskScenario
    return cls(**{"task_type": task_type, **kwargs, "scenario_name": name})


from functools import partial
from .fly_task_specs import TASK_NAMES, task_spec
for _task_type in TASK_NAMES:
    for _neural in (False, True):
        _name = f"fly_{_task_type}" + ("_neural" if _neural else "")
        _REGISTRY[_name] = (partial(_fly_task_factory, task_type=_task_type, neural=_neural, name=_name),
                            task_spec(_name, _task_type, neural=_neural))

from .full_brain_sandbox import full_brain_spec
_REGISTRY["full_brain_sandbox"] = (_full_brain_factory, full_brain_spec())


def _doom_defense_factory(kwargs, *, layout, neural):
    from .doom_defense import DoomDefenseScenario, DoomDefenseNeuralScenario
    cls = DoomDefenseNeuralScenario if neural else DoomDefenseScenario
    return cls(**{"layout": layout, **kwargs})


from .doom_defense import defense_spec
for _layout in ("center", "line"):
    for _neural in (False, True):
        _spec = defense_spec(_layout, neural=_neural)
        _REGISTRY[_spec["name"]] = (partial(_doom_defense_factory, layout=_layout, neural=_neural), _spec)

for _, _spec in _REGISTRY.values():
    _spec["supports_3d_feedback"] = _spec.get("embodiment") == "physical_3d"
    if _spec.get("neural_input"):
        _spec.setdefault("task_updates", {})["general_stimulation"] = "Persistent bounded currents to annotated groups or original neuron IDs; next real neural step"

# Separate spaces/layout identities: never reinterpret an existing fly policy.
from .flybody_specs import flybody_spec
for _neural in (False, True):
    _spec = flybody_spec(neural=_neural)
    _REGISTRY[_spec['name']] = (partial(_unified_flybody_factory, neural=_neural), _spec)


def register_scenario(name: str, factory: Callable[[dict], gym.Env], spec: dict):
    """Register a trusted Python factory accepting one env_kwargs dictionary.

    Factories return gym.Env objects with scenario_spec, task_state(), and
    apply_task(dict). Duplicate registration never overwrites existing entries.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name):
        raise ValueError("Scenario names must be 2-64 lowercase letters/digits/underscores.")
    if name in _REGISTRY:
        raise ValueError(f"Scenario already registered: {name}")
    if not callable(factory) or not isinstance(spec, dict):
        raise TypeError("A callable factory and a spec dictionary are required.")
    if spec.get("name") != name:
        raise ValueError("spec.name must match the registry name.")
    for field in ("observation_kind", "policy_type", "render"):
        if field not in spec:
            raise ValueError(f"Scenario spec is missing {field}.")
    _REGISTRY[name] = (factory, deepcopy(spec))


def load_plugins(names: list[str]):
    """Explicitly load installed entry points in group cyberfly.scenarios.

    An entry point is a callable taking register_scenario. This executes trusted
    installed Python code. Nothing is imported from natural-language task text,
    arbitrary paths, URLs, or config-supplied module names.
    """
    if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
        raise TypeError("plugins must be a list of installed entry-point names.")
    available = {}
    for ep in entry_points(group="cyberfly.scenarios"):
        if ep.name in available:
            raise ValueError(f"Ambiguous duplicate plugin entry point: {ep.name}")
        available[ep.name] = ep
    for name in names:
        if name in _LOADED_PLUGINS:
            continue
        if name not in available:
            raise ValueError(f"Installed scenario plugin not found: {name}")
        before = set(_REGISTRY)
        try:
            available[name].load()(register_scenario)
        except BaseException:
            for added in set(_REGISTRY) - before:
                del _REGISTRY[added]
            raise
        _LOADED_PLUGINS.add(name)


def list_scenarios():
    """List serializable specs without initializing a simulator or loading plugins."""
    return [deepcopy(spec) for _, spec in _REGISTRY.values()]


def create_scenario(config: dict) -> gym.Env:
    """Build {'name': str, 'env_kwargs': dict, 'task': dict?, 'plugins': list?}.

    The caller must reset() before step()/render(). Seeds belong to reset(seed=)
    so training and evaluation control their own reproducible seed schedules.
    """
    if not isinstance(config, dict):
        raise TypeError("Scenario config must be a dictionary.")
    unknown = set(config) - {"name", "env_kwargs", "task", "plugins"}
    if unknown:
        raise ValueError(f"Unknown scenario configuration keys: {sorted(unknown)}")
    load_plugins(config.get("plugins", []))
    name = config.get("name")
    if name not in _REGISTRY:
        raise ValueError(f"Unknown scenario {name!r}; available: {sorted(_REGISTRY)}")
    kwargs = config.get("env_kwargs", {})
    if not isinstance(kwargs, dict):
        raise TypeError("env_kwargs must be a dictionary.")
    factory, spec = _REGISTRY[name]
    env = factory(deepcopy(kwargs))
    try:
        if not isinstance(env, gym.Env):
            raise TypeError("A scenario factory must return gymnasium.Env.")
        if not isinstance(getattr(env, "scenario_spec", None), dict):
            raise TypeError("Scenario environment must expose scenario_spec as a dictionary.")
        for method in ("task_state", "apply_task"):
            if not callable(getattr(env, method, None)):
                raise TypeError(f"Scenario environment must implement {method}().")
        if env.scenario_spec.get("name") != name:
            raise ValueError("Environment spec name does not match its registered name.")
        if config.get("task") is not None:
            env.apply_task(deepcopy(config["task"]))
        return env
    except BaseException:
        if isinstance(env, gym.Env):
            env.close()
        raise
