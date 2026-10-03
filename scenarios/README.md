# Shared scenario interface

Both built-in environments implement **the same Gymnasium interface** for human
demonstrations, behavior cloning, PPO, a language planner, and real rendering.

```python
from scenarios import create_scenario, list_scenarios

env = create_scenario({
    "name": "doom_basic",
    "env_kwargs": {"frame_skip": 4, "max_episode_steps": 75},
})
obs, info = env.reset(seed=123)
obs, reward, terminated, truncated, info = env.step(2)  # shoot
rgb = env.render()  # uint8 HWC, genuine game renderer
planner_telemetry = env.task_state()
env.close()
```

`create_scenario` accepts only `name`, `env_kwargs`, optional `task`, and optional
`plugins`. Seeds are passed to `reset(seed=...)`, keeping train/evaluation seed
schedules explicit. It never silently substitutes a different environment.

| Scenario | Observation | Action | Policy | Renderer |
| --- | --- | --- | --- | --- |
| `fly_navigation` | 22 float32 values | Box(−1, 1), two descending signals | `MlpPolicy` | MuJoCo, 480×640 RGB |
| `doom_basic` | uint8 RGB `(3,84,84)` | Discrete(3): left strafe, right strafe, shoot | `CnnPolicy` | ViZDoom, 240×320 RGB |

Each environment exposes `scenario_spec` as a JSON-compatible dictionary,
`task_state()`, `apply_task(dict)`, `observe()`, and `render()`. `observe()` is
useful immediately after a language planner changes the task. All actions still
go through `step`, never through planner-generated shell or console commands.

For the fly, `apply_task({"goal_mm": [4, 2]})` changes the target and preserves it
across resets. It does not move the body. Constructor parameters in
`env_kwargs` are those accepted by `rl.env.FlyNavigationEnv`, plus `goal_mm`.
`max_episode_steps` is the episode limit; `episode_steps` is not a constructor
argument.

For Doom, `apply_task({"step_limit": 75, "target_kills": 1})` changes the wrapper
time limit. The official basic map contains one target, so other kill counts are
rejected. Unsupported tasks are rejected instead of appearing to work. The
policy sees only resized RGB; health/ammo/kills are planner/evaluation telemetry,
not inputs secretly added to the image policy. Reward is the official engine's
reward multiplied by `reward_scale` (default 0.01). A time limit is reported as
`truncated`; a kill/death/level ending is `terminated`. After native termination,
ViZDoom provides no new screen, so `render()` retains the last real screen.

Install the optional Doom dependencies after the RL environment:

```bash
~/.cache/cyberfly/rl-env/bin/python -m pip install -r scenarios/requirements.txt
~/.cache/cyberfly/rl-env/bin/python -m unittest scenarios.tests.test_scenarios -v
```

The five integration tests run real engines, verify deterministic pixel resets,
physics movement, task rejection, time-limit semantics, rendering, and custom
registration. Test screenshots are written under `scenarios/test_artifacts`.

## Add an existing Gymnasium environment

For a single process, register a trusted factory directly. An ordinary Gymnasium
environment can use `GymScenarioAdapter`; unsupported dynamic task updates then
raise a clear error. A custom subclass can implement validated updates.

```python
import gymnasium as gym
from scenarios import register_scenario, GymScenarioAdapter

SPEC = {
    "name": "my_cartpole", "observation_kind": "vector",
    "policy_type": "MlpPolicy", "render": "rgb_array",
}

def make_cartpole(env_kwargs):
    env = gym.make("CartPole-v1", render_mode="rgb_array", **env_kwargs)
    return GymScenarioAdapter(env, SPEC)

register_scenario("my_cartpole", make_cartpole, SPEC)
```

For independent training worker processes, package the factory as an installed
entry point so each worker can load it reproducibly. In your own package's
`pyproject.toml`:

```toml
[project.entry-points."cyberfly.scenarios"]
my_environments = "my_environments:register"
```

The entry-point callable accepts the registration function:

```python
def register(register_scenario):
    register_scenario("my_cartpole", make_cartpole, SPEC)
```

Use `{"name": "my_cartpole", "plugins": ["my_environments"]}`. Plugins are
explicitly loaded from installed Python packages. No arbitrary module path or
URL is executed from task text. Duplicate names cannot overwrite a built-in.
Plugins execute Python with the current user's permissions, so install trusted
packages. A plugin must supply its own real renderer; the adapter does not turn
a nonvisual or 2D environment into 3D.

Upstream sources: [ViZDoom](https://github.com/Farama-Foundation/ViZDoom),
[basic scenario configuration](https://github.com/Farama-Foundation/ViZDoom/blob/master/scenarios/basic.cfg),
[DoomGame API](https://vizdoom.farama.org/api/python/doom_game/),
[NeuroMechFly](https://github.com/NeLy-EPFL/flygym). Doom uses the assets bundled
with the official installed wheel; no proprietary game download is required.

## Direct neural-input combinations

The following scenarios make the **actual connectome part of every environment
step**. A PPO/BC action has three values in `[-1,1]`, mapped to normalized currents
`(action+1)/2` at the verified `retina_left`, `retina_right`, and `sugar` ports.
These input neurons are identified against the actual retained MaleCNS graph.
Actions cannot bypass neural integration to directly move the body or shoot.

| Name | Embodiment | Observation | Action reaches |
| --- | --- | --- | --- |
| `fly_neural_link` | MuJoCo anatomical fly | 33 vector values | Real graph → fixed BCI decoder → two CPG body commands |
| `doom_neural_link` | ViZDoom 3D game | 81 vector values | Real graph → fixed BCI decoder → left/right strafe or attack |
| `brain_sandbox` | No body or game | 12 vector values | Real graph; reward targets a chosen population firing rate |

All advertise `scenario_spec['neural_input'] = True`, `embodiment`, and supported
`control_modes = ['manual', 'policy', 'llm_brain']`. Human demonstrations can teach
an input adapter; PPO can continue training that same adapter. `learning=False`
(default) freezes graph weights while the external adapter learns. Optional
`learning=True` enables the upstream graph's separate plasticity machinery;
that is a different learning mechanism and does not make PPO gradients flow
through the graph. Ports and decoders are explicit engineering interfaces, not
claims that the connectome innately understands language or Doom.

```python
import numpy as np

env = create_scenario({
    'name': 'fly_neural_link',
    'env_kwargs': {'max_episode_steps': 150, 'learning': False},
})
obs, info = env.reset(seed=42)
# A real language-model planner may choose these amplitudes.
plan = {'retina_left': 1.0, 'retina_right': 0.0, 'sugar': 0.3}
env.apply_task({'neural_drive': plan})
# apply_task updates semantic context. This explicit step delivers the currents.
action = np.array([2 * plan[k] - 1 for k in ('retina_left','retina_right','sugar')], dtype=np.float32)
obs, reward, terminated, truncated, info = env.step(action)
assert info['neural']['backend'] == 'malecns-v6-real'
env.close()
```

The body/game RGB is really fed to the brain each step. The fly adapter also
observes 22 body/goal state values. The Doom adapter observes an 8×8 downsampled
real grayscale image plus engine telemetry; it is distinct from `doom_basic`'s
CNN that directly selects game actions. Neural-input observations additionally
include semantic drive context and actual neural statistics/readouts.

`env.brain` exposes the owned `ConnectomeClient`. `set_learning(bool)` changes
plasticity, and `queue_aversive()` schedules a deliberately requested event for
the next neural interval. Terminal or frozen-state requests are rejected. This
adapter does not silently convert terminal failure into a punishment that is
then lost during reset. Cancelling a queued event by reset is explicitly counted
in `task_state`; the brain records cancellation of an already-scheduled pulse in
its own event log. `task_state` contains the last true neural summary and compact
port definitions/hashes, not thousands of neuron IDs per frame.

Neural integration is 28.6 ms per control decision. Body/game intervals are
reported separately in state: by default 10 ms for the fly or 4/35 s for Doom.
This timing and the motor decoder are engineering choices. Reward remains the
underlying body's navigation reward or the game's reward; sandbox activity
tracking is explicitly an engineered laboratory objective.

A 32-transition real full-graph PPO adapter test is preserved at
`runs/neural_link_ppo32`: actor parameters changed and the checkpoint reloaded
exactly. That verifies the complete training path, not mastery of navigation.

## Task family and real 3D feedback

The registry now includes `fly_food`, `fly_follow`, `fly_waypoints`, and
`fly_hazard`, plus each name with `_neural`. These use the same articulated
NeuroMechFly and real MuJoCo contacts as navigation. Direct actors output two
CPG descending amplitudes. Neural actors output three currents, which enter the
complete connectome before the engineering motor decoder drives the body.
The original `fly_navigation` 22-state and `fly_neural_link` 33-state policy
identities are preserved. New body tasks have 36 observations; neural task
variants have 47.

```python
from scenarios import create_scenario

env = create_scenario({
    "name": "fly_waypoints_neural",
    "env_kwargs": {"difficulty": 1, "randomization": True,
                   "waypoint_count": 3, "max_episode_steps": 300,
                   "learning": False},
})
observation, info = env.reset(seed=42)
# Neural action: [-1,1]^3 -> retina_left/right/sugar currents [0,1].
observation, reward, terminated, truncated, info = env.step([-.4, -.4, -.8])
frame = env.render()
env.close()
```

`scenario_spec.parameters` contains parameter schemas and
`scenario_spec.curriculum_levels` lists four suggested levels. `difficulty`
and optional `curriculum_level` are integers 0–3; an explicit curriculum level
overrides difficulty. `randomization=False` uses a fixed route; otherwise
`reset(seed)` controls reproducible task generation. Task parameters passed to
`apply_task` are staged, report `requires_reset=True`, and start on the next
reset. A task update never teleports the body. A training coordinator must
explicitly advance curriculum levels using measured evaluation.

| Task | Parameters | Success and failure |
|---|---|---|
| Food | `food_count` 1–8, optional `goal_mm` | Collect the requested number of visible food markers within 0.6 mm; falling fails. |
| Follow | `target_speed_mm_s` 0–10, `follow_steps`, optional `goal_mm` | Remain within the difficulty-dependent radius for consecutive steps; target moves along a bounded scripted path. |
| Waypoints | `waypoint_count` 2–8 or explicit `waypoints_mm` | Reach the route in order; each marker pays a checkpoint reward; final marker completes the task. |
| Danger | `hazard_count` 1–4, `hazards_mm=[[x,y,r],...]`, `hazard_terminates` | Reach the goal while avoiding visible red disks. Entering a disk costs 8; by default it ends the episode as failure. |

Explicit `hazards_mm` can add danger zones to any of the four tasks. Zone
crossing tests the swept thorax XY segment, so a fast step cannot skip through
a disk undetected. These are **non-contact danger regions**, not physical
collision obstacles. Food collection is a task event, not simulated digestion
or detailed feeding biomechanics. All rewards are engineering objectives.
Falling or terminating danger takes precedence over proximity success.

The 36 observations are the original 22 body/goal features, two target-velocity
features, four task indicators, task progress, three nearest-danger features,
following progress, remaining episode-time fraction, difficulty, and configured
target speed. They include privileged simulator state. No visual steering
script moves the fly.

MuJoCo scenes advertise `supports_3d_feedback=True` and expose:

```python
env.show_feedback(kind="aversive", intensity=1.0, duration_ms=800)
# kind also accepts "food" and "reward".
```

This draws actual non-contact 3D electrodes and an arc, or golden food particles,
in the body's camera. Props track the real body, expire by wall time, and never
change `qpos`, velocity, or neural currents. A controller must call the aversive
visual **after confirming actual neural feedback delivery**. The rendering API
alone is not evidence of neural stimulation. Doom scenes do not support these
MuJoCo props. `task_state().feedback_visual` reports the remaining visible time.

## Full-brain adapter training

`full_brain_sandbox` keeps all 166,700 retained neurons and 25,582,938 edges in
the real graph. Its 47 actions control macro-group currents grouped by annotated
superclass and root side; this changes input granularity, not graph coverage.
The annotation catalog additionally exposes 23,163 fine groups and original
neuron IDs for explicit interventions.

The action is a 47-vector in `[-1,1]`, scaled to `[-30,30]` mV-equivalent currents.
The observation is 146-dimensional: 47 measured firing rates / 20, 47 measured
mean membrane voltages / 30, 47 previous actions, three semantic context values,
target rate / 20, and plasticity enabled. Macro group ordering and partition hash
are checked against the live brain before use. The reward tracks per-group
activity against an engineering rate target, weighted by neuron count, with
an input cost; it is not a natural fly preference.

All neural scenes support `set_general_stimulation(list)` and
`apply_task({"general_stimulation": [...]})`. Additional inputs persist until
replaced or cleared and are delivered on the next actual brain step. Formats:

```python
[{"group_id": "m_4a926a3e02c56721", "current_mv": 5.0}]
[{"neuron_ids": ["ACTUAL_ID_FROM_CATALOG"], "current_mv": -5.0}]
[{"macro_group_currents_mv": [0.0] * 47}]
```

Use actual neuron IDs returned by the full-brain API; the string above is only
a placeholder. The worker rejects unknown identities and incorrect vector
lengths. `general_stimulation=[]` clears the additional inputs. Original
three-port context is separate from the full-brain actor's macro-current action.
Every `task_state().group_state` carries measured statistics over all 47 macro
groups, whose counts cover the full retained brain. The old three-port policy
observation shapes remain unchanged.

`runs/full_brain_ppo32/` contains an actual 32-step PPO run: actor parameters
changed, policy reload matched exactly, and paired brain checkpoints restored
with matching memory hashes. Its single four-step heldout evaluation is a
pipeline check, not proof of general mastery. `runs/fly_food_ppo32/` similarly
verifies direct task training; that short task's baseline already succeeded.

## Additional official Doom tasks

`doom_defend_center` and `doom_defend_line`, plus `_neural` variants, load the
installed ViZDoom package's official `defend_the_center` and `defend_the_line`
CFG/WAD assets. Direct actions are turn left, turn right, and shoot. Direct
actors observe real `3×84×84` RGB pixels; neural adapters retain the separate
81-dimensional image-summary/telemetry observation and enter the full brain
before actions reach the game. Difficulty 0–3 maps to Doom skill 1–4.
`target_kills` (default 5) requires that many real kills while alive; death is
failure and the configured decision limit is truncation.

Sources: [official defense configuration](https://github.com/Farama-Foundation/ViZDoom/blob/master/scenarios/defend_the_center.cfg),
[official defense-line configuration](https://github.com/Farama-Foundation/ViZDoom/blob/master/scenarios/defend_the_line.cfg).

Verification artifacts: `runs/task_family_checks/` includes real frames,
terminal-outcome checks and neural Doom checks; `runs/full_brain_scene_checks/`
contains actual full-group stimulation and legacy-observation-shape evidence.
Run the real-engine regression suite with
`python -m unittest discover -s scenarios/tests -v` (nine tests).
