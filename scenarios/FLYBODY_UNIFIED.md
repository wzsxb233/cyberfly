# One FlyBody for walking, flight and landing

The new `flybody_unified` and `flybody_neural_link` scenarios use the same complete
official FlyBody MuJoCo model. The original 109 position coordinates, 108 velocity
coordinates, 78 actuators, legs and wings remain present. `apply_task({"mode": ...})`
switches the low-level controller without recreating the model, resetting the
body, changing its pose, or resetting its brain.

```python
from scenarios import create_scenario

scene = create_scenario({
    "name": "flybody_neural_link",
    "env_kwargs": {"mode": "flight", "learning": False, "max_episode_steps": 150},
})
try:
    observation, state = scene.reset(seed=42)
    # Only reset(options={"height_mm": 8}) may set an explicit episode initial pose.
    # Mode changes never lift or teleport the body.
    observation, reward, terminated, truncated, state = scene.step(
        scene.shared_passthrough_action()
    )
    scene.apply_task({"mode": "landing"})
    image = scene.render()
finally:
    scene.close()
```

`flybody_unified` exposes nine body actions for PPO/A2C/SAC/TD3 or demonstrations.
All are bounded to `[-1,1]`: two leg gait drives, six wing-angle residuals in
left yaw/roll/pitch then right yaw/roll/pitch order, and one wing-frequency delta.
The fixed low-level wing generator runs near 218 Hz; body actions allow ±0.25 rad
wing residuals and ±5% frequency changes. Those commands enter actual joint torque
controllers and the MuJoCo fluid model. No artificial lift force is applied.

The current `unified_wbpg_v2` baseline uses the official NumPy wingbeat generator
with cycle interpolation and frequency smoothing. Flight/landing reset begins at
8 mm, with the official −47.5° thorax pitch, moving-wing initial coordinates and
retracted legs. These are episode initialization choices, never changes during a
mode switch. The untrained zero-residual baseline still touches down after about
43.2 ms in the measured test; it has not learned to hover.

`flybody_neural_link` exposes the same three neural-input actions as the earlier
neural scenarios: left retina, right retina, and sugar. Existing visual inputs,
persistent group currents and explicit one-interval model additions are preserved.
The actual full 166,700-cell graph advances first. Its measured BCI forward/turn
readouts drive the legs and small wing residuals through the explicit untrained
mapping in `info.motor_readout`. The maximum BCI wing-angle residual is 0.0375 rad
and the maximum frequency change is 2.18 Hz. This is an engineering connection,
not a reconstructed biological motor-neuron-to-muscle map. Old two-output motor
checkpoints are rejected before a new environment starts.

The body advances 10 ms per outer action by default. It executes 50 wing-controller
updates of 200 µs, each with four 50 µs physics steps. A neural scenario calls the
full brain once per outer body action for 28.6 ms. These are explicitly different
simulation clocks; no strict biological time synchronization is claimed.

The task wrapper provides measured walking-goal, hover-height, and landing rewards.
They combine change in goal error, target proximity, uprightness, motion/control
costs and explicit success/failure terms. Hover success requires at least 500 ms
within 0.5 mm of target altitude, upright, with no floor contact and speed at most
20 mm/s. Landing requires at least 100 ms of floor contact, height at most 1.2 mm,
upright and speed at most 20 mm/s. Failure takes priority over success. These are
privileged simulator objectives, separate from neural plasticity or human PPL
feedback. Switching modes never automatically applies a neural punishment.
Flight uprightness uses the actual `hover_up_dir` site, respecting the intended
flight pitch. Landing/walking use the thorax reference. The common evaluator reads
the explicit `is_success` field; cumulative and per-step rewards are both exposed.

Already delivered neural aversion is visible as actual non-contact 3D electrodes
and arcs in the simulator camera. Food/reward markers also use real scene geometry.
Only the existing session's confirmed positive `delivered_ms_this_step` triggers
the aversion marker; rendering cannot inject a pulse or claim neural learning.

Body-policy observations have 224 values: normalized 109 qpos, 108 qvel, three mode
flags, walking goal XY, hover target height, and accumulated stable duration. Neural
observations have 235 values, adding three sensory context values and eight measured
brain summaries. Goal changes therefore remain visible to a policy. Existing
22/33-value navigation policies cannot be loaded into these spaces.

The full body-sensory packet has a distinct **922-feature** layout, including
actual contacts, forces, sensors and body orientation with CGS-to-mm conversion.
Existing 758-feature NeuroMechFly heads are incompatible. New body-to-sensory-cell
and body-to-model-embedding heads were initialized and saved with **zero training
steps**, then loaded and compared exactly. Their new layout is
`2829eb7ecba8f8067a4e443c4c7e43295939304d6bd97169b545f1c02e1f9050`.
No original MiniCPM or old interface weight was changed.

Actual integration evidence: `artifacts/unified_flybody_checks/58e8a88b41ea`.
Five body steps and three full-brain/body steps passed; all owned processes closed.
The three neural intervals produced 13,757 / 15,997 / 16,606 actual spikes and
nine executed body actions while retaining the real camera and sugar input.
The test explicitly began at an 8 mm episode height and then descended; its
in-air screenshots **do not establish learned flight or successful landing**.
No model inference, GPU training, or active Lab mutation occurred in this test.

This first integration archive predates the improved WBPG reset. The current
physics revision and reward interface were separately checked through actual
CPU PPO construction, two physical prediction steps and exact save/reload in
`artifacts/unified_training_compat/f301ab0618c2`. Optimizer updates were zero.
Its `request-not-executed.json` is an immediately usable `/api/train` request for
the separate nine-dimensional flight policy: PPO, 8,192 steps, two CPU environments,
128-step rollouts, seed 42, five fixed evaluation episodes and no old checkpoint.
That request has been prepared, **not launched**.

The browser defaults new, empty sessions to the unified neural body. Existing
active sessions remain selected; reloading the page never creates a body. Select
the body mode and hover target before loading or training. The explicit “应用到当前
身体 · 不重置” button invokes `/api/body/task`, without calling a language model.
The direct body scene shows all nine controls. The neural scene keeps three sensory
inputs, disables the incompatible old two-output motor form, and sends no old
motor checkpoint. Fleet copies motor state only when the source scenario matches.

End-to-end isolated API/browser verification with the frozen `unified_wbpg_v2`
physics is in `artifacts/unified_body_ui/5b8111a69d61`. The mode endpoint retained
actual time, pose and episode identity; nine manual actions and the prepared flight
training request passed, with no training dispatched. The real neural aversion
test delivered its first 28.6 ms PPL interval before drawing the 3D electrodes and
arc. At that point changed plastic edges were **zero**; neither completed learning
nor delivery of the entire 200 ms pulse is claimed. The owned port18801 server and
its full-brain process were closed; the active port18800 Lab was untouched.
