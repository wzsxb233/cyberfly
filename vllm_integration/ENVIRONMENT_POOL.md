# Independent CPU environment pool

`EnvironmentPool` does not start processes until `start()` is called. The initial
implementation supports one or two built-in physical neural scenarios. Each owns
a MuJoCo body process and a separate full MaleCNS brain process. Brain dynamics,
weights, episode state and body random seeds are independent. The seed controls
the body/environment RNG; it does not invent random biological wiring.

```python
from vllm_integration.environment_pool import EnvironmentPool

with EnvironmentPool() as pool:
    inputs = pool.capture_inputs(include_neural_state=True)
    # An external dispatcher may now process these actual inputs.
    # This module itself never calls an LLM.
    results = pool.step_many({env_id: {
        "input_id": packet["input_id"],
        "capture_neural_state": True,
    } for env_id, packet in inputs.items()})
    pool.select("fly-1")
    jpeg = pool.render_selected()  # only this environment encodes display JPEG
```

Each input has immutable `rgb_file` (uint8 RGB NPY), `body_packet_file` (the
existing raw MuJoCo sensor packet), and optionally `neural_state_file` (full
166,700-cell voltage/spike NPZ descriptor). `input_id` rejects stale or foreign
outputs; the actual neural input RGB SHA must match the pinned record. A step
with `capture_neural_state=True` returns `feedback`, including the actual after
state and `source_input_id`. `capture_feedback(include_neural_state=True)` reads
the current body/brain without advancing or reserving the next input.

For model additions, pass `model_currents_file`. The worker budgets only this
new component against the actual existing general currents and executed sensory
ports, then verifies the full brain's delivered-current SHA. Returned
`parallel_inputs` records requested/applied differences and preservation evidence.
The separate `neuron_currents_file` entry is an explicit direct experiment and
does not claim the same model budget. They cannot be combined in one command.
Visual input remains enabled in every instance. One-interval file inputs are not
replayed; persistent group inputs remain after the step.

Other calls: `status()` reads cached status, `set_task(id, task)` changes an explicit
task, `reset(id, seed=...)` preserves that brain's weights, `save(id)` creates an
independent full brain checkpoint, and `close()` releases only the pool's owned
workers. Episodes do not reset automatically. A partial batch failure reports
completed results and failures; successful environments must not be replayed.

Rendering uses Mesa llvmpipe with CUDA hidden. Every instance still renders the
real sensory camera for its brain; only the selected instance adds UI JPEG
encoding/transmission. Saved before/after records have separate brain and body
clocks: default intervals are 28.6 ms and 10 ms respectively, not synchronized
biological time. Callers own retention of immutable input/feedback artifacts.

Measured environment-only verification: `artifacts/environment_pool/b470557cf7ec`.
Two instances initialized in 11.38 s; three parallel step batches took
0.357/0.280/0.308 s, excluding separate capture/model work. Each instance used
about 1.53 GiB RSS for its two processes. This is an environment smoke test, not
an LLM integration or learned-behavior result. Native before/after snapshots and
preserved-input budget verification: `artifacts/environment_pool/ee6c7171d7e2`.

## Local fleet controller and API

`FleetController` owns a command thread around `EnvironmentPool` and the actual
`NativeOmniFlow`. Construction does not create environments or make model
requests. `/api/fleet/start` accepts `{count: 2}` and returns immediately after
validation; initialization, native model requests and body steps run separately
from cached `/api/fleet` and `/api/fleet/frame?id=...` reads. `/select` queues the
selection without resetting either body. `/stop` finishes an in-flight step,
saves each independent brain, then closes the owned pool. Failed saves leave the
environment alive and paused so stopping can be retried. Already in-flight model
requests may finish, but their results no longer drive the stopped environments;
the model services themselves are never stopped by this controller.

Start requires `artifacts/vllm_omni_migration/acceptance.json` to contain
`status="passed"`, `model_bidirectional_verified=true`, and
`native_speech_verified=true` with actual boolean types. No missing or partial
marker starts a fleet. The current server explicitly configures the existing
`brain_checkpoints/live-scheduling-1789312227` seed when no newer explicit saved
source is supplied. Its generation is pinned before startup and reported in
`brain_seed_source`; it is **not a copy of current live brain state**. The live
Lab remains untouched. Cached scenario, current persistent sensory inputs and
selected motor readout are inherited; one-interval current files are excluded.

Optional start fields: `scenario`, `brain_checkpoint`, `motor_checkpoint`, `seed`,
`text`, and `speak`. Each copied environment gets a distinct seed. Defaults are
two flies, seeds 20260913/20260914, a short exploration instruction and no speech
request. Only selected, changed camera frames are JPEG encoded by the owner and
cached for HTTP. Below a configurable 2 GiB free-space threshold, only this
pool's advancement/writes pause; no GPU service is stopped. Temporary unpinned
budget snapshots are removed after verified current delivery, retaining their
hashes and times. Input observations needed by model requests are preserved.

Controller protocol tests inject mock pools/flows and never invoke a model or
GPU. Run `python -m unittest vllm_integration.tests.test_fleet_controller -v` in
the RL environment. Those tests establish lifecycle/cache behavior, not native
model integration; the actual model acceptance marker remains a separate gate.

## Independent online replay and responses

Fleet online learning is off by default. `POST /api/fleet/online/start {}` and
`POST /api/fleet/online/stop {}` control only this fleet's measured-data source;
`GET /api/fleet/online` reads its status. It does not borrow the single-Lab online
loop or its episodes. One actual transition is sampled at most every two seconds,
round-robin across the flies. Before and after RGB/body/full-brain evidence is
copied into the learner's own bounded storage, with source SHA checks and named
retention leases. Observer-only source observations become eligible for cleanup
after copying; observations still needed by a model turn remain protected.

The unchanged grounding verifier derives labels from measured displacement,
contact changes and spikes in the 2,129 annotated output cells. An external model
can drive the body, but its generated words never become supervision labels.
`model_called=false` describes the collector only; records explicitly preserve
`external_model_may_have_driven_body=true`. Four new verified records trigger an
eight-step candidate job through the same GPU1 gate as existing training. Busy
jobs queue; they do not stop or replace another training process. Fixed test
records remain sealed, validation is separate, and candidates are not deployed
automatically. Stop cancels only this source's queued or owned training job.

Actual archived-transition format validation is in
`artifacts/fleet_online_validation/2d0e351d5caa`: displacement 0.05688446 mm,
contacts 6→6 and 903 spikes over the selected output neurons. This verifies
collection/label integrity; the GPU dispatch checks there use an explicit mock
and do not claim a new model training run.

Fleet state includes each environment's latest completed text, actual model
application count, generation phase and body steps during model computation.
Optional `start.speak=true` requests native model speech. The UI never plays it
automatically. `GET /api/fleet/audio?id=fly0&sha256=...` serves only the selected
environment's current response, within this flow's owned directory, after file
SHA and nonempty 24 kHz mono PCM16 validation. Missing/changed audio clears the
old player. This route cannot read arbitrary files or synthesize replacement
speech. Browser tests using fixtures establish the UI contract, separately from
the mandatory actual-model acceptance gate.

The original single-Lab online panel independently polls `GET /api/online` about
every two seconds. Once received, that direct status takes precedence over older
actor snapshots, so pausing a body no longer freezes its displayed training
progress. Errors remain local to the training panel.
