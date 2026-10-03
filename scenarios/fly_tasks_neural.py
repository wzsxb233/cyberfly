"""Task-family neural adapters; original neural checkpoint observation identities stay intact."""
from copy import deepcopy
from connectome_adapter.ports import validate_neural_drive
from .neural_link import FlyNeuralLink
from .fly_tasks import FlyTaskScenario, task_spec
from .neural_inputs import validate_general_stimulation

class FlyTaskNeuralScenario(FlyNeuralLink):
    def __init__(self, *, scenario_name=None, task_type="food", max_episode_steps=300, **kwargs):
        name=scenario_name or f"fly_{task_type}_neural"
        self.SCENARIO_SPEC=task_spec(name,task_type,neural=True)
        super().__init__(task_type=task_type,scenario_name=name.removesuffix("_neural"),max_episode_steps=max_episode_steps,**kwargs)

    def reset(self, *, seed=None, options=None):
        result=super().reset(seed=seed,options=options)
        self.scenario_spec=task_spec(self.scenario_spec["name"],self.body.task_state()["task_type"],neural=True)
        return result

    def _make_body(self,max_episode_steps,render_mode,kwargs):
        return FlyTaskScenario(max_episode_steps=max_episode_steps,render_mode=render_mode,**kwargs)

    def apply_task(self,task):
        if not isinstance(task,dict) or not task:
            raise ValueError("Task must be a nonempty object.")
        remaining=deepcopy(task)
        context=validate_neural_drive(remaining.pop("neural_drive")) if "neural_drive" in remaining else None
        general=validate_general_stimulation(remaining.pop("general_stimulation")) if "general_stimulation" in remaining else None
        result={"applied":{}}
        if remaining:
            result=self.body.apply_task(remaining)
        if context is not None:
            self.semantic_drive=context
            result.setdefault("applied",{})["neural_drive"]=deepcopy(context)
            result["neural_context_effect"]="Explicit next neural action delivers currents; context does not move the body."
        if general is not None:
            self.general_stimulation=general
            result.setdefault("applied",{})["general_stimulation"]=deepcopy(general)
        return result
