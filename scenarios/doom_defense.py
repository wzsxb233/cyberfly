"""Official bundled ViZDoom defend-center/line assets with real pixel observations."""
from copy import deepcopy
from .doom_basic import DoomBasicScenario
from .doom_neural_link import DoomNeuralLink, DOOM_NEURAL_LINK_SPEC
from .registry import DOOM_SPEC

ASSETS={'center':'defend_the_center','line':'defend_the_line'}

def defense_spec(layout,*,neural=False):
    asset=ASSETS[layout]
    name=f'doom_defend_{layout}'+('_neural' if neural else '')
    spec=deepcopy(DOOM_NEURAL_LINK_SPEC if neural else DOOM_SPEC)
    spec.update(name=name,title=('神经直连 · ' if neural else '')+('Doom 环形防守' if layout=='center' else 'Doom 防线守卫'),
        engine=f"{'Real full connectome → ' if neural else ''}ViZDoom / official {asset}.cfg and .wad",
        parameters={'difficulty':{'type':'integer','minimum':0,'maximum':3,'default':1},
            'target_kills':{'type':'integer','minimum':1,'maximum':100,'default':5},
            'max_episode_steps':{'type':'integer','minimum':1,'maximum':1000,'default':300}},
        task_updates={'target_kills':'1..100 actual kills before death','step_limit':'1..1000 decisions',
            **({'neural_drive':'Three-port semantic context','general_stimulation':'Persistent full-brain currents'} if neural else {})},
        limitations=['Official defense scenario with real enemies and damage, not the full Doom campaign',
            'Success requires target_kills before dying; time limit is truncation',
            'Direct policy sees RGB pixels; optional neural adapter uses measured brain readouts'])
    if not neural:spec['action_labels']=['turn_left','turn_right','attack']
    return spec

class DoomDefenseScenario(DoomBasicScenario):
    BUTTON_NAMES=('TURN_LEFT','TURN_RIGHT','ATTACK')
    DEFAULT_TARGET_KILLS=5
    def __init__(self,*,layout='center',difficulty=1,max_episode_steps=300,**kwargs):
        if layout not in ASSETS:raise ValueError('Doom defense layout must be center or line.')
        if type(difficulty) is not int or not 0<=difficulty<=3:raise ValueError('difficulty must be 0..3.')
        self.ASSET_NAME=ASSETS[layout]
        self.SCENARIO_SPEC=defense_spec(layout)
        self.difficulty=difficulty
        super().__init__(max_episode_steps=max_episode_steps,**{'doom_skill':difficulty+1,**kwargs})

class DoomDefenseNeuralScenario(DoomNeuralLink):
    def __init__(self,*,layout='center',max_episode_steps=300,**kwargs):
        if layout not in ASSETS:raise ValueError('Doom defense layout must be center or line.')
        self.SCENARIO_SPEC=defense_spec(layout,neural=True)
        super().__init__(layout=layout,max_episode_steps=max_episode_steps,**kwargs)

    def _make_body(self,max_episode_steps,render_mode,kwargs):
        return DoomDefenseScenario(max_episode_steps=max_episode_steps,render_mode=render_mode,**kwargs)

    def apply_task(self,task):
        if not isinstance(task,dict) or not task:raise ValueError('Task must be a nonempty object.')
        brain={key:value for key,value in task.items() if key in {'neural_drive','general_stimulation'}}
        body={key:value for key,value in task.items() if key not in brain}
        # Validate neural fields before changing the defense objective.
        from connectome_adapter.ports import validate_neural_drive,validate_general_stimulation
        if 'neural_drive' in brain:validate_neural_drive(brain['neural_drive'])
        if 'general_stimulation' in brain:validate_general_stimulation(brain['general_stimulation'])
        result=self.body.apply_task(body) if body else {'applied':{}}
        if brain:result['applied'].update(super().apply_task(brain)['applied'])
        return result

    def task_state(self):
        result=super().task_state()
        result['decoder_note']='Engineered: actual neural attack selects shooting; otherwise the native turn sign rotates the real camera left/right.'
        return result
