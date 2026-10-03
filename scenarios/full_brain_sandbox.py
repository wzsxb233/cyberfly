"""All-neuron, macro-group current-control task with real measured observations."""
from __future__ import annotations
from copy import deepcopy
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image, ImageDraw
from .brain_sandbox import BrainSandboxScenario
from .neural_inputs import validate_general_stimulation

CATALOG_PATH=Path(__file__).resolve().parents[1]/'artifacts/full_brain/catalog.json'

def load_catalog():
    if not CATALOG_PATH.is_file():
        raise FileNotFoundError(f"Full brain group catalog is missing: {CATALOG_PATH}")
    catalog=json.loads(CATALOG_PATH.read_text())
    groups=catalog['macro_groups']
    if len(groups)!=catalog['macro_group_count'] or sum(g['count'] for g in groups)!=catalog['coverage']:
        raise ValueError('Full-brain catalog coverage is inconsistent.')
    if [g['index'] for g in groups]!=list(range(len(groups))):
        raise ValueError('Full-brain macro group indices are not contiguous.')
    return catalog

def full_brain_spec():
    catalog=load_catalog();groups=catalog['macro_groups'];count=len(groups)
    return {'name':'full_brain_sandbox','title':'全脑训练 · 47群电流控制',
        'observation_kind':'vector','observation_shape':[3*count+5],'policy_type':'MlpPolicy',
        'action_kind':'continuous','action_shape':[count],'action_labels':[g['group_id'] for g in groups],
        'action_interface':'group_currents','current_scale_mv':30,
        'macro_groups':deepcopy(groups),'macro_partition_sha256':catalog['macro_partition_sha256'],
        'neural_input':True,'embodiment':'none','control_modes':['manual','policy','llm_brain'],
        'supports_3d_feedback':False,
        'render':'rgb_array','render_shape':[480,640,3],'supports_demonstration':True,
        'engine':'Real full MaleCNS graph / macro-group currents / all retained neurons',
        'neuron_coverage':catalog['coverage'],'fine_group_count':catalog['group_count'],
        'parameters':{'target_rate_hz':{'type':'number','minimum':.1,'maximum':20,'default':2},
            'max_episode_steps':{'type':'integer','minimum':1,'maximum':1000,'default':64},
            'learning':{'type':'boolean','default':False}},
        'task_updates':{'general_stimulation':'Additional fine-group, macro-group or original neuron-ID currents',
            'target_rate_hz':'Engineered group-activity tracking target, 0.1..20 Hz',
            'neural_drive':'Three-port semantic observation context only'},
        'limitations':['All neurons and edges remain in the full simulation; actions group currents by superclass and root side',
            'PPO/BC learns a macro-current adapter; synaptic plasticity is a separate setting',
            'Observation uses real per-group rate, membrane-voltage mean and previous action, plus task context',
            'Reward is engineered activity tracking, not natural preference or task mastery',
            'Rendered grid shows measured group activity; it does not depict biological anatomy']}


class FullBrainSandboxScenario(BrainSandboxScenario):
    def __init__(self, *, general_stimulation=None, **kwargs):
        self._catalog=load_catalog()
        self._groups=self._catalog['macro_groups']
        self._group_ids=[g['group_id'] for g in self._groups]
        self._group_count=len(self._groups)
        self._last_macro_action=np.zeros(self._group_count,dtype=np.float32)
        self._macro_rates=np.zeros(self._group_count,dtype=float)
        self._macro_voltage=np.zeros(self._group_count,dtype=float)
        self._verified_partition=False
        super().__init__(**kwargs)
        self.general_stimulation=validate_general_stimulation(general_stimulation or [])
        if len(self.general_stimulation)>15:
            raise ValueError('Full-brain policy reserves one input slot; at most 15 additional targets are allowed.')
        self.scenario_spec=full_brain_spec()
        self.action_space=spaces.Box(-1,1,(self._group_count,),dtype=np.float32)
        self.observation_space=spaces.Box(-10,10,(3*self._group_count+5,),dtype=np.float32)
        self._obs=np.zeros(self.observation_space.shape,dtype=np.float32)
        self.group_state={}

    def _ensure_brain(self):
        super()._ensure_brain()
        if not self._verified_partition:
            actual=self.brain.group_catalog()
            if actual['macro_partition_sha256']!=self._catalog['macro_partition_sha256'] or actual['coverage']!=self._catalog['coverage']:
                self.close()
                raise RuntimeError('Brain partition differs from the full-brain policy observation/action identity.')
            self._verified_partition=True

    def _update(self,result):
        self._last_neural=result
        self._applied_drive=result['neural_drive']['normalized'].copy()
        groups=result['group_state']
        if groups['group_ids']!=self._group_ids:
            raise RuntimeError('Neural group order changed; refusing to misalign policy actions and observations.')
        counts=np.asarray(groups['counts'],dtype=float)
        if counts.sum()!=self._catalog['coverage']:
            raise RuntimeError('Neural group observation does not cover the full retained brain.')
        self._macro_rates=np.asarray(groups['spikes'],dtype=float)/np.maximum(counts,1)/(result['neural']['duration_ms']/1000)
        self._macro_voltage=np.asarray(groups['voltage_mean'],dtype=float)
        self.group_state=deepcopy(groups)
        self._obs=np.clip(np.r_[self._macro_rates/20,self._macro_voltage/30,self._last_macro_action,
            [self.semantic_drive[k] for k in ('retina_left','retina_right','sugar')],self.target_rate_hz/20,float(self.learning)],-10,10).astype(np.float32)
        if not self.observation_space.contains(self._obs):
            raise RuntimeError('Full-brain policy received invalid measured observations.')

    def _info(self,reward=0.0,*,truncated=False):
        info=super()._info(reward,truncated=truncated)
        info.update(scenario='full_brain_sandbox',group_state=deepcopy(self.group_state),
            general_stimulation=deepcopy(self.general_stimulation),
            macro_action=self._last_macro_action.tolist(),macro_currents_mv=(30*self._last_macro_action).tolist(),
            macro_group_count=self._group_count,neuron_coverage=self._catalog['coverage'],
            macro_partition_sha256=self._catalog['macro_partition_sha256'],
            reward_definition='-population-weighted mean abs(group_Hz-target_Hz)/max(target_Hz,1) - 0.01*mean(normalized_macro_current**2)')
        return info

    def reset(self,*,seed=None,options=None):
        if self._closed:
            raise RuntimeError('This full-brain environment is closed.')
        gym.Env.reset(self,seed=seed)
        if options:self.apply_task(options)
        self._ensure_brain()
        self.brain.reset(preserve_weights=True)
        self.general_stimulation=[item for item in self.general_stimulation if 'neuron_currents_file' not in item]
        self._steps,self._return,self._pending_aversive=0,0.0,False
        self._started,self._done=True,False
        self._last_macro_action=np.zeros(self._group_count,dtype=np.float32)
        self._update(self.brain.step(self._blank_rgb,duration_ms=self.neural_duration_ms,
            general_stimulation=self.general_stimulation,visual_input_enabled=self.visual_input_enabled,
            include_group_stats=True))
        self._last_info=self._info()
        return self._obs.copy(),deepcopy(self._last_info)

    def step(self,action):
        if not self._started or self._done:
            raise RuntimeError('Call reset() before stepping a new or finished episode.')
        action=np.asarray(action,dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f'Full-brain action must have {self._group_count} finite values within [-1,1].')
        stimuli=[{'macro_group_currents_mv':(30*action.astype(float)).tolist()},*deepcopy(self.general_stimulation)]
        self.general_stimulation=[item for item in self.general_stimulation if 'neuron_currents_file' not in item]
        result=self.brain.step(self._blank_rgb,duration_ms=self.neural_duration_ms,
            general_stimulation=stimuli,include_group_stats=True,
            visual_input_enabled=self.visual_input_enabled,
            reward_event={'aversive':self._pending_aversive})
        self._pending_aversive=False
        self._last_macro_action=action.copy()
        self._update(result)
        counts=np.asarray(self.group_state['counts'],dtype=float)
        error=float(np.average(np.abs(self._macro_rates-self.target_rate_hz),weights=counts))
        reward=-error/max(self.target_rate_hz,1)-.01*float(np.mean(np.square(action)))
        self._steps+=1;self._return+=reward;self._done=self._steps>=self.max_episode_steps
        self._last_info=self._info(reward,truncated=self._done)
        return self._obs.copy(),float(reward),False,self._done,deepcopy(self._last_info)

    def set_general_stimulation(self,stimulation):
        value=validate_general_stimulation(stimulation)
        if len(value)>15:
            raise ValueError('At most 15 additional targets; the full-brain action occupies one slot.')
        self.general_stimulation=value
        return {'general_stimulation':deepcopy(value),'delivery':'next real brain interval'}

    def shared_passthrough_action(self):
        """Retain the existing macro policy input while model currents add to it."""
        return self._last_macro_action.copy()

    def apply_task(self,task):
        if not isinstance(task,dict) or not task:
            raise ValueError('Task must be a nonempty object.')
        remaining=deepcopy(task)
        general=validate_general_stimulation(remaining.pop('general_stimulation')) if 'general_stimulation' in remaining else None
        if general is not None and len(general)>15:
            raise ValueError('At most 15 additional stimulation targets are allowed.')
        result=super().apply_task(remaining) if remaining else {'applied':{}}
        if general is not None:
            self.general_stimulation=general
            result['applied']['general_stimulation']=deepcopy(general)
        return result

    def task_state(self):
        return {**super().task_state(),'scenario':'full_brain_sandbox','group_state':deepcopy(self.group_state),
            'macro_group_count':self._group_count,'neuron_coverage':self._catalog['coverage'],
            'general_stimulation':deepcopy(self.general_stimulation),
            'macro_action':self._last_macro_action.tolist(),'macro_currents_mv':(30*self._last_macro_action).tolist(),
            'macro_partition_sha256':self._catalog['macro_partition_sha256']}

    def render(self):
        if not self._started:raise RuntimeError('Call reset() before render().')
        image=Image.new('RGB',(640,480),'#101923');draw=ImageDraw.Draw(image)
        neural=self._last_neural['neural']
        draw.text((16,16),'FULL BRAIN / MEASURED MACRO-GROUP ACTIVITY',fill='#dfebf5')
        draw.text((16,42),f"{neural['neurons']:,} neurons | {neural['edges']:,} edges | {self._group_count} action groups",fill='#91b8cb')
        draw.text((16,66),f"Step {self._steps}/{self.max_episode_steps} | target {self.target_rate_hz:.2f} Hz | actual {self._mean_rate():.2f} Hz",fill='#a6e296')
        for i,group in enumerate(self._groups):
            x=16+(i%8)*77;y=105+(i//8)*53
            rate=self._macro_rates[i];strength=float(np.clip(rate/20,0,1))
            color=(int(30+70*strength),int(52+130*strength),int(63+70*strength))
            draw.rounded_rectangle((x,y,x+70,y+46),radius=5,fill=color)
            draw.text((x+4,y+3),group['superclass'][:9],fill='#dceaf2')
            draw.text((x+4,y+17),f'{rate:.2f} Hz',fill='#bdec94')
            draw.text((x+4,y+30),f"n={group['count']}",fill='#99b1c3')
        draw.text((16,446),'Measured group grid, not anatomy. All retained neurons remain simulated.',fill='#99b1c3')
        return np.asarray(image).copy()
