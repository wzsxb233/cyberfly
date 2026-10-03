"""Measured MuJoCo body inputs and a separately trainable neural encoder.

The packet is raw simulator data. The encoder's body-to-vnc_sensory pairing and
normalization are engineering choices, not recovered anatomical wiring. Its
small initial outputs are explicitly untrained; original senses stay separate.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
GRAPH_PATH = ROOT / 'vendor/doomfly/outputs/doom/malecns_v1/graph.npz'
CATALOG_PATH = ROOT / 'artifacts/full_brain/catalog.json'
N = 166700
SENSORY_N = 6370
SENSORY_IDS_SHA256 = 'b6c53381507c727c5621579cc077645585e095c0b9053380d2484be6aa69347b'


def json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _name(model, kind, index):
    import mujoco
    return mujoco.mj_id2name(model, kind, index) or f'unnamed_{index}'


def sample_body_senses(env, *, rgb=None):
    """Read one instant, all contacts, with native coordinate/force conventions."""
    import mujoco
    m, d = env.sim.mj_model, env.sim.mj_data
    before = float(d.time)
    force_unit = 'model_mass_unit*mm/s^2'
    torque_unit = 'model_mass_unit*mm^2/s^2'
    qpos_units, qvel_units, joints = [], [], []
    for index in range(m.njnt):
        kind = int(m.jnt_type[index])
        pos_units = {0:['mm']*3+['unit quaternion wxyz']*4, 1:['unit quaternion wxyz']*4, 2:['mm'], 3:['rad']}[kind]
        vel_units = {0:['mm/s']*3+['rad/s']*3, 1:['rad/s']*3, 2:['mm/s'], 3:['rad/s']}[kind]
        joints.append({'name':_name(m,mujoco.mjtObj.mjOBJ_JOINT,index), 'type':('free','ball','slide','hinge')[kind],
            'qpos_address':int(m.jnt_qposadr[index]),'dof_address':int(m.jnt_dofadr[index]),
            'qpos_size':len(pos_units),'dof_size':len(vel_units)})
        qpos_units.extend(pos_units);qvel_units.extend(vel_units)
    bodies = [_name(m,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(m.nbody)]
    geoms = [_name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
    actuators = [{'name':_name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i),
                  'transmission_type':int(m.actuator_trntype[i]),'transmission_ids':m.actuator_trnid[i].tolist(),
                  'force_units': torque_unit if int(m.actuator_trntype[i])==0 else force_unit if int(m.actuator_trntype[i])==5 else 'native transmission-dependent actuator scalar'} for i in range(m.nu)]
    sensors = [{'name':_name(m,mujoco.mjtObj.mjOBJ_SENSOR,i),'type':int(m.sensor_type[i]),
                'address':int(m.sensor_adr[i]),'dimensions':int(m.sensor_dim[i])} for i in range(m.nsensor)]
    layout = {'nq':int(m.nq),'nv':int(m.nv),'nu':int(m.nu),'nbody':int(m.nbody),'ngeom':int(m.ngeom),
        'nsensor':int(m.nsensor),'nsensordata':int(m.nsensordata),'joints':joints,'body_names':bodies,
        'geom_names':geoms,'geom_body_ids':m.geom_bodyid.tolist(),'actuators':actuators,'sensors':sensors,
        'qpos_units':qpos_units,'qvel_units':qvel_units,
        'units':{'length':'mm','time':'s','angles':'rad','mass':'native model mass unit; SI mass calibration is not asserted',
                 'force':force_unit,'torque':torque_unit,'actuator_force':'per-actuator transmission units above',
                 'qfrc_actuator':'generalized force: translational DOFs use force, rotational DOFs use torque',
                 'sensordata':'per MuJoCo sensor type; no synthetic values for absent sensors'},
        'cameras':[_name(m,mujoco.mjtObj.mjOBJ_CAMERA,i) for i in range(m.ncam)]}
    contact_force = np.zeros((m.nbody,3)); contact_torque = np.zeros((m.nbody,3)); contact_count = np.zeros(m.nbody,dtype=np.int32)
    contacts=[]
    for index in range(d.ncon):
        c=d.contact[index];wrench=np.zeros(6)
        mujoco.mj_contactForce(m,d,index,wrench)
        frame=c.frame.reshape(3,3).copy();world_force=frame.T@wrench[:3];world_torque=frame.T@wrench[3:]
        geom1,geom2=int(c.geom1),int(c.geom2)
        contacts.append({'index':index,'geom_ids':[geom1,geom2],
            'geom_names':[geoms[g] if g>=0 else 'non-geom contact' for g in (geom1,geom2)],
            'position_world_mm':c.pos.tolist(),'distance_mm':float(c.dist),'dimension':int(c.dim),
            'exclude':int(c.exclude),'frame_rows_world':frame.tolist(),
            'wrench_contact_force_then_torque':wrench.tolist(),
            'world_force_on_geom2':world_force.tolist(),'world_torque_at_contact_on_geom2':world_torque.tolist()})
        if not c.exclude:
            for geom,sign in ((geom1,-1),(geom2,1)):
                if geom>=0:
                    body=int(m.geom_bodyid[geom]);force=sign*world_force
                    contact_force[body]+=force
                    contact_torque[body]+=sign*world_torque+np.cross(c.pos-d.xipos[body],force)
                    contact_count[body]+=1
    velocity=np.zeros(6)
    mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,env.thorax_body_id,velocity,0)
    packet={'schema':1,'source':'real MuJoCo NeuroMechFly body','simulation_time_s':before,
        'decision_step':int(env._step_count),'physics_timestep_s':float(m.opt.timestep),'body_interval_s':float(env.action_dt),
        'layout':layout,'model_layout_sha256':json_sha(layout),
        'qpos':d.qpos.copy().tolist(),'qvel':d.qvel.copy().tolist(),'qacc':d.qacc.copy().tolist(),
        'actuator_force':d.actuator_force.copy().tolist(),'qfrc_actuator':d.qfrc_actuator.copy().tolist(),
        'sensordata':d.sensordata.copy().tolist(),'contacts':contacts,'contact_count':int(d.ncon),
        'body_contact_force_world':contact_force.tolist(),'body_contact_torque_at_com_world':contact_torque.tolist(),
        'body_contact_count':contact_count.tolist(),
        'thorax':{'body_id':int(env.thorax_body_id),'position_world_mm':d.xpos[env.thorax_body_id].copy().tolist(),
                  'quaternion_world_wxyz':d.xquat[env.thorax_body_id].copy().tolist(),
                  'velocity_world_rot_rad_s_then_lin_mm_s':velocity.tolist()},
        'sensor_availability':{'configured_mujoco_sensors':int(m.nsensor),'raw_contacts_available':True,
            'note':'No dedicated touch sensor was instantiated when nsensor=0; contacts are actual solver measurements'},
        'interpretation':'Raw physical state includes simulator joint coordinates and pose; biological sensory transduction and anatomical neuron pairing are not established'}
    if rgb is not None:
        pixels=np.asarray(rgb)
        if pixels.dtype!=np.uint8 or pixels.ndim!=3 or pixels.shape[-1]!=3:raise ValueError('Expected actual RGB uint8 camera frame')
        packet['vision']={'rgb_sha256':array_sha(pixels),'shape':list(pixels.shape),'source':'existing actual MuJoCo tracking camera; no second camera capture','time_s':before}
    if float(d.time)!=before:raise RuntimeError('Physics advanced while sampling body senses')
    packet['packet_sha256']=json_sha(packet)
    return packet


class BodySenseCodec:
    def __init__(self, packet):
        self._validate_packet(packet)
        self.layout=deepcopy(packet['layout']);self.layout_sha=packet['model_layout_sha256']
        self.blocks=[('qpos',(self.layout['nq'],)),('qvel',(self.layout['nv'],)),
            ('actuator_force',(self.layout['nu'],)),('qfrc_actuator',(self.layout['nv'],)),
            ('sensordata',(self.layout['nsensordata'],)),('body_contact_force_world',(self.layout['nbody'],3)),
            ('body_contact_torque_at_com_world',(self.layout['nbody'],3)),('body_contact_count',(self.layout['nbody'],)),
            ('thorax.quaternion_world_wxyz',(4,)),('thorax.velocity_world_rot_rad_s_then_lin_mm_s',(6,))]
        self.feature_dim=sum(int(np.prod(shape)) for _,shape in self.blocks)
        self.dimensions=[]
        for name,shape in self.blocks:
            for i in range(int(np.prod(shape))):
                if name=='qpos':
                    unit=self.layout['qpos_units'][i];scale=10.0 if unit=='mm' else float(np.pi) if unit=='rad' else 1.0
                elif name=='qvel':unit=self.layout['qvel_units'][i];scale=50.0
                elif name=='actuator_force':unit=self.layout['actuators'][i]['force_units'];scale=1000.0
                elif name=='qfrc_actuator':unit=self.layout['units']['force'] if self.layout['qvel_units'][i]=='mm/s' else self.layout['units']['torque'];scale=1000.0
                elif name=='body_contact_force_world':unit=self.layout['units']['force'];scale=1000.0
                elif name=='body_contact_torque_at_com_world':unit=self.layout['units']['torque'];scale=1000.0
                elif name=='body_contact_count':unit='contact count';scale=10.0
                elif name=='thorax.velocity_world_rot_rad_s_then_lin_mm_s':unit='rad/s' if i<3 else 'mm/s';scale=50.0
                elif name=='thorax.quaternion_world_wxyz':unit='unit quaternion wxyz';scale=1.0
                else:unit='native MuJoCo sensor value';scale=1.0
                self.dimensions.append({'index':len(self.dimensions),'name':f'{name}[{i}]','original_unit':unit,'divide_by':scale})
        self.identity={'schema':1,'feature_dim':self.feature_dim,'model_layout_sha256':self.layout_sha,
            'blocks':[{'name':name,'shape':list(shape)} for name,shape in self.blocks],
            'normalization':'qpos mm/10,rad/pi,quaternion/1; qvel/50; native forces and torques/1000; contact_count/10; thorax angular+linear velocity/50; configured sensordata raw',
            'contact_reduction':'Raw packet retains every contact; feature vector sums all active contacts per body with torque about its center of mass',
            'dimensions':self.dimensions,
            'interpretation':'Numerical engineering features, including privileged physical joint coordinates; no natural receptor model'}
        self.identity['feature_layout_sha256']=json_sha(self.identity)

    @staticmethod
    def _validate_packet(packet):
        if not isinstance(packet,dict) or packet.get('schema')!=1:raise ValueError('Expected body sensory packet schema 1')
        value={k:v for k,v in packet.items() if k!='packet_sha256'}
        if json_sha(value)!=packet.get('packet_sha256'):raise ValueError('Body packet hash changed')
        if json_sha(packet.get('layout'))!=packet.get('model_layout_sha256'):raise ValueError('Body layout identity changed')
        if not np.isfinite(packet.get('simulation_time_s',np.nan)):raise ValueError('Body packet needs an actual finite simulation time')

    def encode(self, packet):
        self._validate_packet(packet)
        if packet['model_layout_sha256']!=self.layout_sha:raise ValueError('This body encoder belongs to another model layout')
        result=[]
        for name,shape in self.blocks:
            value=packet
            for key in name.split('.'):value=value[key]
            array=np.asarray(value,dtype=np.float64)
            if array.shape!=shape or not np.isfinite(array).all():raise ValueError('Invalid measured body feature: '+name)
            if name=='qpos':array=array/np.array([10 if u=='mm' else np.pi if u=='rad' else 1 for u in self.layout['qpos_units']])
            elif name in ('qvel','thorax.velocity_world_rot_rad_s_then_lin_mm_s'):array=array/50
            elif 'force' in name or 'torque' in name or name=='qfrc_actuator':array=array/1000
            elif name=='body_contact_count':array=array/10
            result.append(array.reshape(-1))
        features=np.concatenate(result).astype(np.float32)
        if not np.isfinite(features).all():raise ValueError('Body features overflowed float32')
        return features


class BodySenseAdapter(nn.Module):
    def __init__(self, packet, *, hidden_width=32, current_scale_mv=1.0, seed=42):
        super().__init__()
        if type(hidden_width)is not int or not 4<=hidden_width<=256 or not np.isfinite(current_scale_mv) or not 0<current_scale_mv<=30:raise ValueError('Invalid body-sense adapter dimensions/current scale')
        self.codec=BodySenseCodec(packet);self.hidden_width=hidden_width;self.current_scale_mv=float(current_scale_mv)
        with np.load(GRAPH_PATH,allow_pickle=False) as graph:self.ids=graph['ids'];classes=graph['superclass']
        catalog=json.loads(CATALOG_PATH.read_text());self.indices=np.flatnonzero(classes=='vnc_sensory')
        if len(self.indices)!=SENSORY_N or array_sha(self.ids)!=catalog['ids_sha256']:raise ValueError('Body sensory target identity differs from the full graph')
        self.target_ids_sha256=array_sha(self.ids[self.indices]);self.graph_ids_sha256=catalog['ids_sha256']
        if self.target_ids_sha256!=SENSORY_IDS_SHA256:raise ValueError('The sensory-cell annotation selection changed')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.network=nn.Sequential(nn.Linear(self.codec.feature_dim,hidden_width),nn.Tanh(),nn.Linear(hidden_width,SENSORY_N),nn.Tanh())
            nn.init.normal_(self.network[2].weight,mean=0,std=1e-4);nn.init.zeros_(self.network[2].bias)
        self.training_steps=0;self.label_source=None

    def forward(self, features):
        if not isinstance(features,torch.Tensor) or features.shape[-1]!=self.codec.feature_dim or not torch.isfinite(features).all():raise ValueError('Invalid body encoder features')
        return self.network(features)*self.current_scale_mv

    def adapter_sha256(self):
        h=hashlib.sha256()
        for name,value in self.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def describe(self):
        return {'schema':1,'component':'body_senses_to_neurons','feature_dim':self.codec.feature_dim,
            'feature_layout_sha256':self.codec.identity['feature_layout_sha256'],'model_layout_sha256':self.codec.layout_sha,
            'target_superclass':'vnc_sensory','target_neurons':SENSORY_N,'target_ids_sha256':self.target_ids_sha256,
            'graph_ids_sha256':self.graph_ids_sha256,'neuron_coverage':N,'current_scale_mv':self.current_scale_mv,
            'hidden_width':self.hidden_width,'adapter_sha256':self.adapter_sha256(),'training_steps':self.training_steps,
            'label_source':self.label_source,'initialization':'Small random encoder; untrained when training_steps=0',
            'interpretation':'Trainable engineering pairing of physical body measurements to annotated sensory cells; natural proprioceptive wiring has not been recovered'}

    def predict(self, packet, *, directory=None):
        from connectome_adapter.neuron_currents import write_neuron_currents
        features=self.codec.encode(packet)
        with torch.no_grad():values=self(torch.from_numpy(features).to(next(self.parameters()).device)).cpu().numpy().astype(np.float32)
        currents=np.zeros(N,dtype=np.float32);currents[self.indices]=values
        descriptor=write_neuron_currents(currents,self.ids,directory=directory)
        return {**self.describe(),'packet_sha256':packet['packet_sha256'],'body_simulation_time_s':packet['simulation_time_s'],
            'features_sha256':array_sha(features),'currents_file':descriptor,'source':'real_body_sensory_packet',
            'minimum_current_mv':float(values.min()),'maximum_current_mv':float(values.max()),
            'non_target_nonzero':int(np.count_nonzero(currents[np.setdiff1d(np.arange(N),self.indices)]))}

    def export(self, packet, outdir):
        """Archive one shared packet and exact float32 feature vector for both consumers."""
        features=self.codec.encode(packet);outdir=Path(outdir).resolve()
        if not outdir.is_relative_to(ROOT/'artifacts'):raise ValueError('Shared body data must be archived in project artifacts')
        outdir.mkdir(parents=True,exist_ok=True)
        paths={name:outdir/name for name in ('features.npz','body_packet.json','feature_layout.json')}
        if any(path.exists() for path in paths.values()):raise FileExistsError('Use a fresh body-sensory event directory')
        paths['body_packet.json'].write_text(json.dumps(packet,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
        paths['feature_layout.json'].write_text(json.dumps(self.codec.identity,indent=2,allow_nan=False)+'\n')
        np.savez(paths['features.npz'],features=features)
        return {'schema':1,'path':str(paths['features.npz']),'sha256':hashlib.sha256(paths['features.npz'].read_bytes()).hexdigest(),
            'layout_sha256':self.codec.identity['feature_layout_sha256'],'shape':[self.codec.feature_dim],'dtype':'float32',
            'simulation_time_s':packet['simulation_time_s'],'packet_path':str(paths['body_packet.json']),
            'packet_sha256':packet['packet_sha256'],'packet_file_sha256':hashlib.sha256(paths['body_packet.json'].read_bytes()).hexdigest(),
            'layout_path':str(paths['feature_layout.json']),'features_sha256':array_sha(features)}

    def prepare(self, packet, outdir):
        features_file=self.export(packet,outdir)
        prediction=self.predict(packet)
        return {**prediction,'body_features_file':features_file}

    def fit_supervised(self, packets, target_currents_mv, *, label_source, steps=2, learning_rate=1e-3, progress_callback=None):
        features=np.stack([self.codec.encode(p) for p in packets]);target=np.asarray(target_currents_mv,dtype=np.float32)
        if target.shape!=(len(features),SENSORY_N) or not np.isfinite(target).all() or np.any(np.abs(target)>self.current_scale_mv):raise ValueError('Explicit teaching currents need one bounded value for each of the 6370 real target IDs')
        if not isinstance(label_source,str) or not label_source.strip() or type(steps)is not int or not 1<=steps<=10000:raise ValueError('Explicit teaching provenance and 1..10000 integer steps required')
        if not np.isfinite(learning_rate) or not 0<learning_rate<=.1:raise ValueError('Invalid learning rate')
        if progress_callback is not None and not callable(progress_callback):raise ValueError('progress_callback must be callable or None')
        device=next(self.parameters()).device;x=torch.from_numpy(features).to(device);y=torch.from_numpy(target).to(device)
        initial=torch.cat([p.detach().cpu().reshape(-1) for p in self.parameters()]).clone();initial_sha=self.adapter_sha256()
        optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate);losses=[];interrupted=False
        for _ in range(steps):
            optimizer.zero_grad();loss=torch.mean((self(x)-y)**2);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(self.parameters(),1.0);optimizer.step()
            losses.append(float(loss.detach()));self.training_steps+=1
            if progress_callback is not None:
                current=torch.cat([p.detach().cpu().reshape(-1) for p in self.parameters()])
                if progress_callback({'step':len(losses),'optimizer_updates':self.training_steps,'loss':losses[-1],
                    'grad_norm':float(gradient),'changed_parameters':int(torch.count_nonzero(current!=initial)),
                    'parameter_l2_change':float(torch.linalg.vector_norm(current-initial))}) is False:
                    interrupted=True;break
        final=torch.cat([p.detach().cpu().reshape(-1) for p in self.parameters()]);self.label_source=label_source.strip()
        with torch.no_grad():after=float(torch.mean((self(x)-y)**2))
        if not torch.isfinite(final).all():raise RuntimeError('Nonfinite body-sense adapter update')
        return {'optimizer_steps':len(losses),'requested_steps':steps,'training_steps':self.training_steps,'interrupted':interrupted,'losses':losses,'loss_after':after,
            'changed_parameters':int(torch.count_nonzero(final!=initial)),'parameter_l2_change':float(torch.linalg.vector_norm(final-initial)),
            'initial_adapter_sha256':initial_sha,'final_adapter_sha256':self.adapter_sha256(),'label_source':self.label_source,
            'interpretation':'Supervised encoder update only; target currents are explicit engineering/human labels, not measured natural sensory responses'}

    def save(self, path, packet):
        self.codec.encode(packet);path=Path(path).resolve();path.mkdir(parents=True,exist_ok=True)
        if any(path.iterdir()):raise FileExistsError('Use an empty body-sense checkpoint directory')
        np.savez(path/'weights.npz',**{k:v.detach().cpu().numpy() for k,v in self.state_dict().items()})
        metadata={**self.describe(),'codec':self.codec.identity,'layout_packet':packet,
            'weights_sha256':hashlib.sha256((path/'weights.npz').read_bytes()).hexdigest()}
        (path/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
        return {'path':str(path),'adapter_sha256':self.adapter_sha256()}

    @classmethod
    def load(cls,path):
        path=Path(path);meta=json.loads((path/'metadata.json').read_text())
        if meta.get('schema')!=1 or meta.get('component')!='body_senses_to_neurons':raise ValueError('Invalid body-sense checkpoint')
        if hashlib.sha256((path/'weights.npz').read_bytes()).hexdigest()!=meta['weights_sha256']:raise ValueError('Body-sense weight integrity mismatch')
        model=cls(meta['layout_packet'],hidden_width=meta['hidden_width'],current_scale_mv=meta['current_scale_mv'])
        if model.codec.identity!=meta['codec'] or model.target_ids_sha256!=meta['target_ids_sha256']:raise ValueError('Body or sensory target identity differs from checkpoint')
        with np.load(path/'weights.npz',allow_pickle=False) as data:
            expected=model.state_dict()
            if set(data.files)!=set(expected):raise ValueError('Body-sense checkpoint parameter names differ')
            state={}
            for key,value in expected.items():
                a=data[key]
                if a.dtype!=np.float32 or a.shape!=tuple(value.shape) or not np.isfinite(a).all():raise ValueError('Invalid body-sense checkpoint tensor')
                state[key]=torch.from_numpy(a.copy())
        model.load_state_dict(state);model.training_steps=meta['training_steps'];model.label_source=meta['label_source']
        if model.adapter_sha256()!=meta['adapter_sha256']:raise ValueError('Body-sense checkpoint reload identity differs')
        return model.eval()
