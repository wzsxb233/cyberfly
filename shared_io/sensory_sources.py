"""Persistent, separately trainable sources appended to original brain inputs."""
from pathlib import Path
import json
import uuid

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/sensory_checkpoints'

def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');temporary.replace(path)

class SharedSensorySources:
    def __init__(self):self.body={};self.modal={}
    def body_adapter(self,packet):
        from .body_senses import BodySenseCodec,BodySenseAdapter
        key=BodySenseCodec(packet).identity['feature_layout_sha256'];base=BASE/'body'/key
        pointer=base/'current.json'
        if pointer.exists():
            current=json.loads(pointer.read_text())
            if key not in self.body or self.body[key][0]!=current['path']:
                adapter=BodySenseAdapter.load(current['path'])
                if adapter.adapter_sha256()!=current['adapter_sha256']:raise ValueError('Body sensory pointer changed its adapter identity')
                self.body[key]=(current['path'],adapter)
        else:
            adapter=BodySenseAdapter(packet);saved=adapter.save(base/'versions'/uuid.uuid4().hex,packet)
            _write(pointer,saved);self.body[key]=(saved['path'],adapter)
        return self.body[key][1]
    def modal_adapter(self,modality):
        from .modal_currents import ModalCurrentAdapter
        base=BASE/modality;pointer=base/'current.json'
        if pointer.exists():
            current=json.loads(pointer.read_text())
            if modality not in self.modal or self.modal[modality][0]!=current['path']:
                adapter=ModalCurrentAdapter.load(current['path'])
                identity=adapter.describe()
                expected=current.get('identity',{})
                if adapter.modality!=modality or any(identity.get(k)!=expected.get(k) for k in ('weights_sha256','configuration_sha256','optimizer_updates')):
                    raise ValueError('Sensory checkpoint pointer changed its modality or parameter identity')
                self.modal[modality]=(current['path'],adapter)
        else:
            adapter=ModalCurrentAdapter(modality);path=base/'versions'/uuid.uuid4().hex
            adapter.save(path);_write(pointer,{'path':str(path),'identity':adapter.describe()});self.modal[modality]=(str(path),adapter)
        return self.modal[modality][1]
    def prepare_body(self,packet,output):
        if packet is None:return None
        return self.body_adapter(packet).prepare(packet,output)
    def feedback_body(self,packet,output):
        if packet is None:return None
        return self.body_adapter(packet).export(packet,output)
    def project_modalities(self,encoded,event_id):
        descriptor=encoded.get('modal_embeddings_file')
        if descriptor is None:raise RuntimeError('The model service has not exported its actual original modality tensors')
        identity={key:encoded[key] for key in ('model_revision','weights_manifest_sha256')}
        return {name:self.modal_adapter(name).project(descriptor,event_id=event_id,model_identity=identity) for name in ('vision','audio')}
    def save_body_training(self,adapter,packet):
        base=BASE/'body'/adapter.codec.identity['feature_layout_sha256']
        saved=adapter.save(base/'versions'/uuid.uuid4().hex,packet);_write(base/'current.json',saved)
        self.body.pop(adapter.codec.identity['feature_layout_sha256'],None)
        return saved
    def save_modal_training(self,adapter):
        base=BASE/adapter.modality;path=base/'versions'/uuid.uuid4().hex
        adapter.save(path);result={'path':str(path),'identity':adapter.describe()};_write(base/'current.json',result)
        self.modal.pop(adapter.modality,None)
        return result
