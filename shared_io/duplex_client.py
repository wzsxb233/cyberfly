"""Transport for the original stateful MiniCPM audio/video stream, with brain turns."""
import uuid
from .bus import SharedLinkClient

class DuplexLinkClient(SharedLinkClient):
    is_duplex=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.session_id=uuid.uuid4().hex;self.epoch=None;self.sequence=0;self.failed_close=None
    def call(self,operation,payload):
        if operation=='encode':
            if self.failed_close is not None:self.close()
            if self.epoch is None:
                opened=super().call('duplex/open',{'session_id':self.session_id,'connection_mode':'neuron_direct',
                    'io_scope':payload['io_scope'],'brain_state':payload['brain_state'],
                    'system_prompt':'Streaming Omni Conversation.','decision_anchor':'native_boundary',
                    'decode_mode':'greedy','diagnostic_baseline':False,'numeric_placement':'before_sensory'})
                self.epoch=opened['epoch'];self.sequence=0
            return super().call('duplex/prefill',{**payload,'session_id':self.session_id,'epoch':self.epoch,'sequence':self.sequence})
        if operation=='decode':
            if self.epoch is None:raise RuntimeError('Native stream was closed before the brain returned')
            result=super().call('duplex/generate',{**payload,'session_id':self.session_id,'epoch':self.epoch,'sequence':self.sequence})
            self.sequence+=1
            return result
        return super().call(operation,payload)
    def close(self):
        owner={'session_id':self.session_id,'epoch':self.epoch} if self.epoch is not None else self.failed_close
        if owner is None:return
        timeout=self.timeout
        try:
            self.timeout=min(timeout,10)
            super().call('duplex/close',owner)
            self.failed_close=None
        except Exception:
            self.failed_close=owner
            raise
        finally:
            self.timeout=timeout;self.epoch=None;self.sequence=0
