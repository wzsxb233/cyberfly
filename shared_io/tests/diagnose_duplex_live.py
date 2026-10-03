"""Real-model diagnostics with recorded brain/body fixtures, not a new brain run."""
from pathlib import Path
import argparse,json,urllib.request,urllib.error,uuid

ROOT=Path(__file__).resolve().parents[2]
def call(route,value):
    req=urllib.request.Request('http://127.0.0.1:18647'+route,data=json.dumps(value,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=300) as response:return json.load(response)
    except urllib.error.HTTPError as error:raise RuntimeError(error.read().decode()) from error

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',choices=['baseline_av','baseline_text','fusion_av','anchor_av','boundary_av','baseline_sampling','anchor_sampling','boundary_sampling'],required=True);p.add_argument('--units',type=int,default=8);p.add_argument('--facts',action='store_true');p.add_argument('--ablation',action='store_true');p.add_argument('--placement',choices=['before_sensory','after_sensory'],default='after_sensory');p.add_argument('--source-directory',type=Path);a=p.parse_args()
    source=a.source_directory or ROOT/'artifacts/full_duplex_checks/105130a2fd2b'
    fixtures=sorted((json.loads(x.read_text()) for x in source.glob('original-event-*.json')),key=lambda x:x['sequence'])
    if not fixtures:
        progress=json.loads((source/'progress.json').read_text())
        fixtures=[json.loads((ROOT/'artifacts/shared_io'/r['event_id']/'event.json').read_text()) for r in progress['units']]
    out=ROOT/'artifacts/duplex_diagnostics'/f'{a.case}-{a.placement}-{uuid.uuid4().hex[:12]}';out.mkdir(parents=True)
    session={'session_id':out.name,'connection_mode':'neuron_direct','io_scope':'sensory_motor',
        'diagnostic_baseline':a.case.startswith('baseline'),'decision_anchor':'last_sensory' if a.case.startswith('anchor') else 'native_boundary' if a.case.startswith('boundary') else 'none',
        'decode_mode':'sampling' if a.case.endswith('sampling') else 'greedy','system_prompt':'Streaming Omni Conversation.',
        'brain_state':fixtures[0]['brain_state_before'],'numeric_placement':a.placement}
    opened=call('/duplex/open',session);session.update(epoch=opened['epoch']);results=[]
    (out/'open.json').write_text(json.dumps(opened,indent=2,ensure_ascii=False))
    try:
        for seq,x in enumerate(fixtures[:a.units]):
            req={**session,'sequence':seq,'event_id':uuid.uuid4().hex,'text':x.get('text',''),
                'brain_state':x['brain_state_before'],'body_features_file':x['body_features_file']}
            if a.case=='baseline_text':req['text']='你好，请用中文说一句问候。' if seq==0 else '请继续。'
            else:
                req['image_path']=x['image_path']
                if x.get('audio_path'):req['audio_path']=x['audio_path']
            pre=call('/duplex/prefill',req)
            feedback={**session,'sequence':seq,'event_id':req['event_id'],
                'brain_state':x['brain_state_after'],'body_feedback_file':x['body_feedback_file'],'generate_audio':True}
            if a.ablation:feedback['diagnostic_brain_ablation']=True
            if a.facts:
                from shared_io.measurements import compact_measurements
                feedback['facts']=compact_measurements({'neural':x['actual_neural'],'stimulus':x['actual_stimulation']},x.get('body_feedback'),x.get('body_state'))
            gen=call('/duplex/generate',feedback)
            record={'fixture_event_id':x['event_id'],'prefill':pre,'generate':gen};results.append(record)
            (out/f'{seq:04d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False))
            print(json.dumps({'case':a.case,'seq':seq,'say':gen['say'],'is_listen':gen['is_listen'],'n_tokens':gen['n_tokens'],
                'raw_tokens':gen['decision_trace']['actual_native_tokens'],'audio':bool(gen['audio_file']),
                'before':pre['decision_trace']['original_pending'][:2],'after':gen['decision_trace'].get('after_anchor',gen['decision_trace'].get('after_brain_body_feedback',[]))[:2]},ensure_ascii=False),flush=True)
    finally:closed=call('/duplex/close',session)
    summary={'status':'diagnostic_completed','case':a.case,'source':'Actual MiniCPM inference; real recorded brain/body snapshots from prior run. New currents were not applied to these historical snapshots; this is NOT a new closed loop.',
        'fixture_directory':str(source),'output':str(out),'open':opened,'close':closed,
        'semantic_units':sum(bool(r['generate']['say']) and r['generate']['n_tokens']>0 for r in results),
        'audible_semantic_units':sum(bool(r['generate']['audio_file']) for r in results)}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
