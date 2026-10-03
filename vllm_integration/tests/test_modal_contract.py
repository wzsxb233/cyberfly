"""Parser fixtures and real archived HF-token/head compatibility, no vLLM inference."""
import json
import unittest
from pathlib import Path
from vllm_integration.latent_contract import ROOT,LatentCoupler
from vllm_integration.modal_export import modal_token_positions,ScheduledModalCapture,ModalCaptureBridge


class ModalPositionTests(unittest.TestCase):
    def test_original_mask_and_bounds_select_only_encoder_rows(self):
        ids=[872,151669,128244,128244,151670,151697,128244,151699,151662]
        mask=[False,False,True,True,False,False,True,False,False]
        positions,active=modal_token_positions(ids,mask)
        self.assertEqual(positions,{'vision':[2,3],'audio':[6]});self.assertIsNone(active)

    def test_printed_placeholder_and_unbounded_row_rejected(self):
        for ids,mask in [([151669,128244,151670],[False,False,False]),([128244],[True]),
                         ([151697,128244,151670],[False,True,False])]:
            with self.assertRaises(ValueError):modal_token_positions(ids,mask)

    def test_real_chunk_offsets_keep_original_rows_and_do_not_mutate_input(self):
        import torch
        capture=ScheduledModalCapture();emb=torch.arange(7*4096,dtype=torch.float32).reshape(7,4096)
        old=emb.clone()
        self.assertIsNone(capture.consume('fixture',0,[872,151669,128244],[False,False,True],emb[:3],complete=False))
        out=capture.consume('fixture',3,[128244,151670,151697,128244],[True,False,False,True],emb[3:],complete=False)
        self.assertIsNone(out)
        last=torch.zeros((1,4096))
        result=capture.consume('fixture',7,[151699],[False],last,complete=True)
        self.assertEqual(result['actual_input_positions'],{'vision':[2,3],'audio':[6]})
        self.assertTrue(torch.equal(emb,old))
        self.assertTrue((result['arrays']['vision']==emb[[2,3]].numpy()).all())

    def test_bridge_requires_the_actual_same_scheduled_ids(self):
        import torch
        bridge=ModalCaptureBridge();ids=torch.tensor([151669,128244,151670]);emb=torch.zeros((3,4096))
        metadata={'request_id':'protocol_fixture','_omni_is_prefill':True,'_omni_num_computed_tokens':0,'_omni_prompt_len':3}
        bridge.record_mask(ids,torch.tensor([False,True,False]))
        wrong=ids.clone();wrong[1]=42
        with self.assertRaises(ValueError):bridge.capture(wrong,emb,metadata)
        bridge.record_mask(ids,torch.tensor([False,True,False]))
        result=bridge.capture(ids,emb,metadata)
        self.assertEqual(result['actual_input_positions'],{'vision':[1],'audio':[]})


class ActualSavedHeadTests(unittest.TestCase):
    def test_existing_actual_modal_tensors_load_both_saved_heads(self):
        import torch
        torch.set_num_threads(1)
        record=json.loads((ROOT/'datasets/latent_grounding/sixseeds_v2/manifest.json').read_text())['records'][0]
        coupler=LatentCoupler(record['input']['brain_state_after'])
        source=json.loads((ROOT/'artifacts/modal_currents_checks/db2eddb03e65/evidence.json').read_text())['modal_embeddings_file']
        result=coupler.modal_currents(source,event_id=source['event_id'])
        for name,count,prefix in [('vision',6098,'3940f1ba'),('audio',6370,'d1d1605b')]:
            self.assertTrue(result[name]['active']);self.assertEqual(result[name]['selected_neurons'],count)
            self.assertEqual(result[name]['outside_selection_nonzero'],0)
            self.assertTrue(result[name]['adapter_identity']['weights_sha256'].startswith(prefix))
            self.assertEqual(result[name]['adapter_identity']['optimizer_updates'],2)
        # This verifies migration of existing heads using an archived actual HF
        # encoder output. It does not claim a new vLLM model request occurred.


if __name__=='__main__':unittest.main()
