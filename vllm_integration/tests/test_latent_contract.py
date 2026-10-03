"""CPU migration checks using actual archived numeric measurements/checkpoints."""
import json
import unittest
from pathlib import Path

from vllm_integration.latent_contract import ROOT,LatentCoupler,load_condition,build_latent_content_prefix
from vllm_integration.worker_plugin import slot_overlap


class SlotTests(unittest.TestCase):
    def test_chunk_positions_cover_only_reserved_span(self):
        positions=[]
        for offset,span in [(0,5),(5,3),(8,12)]:
            overlap=slot_overlap(3,6,offset,span)
            if overlap:
                target,source=overlap
                self.assertEqual(target.stop-target.start,source.stop-source.start)
                positions.extend(range(offset+target.start,offset+target.stop))
        self.assertEqual(positions,list(range(3,9)))
        self.assertIsNone(slot_overlap(3,6,30,1))

    def test_invalid_offsets_rejected(self):
        for args in [(0,3,0,5),(0,6,-1,4),(False,6,0,4)]:
            with self.assertRaises(ValueError):slot_overlap(*args)


class ActualCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        record=json.loads((ROOT/'datasets/latent_grounding/sixseeds_v2/manifest.json').read_text())['records'][0]
        cls.input=record['input'];cls.coupler=LatentCoupler(cls.input['brain_state_after'])
        cls.condition=cls.coupler.condition('vllm-cpu-contract-actual-record',cls.input['brain_state_after'],
            body_features_file=cls.input['body_features_file'],body_feedback_file=cls.input['body_feedback_file'])

    def test_existing_heads_and_raw_numbers_produce_six_real_soft_tokens(self):
        values=load_condition(self.condition)
        self.assertEqual(values.shape,(6,4096))
        self.assertGreater(float(abs(values).max()),0)
        self.assertEqual(self.condition['coupler_sha256'],'4c8c62aaa1bc867e156700f059466d85bc209d92e04abe8f68fcc9fd2ebf75d8')
        self.assertEqual(self.condition['body_input']['body_embedding_sha256'],'4ef562c73e6cb264908d186f284f2f075ada829149ed7c277b13a579a80af2f7')

    def test_template_slots_precede_media_and_match_original_ids(self):
        result=build_latent_content_prefix(self.condition)
        descriptor=result['additional_information']['cyberfly_latent']
        self.assertEqual(descriptor['slot_token_id'],151662)
        self.assertEqual(result['content_part']['text'],'<|fim_pad|>'*6)
        self.assertEqual(descriptor['slot_start'],3)

    def test_modified_numeric_file_identity_rejected(self):
        with self.assertRaises(ValueError):load_condition({**self.condition,'sha256':'0'*64})


if __name__=='__main__':unittest.main()
