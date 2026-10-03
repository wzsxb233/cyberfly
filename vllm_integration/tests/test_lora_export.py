"""Actual saved tensors and official CPU LoRA reader/packing; no model load."""
import hashlib
import json
import unittest
from pathlib import Path
from vllm_integration.latent_contract import ROOT
from vllm_integration.lora_export import read_source,validate_export,PREFIX


class LoraExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.folder=ROOT/'artifacts/vllm_lora_exports/grounded-run64-v2-01'
        cls.proof=json.loads((cls.folder/'export-evidence.json').read_text())
        cls.state,cls.metadata,_=read_source(cls.proof['source']['checkpoint'],cls.proof['source']['training_evidence'])

    def test_all_144_original_trained_tensors_are_preserved(self):
        result=validate_export(self.folder)
        self.assertEqual(result['tensor_count'],144);self.assertTrue(result['mapped_tensors_bitexact'])
        self.assertFalse(result['deployed']);self.assertEqual(self.metadata['training_steps'],64)

    def test_changed_source_sha_is_rejected(self):
        bad={**self.proof['source']['checkpoint'],'sha256':'0'*64}
        with self.assertRaises(ValueError):read_source(bad,self.proof['source']['training_evidence'])

    def test_official_reader_and_q_no_k_v_packing_preserve_actual_deltas(self):
        import torch
        from vllm.lora.peft_helper import PEFTHelper
        from vllm.lora.lora_model import LoRAModel
        from vllm.lora.lora_weights import PackedLoRALayerWeights
        from vllm.config.lora import LoRAConfig
        helper=PEFTHelper.from_local_dir(str(self.folder),max_position_embeddings=2048)
        helper.validate_legal(LoRAConfig(max_lora_rank=8))
        loaded=LoRAModel.from_local_checkpoint(str(self.folder),{'q_proj','v_proj'},helper,
            lora_model_id=64,device='cpu',dtype=torch.float32)
        self.assertEqual(len(loaded.loras),72)
        generator=torch.Generator().manual_seed(42);x=torch.randn((2,4096),generator=generator)
        maximum=0.
        for layer in range(36):
            base=f'model.layers.{layer}.self_attn.'
            packed=PackedLoRALayerWeights.pack([loaded.loras[PREFIX+base+'q_proj'],None,loaded.loras[PREFIX+base+'v_proj']])
            self.assertIsNone(packed.lora_a[1]);self.assertIsNone(packed.lora_b[1])
            for slot,projection in [(0,'q_proj'),(2,'v_proj')]:
                a=self.state[base+projection+'.lora_A.weight'];b=self.state[base+projection+'.lora_B.weight']
                reference=((x@a.T)@b.T)*2
                actual=(x@packed.lora_a[slot].T)@packed.lora_b[slot].T
                maximum=max(maximum,float((actual-reference).abs().max()))
                torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-7)
        self.assertFalse(torch.cuda.is_initialized(),'Run these CPU checks with CUDA_VISIBLE_DEVICES empty')
        path=ROOT/'artifacts/vllm_omni_migration/lora-cpu-20260914';path.mkdir(parents=True,exist_ok=True)
        (path/'official-reader-evidence.json').write_text(json.dumps({'status':'official_cpu_lora_reader_and_packing_passed',
            'source_adapter_sha256':self.metadata['adapter_sha256'],'tensor_count':144,'loaded_modules':72,'qkv_groups':36,
            'k_adapters_present':False,'max_delta_error':maximum,'cuda_initialized':False,'model_loaded':False,
            'runtime_lora_enabled':False,'source_training_steps':64,'synthetic_matmul_input':True},indent=2))


if __name__=='__main__':unittest.main()
