"""Unregistered candidate mixin: no running model acquires LoRA by importing it.

It must be explicitly combined with the verified thinker plugin in a separate
candidate architecture and then tested on a GPU before any live deployment.
"""
from __future__ import annotations


class OriginalLanguageLoRAMixin:
    supports_lora=True
    supports_tower_connector_lora=False
    packed_modules_mapping={'qkv_proj':['q_proj','k_proj','v_proj'],
        'gate_up_proj':['gate_proj','up_proj']}
    embedding_modules={'embed_tokens':'input_embeddings','lm_head':'output_embeddings'}
    language_lora_capability_status='unregistered_candidate_not_gpu_validated'

    def __init__(self,*,vllm_config,prefix=''):
        super().__init__(vllm_config=vllm_config,prefix=prefix)
        if self.model_stage!='llm':raise ValueError('This candidate supports original language LoRA only')
        if self.model is not self.thinker:raise ValueError('Original thinker/model alias identity changed')
        config=vllm_config.lora_config
        if config is not None:
            if config.enable_tower_connector_lora:raise ValueError('Vision, audio and native speech foundations remain frozen')
            if set(config.target_modules or ())!={'q_proj','v_proj'}:
                raise ValueError('First candidate is explicitly limited to original language q/v LoRA')
        self.verify_language_module_paths()

    def get_mm_mapping(self):
        from vllm.model_executor.models.module_mapping import MultiModelKeys
        return MultiModelKeys.from_string_field(language_model='thinker.llm',
            tower_model=['thinker.vpm','thinker.apm'],
            connector=['thinker.resampler','thinker.audio_projection_layer'])

    def verify_language_module_paths(self):
        """Canonical paths must alias the exact original language projections."""
        actual=[]
        for layer in range(36):
            name=f'thinker.llm.model.layers.{layer}.self_attn.qkv_proj'
            canonical=self.get_submodule(name)
            alias=self.get_submodule(name.replace('thinker.','model.',1))
            if canonical is not alias:raise ValueError('Language alias would route through another projection')
            actual.append(name)
        return {'canonical_packed_qkv_modules':actual,'module_aliases_identical':True,
            'tower_connector_lora':False,'native_tts_lora':False,
            'status':self.language_lora_capability_status}
