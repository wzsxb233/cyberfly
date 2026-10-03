"""CPU protocol checks; these do not stand in for the archived 64-step GPU run."""
import unittest
from types import SimpleNamespace

from shared_io.language_adapter import LanguageAdapter,evaluation_splits
from shared_io.model_link import ModelLink


class EvaluationPolicyTests(unittest.TestCase):
    def test_historical_evaluation_default_is_unchanged(self):
        self.assertEqual(evaluation_splits({},{}),['validation','test'])

    def test_online_validation_is_explicit(self):
        self.assertEqual(evaluation_splits({'evaluation_splits':['validation']},{'online_learning':True}),['validation'])
        for request in ({},{'evaluation_splits':['test']},{'evaluation_splits':['validation','test']}):
            with self.subTest(request=request),self.assertRaises(ValueError):
                evaluation_splits(request,{'online_learning':True})

    def test_invalid_split_requests_fail_closed(self):
        for value in ([],['train'],['validation','validation'],'validation',[None]):
            with self.subTest(value=value),self.assertRaises(ValueError):
                evaluation_splits({'evaluation_splits':value},{})


class GenerationBudgetTests(unittest.TestCase):
    def test_zero_budget_skips_language_and_speech_calls(self):
        # No llm or speech object exists: touching either would fail this test.
        host=SimpleNamespace(torch=object())
        say,info=ModelLink._generate_reply(host,None,{'max_new_tokens':0})
        self.assertEqual(say,'');self.assertTrue(info['generation_skipped'])

    def test_zero_budget_cannot_claim_speech(self):
        with self.assertRaises(ValueError):
            ModelLink._generate_reply(SimpleNamespace(torch=object()),None,{'max_new_tokens':0,'generate_audio':True})

    def test_invalid_budget_rejected_before_inference(self):
        for value in (-1,257,True,12.5,float('nan'),'12'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                ModelLink._generate_reply(SimpleNamespace(torch=object()),None,{'max_new_tokens':value})


class FrozenOriginalTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch=torch
        self.model=torch.nn.Linear(3,2).requires_grad_(False)
        self.adapter=LanguageAdapter(SimpleNamespace(torch=torch,model=self.model))
        self.adapter.verify_frozen()

    def test_original_accidentally_trainable_is_rejected(self):
        self.model.weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):self.adapter.verify_frozen()

    def test_inplace_original_update_is_rejected(self):
        with self.torch.no_grad():self.model.weight.add_(1)
        with self.assertRaises(RuntimeError):self.adapter.verify_frozen()

    def test_original_storage_replacement_is_rejected(self):
        self.model.weight.data=self.model.weight.data.clone()
        with self.assertRaises(RuntimeError):self.adapter.verify_frozen()


if __name__=='__main__':unittest.main()
