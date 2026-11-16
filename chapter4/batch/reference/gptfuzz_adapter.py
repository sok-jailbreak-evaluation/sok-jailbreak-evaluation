"""Response-only GPTFuzzer classifier, plugged into the unchanged legacy search."""
from __future__ import annotations
from collections import Counter
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,RobertaTokenizer,RobertaForSequenceClassification,RobertaConfig
from common import ROOT as LEGACY, read_json as read
if str(LEGACY) not in sys.path:sys.path.insert(0,str(LEGACY))
from adapters import EvaluatorAdapter


def compatibility_config(path):
    raw=read(__import__('pathlib').Path(path)/'config.json')
    raw['id2label']={int(k):str(v) for k,v in raw['id2label'].items()}
    raw['label2id']={str(k):int(v) for k,v in raw['label2id'].items()}
    return RobertaConfig.from_dict(raw)


def load_tokenizers(path):
    fast=AutoTokenizer.from_pretrained(path,config=compatibility_config(path),local_files_only=True,use_fast=True,trust_remote_code=False)
    native=RobertaTokenizer.from_pretrained(path,local_files_only=True)
    return fast,native


class GPTFuzzAdapter(EvaluatorAdapter):
    name='gptfuzz'
    protocol_id='gptfuzz/official_response_only/roberta_argmax_512/v1'
    supports_prefix_kv=False
    prefix_kv_unsupported_reason='Bidirectional RoBERTa representations depend on the entire response'
    add_special_tokens=True
    def __init__(self,config,device='cuda:0'):
        self.device=torch.device(device)
        self.tokenizer,self.native_tokenizer=load_tokenizers(config['model'])
        self.model=RobertaForSequenceClassification.from_pretrained(config['model'],local_files_only=True,
                   config=compatibility_config(config['model']),dtype=torch.float32,attn_implementation='eager').to(self.device)
        self.model.eval().requires_grad_(False)
        assert self.model.config.num_labels==2 and self.model.config.hidden_size==1024
        self.context_limit=512
        self._disallowed,self.oom_batch_reductions={},0
        self.last_effective_batch_size=None
        self.counts=Counter()
        self.model.register_forward_pre_hook(lambda *args:self.counts.update(model_forwards=1))

    def parts(self,sample):
        return sample['response'],'',0

    def last_logits(self,input_ids=None,attention_mask=None,inputs_embeds=None,full_logits=False):
        # Explicit positions also cover the differentiable inputs_embeds path.
        # Equal-length unpadded batches have the native RoBERTa positions 2..n+1.
        positions=attention_mask.long().cumsum(-1)*attention_mask.long()+self.model.config.pad_token_id
        return self.model(input_ids=input_ids,attention_mask=attention_mask,inputs_embeds=inputs_embeds,
                          position_ids=positions).logits

    def metrics_tensor(self,logits):
        p=logits.float().softmax(-1)
        return {'score':p[:,1],'p_jailbroken':p[:,1],'p_not_jailbroken':p[:,0]}

    def loss_tensor(self,logits,direction):
        return -logits.float().log_softmax(-1)[:,1 if direction=='up' else 0]

    def score_ids(self,sequences,direction,batch_size):
        self.counts.update(score_sequences=len(sequences),score_api_calls=1)
        return super().score_ids(sequences,direction,batch_size)

    def gradient(self,layout,direction):
        self.counts.update(gradient_calls=1)
        return super().gradient(layout,direction)

    def native_check(self,sample,suffix,direction,tolerance=1e-6):
        text=sample['response']+suffix
        complete=self.native_tokenizer(text,truncation=False)['input_ids']
        if len(complete)>512:raise ValueError('Native 512-token truncation would remove fixed text or suffix; sample/candidate incompatible')
        encoded=self.native_tokenizer([text],padding=True,truncation=True,max_length=512,return_tensors='pt').to(self.device)
        search_ids=self.encode(sample,suffix)
        assert encoded['input_ids'][0].tolist()==complete==search_ids,'Native/fast token IDs differ'
        with torch.no_grad():logits=self.model(**encoded).logits
        score=logits.float().softmax(-1)[0,1].item()
        label=logits.argmax(-1).item()
        self.counts.update(native_checks=1,native_sequences=1)
        fast=self.score_ids([search_ids],direction,1)[0]
        error=abs(score-fast['score'])
        if error>tolerance:raise AssertionError(f'Native/search score mismatch: {error}')
        if int(fast['p_jailbroken']>fast['p_not_jailbroken'])!=label:raise AssertionError('Native/search label mismatch')
        return {'score':score,'p_jailbroken':score,'p_not_jailbroken':1-score,'native_label':label,
                'hit':label==(1 if direction=='up' else 0),'parse_ok':True,'passed':True,
                'native_label_parse':'argmax of 2 logits; class 1 = GPTFuzzer jailbreak success',
                'logits':logits[0].float().cpu().tolist(),'input_ids':search_ids,
                'input_tokens':len(search_ids),'input_text':text,'input_truncated':False,
                'native_score_difference':error,'protocol_id':self.protocol_id}

    def native_stop(self,sample,suffix,direction):
        return self.native_check(sample,suffix,direction)

    def metadata(self):
        return {**super().metadata(),'logits_to_keep':None,'input_object':'response only; original goal is NOT fed to judge',
                'native_tokenizer_class':type(self.native_tokenizer).__name__, 'jailbroken_class':1,
                'source_max_length':512,'selection_policy':'complete inputs <=512; no truncation actually applied',
                'precision':'float32; no quantization','attention':'eager for explicit reference consistency'}
