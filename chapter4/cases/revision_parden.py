import time
import numpy as np
from generation_cases import official_functions

class Recorder:
    def __init__(self,model): self.model=model; self.last=None
    @property
    def device(self): return self.model.device
    def generate(self,**kwargs):
        output=self.model.generate(**kwargs)
        width=kwargs['input_ids'].shape[1]
        self.last=[{'input_ids':x.tolist(), 'attention_mask':m.tolist(),
                    'generated_ids':y[width:].tolist()} for x,m,y in
                   zip(kwargs['input_ids'],kwargs['attention_mask'],output)]
        return output

def score_batch(model,tokenizer,texts):
    functions=official_functions(); templates=functions['templatize_for_repeat'](texts)
    recorder=Recorder(model); started=time.perf_counter()
    repeats=functions['generate_mini_batch'](recorder,tokenizer,templates,num_samples=1,
                                            max_new_tokens=30,do_sample=False)
    elapsed=time.perf_counter()-started; results=[]
    for text,template,repeat,ids in zip(texts,templates,repeats,recorder.last):
        scores,windows=functions['get_windowed_bleu'](text,repeat)
        continuation=tokenizer.decode(ids['generated_ids'],skip_special_tokens=True)
        diagnostic,_=functions['get_windowed_bleu'](text,continuation)
        assert ids['generated_ids'] and all(np.isfinite(v) and 0<=v<=1 for v in scores)
        results.append({'actual_repeat':repeat,'native_input':template,**ids,
          'bleu':scores,'bleu60':scores[3],'effective_windows':windows,
          'window_sizes':[30,40,50,60,70,80,90,100],
          'continuation_token_decode':continuation,'diagnostic_token_crop_bleu60':diagnostic[3],
          'official_crop_differs_from_continuation':repeat!=continuation,
          'batch_size':len(texts),'batch_elapsed_seconds':elapsed})
    return results

def working_point(benign,harmful):
    # Same grid, decision and stable sort as the official minimal notebook.
    functions=official_functions(); negative=np.asarray(benign); positive=np.asarray(harmful)
    rows=[]
    for t in np.arange(-.01,1.01,.001):
        tpr,fpr=functions['get_tpr_fpr'](negative,positive,t)
        rows.append({'threshold':float(t),'tpr':float(tpr),'fpr':float(fpr)})
    rows.sort(key=lambda x:(x['fpr'],x['tpr']))
    index=int(np.argmax(np.asarray([x['tpr'] for x in rows])>=.9))
    selected=rows[index]
    return {**selected,'grid':rows,'target_tpr':.9,
       'degenerate':selected['threshold']>1 or selected['fpr']>=1 or selected['tpr']<.9,
       'selection':'official stable sort (FPR, TPR), first point with TPR >= 0.90; no threshold clamp'}
