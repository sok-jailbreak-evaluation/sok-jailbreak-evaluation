"""Independent down-only configuration for the frozen reference worker."""
import os
from down import ROOT,REPO,OUT,CONF as ALL,read,write,sha,lock,data,key,mapping

REFERENCE=ROOT.parent/'batch/reference'
NAME=os.environ.get('DOWN_SUFFIX_MODEL','llamaguard')
GPU=int(os.environ.get('DOWN_GPU','1'))
CONF={'order':[NAME],'gpu_uuid':os.environ.get('DOWN_GPU_UUID',''),
      'models':{NAME:dict(ALL['suffix_models'][NAME])}}
CONF['models'][NAME]['gpus']=[GPU]

def normalized_gpu_uuid(v):return str(v).lower().removeprefix('gpu-')
def search_sample(row,position):
    r=dict(row);r['source_row_index']=mapping()[row['sample_id']]['search_seed_index']
    if NAME!='harmbench' and r.get('context'):r['prompt']=r['context']+'\n\n'+r['prompt']
    return r

def prepare():
    import torch
    run_key=key(NAME);rows=data()
    free,_=torch.cuda.mem_get_info()
    cfg=CONF['models'][NAME]
    cfg['max_gpu_memory_gib']=min(64,max(1,(free-3*1024**3)/1024**3))
    return run_key,rows
