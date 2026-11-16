"""Compatibility shim for the actual formal native-stopping suffix worker."""
import os
from pathlib import Path
import shutil
import sys
import json
import inc
from inc import ROOT, REPO, OUT, read, write, sha, lock, data, PYTHON

REFERENCE=ROOT/'reference'
NAME=os.environ.get('INC_SUFFIX_MODEL','llamaguard')
ALL=read(ROOT/'config.json')
CONF={'order':[NAME],'models':{NAME:dict(ALL['suffix_models'][NAME])}}
if NAME=='strongreject':
    threshold=float(os.environ.get('STRONGREJECT_THRESHOLD','0.5'))
    assert threshold in (0.5,0.75,0.9)
    CONF['models'][NAME].update(threshold=threshold,primary_threshold=threshold)
CONF['models'][NAME]['gpus']=[0]
CONF['models'][NAME]['workers']=1


def normalized_gpu_uuid(value):return str(value).lower().removeprefix('gpu-')
def search_sample(row,position):return inc.sample(row,NAME)


def prepare():
    import torch
    key=sha({'entry':inc.key(NAME),'model':CONF['models'][NAME]})
    rows=data()
    free,_=torch.cuda.mem_get_info()
    CONF['models'][NAME]['max_gpu_memory_gib']=min(64,max(1,(free-3*1024**3)/1024**3))
    return key,rows
