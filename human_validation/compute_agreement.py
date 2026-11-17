"""Reproduce nominal agreement from labels.csv using Python's standard library."""
from pathlib import Path
from collections import Counter
from itertools import combinations
from fractions import Fraction as F
import argparse,csv,json

CATEGORIES={'H':['是','否','无法判断'],'A':['无','部分','完整','无法判断'],'R':['相关','部分相关','无关','无法判断']}
def pair(x,y,cats):
    n=len(x); m=Counter(zip(x,y)); hit=sum(m[c,c] for c in cats)
    po=F(hit,n); a=Counter(x); b=Counter(y)
    pk=sum(F(a[c]*b[c],n*n) for c in cats)
    p=[F(a[c]+b[c],2*n) for c in cats]
    pe=sum(v*(1-v) for v in p)/(len(cats)-1)
    return dict(n=n,matches=hit,agreement=float(po),kappa=float((po-pk)/(1-pk)),AC1=float((po-pe)/(1-pe)),categories=cats,matrix=[[m[c,d] for d in cats] for c in cats])

def compute(path):
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==len({r['record'] for r in rows})==300
    for r in rows:
        for w in ['AA','BB','CC','evaluator']:
            assert all(r[w+'_'+k] in cats for k,cats in CATEGORIES.items())
    pairs=[];multi=[];majority=[]
    for a,b in [('AA','evaluator'),('BB','evaluator'),('CC','evaluator'),*combinations(['AA','BB','CC'],2)]:
        for k,cats in CATEGORIES.items():
            pairs.append(dict(left=a,right=b,aspect=k,**pair([r[a+'_'+k] for r in rows],[r[b+'_'+k] for r in rows],cats)))
    for k,cats in CATEGORIES.items():
        total=Counter();po=F(0);unanimous=0;tie=[];x=[];y=[]
        for r in rows:
            ct=Counter(r[w+'_'+k] for w in ['AA','BB','CC']);total.update(ct)
            label,count=ct.most_common(1)[0]
            calculated=label if count>=2 else 'no_majority'
            assert calculated==r['majority_'+k]
            unanimous+=count==3
            po+=sum(F(c*(c-1),6) for c in ct.values())/len(rows)
            if count>=2:x.append(label);y.append(r['evaluator_'+k])
            else:tie.append(r['record'])
        p=[F(total[c],3*len(rows)) for c in cats];pk=sum(v*v for v in p)
        pe=sum(v*(1-v) for v in p)/(len(cats)-1)
        multi.append(dict(aspect=k,n=len(rows),unanimous=unanimous,unanimity=unanimous/len(rows),mean_pairwise_agreement=float(po),fleiss_kappa=float((po-pk)/(1-pk)),AC1=float((po-pe)/(1-pe))))
        majority.append(dict(aspect=k,no_majority=tie,**pair(x,y,cats)))
    return dict(category_sets=CATEGORIES,pairwise=pairs,multirater=multi,majority_vs_evaluator=majority)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels',type=Path,default=Path(__file__).with_name('labels.csv'))
    p.add_argument('--output',type=Path)
    a=p.parse_args();s=json.dumps(compute(a.labels),ensure_ascii=False,indent=2)+'\n'
    if a.output:a.output.write_text(s,encoding='utf-8')
    else:print(s,end='')
