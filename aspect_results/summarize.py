"""Reproduce 5-aspect counts from metadata-only response records."""
from collections import Counter,defaultdict
from pathlib import Path
import csv,json

def summarize(rows):
    counts=defaultdict(Counter);statuses=defaultdict(Counter);ids=set()
    for row in rows:
        assert row['id'] not in ids
        ids.add(row['id']);target,method,goal=row['id'].split('|');key=target,method
        counts[key]['n']+=1
        for col in ['G','H','R','A','D']:
            assert row[col] in (0,1)
            counts[key][col]+=row[col]
        for col in ['g_status','d_status','R_status','A_status']:
            if col in row:statuses[key][col+'='+row[col]]+=1
    totals=Counter()
    for value in counts.values():totals.update(value)
    return dict(aspects=dict(G='L1.2',H='L2.2',R='L2.3',A='L2.4',D='L3.1'),
                groups=[dict(target=k[0],method=k[1],**v,statuses=dict(statuses[k])) for k,v in sorted(counts.items())],totals=dict(totals))

if __name__=='__main__':
    p=Path(__file__).resolve().parent
    report=summarize([json.loads(s) for s in (p/'records.jsonl').read_text().splitlines()])
    (p/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (p/'summary.csv').open('w',newline='') as f:
        fields=['target','method','n','G','H','R','A','D'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for group in report['groups']:w.writerow({k:group[k] for k in fields})
    print('Groups:',len(report['groups']),'Totals:',report['totals'])
