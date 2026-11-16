"""Count task-condition passes before and after verification on the same selected slots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

def summarize(path):
    counts=defaultdict(lambda: [0,0,0])
    seen=set()
    for line in path.read_text().splitlines():
        row=json.loads(line)
        target,method,_=row['id'].split('|')
        if row['id'] in seen:
            raise ValueError('Duplicate attempt')
        seen.add(row['id'])
        for name in (method,'ALL'):
            if method=='ALL':
                raise ValueError('Reserved group label')
            group=counts[target,name]
            group[0]+=1
            group[1]+=row['task_before_verification']
            group[2]+=row['task_pass']
    return counts

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('records',type=Path)
    output=csv.writer(sys.stdout)
    output.writerow(['target','method','slots','task_before_verification','task_after_verification'])
    for key,counts in sorted(summarize(parser.parse_args().records).items()):
        output.writerow([*key,*counts])
