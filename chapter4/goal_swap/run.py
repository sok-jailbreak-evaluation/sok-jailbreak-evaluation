import argparse
import asyncio
from study_common import ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('lane',choices=['prepare','api','strongreject','harmbench','report'])
    p.add_argument('--phase',choices=['preflight','full'],default='preflight')
    args=p.parse_args()
    if args.lane=='prepare':
        from prepare import main as prepare
        prepare()
    elif args.lane=='report':
        from report import build
        build()
    elif args.lane=='api':
        from api import run
        asyncio.run(run(args.phase))
    else:
        from local import run
        run(args.lane,args.phase)

if __name__=='__main__': main()
