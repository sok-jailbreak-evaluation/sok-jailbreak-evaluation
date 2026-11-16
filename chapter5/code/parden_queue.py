"""Hand independent, fixed contiguous batches to single-GPU PARDEN workers."""
import fcntl,json,os,subprocess,sys,time
from hashlib import sha256
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def execute():
    os.chdir(root)
    data=Path('input/extension.jsonl').read_bytes().split(b'\n')
    data=[line for line in data if line]
    assert len(data)==4200
    queue=Path('work/parden_shards');queue.mkdir(exist_ok=True)
    dest=Path('work/parden');dest.mkdir(exist_ok=True)
    size=264
    shards=[(f'{i//size:02d}',data[i:i+size]) for i in range(0,len(data),size)]
    while True:
        claimed=False
        for name,lines in shards:
            directory=queue/name;directory.mkdir(exist_ok=True)
            done=directory/'done'
            if done.exists():continue
            with (directory/'lease.lock').open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if done.exists():continue
                source=directory/'input.jsonl'; payload=b'\n'.join(lines)+b'\n'
                if source.exists(): assert source.read_bytes()==payload
                else: source.write_bytes(payload)
                with (directory/'run.log').open('a') as log:
                    log.write('GPU '+os.environ['CUDA_VISIBLE_DEVICES']+'\n');log.flush()
                    subprocess.run([sys.executable,'-Bu','code/auxiliary.py','--data',str(source),'--out',str(directory/'results'),'--model','parden'],stdout=log,stderr=subprocess.STDOUT,check=True)
                for line in lines:
                    row=json.loads(line);fn=sha256(row['attempt_id'].encode()).hexdigest()+'.json'
                    produced=directory/'results'/fn
                    assert json.loads(produced.read_text())['attempt_id']==row['attempt_id']
                    if (dest/fn).exists():assert (dest/fn).read_bytes()==produced.read_bytes()
                    else:os.link(produced,dest/fn)
                done.touch();claimed=True
        if all((queue/n/'done').exists() for n,_ in shards):return
        if not claimed:time.sleep(10)
if __name__=='__main__':execute()
