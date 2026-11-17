"""Run the frozen image-only pipeline for cases 11..100, without loading labels."""
from __future__ import annotations
import concurrent.futures as cf, hashlib, json, shutil, threading, time
from pathlib import Path
import pipeline as p

ROOT=p.ROOT
BASE=ROOT/'image-matching/runs/pilot-v1'
OUT=ROOT/'image-matching/runs/holdout-011-100-v1'
NETWORK=threading.BoundedSemaphore(16)
STATE_LOCK=threading.Lock()
STATE={}

class BoundedAPI(p.API):
    def call(self,*args,**kwargs):
        with NETWORK:return super().call(*args,**kwargs)

def status(offset,stage,**extra):
    with STATE_LOCK:
        STATE[str(offset)]={'stage':stage,**extra}
        p.write_json(OUT/'progress.json',{'updated_at':time.time(),'batches':STATE})
    print(f'cases {offset+1:03d}..{offset+10:03d}: {stage}',flush=True)

def run_batch(offset,inv,key):
    out=OUT/f'batch-{offset+1:03d}-{offset+10:03d}'
    out.mkdir(parents=True,exist_ok=True)
    for name in ('inventory.json','corpus_metadata.json'):
        if not (out/name).exists():shutil.copy2(BASE/name,out/name)
    if not (out/'thumbs').exists():
        (out/'thumbs').symlink_to(BASE/'thumbs',target_is_directory=True)
    api=BoundedAPI(out,key,max_calls=2200)
    cases=inv['cases'][offset:offset+10]
    route=p.plan(api,inv,cases,out)
    for stage,func,args in [('screen',p.screen,(api,ROOT,out,inv,route,8,12)),('verify',p.verify,(api,ROOT,out,inv,route,8))]:
        status(offset,stage)
        for attempt in range(3):
            try:func(*args);break
            except Exception as exc:
                status(offset,stage+'_retry',attempt=attempt+1,error=type(exc).__name__)
                if attempt==2:raise
    p.export(out,inv,cases)
    status(offset,'predictions_complete')
    return str(out)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    inv=p.read_json(BASE/'inventory.json')
    sources=p.read_json(BASE/'experiment.json')['sources']
    for name,sha in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError('Source workbook changed; new inventory required')
    frozen=OUT/'frozen_code';frozen.mkdir(exist_ok=True)
    hashes={}
    for name in ('pipeline.py','evaluator.py','support.py','cohort.py'):
        source=ROOT/'image-matching'/name;hashes[name]=hashlib.sha256(source.read_bytes()).hexdigest()
        target=frozen/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=hashes[name]:
            raise RuntimeError('Frozen code changed; use a new cohort directory')
        if not target.exists():shutil.copy2(source,target)
    p.write_json(OUT/'protocol.json',{'scope':[11,100],'development_cases':[1,10],'source_sha256':sources,'code_sha256':hashes,'model':'deepseek-flash','image_batch':12,'verify_batch':4,'batch_cases':10,'network_concurrency':16,'retrieval_reads_ground_truth':False,'existing_semantic_captions_used_in_vlm':False,'created_at':time.time()})
    key=(ROOT/'.secrets/deepseek.key').read_text().strip()
    failures=[]
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(run_batch,offset,inv,key):offset for offset in range(10,100,10)}
        for future in cf.as_completed(jobs):
            offset=jobs[future]
            try:future.result()
            except Exception as exc:
                status(offset,'failed',error=type(exc).__name__);failures.append(offset)
    if failures:raise RuntimeError(f'Failed offsets {failures}; successful stages remain resumable')
    p.write_json(OUT/'prediction_completion.json',{'cases':90,'batches':9,'complete':True,'completed_at':time.time()})
    print('All holdout predictions complete; labels have not been read.',flush=True)

if __name__=='__main__':main()
