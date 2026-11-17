"""Portable, resumable entry point. Inference and label evaluation are separate."""
import argparse,collections,hashlib,os
from pathlib import Path
import pipeline as p
import support
from evaluator import evaluate

def validate_stages(out,inv,route,batch,verified=False):
    candidate=collections.defaultdict(set)
    for directory in {d for q in route['queries'] for d in q['directories']}:
        rows=[r for r in inv['images'] if r['directory']==directory]
        prefix=hashlib.sha256(directory.encode()).hexdigest()[:8]
        for start in range(0,len(rows),batch):
            file=out/'screen'/f'{prefix}-{start:05d}.json'
            if not file.exists():raise RuntimeError('Screening is incomplete; run --stage screen before continuing')
            obj=p.read_json(file)
            if {r['id'] for r in rows[start:start+batch]}!=set(obj['tested_ids'])|{r['image_id'] for r in obj['errors']}:
                raise RuntimeError('Screening image coverage mismatch')
            for row in obj['candidates']:candidate[row['image_id']].update(row['case_ids'])
    if verified:
        for image,queries in candidate.items():
            if not queries:continue
            file=out/'verify'/f'{image}.json'
            if not file.exists() or {j['case_id'] for j in p.read_json(file)['judgments']}!=queries:
                raise RuntimeError('Verification is incomplete; run --stage verify before continuing')

def parser():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',type=Path,default=p.ROOT);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--offset',type=int,default=0);a.add_argument('--limit',type=int,default=10)
    a.add_argument('--workers',type=int,default=8);a.add_argument('--batch',type=int,default=12)
    a.add_argument('--max-calls',type=int,default=2000);a.add_argument('--key-file',type=Path)
    a.add_argument('--model',default=os.environ.get('DEEPSEEK_MODEL','deepseek-flash'))
    a.add_argument('--stage',choices=['run','prepare','screen','verify','export','evaluate'],default='run')
    return a

def run(args,api_factory=None):
    root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.offset<0 or args.limit<=0 or args.workers<=0 or args.batch<=0:raise ValueError('Invalid scope or worker/batch count')
    sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('E2E_1321cases_1w_data_v19.xlsx','ES_byFileType_user10_20261117170358.xlsx')}
    signature={'sources':sources,'offset':args.offset,'limit':args.limit,'model':args.model,'batch':args.batch,
               'base_url':os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com').rstrip('/')}
    manifest=out/'run_manifest.json'
    if manifest.exists() and p.read_json(manifest)!=signature:raise ValueError('Inputs or inference configuration changed; use a new --out directory')
    if not manifest.exists() and any((out/n).exists() for n in ('inventory.json','predictions.json','plan.json')):
        raise ValueError('Unmanaged historical output; use a new --out directory')
    p.write_json(manifest,signature)
    inv=p.read_json(out/'inventory.json') if (out/'inventory.json').exists() else p.inventory(root,out)
    actual={str(f.relative_to(root/'图片')) for f in (root/'图片').rglob('*') if f.is_file() and f.suffix.lower() in p.PHOTO}
    if actual!={r['path'] for r in inv['images']}:raise ValueError('Image list changed; use a new --out directory')
    for row in inv['images']:
        stat=(root/'图片'/row['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(row['size'],row['mtime_ns']):raise ValueError('Image inventory changed; use a new --out directory')
    cases=inv['cases'][args.offset:args.offset+args.limit]
    if len(cases)!=args.limit:raise ValueError('Requested scope exceeds available cases')
    if not (out/'corpus_metadata.json').exists():support.corpus(root,out)
    if args.stage in ('evaluate','export'):
        route=p.read_json(out/'plan.json');validate_stages(out,inv,route,args.batch,verified=True)
        if args.stage=='evaluate':return evaluate(root,out)
        p.export(out,inv,cases);support.gallery(root,out);return
    key=os.environ.get('DEEPSEEK_API_KEY','').strip()
    if not key and api_factory is None:key=(args.key_file or root/'.secrets/deepseek.key').read_text(encoding='utf-8').strip()
    if api_factory is None and not key:raise ValueError('Missing API credential')
    api=(api_factory or p.API)(out,key,model=args.model,max_calls=args.max_calls)
    route=p.plan(api,inv,cases,out)
    if args.stage in ('run','screen'):p.screen(api,root,out,inv,route,args.workers,args.batch)
    if args.stage in ('run','verify'):
        validate_stages(out,inv,route,args.batch)
        p.verify(api,root,out,inv,route,args.workers)
    if args.stage in ('run','verify'):
        validate_stages(out,inv,route,args.batch,verified=True)
        p.export(out,inv,cases);support.gallery(root,out)
        p.write_json(out/'predictions_seal.json',{'sha256':hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest(),'labels_used_in_inference':False})

if __name__=='__main__':run(parser().parse_args())
