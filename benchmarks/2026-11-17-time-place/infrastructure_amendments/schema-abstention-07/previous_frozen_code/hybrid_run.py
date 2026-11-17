"""Compare frozen visual retrieval and metadata/visual fusion using the same VLM.

Inference only: this program never reads the ground-truth column or evaluates it.
"""
import argparse,collections,concurrent.futures as cf,csv,datetime,hashlib,json,os,shutil,threading,time
from pathlib import Path
import pipeline as p
import support
import run as visual
import hybrid_verify
import rich_metadata
from metadata_search import MetadataIndex,plan as metadata_plan
from providers import CompatibleAPI,RoutedAPI,SingleImageReview,load_vision_config
from provider_guard import BlockRegistry,GuardedVision

CODE=('pipeline.py','run.py','support.py','providers.py','provider_guard.py','metadata_search.py','rich_metadata.py','hybrid_verify.py','hybrid_run.py')

def args_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--offset',type=int,default=0);parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--case-batch',type=int,default=10);parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--parallel-batches',type=int,default=2);parser.add_argument('--network-concurrency',type=int,default=16)
    parser.add_argument('--max-calls',type=int,default=6000);parser.add_argument('--vlm-config',type=Path,required=True)
    parser.add_argument('--llm-key-file',type=Path);parser.add_argument('--llm-model',default='deepseek-flash')
    parser.add_argument('--llm-base-url',default='https://api.deepseek.com');parser.add_argument('--thumbnail-cache',type=Path)
    parser.add_argument('--cases-file',type=Path);parser.add_argument('--as-of-date',default=datetime.date.today().isoformat())
    return parser

def link_readonly(source,destination):
    if destination.exists():return
    destination.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(source,destination)
    except OSError:shutil.copyfile(source,destination)

def execute(args,api_factories=None):
    root=args.root.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if min(args.limit,args.case_batch,args.workers,args.parallel_batches,args.network_concurrency)<=0 or args.offset<0:raise ValueError('Invalid scope or concurrency')
    config,vision_key=load_vision_config(args.vlm_config)
    text_key=(args.llm_key_file or root/'.secrets/deepseek.key').read_text(encoding='utf-8').strip() if api_factories is None else ''
    source_dir=Path(__file__).resolve().parent
    selection=p.read_json(args.cases_file) if args.cases_file else None
    protocol={'version':'hybrid-v2-time-place','scope':{'offset':args.offset,'limit':args.limit},'case_batch':args.case_batch,'workers':args.workers,'reference_date':args.as_of_date,
              'parallel_batches':args.parallel_batches,'network_concurrency':args.network_concurrency,'screen_images':12,'internal_verify_group':4,'verify_images_per_request':1,
              'text_model':args.llm_model,'vision_model':config['model'],'text_endpoint_sha256':hashlib.sha256(args.llm_base_url.rstrip('/').encode()).hexdigest(),
              'vision_endpoint_sha256':hashlib.sha256(config['base_url'].rstrip('/').encode()).hexdigest(),
              'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('E2E_1321cases_1w_data_v19.xlsx','ES_byFileType_user10_20261117170358.xlsx')},
              'code_sha256':{name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in CODE},
              'metadata':{'phrase_all_matches':True,'bm25_peak_ratio':.45,'fixed_top_k':False,'excerpt_limit':1200,'preserve_baseline_positives':True},
              'ground_truth_used_in_inference':False,'baseline':'v1 retrieval rules with Qwen VLM; same text planner as hybrid'}
    protocol['provider_block_handling']='request-scoped inspection rejection; no image resubmission; uncertain not negative'
    if selection:protocol['selection']={'sha256':hashlib.sha256(args.cases_file.read_bytes()).hexdigest(),'case_ids':[q['id'] for q in selection['cases']],'counts':selection.get('counts')}
    if (out/'protocol.json').exists() and p.read_json(out/'protocol.json')!=protocol:raise ValueError('Frozen inference protocol changed; use a new output directory')
    p.write_json(out/'protocol.json',protocol)
    for name in CODE:link_readonly(source_dir/name,out/'frozen_code'/name)
    # Frozen code must not be hard-linked to editable sources.
    for name in CODE:
        target=out/'frozen_code'/name
        if os.stat(target).st_nlink>1:
            temporary=target.with_suffix('.copy');shutil.copyfile(target,temporary);temporary.replace(target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=protocol['code_sha256'][name]:raise ValueError('Frozen source copy mismatch')
    inv=p.read_json(out/'inventory.json') if (out/'inventory.json').exists() else p.inventory(root,out)
    actual={str(f.relative_to(root/'图片')) for f in (root/'图片').rglob('*') if f.is_file() and f.suffix.lower() in p.PHOTO}
    if actual!={r['path'] for r in inv['images']}:raise ValueError('Source image list changed')
    for row in inv['images']:
        stat=(root/'图片'/row['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=(row['size'],row['mtime_ns']):raise ValueError('Source image changed')
    cases=inv['cases'][args.offset:args.offset+args.limit]
    if selection:
        byid={q['id']:q for q in inv['cases']};cases=[byid[q['id']] for q in selection['cases']]
        if any(byid[q['id']]['query']!=q['query'] for q in selection['cases']):raise ValueError('Selected query text changed')
        p.write_json(out/'selected_cases.json',selection)
    if len(cases)!=args.limit or len({q['id'] for q in cases})!=len(cases):raise ValueError('Invalid requested cases')
    if not (out/'corpus_metadata.json').exists():rich_metadata.corpus(root,out)
    print('Building full-library metadata index (no answer columns).',flush=True)
    index=MetadataIndex(p.read_json(out/'corpus_metadata.json'),inv['images'])
    thumb_owner=out
    if args.thumbnail_cache:
        owner=args.thumbnail_cache.resolve().parent
        if args.thumbnail_cache.name!='thumbs' or p.read_json(owner/'inventory.json')['images']!=inv['images']:raise ValueError('Thumbnail cache inventory mismatch')
        thumb_owner=owner
    original_thumbnail=p.thumbnail
    p.thumbnail=lambda source,unused,row,size=1200:original_thumbnail(source,thumb_owner,row,size)
    network=threading.BoundedSemaphore(args.network_concurrency);state={};state_lock=threading.Lock();registry=BlockRegistry(out)
    def status(offset,stage,**extra):
        with state_lock:
            state[str(offset)]={'stage':stage,**extra};p.write_json(out/'progress.json',{'updated_at':time.time(),'batches':state})
        print(f'cohort positions {offset+1:03d}..{min(offset+args.case_batch,len(cases)):03d}: {stage}',flush=True)
    def batch(offset):
        subset=cases[offset:offset+args.case_batch]
        batch_out=out/f'batch-{offset+1:03d}-{offset+len(subset):03d}';base=batch_out/'visual-v1';hybrid=batch_out/'hybrid-v2'
        for directory in (base,hybrid):
            directory.mkdir(parents=True,exist_ok=True)
            for name in ('inventory.json','corpus_metadata.json'):link_readonly(out/name,directory/name)
        text=CompatibleAPI(batch_out,text_key,args.llm_base_url,args.llm_model,'deepseek',args.max_calls)
        vision_api=SingleImageReview(GuardedVision(CompatibleAPI(batch_out,vision_key,config['base_url'],config['model'],'qwen',args.max_calls,network),registry))
        if api_factories:text,vision_api=api_factories(batch_out)
        routed=RoutedAPI(text,vision_api)
        status(offset,'plan')
        route=p.plan(routed,inv,subset,base)
        proposal=metadata_plan(text,subset,hybrid,args.as_of_date)
        candidates={q['id']:index.search(next(c['query'] for c in subset if c['id']==q['id']),q) for q in proposal['queries']}
        p.write_json(hybrid/'metadata_candidates.json',{'version':'metadata-union-v2','predictions':candidates})
        status(offset,'visual_screen',metadata_candidate_pairs=sum(map(len,candidates.values())))
        stages=[('visual_screen',lambda:p.screen(routed,root,base,inv,route,args.workers,12)),
                ('visual_verify',lambda:(visual.validate_stages(base,inv,route,12),p.verify(routed,root,base,inv,route,args.workers)))]
        for name,stage in stages:
            status(offset,name)
            for attempt in range(3):
                try:stage();break
                except Exception as error:
                    status(offset,name+'_retry',attempt=attempt+1,error=type(error).__name__)
                    if attempt==2 or text.stopped.is_set() or vision_api.stopped.is_set():raise
        visual.validate_stages(base,inv,route,12,verified=True);p.export(base,inv,subset)
        baseline=p.read_json(base/'predictions.json');hybrid_verify.export(base,inv,subset,baseline)
        status(offset,'metadata_verify')
        for attempt in range(3):
            try:prediction=hybrid_verify.verify(vision_api,root,hybrid,inv,route,subset,index,candidates,baseline,args.workers,proposal);break
            except Exception as error:
                status(offset,'metadata_verify_retry',attempt=attempt+1,error=type(error).__name__)
                if attempt==2 or vision_api.stopped.is_set():raise
        hybrid_verify.export(hybrid,inv,subset,prediction)
        p.write_json(batch_out/'completion.json',{'complete':True,'cases':[c['id'] for c in subset],
              'visual_sha256':hashlib.sha256((base/'predictions.json').read_bytes()).hexdigest(),
              'hybrid_sha256':hashlib.sha256((hybrid/'predictions.json').read_bytes()).hexdigest(),
              'metadata_candidate_pairs':sum(map(len,candidates.values()))})
        status(offset,'predictions_complete')
    failures=[]
    try:
        with cf.ThreadPoolExecutor(max_workers=args.parallel_batches) as pool:
            jobs={pool.submit(batch,offset):offset for offset in range(0,len(cases),args.case_batch)}
            for future in cf.as_completed(jobs):
                try:future.result()
                except Exception as error:
                    offset=jobs[future];status(offset,'failed',error=type(error).__name__);failures.append(offset)
    finally:p.thumbnail=original_thumbnail
    if failures:raise RuntimeError(f'Incomplete batches at offsets {failures}; labels have not been read')
    seals={}
    for file in sorted(out.glob('batch-*/completion.json')):seals[file.parent.name]=p.read_json(file)
    p.write_json(out/'prediction_completion.json',{'complete':True,'scope':protocol['scope'],'batches':seals,'sealed_before_label_evaluation':True})
    print('Both methods completed and sealed; no labels were read by inference.',flush=True)
    return out

if __name__=='__main__':execute(args_parser().parse_args())
