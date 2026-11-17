"""Rescreen only failed ID-schema batches with larger, explicit image labels.

An infrastructure repair, not a query/semantic-rule or ground-truth adjustment.
The ordinary runner can continue concurrently: only already-failed requests whose
schema-retry response is invalid are eligible, and completed screens are untouched.
"""
import argparse,ast,collections,hashlib,inspect,json,sys,time
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pipeline as p
from providers import CompatibleAPI,ProviderContentBlocked,load_vision_config
from provider_guard import BlockRegistry

def main(args):
    root=args.root.resolve();out=args.out.resolve();config,key=load_vision_config(args.vlm_config)
    inv=p.read_json(out/'inventory.json');groups=collections.defaultdict(list);registry=BlockRegistry(out)
    for row in inv['images']:groups[row['directory']].append(row)
    prefixes={hashlib.sha256(d.encode()).hexdigest()[:8]:d for d in groups}
    owner=args.thumbnail_cache.resolve().parent if args.thumbnail_cache else out
    prefix=next(n.value for n in ast.walk(ast.parse(inspect.getsource(p.screen))) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('逐格看这张'))
    seen={};sheet_cache={};attempted=set();repairs=[];last_activity=time.time()
    audit=out/'screen_identifier_repairs.json'
    if audit.exists():repairs=p.read_json(audit)['repairs']
    while True:
        changed=False
        for batch in sorted(out.glob('batch-*')):
            planfile=batch/'visual-v1/plan.json'
            if not planfile.exists():continue
            plan=p.read_json(planfile);lookup={}
            for sheet in (batch/'visual-v1/sheets').glob('screen-*.jpg'):
                stamp=sheet.stat().st_mtime_ns
                if sheet not in sheet_cache or sheet_cache[sheet][0]!=stamp:sheet_cache[sheet]=(stamp,hashlib.sha256(sheet.read_bytes()).hexdigest())
                lookup[sheet_cache[sheet][1]]=sheet
            for cache in (batch/'cache').glob('qwen-screen-schema-retry-*.json'):
                if cache in seen:continue
                raw=p.read_json(cache);hashes=raw.get('input_image_hashes',[])
                if not hashes or hashes[0] not in lookup:continue
                sheet=lookup[hashes[0]];part,start=sheet.stem[7:].rsplit('-',1);directory=prefixes[part];start=int(start)
                target=batch/'visual-v1/screen'/f'{part}-{start:05d}.json'
                if target.exists():seen[cache]=True;continue
                rows=groups[directory][start:start+12];queries=[q for q in plan['queries'] if directory in q['directories']]
                # The running process owns registry writes; reload for this helper.
                if (out/'provider_blocks.json').exists():
                    blocked=set(p.read_json(out/'provider_blocks.json')['blocked_image_ids'])
                    if any(r['id'] in blocked for r in rows):continue
                ids={r['id'] for r in rows};qids={q['id'] for q in queries};value=raw['result'];candidate=value.get('candidates') if isinstance(value,dict) else None
                valid=isinstance(candidate,list) and all(isinstance(r,dict) and r.get('image_id') in ids and isinstance(r.get('case_ids'),list) and set(r['case_ids'])<=qids for r in candidate)
                if valid:continue
                identity=(batch.name,part,start)
                if identity in attempted:continue
                attempted.add(identity);changed=True;last_activity=time.time()
                canvas=Image.new('RGB',(1280,348*((len(rows)+3)//4)),(238,240,244));draw=ImageDraw.Draw(canvas);errors=[];tested=[]
                for n,row in enumerate(rows):
                    x=n%4*320;y=n//4*348;draw.text((x+8,y+2),row['id'],font=ImageFont.load_default(size=22),fill='black')
                    try:
                        file=p.thumbnail(root,owner,row,1200)
                        with Image.open(file) as original:
                            image=original.copy();image.thumbnail((312,312));canvas.paste(image,(x+(320-image.width)//2,y+28+(320-image.height)//2))
                        tested.append(row['id'])
                    except Exception as error:errors.append({'image_id':row['id'],'error':type(error).__name__})
                repair_sheet=batch/'visual-v1/sheets'/f'id-repair-{part}-{start:05d}.jpg';canvas.save(repair_sheet,quality=90)
                prompt=prefix+json.dumps(sorted(tested))+'。查询:'+json.dumps(queries,ensure_ascii=False)
                prompt+=' 格式复核：不得根据序号推测或续写图片ID；只返回本图和允许列表里的ID。格子位置与ID映射：'+json.dumps([{'row':n//4+1,'column':n%4+1,'image_id':r['id']} for n,r in enumerate(rows)],ensure_ascii=False)
                client=CompatibleAPI(batch,key,config['base_url'],config['model'],'qwen',max_calls=10)
                record={'batch':batch.name,'screen_file':target.name,'failed_cache':cache.name,'repair_image_sha256':hashlib.sha256(repair_sheet.read_bytes()).hexdigest(),'labels_used':False,'semantic_rules_changed':False,'identifier_font_size':22,'time':time.time()}
                for attempt in range(2):
                    try:
                        result=client.call(f'screen-id-repair-{attempt+1}',prompt+' 仅输出candidates JSON对象。', [repair_sheet])
                        candidates=result.get('candidates')
                        if not isinstance(candidates,list):continue
                        candidates=[r for r in candidates if isinstance(r,dict) and r.get('case_ids')!=[]]
                        if not all(r.get('image_id') in set(tested) and isinstance(r.get('case_ids'),list) and set(r['case_ids'])<=qids for r in candidates):continue
                        if not target.exists():p.write_json(target,{'directory':directory,'start':start,'tested_ids':sorted(tested),'case_ids':sorted(qids),'candidates':candidates,'errors':errors,'infrastructure_repair':record})
                        record['status']='repaired';break
                    except ProviderContentBlocked:
                        record['status']='provider_blocked'
                        p.write_json(out/'provider_block_reports'/f'{batch.name}-{part}-{start:05d}.json',{'image_ids':sorted(ids),'stage':'screen-id-repair','media':'image','provider':'qwen','time':time.time(),'action':'no resubmission; uncertain, not negative'})
                        break
                    except Exception as error:record['error']=type(error).__name__
                record.setdefault('status','unresolved');repairs.append(record)
                p.write_json(audit,{'kind':'ID schema/readability repair','ground_truth_used':False,'semantic_rules_changed':False,'helper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'repairs':repairs})
                print(json.dumps({'repair':identity,'status':record['status']},ensure_ascii=False),flush=True)
        if not args.watch or (out/'prediction_completion.json').exists():break
        if time.time()-last_activity>args.idle_seconds:print('Repair watch idle timeout.',flush=True);break
        time.sleep(10)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--vlm-config',type=Path,required=True);parser.add_argument('--thumbnail-cache',type=Path);parser.add_argument('--watch',action='store_true');parser.add_argument('--idle-seconds',type=int,default=7200);main(parser.parse_args())
