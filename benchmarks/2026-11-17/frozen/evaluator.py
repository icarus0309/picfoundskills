"""Post-hoc evaluator: the only matching component that reads case column D."""
import argparse, collections, hashlib, json, re
from pathlib import Path
import openpyxl
from pipeline import ROOT,read_json,write_json

def score_case(gt, mapping, predicted):
    mapped={fid:set(mapping.get(fid,[])) for fid in gt}
    available={fid:ids for fid,ids in mapped.items() if ids}
    hit=[fid for fid,ids in available.items() if ids & predicted]
    return {'expected_fids':len(gt),'locally_mapped_fids':len(available),'unmapped_fids':len(gt)-len(available),'hit_fids':len(hit),'hit_any':bool(hit),'known_fid_recall_all':len(hit)/len(gt) if gt else None,'known_fid_recall_available':len(hit)/len(available) if available else None,
            'missed_fids':[fid for fid in gt if fid not in hit]}

def evaluate(root,out):
    inv=read_json(out/'inventory.json');pred=read_json(out/'predictions.json');base=read_json(out/'baseline.json');meta=read_json(out/'corpus_metadata.json')
    seal=read_json(out/'baseline_seal.json')['sha256']
    assert hashlib.sha256((out/'baseline.json').read_bytes()).hexdigest()==seal,'Baseline changed after sealing'
    write_json(out/'predictions_seal.json',{'sha256':hashlib.sha256((out/'predictions.json').read_bytes()).hexdigest()})
    mapping=collections.defaultdict(set)
    for r in meta:
        # Ambiguous duplicate-name mappings do not count as verified ground truth.
        if r['mapping'] in ('size_and_name','unique_name'):mapping[r['fid']].update(r['image_ids'])
    book=openpyxl.load_workbook(root/'E2E_1321cases_1w_data_v19.xlsx',read_only=True,data_only=True)
    labels={str(row[0]):[f.strip() for f in re.split('[,，;；\\n]',str(row[3] or '')) if f.strip()] for row in book.worksheets[0].iter_rows(min_row=2,max_col=4,values_only=True)};book.close()
    own_path=root/'image-matching'/'codex_visual_sample.json'
    own=read_json(own_path) if own_path.exists() else {'predictions':{}}
    if own_path.exists():
        pre=out/'codex_visual_pre_evaluation_seal.json'
        if pre.exists():assert read_json(pre)['sha256']==hashlib.sha256(own_path.read_bytes()).hexdigest()
        write_json(out/'codex_visual_seal.json',{'sha256':hashlib.sha256(own_path.read_bytes()).hexdigest()})
    result=[];images={r['id']:r for r in inv['images']}
    for q in [q for q in inv['cases'] if q['id'] in pred['states']]:
        gt=list(dict.fromkeys(labels[q['id']]))
        method={im for im,s in pred['states'][q['id']].items() if s=='1'}
        uncertain={im for im,s in pred['states'][q['id']].items() if s=='?'}
        baseline=set(base['predictions'].get(q['id'],[]))
        row={'id':q['id'],'query':q['query'],'vlm_count':len(method),'uncertain_count':len(uncertain),'baseline_count':len(baseline),
            'baseline_available':q['id'] in base['predictions'],'codex_visual_sample_available':q['id'] in own['predictions'],
            'vlm':score_case(gt,mapping,method),'baseline':score_case(gt,mapping,baseline),'codex_visual_sample':score_case(gt,mapping,set(own['predictions'].get(q['id'],[]))),'vlm_with_uncertain':score_case(gt,mapping,method|uncertain),
            'vlm_image_ids':sorted(method),'baseline_image_ids':sorted(baseline)}
        row['missed_images']=[{'fid':fid,'images':[images[i]['path'] for i in mapping.get(fid,[])]} for fid in row['vlm']['missed_fids']]
        stages=collections.Counter()
        for fid in row['vlm']['missed_fids']:
            ids=mapping.get(fid,set())
            if not ids:stages['unmapped']+=1;continue
            statuses=[pred['states'][q['id']].get(i,'S') for i in ids]
            if '?' in statuses:stages['uncertain']+=1
            elif 'E' in statuses:stages['decode_error']+=1
            elif all(s=='S' for s in statuses):stages['directory_route']+=1
            elif any(any(e['case_id']==q['id'] and e['image_id']==i for e in pred['evidence']) for i in ids):stages['individual_verification']+=1
            else:stages['coarse_screen']+=1
        row['missed_by_stage']=dict(stages)
        result.append(row)
    usage=collections.Counter();calls=collections.Counter();seconds=0
    for p in (out/'cache').glob('*.json'):
        r=read_json(p);calls[r['stage']]+=1;seconds+=r['seconds']
        for key in ('prompt_tokens','completion_tokens','total_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens'):usage[key]+=r.get('usage',{}).get(key,0)
    summary={'cases':len(result),'images':len(inv['images']),'vlm_hit_cases':sum(r['vlm']['hit_any'] for r in result),'baseline_hit_cases':sum(r['baseline']['hit_any'] for r in result),
        'vlm_hit_rate':sum(r['vlm']['hit_any'] for r in result)/len(result),'baseline_evaluated_cases':sum(r['baseline_available'] for r in result),'baseline_hit_rate':sum(r['baseline']['hit_any'] for r in result)/sum(r['baseline_available'] for r in result) if any(r['baseline_available'] for r in result) else None,'codex_visual_sample_evaluated_cases':sum(r['codex_visual_sample_available'] for r in result),'codex_visual_sample_hit_cases':sum(r['codex_visual_sample']['hit_any'] for r in result),'calls':dict(calls),'usage':dict(usage),'api_seconds_sum':seconds,
        'limitations':['First ten cases are development cases, not an independent test set.','Only positive labels supplied: true precision and recall of every relevant image are not identifiable.','Metadata baseline uses existing corpus tags/captions; VLM uses images plus shooting date, so this is not an equal-input model benchmark.','Rows beyond pilot are unrun, not negatives.','Route-excluded images and decoding failures remain unjudged.']}
    write_json(out/'evaluation.json',{'summary':summary,'cases':result});print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=ROOT/'image-matching/runs/pilot-v1');a=p.parse_args();evaluate(a.root,a.out)
