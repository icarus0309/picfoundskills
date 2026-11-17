"""Score sealed paired predictions and export an anonymized comparison benchmark."""
import argparse,collections,csv,hashlib,importlib.metadata,json,platform,re,shutil,sys,time
from pathlib import Path
import openpyxl
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'skills/image-case-retrieval/scripts'))
import pipeline as p
from evaluator import score_case
from run import validate_stages

def fid_hash(fid):return 'F'+hashlib.sha256(fid.encode()).hexdigest()

def summary(rows,method):
    eligible=[r for r in rows if r['gt_count']]
    hits=sum(r[method]['hit_any'] for r in eligible);known=sum(r[method]['hit_fids'] for r in eligible);total=sum(r['gt_count'] for r in eligible)
    return {'cases':len(rows),'labeled_cases':len(eligible),'without_labels':len(rows)-len(eligible),'hit_cases':hits,'hit_rate':hits/len(eligible) if eligible else None,
            'known_fid_hits':known,'known_fid_total':total,'known_fid_coverage':known/total if total else None,
            'macro_known_fid_coverage':sum(r[method]['known_fid_recall_all'] for r in eligible)/len(eligible) if eligible else None,
            'returned_pairs':sum(len(r[method+'_image_ids']) for r in rows),'uncertain_pairs':sum(len(r[method+'_uncertain_image_ids']) for r in rows)}

def evaluate(root,out,publish):
    completion=p.read_json(out/'prediction_completion.json')
    if not completion.get('complete'):raise RuntimeError('Inference incomplete; labels will not be opened')
    protocol=p.read_json(out/'protocol.json');inv=p.read_json(out/'inventory.json');selection=p.read_json(out/'selected_cases.json');cases=selection['cases']
    assert hashlib.sha256((out/'selected_cases.json').read_bytes()).hexdigest()==protocol['selection']['sha256'],'Selection input changed'
    for name,sha in protocol['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,'Source workbook changed'
    for name,sha in protocol['code_sha256'].items():assert hashlib.sha256((out/'frozen_code'/name).read_bytes()).hexdigest()==sha,'Frozen inference source changed'
    for image in inv['images']:
        stat=(root/'图片'/image['path']).stat();assert stat.st_size==image['size'] and stat.st_mtime_ns==image['mtime_ns'],'Source image changed'
    assert [q['id'] for q in cases]==protocol['selection']['case_ids']
    predictions={'baseline':{},'hybrid':{}};evidence={'baseline':{},'hybrid':{}};metadata={};visual_candidates={};routes={};seals={}
    for batch_name,record in completion['batches'].items():
        batch=out/batch_name;base=batch/'visual-v1';hybrid=batch/'hybrid-v2'
        plan=p.read_json(base/'plan.json');validate_stages(base,inv,plan,12,verified=True)
        for q in plan['queries']:routes[q['id']]=q['directories'];visual_candidates[q['id']]=set()
        for file in (base/'screen').glob('*.json'):
            for row in p.read_json(file)['candidates']:
                for q in row['case_ids']:visual_candidates[q].add(row['image_id'])
        metadata.update(p.read_json(hybrid/'metadata_candidates.json')['predictions'])
        for method,directory in [('baseline',base),('hybrid',hybrid)]:
            file=directory/'predictions.json';sha=hashlib.sha256(file.read_bytes()).hexdigest();assert sha==record['visual_sha256' if method=='baseline' else 'hybrid_sha256']
            pred=p.read_json(file);assert not set(pred['states']) & set(predictions[method]);predictions[method].update(pred['states'])
            evidence[method].update({(r['case_id'],r['image_id']):r for r in pred['evidence']});seals[str(file.relative_to(out))]=sha
        # Every metadata candidate must have a final explicit judgment or preserve
        # an existing baseline positive. Missing work must not become negative.
        for q,images in p.read_json(hybrid/'metadata_candidates.json')['predictions'].items():
            for image in images:
                assert (q,image) in evidence['hybrid'],'Unreviewed metadata candidate'
                assert predictions['hybrid'][q][image] in ('0','1','?')
    expected={q['id'] for q in cases};assert set(predictions['baseline'])==set(predictions['hybrid'])==expected
    p.write_json(out/'comparison_prediction_seal.json',{'files':seals,'labels_used_in_inference':False,'sealed_before_this_evaluation':True,'time':time.time()})
    corpus=p.read_json(out/'corpus_metadata.json');mapping=collections.defaultdict(set)
    for row in corpus:
        if row['mapping'] in ('size_and_name','unique_name'):mapping[row['fid']].update(row['image_ids'])
    # All label access is below the all-method completion and seal checks.
    workbook=openpyxl.load_workbook(root/'E2E_1321cases_1w_data_v19.xlsx',read_only=True,data_only=True)
    labels={str(r[0]):list(dict.fromkeys(s.strip() for s in re.split(r'[,，;；\n]',str(r[3] or '')) if s.strip())) for r in workbook.worksheets[0].iter_rows(min_row=2,max_col=4,values_only=True) if str(r[0]) in expected};workbook.close()
    provider_blocks=p.read_json(out/'provider_blocks.json') if (out/'provider_blocks.json').exists() else {'blocked_image_ids':[],'events':[]}
    blocked=set(provider_blocks['blocked_image_ids'])
    image_by_id={i['id']:i for i in inv['images']};rows=[];gain_stages=collections.Counter();missed_stages=collections.Counter()
    for query in cases:
        q=query['id'];gt=labels.get(q,[]);row={'id':q,'query':query['query'],'category':query['category'],'gt_count':len(gt)}
        for method in ('baseline','hybrid'):
            state=predictions[method][q];positive={i for i,s in state.items() if s=='1'};unknown={i for i,s in state.items() if s=='?'}
            row[method]=score_case(gt,mapping,positive);row[method+'_image_ids']=sorted(positive);row[method+'_uncertain_image_ids']=sorted(unknown)
            row[method+'_including_uncertain']=score_case(gt,mapping,positive|unknown)
        base=set(row['baseline_image_ids']);hybrid=set(row['hybrid_image_ids']);assert base<=hybrid
        meta=set(metadata[q]);union=meta|visual_candidates[q]
        row['candidate_recall']={'visual':score_case(gt,mapping,visual_candidates[q]),'metadata':score_case(gt,mapping,meta),'union':score_case(gt,mapping,union)}
        row['candidate_counts']={'visual':len(visual_candidates[q]),'metadata':len(meta),'union':len(union)}
        row['provider_blocked_uncertain_pairs']={m:sum(i in blocked for i in row[m+'_uncertain_image_ids']) for m in ('baseline','hybrid')}
        row['new_positive_image_ids']=sorted(hybrid-base)
        row['new_positive_verification']=dict(collections.Counter(evidence['hybrid'][(q,i)].get('verification_method','unknown') for i in hybrid-base))
        row['hybrid_confirmation_sources']={i:evidence['hybrid'][(q,i)].get('verification_method','visual_baseline') for i in sorted(hybrid)}
        row['new_fid_hits']=row['hybrid']['hit_fids']-row['baseline']['hit_fids']
        recovered=collections.Counter()
        for fid in gt:
            images=mapping.get(fid,set())
            if images & base or not images & hybrid:continue
            statuses={predictions['baseline'][q].get(i,'S') for i in images & hybrid}
            if '?' in statuses and any(i in blocked for i in images & hybrid):stage='provider_blocked_visual'
            elif 'S' in statuses:stage='directory_not_searched'
            elif '?' in statuses:stage='baseline_uncertain'
            elif any((q,i) in evidence['baseline'] for i in images & hybrid):stage='baseline_visual_rejection'
            else:stage='coarse_screen_miss'
            recovered[stage]+=1;gain_stages[stage]+=1
        row['recovered_by_stage']=dict(recovered)
        missed=collections.Counter()
        for fid in gt:
            images=mapping.get(fid,set())
            if images & hybrid:continue
            if not images:stage='unmapped_local_fid'
            elif any(predictions['hybrid'][q].get(i)=='?' for i in images):stage='final_uncertain'
            elif all(predictions['hybrid'][q].get(i)=='E' for i in images):stage='decode_error'
            elif images & meta:stage='metadata_candidate_not_confirmed'
            elif images & visual_candidates[q]:stage='visual_candidate_not_confirmed'
            else:stage='not_in_either_candidate_pool'
            missed[stage]+=1;missed_stages[stage]+=1
        row['missed_fid_stages']=dict(missed)
        rows.append(row)
    result={'scope':{'cases':len(rows),'source_start_case_number':101,'selection':'All A/B-classified time/named-location queries after the first 100; no ground-truth selection','categories':selection['counts'],'reference_date':protocol['reference_date']},
            'overall':{m:summary(rows,m) for m in ('baseline','hybrid')},
            'by_category':{category:{m:summary([r for r in rows if r['category']==category],m) for m in ('baseline','hybrid')} for category in ('time','place','both')},
            'recovered_fid_stages':dict(gain_stages),'missed_fid_stages':dict(missed_stages),
            'unmapped_known_fid_relations':sum(r['hybrid']['unmapped_fids'] for r in rows),'cases':rows}
    p.write_json(out/'evaluation.json',result)
    # Publish a semantically identical, pseudonymized scoring data set.
    publish.mkdir(parents=True,exist_ok=True);(publish/'data').mkdir(exist_ok=True)
    shutil.copyfile(out/'selected_cases.json',publish/'selection.json')
    clean_rows=[]
    for row in rows:
        item={**row,'ground_truth_fids':[fid_hash(fid) for fid in labels.get(row['id'],[])]}
        for method in ('baseline','hybrid','baseline_including_uncertain','hybrid_including_uncertain'):item[method]={k:v for k,v in item[method].items() if k!='missed_fids'}
        item['candidate_recall']={k:{a:b for a,b in v.items() if a!='missed_fids'} for k,v in item['candidate_recall'].items()}
        clean_rows.append(item)
    all_fids={fid for q in expected for fid in labels.get(q,[])}
    data={'schema_version':2,'scope':result['scope'],'dataset':{'cases_total':len(inv['cases']),'images_total':len(inv['images']),'excluded_non_images':len(inv['excluded'])},
          'overall':result['overall'],'by_category':result['by_category'],'cases':clean_rows,
          'fid_to_image_ids':{fid_hash(fid):sorted(mapping.get(fid,set())) for fid in sorted(all_fids)}}
    p.write_json(publish/'data/benchmark.json',data)
    with (publish/'data/cases.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.writer(file);writer.writerow(['id','query','category','expected_fids','baseline_hit','hybrid_hit','baseline_fid_hits','hybrid_fid_hits','baseline_fid_coverage','hybrid_fid_coverage','baseline_returns','hybrid_returns','new_returns'])
        for r in rows:writer.writerow([r['id'],r['query'],r['category'],r['gt_count'],int(r['baseline']['hit_any']),int(r['hybrid']['hit_any']),r['baseline']['hit_fids'],r['hybrid']['hit_fids'],r['baseline']['known_fid_recall_all'],r['hybrid']['known_fid_recall_all'],len(r['baseline_image_ids']),len(r['hybrid_image_ids']),len(r['new_positive_image_ids'])])
    for method in ('baseline','hybrid'):
        with (publish/f'data/{method}_matrix.csv').open('w',encoding='utf-8-sig',newline='') as file:
            writer=csv.writer(file);writer.writerow(['case_id','query']+[i['id'] for i in inv['images']])
            for row in rows:writer.writerow([row['id'],row['query']]+[predictions[method][row['id']].get(i['id'],'S') for i in inv['images']])
    calls=collections.Counter();usage=collections.defaultdict(collections.Counter);latencies=[]
    for file in out.glob('batch-*/cache/*.json'):
        record=p.read_json(file);calls[record['stage']]+=1;latencies.append(record['seconds'])
        for field in ('prompt_tokens','completion_tokens','total_tokens'):usage[record['provider']][field]+=record.get('usage',{}).get(field,0)
    field_counts={field:sum(bool(r.get('attributes',{}).get(field)) for r in corpus) for field in ('tagIds','shootingTime','country','province','city','business_terms','place_terms')}
    schema_abstentions={'records':[p.read_json(f) for f in sorted(out.glob('batch-*/schema_abstentions/*.json'))],
                        'final_uncertain_pairs':{m:sum(r.get('schema_abstention',False) and r.get('status')=='uncertain' for r in evidence[m].values()) for m in ('baseline','hybrid')}}
    media_abstentions={'records':[p.read_json(f) for f in sorted(out.glob('batch-*/schema_abstentions/media_transport/*.json'))],
                       'final_uncertain_pairs':{m:sum(r.get('media_transport_abstention',False) and r.get('status')=='uncertain' for r in evidence[m].values()) for m in ('baseline','hybrid')}}
    provenance={'protocol':protocol,'prediction_files':seals,'scope_selection':selection,'calls_by_stage':dict(calls),'successful_response_usage':{k:dict(v) for k,v in usage.items()},
                'environment':{'python':platform.python_version(),'system':platform.system(),'dependencies':{package:importlib.metadata.version(package) for package in ('Pillow','openpyxl','pillow-heif','rawpy')}},
                'metadata_field_nonempty_rows':field_counts,'recovered_fid_stages':dict(gain_stages),'missed_fid_stages':dict(missed_stages),
                'unmapped_known_fid_relations':result['unmapped_known_fid_relations'],
                'provider_blocks':provider_blocks,'provider_blocked_uncertain_pairs':{m:sum(r['provider_blocked_uncertain_pairs'][m] for r in rows) for m in ('baseline','hybrid')},
                'schema_abstentions':schema_abstentions,
                'provider_generation_errors':[p.read_json(f) for f in sorted(out.glob('batch-*/provider_generation_errors/*.json'))],
                'provider_media_errors':[p.read_json(f) for f in sorted(out.glob('batch-*/provider_media_errors/*.json'))],'media_abstentions':media_abstentions,
                'limitations':['Both arms use Qwen; this is a retrieval/workflow comparison, not a DeepSeek versus Qwen model benchmark.','Hybrid preserves baseline positives by construction; this guarantees nondecreasing positive-only recall, not precision.','Structured metadata can confirm pure time/location queries without a VLM; visual constraints still require image review.','Provider cache usage excludes probes, interrupted scope runs and responses not saved successfully.','No photos, original fids, captions, GPS coordinates or private service endpoints are published.']}
    if (out/'screen_identifier_repairs.json').exists():provenance['screen_id_infrastructure_repairs']=p.read_json(out/'screen_identifier_repairs.json')
    p.write_json(publish/'provenance.json',provenance)
    p.write_json(publish/'summary.json',{k:v for k,v in result.items() if k!='cases'})
    for name in protocol['code_sha256']:link=publish/'frozen'/name;link.parent.mkdir(exist_ok=True);shutil.copyfile(out/'frozen_code'/name,link)
    auxiliary=publish/'frozen_auxiliary';auxiliary.mkdir(exist_ok=True)
    helper=auxiliary/'repair_screen_identifiers.py'
    helper_sha=provenance.get('screen_id_infrastructure_repairs',{}).get('helper_sha256')
    sources=[helper,out/'frozen_auxiliary/repair_screen_identifiers.py',REPO/'skills/image-case-retrieval/scripts/repair_screen_identifiers.py',REPO/'tools/repair_screen_identifiers.py']
    candidates=[source for source in sources if source.exists() and (not helper_sha or hashlib.sha256(source.read_bytes()).hexdigest()==helper_sha)]
    assert candidates,'Executed repair helper source snapshot is missing'
    if candidates[0]!=helper:shutil.copyfile(candidates[0],helper)
    if (out/'infrastructure_amendments').exists():shutil.copytree(out/'infrastructure_amendments',publish/'infrastructure_amendments',dirs_exist_ok=True)
    manifest={str(f.relative_to(publish)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(publish.rglob('*')) if f.is_file() and f.name!='checksums.json'}
    p.write_json(publish/'checksums.json',{'files':manifest})
    print(json.dumps({'scope':result['scope'],'overall':result['overall'],'by_category':result['by_category'],'recovered_fid_stages':result['recovered_fid_stages']},ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--publish-dir',type=Path,required=True);args=parser.parse_args();evaluate(args.root.resolve(),args.out.resolve(),args.publish_dir.resolve())
