"""Export anonymized, independently scoreable results from the private experiment.

No photos, captions, filenames, original fids, credentials, or absolute paths are exported.
"""
import argparse,ast,collections,csv,hashlib,json,platform,re,shutil
from importlib.metadata import version
from pathlib import Path
import openpyxl

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def anonymize(fid):return 'F'+hashlib.sha256(fid.encode()).hexdigest()

def main(root,dest):
    workspace=root/'image-matching';base=workspace/'runs/pilot-v1';cohort=workspace/'runs/holdout-011-100-v1'
    result=read(cohort/'evaluation_100.json');protocol=read(cohort/'protocol.json');inv=read(base/'inventory.json')
    assert read(cohort/'prediction_completion.json')['complete']
    for name,sha in protocol['source_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha
    seal=read(cohort/'predictions_seal.json')
    for name,sha in seal['files'].items():assert hashlib.sha256((root/name/'predictions.json').read_bytes()).hexdigest()==sha
    own_seal=read(cohort/'own_review/seal.json')
    assert hashlib.sha256((cohort/'own_review/visual_decisions.json').read_bytes()).hexdigest()==own_seal['prediction_sha256']
    wb=openpyxl.load_workbook(root/'E2E_1321cases_1w_data_v19.xlsx',read_only=True,data_only=True)
    labels={str(r[0]):list(dict.fromkeys(x.strip() for x in re.split(r'[,，;；\n]',str(r[3] or '')) if x.strip())) for r in wb.worksheets[0].iter_rows(min_row=2,max_row=101,max_col=4,values_only=True)};wb.close()
    original_mapping=collections.defaultdict(set)
    for r in read(base/'corpus_metadata.json'):
        if r['mapping'] in ('size_and_name','unique_name'):original_mapping[r['fid']].update(r['image_ids'])
    states={}
    for batch in [base]+[cohort/f'batch-{i+1:03d}-{i+10:03d}' for i in range(10,100,10)]:states.update(read(batch/'predictions.json')['states'])
    baseline=read(base/'baseline.json')['predictions'];cases=[]
    for row in result['cases']:
        clean={k:v for k,v in row.items() if k not in ('skill','codex','skill_with_uncertain')}
        for method in ('skill','codex','skill_with_uncertain'):clean[method]={k:v for k,v in row[method].items() if k!='missed_fids'}
        clean['ground_truth_fids']=[anonymize(fid) for fid in labels[row['id']]]
        clean['skill_uncertain_image_ids']=sorted(i for i,s in states[row['id']].items() if s=='?')
        if row['id'] in baseline:clean['metadata_baseline_image_ids']=baseline[row['id']]
        cases.append(clean)
    fids={f for q in labels.values() for f in q}
    payload={'schema_version':1,'date':'2026-11-17','dataset':{'cases_total':len(inv['cases']),'images_total':len(inv['images']),'non_images_excluded':len(inv['excluded']),'directories':dict(collections.Counter(r['directory'] for r in inv['images']))},
             'summaries':{k:result[k] for k in ('development_001_010','holdout_011_100','combined_001_100')},
             'fid_to_image_ids':{anonymize(f):sorted(original_mapping[f]) for f in sorted(fids)},'cases':cases}
    write(dest/'data/benchmark.json',payload)
    with (dest/'data/cases_100.csv').open('w',encoding='utf-8-sig',newline='') as file:
        fields=['id','query','split','gt_count','mapped_fids','skill_count','skill_hit','skill_fid_hits','skill_fid_coverage','codex_count','codex_hit','codex_fid_hits','codex_fid_coverage','uncertain_count']
        writer=csv.writer(file);writer.writerow(fields)
        for r in cases:writer.writerow([r['id'],r['query'],r['split'],r['gt_count'],r['skill']['locally_mapped_fids'],r['skill_count'],int(r['skill']['hit_any']),r['skill']['hit_fids'],r['skill']['known_fid_recall_all'],r['codex_count'],int(r['codex']['hit_any']),r['codex']['hit_fids'],r['codex']['known_fid_recall_all'],r['uncertain_count']])
    public_hashes={};transforms=[]
    for name,original_sha in protocol['code_sha256'].items():
        path=dest/'frozen'/name;source=(cohort/'frozen_code'/name).read_bytes();assert hashlib.sha256(source).hexdigest()==original_sha
        if name=='pipeline.py':
            text=source.decode('utf-8');text=re.sub(r"ROOT = Path\(os.environ.get\('IMAGE_MATCH_ROOT',r'[^']*'\)\)","ROOT = Path(os.environ.get('IMAGE_MATCH_ROOT','.')).resolve()",text,count=1)
            source=text.encode('utf-8');transforms.append({'file':name,'change':'Default dataset root generalized; inference prompts and logic unchanged.'})
        path.write_bytes(source);public_hashes[name]=hashlib.sha256(source).hexdigest()
    own_rules={}
    for node in ast.parse((workspace/'own_review_100.py').read_text(encoding='utf-8')).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('TERMS','YEAR') for t in node.targets):
            own_rules[node.targets[0].id]=ast.literal_eval(node.value)
    write(dest/'codex_candidate_rules.json',own_rules)
    shutil.copyfile(cohort/'own_review/visual_decisions.json',dest/'data/codex_decisions_011_100.json')
    shutil.copyfile(workspace/'codex_visual_sample.json',dest/'data/codex_decisions_001_010.json')
    actual_models=collections.Counter()
    for batch in [base]+[cohort/f'batch-{i+1:03d}-{i+10:03d}' for i in range(10,100,10)]:
        for cache in (batch/'cache').glob('*.json'):actual_models[read(cache).get('model','unknown')]+=1
    provenance={'experiment_date':'2026-11-17','protocol':protocol,'published_frozen_code_sha256':public_hashes,'publication_transformations':transforms,
                'original_prediction_sha256':{Path(n).name:sha for n,sha in seal['files'].items()},'original_own_seal':own_seal,
                'original_own_decisions':read(cohort/'own_review/visual_decisions.json'),
                'holdout_usage':result['holdout_usage'],'holdout_successful_calls':result['holdout_successful_calls'],'observed_api_model_values':dict(actual_models),
                'development_usage':read(base/'evaluation.json')['summary']['usage'],'development_successful_calls':read(base/'evaluation.json')['summary']['calls'],
                'validation':read(cohort/'validation_100.json'),'runtime':{'python':platform.python_version(),**{name:version(name) for name in ('Pillow','pillow-heif','openpyxl','rawpy')}},
                'privacy':{'photos_included':False,'original_fids_included':False,'captions_included':False,'original_filenames_included':False,'fid_transform':'F + SHA256(original fid), consistent across cases'}}
    # Dates and hashes are evidence of local artifact preservation, not external attestation.
    write(dest/'provenance.json',provenance)
    summary={k:v for k,v in result.items() if k!='cases'};write(dest/'summary.json',summary)
    for name in ('skill_matrix_100.csv','codex_matrix_100.csv'):shutil.copyfile(cohort/name,dest/'data'/name)
    manifest={'files':{str(f.relative_to(dest)).replace('\\','/'):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(dest.rglob('*')) if f.is_file() and f.name not in ('checksums.json','comparison-100.xlsx')},'scope':'All published benchmark source artifacts except workbook and this checksum manifest.'}
    write(dest/'checksums.json',manifest)
    print(json.dumps({'cases':len(cases),'anonymized_fids':len(fids),'observed_api_model_values':dict(actual_models),'runtime':provenance['runtime']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--dest',type=Path,default=Path(__file__).resolve().parents[1]/'benchmarks/2026-11-17');a=parser.parse_args();main(a.root.resolve(),a.dest.resolve())
