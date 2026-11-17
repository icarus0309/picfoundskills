"""Offline verification of the published time/location paired benchmark."""
import argparse,collections,csv,hashlib,json,sys,zipfile
import openpyxl
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/image-case-retrieval/scripts'))
from evaluator import score_case

def main(directory):
    read=lambda name:json.loads((directory/name).read_text(encoding='utf-8'))
    manifest=read('checksums.json')
    for name,sha in manifest['files'].items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==sha,('File checksum mismatch',name)
    data=read('data/benchmark.json');provenance=read('provenance.json');rows=data['cases'];mapping=data['fid_to_image_ids']
    for name,sha in provenance['protocol']['code_sha256'].items():
        assert hashlib.sha256((directory/'frozen'/name).read_bytes()).hexdigest()==sha,('Frozen inference source mismatch',name)
    assert provenance['protocol']['ground_truth_used_in_inference'] is False
    assert hashlib.sha256((directory/'selection.json').read_bytes()).hexdigest()==provenance['protocol']['selection']['sha256']
    if provenance.get('screen_id_infrastructure_repairs'):
        assert hashlib.sha256((directory/'frozen_auxiliary/repair_screen_identifiers.py').read_bytes()).hexdigest()==provenance['screen_id_infrastructure_repairs']['helper_sha256']
    expected=[q['id'] for q in provenance['scope_selection']['cases']]
    assert [q['id'] for q in rows]==expected and len(rows)==data['scope']['cases']
    assert dict(collections.Counter(q['category'] for q in rows))==data['scope']['categories']
    for row in rows:
        assert set(row['baseline_image_ids'])<=set(row['hybrid_image_ids'])
        assert set(row['new_positive_image_ids'])==set(row['hybrid_image_ids'])-set(row['baseline_image_ids'])
        assert all(fid.startswith('F') and len(fid)==65 for fid in row['ground_truth_fids'])
        assert sum(row['missed_fid_stages'].values())==row['gt_count']-row['hybrid']['hit_fids']
        for method in ('baseline','hybrid'):
            score=score_case(row['ground_truth_fids'],mapping,set(row[method+'_image_ids']))
            for key,value in score.items():
                if key!='missed_fids':assert value==row[method][key],(row['id'],method,key)
    for name,subset in [('overall',rows)]+[(c,[r for r in rows if r['category']==c]) for c in ('time','place','both')]:
        target=data['overall'] if name=='overall' else data['by_category'][name]
        eligible=[r for r in subset if r['gt_count']]
        for method in ('baseline','hybrid'):
            metric=target[method]
            assert metric['cases']==len(subset) and metric['without_labels']==len(subset)-len(eligible)
            assert metric['labeled_cases']==len(eligible)
            assert metric['hit_cases']==sum(r[method]['hit_any'] for r in eligible)
            assert metric['known_fid_hits']==sum(r[method]['hit_fids'] for r in eligible)
            assert metric['known_fid_total']==sum(r['gt_count'] for r in eligible)
            assert metric['hit_rate']==(metric['hit_cases']/len(eligible) if eligible else None)
            assert metric['known_fid_coverage']==(metric['known_fid_hits']/metric['known_fid_total'] if metric['known_fid_total'] else None)
            assert metric['macro_known_fid_coverage']==(sum(r[method]['known_fid_recall_all'] for r in eligible)/len(eligible) if eligible else None)
            assert metric['returned_pairs']==sum(len(r[method+'_image_ids']) for r in subset)
            assert metric['uncertain_pairs']==sum(len(r[method+'_uncertain_image_ids']) for r in subset)
    matrices={}
    for method in ('baseline','hybrid'):
        count=collections.Counter();n=0
        with (directory/f'data/{method}_matrix.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);header=next(reader);assert len(header)==data['dataset']['images_total']+2
            for index,row in enumerate(reader):
                assert row[0]==rows[index]['id'] and row[1]==rows[index]['query'] and len(row)==len(header)
                assert set(row[2:])<={'0','1','?','S','E'}
                assert {header[i] for i,s in enumerate(row) if i>=2 and s=='1'}==set(rows[index][method+'_image_ids'])
                assert {header[i] for i,s in enumerate(row) if i>=2 and s=='?'}==set(rows[index][method+'_uncertain_image_ids'])
                count.update(row[2:]);n+=1
        assert n==len(rows);matrices[method]=dict(count)
    bookfile=directory/'comparison.xlsx'
    with zipfile.ZipFile(bookfile) as archive:assert archive.testzip() is None
    book=openpyxl.load_workbook(bookfile,read_only=True,data_only=True)
    assert book.sheetnames==['汇总','逐条结果','未命中','正关联']
    groups=[data['overall']]+[data['by_category'][key] for key in ('time','place','both')]
    expected_metrics=[group[method] for group in groups for method in ('baseline','hybrid')]
    keys=('cases','labeled_cases','hit_cases','hit_rate','known_fid_hits','known_fid_total','known_fid_coverage','macro_known_fid_coverage','returned_pairs','uncertain_pairs')
    for actual,metric in zip(book['汇总'].iter_rows(min_row=5,max_row=12,min_col=3,max_col=12,values_only=True),expected_metrics):
        for value,key in zip(actual,keys):
            expected_value=metric[key]
            if expected_value is None:assert value in (None,'')
            else:assert isinstance(value,(int,float)) and abs(value-expected_value)<1e-10,('Workbook headline differs',key,value,expected_value)
    details=list(book['逐条结果'].iter_rows(min_row=2,values_only=True));assert len(details)==len(rows)
    assert [r[0] for r in details]==[r['id'] for r in rows]
    expected_pairs={(r['id'],i) for r in rows for i in r['hybrid_image_ids']}
    assert {(r[0],r[1]) for r in book['正关联'].iter_rows(min_row=2,values_only=True)}==expected_pairs
    failure_ids=[r[0] for r in book['未命中'].iter_rows(min_row=2,values_only=True) if r[0] is not None]
    assert failure_ids==[r['id'] for r in rows if r['gt_count']>0 and not r['hybrid']['hit_any']]
    book.close()
    print(json.dumps({'verified':True,'cases':len(rows),'categories':data['scope']['categories'],'overall':data['overall'],'matrix_states':matrices},ensure_ascii=False,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark-dir',type=Path,default=ROOT/'benchmarks/2026-11-17-time-place');args=parser.parse_args();main(args.benchmark_dir.resolve())
