"""Recompute published metrics and matrix counts offline; no private data or API key."""
import ast,collections,csv,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/image-case-retrieval/scripts'))
from evaluator import score_case
import openpyxl

def main():
    directory=ROOT/'benchmarks/2026-11-17';data=json.loads((directory/'data/benchmark.json').read_text(encoding='utf-8'))
    expected=json.loads((directory/'checksums.json').read_text(encoding='utf-8'))
    for path,sha in expected['files'].items():assert hashlib.sha256((directory/path).read_bytes()).hexdigest()==sha,('Checksum mismatch',path)
    original=json.loads((directory/'provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((directory/'data/codex_decisions_011_100.json').read_bytes()).hexdigest()==original['original_own_seal']['prediction_sha256']
    functions=lambda file:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(file.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    frozen=functions(directory/'frozen/pipeline.py');published=functions(ROOT/'skills/image-case-retrieval/scripts/pipeline.py')
    for name in ('plan','screen','verify'):assert frozen[name]==published[name],('Inference rules differ from the documented frozen version',name)
    mapping=data['fid_to_image_ids'];cases=data['cases'];assert len(cases)==100
    for row in cases:
        for method in ('skill','codex'):
            score=score_case(row['ground_truth_fids'],mapping,set(row[method+'_image_ids']))
            for key in ('hit_any','expected_fids','locally_mapped_fids','unmapped_fids','hit_fids','known_fid_recall_all'):
                assert score[key]==row[method][key],(row['id'],method,key)
    for name,rows in [('development_001_010',cases[:10]),('holdout_011_100',cases[10:]),('combined_001_100',cases)]:
        for method in ('skill','codex'):
            m=data['summaries'][name][method]
            assert sum(r[method]['hit_any'] for r in rows)==m['hit_cases']
            assert len(rows)==m['evaluated_cases']
            assert sum(r[method]['hit_fids'] for r in rows)==m['known_fid_hits']
            assert sum(r['gt_count'] for r in rows)==m['known_fid_total']
            assert sum(len(r[method+'_image_ids']) for r in rows)==m['returned_pairs']
    baseline=[score_case(r['ground_truth_fids'],mapping,set(r['metadata_baseline_image_ids'])) for r in cases[:10]]
    assert sum(r['hit_any'] for r in baseline)==10 and sum(r['hit_fids'] for r in baseline)==453
    counts={}
    for method in ('skill','codex'):
        c=collections.Counter()
        with (directory/f'data/{method}_matrix_100.csv').open(encoding='utf-8-sig',newline='') as f:
            reader=csv.reader(f);header=next(reader);assert len(header)==10491
            rows=list(reader);assert len(rows)==100
            for index,row in enumerate(rows):
                assert row[0]==cases[index]['id'] and len(row)==len(header)
                assert {header[i] for i,v in enumerate(row) if i>=2 and v=='1'}==set(cases[index][method+'_image_ids'])
                if method=='skill':assert {header[i] for i,v in enumerate(row) if i>=2 and v=='?'}==set(cases[index]['skill_uncertain_image_ids'])
                c.update(row[2:])
        counts[method]=dict(c)
    assert counts['skill']['1']==7659 and counts['codex']['1']==362
    workbook=directory/'comparison-100.xlsx'
    with zipfile.ZipFile(workbook) as archive:assert archive.testzip() is None
    wb=openpyxl.load_workbook(workbook,read_only=True,data_only=True)
    assert wb.sheetnames==['汇总','逐条结果','未命中分析','正关联']
    assert [wb['汇总'][f'E{r}'].value for r in range(5,12)]==[87/90,84/90,96/100,94/100,9/10,1,1]
    assert sum(1 for row in wb['逐条结果'].iter_rows(values_only=True))==101
    assert sum(1 for row in wb['正关联'].iter_rows(values_only=True))==8022
    wb.close()
    print(json.dumps({'verified':True,'cases':100,'images':10489,'skill_hit_cases':96,'codex_hit_cases':94,'matrix_states':counts,'metadata_baseline_development_fid_hits':453},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
