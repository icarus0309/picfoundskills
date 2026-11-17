"""Score both methods only after the whole selected cohort is complete and sealed."""
import argparse,hashlib,json
from pathlib import Path
import pipeline as p
from evaluator import evaluate

def score(root,out):
    file=out/'prediction_completion.json'
    if not file.exists() or not p.read_json(file).get('complete'):raise RuntimeError('All cohort predictions must complete before opening labels')
    completion=p.read_json(file);results={method:[] for method in ('baseline','hybrid')}
    for name,record in completion['batches'].items():
        for method,folder,key in [('baseline','visual-v1','visual_sha256'),('hybrid','hybrid-v2','hybrid_sha256')]:
            directory=out/name/folder
            if hashlib.sha256((directory/'predictions.json').read_bytes()).hexdigest()!=record[key]:raise ValueError('Prediction seal changed')
    for name in completion['batches']:
        for method,folder in [('baseline','visual-v1'),('hybrid','hybrid-v2')]:results[method].append(evaluate(root,out/name/folder))
    summaries={}
    for method,rows in results.items():
        total={key:sum(r[key] for r in rows) for key in ('cases_run','cases_with_labels','cases_without_labels','hit_cases','known_fid_hits','known_fid_total','returned_pairs','uncertain_pairs')}
        total['hit_rate']=total['hit_cases']/total['cases_with_labels'] if total['cases_with_labels'] else None
        total['known_fid_coverage']=total['known_fid_hits']/total['known_fid_total'] if total['known_fid_total'] else None
        summaries[method]=total
    p.write_json(out/'scoring_summary.json',summaries);print(json.dumps(summaries,ensure_ascii=False,indent=2));return summaries

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();score(args.root.resolve(),args.out.resolve())
