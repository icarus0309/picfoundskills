"""Post-prediction scoring. Only this stage reads case ground-truth column D."""
import argparse,collections,hashlib,json,re
from pathlib import Path
import openpyxl
from pipeline import read_json,write_json,ROOT

def score_case(gt,mapping,predicted):
    gt=list(dict.fromkeys(gt))
    available={fid:set(mapping.get(fid,[])) for fid in gt if mapping.get(fid)}
    hits=[fid for fid,ids in available.items() if ids & set(predicted)]
    return {'expected_fids':len(gt),'locally_mapped_fids':len(available),'unmapped_fids':len(gt)-len(available),
            'hit_fids':len(hits),'hit_any':bool(hits),'known_fid_recall_all':len(hits)/len(gt) if gt else None,
            'known_fid_recall_available':len(hits)/len(available) if available else None,
            'missed_fids':[fid for fid in gt if fid not in hits]}

def evaluate(root,out):
    pred_path=out/'predictions.json';pred=read_json(pred_path)
    write_json(out/'predictions_seal.json',{'sha256':hashlib.sha256(pred_path.read_bytes()).hexdigest(),'sealed_before_this_evaluation_label_read':True})
    mapping=collections.defaultdict(set)
    for r in read_json(out/'corpus_metadata.json'):
        if r['mapping'] in ('size_and_name','unique_name'):mapping[r['fid']].update(r['image_ids'])
    wb=openpyxl.load_workbook(root/'E2E_1321cases_1w_data_v19.xlsx',read_only=True,data_only=True)
    labels={str(r[0]):list(dict.fromkeys(x.strip() for x in re.split(r'[,，;；\n]',str(r[3] or '')) if x.strip()))
            for r in wb.worksheets[0].iter_rows(min_row=2,max_col=4,values_only=True) if str(r[0]) in pred['states']}
    wb.close();queries={q['id']:q['query'] for q in read_json(out/'inventory.json')['cases']};rows=[]
    for q,state in pred['states'].items():
        positive={i for i,s in state.items() if s=='1'};unknown={i for i,s in state.items() if s=='?'}
        gt=labels.get(q,[])
        rows.append({'id':q,'query':queries[q],'returned_pairs':len(positive),'uncertain_pairs':len(unknown),
                     **score_case(gt,mapping,positive),'including_uncertain':score_case(gt,mapping,positive|unknown)})
    eligible=[r for r in rows if r['expected_fids']]
    summary={'cases_run':len(rows),'cases_with_labels':len(eligible),'cases_without_labels':len(rows)-len(eligible),
             'hit_cases':sum(r['hit_any'] for r in eligible),'hit_rate':sum(r['hit_any'] for r in eligible)/len(eligible) if eligible else None,
             'known_fid_hits':sum(r['hit_fids'] for r in eligible),'known_fid_total':sum(r['expected_fids'] for r in eligible),
             'returned_pairs':sum(r['returned_pairs'] for r in rows),'uncertain_pairs':sum(r['uncertain_pairs'] for r in rows)}
    write_json(out/'evaluation.json',{'summary':summary,'cases':rows,
               'limitations':['Positive-only labels do not establish true precision or complete semantic recall.','Unsearched cells and uncertain judgments are not verified negatives.']})
    print(json.dumps(summary,ensure_ascii=False));return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();evaluate(args.root.resolve(),args.out.resolve())
