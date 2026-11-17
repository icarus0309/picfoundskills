"""Classify A/B-only queries by capture-time and named-location constraints."""
import argparse,collections,concurrent.futures as cf,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/image-case-retrieval/scripts'))
import pipeline as p
from providers import CompatibleAPI

def classify(root,out,start_offset=100):
    out.mkdir(parents=True,exist_ok=True);cases=p.read_cases(root)[start_offset:]
    signature=hashlib.sha256(json.dumps(cases,ensure_ascii=False).encode()).hexdigest()
    key=(root/'.secrets/deepseek.key').read_text(encoding='utf-8').strip()
    api=CompatibleAPI(out,key,'https://api.deepseek.com','deepseek-flash','deepseek',max_calls=100)
    def batch(group):
        file=out/f"classification-{group[0]['excel_row']:04d}.json"
        if file.exists():return p.read_json(file)
        prompt='仅根据以下用例ID和查询，判断是否包含可用图片元信息检索的拍摄时间或具名地点条件。不得查看、猜测或生成预期fid。输出JSON {"cases":[{"id":"...","category":"time|place|both|none","time_terms":[],"place_terms":[],"reason":"简短说明"}]}。time指拍摄年份、年月日、相对时间、季节、节日或时段；画面中历史朝代/时间表/钟表不等同拍摄时间。place指国家、省、市、区县、具名商圈/景区/校园/机构/地标等；泛称建筑、室内、街道、宿舍等无具体地名通常none，日本料理等风格/产地修饰不等于在日本拍摄。具名地点的内容查询也属于place。两类都有为both，均无为none。不要为了凑样本放宽条件。必须逐条覆盖且仅返回给定ID。查询：'+json.dumps(group,ensure_ascii=False)
        result=api.call('case-classification',prompt)
        valid=lambda obj:isinstance(obj.get('cases'),list) and len(obj['cases'])==len(group) and {c.get('id') for c in obj['cases']}=={c['id'] for c in group} and all(c.get('category') in ('time','place','both','none') for c in obj['cases'])
        if not valid(result):result=api.call('case-classification-retry',prompt+' 严格逐条覆盖，不得漏ID。')
        if not valid(result):raise ValueError('Case classification schema mismatch')
        p.write_json(file,result);return result
    groups=[cases[i:i+40] for i in range(0,len(cases),40)];rows=[]
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        for n,result in enumerate(pool.map(batch,groups),1):rows.extend(result['cases']);print(f'Classification {n}/{len(groups)}',flush=True)
    byid={r['id']:r for r in rows};rows=[{**q,**byid[q['id']]} for q in cases]
    result={'query_source_sha256':signature,'source_start_offset':start_offset,'labels_read':False,'counts':dict(collections.Counter(r['category'] for r in rows)),'cases':rows}
    p.write_json(out/'classification.json',result);print(json.dumps(result['counts'],ensure_ascii=False));return result

def select(out,limit=100,all_cases=False):
    data=p.read_json(out/'classification.json');groups={key:[q for q in data['cases'] if q['category']==key] for key in ('time','place','both')}
    if all_cases:chosen=[q for q in data['cases'] if q['category']!='none']
    else:
        chosen=[];offset={k:0 for k in groups}
        while len(chosen)<limit:
            progress=False
            for key,rows in groups.items():
                if offset[key]<len(rows) and len(chosen)<limit:chosen.append(rows[offset[key]]);offset[key]+=1;progress=True
            if not progress:raise ValueError('Insufficient eligible queries for requested sample')
        chosen.sort(key=lambda q:q['excel_row'])
    manifest={'version':'time-place-selection-v2','selection':'all eligible' if all_cases else 'deterministic round-robin stratification; original order within strata',
              'query_source_sha256':data['query_source_sha256'],'labels_used':False,'cases':chosen,'counts':dict(collections.Counter(q['category'] for q in chosen))}
    p.write_json(out/'selected_cases.json',manifest);print(json.dumps({'selected':len(chosen),'counts':manifest['counts']},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['classify','select']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--start-offset',type=int,default=100);parser.add_argument('--limit',type=int,default=100);parser.add_argument('--all',action='store_true');a=parser.parse_args()
    classify(a.root.resolve(),a.out.resolve(),a.start_offset) if a.stage=='classify' else select(a.out.resolve(),a.limit,a.all)
