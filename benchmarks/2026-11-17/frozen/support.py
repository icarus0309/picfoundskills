"""Corpus metadata and independent text baseline. Does not read case labels."""
import argparse, collections, csv, hashlib, html, json, re
from pathlib import Path
import openpyxl
from pipeline import ROOT,read_json,write_json,contact

def corpus(root,out):
    inv=read_json(out/'inventory.json');byname=collections.defaultdict(list)
    for r in inv['images']:byname[Path(r['path']).name.casefold()].append(r)
    book=openpyxl.load_workbook(root/'ES_byFileType_user10_20261117170358.xlsx',read_only=True,data_only=True)
    rows=[]
    for sheet in book:
        it=sheet.iter_rows(values_only=True);headers=next(it)
        for values in it:
            r=dict(zip(headers,values));paths=byname.get(str(r.get('fileName','')).casefold(),[])
            exact=[p for p in paths if str(p['size'])==str(r.get('fileSize'))]
            mapped=exact or paths
            # Retain ambiguity; do not arbitrarily assign duplicate basenames.
            rows.append({'fid':str(r.get('fid','')),'filename':r.get('fileName'),'image_ids':[p['id'] for p in mapped],'mapping':'size_and_name' if exact else 'unique_name' if len(paths)==1 else 'ambiguous_name' if paths else 'missing',
                         'text':' '.join(str(r.get(k) or '') for k in ['tagIds','description','ocr','aiDescription','keyword','abstract','search_term','summary','business_terms','place_terms']), 'shooting_time':str(r.get('shootingTime') or '')})
    book.close();write_json(out/'corpus_metadata.json',rows)
    print(json.dumps(dict(collections.Counter(r['mapping'] for r in rows)),ensure_ascii=False),flush=True)

# Fixed before reading labels. This is an explainable metadata baseline, not an
# exhaustive visual annotation by Codex. Its richer inputs are disclosed.
RULES={
 'TC0001':lambda t: '火烈鸟' in t or 'flamingo' in t.lower(),
 'TC0002':lambda t: '小熊猫' in t or '红熊猫' in t or 'red panda' in t.lower(),
 'TC0003':lambda t: '白虎' in t,
 'TC0004':lambda t: bool(re.search('鸟|鹦鹉|麻雀|喜鹊|鸽子|翠鸟',t) and re.search('树枝|枝头|枝干|枝条',t)),
 'TC0005':lambda t: bool(re.search('大门|校门|院门|山门|园门|门楼|牌楼|城门|拱门|门洞|入口处|正门',t)),
 'TC0006':lambda t: bool(re.search('海洋世界|海洋馆|水族馆|水族箱|水族展示|海底世界|水下隧道|海洋公园',t)),
 'TC0007':lambda t: '白虎' in t and not re.search('两只白虎|多只白虎|三只白虎|一对白虎',t),
 'TC0008':lambda t: bool(re.search('老虎|白虎|东北虎|孟加拉虎|华南虎',t)),
 'TC0009':lambda t: '鳄鱼' in t,
 'TC0010':lambda t: bool(re.search('鱼|锦鲤|鲤鱼|金鱼',t) and re.search('特写|近景|近距离',t) and not re.search('鱿鱼|章鱼|墨鱼|鳄鱼|钓鱼竿',t)),
}

def baseline(root,out):
    if (out/'baseline.json').exists():
        obj=read_json(out/'baseline.json')
        assert hashlib.sha256((out/'baseline.json').read_bytes()).hexdigest()==read_json(out/'baseline_seal.json')['sha256']
        print('Using sealed metadata baseline',flush=True);return
    inv=read_json(out/'inventory.json');rows=read_json(out/'corpus_metadata.json')
    predictions={q['id']:set() for q in inv['cases'][:10]};evidence=[]
    for r in rows:
        for q,rule in RULES.items():
            if q=='TC0009' and not r['shooting_time'].startswith('2013'):continue
            if rule(r['text']):
                for image_id in r['image_ids']:
                    predictions[q].add(image_id);evidence.append({'case_id':q,'image_id':image_id,'excerpt':r['text'][:700]})
    obj={'method':'Codex-designed metadata rules (no labels), not exhaustive Codex visual judgments','predictions':{q:sorted(v) for q,v in predictions.items()},'evidence':evidence}
    write_json(out/'baseline.json',obj)
    # Save prior-to-evaluation prediction hash for transparent experiment order.
    write_json(out/'baseline_seal.json',{'sha256':hashlib.sha256((out/'baseline.json').read_bytes()).hexdigest()})
    allids=set().union(*predictions.values());byid={r['id']:r for r in inv['images']};items=[byid[i] for i in sorted(allids)]
    for i in range(0,len(items),24):contact(root,out,items[i:i+24],f'baseline-{i//24:03d}',tile=240,cols=6)
    print(json.dumps({q:len(v) for q,v in predictions.items()},ensure_ascii=False),flush=True)

def gallery(root,out):
    inv=read_json(out/'inventory.json');pred=read_json(out/'predictions.json');byid={r['id']:r for r in inv['images']}
    fids=collections.defaultdict(set)
    for r in read_json(out/'corpus_metadata.json'):
        if r['mapping'] in ('size_and_name','unique_name'):
            for i in r['image_ids']:fids[i].add(r['fid'])
    with (out/'matches.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['case_id','image_id','fids','path','status','reason'])
        for e in pred['evidence']:
            if e['status']!='irrelevant':w.writerow([e['case_id'],e['image_id'],','.join(sorted(fids[e['image_id']])),e['path'],e['status'],e['reason']])
    blocks=[]
    for q in [q for q in inv['cases'] if q['id'] in pred['states']]:
        blocks.append('<h2>'+html.escape(q['id']+' '+q['query'])+'</h2><div class="grid">')
        for image_id,status in pred['states'][q['id']].items():
            if status not in ('1','?'):continue
            row=byid[image_id];path=root/'图片'/row['path'];thumb=out/'thumbs'/f'{image_id}-1200.jpg'
            reason=next((j['reason'] for j in pred['evidence'] if j['case_id']==q['id'] and j['image_id']==image_id),'')
            blocks.append(f'<article><a href="{path.as_uri()}"><img loading="lazy" src="{thumb.as_uri()}"></a><p>{image_id} / {status}</p><p>{html.escape(row["path"])}</p><p>{html.escape(reason)}</p></article>')
        blocks.append('</div>')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>图片匹配复核</title><style>body{font:15px Arial;margin:32px;background:#f4f6f8;color:#17212e}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}article{padding:12px;background:white;overflow-wrap:anywhere}img{width:100%;height:220px;object-fit:contain}p{margin:8px 0}</style><h1>前10条用例视觉匹配复核</h1><p>1=单图复核相关；?=待核实。点击图片打开原图。</p>'+''.join(blocks),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['corpus','baseline','gallery']);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=ROOT/'image-matching/runs/pilot-v1');a=p.parse_args()
    {'corpus':corpus,'baseline':baseline,'gallery':gallery}[a.stage](a.root,a.out)
