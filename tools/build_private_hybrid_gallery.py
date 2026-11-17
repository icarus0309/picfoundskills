"""Build a local, label-free visual review page. Never publish this private HTML."""
import argparse,html,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'skills/image-case-retrieval/scripts'))
import pipeline as p

def main(root,out,thumbs):
    inv=p.read_json(out/'inventory.json');images={r['id']:r for r in inv['images']}
    if (out/'selected_cases.json').exists():selected=p.read_json(out/'selected_cases.json')['cases']
    else:
        scope=p.read_json(out/'protocol.json')['scope'];selected=inv['cases'][scope['offset']:scope['offset']+scope['limit']]
    baseline={};hybrid={};proof={}
    for batch in sorted(out.glob('batch-*')):
        for name,target in [('visual-v1',baseline),('hybrid-v2',hybrid)]:
            file=batch/name/'predictions.json'
            if not file.exists():continue
            data=p.read_json(file);target.update(data['states'])
            if name=='hybrid-v2':proof.update({(e['case_id'],e['image_id']):e for e in data['evidence']})
    parts=['<!doctype html><meta charset="utf-8"><title>时间地点混合检索复核</title><style>body{font:15px Arial;margin:24px;background:#f3f6fa;color:#17212e}details{padding:14px;margin:12px 0;background:white}summary{font-size:18px;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}article{border:1px solid #d5deea;padding:10px;overflow-wrap:anywhere}img{width:100%;height:210px;object-fit:contain}input{padding:10px;width:400px}.new{color:#12634b;font-weight:bold}</style><h1>时间与地点图片检索复核</h1><p>本地查看页面，不读取用例答案。显示两种方法的正关联，以及优化版新增图片的确认来源。未完成用例单列，不当作未命中。</p><input id="filter" placeholder="按用例ID或查询筛选">']
    n=0
    for case in selected:
        q=case['id'];base={i for i,s in baseline.get(q,{}).items() if s=='1'};new={i for i,s in hybrid.get(q,{}).items() if s=='1'}
        if q not in hybrid:parts.append('<details><summary>'+html.escape(q+' '+case['query']+'：预测尚未完成')+'</summary></details>');continue
        n+=1;parts.append('<details><summary>'+html.escape(f"{q} {case['query']} / 基线 {len(base)} / 混合 {len(new)} / 新增 {len(new-base)}")+'</summary><div class="grid">')
        for image in sorted(new,key=lambda i:(i in base,i)):
            e=proof.get((q,image),{});original=root/'图片'/images[image]['path'];thumb=thumbs/f'{image}-1200.jpg'
            if not thumb.exists():thumb=p.thumbnail(root,out,images[image])
            badge='新增' if image not in base else '原视觉分支'
            parts.append('<article><a href="'+original.as_uri()+'"><img loading="lazy" src="'+thumb.as_uri()+'"></a><p class="new">'+html.escape(image+' '+badge)+'</p><p>'+html.escape(e.get('verification_method','visual_baseline'))+'</p><p>'+html.escape(e.get('reason',''))+'</p></article>')
        parts.append('</div></details>')
    parts.append('<script>document.getElementById("filter").addEventListener("input",e=>{const q=e.target.value.toLowerCase();document.querySelectorAll("details").forEach(d=>d.hidden=!d.querySelector("summary").textContent.toLowerCase().includes(q));});</script>')
    (out/'review.html').write_text(''.join(parts),encoding='utf-8');print(json.dumps({'completed_queries':n,'selected_queries':len(selected)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--thumbnail-cache',type=Path,required=True);args=parser.parse_args();main(args.root.resolve(),args.out.resolve(),args.thumbnail_cache.resolve())
