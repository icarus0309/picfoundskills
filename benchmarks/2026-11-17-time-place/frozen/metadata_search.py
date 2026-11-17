"""Label-blind full-library phrase search plus Chinese-bigram BM25 expansion."""
import collections,hashlib,json,math,re,unicodedata
import pipeline as p
from rich_metadata import aggregate,structured_match

VERSION='metadata-union-v2'
def normalize(text):return unicodedata.normalize('NFKC',str(text)).casefold()
def grams(text):
    text=re.sub(r'\s+','',normalize(text))
    return [text[i:i+2] for i in range(len(text)-1)]

class MetadataIndex:
    def __init__(self,metadata,images):
        self.ids=[r['id'] for r in images];parts=collections.defaultdict(set);dates=collections.defaultdict(set);attributes=collections.defaultdict(list)
        for row in metadata:
            for image in row['image_ids']:
                if row.get('text'):parts[image].add(row['text'])
                if row.get('shooting_time'):dates[image].add(row['shooting_time'])
                attributes[image].append(row.get('attributes',{'tagIds':'','shootingTime':row.get('shooting_time','')}))
        self.original={image:'\n'.join(sorted(parts[image])) for image in self.ids}
        self.texts={image:normalize(self.original[image]) for image in self.ids};self.dates={i:sorted(dates[i]) for i in self.ids}
        self.attributes={image:aggregate(attributes[image]) for image in self.ids}
        self.postings=collections.defaultdict(dict);self.lengths={};self.average=1
        for image,text in self.texts.items():
            counts=collections.Counter(grams(text));self.lengths[image]=sum(counts.values())
            for term,tf in counts.items():self.postings[term][image]=tf
        self.average=sum(self.lengths.values())/max(1,len(self.ids)) or 1
    def phrase(self,term):
        term=normalize(term).strip()
        if not term:return set()
        tokens=set(grams(term))
        if not tokens:return {i for i,t in self.texts.items() if term in t}
        groups=[self.postings.get(g,{}) for g in tokens]
        if any(not g for g in groups):return set()
        groups.sort(key=len);found=set(groups[0])
        for group in groups[1:]:found.intersection_update(group)
        return {i for i in found if term in self.texts[i]}
    def search(self,query,plan):
        literal=plan.get('literal',False)
        terms=list(dict.fromkeys(t.strip() for t in plan.get('subject_terms',[]) if isinstance(t,str) and t.strip()))
        if literal:terms=[query.strip()]
        candidates=collections.defaultdict(lambda:{'matched_terms':[],'sources':[],'bm25':0.0})
        for term in terms:
            for image in self.phrase(term):
                candidates[image]['matched_terms'].append(term)
                if 'phrase' not in candidates[image]['sources']:candidates[image]['sources'].append('phrase')
        if not literal:
            # No fixed top-k: retain every exact-subject hit and all sufficiently
            # strong fuzzy lexical matches. Conditions beyond subjects are checked visually.
            tokens=set(g for term in terms for g in grams(term));scores=collections.Counter();overlap=collections.Counter();n=len(self.ids)
            for term in tokens:
                posting=self.postings.get(term,{})
                if not posting:continue
                idf=math.log(1+(n-len(posting)+0.5)/(len(posting)+0.5))
                for image,tf in posting.items():
                    scores[image]+=idf*tf*2.2/(tf+1.2*(.25+.75*self.lengths[image]/self.average));overlap[image]+=1
            cutoff=max(scores.values(),default=0)*.45
            for image,score in scores.items():
                if score>=cutoff and overlap[image]>=min(2,len(tokens)):
                    candidates[image]['bm25']=round(score,5)
                    if 'bm25' not in candidates[image]['sources']:candidates[image]['sources'].append('bm25')
        # Metadata conditions can retrieve uncaptained images. Explicit year
        # mismatches are excluded; unknown or conflicting years remain eligible.
        if plan.get('time_years') or plan.get('location_groups'):
            for image,attributes in self.attributes.items():
                match=structured_match(attributes,plan)
                if match['direct_match'] or (plan.get('location_groups') and all(match['location_group_support']) and not match['hard_time_mismatch']):
                    if 'structured' not in candidates[image]['sources']:candidates[image]['sources'].append('structured')
                elif plan.get('time_years') and match['time_match'] and not plan.get('location_groups') and not plan.get('visual_terms'):
                    if 'structured_time' not in candidates[image]['sources']:candidates[image]['sources'].append('structured_time')
            for image in list(candidates):
                match=structured_match(self.attributes[image],plan)
                if match['hard_time_mismatch']:del candidates[image]
                else:candidates[image]['structured']=match
        return dict(sorted(candidates.items()))
    def excerpt(self,image,terms,limit=1200):
        text=self.original.get(image,'');folded=normalize(text);chunks=[]
        for term in terms:
            index=folded.find(normalize(term))
            if index>=0:chunks.append(text[max(0,index-60):index+240])
        return (' … '.join(dict.fromkeys(chunks)) or text[:limit])[:limit]

def plan(api,cases,out,reference_date='2026-11-17'):
    file=out/'metadata_plan.json';signature=hashlib.sha256(json.dumps(cases,ensure_ascii=False).encode()).hexdigest()
    if file.exists():
        result=p.read_json(file)
        if result['case_signature']!=signature:raise ValueError('Metadata scope changed; use a new output directory')
        return result
    literal={q['id']:bool(re.match(r'https?://',q['query'].strip(),re.I) or re.fullmatch(r'(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9_-]{5,}',q['query'].strip())) for q in cases}
    prompt='为图片已有描述/OCR及拍摄元信息设计高召回检索条件，不提供或读取答案标签。每个查询输出JSON {"queries":[{"id":"...","literal":false,"subject_terms":["核心词","可靠别名"],"visual_terms":["必须从图像确认的对象或属性"],"time_years":[2023],"time_other_constraints":"年份之外的时段/节日等，没有则空字符串","location_groups":[{"kind":"country|province|city|business|poi|any","terms":["原名","同义别名"]}],"reason":"..."}]}。参考日期：'+reference_date+'。地理字段包括country、province、city、business_terms、place_terms，国家可能中英文混用，因此提供可靠中英文别名。每个location_group表示一个必要的具名地点，组内terms为同一实体别名，组间同时满足；不要把具体地标的上级城市误当该地标别名。subject_terms通常3至12个，不得将明确实体泛化为任意建筑/动物。视觉对象查询须填visual_terms，例如2023年猫需要猫；只要求某年份或某地点照片时visual_terms为空，地点或日期本身不应强制可见招牌或地标。具名地标照片可用明确place_terms支持，但画面若另有对象条件须检查。时间同时读取tagIds中年份与shootingTime；查询原文的明确年份必须保留，其他时间约束如月份、节日、最近、季节填time_other_constraints，不得仅用年份判断。未指定时间/地点时对应字段为空，不得虚构。不读取文件名猜图。随机字符串或URL须literal=true，仅subject_terms保留原文，其他条件为空。多个主体用于候选可分别取词，但最终必要条件不应放宽。查询：'+json.dumps(cases,ensure_ascii=False)
    result=api.call('metadata-plan',prompt)
    def valid(obj):return isinstance(obj.get('queries'),list) and len(obj['queries'])==len(cases) and {q.get('id') for q in obj['queries']}=={q['id'] for q in cases} and all(isinstance(q.get('subject_terms'),list) and q['subject_terms'] and all(isinstance(t,str) and t.strip() for t in q['subject_terms']) for q in obj['queries'])
    if not valid(result):result=api.call('metadata-plan-retry',prompt+' 必须逐条覆盖且subject_terms非空。')
    if not valid(result):raise ValueError('Metadata planning schema invalid')
    for q in result['queries']:
        if literal[q['id']]:q['literal']=True;q['subject_terms']=[next(c['query'] for c in cases if c['id']==q['id'])]
        query=next(c['query'] for c in cases if c['id']==q['id'])
        explicit_years=sorted({int(y) for y in re.findall(r'(?<!\d)((?:19|20)\d{2})(?:年)?(?!\d)',query)})
        if explicit_years:q['time_years']=explicit_years
        q.setdefault('time_years',[]);q.setdefault('time_other_constraints','');q.setdefault('location_groups',[]);q.setdefault('visual_terms',[query])
    result['case_signature']=signature;result['version']=VERSION;p.write_json(file,result);return result
