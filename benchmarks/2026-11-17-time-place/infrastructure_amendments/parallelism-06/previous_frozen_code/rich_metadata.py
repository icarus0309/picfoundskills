"""Named-source time and location attributes from the image index, never case labels."""
import collections,re
from pathlib import Path
import openpyxl
import pipeline as p

FIELDS=('tagIds','tag解析','shootingTime','country','province','city','business_terms','place_terms')

def corpus(root,out):
    inv=p.read_json(out/'inventory.json');names=collections.defaultdict(list)
    for image in inv['images']:names[Path(image['path']).name.casefold()].append(image)
    wb=openpyxl.load_workbook(root/'ES_byFileType_user10_20261117170358.xlsx',read_only=True,data_only=True);records=[]
    for sheet in wb:
        rows=sheet.iter_rows(values_only=True);headers=next(rows)
        for values in rows:
            row=dict(zip(headers,values));matches=names.get(str(row.get('fileName') or '').casefold(),[])
            exact=[image for image in matches if str(image['size'])==str(row.get('fileSize'))];mapped=exact or matches
            attributes={field:str(row.get(field) or '') for field in FIELDS}
            text=' '.join(str(row.get(field) or '') for field in ('tagIds','tag解析','country','province','city','business_terms','place_terms','description','ocr','aiDescription','keyword','abstract','search_term','summary'))
            records.append({'fid':str(row.get('fid') or ''),'filename':row.get('fileName'),'image_ids':[i['id'] for i in mapped],
                            'mapping':'size_and_name' if exact else 'unique_name' if len(matches)==1 else 'ambiguous_name' if matches else 'missing',
                            'text':text,'shooting_time':attributes['shootingTime'],'attributes':attributes})
    wb.close();p.write_json(out/'corpus_metadata.json',records)
    print('Rich metadata extracted; time and location source fields preserved.',flush=True)
    return records

def years(text):return sorted({int(y) for y in re.findall(r'(?<!\d)((?:19|20)\d{2})(?:年)?(?!\d)',str(text))})

def aggregate(rows):
    result={field:sorted({r.get(field,'') for r in rows if r.get(field)}) for field in FIELDS}
    result['tag_years']=sorted({year for r in rows for year in years(r.get('tagIds',''))})
    result['shooting_years']=sorted({year for r in rows for year in years(r.get('shootingTime','')[:4])})
    result['year_conflict']=bool(result['tag_years'] and result['shooting_years'] and set(result['tag_years'])!=set(result['shooting_years']))
    return result

def normalized_place(text):return re.sub(r'[\s,，;；、()（）]','',str(text).casefold())

def structured_match(attributes,plan):
    """Return support only for fully explicit metadata conditions, never inferred geography."""
    years_needed=set(plan.get('time_years',[]));locations=plan.get('location_groups',[])
    visual_terms=[t for t in plan.get('visual_terms',[]) if t.strip()]
    available=set(attributes.get('tag_years',[]))|set(attributes.get('shooting_years',[]))
    time_match=not years_needed or bool(years_needed & available) and not attributes['year_conflict']
    # Relative/holiday/month constraints require separate evidence review, not year-only acceptance.
    extra_time=bool(plan.get('time_other_constraints'))
    place_support=[]
    for group in locations:
        kind=group.get('kind','any');terms=[normalized_place(t) for t in group.get('terms',[]) if t]
        fields={'country':('country',),'province':('province',),'city':('city','province'),
                'business':('business_terms','place_terms'),'poi':('place_terms','business_terms')}.get(kind,('country','province','city','business_terms','place_terms'))
        values=[normalized_place(value) for field in fields for value in attributes.get(field,[])]
        if kind in ('country','province','city'):
            canonical=lambda text:re.sub(r'(省|市|自治区|特别行政区)$','',text)
            supported=any(canonical(term)==canonical(value) for term in terms for value in values)
        else:supported=any(term in value for term in terms for value in values)
        place_support.append(supported)
    condition_present=bool(years_needed or locations)
    direct=condition_present and not visual_terms and not extra_time and time_match and all(place_support)
    hard_time_mismatch=bool(years_needed and available and not years_needed & available and not attributes['year_conflict'])
    return {'direct_match':direct,'time_match':time_match,'location_group_support':place_support,'hard_time_mismatch':hard_time_mismatch}
