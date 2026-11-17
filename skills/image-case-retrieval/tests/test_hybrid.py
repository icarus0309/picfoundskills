import io,json,sys,tempfile,threading,unittest,urllib.error
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import openpyxl
from PIL import Image
import pipeline as p
from metadata_search import MetadataIndex
from providers import SingleImageReview
from rich_metadata import aggregate,structured_match
import hybrid_run
from provider_guard import BlockRegistry,GuardedVision
from providers import CompatibleAPI,ProviderContentBlocked,ProviderOutputInvalid,ProviderMediaUnavailable,RoutedAPI

class Fake:
    prompts=[]
    def __init__(self):self.stopped=threading.Event()
    def call(self,stage,prompt,images=()):
        self.prompts.append(prompt)
        if stage=='route':return {'queries':[{'id':'TC0001','directories':['动物'],'visual_criteria':'红色方块','time_constraint':'','reason':'fixture'}]}
        if stage=='metadata-plan':return {'queries':[{'id':'TC0001','subject_terms':['红色方块'],'literal':False,'reason':'fixture'}]}
        if stage=='screen':return {'candidates':[]}
        if stage=='verify-metadata':
            specs=json.loads(prompt.split('逐图逐查询完整覆盖：',1)[1])
            return {'images':[{'image_id':s['image_id'],'judgments':[{'case_id':q['id'],'status':'relevant' if s['image_id']=='I00001' else 'irrelevant','reason':'fixture image','confidence':1} for q in s['queries']]} for s in specs]}
        raise AssertionError(stage)

class HybridTests(unittest.TestCase):
    def test_media_transport_failure_abstains_without_repeating_id_repair(self):
        class Inner:
            stopped=threading.Event();calls=0
            def call(self,*args):self.calls+=1;raise ProviderMediaUnavailable()
        inner=Inner();prompt='必须逐图逐条覆盖:'+json.dumps([{'image_id':'I0','queries':[{'id':'TC1'}]}])
        result=SingleImageReview(inner).call('verify',prompt,['image'])
        self.assertEqual(inner.calls,1);self.assertTrue(result['images'][0]['judgments'][0]['media_transport_abstention'])
        self.assertEqual(result['images'][0]['judgments'][0]['status'],'uncertain')
    def test_data_uri_rejection_retries_and_diagnostics_redact_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);photo=root/'fixture.jpg';Image.new('RGB',(60,40),'red').save(photo)
            key='fixture-only-credential';endpoint='https://example.invalid/v1'
            body=json.dumps({'error':{'code':'invalid_parameter_error','message':'The provided URL does not appear to be valid. '+key+' '+endpoint}}).encode()
            error=urllib.error.HTTPError(endpoint,400,'fixture',{},io.BytesIO(body))
            response=io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"images":[]}'}}]}).encode())
            api=CompatibleAPI(root,key,endpoint,'fixture-vision')
            with patch('providers.urllib.request.urlopen',side_effect=[error,response]),patch('providers.time.sleep'):
                self.assertEqual(api.call('verify','fixture',[photo]),{'images':[]})
            self.assertFalse(api.stopped.is_set());self.assertEqual(len(list((root/'provider_media_errors').glob('*.json'))),1)
            diagnostics=''.join(p.read_text(encoding='utf-8') for p in (root/'runtime_diagnostics').glob('*.json'))
            self.assertNotIn(key,diagnostics);self.assertNotIn('example.invalid',diagnostics)
    def test_http400_json_generation_abort_retries_without_stopping_session(self):
        with tempfile.TemporaryDirectory() as directory:
            body=json.dumps({'error':{'code':'invalid_parameter_error','message':'Model output became abnormal while generating a JSON response. Please retry.'}}).encode()
            error=urllib.error.HTTPError('https://example.invalid',400,'fixture',{},io.BytesIO(body))
            response={'choices':[{'finish_reason':'stop','message':{'content':'{"images":[]}'}}]}
            class Response(io.BytesIO):
                def __enter__(self):return self
                def __exit__(self,*args):self.close()
            api=CompatibleAPI(Path(directory),'fixture-only','https://example.invalid/v1','fixture-vision')
            with patch('providers.urllib.request.urlopen',side_effect=[error,Response(json.dumps(response).encode())]),patch('providers.time.sleep'):
                self.assertEqual(api.call('verify','fixture'),{'images':[]})
            self.assertFalse(api.stopped.is_set());self.assertEqual(len(list((Path(directory)/'provider_generation_errors').glob('*.json'))),1)
    def test_route_repair_requires_original_ids_and_valid_directory_names(self):
        class Text:
            calls=[]
            def call(self,stage,prompt,images):
                self.calls.append(stage)
                return {'queries':[{'id':'wrong' if len(self.calls)==1 else 'TC1','directories':['animals']}]}
        text=Text();prompt='schema。用例:'+json.dumps([{'id':'TC1'}])+'。目录:'+json.dumps([{'name':'animals'}])
        result=RoutedAPI(text,None).call('route',prompt)
        self.assertEqual(result['queries'][0]['id'],'TC1');self.assertEqual(text.calls,['route','route-schema-retry-1'])
    def test_provider_generation_abort_is_an_abstention_without_stopping_other_work(self):
        class Inner:
            stopped=threading.Event();calls=0
            def call(self,*args):self.calls+=1;raise ProviderOutputInvalid()
        inner=Inner();prompt='必须逐图逐条覆盖:'+json.dumps([{'image_id':'I0','queries':[{'id':'TC1'}]}])
        result=SingleImageReview(inner).call('verify',prompt,['image'])
        self.assertEqual(inner.calls,3);self.assertFalse(inner.stopped.is_set());self.assertEqual(result['images'][0]['judgments'][0]['status'],'uncertain')
    def test_bounded_schema_failure_is_audited_as_uncertain(self):
        with tempfile.TemporaryDirectory() as directory:
            class Inner:
                stopped=threading.Event();calls=0
                def call(self,*args):self.calls+=1;return {'images':[]}
            inner=Inner();prompt='必须逐图逐条覆盖:'+json.dumps([{'image_id':'I0','queries':[{'id':'TC1'}]}])
            result=SingleImageReview(inner,Path(directory)).call('verify',prompt,['image'])
            self.assertEqual(inner.calls,3);self.assertEqual(result['images'][0]['judgments'][0]['status'],'uncertain')
            self.assertTrue(result['images'][0]['judgments'][0]['schema_abstention']);self.assertEqual(len(list(Path(directory).glob('*.json'))),1)
    def test_identical_duplicate_review_rows_collapse_but_conflicts_remain(self):
        a={'image_id':'I0','judgments':[{'case_id':'TC1','status':'uncertain'}]}
        b={'image_id':'I0','judgments':[{'case_id':'TC1','status':'relevant'}]}
        result=SingleImageReview.normalize_review({'images':[a,a,b]})
        self.assertEqual(result['images'],[a,b])
    def test_single_image_identity_repair_resubmits_instead_of_accepting_extra_ids(self):
        class Inner:
            stopped=threading.Event();calls=[]
            def call(self,stage,prompt,images):
                self.calls.append(stage)
                row={'image_id':'I0','judgments':[{'case_id':'TC1','status':'uncertain'}]}
                return {'images':[row,{'image_id':'I9','judgments':row['judgments']}]} if len(self.calls)==1 else {'images':[row]}
        inner=Inner();prompt='prefix 必须逐图逐条覆盖:'+json.dumps([{'image_id':'I0','queries':[{'id':'TC1'}]}])
        result=SingleImageReview(inner).call('verify',prompt,['image'])
        self.assertEqual(len(result['images']),1);self.assertEqual(result['images'][0]['judgments'][0]['status'],'uncertain')
        self.assertEqual(inner.calls,['verify','verify-identity-repair-1'])
    def test_review_normalization_preserves_judgments_and_drops_input_echoes(self):
        judgment={'case_id':'TC1','status':'uncertain','reason':'fixture'}
        raw={'images':[{'image_id':'I0','judgments':[judgment,'note']},'image_id',{'image_id':'I0','queries':[{'id':'TC1'}]}]}
        cleaned=SingleImageReview.normalize_review(raw)
        self.assertEqual(cleaned,{'images':[{'image_id':'I0','judgments':[judgment]}]})
    def test_provider_block_remains_uncertain_without_resubmission(self):
        with tempfile.TemporaryDirectory() as directory:
            class Inner:
                stopped=threading.Event();count=0
                def call(self,*args):self.count+=1;raise ProviderContentBlocked()
            inner=Inner();guard=GuardedVision(inner,BlockRegistry(Path(directory)))
            prompt='允许ID:'+json.dumps(['I0','I1'])+'。查询:'+json.dumps([{'id':'TC1'}])
            self.assertEqual(len(guard.call('screen',prompt,['plate'])['candidates']),2)
            review='必须逐图逐条覆盖:'+json.dumps([{'image_id':'I0','queries':[{'id':'TC1'}]}])
            result=guard.call('verify',review,['image'])
            self.assertEqual(result['images'][0]['judgments'][0]['status'],'uncertain');self.assertEqual(inner.count,1)
    def test_time_and_city_conditions_retrieve_without_caption(self):
        images=[{'id':'I0'},{'id':'I1'},{'id':'I2'}]
        records=[{'image_ids':['I0'],'text':'普通照片','shooting_time':'2023-01-01','attributes':{'tagIds':'2023年','shootingTime':'2023-01-01','country':'中国','province':'北京市','city':'北京市'}},
                 {'image_ids':['I1'],'text':'普通照片','shooting_time':'2023-01-01','attributes':{'tagIds':'2023年','shootingTime':'2023-01-01','city':'上海市'}},
                 {'image_ids':['I2'],'text':'普通照片','shooting_time':'2024-01-01','attributes':{'tagIds':'2024年','shootingTime':'2024-01-01','city':'北京市'}}]
        plan={'subject_terms':['北京'],'visual_terms':[],'time_years':[2023],'location_groups':[{'kind':'city','terms':['北京','Beijing']}]}
        candidates=MetadataIndex(records,images).search('2023年的北京照片',plan)
        self.assertEqual(set(candidates),{'I0'});self.assertTrue(candidates['I0']['structured']['direct_match'])
    def test_conflicting_years_are_not_automatically_positive(self):
        attributes=aggregate([{'tagIds':'2023年','shootingTime':'2024-01-01'}])
        match=structured_match(attributes,{'time_years':[2023],'visual_terms':[]})
        self.assertFalse(match['direct_match']);self.assertFalse(match['hard_time_mismatch'])
    def test_independent_image_adapter_retains_ids_and_query_specs(self):
        class Inner:
            stopped=threading.Event()
            def call(self,stage,prompt,images):
                specs=json.loads(prompt.split('必须逐图逐条覆盖:',1)[1]);assert len(images)==len(specs)==1
                return {'images':[{'image_id':specs[0]['image_id'],'judgments':[]}]}
        result=SingleImageReview(Inner()).call('verify','prefix 必须逐图逐条覆盖:'+json.dumps([{'image_id':'I0'},{'image_id':'I1'}]),['a','b'])
        self.assertEqual([r['image_id'] for r in result['images']],['I0','I1'])
    def test_phrase_recall_is_not_capped_at_top_k(self):
        images=[{'id':f'I{i:05d}'} for i in range(250)]
        metadata=[{'image_ids':[i['id']],'text':'教堂、尖塔、拱窗','shooting_time':''} for i in images]
        index=MetadataIndex(metadata,images)
        self.assertEqual(len(index.search('教堂',{'subject_terms':['教堂'],'literal':False})),250)
    def test_literal_query_does_not_expand_into_partial_matches(self):
        index=MetadataIndex([{'image_ids':['I0'],'text':'receipt mm998877','shooting_time':''},{'image_ids':['I1'],'text':'receipt mm9988','shooting_time':''}],[{'id':'I0'},{'id':'I1'}])
        self.assertEqual(set(index.search('mm998877',{'subject_terms':['receipt'],'literal':True})),{'I0'})
    def test_subject_candidate_union_does_not_require_all_words(self):
        index=MetadataIndex([{'image_ids':['I0'],'text':'梅花鹿','shooting_time':''},{'image_ids':['I1'],'text':'锦鲤','shooting_time':''}],[{'id':'I0'},{'id':'I1'}])
        self.assertEqual(set(index.search('梅花鹿和锦鲤',{'subject_terms':['梅花鹿','锦鲤']})),{'I0','I1'})
    def test_hybrid_recovers_unrouted_image_and_remains_label_blind(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'output'
            for folder,name,color in [('动物','blue.jpg','blue'),('物品','red.jpg','red')]:
                parent=root/'图片'/folder;parent.mkdir(parents=True);Image.new('RGB',(60,40),color).save(parent/name)
            wb=openpyxl.Workbook();sheet=wb.active;sheet.append(['id','query','unused','expected']);sheet.append(['TC0001','红色方块',None,'label-not-for-inference']);wb.save(root/'E2E_1321cases_1w_data_v19.xlsx');wb.close()
            wb=openpyxl.Workbook();sheet=wb.active;sheet.append(['fid','fileName','fileSize','shootingTime','description'])
            for folder,name,desc in [('动物','blue.jpg','蓝色方块'),('物品','red.jpg','红色方块')]:sheet.append(['opaque-'+name,name,(root/'图片'/folder/name).stat().st_size,'2024-01-01',desc])
            wb.save(root/'ES_byFileType_user10_20261117170358.xlsx');wb.close()
            (root/'fixture.key').write_text('fixture-only',encoding='utf-8')
            config=root/'config.json';config.write_text(json.dumps({'base_url':'https://example.invalid/v1','model':'fixture-vision','key_file':'fixture.key'}),encoding='utf-8')
            args=hybrid_run.args_parser().parse_args(['--root',str(root),'--out',str(out),'--limit','1','--vlm-config',str(config)])
            Fake.prompts=[];hybrid_run.execute(args,api_factories=lambda _: (Fake(),Fake()))
            base=p.read_json(out/'batch-001-001/visual-v1/predictions.json');new=p.read_json(out/'batch-001-001/hybrid-v2/predictions.json')
            self.assertFalse(any(s=='1' for s in base['states']['TC0001'].values()))
            self.assertEqual(new['states']['TC0001']['I00001'],'1')
            self.assertTrue(p.read_json(out/'prediction_completion.json')['complete'])
            self.assertTrue(all('label-not-for-inference' not in text for text in Fake.prompts))
            self.assertFalse((out/'evaluation.json').exists())
            count=len(Fake.prompts);hybrid_run.execute(args,api_factories=lambda _: (Fake(),Fake()));self.assertEqual(len(Fake.prompts),count)
            (out/'batch-001-001/visual-v1/predictions.json').write_text('{}',encoding='utf-8')
            with self.assertRaises(RuntimeError):hybrid_run.execute(args,api_factories=lambda _: (Fake(),Fake()))
            self.assertFalse((out/'evaluation.json').exists());self.assertEqual(len(Fake.prompts),count)

if __name__=='__main__':unittest.main()
