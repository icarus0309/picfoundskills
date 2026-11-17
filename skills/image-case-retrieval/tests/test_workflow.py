import concurrent.futures as cf,http.client,sys,tempfile,unittest,urllib.error
from pathlib import Path
from unittest.mock import MagicMock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import openpyxl
from PIL import Image
import pipeline as p
import run
from evaluator import evaluate,score_case

class FakeAPI:
    prompts=[]
    def __init__(self,*args,**kwargs):pass
    def call(self,stage,prompt,images=()):
        self.prompts.append(prompt)
        if stage=='route':return {'queries':[{'id':'TC0001','directories':['动物'],'visual_criteria':'红色方块','time_constraint':'','reason':'fixture'}]}
        if stage=='screen':return {'candidates':[{'image_id':'I00000','case_ids':['TC0001']}]}
        if stage=='verify':return {'images':[{'image_id':'I00000','judgments':[{'case_id':'TC0001','status':'relevant','reason':'red fixture','confidence':1.0}]}]}
        raise AssertionError(stage)

class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.out=self.root/'out'
    def tearDown(self):self.temp.cleanup()
    def fixture(self):
        photos=self.root/'图片/动物';photos.mkdir(parents=True)
        for name,color in [('a.jpg','red'),('b.jpg','blue')]:Image.new('RGB',(60,40),color).save(photos/name)
        wb=openpyxl.Workbook();ws=wb.active;ws.append(['id','query','unused','expected']);ws.append(['TC0001','红色方块',None,'fid-known,fid-absent']);wb.save(self.root/'E2E_1321cases_1w_data_v19.xlsx');wb.close()
        wb=openpyxl.Workbook();ws=wb.active;ws.append(['fid','fileName','fileSize','shootingTime','description']);ws.append(['fid-known','a.jpg',(photos/'a.jpg').stat().st_size,'2024-01-01','fixture metadata']);wb.save(self.root/'ES_byFileType_user10_20261117170358.xlsx');wb.close()
        return run.parser().parse_args(['--root',str(self.root),'--out',str(self.out),'--limit','1'])
    def test_any_hit_preserves_missing_label(self):
        m=score_case(['a','b','c'],{'a':['I1'],'b':['I2']},{'I2','other'})
        self.assertTrue(m['hit_any']);self.assertEqual(m['unmapped_fids'],1);self.assertAlmostEqual(m['known_fid_recall_all'],1/3)
    def test_empty_labels_are_not_scored_as_negatives(self):
        self.assertIsNone(score_case([],{},set())['known_fid_recall_all'])
    def test_multiple_local_matches_for_one_fid(self):
        self.assertEqual(score_case(['a'],{'a':['I1','I2']},{'I2'})['hit_fids'],1)
    def test_reader_requests_only_query_columns(self):
        wb=MagicMock();ws=wb.worksheets[0];ws.iter_rows.return_value=[('TC0001','query')]
        with patch.object(p.openpyxl,'load_workbook',return_value=wb):p.read_cases(self.root)
        self.assertEqual(ws.iter_rows.call_args.kwargs['max_col'],2)
    def test_end_to_end_export_then_evaluate_and_resume(self):
        args=self.fixture();FakeAPI.prompts=[];run.run(args,api_factory=FakeAPI)
        self.assertTrue((self.out/'predictions.json').exists());self.assertTrue((self.out/'review.html').exists())
        self.assertFalse((self.out/'evaluation.json').exists())
        self.assertTrue(all('fid-known' not in text and 'fid-absent' not in text for text in FakeAPI.prompts))
        count=len(FakeAPI.prompts);run.run(args,api_factory=FakeAPI);self.assertEqual(len(FakeAPI.prompts),count)
        args.stage='evaluate';m=run.run(args)
        self.assertEqual(m['hit_cases'],1);self.assertEqual(m['known_fid_total'],2);self.assertEqual(m['known_fid_hits'],1)
    def test_input_drift_rejected_before_model_calls(self):
        args=self.fixture();run.run(args,api_factory=FakeAPI)
        Image.new('RGB',(60,40),'green').save(self.root/'图片/动物/c.jpg')
        with self.assertRaisesRegex(ValueError,'Image list changed'):run.run(args,api_factory=FakeAPI)
    def test_incomplete_screening_cannot_be_exported(self):
        args=self.fixture();args.stage='prepare';run.run(args,api_factory=FakeAPI);args.stage='export'
        with self.assertRaisesRegex(RuntimeError,'Screening is incomplete'):run.run(args)
    def test_concurrent_readers_receive_complete_jpeg(self):
        photos=self.root/'图片';photos.mkdir();Image.new('RGB',(500,300),'red').save(photos/'image.jpg')
        row={'id':'I00000','path':'image.jpg'}
        def read(_):
            file=p.thumbnail(self.root,self.out,row)
            with Image.open(file) as image:image.load();return image.size
        with cf.ThreadPoolExecutor(max_workers=16) as pool:self.assertEqual(list(pool.map(read,range(32))),[(500,300)]*32)
    def test_truncated_http_response_has_bounded_retries(self):
        with patch('pipeline.urllib.request.urlopen',side_effect=http.client.IncompleteRead(b'')) as request,patch('pipeline.time.sleep'):
            with self.assertRaisesRegex(RuntimeError,'failed after 3'):p.API(self.out,'fixture-only').call('test','{}')
            self.assertEqual(request.call_count,3)
    def test_auth_failure_stops_following_requests(self):
        error=urllib.error.HTTPError('https://api.deepseek.com/chat/completions',401,'fixture',{},None)
        with patch('pipeline.urllib.request.urlopen',side_effect=error) as request:
            api=p.API(self.out,'fixture-only')
            with self.assertRaisesRegex(RuntimeError,'HTTP 401'):api.call('test','{}')
            with self.assertRaisesRegex(RuntimeError,'stopped'):api.call('test','again')
            self.assertEqual(request.call_count,1)

if __name__=='__main__':unittest.main()
