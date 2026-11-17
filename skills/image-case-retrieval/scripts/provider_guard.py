"""Respect image-inspection rejections without turning unreviewed content negative."""
import collections,hashlib,json,threading,time
from providers import ProviderContentBlocked,ProviderOutputInvalid,ProviderMediaUnavailable
import pipeline as p

class BlockRegistry:
    def __init__(self,out):
        self.file=out/'provider_blocks.json';self.lock=threading.Lock()
        data=p.read_json(self.file) if self.file.exists() else {}
        self.images=set(data.get('blocked_image_ids',[]));self.events=data.get('events',[]);self.reports=set(data.get('processed_reports',[]))
    def _save(self):
        p.write_json(self.file,{'blocked_image_ids':sorted(self.images),'events':self.events,'processed_reports':sorted(self.reports),
                     'policy':'An image rejection of a composite conservatively blocks every constituent image; blocked images are not resubmitted.'})
    def _refresh(self):
        changed=False
        for file in (self.file.parent/'provider_block_reports').glob('*.json'):
            if file.name in self.reports:continue
            report=p.read_json(file);self.images.update(report['image_ids']);self.events.append(report);self.reports.add(file.name);changed=True
        if changed:self._save()
    def contains(self,image):
        with self.lock:
            self._refresh();return image in self.images
    def register(self,images,stage,media,provider='qwen'):
        with self.lock:
            self._refresh()
            if media=='image':self.images.update(images)
            self.events.append({'image_ids':sorted(images),'stage':stage,'media':media,'provider':provider,'time':time.time(),'action':'no resubmission; uncertain, not negative'})
            self._save()

class GuardedVision:
    def __init__(self,inner,registry):self.inner=inner;self.registry=registry
    @property
    def stopped(self):return self.inner.stopped
    def call(self,stage,prompt,images=()):
        if stage.startswith('screen'):
            ids,tail=json.JSONDecoder().raw_decode(prompt.split('允许ID:',1)[1])
            raw_queries=prompt.split('。查询:',1)[1];queries,_=json.JSONDecoder().raw_decode(raw_queries)
            qids=[q['id'] for q in queries]
            def uncertain_candidates():return {'candidates':[{'image_id':image,'case_ids':qids} for image in ids]}
            if any(self.registry.contains(image) for image in ids):return uncertain_candidates()
            try:return self.inner.call(stage,prompt,images)
            except (ProviderOutputInvalid,ProviderMediaUnavailable):return uncertain_candidates()
            except ProviderContentBlocked as blocked:
                self.registry.register(ids,stage,blocked.media);return uncertain_candidates()
        if stage.startswith('verify'):
            marker='逐图逐查询完整覆盖：' if stage.startswith('verify-metadata') else '必须逐图逐条覆盖:'
            specs,_=json.JSONDecoder().raw_decode(prompt.split(marker,1)[1])
            ids=[s['image_id'] for s in specs]
            def uncertain():
                return {'images':[{'image_id':s['image_id'],'judgments':[{'case_id':q.get('id',q.get('case_id')),'status':'uncertain','reason':'服务商图像审核拦截，未取得视觉结论','confidence':0.0} for q in s['queries']]} for s in specs]}
            if any(self.registry.contains(image) for image in ids):return uncertain()
            try:return self.inner.call(stage,prompt,images)
            except ProviderContentBlocked as blocked:
                self.registry.register(ids,stage,blocked.media);return uncertain()
        return self.inner.call(stage,prompt,images)
