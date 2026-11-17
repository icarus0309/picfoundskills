"""Record the request-scoped content-inspection repair before any label evaluation."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'skills/image-case-retrieval/scripts'))
import pipeline as p
from hybrid_run import CODE

def main(out,name='provider-guard-01',reason=None,runtime=None):
    assert not (out/'evaluation.json').exists() and not (out/'comparison_prediction_seal.json').exists(),'Create a new experiment after evaluation'
    old=p.read_json(out/'protocol.json');source=REPO/'skills/image-case-retrieval/scripts'
    new_hashes={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in CODE}
    changed={name for name,sha in new_hashes.items() if old['code_sha256'].get(name)!=sha}
    assert changed<={'providers.py','provider_guard.py','hybrid_run.py'},('Semantic modules changed',changed)
    for module,sha in old['code_sha256'].items():assert hashlib.sha256((out/'frozen_code'/module).read_bytes()).hexdigest()==sha
    archive=out/'infrastructure_amendments'/name;archive.mkdir(parents=True,exist_ok=False)
    shutil.copytree(out/'frozen_code',archive/'previous_frozen_code');shutil.copyfile(out/'protocol.json',archive/'previous_protocol.json')
    new={**old,'code_sha256':new_hashes,'provider_block_handling':'request-scoped inspection rejection; no image resubmission; uncertain not negative'}
    runtime=runtime or {}
    assert set(runtime)<={'workers','parallel_batches','network_concurrency'} and all(isinstance(v,int) and v>0 for v in runtime.values())
    new.update(runtime)
    for module in CODE:shutil.copyfile(source/module,out/'frozen_code'/module)
    p.write_json(out/'protocol.json',new)
    p.write_json(archive/'amendment.json',{'time':time.time(),'reason':reason or 'Image content inspection failures must not stop the entire provider session or turn unreviewed images negative.',
                 'changed_modules':sorted(changed),'ground_truth_used':False,'relevance_criteria_changed':False,
                 'runtime_changes':{k:{'before':old[k],'after':v} for k,v in runtime.items() if old[k]!=v},
                 'previous_protocol_sha256':hashlib.sha256((archive/'previous_protocol.json').read_bytes()).hexdigest(),
                 'new_protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest()})
    print(json.dumps({'infrastructure_amended':True,'changed_modules':sorted(changed),'relevance_criteria_changed':False}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--name',default='provider-guard-01');parser.add_argument('--reason')
    for setting in ('workers','parallel-batches','network-concurrency'):parser.add_argument('--'+setting,type=int)
    args=parser.parse_args();runtime={k:getattr(args,k) for k in ('workers','parallel_batches','network_concurrency') if getattr(args,k) is not None};main(args.out.resolve(),args.name,args.reason,runtime)
