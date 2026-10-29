"""Read-only replay/cost audit of run-001, with a separate durable report."""
import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
import numpy as np
from diagnostics import (HERE,ROOT,SCENES,COUNTS,map_mode,top2,Q_pvalue,
                          physical_modes,propose,oracle_indices,sha,dump)


def run_review():
    out=HERE/'results/run-001'
    manifest=json.loads((out/'evidence.json').read_text())
    for path,digest in manifest['sha256'].items():
        if sha(out/path)!=digest: raise AssertionError('archive mismatch '+path)
    config=json.loads((out/'config.json').read_text())
    updated_sources=[]
    for path,digest in config['sources'].items():
        # Run-001 certifies its committed source and snapshot, not later tests.
        content=subprocess.check_output(['git','show',config['code_commit']+':'+path],cwd=ROOT)
        if hashlib.sha256(content).hexdigest()!=digest:
            raise AssertionError('source commit mismatch '+path)
        if sha(out/'source_snapshot'/Path(path).name)!=digest:
            raise AssertionError('snapshot mismatch '+path)
        if sha(ROOT/path)!=digest:
            if Path(path).name!='test_diagnostics.py':
                raise AssertionError('scientific source changed '+path)
            updated_sources.append(path)
    for path,digest in config['input_hashes'].items():
        if sha(ROOT/path)!=digest: raise AssertionError('input changed '+path)
    data=np.load(out/'mode_energies.npz')['energy']
    replay=0
    for si,(_,truth,sigma) in enumerate(SCENES):
        _,_,a,_=physical_modes(sigma)
        for rep in range(128):
            for part in range(2):
                rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261029,si,rep,part])))
                sample=rng.chisquare(COUNTS[:,None],size=(8,2))*(1+np.array(truth)[:,None]*a)
                np.testing.assert_array_equal(sample,data[si,rep,part]); replay+=1
    by_scene=json.loads((out/'candidate_summary.json').read_text())
    original=json.loads((out/'trial_records.json').read_text())
    for row in original:
        si=[s[0] for s in SCENES].index(row['scene']); rep=row['replicate']
        _,truth,sigma=SCENES[si]; truth=np.array(truth); true=truth>1
        _,_,a,_=physical_modes(sigma)
        ep,ea=data[si,rep]; method=row['method']
        if method.startswith('split_'):
            if method=='split_oracle':
                chosen=oracle_indices(truth,by_scene[row['scene']]['split_power'])
            else: chosen=propose(ep,COUNTS//2,a)[method[6:]]
            p=[Q_pvalue(e,int(n),a) for e,n in zip(ea,COUNTS//2)]; alpha=.025
        else:
            p=[Q_pvalue(e,int(n),a) for e,n in zip(ep+ea,COUNTS)]; alpha=.00625
            if method=='full_all8': chosen=list(range(8))
            elif method=='full_oracle':
                chosen=oracle_indices(truth,by_scene[row['scene']]['full_family_power'])
            else: chosen=propose(ep+ea,COUNTS,a)['score']
        supported=[j for j in chosen if p[j]<=alpha]
        assert chosen==row['selected'] and supported==row['supported']
        assert sum(true[j] for j in supported)==row['true_support']
        assert sum(not true[j] for j in supported)==row['false_support']
    # Cost supplement uses the first archived replicate, no new observations.
    # Each rule is timed separately to avoid attributing MAP work to score.
    timing={name:[] for name in ('fixed','map','score','fisher','audit_all8')}
    for si,(_,_,sigma) in enumerate(SCENES):
        ep,ea=data[si,0]; n=COUNTS//2; _,_,a,_=physical_modes(sigma); r=a/(1+a)
        funcs={
            'fixed': lambda: [0,1],
            'map': lambda: top2([map_mode(e,int(k),a)[0] for e,k in zip(ep,n)]),
            'score': lambda: top2(.5*np.sum((ep/(1+a)-2*n[:,None])*r,axis=1)/np.sqrt(n*np.sum(r*r))),
            'fisher': lambda: top2(n*np.sum(r*r)),
            'audit_all8': lambda: [Q_pvalue(e,int(k),a) for e,k in zip(ea,n)]}
        expected=propose(ep,n,a)
        for name,func in funcs.items():
            for _ in range(10):
                start=time.process_time_ns(); result=func()
                timing[name].append((time.process_time_ns()-start)*1e-9)
                if name!='audit_all8': assert result==expected[name]
    base='b57ba0e8bbe2c930a7b473a34a0d90901708bda1'
    changed=subprocess.check_output(['git','diff',base,'--name-only'],cwd=ROOT,text=True).splitlines()
    prefix=str(HERE.relative_to(ROOT))+'/'
    assert all(p.startswith(prefix) for p in changed),changed
    status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    assert all(line[3:].startswith(prefix) for line in status.splitlines()),status
    return dict(status='passed',replayed_seed_parts=replay,verified_trial_rows=len(original),
        later_test_only_source_changes=updated_sources,
        archive_files_verified=len(manifest['sha256']),baseline=base,
        changed_paths=changed,protected_historical_paths_unchanged=True,
        timing={k:dict(median_seconds=statistics.median(v),min_seconds=min(v),
                       max_seconds=max(v),measurements=len(v)) for k,v in timing.items()},
        config_sha256=sha(out/'config.json'),evidence_sha256=sha(out/'evidence.json'),
        observations_sha256=sha(out/'mode_energies.npz'),
        review_source_sha256=sha(Path(__file__)),
        classification='exploratory output replay and timing audit; not independent research validation')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',required=True); args=parser.parse_args()
    target=HERE/'verification'/(args.tag+'.json')
    if target.exists(): raise FileExistsError('never overwrite review evidence')
    try:
        result=run_review()
    except Exception as exc:
        dump(target,dict(status='failed',error=repr(exc))); raise
    dump(target,result); print(json.dumps(result,indent=2))
