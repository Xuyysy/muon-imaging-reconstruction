"""Static delivery audit: hashes, baseline preservation, links and CPU ledger."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from diagnostics import HERE, ROOT, dump, sha


def main():
    target=HERE/'verification/final-archive-001.json'
    if target.exists(): raise FileExistsError('do not overwrite final audit')
    base='b57ba0e8bbe2c930a7b473a34a0d90901708bda1'
    records=subprocess.check_output(['git','ls-tree','-rz',base],cwd=ROOT).split(b'\0')
    protected=0
    for record in records:
        if not record: continue
        info,name=record.split(b'\t',1); mode,kind,expected=info.split()
        if kind!=b'blob': raise AssertionError('unexpected non-blob')
        path=ROOT/os.fsdecode(name)
        raw=os.fsencode(os.readlink(path)) if mode==b'120000' else path.read_bytes()
        digest=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        assert digest==expected.decode(),os.fsdecode(name)
        protected+=1
    run=HERE/'results/run-001'
    manifest=json.loads((run/'evidence.json').read_text())
    for name,digest in manifest['sha256'].items(): assert sha(run/name)==digest,name
    links=0
    for path in HERE.glob('*.md'):
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            if '://' in link: continue
            assert (path.parent/link.split('#')[0]).exists(),(str(path),link)
            links+=1
    assert json.loads((HERE/'verification/archive-review-002.json').read_text())['status']=='passed'
    assert 'Ran 20 tests' in (HERE/'verification/tests-review-001.log').read_text()
    ledger=[json.loads(p.read_text()) for p in (HERE/'verification').glob('*.json')]
    cpu=sum(x.get('child_cpu_seconds',0) for x in ledger)
    assert cpu<3600
    diff=subprocess.check_output(['git','diff',base,'--name-only'],cwd=ROOT,text=True)
    prefix=str(HERE.relative_to(ROOT))+'/'
    assert all(p.startswith(prefix) for p in diff.splitlines())
    dump(target,dict(status='passed',baseline=base,protected_blobs=protected,
        local_markdown_links_checked=links,prior_measured_child_cpu_seconds=cpu,
        note='final command own child CPU is recorded by delivery-command-001.json after exit',
        files_sha256={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.rglob('*'))
            if p.is_file() and p.name not in ('final-archive-001.json','delivery-command-001.json','delivery-command-001.log')}))
    print(json.dumps(dict(status='passed',protected_blobs=protected,
                         local_links=links,prior_child_cpu_seconds=cpu),indent=2))


if __name__=='__main__': main()
