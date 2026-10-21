"""Archive the complete unit suite and injected gate failures; never run a study."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
BASE="35e6553b24cb5db880944b3938250fc3aef2fad3"
PROTECTED=("sci_method/ADAPTIVE_AUDIT_FIX_PLAN.md",
    "sci_method/studies/adaptive-prototype-001/results",
    "sci_method/studies/adaptive-audit-fix-001/results",
    "sci_method/studies/optimizer-stability-001/results",
    "sci_method/studies/optimizer-stability-001/verification")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git",*args],cwd=REPO,text=True).strip()


def protected_hashes():
    files=git("ls-tree","-r","--name-only",BASE,"--",*PROTECTED).splitlines()
    return {name:sha(REPO/name) for name in files}


def run(run_id):
    if not run_id.startswith("run-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in run_id):
        raise ValueError("invalid run ID")
    output=ROOT/"studies/low-budget-execution-gate-001/unit-tests"/run_id
    output.mkdir(parents=True,exist_ok=False)
    before=protected_hashes()
    sources=[Path(__file__),ROOT/"studies/low_budget_execution_gate.py",
        ROOT/"studies/run_adaptive_audit_scenarios.py",ROOT/"studies/run_adaptive_split_merge.py",
        ROOT/"src/sci_method/covariance_optimizer.py",ROOT/"src/sci_method/split_merge.py",
        ROOT/"src/sci_method/forward_model.py",ROOT/"src/sci_method/identifiability.py",
        ROOT/"studies/adaptive_audit_analysis.py",ROOT/"src/sci_method/study_io.py",
        ROOT/"tests/test_low_budget_execution_gate.py",ROOT/"LOW_BUDGET_EXECUTION_GATE_PLAN.md",
        ROOT/"ADAPTIVE_AUDIT_FIX_PLAN.md"]
    snapshot=output/"source_snapshot";snapshot.mkdir()
    for source in sources:
        shutil.copyfile(source,snapshot/source.name)
    environment=os.environ.copy()
    environment["MUON_GATE_TEST_EVIDENCE_DIR"]=str(output/"fixtures")
    command=[sys.executable,"-m","unittest","discover","-s","sci_method/tests","-v"]
    result=subprocess.run(command,cwd=REPO,env=environment,capture_output=True,text=True)
    log=result.stdout+result.stderr
    (output/"test.log").write_text(log)
    count=re.search(r"Ran (\d+) tests?",log)
    after=protected_hashes()
    baseline_diff=subprocess.run(["git","diff","--exit-code",BASE,"--",*PROTECTED,"B1","muon_analysis"],
                                  cwd=REPO,capture_output=True,text=True)
    summary=dict(unit_test_exit_code=result.returncode,unit_test_count=int(count.group(1)) if count else None,
        unit_tests_only=True,six_scenario_experiment_executed=False,geant4_invoked=False,
        deterministic_fault_fixture_folders=len(list((output/"fixtures").glob("test_*"))),
        history_hashes_unchanged=before==after,protected_git_diff_empty=baseline_diff.returncode==0,
        protected_file_count=len(before),frozen_plan_sha256=sha(ROOT/"ADAPTIVE_AUDIT_FIX_PLAN.md"))
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (output/"protected_history.json").write_text(json.dumps(dict(before=before,after=after),indent=2)+"\n")
    evidence=dict(base_commit=BASE,source_commit=git("rev-parse","HEAD"),branch=git("branch","--show-current"),
        command=command,fixture_capture_environment_variable="MUON_GATE_TEST_EVIDENCE_DIR",
        python=sys.version,numpy=np.__version__,platform=platform.platform(),
        source_file_sha256={str(p.relative_to(REPO)):sha(p) for p in sources},
        result_file_sha256={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
        interpretation="Software fault-injection/unit fixtures only; no six-cell calibration or evaluation study was run.")
    (output/"evidence.json").write_text(json.dumps(evidence,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    print(output)
    return result.returncode or (0 if before==after and baseline_diff.returncode==0 else 1)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    sys.exit(run(parser.parse_args().run_id))
