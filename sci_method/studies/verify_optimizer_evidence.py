"""Verify audit manifests, archived inputs and four protected history trees."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/"studies/optimizer-stability-001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output):
    output.mkdir(parents=True,exist_ok=False)
    checks = []
    for path in sorted((STUDY/"results").glob("*/evidence.json")):
        manifest = json.loads(path.read_text())
        for relative, expected in manifest.get("results_sha256",{}).items():
            checks.append(dict(file=str((path.parent/relative).relative_to(ROOT)),
                match=sha(path.parent/relative)==expected, expected_sha256=expected))
    canonical=STUDY/"results/run-004"
    protected=json.loads((canonical/"historical_protection.json").read_text())
    # Only canonical source/data files belong to the historical evidence tree;
    # the early importer's two newly generated .pyc files were incidental caches.
    ignored_caches=[name for name in protected["before_sha256"] if "__pycache__" in Path(name).parts]
    history_checks={name:sha(ROOT/name)==expected for name,expected in protected["before_sha256"].items()
                    if name not in ignored_caches}
    with (ROOT/"studies/adaptive-prototype-001/results/run-002/sample_covariance_sufficient_statistics.csv").open() as stream:
        archived={(r["phase"],int(r["seed"]),r["group"]):r for r in csv.DictReader(stream)}
    statistic_checks=[]
    for record in json.loads((canonical/"summary.json").read_text()):
        case=record["case"]
        with np.load(canonical/f"{case['name']}_observations.npz",allow_pickle=False) as data:
            for key in data.files:
                empty=key.startswith("empty_")
                phase="evaluation_empty_field" if empty else case["phase"]
                seed=case["seed"]+2 if empty else case["seed"]
                name=key.removeprefix("empty_")
                row=archived[(phase,seed,name)]
                # NpzFile.__getitem__ reloads on each call. Cache once to
                # reproduce the original symmetric Y.T@Y reduction, rather
                # than a general GEMM between two separately loaded copies.
                observations=data[key]
                s=observations.T@observations
                match=int(row["n_events"])==len(observations) and all(
                    s[i,j]==float(row[f"s{i}{j}_mm2"]) for i in range(4) for j in range(i,4))
                statistic_checks.append(dict(case=case["name"],group=key,match=match))
    paths=["sci_method/studies/"+study+"/results/"+run
           for study in ("adaptive-prototype-001","adaptive-audit-fix-001")
           for run in ("run-001","run-002")]
    diff=subprocess.run(["git","diff","--exit-code","5334f3533e1ada58f0afb3e98f8de10a6f52aeb8","--",*paths],
        cwd=ROOT.parent,capture_output=True,text=True)
    latest=json.loads((STUDY/"results/run-006/test_status.json").read_text())
    result=dict(all_passed=all(c["match"] for c in checks+statistic_checks) and all(history_checks.values())
        and diff.returncode==0 and latest["exit_code"]==0,
        artifact_hash_checks=checks,protected_history_checks=history_checks,noncanonical_import_caches=ignored_caches,
        historical_git_diff_empty=diff.returncode==0,observed_sufficient_statistic_checks=statistic_checks,
        final_test_status=latest,source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT.parent,text=True).strip(),
        verifier_source_sha256=sha(Path(__file__)),python=platform.python_version(),numpy=np.__version__,
        command="PYTHONPATH=sci_method/src python3 sci_method/studies/verify_optimizer_evidence.py --run-id "+output.name,
        result_files_sha256={str(p.relative_to(STUDY)):sha(p) for p in sorted((STUDY/"results").rglob("*")) if p.is_file()})
    (output/"verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(all_passed=result["all_passed"],artifact_files=len(checks),
        protected_history_files=len(history_checks),statistic_groups=len(statistic_checks),output=str(output))))
    return result["all_passed"]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    if not args.run_id.startswith("run-") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        raise ValueError("invalid verification run ID")
    sys.exit(0 if verify(STUDY/"verification"/args.run_id) else 1)
