#!/usr/bin/env python3
"""Read-only SCI evidence gate for the archived B2 stage-1 diagnostic.

This is a CI integrity/replay aid, NOT a proof of scientific validity.
It must never rewrite or repair an archived run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "sci_method/studies/ml-nonneural-feasibility-001"
RUN = STUDY / "results/run-001"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def protect_old_evidence(base):
    """Reject modified/deleted/renamed archived result files relative to PR base."""
    if not base or set(base) == {"0"}:
        return {"comparison": "unavailable", "protected_changes": []}
    result = subprocess.run(
        ["git", "diff", "--name-status", "-z", base, "HEAD"],
        cwd=ROOT, check=True, stdout=subprocess.PIPE,
    )
    tokens = result.stdout.decode("utf-8", "surrogateescape").split("\0")
    i, protected = 0, []
    while i < len(tokens) and tokens[i]:
        status = tokens[i]
        path = tokens[i + 1]
        i += 2
        paths = [path]
        if status[0] in ("R", "C"):
            paths.append(tokens[i])
            i += 1
        for name in paths:
            sections = name.split("/")
            is_historical = (len(sections) >= 5 and
                             sections[:2] == ["sci_method", "studies"] and
                             sections[3] in ("results", "verification"))
            if is_historical and status[0] != "A":
                protected.append({"status": status, "path": name})
    if protected:
        raise RuntimeError("Existing scientific result/verification paths modified: "
                           + json.dumps(protected, ensure_ascii=False))
    return {"comparison": base[:12] + "...HEAD", "protected_changes": protected}


def verify_archive():
    manifest = json.loads((RUN / "evidence.json").read_text())
    checks = manifest["sha256"]
    if not checks:
        raise AssertionError("Empty archive manifest")
    mismatches = []
    for relative, expected in checks.items():
        p = RUN / relative
        if not p.is_file() or digest(p) != expected:
            mismatches.append(relative)
    if mismatches:
        raise AssertionError("Archive SHA256 mismatch: " + repr(mismatches))
    conf = json.loads((RUN / "config.json").read_text())
    if conf["code_commit"] != manifest["code_commit"]:
        raise AssertionError("Archive code commit does not match manifest")
    keys = json.loads((RUN / "seed_keys.json").read_text())["seedsequence_keys"]
    records = json.loads((RUN / "trial_records.json").read_text())
    if len(keys) != 1536 or len(records) != 6144:
        raise AssertionError("Unexpected archived seed/trial counts")
    if not all("true_support" in r and "false_support" in r for r in records):
        raise AssertionError("Incomplete trial rows")
    return {
        "archive_files_sha256_verified": len(checks),
        "seed_components": len(keys),
        "trial_rows": len(records),
        "recorded_scientific_code_commit": conf["code_commit"],
        "reported_classification": conf["classification"],
    }


def replay():
    """Recompute archived RNG inputs and decisions; no new scientific experiment."""
    import numpy as np
    sys.path.insert(0, str(STUDY))
    from diagnostics import (SCENES, COUNTS, physical_modes, propose,
                             Q_pvalue, oracle_indices)
    energies = np.load(RUN / "mode_energies.npz", allow_pickle=False)["energy"]
    if energies.shape != (6, 128, 2, 8, 2):
        raise AssertionError("Unexpected archived energy tensor")
    summaries = json.loads((RUN / "candidate_summary.json").read_text())
    original = json.loads((RUN / "trial_records.json").read_text())
    idx = {name: i for i, (name, _, _) in enumerate(SCENES)}
    seeds_checked = 0
    for si, (_, truth, sigma) in enumerate(SCENES):
        _, _, a, _ = physical_modes(sigma)
        for rep in range(128):
            for part in range(2):
                rng = np.random.Generator(np.random.PCG64(
                    np.random.SeedSequence([20261029, si, rep, part])))
                computed = rng.chisquare(COUNTS[:, None], size=(8, 2)) * (
                    1 + np.asarray(truth)[:, None] * a)
                np.testing.assert_array_equal(computed, energies[si, rep, part])
                seeds_checked += 1
    for row in original:
        si = idx[row["scene"]]
        _, truth, sigma = SCENES[si]
        _, _, a, _ = physical_modes(sigma)
        true = np.asarray(truth) > 1
        ep, ea = energies[si, row["replicate"]]
        method = row["method"]
        if method.startswith("split_"):
            if method == "split_oracle":
                selected = oracle_indices(truth, summaries[row["scene"]]["split_power"])
            else:
                selected = propose(ep, COUNTS // 2, a)[method[6:]]
            p = [Q_pvalue(e, int(n), a) for e, n in zip(ea, COUNTS // 2)]
            alpha = 0.025
        else:
            p = [Q_pvalue(e, int(n), a) for e, n in zip(ep + ea, COUNTS)]
            alpha = 0.00625
            if method == "full_all8":
                selected = list(range(8))
            elif method == "full_oracle":
                selected = oracle_indices(truth,
                    summaries[row["scene"]]["full_family_power"])
            else:
                selected = propose(ep + ea, COUNTS, a)["score"]
        supported = [j for j in selected if p[j] <= alpha]
        if (selected != row["selected"] or supported != row["supported"] or
                sum(bool(true[j]) for j in supported) != row["true_support"] or
                sum(not bool(true[j]) for j in supported) != row["false_support"]):
            raise AssertionError("Replay mismatch: " + str(row["scene"]))
    return {"replayed_seed_components": seeds_checked,
            "replayed_selection_and_audit_rows": len(original)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--report", default="ci-artifacts/evidence-report.json")
    args = parser.parse_args()
    report = {
        "check": "read-only, existing archived study data",
        "is_independent_scientific_validation": False,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                           cwd=ROOT, text=True).strip(),
        "mode": "replay" if args.replay else "integrity",
    }
    status = 0
    try:
        report["historical_protection"] = protect_old_evidence(args.base)
        report["archive"] = verify_archive()
        if args.replay:
            report["replay"] = replay()
        report["status"] = "passed"
    except Exception as exc:
        status = 1
        report.update(status="failed", error=repr(exc),
                      traceback=traceback.format_exc())
    out = ROOT / args.report
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"},
                     indent=2, ensure_ascii=False))
    return status


if __name__ == "__main__":
    sys.exit(main())
