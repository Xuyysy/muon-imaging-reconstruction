"""Durable numerical checks for the existing known_fixed six-cell runner.

This adapter records/aborts; it changes no scientific threshold or solver.
Inputs are observed residual groups and the declared geometry/noise model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback

import numpy as np

from sci_method.covariance_optimizer import (
    CovarianceObjective, fit_covariance_detailed, kkt_residual,
)
from sci_method.split_merge import grouped_gaussian_nll_and_gradient

KKT_TOLERANCE = 1e-6
MAX_ITERATIONS = 500
OBJECTIVE_ABSOLUTE_BUDGET = 1e-8
OBJECTIVE_EPS_FACTOR = 512


def plain(value):
    """Strict JSON, with undefined nonfinite numerics explicitly null."""
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)):
        return [plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value,(str,bool,int,float)):
        return value
    return repr(value)


def write_json(path, record):
    """Atomically replace a run-local state file after flushing its content."""
    path=Path(path)
    temporary=path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8") as stream:
        json.dump(plain(record),stream,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


class ExecutionGateError(RuntimeError):
    def __init__(self,code,context,details):
        self.code,self.context,self.details=code,context,details
        super().__init__(f"{code}: {context}")


def validate_known_fixed_nesting(coarse,fine):
    """Verify (parent,other)->(parent,parent,other) against actual group kernels."""
    coarse,fine=tuple(coarse),tuple(fine)
    if not coarse or len(coarse)!=len(fine):
        raise ValueError("coarse/fine path groups differ")
    mapping=np.array([[1.,0.],[1.,0.],[0.,1.]])
    for c,f in zip(coarse,fine):
        if c.name!=f.name or not np.array_equal(c.residuals_mm,f.residuals_mm):
            raise ValueError("coarse/fine observations differ")
        if not np.array_equal(c.base_covariance_mm2,f.base_covariance_mm2):
            raise ValueError("coarse/fine fixed background or measurement covariance differs")
        if c.covariance_derivatives_mm2.shape!=(2,4,4) or f.covariance_derivatives_mm2.shape!=(3,4,4):
            raise ValueError("known_fixed requires coarse=2 and fine=3 parameters")
        embedded=np.einsum("ij,iab->jab",mapping,f.covariance_derivatives_mm2)
        if not np.array_equal(c.covariance_derivatives_mm2,embedded):
            raise ValueError("coarse derivatives do not match the declared fine embedding")
    return mapping


def objective_budget(value):
    return max(OBJECTIVE_ABSOLUTE_BUDGET,
               OBJECTIVE_EPS_FACTOR*np.finfo(float).eps*max(1.,abs(value)))


class ExecutionGate:
    def __init__(self,output):
        self.output=Path(output)
        if not self.output.is_dir():
            raise ValueError("a new study output directory must already exist")

    def log(self,filename,record):
        with (self.output/filename).open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(plain(record),allow_nan=False)+"\n")
            stream.flush();os.fsync(stream.fileno())

    def fail(self,code,context,details):
        self.log("failures.jsonl",dict(code=code,context=context,details=details))
        raise ExecutionGateError(code,context,details)

    def start_trial(self,context):
        self.log("trial_events.jsonl",dict(event="started",context=context))
        write_json(self.output/"current_trial.json",context)

    def record_sample(self,data,context,*,role):
        actual={name:len(values) for name,values in data.items()}
        expected=context.get("counts",actual)
        # Save the observed input before checking allocation, so a bad generator
        # or caller still leaves inspectable data/seed provenance.
        for name,values in data.items():
            y=np.asarray(values,dtype=float)
            if y.ndim!=2 or y.shape[1]!=4 or not np.isfinite(y).all():
                self.fail("invalid_observations",context,dict(group=name,role=role,shape=y.shape))
            content=np.ascontiguousarray(y,dtype="<f8")
            digest=hashlib.sha256(content.tobytes(order="C")).hexdigest()
            self.log("inputs.jsonl",dict(context=context,role=role,group=name,
                n_events=len(y),scatter_mm2=y.T@y,observation_sha256=digest,
                hash_encoding="C-order little-endian float64; shape recorded separately",
                shape=y.shape,actual_allocation=actual,declared_allocation=expected))
        if actual!=expected:
            self.fail("event_allocation_mismatch",context,dict(actual=actual,declared=expected,role=role))

    def _fit_one(self,groups,validation,initial,model,context,solver):
        start=time.perf_counter()
        record=dict(context=context,model=model,initial_parameters=initial,
                    tolerance=KKT_TOLERANCE,coordinate_scale=np.maximum(1.,initial))
        try:
            result=solver(groups,initial=initial,tolerance=KKT_TOLERANCE,max_iter=MAX_ITERATIONS)
        except Exception as exc:
            record.update(status="exception",certified=False,reason="solver_exception",
                          exception=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc(),
                          parameters=None,reported_kkt_inf=None,solver_nll=None,
                          fit_ms=1000*(time.perf_counter()-start))
            self.log("fits.jsonl",record)
            self.fail("solver_exception",context,record)
        solver_finished=time.perf_counter()
        record.update(status=getattr(result,"status",None),reason=getattr(result,"reason",None),
            parameters=getattr(result,"parameters",None),solver_nll=getattr(result,"nll",None),
            reported_gradient=getattr(result,"gradient",None),reported_kkt_inf=getattr(result,"kkt_inf",None),
            iterations=getattr(result,"iterations",None),claimed_converged=getattr(result,"converged",None))
        issues=[]
        status=record["status"]
        known_status=isinstance(status,str) and status in ("converged","precision_limited","unproven","failed")
        if not known_status:
            issues.append("unknown_status")
        if not known_status or status!="converged" or record["claimed_converged"] is not True:
            issues.append("status_not_certified")
        try:
            theta=np.asarray(record["parameters"],dtype=float)
            reported_gradient=np.asarray(record["reported_gradient"],dtype=float)
            reported_kkt=float(record["reported_kkt_inf"])
            solver_nll=float(record["solver_nll"])
            if (theta.shape!=initial.shape or reported_gradient.shape!=initial.shape
                or not np.isfinite(theta).all() or np.any(theta<0)
                or not np.isfinite(reported_gradient).all()
                or not math.isfinite(reported_kkt) or reported_kkt<0
                or not math.isfinite(solver_nll)):
                raise ValueError("nonfinite, malformed or infeasible numerical report")
            iteration=record["iterations"]
            if isinstance(iteration,(bool,np.bool_)) or not isinstance(iteration,(int,np.integer)) or not 0<=iteration<=MAX_ITERATIONS:
                raise ValueError("invalid iteration count")
            objective=CovarianceObjective(groups)
            value,gradient,hessian,_=objective.evaluate(theta)
            raw_nll=grouped_gaussian_nll_and_gradient(groups,theta)[0]
            residual,norm=kkt_residual(theta,gradient,np.maximum(1.,initial))
            budget=objective_budget(raw_nll)
            interior=theta>0
            eig=np.linalg.eigvalsh(hessian[np.ix_(interior,interior)]) if np.any(interior) else np.array([])
            negative=bool(len(eig) and eig.min() < -64*np.finfo(float).eps*max(1.,float(np.max(np.abs(eig)))))
            if norm>KKT_TOLERANCE or reported_kkt>KKT_TOLERANCE:
                issues.append("kkt_not_satisfied")
            if max(abs(value-solver_nll),abs(value-raw_nll))>budget:
                issues.append("nll_inconsistent")
            if np.max(np.abs(gradient-reported_gradient)*np.maximum(1.,initial))>KKT_TOLERANCE:
                issues.append("gradient_inconsistent")
            if negative:
                issues.append("negative_interior_curvature")
            eval_nll=grouped_gaussian_nll_and_gradient(validation if validation is not None else groups,theta)[0]
            if not all(math.isfinite(v) for v in (raw_nll,value,norm,eval_nll)):
                issues.append("nonfinite_recomputed_values")
            record.update(recomputed_nll=value,train_nll=raw_nll,eval_nll=eval_nll,
                full_train_nll=raw_nll+2*sum(len(g.residuals_mm) for g in groups)*math.log(2*math.pi),
                recomputed_gradient=gradient,kkt_residual=residual,recomputed_kkt_inf=norm,
                interior_hessian_eigenvalues=eig,objective_budget=budget,
                objective_discrepancy=max(abs(value-solver_nll),abs(value-raw_nll)))
        except (ValueError,TypeError,OverflowError,np.linalg.LinAlgError) as exc:
            issues.append("invalid_numeric_report")
            record["validation_error"]=f"{type(exc).__name__}: {exc}"
        record.update(certification_issues=issues,certified=not issues,
                      fit_ms=1000*(solver_finished-start),
                      certification_and_scoring_ms=1000*(time.perf_counter()-solver_finished))
        self.log("fits.jsonl",record)
        if issues:
            self.fail("fit_not_certified",context,record)
        record["converged"]=True
        return record

    def fit_pair(self,coarse,fine,context,*,validation=None,solver=None):
        try:
            mapping=validate_known_fixed_nesting(coarse,fine)
        except ValueError as exc:
            self.fail("not_known_fixed_nested_models",context,dict(error=str(exc)))
        solver=fit_covariance_detailed if solver is None else solver
        cval,fval=(None,None) if validation is None else validation
        try:
            c=self._fit_one(coarse,cval,np.ones(2),"coarse",context,solver)
        except ExecutionGateError:
            self.log("not_attempted.jsonl",dict(context=context,model="fine",reason="preceding_fit_failed"))
            raise
        f=self._fit_one(fine,fval,np.ones(3),"fine",context,solver)
        raw_signed=2*(c["train_nll"]-f["train_nll"])
        difference_error=None
        try:
            stable_signed=2*CovarianceObjective(fine).difference(np.asarray(f["parameters"]),mapping@c["parameters"])
        except Exception as exc:
            stable_signed=math.nan
            difference_error=f"{type(exc).__name__}: {exc}"
        allowance=2*sum(r["objective_budget"]+r["objective_discrepancy"] for r in (c,f))
        valid=all(math.isfinite(v) for v in (raw_signed,stable_signed,allowance))
        record=dict(context=context,mapping=mapping,nesting_verified=True,
            raw_signed_lrt=raw_signed,raw_lrt=max(0.,raw_signed) if math.isfinite(raw_signed) else None,
            stable_signed_lrt=stable_signed,stable_lrt=max(0.,stable_signed) if math.isfinite(stable_signed) else None,
            numerical_allowance=allowance,numerical_allowance_is_rigorous_bound=False,
            stable_difference_error=difference_error,
            score_below_resolution=valid and max(abs(raw_signed),abs(stable_signed))<=allowance)
        inconsistent=not valid or min(raw_signed,stable_signed)<-allowance or abs(raw_signed-stable_signed)>allowance
        record["ambiguous"]=inconsistent
        self.log("lrt_checks.jsonl",record)
        if inconsistent:
            self.fail("lrt_numeric_inconsistency",context,record)
        return c,f,record

    def check_lrt(self,record,cutoffs,context):
        comparison={}
        for name,spec in cutoffs.items():
            try:
                cutoff=float(spec["value"]);calibration_allowance=float(spec.get("allowance",0.))
            except (ValueError,TypeError,OverflowError,KeyError):
                self.fail("invalid_cutoff_state",context,dict(name=name,spec=spec))
            if not math.isfinite(cutoff) or cutoff<0 or not math.isfinite(calibration_allowance) or calibration_allowance<0:
                self.fail("invalid_cutoff_state",context,dict(name=name,spec=spec))
            radius=record["numerical_allowance"]+calibration_allowance
            raw,stable=record["raw_lrt"],record["stable_lrt"]
            ambiguous=(raw>cutoff)!=(stable>cutoff) or min(abs(raw-cutoff),abs(stable-cutoff))<=radius
            comparison[name]=dict(value=cutoff,calibration_allowance=calibration_allowance,
                total_allowance=radius,raw_exceeds=raw>cutoff,stable_exceeds=stable>cutoff,ambiguous=ambiguous)
        checked={**record,"context":context,"cutoffs":comparison,
                 "ambiguous":any(s["ambiguous"] for s in comparison.values())}
        self.log("lrt_checks.jsonl",checked)
        if checked["ambiguous"]:
            self.fail("lrt_threshold_ambiguous",context,checked)
        return checked
