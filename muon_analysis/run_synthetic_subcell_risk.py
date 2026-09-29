"""运行预定四子格合成检查，保留全部配置，不训练/扫描参数。"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from synthetic_subcell_risk import reconstruct, squared_risk, weighted_population_target, METHODS

ROOT = Path(__file__).resolve().parent
COUNTS = (0, 1, 2, 4, 8, 32, 128)
TRIALS = 10000  # 合成样本独立；这与Geant4母CSV重复抽样不同。
SEED = 2026093001
PROBABILITIES = {'uniform':[.25,.25,.25,.25],
                 'one_favored':[.7,.1,.1,.1], 'two_child_boundary':[.5,.5,0.,0.]}
REGIMES = ('equal', 'independent_lognormal', 'child_correlated')
MEAN_WEIGHTS = {'equal':[1.]*4, 'independent_lognormal':[1.]*4,
                'child_correlated':[1.,2.,4.,8.]}


def paired_difference(values, reference):
    """同次模拟方法相减，给出Monte Carlo均值及标准误，不是物理泛化。"""
    difference = values-reference
    return {'mean':float(difference.mean()),
            'mc_standard_error':float(difference.std(ddof=1)/np.sqrt(len(difference)))}


def run():
    # 顺序及每配置种子固定；新增别的配置不会改变现有抽样流。
    results = []
    for pi, (name, probability) in enumerate(PROBABILITIES.items()):
        for ri, regime in enumerate(REGIMES):
            truth = weighted_population_target(probability, MEAN_WEIGHTS[regime])
            for ni, count in enumerate(COUNTS):
                rng = np.random.default_rng(SEED+1000*pi+100*ri+ni)
                children = rng.choice(4, size=(TRIALS,count), p=probability)
                if regime == 'independent_lognormal':
                    # E[w]=1，log(w)均值-1/2、标准差1；位置与权重独立。
                    weights = rng.lognormal(mean=-.5, sigma=1., size=children.shape)
                elif regime == 'child_correlated':
                    weights = np.asarray(MEAN_WEIGHTS[regime])[children]
                else:
                    weights = np.ones_like(children, dtype=float)
                child_w = np.zeros((TRIALS,4))
                child_w2 = np.zeros_like(child_w)
                for child in range(4):
                    mask = children == child
                    child_w[:,child] = (weights*mask).sum(axis=1)
                    child_w2[:,child] = (weights**2*mask).sum(axis=1)
                images, diagnostic = reconstruct(child_w,child_w2)
                risks = {method:squared_risk(images[method],truth) for method in METHODS}
                metrics = {method:{'mean_squared_risk':float(risk.mean()),
                           'mc_standard_error':float(risk.std(ddof=1)/np.sqrt(TRIALS))}
                           for method,risk in risks.items()}
                result = {'probability_case':name, 'weight_regime':regime,
                          'event_count':count, 'trials':TRIALS,
                          'event_probability':probability, 'weighted_target':truth.tolist(),
                          'sample_probability_bias':(images['raw'].mean(axis=0)-truth).tolist(),
                          'methods':metrics, 'candidate_minus':{
                              ref:paired_difference(risks['candidate'],risks[ref])
                              for ref in ('coarse','raw','neff')},
                          'candidate_alpha_mean':float(diagnostic['candidate_alpha'].mean()),
                          'candidate_alpha_one_fraction':float((diagnostic['candidate_alpha'] >= 1.-1e-12).mean()),
                          'variance_proxy_mean':float(diagnostic['variance_proxy'].mean()),
                          'contrast_mean':float(diagnostic['contrast'].mean()),
                          'variance_zero_fraction':float((diagnostic['variance_proxy'] == 0).mean()),
                          'neff_mean':float(diagnostic['effective_count'].mean())}
                results.append(result)
    sparse = next(r for r in results if r['probability_case']=='uniform'
                  and r['weight_regime']=='equal' and r['event_count']==1)
    # 最基本必要检查：均匀总体/单事件不应比已有温和收缩更极端。
    # 本轮既不事后取消这一档，也不修改原候选以绕过失败。
    verdict = {'single_event_uniform_sanity_passed':
               sparse['methods']['candidate']['mean_squared_risk'] <= sparse['methods']['neff']['mean_squared_risk'],
               'next_action':'stop before Geant4; diagnose finite-sample variance degeneracy',
               'scope':'synthetic weighted-proportion risk, not muon imaging or 3D voxel validation'}
    report = {'config':{'counts':COUNTS,'trials':TRIALS,'seed':SEED,'probabilities':PROBABILITIES,
                         'weight_regimes':REGIMES,'conditional_mean_weights':MEAN_WEIGHTS},
              'code_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                             for name in ('synthetic_subcell_risk.py','run_synthetic_subcell_risk.py')},
              'results':results,'verdict':verdict}
    out = ROOT/'synthetic_subcell_risk_results'
    out.mkdir(exist_ok=True)
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    fig, axes = plt.subplots(3,3,figsize=(12,9),constrained_layout=True)
    for pi,name in enumerate(PROBABILITIES):
        for ri,regime in enumerate(REGIMES):
            ax=axes[pi,ri]
            part=[r for r in results if r['probability_case']==name and r['weight_regime']==regime]
            for method in METHODS:
                ax.plot(COUNTS,[r['methods'][method]['mean_squared_risk'] for r in part],'o-',label=method,ms=3)
            ax.set_xscale('symlog',linthresh=1)
            ax.set(title=f'{name}\n{regime}',xlabel='Events in coarse cell',ylabel='Mean squared proportion error')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'synthetic_risk.png',dpi=160)
    plt.close(fig)
    # 真值已知的均匀总体中，coarse误差为0是定义性质，不是可部署优势。
    print(json.dumps({'configurations':len(results),'method_summaries':len(results)*len(METHODS),
                      'single_event_uniform_equal':sparse,'verdict':verdict},indent=2))


if __name__ == '__main__':
    run()
