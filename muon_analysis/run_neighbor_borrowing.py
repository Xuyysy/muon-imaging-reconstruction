"""两粗格固定事件预算合成检查；完整记录共享与越界错配。"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from neighbor_borrowing import reconstruct_pair
from synthetic_subcell_risk import squared_risk
from run_synthetic_subcell_risk import paired_difference

ROOT=Path(__file__).resolve().parent
TRIALS=10000
SEED=2026093002
PAIRS=((1,8),(2,8),(4,16),(8,32))
UNIFORM=np.array([.25]*4)
FAVORED=np.array([.7,.1,.1,.1])
TWO=np.array([.5,.5,0.,0.])
SCENES={'uniform_shared':(UNIFORM,UNIFORM),
        'feature_shared':(FAVORED,FAVORED),
        'rotated_boundary':(FAVORED,np.roll(FAVORED,1)),
        'disjoint_boundary':(TWO,np.roll(TWO,2))}


def run():
    results=[]
    for si,(scene,(truth_a,truth_b)) in enumerate(SCENES.items()):
        for pi,(na,nb) in enumerate(PAIRS):
            rng=np.random.default_rng(SEED+100*si+pi)
            events_a=rng.choice(4,size=(TRIALS,na),p=truth_a)
            events_b=rng.choice(4,size=(TRIALS,nb),p=truth_b)
            # 反事实：将邻区nb个事件预算全部投到A；不是现实免费样本。
            events_extra_a=rng.choice(4,size=(TRIALS,nb),p=truth_a)
            counts=lambda events:np.stack([(events==j).sum(axis=1) for j in range(4)],axis=1).astype(float)
            a,b=counts(events_a),counts(events_b)
            outputs,diagnostic=reconstruct_pair(a,b)
            own_a=(1.+a+counts(events_extra_a))/(4.+na+nb)
            # 场景oracle读真值“是否同结构”，只展示可解释的两种端点。
            same=np.array_equal(truth_a,truth_b)
            outputs['scenario_oracle']=outputs['pooled' if same else 'independent']
            risk={method:{'A':squared_risk(cells[0],truth_a),
                          'B':squared_risk(cells[1],truth_b)}
                  for method,cells in outputs.items()}
            full_own_risk=squared_risk(own_a,truth_a)
            metrics={method:{cell:{'mean_squared_risk':float(values.mean()),
                                   'mc_standard_error':float(values.std(ddof=1)/np.sqrt(TRIALS))}
                             for cell,values in cell_risk.items()}
                     for method,cell_risk in risk.items()}
            results.append({'scene':scene,'n_a':na,'n_b':nb,'total_events':na+nb,
                            'trials':TRIALS,'truth_a':truth_a.tolist(),'truth_b':truth_b.tolist(),
                            'methods':metrics,
                            'paired_minus_independent':{
                                method:{cell:paired_difference(risk[method][cell],risk['independent'][cell])
                                        for cell in ('A','B')}
                                for method in ('pooled','limited','gated','scenario_oracle')},
                            'own_a_full_budget':{
                                'mean_squared_risk':float(full_own_risk.mean()),
                                'paired_minus_borrowed_a':paired_difference(full_own_risk,risk['gated']['A'])},
                            'mean_gate':float(diagnostic['gate'].mean()),
                            'mean_borrowed_pseudo_events_a':float(diagnostic['borrowed_a'].mean()),
                            'mean_borrowed_pseudo_events_b':float(diagnostic['borrowed_b'].mean())})
    if len(results)!=16 or len({(r['scene'],r['n_a'],r['n_b']) for r in results})!=16:
        raise AssertionError('Missing fixed-budget conditions')
    focus_shared=[r for r in results if r['scene']=='feature_shared' and (r['n_a'],r['n_b']) in ((1,8),(2,8))]
    focus_boundary=[r for r in results if r['scene']=='disjoint_boundary']
    shared_checks=[r['paired_minus_independent']['gated']['A']['mean']<0 for r in focus_shared]
    boundary_checks=[r['paired_minus_independent']['gated'][cell]['mean']<=1e-12
                     for r in focus_boundary for cell in ('A','B')]
    verdict={'shared_feature_sparse_a_passing':sum(shared_checks),'shared_feature_required':2,
             'disjoint_boundary_cells_passing':sum(boundary_checks),'disjoint_boundary_required':8,
             'development_screening_passed':all(shared_checks) and all(boundary_checks),
             'scope':'synthetic fixed-total-budget development, not independent Geant4 or 3D imaging'}
    out=ROOT/'neighbor_borrowing_results'
    out.mkdir(exist_ok=True)
    report={'config':{'seed':SEED,'trials':TRIALS,'pairs':PAIRS,
                      'scenes':{name:[p.tolist() for p in pair] for name,pair in SCENES.items()}},
            'code_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                           for name in ('neighbor_borrowing.py','run_neighbor_borrowing.py')},
            'results':results,'verdict':verdict}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(4,2,figsize=(11,12),constrained_layout=True)
    for row,scene in enumerate(SCENES):
        part=[r for r in results if r['scene']==scene]
        for col,cell in enumerate(('A','B')):
            ax=axes[row,col]
            for method in ('pooled','limited','gated'):
                d=[r['paired_minus_independent'][method][cell] for r in part]
                ax.errorbar(range(4),[x['mean'] for x in d],
                    yerr=[x['mc_standard_error'] for x in d],fmt='o-',capsize=2,label=method)
            ax.axhline(0,color='black',lw=.7)
            ax.set_xticks(range(4),[f'{a}/{b}' for a,b in PAIRS])
            ax.set(title=f'{scene}: cell {cell}',xlabel='nA/nB, total fixed',
                   ylabel='Paired risk difference vs independent')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'neighbor_borrowing_risk.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'verdict':verdict,'focus_shared':[
        {'pair':[r['n_a'],r['n_b']], 'A_gated_minus_I':r['paired_minus_independent']['gated']['A']}
        for r in focus_shared], 'focus_boundary':[
        {'pair':[r['n_a'],r['n_b']], 'A':r['paired_minus_independent']['gated']['A'],
         'B':r['paired_minus_independent']['gated']['B']}
        for r in focus_boundary]},indent=2))


if __name__=='__main__':
    run()
