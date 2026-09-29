"""冻结E/ME在新偏心H目标两个Geant4流上的检验。

必须先读B1/EDGE_HELDOUT_H_PROTOCOL.md及本轮frozen_config.json。
本脚本核对重建摘要，不选择参数、不根据得分回退、不混淆子抽样与物理流。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diagnose_split_confidence import ROOT
from run_adaptive_voxel_experiment import RANDOM_SEED, imaging_edges, Z_LIMIT_MM
from run_detector_resolution_robustness import recompute_from_noisy_hits
from run_event_uncertainty_experiment import measured_hits
from run_soft_z_gate_experiment import candidate_events, xy_image
from run_frame_z_gate_validation import smooth_hard_image
from run_uncertainty_aware_angle_experiment import same_side_layer_spacing_mm
from run_fixed_budget_refinement import fractional_pair_metrics, continuous_image_metrics
from run_shape_smoothing_control import shape_scores
from local_scale_reconstruction import native_support, local_scale_image, matched_isotropic_image

KEYS = ['stream', 'event_count', 'repeat', 'position_sigma_mm', 'grid']
METHODS = ('hard', 'smoothed', 'edge_only', 'matched_edge')
METRICS = ('fractional_iou', 'target_mass_fraction', 'empty_mass_fraction',
           'empty_fractional_occupancy', 'footprint_proxy_distance')


def h_mask(cx, cy):
    """三块仿真盒子的XY足迹，仅用于评分。"""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    left = (x >= -40)&(x <= -20)&(y >= -65)&(y <= 45)
    right = (x >= 50)&(x <= 70)&(y >= -65)&(y <= 45)
    middle = (x >= -20)&(x <= 50)&(y >= -20)&(y <= 0)
    return left | right | middle


def h_empty(cx, cy):
    """上/下开口合并；不包含柱或连接横梁，不给重建函数。"""
    x, y = np.meshgrid(cx, cy, indexing='ij')
    return (x > -20)&(x < 50)&(((y > 0)&(y <= 45))|((y >= -65)&(y < -20)))


def frozen_verdict(primary_summary):
    """仅执行运行前四组判据，不事后重新指定主条件。"""
    groups = primary_summary.copy()
    groups['passes'] = ((groups.edge_only_minus_matched_edge_fractional_iou > 0)
                       & (groups.edge_only_minus_matched_edge_empty_mass_fraction < 0)
                       & (groups.valid_matches == 5))
    return {'primary_groups': len(groups), 'groups_passing': int(groups.passes.sum()),
            'mechanism_reproduced': bool(len(groups) == 4 and groups.passes.all()),
            'scope': 'one held-out geometry/two Geant4 streams; not E>S or 3D innovation'}, groups


def run(output_dir):
    frozen = json.loads((ROOT/'muon_analysis/edge_h_heldout_results/frozen_config.json').read_text())
    core = ROOT/'muon_analysis/local_scale_reconstruction.py'
    if hashlib.sha256(core.read_bytes()).hexdigest() != frozen['reconstruction_sha256']:
        raise AssertionError('Frozen reconstruction changed; do not replace saved hash')
    edges = imaging_edges(32)
    cx, cy = [(a[:-1]+a[1:])/2 for a in edges[:2]]
    truth, empty = h_mask(cx, cy), h_empty(cx, cy)
    if (truth & empty).any() or not empty.any():
        raise AssertionError('Invalid held-out target/empty region')
    rows, counts, pictures = [], [], {}
    for index, stream in enumerate(frozen['geant4_seeds']):
        path = ROOT/f'muon_analysis/muon_tracks_h_offset_{stream}.csv'
        data = pd.read_csv(path)
        if len(data) < max(frozen['counts']) or data.event.duplicated().any():
            raise ValueError('Incomplete or duplicate held-out tracks; do not swap seeds')
        counts.append({'stream': stream, 'generated_events': frozen['generated_events_per_stream'],
                       'complete_tracks': len(data), 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        spacing = same_side_layer_spacing_mm(data)
        rng = np.random.default_rng(RANDOM_SEED+9107+1000*index)
        for count in frozen['counts']:
            for repeat in range(1, frozen['repeats']+1):
                subset = data.iloc[rng.choice(len(data), count, replace=False)].copy()
                draws = {f'{plane}_{axis}': rng.normal(0,1,count)
                         for plane in ('up1','up2','down1','down2') for axis in ('x','y')}
                for sigma in frozen['sigmas_mm']:
                    measured = measured_hits(subset, sigma, draws)
                    observed = recompute_from_noisy_hits(measured, 0., np.random.default_rng(0))
                    ids, weight = candidate_events(observed, sigma, spacing)
                    weight *= (np.abs(observed.poca_z_mm.to_numpy()[ids]) <= Z_LIMIT_MM)
                    xy = observed[['poca_x_mm','poca_y_mm']].to_numpy()[ids]
                    for bins in frozen['grids']:
                        base = xy_image(xy,weight,bins)
                        fixed = smooth_hard_image(base,bins)
                        neff, guide = native_support(xy,weight,bins)
                        edged, edge_diag = local_scale_image(base,bins,neff,guide,False,True)
                        check, fixed_diag = local_scale_image(base,bins,neff,guide,False,False)
                        if not np.allclose(check,fixed,rtol=1e-12,atol=1e-9):
                            raise AssertionError('Fixed S displacement operator mismatch')
                        images = {'hard':base, 'smoothed':fixed, 'edge_only':edged}
                        diagnostics = {'hard':{'expected_squared_displacement_mm2':0.},
                                       'smoothed':fixed_diag, 'edge_only':edge_diag}
                        try:
                            images['matched_edge'], diagnostics['matched_edge'] = matched_isotropic_image(
                                base,fixed,edge_diag['expected_squared_displacement_mm2'],fixed_diag['expected_squared_displacement_mm2'])
                        except ValueError as error:
                            diagnostics['matched_edge'] = {'reason':str(error)}
                        for method in METHODS:
                            key = {'stream':stream,'event_count':count,'repeat':repeat,'position_sigma_mm':sigma,'grid':bins,'method':method}
                            if method not in images:
                                rows.append({**key,'match_valid':False,'total_image_weight':np.nan,
                                             **{m:np.nan for m in METRICS},**diagnostics[method]})
                                continue
                            image = images[method]
                            if not np.isclose(image.sum(),base.sum(),rtol=1e-12,atol=1e-9):
                                raise AssertionError('Held-out mass changed')
                            rows.append({**key,'match_valid':True,'total_image_weight':float(image.sum()),
                                         **fractional_pair_metrics(image,base,truth),**continuous_image_metrics(image,base,truth),
                                         **shape_scores(image,truth,empty,cx,'h_offset'),**diagnostics[method]})
                            if count==5000 and repeat==1 and sigma==.05 and bins==32:
                                pictures[(stream,method)] = image
                print(f'H held-out: {stream}, N={count}, repeat={repeat}',flush=True)
    frame = pd.DataFrame(rows)
    if len(frame)!=960 or frame.duplicated(KEYS+['method']).any():
        raise AssertionError('Incomplete validation conditions')
    views = {m:frame[frame.method==m].set_index(KEYS).sort_index() for m in METHODS}
    paired = pd.DataFrame({
        **{f'{m}_{metric}':views[m][metric] for m in METHODS for metric in METRICS},
        **{f'edge_only_minus_{reference}_{metric}':views['edge_only'][metric]-views[reference][metric]
           for reference in ('matched_edge','smoothed','hard') for metric in METRICS},
        'match_valid':views['matched_edge'].match_valid,
    }).reset_index()
    gkeys = [k for k in KEYS if k!='repeat']
    summary = paired.groupby(gkeys,as_index=False).mean(numeric_only=True).drop(columns='repeat')
    spread = paired.groupby(gkeys,as_index=False).agg(
        valid_matches=('match_valid','sum'), repeats=('repeat','size'),
        iou_diff_matched_std=('edge_only_minus_matched_edge_fractional_iou','std'),
        iou_wins_matched=('edge_only_minus_matched_edge_fractional_iou',lambda x:int((x>1e-12).sum())))
    summary = summary.merge(spread,on=gkeys,validate='one_to_one')
    primary = frozen['primary']
    focus = summary[(summary.event_count==primary['event_count'])&(summary.grid==primary['grid'])
                    &summary.position_sigma_mm.isin(primary['sigmas_mm'])]
    verdict, decisions = frozen_verdict(focus)
    output_dir.mkdir(parents=True,exist_ok=True)
    frame.to_csv(output_dir/'by_repeat.csv',index=False)
    paired.to_csv(output_dir/'paired_by_repeat.csv',index=False)
    summary.to_csv(output_dir/'summary.csv',index=False)
    decisions.to_csv(output_dir/'primary_decisions.csv',index=False)
    pd.DataFrame(counts).to_csv(output_dir/'source_counts.csv',index=False)
    (output_dir/'verdict.json').write_text(json.dumps(verdict,indent=2)+'\n')
    fig, axes = plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for j,stream in enumerate(frozen['geant4_seeds']):
        part = summary[(summary.stream==stream)&(summary.event_count==5000)&(summary.grid==32)]
        for reference in ('matched_edge','smoothed'):
            axes[0,j].plot(part.position_sigma_mm,part[f'edge_only_minus_{reference}_fractional_iou'],'o-',label=f'E vs {reference}')
            axes[1,j].plot(part.position_sigma_mm,part[f'edge_only_minus_{reference}_empty_mass_fraction'],'o-')
        for ax in axes[:,j]:
            ax.axhline(0,color='black',lw=.7); ax.grid(alpha=.2); ax.set(title=stream,xlabel='Hit-position sigma [mm]')
    axes[0,0].set_ylabel('Fractional IoU difference (higher better)')
    axes[1,0].set_ylabel('Empty-region mass difference (lower better)')
    axes[0,0].legend(fontsize=8)
    fig.savefig(output_dir/'h_heldout_metrics.png',dpi=170); plt.close(fig)
    if len(pictures)==8:
        fig,axes = plt.subplots(2,3,figsize=(11,7),constrained_layout=True)
        for i,stream in enumerate(frozen['geant4_seeds']):
            vmax = max(np.log1p(pictures[(stream,m)]).max() for m in ('smoothed','matched_edge','edge_only'))
            for j,method in enumerate(('smoothed','matched_edge','edge_only')):
                ax = axes[i,j]
                im = ax.imshow(np.log1p(pictures[(stream,method)]).T,origin='lower',extent=[-90,90,-90,90],vmin=0,vmax=vmax,cmap='inferno')
                ax.contour(cx,cy,truth.T,levels=[.5],colors='cyan',linewidths=.8)
                ax.set(title=f'{stream}: {method}',xlabel='x [mm]',ylabel='y [mm]')
            fig.colorbar(im,ax=axes[i,:].tolist(),label='log(1 + scattering weight)')
        fig.suptitle('Frozen H-target validation; N=5000, sigma=.05 mm, repeat 1')
        fig.savefig(output_dir/'h_heldout_images.png',dpi=160); plt.close(fig)
    print(json.dumps(verdict),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'muon_analysis/edge_h_heldout_results')
    run(parser.parse_args().output_dir)
