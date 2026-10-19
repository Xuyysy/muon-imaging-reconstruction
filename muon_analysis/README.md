# Muon-imaging research resource index

This page is a navigation and provenance index for the simulation, experiment records, plots, and paper assets already present in the repository. It does not modify scientific data, experiment outcomes, evaluation criteria, or paper conclusions.

## Audit scope and rules

- Inventory snapshot: **2026-10-19**. The tracked-resource baseline inspected for this index is commit [`7e66b72`](https://github.com/Xuyysy/muon-imaging-reconstruction/commit/7e66b72dead8028a9aa3f5958cdecd6ae12bfa0c) (2026-10-06). Per-folder dates below are the latest commit dates touching that path at the baseline, not experiment run dates.
- This branch adds only this index and a link to it in the repository README. It does not add raw event data, PDFs, slide decks, notebooks, generated plots, or other experiment outputs.
- “Tracked” means present in Git at the baseline. “Local only” means observed in the researcher’s source checkout but not tracked; these entries are documentation of their status, not copies in this branch. Local-only paths may be absent from a clean GitHub checkout.
- To avoid duplicating existing documentation, the paper-writing guide remains in [`B1/paper/README.md`](../B1/paper/README.md); the paper source, bibliography, and manuscript figure copies remain under [`B1/paper/`](../B1/paper/).
- Relative links are used so the index works in VS Code and on GitHub. The “last updated” date is the date of the last Git change to the resource path, unless explicitly marked as a local file timestamp.

## At-a-glance inventory

| Resource class | Audit result | Git status / handling |
|---|---:|---|
| LaTeX paper source and bibliography | `B1/paper/main.tex`, `references.bib` | Tracked |
| Manuscript figure copies | 56 PNG files in `B1/paper/figures/` | All tracked; each has an exact byte-identical plot in an experiment-results folder |
| Experiment result folders | 74 folders named `*_results/` | The table below records their latest Git update and key record files |
| Research notes | 57 `research_note.md` files in the audited repository snapshot | Tracked where listed below; stored beside the associated experiment records |
| Machine-readable run evidence | `evidence.json` in 5 experiment folders | Tracked; listed in the experiment table |
| Other plots | 162 PNGs found across paper and analysis resources: 114 tracked, 48 local-only | Local-only plots are not added by this task |
| Notebook | `muon_analysis/analysis.ipynb` | Tracked; the source checkout also has local edits, which are not included here |
| PDF, slide decks, thesis draft template | One local compiled PDF, three presentation PPTX files, one Word template | Local-only; not added or uploaded |
| Bibliographic full-text PDFs | None found in the repository snapshot | The user-provided Zotero paper is outside this repository and is not copied or indexed as a distributable file |

## Paper figures: source and generation links

Each manuscript PNG is a tracked copy used by LaTeX. The source plot listed beside it has the same file hash in the audited checkout. Where an `evidence.json` exists, it is linked; otherwise consult the adjacent `config.json`/experiment config, summary, and research note. The script link identifies the associated analysis/diagnostic entry point; it does not imply that the script alone contains the complete run environment.

| Manuscript figure | Source plot / experiment record | Associated script |
|---|---|---|
| [`adaptive_voxel_metrics_vs_refinement.png`](../B1/paper/figures/adaptive_voxel_metrics_vs_refinement.png) | [`adaptive_voxel_results/`](adaptive_voxel_results/) | [`run_adaptive_voxel_experiment.py`](run_adaptive_voxel_experiment.py) |
| [`angle_weighting_metrics_vs_cap.png`](../B1/paper/figures/angle_weighting_metrics_vs_cap.png) | [`angle_weighting_results/`](angle_weighting_results/) | [`run_angle_weighting_experiment.py`](run_angle_weighting_experiment.py) |
| [`baseline_metrics_vs_event_count.png`](../B1/paper/figures/baseline_metrics_vs_event_count.png) | [`baseline_results/`](baseline_results/) | [`run_baseline_experiment.py`](run_baseline_experiment.py) |
| [`boundary_poca_scale_5000.png`](../B1/paper/figures/boundary_poca_scale_5000.png) | [`boundary_poca_diagnosis_results/`](boundary_poca_diagnosis_results/) | [`diagnose_boundary_poca_scale.py`](diagnose_boundary_poca_scale.py) |
| [`continuous_image_audit.png`](../B1/paper/figures/continuous_image_audit.png) | [`continuous_image_audit_results/`](continuous_image_audit_results/) | [`audit_refinement_checkpoint.py`](audit_refinement_checkpoint.py) |
| [`detector_resolution_metrics.png`](../B1/paper/figures/detector_resolution_metrics.png) | [`detector_resolution_results/`](detector_resolution_results/) | [`run_detector_resolution_robustness.py`](run_detector_resolution_robustness.py) |
| [`disjoint_spatial_support.png`](../B1/paper/figures/disjoint_spatial_support.png) | [`disjoint_spatial_support_results/`](disjoint_spatial_support_results/) · [`evidence.json`](disjoint_spatial_support_results/evidence.json) | [`diagnose_disjoint_spatial_support.py`](diagnose_disjoint_spatial_support.py) |
| [`double_block_metrics_5000.png`](../B1/paper/figures/double_block_metrics_5000.png) | [`double_block_validation_results/`](double_block_validation_results/) | [`run_double_block_validation.py`](run_double_block_validation.py) |
| [`double_block_reconstruction_5000.png`](../B1/paper/figures/double_block_reconstruction_5000.png) | [`double_block_validation_results/`](double_block_validation_results/) | [`run_double_block_validation.py`](run_double_block_validation.py) |
| [`empty_target_control.png`](../B1/paper/figures/empty_target_control.png) | [`empty_target_control_results/`](empty_target_control_results/) · [`evidence.json`](empty_target_control_results/evidence.json) | [`run_empty_target_control.py`](run_empty_target_control.py) |
| [`finite_sample_risk.png`](../B1/paper/figures/finite_sample_risk.png) | [`finite_sample_subcell_risk_results/`](finite_sample_subcell_risk_results/) | [`run_finite_sample_subcell_risk.py`](run_finite_sample_subcell_risk.py) |
| [`fixed_budget_refinement.png`](../B1/paper/figures/fixed_budget_refinement.png) | [`fixed_budget_refinement_results/`](fixed_budget_refinement_results/) · [`continuous_image_audit_results/`](continuous_image_audit_results/) | [`run_fixed_budget_refinement.py`](run_fixed_budget_refinement.py) |
| [`fixed_image_support_pool.png`](../B1/paper/figures/fixed_image_support_pool.png) | [`fixed_image_support_pool_results/`](fixed_image_support_pool_results/) | [`run_fixed_image_support_pool.py`](run_fixed_image_support_pool.py) |
| [`fractional_iou_comparison.png`](../B1/paper/figures/fractional_iou_comparison.png) | [`fractional_iou_audit_results/`](fractional_iou_audit_results/) · [`continuous_image_audit_results/`](continuous_image_audit_results/) | [`audit_refinement_checkpoint.py`](audit_refinement_checkpoint.py) |
| [`frame_gate_iou_5000_32.png`](../B1/paper/figures/frame_gate_iou_5000_32.png) | [`frame_z_gate_results/`](frame_z_gate_results/) | [`run_frame_z_gate_validation.py`](run_frame_z_gate_validation.py) |
| [`frame_gate_score_audit.png`](../B1/paper/figures/frame_gate_score_audit.png) | [`frame_gate_score_audit_results/`](frame_gate_score_audit_results/) | [`run_frame_z_gate_validation.py`](run_frame_z_gate_validation.py) |
| [`frame_smoothing_control.png`](../B1/paper/figures/frame_smoothing_control.png) | [`frame_smoothing_control_results/`](frame_smoothing_control_results/) | [`run_frame_z_gate_validation.py`](run_frame_z_gate_validation.py) |
| [`h_heldout_images.png`](../B1/paper/figures/h_heldout_images.png) | [`edge_h_heldout_results/`](edge_h_heldout_results/) | [`run_edge_h_heldout_validation.py`](run_edge_h_heldout_validation.py) |
| [`h_heldout_metrics.png`](../B1/paper/figures/h_heldout_metrics.png) | [`edge_h_heldout_results/`](edge_h_heldout_results/) | [`run_edge_h_heldout_validation.py`](run_edge_h_heldout_validation.py) |
| [`heldout_l_shape_5000.png`](../B1/paper/figures/heldout_l_shape_5000.png) | [`heldout_l_shape_results/`](heldout_l_shape_results/) | [`run_heldout_l_shape_validation.py`](run_heldout_l_shape_validation.py) |
| [`heldout_l_shape_reconstruction.png`](../B1/paper/figures/heldout_l_shape_reconstruction.png) | [`heldout_l_shape_results/`](heldout_l_shape_results/) | [`run_heldout_l_shape_validation.py`](run_heldout_l_shape_validation.py) |
| [`heldout_subcell_score_5000.png`](../B1/paper/figures/heldout_subcell_score_5000.png) | [`heldout_subcell_score_results/`](heldout_subcell_score_results/) | [`diagnose_heldout_subcell_score.py`](diagnose_heldout_subcell_score.py) |
| [`local_scale_ablation.png`](../B1/paper/figures/local_scale_ablation.png) | [`local_scale_candidate_results/`](local_scale_candidate_results/) · [`matched_diffusion_control_results/`](matched_diffusion_control_results/) | [`run_local_scale_candidate.py`](run_local_scale_candidate.py) |
| [`local_scale_images.png`](../B1/paper/figures/local_scale_images.png) | [`local_scale_candidate_results/`](local_scale_candidate_results/) · [`matched_diffusion_control_results/`](matched_diffusion_control_results/) | [`run_local_scale_candidate.py`](run_local_scale_candidate.py) |
| [`local_support_count_mechanism.png`](../B1/paper/figures/local_support_count_mechanism.png) | [`local_support_count_mechanism_results/`](local_support_count_mechanism_results/) | [`diagnose_local_support_count_mechanism.py`](diagnose_local_support_count_mechanism.py) |
| [`local_support_gate_75mm.png`](../B1/paper/figures/local_support_gate_75mm.png) | [`local_support_gate_validation_results/`](local_support_gate_validation_results/) | [`run_local_support_gate_validation.py`](run_local_support_gate_validation.py) |
| [`localizer_ablation_5000.png`](../B1/paper/figures/localizer_ablation_5000.png) | [`event_quality_localizer_ablation_results/`](event_quality_localizer_ablation_results/) | [`run_event_quality_localizer_ablation.py`](run_event_quality_localizer_ablation.py) |
| [`matched_diffusion_metrics.png`](../B1/paper/figures/matched_diffusion_metrics.png) | [`matched_diffusion_control_results/`](matched_diffusion_control_results/) | [`run_local_scale_candidate.py`](run_local_scale_candidate.py) |
| [`neff8_interaction.png`](../B1/paper/figures/neff8_interaction.png) | [`neff8_interaction_results/`](neff8_interaction_results/) | [`diagnose_neff8_interaction.py`](diagnose_neff8_interaction.py) |
| [`neighbor_borrowing_risk.png`](../B1/paper/figures/neighbor_borrowing_risk.png) | [`neighbor_borrowing_results/`](neighbor_borrowing_results/) | [`run_neighbor_borrowing.py`](run_neighbor_borrowing.py) |
| [`neighbor_signal_coverage.png`](../B1/paper/figures/neighbor_signal_coverage.png) | [`neighbor_signal_coverage_results/`](neighbor_signal_coverage_results/) · [`evidence.json`](neighbor_signal_coverage_results/evidence.json) | [`audit_neighbor_signal_coverage.py`](audit_neighbor_signal_coverage.py) |
| [`nested_count_gate_diagnostic.png`](../B1/paper/figures/nested_count_gate_diagnostic.png) | [`nested_count_gate_diagnostic_results/`](nested_count_gate_diagnostic_results/) | [`run_nested_count_gate_diagnostic.py`](run_nested_count_gate_diagnostic.py) |
| [`observable_rescue_support_5000_32.png`](../B1/paper/figures/observable_rescue_support_5000_32.png) | [`observable_rescue_support_results/`](observable_rescue_support_results/) | [`diagnose_observable_rescue_support.py`](diagnose_observable_rescue_support.py) |
| [`observed_features_vs_gain_5000_32.png`](../B1/paper/figures/observed_features_vs_gain_5000_32.png) | [`observed_z_gate_risk_results/`](observed_z_gate_risk_results/) | [`diagnose_observed_z_gate_risk.py`](diagnose_observed_z_gate_risk.py) |
| [`offcenter_5000_r1_sigma0.10.png`](../B1/paper/figures/offcenter_5000_r1_sigma0.10.png) | [`event_quality_failure_diagnosis/`](event_quality_failure_diagnosis/) | [`diagnose_event_quality_localization.py`](diagnose_event_quality_localization.py) |
| [`paired_tradeoff_5000.png`](../B1/paper/figures/paired_tradeoff_5000.png) | [`paired_event_quality_results/`](paired_event_quality_results/) | [`run_paired_event_quality_validation.py`](run_paired_event_quality_validation.py) |
| [`plus_iron_fallback_5000_32.png`](../B1/paper/figures/plus_iron_fallback_5000_32.png) | [`plus_iron_fallback_results/`](plus_iron_fallback_results/) | [`run_plus_iron_fallback_validation.py`](run_plus_iron_fallback_validation.py) |
| [`plus_material_control_5000_32.png`](../B1/paper/figures/plus_material_control_5000_32.png) | [`plus_material_control_results/`](plus_material_control_results/) | [`run_plus_material_control.py`](run_plus_material_control.py) |
| [`plus_thickness_control_5000_32.png`](../B1/paper/figures/plus_thickness_control_5000_32.png) | [`plus_thickness_control_results/`](plus_thickness_control_results/) | [`run_plus_thickness_control.py`](run_plus_thickness_control.py) |
| [`plus_thickness_mechanism_steps.png`](../B1/paper/figures/plus_thickness_mechanism_steps.png) | [`plus_thickness_mechanism_results/`](plus_thickness_mechanism_results/) | [`diagnose_plus_thickness_mechanism.py`](diagnose_plus_thickness_mechanism.py) |
| [`poca_z_strata_5000.png`](../B1/paper/figures/poca_z_strata_5000.png) | [`poca_z_stability_results/`](poca_z_stability_results/) | [`diagnose_poca_z_stability.py`](diagnose_poca_z_stability.py) |
| [`selection_attribute_5000.png`](../B1/paper/figures/selection_attribute_5000.png) | [`selection_attribute_results/`](selection_attribute_results/) | [`decompose_selection_attribute_effect.py`](decompose_selection_attribute_effect.py) |
| [`selection_gate_transitions_5000.png`](../B1/paper/figures/selection_gate_transitions_5000.png) | [`selection_gate_results/`](selection_gate_results/) | [`diagnose_selection_gate_transitions.py`](diagnose_selection_gate_transitions.py) |
| [`shape_images_sigma_0.05.png`](../B1/paper/figures/shape_images_sigma_0.05.png) | [`shape_smoothing_control_results/`](shape_smoothing_control_results/) | [`run_shape_smoothing_control.py`](run_shape_smoothing_control.py) |
| [`shape_smoothing_metrics.png`](../B1/paper/figures/shape_smoothing_metrics.png) | [`shape_smoothing_control_results/`](shape_smoothing_control_results/) | [`run_shape_smoothing_control.py`](run_shape_smoothing_control.py) |
| [`simple_vs_propagated_5000_32.png`](../B1/paper/figures/simple_vs_propagated_5000_32.png) | [`simple_z_gate_results/`](simple_z_gate_results/) | [`run_simple_z_gate_baseline.py`](run_simple_z_gate_baseline.py) |
| [`single_cell_refinement_signal.png`](../B1/paper/figures/single_cell_refinement_signal.png) | [`single_cell_refinement_signal_results/`](single_cell_refinement_signal_results/) | [`diagnose_single_cell_refinement_signal.py`](diagnose_single_cell_refinement_signal.py) |
| [`smoothing_support_diagnosis.png`](../B1/paper/figures/smoothing_support_diagnosis.png) | [`smoothing_support_diagnosis_results/`](smoothing_support_diagnosis_results/) | [`diagnose_smoothing_support.py`](diagnose_smoothing_support.py) |
| [`split_confidence_gate_diagnosis_5000.png`](../B1/paper/figures/split_confidence_gate_diagnosis_5000.png) | [`split_confidence_diagnosis_results/`](split_confidence_diagnosis_results/) | [`diagnose_split_confidence.py`](diagnose_split_confidence.py) |
| [`split_gate_ablation_5000.png`](../B1/paper/figures/split_gate_ablation_5000.png) | [`split_gate_ablation_results/`](split_gate_ablation_results/) | [`run_split_gate_ablation.py`](run_split_gate_ablation.py) |
| [`synthetic_risk.png`](../B1/paper/figures/synthetic_risk.png) | [`synthetic_subcell_risk_results/`](synthetic_subcell_risk_results/) | [`run_synthetic_subcell_risk.py`](run_synthetic_subcell_risk.py) |
| [`t_shape_metrics_5000.png`](../B1/paper/figures/t_shape_metrics_5000.png) | [`t_shape_independent_results/`](t_shape_independent_results/) | [`run_t_shape_independent_validation.py`](run_t_shape_independent_validation.py) |
| [`t_shape_reconstruction_5000.png`](../B1/paper/figures/t_shape_reconstruction_5000.png) | [`t_shape_independent_results/`](t_shape_independent_results/) | [`run_t_shape_independent_validation.py`](run_t_shape_independent_validation.py) |
| [`three_block_q_validation.png`](../B1/paper/figures/three_block_q_validation.png) | [`three_block_q_validation_results/`](three_block_q_validation_results/) | [`run_three_block_q_validation.py`](run_three_block_q_validation.py) |
| [`topk_tie_audit.png`](../B1/paper/figures/topk_tie_audit.png) | [`topk_tie_audit_results/`](topk_tie_audit_results/) | [`diagnose_combination_stability.py`](diagnose_combination_stability.py) |
| [`uncertainty_aware_angle_metrics.png`](../B1/paper/figures/uncertainty_aware_angle_metrics.png) | [`uncertainty_aware_angle_results/`](uncertainty_aware_angle_results/) | [`run_uncertainty_aware_angle_experiment.py`](run_uncertainty_aware_angle_experiment.py) |

## Experiment folders and record locations

The folder names below are the experiment/diagnostic identifiers already used by the project. Dates are the latest Git commit dates for those folders in the audited baseline; a date is not a claim about when a simulation was physically run. Follow the linked folder to its configuration, run summary, repeat-level data, plots, and `research_note.md` when present. The five `evidence.json` records are explicitly linked.

| Experiment or diagnostic | Result folder | Last Git update | Primary record / evidence |
|---|---|---:|---|
| Adaptive voxel | [`adaptive_voxel_results/`](adaptive_voxel_results/) | 2026-09-22 | experiment config, experiment summary, repeat metrics |
| Angle filter | [`angle_filter_results/`](angle_filter_results/) | 2026-09-22 | config, summary, repeat metrics |
| Angle weighting | [`angle_weighting_results/`](angle_weighting_results/) | 2026-09-22 | config, summary, repeat metrics |
| Baseline | [`baseline_results/`](baseline_results/) | 2026-09-21 | config, summary, repeat metrics |
| Boundary PoCA diagnosis | [`boundary_poca_diagnosis_results/`](boundary_poca_diagnosis_results/) | 2026-09-25 | [`research_note.md`](boundary_poca_diagnosis_results/research_note.md), config, summary |
| Combination stability | [`combination_stability_results/`](combination_stability_results/) | 2026-09-27 | [`research_note.md`](combination_stability_results/research_note.md), config, repeat/association records |
| Continuous event quality | [`continuous_event_quality_results/`](continuous_event_quality_results/) | 2026-09-23 | config, summary, repeat records |
| Continuous image audit | [`continuous_image_audit_results/`](continuous_image_audit_results/) | 2026-09-27 | [`research_note.md`](continuous_image_audit_results/research_note.md), config, image and audit records |
| Detector resolution | [`detector_resolution_results/`](detector_resolution_results/) | 2026-09-22 | config, summary, repeat metrics |
| Disjoint spatial support | [`disjoint_spatial_support_results/`](disjoint_spatial_support_results/) | 2026-10-03 | [`research_note.md`](disjoint_spatial_support_results/research_note.md), [`evidence.json`](disjoint_spatial_support_results/evidence.json) |
| Double-block validation | [`double_block_validation_results/`](double_block_validation_results/) | 2026-09-24 | [`research_note.md`](double_block_validation_results/research_note.md), config, paired summary |
| Edge-H held-out | [`edge_h_heldout_results/`](edge_h_heldout_results/) | 2026-09-29 | [`research_note.md`](edge_h_heldout_results/research_note.md), frozen config, paired records |
| Empty-target control | [`empty_target_control_results/`](empty_target_control_results/) | 2026-10-06 | [`research_note.md`](empty_target_control_results/research_note.md), [`evidence.json`](empty_target_control_results/evidence.json) |
| Event-quality localizer ablation | [`event_quality_localizer_ablation_results/`](event_quality_localizer_ablation_results/) | 2026-09-23 | [`research_note.md`](event_quality_localizer_ablation_results/research_note.md), config, summary |
| Event-selection shift | [`event_selection_shift_results/`](event_selection_shift_results/) | 2026-09-23 | config, summary, diagnostics |
| Event uncertainty | [`event_uncertainty_results/`](event_uncertainty_results/) | 2026-09-23 | config, summary, repeat metrics |
| Finite-sample subcell risk | [`finite_sample_subcell_risk_results/`](finite_sample_subcell_risk_results/) | 2026-09-29 | [`research_note.md`](finite_sample_subcell_risk_results/research_note.md), results JSON, plot |
| Fixed-budget refinement | [`fixed_budget_refinement_results/`](fixed_budget_refinement_results/) | 2026-09-26 | [`research_note.md`](fixed_budget_refinement_results/research_note.md), config, paired/repeat records |
| Fixed-image support pool | [`fixed_image_support_pool_results/`](fixed_image_support_pool_results/) | 2026-09-26 | [`research_note.md`](fixed_image_support_pool_results/research_note.md), config, identity/paired records |
| Fractional-IoU audit | [`fractional_iou_audit_results/`](fractional_iou_audit_results/) | 2026-09-27 | [`research_note.md`](fractional_iou_audit_results/research_note.md), config, audit records |
| Frame-gate score audit | [`frame_gate_score_audit_results/`](frame_gate_score_audit_results/) | 2026-09-28 | [`research_note.md`](frame_gate_score_audit_results/research_note.md), config, paired summary |
| Frame smoothing control | [`frame_smoothing_control_results/`](frame_smoothing_control_results/) | 2026-09-28 | [`research_note.md`](frame_smoothing_control_results/research_note.md), config, paired summary |
| Frame-z gate | [`frame_z_gate_results/`](frame_z_gate_results/) | 2026-09-25 | [`research_note.md`](frame_z_gate_results/research_note.md), config, paired summary |
| Held-out L shape | [`heldout_l_shape_results/`](heldout_l_shape_results/) | 2026-09-24 | [`research_note.md`](heldout_l_shape_results/research_note.md), config, paired records |
| Held-out subcell score | [`heldout_subcell_score_results/`](heldout_subcell_score_results/) | 2026-09-25 | [`research_note.md`](heldout_subcell_score_results/research_note.md), config, cell/summary records |
| Hotspot diagnosis | [`hotspot_diagnosis_results/`](hotspot_diagnosis_results/) | 2026-09-23 | config, summary, diagnostic plots |
| Local region | [`local_region_results/`](local_region_results/) | 2026-09-23 | config, summary, repeat records |
| Local-scale candidate | [`local_scale_candidate_results/`](local_scale_candidate_results/) | 2026-09-28 | [`research_note.md`](local_scale_candidate_results/research_note.md), config, paired records |
| Local-support count mechanism | [`local_support_count_mechanism_results/`](local_support_count_mechanism_results/) | 2026-09-26 | [`research_note.md`](local_support_count_mechanism_results/research_note.md), config, summary |
| Local-support gate validation | [`local_support_gate_validation_results/`](local_support_gate_validation_results/) | 2026-09-26 | [`research_note.md`](local_support_gate_validation_results/research_note.md), config, paired records |
| Local-support refinement | [`local_support_refinement_results/`](local_support_refinement_results/) | 2026-09-24 | [`research_note.md`](local_support_refinement_results/research_note.md), config, paired records |
| Matched-diffusion control | [`matched_diffusion_control_results/`](matched_diffusion_control_results/) | 2026-09-28 | [`research_note.md`](matched_diffusion_control_results/research_note.md), config, paired summary |
| Multiscale localization | [`multiscale_localization_results/`](multiscale_localization_results/) | 2026-09-23 | config(s), summaries, repeat records |
| Neff-8 interaction | [`neff8_interaction_results/`](neff8_interaction_results/) | 2026-09-26 | [`research_note.md`](neff8_interaction_results/research_note.md), config, summary |
| Neighbor-borrowing risk | [`neighbor_borrowing_results/`](neighbor_borrowing_results/) | 2026-09-30 | [`research_note.md`](neighbor_borrowing_results/research_note.md), results JSON, plot |
| Neighbor-signal coverage | [`neighbor_signal_coverage_results/`](neighbor_signal_coverage_results/) | 2026-09-30 | [`research_note.md`](neighbor_signal_coverage_results/research_note.md), [`evidence.json`](neighbor_signal_coverage_results/evidence.json) |
| Nested count-gate diagnostic | [`nested_count_gate_diagnostic_results/`](nested_count_gate_diagnostic_results/) | 2026-09-26 | [`research_note.md`](nested_count_gate_diagnostic_results/research_note.md), config, identity/paired records |
| Noise-aware refinement gate-8 | [`noise_aware_refinement_gate8_results/`](noise_aware_refinement_gate8_results/) | 2026-09-24 | config, comparison, paired summary |
| Noise-aware refinement | [`noise_aware_refinement_results/`](noise_aware_refinement_results/) | 2026-09-24 | [`research_note.md`](noise_aware_refinement_results/research_note.md), config, paired summary |
| Noise-floor fallback | [`noise_floor_fallback_results/`](noise_floor_fallback_results/) | 2026-09-26 | config, summary, repeat records |
| Observable event schema | [`observable_event_schema_results/`](observable_event_schema_results/) | 2026-09-30 | [`research_note.md`](observable_event_schema_results/research_note.md), [`evidence.json`](observable_event_schema_results/evidence.json) |
| Observable rescue support | [`observable_rescue_support_results/`](observable_rescue_support_results/) | 2026-09-26 | [`research_note.md`](observable_rescue_support_results/research_note.md), config, summary |
| Observed-z gate risk | [`observed_z_gate_risk_results/`](observed_z_gate_risk_results/) | 2026-09-26 | [`research_note.md`](observed_z_gate_risk_results/research_note.md), config, summary |
| Off-center generalization | [`offcenter_generalization_results/`](offcenter_generalization_results/) | 2026-09-22 | config, summary, repeat metrics |
| Paired event quality | [`paired_event_quality_results/`](paired_event_quality_results/) | 2026-09-23 | [`research_note.md`](paired_event_quality_results/research_note.md), paired records |
| Plus-iron fallback | [`plus_iron_fallback_results/`](plus_iron_fallback_results/) | 2026-09-26 | [`research_note.md`](plus_iron_fallback_results/research_note.md), config, paired summary |
| Plus-material control | [`plus_material_control_results/`](plus_material_control_results/) | 2026-09-26 | [`research_note.md`](plus_material_control_results/research_note.md), config, material/paired records |
| Plus-thickness control | [`plus_thickness_control_results/`](plus_thickness_control_results/) | 2026-09-26 | [`research_note.md`](plus_thickness_control_results/research_note.md), config, paired/source counts |
| Plus-thickness mechanism | [`plus_thickness_mechanism_results/`](plus_thickness_mechanism_results/) | 2026-09-26 | [`research_note.md`](plus_thickness_mechanism_results/research_note.md), config, summary |
| PoCA-z calibration | [`poca_z_calibration_results/`](poca_z_calibration_results/) | 2026-09-25 | [`research_note.md`](poca_z_calibration_results/research_note.md), config, calibration summaries |
| PoCA-z incremental value | [`poca_z_incremental_results/`](poca_z_incremental_results/) | 2026-09-25 | [`research_note.md`](poca_z_incremental_results/research_note.md), config, paired records |
| PoCA-z stability | [`poca_z_stability_results/`](poca_z_stability_results/) | 2026-09-25 | [`research_note.md`](poca_z_stability_results/research_note.md), config, stratified summary |
| Refinement checkpoint | [`refinement_checkpoint_results/`](refinement_checkpoint_results/) | 2026-09-29 | [`research_note.md`](refinement_checkpoint_results/research_note.md), [`evidence.json`](refinement_checkpoint_results/evidence.json) |
| Refinement fallback | [`refinement_fallback_results/`](refinement_fallback_results/) | 2026-09-24 | [`research_note.md`](refinement_fallback_results/research_note.md), config, policy summary |
| Robust localization | [`robust_localization_results/`](robust_localization_results/) | 2026-09-23 | config(s), summaries, repeat records |
| Selection attribute | [`selection_attribute_results/`](selection_attribute_results/) | 2026-09-25 | [`research_note.md`](selection_attribute_results/research_note.md), config, summary |
| Selection gate | [`selection_gate_results/`](selection_gate_results/) | 2026-09-25 | [`research_note.md`](selection_gate_results/research_note.md), config, summary |
| Shape-smoothing control | [`shape_smoothing_control_results/`](shape_smoothing_control_results/) | 2026-09-28 | [`research_note.md`](shape_smoothing_control_results/research_note.md), config, paired/image records |
| Simple-z gate baseline | [`simple_z_gate_results/`](simple_z_gate_results/) | 2026-09-25 | [`research_note.md`](simple_z_gate_results/research_note.md), config, paired summary |
| Single-cell refinement signal | [`single_cell_refinement_signal_results/`](single_cell_refinement_signal_results/) | 2026-09-26 | [`research_note.md`](single_cell_refinement_signal_results/research_note.md), config, cell/summary records |
| Small-target generalization | [`small_target_generalization_results/`](small_target_generalization_results/) | 2026-09-22 | config, summary, repeat metrics |
| Smoothing-support diagnosis | [`smoothing_support_diagnosis_results/`](smoothing_support_diagnosis_results/) | 2026-09-28 | [`research_note.md`](smoothing_support_diagnosis_results/research_note.md), config, diagnostic records |
| Soft uncertainty | [`soft_uncertainty_results/`](soft_uncertainty_results/) | 2026-09-23 | config, summary, repeat metrics |
| Soft-z gate | [`soft_z_gate_results/`](soft_z_gate_results/) | 2026-09-25 | [`research_note.md`](soft_z_gate_results/research_note.md), config, paired summary |
| Soft-z mechanism | [`soft_z_mechanism_results/`](soft_z_mechanism_results/) | 2026-09-25 | [`research_note.md`](soft_z_mechanism_results/research_note.md), config, paired summary |
| Split-confidence diagnosis | [`split_confidence_diagnosis_results/`](split_confidence_diagnosis_results/) | 2026-09-24 | [`research_note.md`](split_confidence_diagnosis_results/research_note.md), config, summary |
| Split-gate ablation | [`split_gate_ablation_results/`](split_gate_ablation_results/) | 2026-09-24 | [`research_note.md`](split_gate_ablation_results/research_note.md), config, summary |
| Synthetic subcell risk | [`synthetic_subcell_risk_results/`](synthetic_subcell_risk_results/) | 2026-09-29 | [`research_note.md`](synthetic_subcell_risk_results/research_note.md), results JSON |
| T-shape independent validation | [`t_shape_independent_results/`](t_shape_independent_results/) | 2026-09-24 | [`research_note.md`](t_shape_independent_results/research_note.md), frozen config, paired summary |
| Three-block Q validation | [`three_block_q_validation_results/`](three_block_q_validation_results/) | 2026-09-26 | [`research_note.md`](three_block_q_validation_results/research_note.md), config, paired/source counts |
| Top-k tie audit | [`topk_tie_audit_results/`](topk_tie_audit_results/) | 2026-09-27 | [`research_note.md`](topk_tie_audit_results/research_note.md), config, bootstrap/sensitivity records |
| Uncertainty-aware angle | [`uncertainty_aware_angle_results/`](uncertainty_aware_angle_results/) | 2026-09-23 | config, summary, repeat metrics |
| Uncertainty cross-scenario | [`uncertainty_cross_scenario_results/`](uncertainty_cross_scenario_results/) | 2026-09-23 | config, summary, repeat records |
| Z-pooled refinement | [`z_pooled_refinement_results/`](z_pooled_refinement_results/) | 2026-09-24 | [`research_note.md`](z_pooled_refinement_results/research_note.md), config, paired summary |

Other research notes not stored in a `*_results/` folder include [`event_quality_failure_diagnosis/research_note.md`](event_quality_failure_diagnosis/research_note.md). Run plans and analysis protocols are also kept in `muon_analysis/` as `*_PLAN.md`/design notes; the existing paper guide links several of them to their matching experiment folder. These are records of planned methods and interpretation constraints, not additional experimental outcomes.

## Papers, reports, slides, and other document files

| File/resource | Purpose and provenance | Tracking status / handling |
|---|---|---|
| [`B1/paper/main.tex`](../B1/paper/main.tex) | Editable LaTeX manuscript source | Tracked |
| [`B1/paper/references.bib`](../B1/paper/references.bib) | Bibliography used by the manuscript | Tracked |
| [`B1/paper/README.md`](../B1/paper/README.md) | Existing detailed writing, result provenance, and scientific interpretation cautions | Tracked; retained as the authoritative writing guide |
| `B1/paper/低统计量条件下缪子散射成像方法研究_基线结果草稿.docx` | Earlier Word baseline-results draft | Tracked in the audited baseline |
| `B1/paper/main.pdf` | Locally compiled PDF output from the manuscript | Local-only/untracked in the source checkout; do not add as a second manuscript source |
| `B1/paper/低统计量条件下缪子散射成像方法研究_论文草稿模板.docx` | Editable Word paper template | Local-only/untracked; do not upload without an explicit decision |
| `B1/presentation-output/muon-imaging-research-direction-v2.pptx` | Research-direction presentation | Local-only/untracked |
| `B1/presentation-output/muon-imaging-research-direction.pptx` | Earlier research-direction presentation | Local-only/untracked |
| `B1/presentation-output/低统计量缪子散射成像研究方向汇报.pptx` | Chinese presentation for research reporting | Local-only/untracked |
| `B1/.codex-ppt-build/` and `B1/.codex-finalizer/` presentation candidates | Temporary generation/validation outputs, not curated research deliverables | Local/ignored scratch; excluded from this index’s deliverable list and this branch |
| PDFs of external literature | No full-text literature PDF was found in the repository at audit time | Do not copy or upload papers from Zotero without rights/authorization; the cited source remains outside this project |

The three presentation decks and compiled PDF were identified in the source checkout, but intentionally are **not present in this branch**. Their content is not reproduced here. This status should be re-audited before any future decision to publish or share them.

## Tracking, duplicates, size, and sensitivity

### Git tracking and duplicate files

- At the audited baseline, the tracked research-resource scan counted **548 tracked** and **57 non-tracked/local-only** resource files among the selected document/data/image extensions. These counts include ignored local artifacts in the “non-tracked” group; a clean checkout will not contain them.
- The 56 files in `B1/paper/figures/` are exact byte-for-byte copies of source plots under the linked result folders. They are retained as the LaTeX manuscript’s figure assets, not treated as disposable duplicates.
- Exact-content duplicate CSV groups found in different experiment folders:
  - `combination_stability_results/summary.csv` and `topk_tie_audit_results/summary.csv`.
  - `local_support_gate_validation_results/source_counts.csv` and `nested_count_gate_diagnostic_results/source_counts.csv`.
  - `fractional_iou_audit_results/selected_cells.csv`, `continuous_image_audit_results/selected_cells.csv`, and `fixed_budget_refinement_results/selected_cells.csv`.
- Duplicate files were not deleted or rewritten. Identical bytes do not establish identical experimental intent; keep each copy with its experiment record unless a researcher reviews the provenance.

### Large files and raw data

- The source checkout contains **29 tracked `muon_analysis/muon_tracks*.csv` event files**, about 1.4 MB each. These are raw/simulation event data already tracked before this index; this task does not add more.
- Two tracked diagnostic tables exceed 1 MB: [`poca_z_incremental_results/event_features.csv`](poca_z_incremental_results/event_features.csv) and [`single_cell_refinement_signal_results/cells.csv`](single_cell_refinement_signal_results/cells.csv).
- `muon_analysis/analysis.ipynb` is about 9 MB. It is already tracked; local notebook edits from the source checkout are not included in this branch.
- The local compiled `B1/paper/main.pdf` is about 8.3 MiB and remains untracked. No large file or raw data is being committed by this change.
- The source checkout also showed a local `muon_analysis/muon_tracks.csv` of about 1.4 MB that was not part of the tracked baseline. It is not copied or staged here. Obtain explicit researcher approval before adding any new raw event dataset to Git.

### Sensitive information

- A marker scan of tracked text/config/source files found no GitHub-token, AWS access-key, or private-key markers. This is a limited pattern scan, not a guarantee that all sensitive information has been ruled out.
- Some pre-existing tracked notes/code contain machine-specific absolute paths with the local account name. They were not reproduced in this index and were not modified during the audit. Review those files before making the repository public; the current repository is intended to remain private.
- No credentials, private literature PDFs, local presentation files, or user-specific local-only documents were added to this branch.

## How to keep this index current

When a researcher approves a new experiment or document for repository tracking:

1. Preserve its run configuration, input/data provenance, random seeds, and exact reproduction command alongside the result; retain `evidence.json` when the run workflow produces one.
2. Keep outputs in the existing matching `muon_analysis/<experiment>_results/` folder. Link each figure to its producing script and source result folder; copy a plot into `B1/paper/figures/` only when it is intentionally selected for the manuscript.
3. Record the Git status and file size before staging. Never include new raw data, local-only documents, caches, credentials, or third-party full text without the researcher’s authorization.
4. Update the experiment row and the audit date here. Do not change scientific thresholds, metrics, validation labels, or conclusions as part of catalog maintenance.
