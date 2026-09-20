# Muon Imaging Reconstruction

This repository is a reproducible baseline for a four-plane Geant4 muon-scattering imaging study.  It contains the detector simulation and the Python/Notebook analysis in one project.

## Baseline scope

- Four scintillator detector planes: two upstream and two downstream;
- A central U-shaped aluminium target;
- Event-level hit recording and two-segment track fitting;
- Point of Closest Approach (PoCA) reconstruction;
- Fixed-voxel, scattering-angle-weighted imaging in a Jupyter notebook.

## Layout

```text
B1/                 Geant4 simulation and macro files
muon_analysis/      CSV output location and analysis.ipynb
```

## Quick start

Configure and build Geant4 in `B1/build`, then run the 10,000-event baseline:

```bash
cd B1/build
./exampleB1 ../run3.mac
```

The simulation writes `muon_analysis/muon_tracks.csv`.  Open `muon_analysis/analysis.ipynb` to create the PoCA and fixed-voxel images.

## Version plan

`v0.1-poca-baseline` is the starting point.  Future work should use separate branches for adaptive voxel reconstruction, low-statistics event selection, and learning-based denoising.
