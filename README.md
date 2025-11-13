# LB EEG (fsaverage)

This repo includes three small scripts that work:

- `compute_phi_lapy.py` — Laplace–Beltrami (LB) eigenmodes on **fsaverage** using LaPy; reindexed to an MNE source space.
- `make_fsaverage_lb_dictionary.py` — builds a sensor-space dictionary `D = G @ Phi` on **fsaverage** (BEM + forward).
- `lemon_batch_tfr.py` — batch Morlet TFR for LEMON EEG (EO/EC), with boundary-aware epoching and per-subject channel masks.

## Requirements

- Python ≥ 3.9  
- `mne`, `nibabel`, `numpy`, `pandas`, `requests`, `lapy` (and MNE’s fsaverage download)
- (for `make_fsaverage_lb_dictionary.py`) MNE BEM tools (no FreeSurfer install required; uses MNE’s packaged fsaverage)

Install (example):
```bash
pip install mne nibabel numpy pandas requests lapy



# Make sure SUBJECTS_DIR isn’t pointing somewhere odd...
unset SUBJECTS_DIR

python - <<'PY'
import os, mne
sd = os.path.expanduser('~/mne_data/MNE-fsaverage-data')
os.makedirs(sd, exist_ok=True)
mne.set_config('SUBJECTS_DIR', sd, set_env=True)   # set both config and env
p = mne.datasets.fetch_fsaverage(verbose=True)
print('SUBJECTS_DIR =', mne.get_config('SUBJECTS_DIR'))
print('fsaverage dir:', p)
PY

## Compute LB eigenmodes (Phi)
python - <<'PY'
from compute_phi_lapy import get_phi_fsaverage
import numpy as np
Phi, evals_lh, evals_rh, src = get_phi_fsaverage(
    K=60, spacing='ico4', combine='sym', verbose=True
)
np.savez_compressed('phi_fsavg_sym_K60_ico4.npz',
                    Phi=Phi, evals_lh=evals_lh, evals_rh=evals_rh)
print('[ok] Saved phi_fsavg_sym_K60_ico4.npz')
PY


python make_fsaverage_lb_dictionary.py \
  --canonical-file canonical_59.txt \
  --outdir ~/lemon_work \
  --K 60 \
  --spacing ico4 \
  --bem-ico 4


python lemon_batch_tfr.py \
  --outdir ~/lemon_work \
  --canonical-file canonical_59.txt \
  --win-len 10 --fmin 2 --fmax 30 --n-freqs 15 --cycles-rule constQ --decim 2



# LB (59, 32, 19) (note per-subset subdir) 
python run_lb_fits.py \
  --dict-npz  ~/lemon_work/derivatives/Dict/fsaverage_D_sym_K60_M59_ico4.npz \
  --manifest  ~/lemon_work/derivatives/Y_tfr/manifest.csv \
  --outdir    ~/lemon_work \
  --use-bands

python run_lb_fits.py \
  --dict-npz  ~/lemon_work/derivatives/Dict/fsaverage_D_sym_K60_M59_ico4.npz \
  --manifest  ~/lemon_work/derivatives/Y_tfr/manifest.csv \
  --outdir    ~/lemon_work \
  --use-bands \
  --subset-file canonical_32.txt

python run_lb_fits.py \
  --dict-npz  ~/lemon_work/derivatives/Dict/fsaverage_D_sym_K60_M59_ico4.npz \
  --manifest  ~/lemon_work/derivatives/Y_tfr/manifest.csv \
  --outdir    ~/lemon_work \
  --use-bands \
  --subset-file canonical_19.txt


## Baselines (59, 32, 19) (note per-subset subdir) uses dict-npz only for channel order 
python run_baselines_pca_ica_sph.py \
  --dict-npz  ~/lemon_work/derivatives/Dict/fsaverage_D_sym_K60_M59_ico4.npz \
  --canonical-file canonical_59.txt \
  --manifest  ~/lemon_work/derivatives/Y_tfr/manifest.csv \
  --outdir    ~/lemon_work \
  --use-bands

python run_baselines_pca_ica_sph.py \
  --dict-npz  ~/lemon_work/derivatives/Dict/fsaverage_D_sym_K60_M59_ico4.npz \
  --canonical-file canonical_59.txt \
  --manifest  ~/lemon_work/derivatives/Y_tfr/manifest.csv \
  --outdir    ~/lemon_work \
  --use-bands \
  --subset-file canonical_32.txt

python run_baselines_pca_ica_sph.py \
  --dict-npz  ~/lemon_work/derivatives/Dict/fsaverage_D_sym_K60_M59_ico4.npz \
  --canonical-file canonical_59.txt \
  --manifest  ~/lemon_work/derivatives/Y_tfr/manifest.csv \
  --outdir    ~/lemon_work \
  --use-bands \
  --subset-file canonical_19.txt


## After running your 3 LB runs and 3 Baseline runs (ALL59 / canonical_32 / canonical_19):
python bootstrap_curves.py \
  --root   ~/lemon_work/derivatives \
  --outdir ~/lemon_work/derivatives/Merged_Report \
  --B 2000 --alpha 0.05
