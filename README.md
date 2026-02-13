# LB EEG (fsaverage)

This repository includes three small scripts that work:

- `compute_phi_lapy.py` — Laplace–Beltrami (LB) eigenmodes on **fsaverage** using LaPy; reindexed to an MNE source space.
- `make_fsaverage_lb_dictionary.py` — builds a sensor-space dictionary `D = G @ Phi` on **fsaverage** (BEM + forward).
- `lemon_batch_tfr.py` — batch Morlet TFR for LEMON EEG (EO/EC), with boundary-aware epoching and per-subject channel masks.

## Requirements

- Python ≥ 3.9  
- `mne`, `nibabel`, `numpy`, `pandas`, `requests`, `lapy` (and MNE’s fsaverage download)
- (for `make_fsaverage_lb_dictionary.py`) MNE BEM tools (no FreeSurfer install required; uses MNE’s packaged fsaverage templates)

Install (example):
```bash
pip install mne nibabel numpy pandas requests lapy



# Make sure SUBJECTS_DIR is not pointing somewhere odd...
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
```




# Build and export the LB sensor dictionary

If you only want the final **cortex-informed sensor-space dictionary** (e.g., no span tests), use:

- `build_lb_sensor_dictionary.py` — builds and exports **two products**:
  - **`D_native`**: the forward-projected LB dictionary \(D = L\Phi\) (native leadfield scaling)
  - **`D_unitnorm`**: column-wise unit-norm version of `D_native` for comparable mode scoring

The **mode index \(k\)** follows the LB ordering (increasing eigenvalue; coarse → fine spatial scale).

### Requirements
- Python ≥ 3.9
- `mne`, `numpy`, `nibabel`, `lapy` (and MNE’s fsaverage download)

Install:
```bash
pip install mne numpy nibabel lapy
```

### Run (one command)

This will download/verify fsaverage (first run may take a few minutes) and then build the dictionary (also a few minutes).

```bash
python3 build_lb_sensor_dictionary.py \
  --outdir ./lb_dictionary_release \
  --channels-file canonical_59.txt \
  --K 60 --spacing ico4 --bem-ico 4 \
  --conductivity 0.3 0.006 0.3 \
  --combine sym \
  --overwrite
```

### Outputs

Written under ./lb_dictionary_release/:
- D_native_fsaverage_ico4_K60_M59_bemico4.npz
- D_unitnorm_fsaverage_ico4_K60_M59_bemico4.npz
- metadata.json (build parameters + versions) 

Each .npz contains:
- D (shape M × K)
- channels (row order)
- col_norms where col_norms[k] = ||D_native[:,k]||2 (used to convert between native and unit-norm)
- evals_lh, evals_rh (hemispheric LB eigenvalues; increasing order)

### Load in Python

```python
import numpy as np

d = np.load("./lb_dictionary_release/D_native_fsaverage_ico4_K60_M59_bemico4.npz", allow_pickle=True)
D = d["D"]                      # (M, K)
channels = list(d["channels"])  # row order of D
col_norms = d["col_norms"]      # ||D_native[:,k]||2
evals_lh = d["evals_lh"]        # LH eigenvalues (in increasing order)
evals_rh = d["evals_rh"]        # RH eigenvalues (in increasing order)
```

### Native vs unit-norm dictionary
- **Native scaling (D_native)**: most literal interpretation (“column k is the scalp projection of cortical mode k”).
- **Unit-norm (D_unitnorm)**: removes mode-dependent "gain" differences induced by the leadfield, useful for dot-products / correlations / “activation strength” comparisons across modes.

Load unit-norm:
```python
import numpy as np

du = np.load("./lb_dictionary_release/D_unitnorm_fsaverage_ico4_K60_M59_bemico4.npz", allow_pickle=True)
D_unit = du["D"]
```

### Project a scalp map y onto the dictionary
Assuming y is in the same channel order as channels:
```python
import numpy as np

# OLS coefficients in native column coordinates
c_native, *_ = np.linalg.lstsq(D, y, rcond=None)
y_hat = D @ c_native
```
