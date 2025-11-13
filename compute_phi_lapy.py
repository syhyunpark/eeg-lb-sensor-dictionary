#!/usr/bin/env python3
# compute_phi_lapy.py — fsaverage LB modes via LaPy; map to MNE src; simple combos

import os
import numpy as np
import nibabel as nib
from lapy import TriaMesh, Solver
import mne
from scipy.spatial import cKDTree

def _get_subjects_dir():
    p = mne.datasets.fetch_fsaverage(verbose=True)
    return os.path.dirname(p) if os.path.basename(p) == "fsaverage" else p

def _read_surf(path):
    v, f = nib.freesurfer.read_geometry(path)
    return v.astype(np.float64), f.astype(np.int32)

def _eigs_lapy_per_hemi(v, f, K, lump=False):
    # first K non-DC
    mesh = TriaMesh(v, f)
    sol = Solver(mesh, lump=lump)
    evals, evecs = sol.eigs(K + 1)
    return evals[1:K+1], evecs[:, 1:K+1]

def _pair_hemis_src(src, lh_verts, rh_verts):
    # NN pairing after x->-x mirror
    lh_xyz = src[0]['rr'][lh_verts]
    rh_xyz = src[1]['rr'][rh_verts]
    rh_m = rh_xyz.copy(); rh_m[:, 0] *= -1.0
    lh_m = lh_xyz.copy(); lh_m[:, 0] *= -1.0
    _, nn_L = cKDTree(lh_xyz).query(rh_m, k=1)
    _, nn_R = cKDTree(rh_xyz).query(lh_m, k=1)
    return nn_L.astype(int), nn_R.astype(int)

def get_phi_fsaverage(K=50, spacing='ico4', subjects_dir=None, combine='sym', src=None, verbose=True):
    # returns (Phi_src, evals_lh, evals_rh, src)
    if subjects_dir is None:
        subjects_dir = _get_subjects_dir()
    subject = "fsaverage"

    # meshes
    lh_p = os.path.join(subjects_dir, subject, "surf", "lh.pial")
    rh_p = os.path.join(subjects_dir, subject, "surf", "rh.pial")
    vL, fL = _read_surf(lh_p)
    vR, fR = _read_surf(rh_p)
    VL = vL.shape[0]

    # eigs per hemi
    evalL, evecL = _eigs_lapy_per_hemi(vL, fL, K, lump=False)
    evalR, evecR = _eigs_lapy_per_hemi(vR, fR, K, lump=False)

    # block [LH | RH] in mesh order
    Phi_block_mesh = np.zeros((VL + vR.shape[0], 2*K), float)
    Phi_block_mesh[:VL, :K] = evecL
    Phi_block_mesh[VL:, K:] = evecR

    # src + vert mapping
    if src is None:
        src = mne.setup_source_space(subject=subject, spacing=spacing, add_dist=False,
                                     subjects_dir=subjects_dir, verbose=verbose)
    lh_verts = src[0]['vertno']; rh_verts = src[1]['vertno']
    vert_idx = np.concatenate([lh_verts, rh_verts + VL])
    Phi_block_src = Phi_block_mesh[vert_idx, :]  # (Vsrc x 2K)

    if combine == 'block':
        Phi_src = Phi_block_src
        K_out = 2*K
        if verbose:
            print(f"[phi] spacing={spacing} Vsrc={Phi_src.shape[0]} K_out={K_out} combine=block")
        return Phi_src, evalL, evalR, src

    # split by hemi in src order
    nL = len(lh_verts); nR = len(rh_verts)
    PhiL = Phi_block_src[:nL, :K]
    PhiR = Phi_block_src[nL:, K:]

    # sign align RH to LH (per mode) using mirrored NN pairing
    map_R2L, map_L2R = _pair_hemis_src(src, lh_verts, rh_verts)
    R_to_L = PhiR[map_R2L, :]  # RH mirrored onto LH verts
    sgn = np.where(np.sum(PhiL * R_to_L, axis=0) >= 0.0, 1.0, -1.0)[None, :]
    PhiR_aligned = PhiR * sgn

    if combine == 'sym':
        # just stack halves (no mixing), after sign-align
        Phi_src = np.vstack([PhiL, PhiR_aligned])  # (Vsrc x K)
        K_out = K
    elif combine == 'sym+antisym':
        L_to_R = PhiL[map_L2R, :]
        sgn_L = np.where(np.sum(PhiL * R_to_L, axis=0) >= 0.0, 1.0, -1.0)[None, :]
        sgn_R = np.where(np.sum(PhiR * L_to_R, axis=0) >= 0.0, 1.0, -1.0)[None, :]
        R_to_L *= sgn_L; L_to_R *= sgn_R
        Sym = np.vstack([(PhiL + R_to_L)/np.sqrt(2.0), (PhiR + L_to_R)/np.sqrt(2.0)])
        Anti = np.vstack([(PhiL - R_to_L)/np.sqrt(2.0), (PhiR - L_to_R)/np.sqrt(2.0)])
        Phi_src = np.hstack([Sym, Anti])
        K_out = 2*K
    else:
        raise ValueError("combine must be one of {'sym','block','sym+antisym'}")

    if verbose:
        Vsrc = Phi_src.shape[0]
        print(f"[phi] spacing={spacing} Vsrc={Vsrc} K_out={K_out} combine={combine} "
              f"col0 μ={np.mean(Phi_src[:,0]):.3g} σ={np.std(Phi_src[:,0]):.3g}")
    return Phi_src, evalL, evalR, src
