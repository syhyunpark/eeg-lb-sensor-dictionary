#!/usr/bin/env python3
# make_fsaverage_lb_dictionary.py — D = G @ Phi on fsaverage (combine='sym')

import os, argparse
import numpy as np
import mne
from compute_phi_lapy import get_phi_fsaverage

def _get_subjects_dir():
    p = mne.datasets.fetch_fsaverage(verbose=True)
    return os.path.dirname(p) if os.path.basename(p) == "fsaverage" else p

def read_canonical_list(path):
    # one per line, keep order, drop dups
    with open(path, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    out, seen = [], set()
    for ch in names:
        if ch not in seen:
            out.append(ch); seen.add(ch)
    return out

def main():
    ap = argparse.ArgumentParser(description="Make fsaverage LB dictionary D = G @ Phi (symmetric modes)")
    ap.add_argument("--canonical-file", required=True)
    ap.add_argument("--outdir", default=os.path.expanduser("~/lemon_work"))
    ap.add_argument("--K", type=int, default=60)
    ap.add_argument("--spacing", default="ico4", choices=["ico3","ico4","ico5","oct6"])
    ap.add_argument("--bem-ico", type=int, default=4)
    args = ap.parse_args()

    out_root = os.path.join(os.path.expanduser(args.outdir), "derivatives", "Dict")
    os.makedirs(out_root, exist_ok=True)

    canonical = read_canonical_list(args.canonical_file)
    M = len(canonical)
    print(f"[dict] canonical M={M}")

    # minimal Info (positions from montage)
    info = mne.create_info(ch_names=canonical, sfreq=250.0, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")

    subjects_dir = _get_subjects_dir()
    subject = "fsaverage"

    # src + BEM
    src = mne.setup_source_space(subject=subject, spacing=args.spacing, add_dist=False,
                                 subjects_dir=subjects_dir, verbose=True)
    conductivity = (0.3, 0.006, 0.3)
    model = mne.make_bem_model(subject=subject, ico=args.bem_ico,
                               conductivity=conductivity, subjects_dir=subjects_dir, verbose=True)
    bem = mne.make_bem_solution(model)

    # forward (surface, fixed ori)
    fwd = mne.make_forward_solution(info=info, trans="fsaverage", src=src, bem=bem,
                                    meg=False, eeg=True, mindist=5.0, n_jobs=1, verbose=True)
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    G = fwd_fixed["sol"]["data"]  # (M x Vsrc)
    print(f"[dict] G shape={G.shape}")

    # Phi on the SAME src; combine='sym' stacks LH + sign-aligned RH
    Phi_src, evalL, evalR, _ = get_phi_fsaverage(
        K=args.K, spacing=args.spacing, subjects_dir=subjects_dir, combine="sym", src=src, verbose=True
    )
    assert G.shape[1] == Phi_src.shape[0], "G columns must match Phi rows"
    print(f"[dict] Phi shape={Phi_src.shape}")

    # tiny sanity ping on first col
    c0 = Phi_src[:, 0]
    print(f"[dict] Phi[:,0] mu={float(np.mean(c0)):.3g} sd={float(np.std(c0)):.3g}")

    # dictionary
    D = G @ Phi_src  # (M x K)
    print(f"[dict] D shape={D.shape}")

    # save Phi
    phi_path = os.path.join(out_root, f"fsaverage_phi_sym_K{args.K}_{args.spacing}.npz")
    np.savez_compressed(
        phi_path,
        Phi=Phi_src,
        evals_lh=evalL,
        evals_rh=evalR,
        spacing=np.array([args.spacing], dtype=object),
        K=np.array([args.K], dtype=int),
        combine=np.array(["sym"], dtype=object),
    )
    print("Saved Phi ->", phi_path)

    # save D
    d_path = os.path.join(out_root, f"fsaverage_D_sym_K{args.K}_M{M}_{args.spacing}.npz")
    np.savez_compressed(
        d_path,
        D=D,
        channels=np.array(canonical, dtype=object),
        spacing=np.array([args.spacing], dtype=object),
        K=np.array([args.K], dtype=int),
        subject=np.array(["fsaverage"], dtype=object),
        combine=np.array(["sym"], dtype=object),
    )
    print("Saved D ->", d_path, "\nDone.")

if __name__ == "__main__":
    main()
