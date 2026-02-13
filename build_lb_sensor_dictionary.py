#!/usr/bin/env python3
"""
build_lb_sensor_dictionary.py

Build the forward-projected cortical-eigenmode EEG sensor dictionary on fsaverage.

Outputs:
  - D_native:   D[:,k] = L @ phi_k  (native LB index order and native forward-model scaling)
  - D_unitnorm: column-wise unit-norm version of D_native:
                 D_unitnorm[:,k] = D_native[:,k] / ||D_native[:,k]||_2

Also saves:
  - col_norms[k] = ||D_native[:,k]||_2   (L2 norm = sqrt(sum of squares), NOT squared)
  - evals_lh[k], evals_rh[k]             (per-hemi eigenvalues; ordering info)
  - metadata.json                    

Requires:
  - compute_phi_lapy.py providing get_phi_fsaverage(...)
"""

import os
import sys
import json
import argparse
import platform
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import mne

from compute_phi_lapy import get_phi_fsaverage


# helpers ------------------------

def ensure_dir(p: str) -> str:
    p = os.path.expanduser(p)
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def read_channels_file(path: str) -> list[str]:
    with open(os.path.expanduser(path), "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    # de-duplicate, preserve order
    out, seen = [], set()
    for ch in names:
        if ch not in seen:
            out.append(ch); seen.add(ch)
    return out

def set_subjects_dir(subjects_dir: str | None, verbose: bool) -> str:
    """
    Use a safe writable SUBJECTS_DIR by default, fetch fsaverage, return subjects_dir.
    """
    if subjects_dir is None:
        subjects_dir = os.path.expanduser("~/mne_data/MNE-fsaverage-data")
    subjects_dir = ensure_dir(subjects_dir)
    mne.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)

    fs_path = mne.datasets.fetch_fsaverage(verbose=verbose)
    # fetch_fsaverage returns either .../fsaverage or its parent; normalize to parent
    fs_dir = fs_path if os.path.basename(fs_path) == "fsaverage" else os.path.join(fs_path, "fsaverage")
    if verbose:
        print("[env] SUBJECTS_DIR =", mne.get_config("SUBJECTS_DIR"))
        print("[env] fsaverage dir =", fs_dir)
    return os.path.dirname(fs_dir) if os.path.basename(fs_dir) == "fsaverage" else fs_dir

def stable_rank(A: np.ndarray, rtol: float = 1e-12) -> int:
    if A.size == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    if s.size == 0:
        return 0
    return int(np.sum(s >= (rtol * s[0])))

def lr_abs_energy_ratio(D: np.ndarray, ch_pos: dict, ch_names: list[str], n_show: int = 10) -> dict:
    xyz = np.array([ch_pos.get(ch, [np.nan, np.nan, np.nan]) for ch in ch_names], float)
    left = np.where(np.isfinite(xyz[:, 0]) & (xyz[:, 0] < 0))[0]
    right = np.where(np.isfinite(xyz[:, 0]) & (xyz[:, 0] > 0))[0]
    out = {"n_left": int(left.size), "n_right": int(right.size), "ratios_first": []}
    if left.size == 0 or right.size == 0:
        return out

    K = D.shape[1]
    for k in range(min(n_show, K)):
        num = float(np.nansum(np.abs(D[left, k])))
        den = float(np.nansum(np.abs(D[right, k])) + 1e-12)
        out["ratios_first"].append(num / den)

    ratios_all = []
    for k in range(K):
        num = float(np.nansum(np.abs(D[left, k])))
        den = float(np.nansum(np.abs(D[right, k])) + 1e-12)
        ratios_all.append(num / den)
    ratios_all = np.asarray(ratios_all, float)
    out["ratio_median"] = float(np.nanmedian(ratios_all))
    out["ratio_iqr"] = float(np.nanpercentile(ratios_all, 75) - np.nanpercentile(ratios_all, 25))
    return out


# main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Build forward-projected LB sensor dictionary (native + unitnorm).")
    ap.add_argument("--outdir", required=True, help="Output directory to write artifacts.")
    ap.add_argument("--channels-file", default=None, help="Text file with one channel name per line (recommended).")
    ap.add_argument("--channels", default=None, help="Comma-separated channel names (alternative to --channels-file).")
    ap.add_argument("--montage", default="standard_1005", help="MNE montage name (default: standard_1005).")

    ap.add_argument("--subject", default="fsaverage", help="FreeSurfer subject (default: fsaverage).")
    ap.add_argument("--subjects-dir", default=None, help="FreeSurfer SUBJECTS_DIR (default: ~/mne_data/MNE-fsaverage-data).")
    ap.add_argument("--trans", default="fsaverage", help="MNE transform for forward solution (default: fsaverage).")

    ap.add_argument("--K", type=int, default=60, help="Number of modes per hemisphere (default: 60).")
    ap.add_argument("--combine", default="sym", choices=["sym", "block", "sym+antisym"],
                    help="Mode combination in compute_phi_lapy.get_phi_fsaverage (default: sym).")
    ap.add_argument("--spacing", default="ico4", choices=["ico3", "ico4", "ico5", "oct6"],
                    help="MNE source-space spacing (default: ico4).")

    ap.add_argument("--bem-ico", type=int, default=4, help="BEM ico subdivision (default: 4).")
    ap.add_argument("--conductivity", nargs=3, type=float, default=[0.3, 0.006, 0.3],
                    metavar=("SCALP", "SKULL", "BRAIN"),
                    help="Conductivities (S/m) scalp skull brain (default: 0.3 0.006 0.3).")
    ap.add_argument("--mindist", type=float, default=5.0, help="Minimum distance from inner skull in mm (default: 5).")
    ap.add_argument("--n-jobs", type=int, default=1, help="n_jobs for forward computation (default: 1).")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    ap.add_argument("--no-diagnostics", action="store_true", help="Skip diagnostics.")
    ap.add_argument("--verbose", action="store_true", help="Verbose MNE output.")
    args = ap.parse_args()

    verbose = bool(args.verbose)

    # output dirs
    outdir = ensure_dir(args.outdir)
    diag_dir = ensure_dir(os.path.join(outdir, "diagnostics"))

    # channels
    if args.channels_file is None and args.channels is None:
        raise ValueError("Provide either --channels-file or --channels.")
    if args.channels_file is not None:
        channels = read_channels_file(args.channels_file)
    else:
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    M = len(channels)
    if M < 4:
        raise ValueError("Channel list too short; expected EEG channels.")

    # subjects_dir + fsaverage
    subjects_dir = set_subjects_dir(args.subjects_dir, verbose=verbose)

    # Info + montage
    info = mne.create_info(ch_names=channels, sfreq=250.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage(args.montage)
    info.set_montage(montage, on_missing="ignore", verbose=False)

    ch_pos = info.get_montage().get_positions().get("ch_pos", {})
    missing_pos = [ch for ch in channels if ch not in ch_pos]
    if missing_pos:
        print("[warn] Missing sensor positions for:", missing_pos)

    # source space
    src = mne.setup_source_space(
        subject=args.subject, spacing=args.spacing, add_dist=False,
        subjects_dir=subjects_dir, verbose=verbose
    )

    # BEM
    conductivity = (float(args.conductivity[0]), float(args.conductivity[1]), float(args.conductivity[2]))
    model = mne.make_bem_model(
        subject=args.subject, ico=args.bem_ico, conductivity=conductivity,
        subjects_dir=subjects_dir, verbose=verbose
    )
    bem = mne.make_bem_solution(model, verbose=verbose)

    # forward solution (EEG)
    fwd = mne.make_forward_solution(
        info=info, trans=args.trans, src=src, bem=bem,
        meg=False, eeg=True, mindist=float(args.mindist),
        n_jobs=int(args.n_jobs), verbose=verbose
    )
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=verbose)
    L = fwd_fixed["sol"]["data"]  # (M x Vsrc)
    Vsrc = int(L.shape[1])

    # LB modes on same src
    Phi_src, evals_lh, evals_rh, _ = get_phi_fsaverage(
        K=int(args.K), spacing=args.spacing, subjects_dir=subjects_dir,
        combine=args.combine, src=src, verbose=verbose
    )
    if int(Phi_src.shape[0]) != Vsrc:
        raise RuntimeError(f"Vertex mismatch: leadfield Vsrc={Vsrc} but Phi rows={Phi_src.shape[0]}")
    K_out = int(Phi_src.shape[1])

    # dictionary
    D_native = L @ Phi_src  # (M x K_out)

    # L2 norms (NOT squared): sqrt(sum of squares)
    col_norms = np.linalg.norm(D_native, axis=0, ord=2)
    safe_norms = np.where(col_norms > 0, col_norms, 1.0)
    D_unit = D_native / safe_norms[None, :]

    tag = f"{args.subject}_{args.spacing}_K{K_out}_M{M}_bemico{args.bem_ico}"
    native_path = os.path.join(outdir, f"D_native_{tag}.npz")
    unit_path = os.path.join(outdir, f"D_unitnorm_{tag}.npz")
    meta_path = os.path.join(outdir, "metadata.json")
    diag_path = os.path.join(diag_dir, "diagnostics.json")

    if (os.path.exists(native_path) or os.path.exists(unit_path) or os.path.exists(meta_path)) and not args.overwrite:
        raise FileExistsError("Outputs already exist. Use --overwrite to replace them.")

    # save npz artifacts
    common_npz = dict(
        channels=np.array(channels, dtype=object),
        col_norms=col_norms.astype(np.float64),
        evals_lh=np.asarray(evals_lh, dtype=np.float64),
        evals_rh=np.asarray(evals_rh, dtype=np.float64),
        subject=np.array([args.subject], dtype=object),
        spacing=np.array([args.spacing], dtype=object),
        combine=np.array([args.combine], dtype=object),
        conductivity=np.array(conductivity, dtype=np.float64),
        bem_ico=np.array([args.bem_ico], dtype=int),
        mindist=np.array([float(args.mindist)], dtype=np.float64),
        Vsrc=np.array([Vsrc], dtype=int),
        K=np.array([K_out], dtype=int),
    )
    np.savez_compressed(native_path, D=D_native.astype(np.float64), **common_npz)
    np.savez_compressed(unit_path, D=D_unit.astype(np.float64), **common_npz)

    # diagnostics
    diag = {
        "shapes": {"M": int(M), "Vsrc": int(Vsrc), "K": int(K_out)},
        "rank_D_native": int(stable_rank(D_native)),
        "rank_D_unitnorm": int(stable_rank(D_unit)),
        "col_norms_min": float(np.min(col_norms)),
        "col_norms_median": float(np.median(col_norms)),
        "col_norms_max": float(np.max(col_norms)),
    }
    if not args.no_diagnostics:
        diag["lr_abs_energy_ratio"] = lr_abs_energy_ratio(D_native, ch_pos, channels, n_show=10)
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2)

    # metadata (timezone-aware UTC timestamp)
    meta = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subject": args.subject,
        "subjects_dir": subjects_dir,
        "spacing": args.spacing,
        "combine": args.combine,
        "K_requested_per_hemi": int(args.K),
        "K_out": int(K_out),
        "M": int(M),
        "channels_file": os.path.expanduser(args.channels_file) if args.channels_file else None,
        "channels_arg": args.channels if args.channels else None,
        "montage": args.montage,
        "bem_ico": int(args.bem_ico),
        "conductivity": list(conductivity),
        "mindist_mm": float(args.mindist),
        "trans": args.trans,
        "forward_fixed_orientation": True,
        "artifacts": {
            "D_native_npz": os.path.basename(native_path),
            "D_unitnorm_npz": os.path.basename(unit_path),
            "diagnostics_json": "diagnostics/diagnostics.json" if (not args.no_diagnostics) else None,
        },
        "how_to_access": {
            "D_native": "np.load(D_native_npz, allow_pickle=True)['D']",
            "D_unitnorm": "np.load(D_unitnorm_npz, allow_pickle=True)['D']",
            "col_norms": "np.load(D_native_npz, allow_pickle=True)['col_norms']",
            "evals_lh": "np.load(D_native_npz, allow_pickle=True)['evals_lh']",
            "evals_rh": "np.load(D_native_npz, allow_pickle=True)['evals_rh']",
            "channels": "list(np.load(D_native_npz, allow_pickle=True)['channels'])",
        },
        "versions": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mne": mne.__version__,
        },
        "diagnostics_summary": diag,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[done] Wrote artifacts:")
    print("  ", native_path)
    print("  ", unit_path)
    print("  ", meta_path)
    if not args.no_diagnostics:
        print("  ", diag_path)

    print("\nAccessors:")
    print("  D_native   = np.load(D_native_npz, allow_pickle=True)['D']")
    print("  D_unitnorm = np.load(D_unitnorm_npz, allow_pickle=True)['D']")
    print("  col_norms  = np.load(D_native_npz, allow_pickle=True)['col_norms']  # L2 norms (not squared)")
    print("  evals_lh   = np.load(D_native_npz, allow_pickle=True)['evals_lh']")
    print("  evals_rh   = np.load(D_native_npz, allow_pickle=True)['evals_rh']")


if __name__ == "__main__":
    main()