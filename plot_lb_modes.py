#!/usr/bin/env python3
# plot_lb_modes.py — quick sensor topomaps for LB modes (e.g., 1,4,5,8..)

import os, argparse
import numpy as np
import mne
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict-npz", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--modes", default="6,14,18")
    ap.add_argument("--sfreq", type=float, default=250.0)
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    dpath = os.path.expanduser(args.dict_npz)
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    d = np.load(dpath, allow_pickle=True)
    D = d["D"]                      # (M x K)
    ch_names = list(d["channels"])
    K = int(D.shape[1])

    # parse modes
    modes = []
    for x in args.modes.split(","):
        x = x.strip()
        if not x: continue
        try:
            modes.append(int(x))
        except ValueError:
            print(f"[warn] bad mode '{x}' (skip)")
    if not modes:
        raise SystemExit("no valid modes to plot")

    print(f"[plot] dict={os.path.basename(dpath)}  M={D.shape[0]} K={K}  modes={modes}")

    # info + montage (... MNE can plot with info directly)
    info = mne.create_info(ch_names=ch_names, sfreq=args.sfreq, ch_types="eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="ignore")

    for k in modes:
        if not (1 <= k <= K):
            print(f"[skip] mode {k} not in 1..{K}")
            continue
        v = np.asarray(D[:, k-1], float)
        v /= (np.linalg.norm(v) + 1e-12)

        fig, ax = plt.subplots(figsize=(4, 4))
        mne.viz.plot_topomap(v, info, axes=ax, show=False, contours=6, cmap="RdBu_r")
        ax.set_title(f"LB mode {k}")
        out = os.path.join(outdir, f"LB_mode_{k:02d}.png")
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print("[ok] wrote", out)

if __name__ == "__main__":
    main()
