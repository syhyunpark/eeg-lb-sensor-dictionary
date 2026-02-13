#!/usr/bin/env python3
# plot_lb_modes_cortex.py — fsaverage cortical LB modes (matplotlib backend, headless-safe)
# Fixes:
#   - true diverging display with explicit pos_lims/neg_lims (avoids "transparent negative")
#   - white near zero (threshold set to 0)
#   - colorbar ON by default (use --no-colorbar to disable)

import os
import argparse
import numpy as np
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
from PIL import Image

VALID_VIEWS = ("cau","dor","fro","lat","med","par","ros","ven")


def get_subjects_dir(maybe_dir: str | None):
    if maybe_dir:
        p = os.path.expanduser(maybe_dir)
        if os.path.isdir(p):
            return p
    fs_path = mne.datasets.fetch_fsaverage(verbose=True)
    return os.path.dirname(fs_path) if os.path.basename(fs_path) == "fsaverage" else fs_path


def _diverging_clim(v, q=95.0, eps=1e-12):
    """
    MNE diverging clim using pos_lims/neg_lims.
    Set pos_lims[0]=0 and neg_lims[2]=0 so values near 0 are WHITE (not transparent).
    """
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        a = 1.0
    else:
        a = np.percentile(np.abs(v), q)
        if not np.isfinite(a) or a < eps:
            a = np.max(np.abs(v)) if v.size else 1.0
        if not np.isfinite(a) or a < eps:
            a = 1.0

    mid = 0.5 * a
    # explicit diverging limits
    return dict(
        kind="value",
        pos_lims=[0.0, mid, a],
        neg_lims=[-a, -mid, 0.0],
    )


def plot_one_view(stc_twohem, subjects_dir, hemi, *, surface, smoothing, view,
                  cmap, out_png, qlim=95.0, colorbar=True):
    v = stc_twohem.data.ravel()
    clim = _diverging_clim(v, q=qlim)

    fig = stc_twohem.plot(
        subjects_dir=subjects_dir,
        hemi=hemi,
        surface=surface,
        time_viewer=False,
        smoothing_steps=smoothing,
        views=view,
        colormap=cmap,
        clim=clim,
        colorbar=colorbar,
        backend="matplotlib",
        size=(800, 400),
    )
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def stitch_horizontal(png_paths, out_png):
    imgs = [Image.open(p).convert("RGBA") for p in png_paths if Path(p).exists()]
    if not imgs:
        return
    H = max(im.height for im in imgs)
    W = sum(im.width for im in imgs)
    canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(out_png)


def main():
    ap = argparse.ArgumentParser(description="Plot fsaverage cortical LB modes (matplotlib backend)")
    ap.add_argument("--phi-npz", required=True, help="fsaverage_phi_sym_K*.npz")
    ap.add_argument("--subjects-dir", default=None)
    ap.add_argument("--spacing", default=None)
    ap.add_argument("--modes", default="6,14,18")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--views", default="lat,med")
    ap.add_argument("--surface", default="inflated")
    # Use RdBu_r: blue negative, white ~ 0, red positive
    ap.add_argument("--cmap", default="RdBu_r")
    ap.add_argument("--smoothing", type=int, default=10)
    ap.add_argument("--qlim", type=float, default=95.0,
                    help="Percentile of |data| used for symmetric limits (default 95).")
    ap.add_argument("--no-colorbar", action="store_true",
                    help="Disable colorbar (default: on).")
    ap.add_argument("--keep-intermediate", action="store_true")
    args = ap.parse_args()

    d = np.load(os.path.expanduser(args.phi_npz), allow_pickle=True)
    Phi = d["Phi"]
    spacing = args.spacing or str(d.get("spacing", ["ico4"])[0])
    K = int(Phi.shape[1])
    modes = [int(x) for x in args.modes.split(",") if x.strip()]

    outdir = Path(os.path.expanduser(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    subjects_dir = get_subjects_dir(args.subjects_dir)
    src = mne.setup_source_space(
        subject="fsaverage", spacing=spacing, add_dist=False,
        subjects_dir=subjects_dir, verbose=False
    )
    lh_vertno, rh_vertno = src[0]["vertno"], src[1]["vertno"]
    Vlh, Vrh = len(lh_vertno), len(rh_vertno)
    assert Phi.shape[0] == (Vlh + Vrh), f"Phi rows ({Phi.shape[0]}) != src vertices ({Vlh+Vrh})"

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    for v in views:
        if v not in VALID_VIEWS:
            raise ValueError(f"Invalid view '{v}'. Allowed: {VALID_VIEWS}")

    colorbar = (not args.no_colorbar)

    print(f"[plot] spacing={spacing}, Vlh={Vlh}, Vrh={Vrh}, K={K}, modes={modes}, views={views}, cmap={args.cmap}, colorbar={colorbar}")

    for k in modes:
        if not (1 <= k <= K):
            print(f"[skip] k={k} out of 1..{K}")
            continue

        vec = Phi[:, k-1].astype(float)
        lh_data = vec[:Vlh][:, None]
        rh_data = vec[Vlh:][:, None]

        stc_lh_two = mne.SourceEstimate(
            data=np.vstack([lh_data, np.zeros((Vrh, 1), float)]),
            vertices=[lh_vertno, rh_vertno], tmin=0.0, tstep=1.0, subject="fsaverage"
        )
        stc_rh_two = mne.SourceEstimate(
            data=np.vstack([np.zeros((Vlh, 1), float), rh_data]),
            vertices=[lh_vertno, rh_vertno], tmin=0.0, tstep=1.0, subject="fsaverage"
        )

        with tempfile.TemporaryDirectory() as tmpd:
            tmpd = Path(tmpd)

            # LH
            lh_pngs = []
            for v in views:
                p = tmpd / f"mode{k:02d}_lh_{v}.png"
                plot_one_view(
                    stc_lh_two, subjects_dir, "lh",
                    surface=args.surface, smoothing=args.smoothing,
                    view=v, cmap=args.cmap, out_png=str(p),
                    qlim=args.qlim, colorbar=colorbar
                )
                lh_pngs.append(str(p))
            lh_out = outdir / f"LB_cortex_mode_{k:02d}_lh.png"
            stitch_horizontal(lh_pngs, str(lh_out))

            # RH
            rh_pngs = []
            for v in views:
                p = tmpd / f"mode{k:02d}_rh_{v}.png"
                plot_one_view(
                    stc_rh_two, subjects_dir, "rh",
                    surface=args.surface, smoothing=args.smoothing,
                    view=v, cmap=args.cmap, out_png=str(p),
                    qlim=args.qlim, colorbar=colorbar
                )
                rh_pngs.append(str(p))
            rh_out = outdir / f"LB_cortex_mode_{k:02d}_rh.png"
            stitch_horizontal(rh_pngs, str(rh_out))

            # final
            final = outdir / f"LB_cortex_mode_{k:02d}.png"
            stitch_horizontal([str(lh_out), str(rh_out)], str(final))
            print("[ok] wrote", final)

            if args.keep_intermediate:
                for p in lh_pngs + rh_pngs:
                    Path(p).replace(outdir / Path(p).name)


if __name__ == "__main__":
    main()