#!/usr/bin/env python3
# fig_checkpoints.py — mean R^2(K) with 95% CIs at a few K’s, split by condition

import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def build_style_maps():
    # colors/lines/markers (keep it simple)
    colors = {"LB":"#1f77b4","SPH":"#ff7f0e","PCA":"#2ca02c","ICA":"#9467bd"}
    linestyles = {"LB":"-","SPH":"--","PCA":":","ICA":"-."}
    markers = {"LB":"o","SPH":"s","PCA":"^","ICA":"D"}
    labels = {"LB":"LB","SPH":"SPH","PCA":"PCA","ICA":"ICA"}
    return colors, linestyles, markers, labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",
                   default=os.path.join(os.path.expanduser("~"),
                                        "lemon_work","derivatives","Merged_Report",
                                        "group_curves_bootstrap.csv"),
                   help="Path to group_curves_bootstrap.csv")
    p.add_argument("--subset", default="ALL59")
    p.add_argument("--methods", default="LB,SPH,PCA,ICA")
    p.add_argument("--conds", default="EC,EO")
    p.add_argument("--ks", default="5,10,20,30")
    p.add_argument("--xspacing", choices=["numeric","categorical"], default="numeric")
    p.add_argument("--hspace", type=float, default=0.34)
    p.add_argument("--legend-fontsize", type=float, default=12.0)
    p.add_argument("--out-png", default=None)
    p.add_argument("--out-svg", default=None)
    args = p.parse_args()

    csv_path = Path(os.path.expanduser(args.csv))
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    subset = args.subset.strip()
    alias = {"PK":"PCA","SP":"SPH","IC":"ICA"}
    methods = [alias.get(m.strip(), m.strip()) for m in args.methods.split(",") if m.strip()]
    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    k_ticks = [int(k) for k in args.ks.split(",") if k.strip()]

    df = pd.read_csv(csv_path)
    df = df[(df["subset"] == subset) & (df["method"].isin(methods)) & (df["K"].isin(k_ticks))]

    out_png = Path(args.out_png).expanduser() if args.out_png else (csv_path.parent / f"fig_{subset}_checkpoints.png")
    out_svg = Path(args.out_svg).expanduser() if args.out_svg else (csv_path.parent / f"fig_{subset}_checkpoints.svg")

    colors, linestyles, markers, labels = build_style_maps()

    # drop methods we don't recognize (quietly warn)
    known = set(colors.keys())
    bad = [m for m in methods if m not in known]
    if bad:
        print(f"[warn] unknown methods: {bad} (skip)")
        methods = [m for m in methods if m in known]

    # look
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "legend.fontsize": args.legend_fontsize,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(nrows=len(conds), ncols=1,
                             figsize=(8.6, 3.8*len(conds)), sharex=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    #  x positions..
    if args.xspacing == "numeric":
        x_positions = np.array(k_ticks, float)
        xticks = x_positions
    else:
        x_positions = np.arange(len(k_ticks), float)
        xticks = x_positions
    xticklabels = [str(k) for k in k_ticks]

    # quick ping
    print(f"[plot] subset={subset} methods={methods} conds={conds} K={k_ticks} rows={len(df)}")

    for ax, cond in zip(axes, conds):
        # warn if some method lacks K entries..
        for m in methods:
            have = set(df[(df["condition"]==cond) & (df["method"]==m)]["K"].tolist())
            miss = sorted(set(k_ticks) - have)
            if miss:
                print(f"[warn] missing K {miss} for {m}/{cond}")

        for m in methods:
            sub = (df[(df["condition"]==cond) & (df["method"]==m)]
                   .sort_values("K").set_index("K").reindex(k_ticks).sort_index())
            if sub.empty or sub["R2_mean"].isna().all():
                continue

            y = sub["R2_mean"].values
            lo = sub["R2_lo"].values
            hi = sub["R2_hi"].values

            ax.plot(x_positions, y,
                    ls=linestyles[m], color=colors[m], lw=2.2,
                    marker=markers[m], ms=6.0, mfc="white", mew=1.2, mec=colors[m],
                    label=labels[m], zorder=3)
            yerr = np.vstack([y - lo, hi - y])
            ax.errorbar(x_positions, y, yerr=yerr, fmt="none",
                        ecolor=colors[m], elinewidth=1.2, capsize=5, zorder=2)

        ax.set_title(cond, pad=6)
        ax.set_ylabel(r"Mean $R^2(K)$")
        ax.set_ylim(0.0, 1.05)
        if args.xspacing == "numeric":
            ax.set_xlim(x_positions.min()-1, x_positions.max()+1)
        else:
            ax.set_xlim(x_positions.min()-0.5, x_positions.max()+0.5)
        ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.set_xlabel("K (number of modes)")

    #  shared legend.
    handles = [Line2D([0],[0], color=colors[m], lw=2.2, ls=linestyles[m],
                      marker=markers[m], markersize=6.5, mfc="white", mew=1.2,
                      label=labels[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", ncol=min(4, len(handles)),
               frameon=False, borderaxespad=0.6, handlelength=3.0, columnspacing=1.2)

    fig.tight_layout(rect=[0.05, 0.10, 1.0, 0.92])
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.10, right=0.98, hspace=args.hspace)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300)
    if out_svg.suffix.lower() == ".svg":
        fig.savefig(out_svg)

    print(f"[ok] wrote:\n  {out_png}\n  {out_svg}")

if __name__ == "__main__":
    main()
