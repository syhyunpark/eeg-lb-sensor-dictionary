#!/usr/bin/env python3
# make_kXX_boxplots.py — boxplots for ... K70/K75/K90 (and /M)

import argparse, glob, os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def _set_rcparams():
    # plain readable defaults
    matplotlib.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "boxplot.flierprops.marker": "o",
        "boxplot.flierprops.markersize": 2.5,
    })

def _load_lb_summaries(derivatives_root: str) -> pd.DataFrame:
    pat = os.path.join(os.path.expanduser(derivatives_root),
                       "LB_fits_OLS", "*", "lb_ols_summary__*.csv")
    paths = glob.glob(pat)
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["method"] = "LB"
            df["subset"] = os.path.basename(os.path.dirname(p))
            frames.append(df)
        except Exception as e:
            print(f"[warn] skip LB {p}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=[
        "subject","condition","subset","method","m_used",
        "K70","K70_norm","attained_70",
        "K75","K75_norm","attained_75",
        "K90","K90_norm","attained_90"
    ])

def _load_baseline_summaries(derivatives_root: str) -> pd.DataFrame:
    pat = os.path.join(os.path.expanduser(derivatives_root),
                       "Baselines_OLS", "*", "baselines_ols_summary__*.csv")
    paths = glob.glob(pat)
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["subset"] = os.path.basename(os.path.dirname(p))
            frames.append(df)
        except Exception as e:
            print(f"[warn] skip BASE {p}: {e}")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=[
        "subject","condition","subset","method","m_used",
        "K70","K70_norm","attained_70",
        "K75","K75_norm","attained_75",
        "K90","K90_norm","attained_90"
    ])

def _prepare_long(df_all: pd.DataFrame,
                  keep_methods=("LB","SPH","PCA","ICA"),
                  keep_conditions=("EC","EO")) -> pd.DataFrame:
    # filter + coerce numerics we need
    out = df_all.copy()
    out["condition"] = out["condition"].astype(str).str.upper()
    out = out[out["condition"].isin(keep_conditions)]
    out["method"] = out["method"].astype(str).str.upper().replace({"SPHARM": "SPH"})
    out = out[out["method"].isin(keep_methods)]
    num_cols = ["m_used",
                "K70","K70_norm","attained_70",
                "K75","K75_norm","attained_75",
                "K90","K90_norm","attained_90"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = np.nan
    return out

def _metric_to_fields(metric: str):
    # metric -> (value, attained flag, y label, ylim if normalized)
    m = metric.upper()
    if m == "K70":       return "K70", "attained_70",  "K₇₀ (modes)", None
    if m == "K70_NORM":  return "K70_norm", "attained_70", "K₇₀ / M", (0.0, 1.05)
    if m == "K75":       return "K75", "attained_75", "K₇₅ (modes)", None
    if m == "K75_NORM":  return "K75_norm", "attained_75", "K₇₅ / M", (0.0, 1.05)
    if m == "K90":       return "K90", "attained_90", "K₉₀ (modes)", None
    if m == "K90_NORM":  return "K90_norm", "attained_90", "K₉₀ / M", (0.0, 1.05)
    raise ValueError(f"Unknown metric: {metric}")

def _collect_arrays_for_boxplot(df: pd.DataFrame,
                                subset: str,
                                methods=("LB","SPH","PCA","ICA"),
                                conditions=("EC","EO"),
                                metric: str = "K90"):
    # build arrays/labels/counts for one subset+metric
    val_col, att_col, _, _ = _metric_to_fields(metric)
    data_arrays, labels, counts = [], [], []
    sub_df = df[df["subset"] == subset]

    for cond in conditions:
        for meth in methods:
            g = sub_df[(sub_df["condition"] == cond) & (sub_df["method"] == meth)]
            if g.empty: continue
            g_att = g[g[att_col] == 1]
            vals = g_att[val_col].dropna().to_numpy(float)
            data_arrays.append(vals)   # may be empty
            labels.append(f"{cond}-{meth}")
            counts.append((int(vals.size), int(g.shape[0])))
    return data_arrays, labels, counts

def _draw_boxplot(data_arrays, labels, counts, title, ylabel, outfile_png, outfile_pdf,
                  dpi=300, figsize=(10, 6), show_title=True, annotate=False, grid=False, ylims=None):
    # single figure (means shown as red lines)
    plt.figure(figsize=figsize, dpi=dpi)
    idx = [i for i, arr in enumerate(data_arrays) if arr.size > 0]
    if not idx:
        print(f"[warn] no data to plot: {title}")
        plt.close()
        return

    data_nonempty = [data_arrays[i] for i in idx]
    labels_nonempty = [labels[i] for i in idx]
    counts_nonempty = [counts[i] for i in idx]

    plt.boxplot(
        data_nonempty,
        patch_artist=False,
        labels=labels_nonempty,
        showmeans=True,
        meanline=True,
        meanprops={"color": "red", "linewidth": 2.0},
        medianprops={"linewidth": 0.0},
        whis=1.5,
    )
    if grid:
        plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    plt.ylabel(ylabel)
    if show_title and title:
        plt.title(title)
    plt.xticks(rotation=30, ha="right")

    if ylims is not None:
        plt.ylim(*ylims)

    if annotate:
        ymax = plt.gca().get_ylim()[1]
        ytxt = ymax - 0.02 * ymax
        for i, (na, nt) in enumerate(counts_nonempty, start=1):
            plt.text(i, ytxt, f"n={na}/{nt}", ha="center", va="top", fontsize=8)

    plt.tight_layout()
    plt.savefig(outfile_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(outfile_pdf, bbox_inches="tight")
    plt.close()

def main():
    _set_rcparams()

    ap = argparse.ArgumentParser(description="K70/K75/K90 boxplots (and normalized) from per-subject summaries")
    ap.add_argument("--derivatives", required=True, help="~/lemon_work/derivatives")
    ap.add_argument("--outdir", default="./figs")
    ap.add_argument("--subsets", default="ALL59,canonical_32,canonical_19")
    ap.add_argument("--methods", default="LB,SPH,PCA,ICA")
    ap.add_argument("--conditions", default="EC,EO")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--figsize", default="10,6")
    ap.add_argument("--no-title", action="store_true")
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--grid", action="store_true")
    args = ap.parse_args()

    derivatives_root = os.path.expanduser(args.derivatives)
    outdir = os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    methods = tuple(m.strip().upper() for m in args.methods.split(",") if m.strip())
    conditions = tuple(c.strip().upper() for c in args.conditions.split(",") if c.strip())
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    w, h = (float(x) for x in args.figsize.split(","))

    lb = _load_lb_summaries(derivatives_root)
    base = _load_baseline_summaries(derivatives_root)
    if lb.empty and base.empty:
        raise FileNotFoundError("No summaries under LB_fits_OLS/ or Baselines_OLS/.")

    df_all = pd.concat([lb, base], ignore_index=True, sort=False)
    df = _prepare_long(df_all, keep_methods=methods, keep_conditions=conditions)

    print(f"[boxplots] N={len(df)} rows | subsets={subsets} | methods={methods} | conds={conditions}")

    metrics = [
        ("K70",      "K₇₀ (modes)"),
        ("K70_norm", "K₇₀ / M"),
        ("K75",      "K₇₅ (modes)"),
        ("K75_norm", "K₇₅ / M"),
        ("K90",      "K₉₀ (modes)"),
        ("K90_norm", "K₉₀ / M"),
    ]

    for subset in subsets:
        for metric, ylabel in metrics:
            data_arrays, labels, counts = _collect_arrays_for_boxplot(
                df, subset, methods=methods, conditions=conditions, metric=metric
            )

            _, _, _, ylim = _metric_to_fields(metric)
            if metric.endswith("_NORM"):
                data_arrays = [np.clip(arr, 0.0, 1.0) for arr in data_arrays]

            title = f"{subset}: {ylabel} by Condition × Method"
            base_name = f"{subset}__{metric.replace('_','')}_boxplot"
            png_path = os.path.join(outdir, base_name + ".png")
            pdf_path = os.path.join(outdir, base_name + ".pdf")

            _draw_boxplot(
                data_arrays, labels, counts,
                title=None if args.no_title else title,
                ylabel=ylabel,
                outfile_png=png_path,
                outfile_pdf=pdf_path,
                dpi=args.dpi, figsize=(w, h),
                show_title=not args.no_title,
                annotate=args.annotate,
                grid=args.grid,
                ylims=ylim
            )

    print("[done] figures ->", outdir)

if __name__ == "__main__":
    main()
