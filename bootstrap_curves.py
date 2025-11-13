#!/usr/bin/env python3
# bootstrap_curves.py — merge LB + Baselines; bootstrap CIs for R2(K) and K70/K75/K90

import os, glob, argparse
import numpy as np
import pandas as pd

def bootstrap_mean_ci(vals, B=2000, alpha=0.05, rng=None):
    # mean CI via simple bootstrap
    vals = np.asarray(vals, float)
    if rng is None:
        rng = np.random.default_rng(0)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return np.nan, (np.nan, np.nan)
    n = vals.shape[0]
    boot = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boot.append(np.nanmean(vals[idx]))
    boot = np.asarray(boot, float)
    return float(np.nanmean(vals)), (np.nanpercentile(boot, 100*alpha/2),
                                     np.nanpercentile(boot, 100*(1-alpha/2)))

def bootstrap_prop_ci(flags, B=2000, alpha=0.05, rng=None):
    # proportion CI (bootstrap on 0/1)
    flags = np.asarray(flags, float)
    if rng is None:
        rng = np.random.default_rng(0)
    if flags.size == 0 or np.all(np.isnan(flags)):
        return np.nan, (np.nan, np.nan)
    flags = np.nan_to_num(flags, nan=0.0)
    n = flags.shape[0]
    p = float(flags.mean())
    boot = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boot.append(np.nanmean(flags[idx]))
    boot = np.asarray(boot, float)
    return p, (np.nanpercentile(boot, 100*alpha/2),
               np.nanpercentile(boot, 100*(1-alpha/2)))

def main():
    ap = argparse.ArgumentParser(description="Merge LB+Baselines; bootstrap CIs for R²(K) and K70/K75/K90")
    ap.add_argument("--root", required=True, help="~/lemon_work/derivatives")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=2000)
    args = ap.parse_args()

    derivatives = os.path.expanduser(args.root)
    outdir = os.path.join(derivatives, "Merged_Report") if args.outdir is None else os.path.expanduser(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    # load per-subject summaries
    lb_paths = glob.glob(os.path.join(derivatives, "LB_fits_OLS", "*", "lb_ols_summary__*.csv"))
    lb_frames = []
    for p in lb_paths:
        try:
            df = pd.read_csv(p)
            df["method"] = "LB"
            lb_frames.append(df)
        except Exception as e:
            print(f"[warn] skip LB {p}: {e}")
    lb_df = pd.concat(lb_frames, ignore_index=True) if lb_frames else pd.DataFrame()

    base_paths = glob.glob(os.path.join(derivatives, "Baselines_OLS", "*", "baselines_ols_summary__*.csv"))
    base_df = pd.concat([pd.read_csv(p) for p in base_paths], ignore_index=True) if base_paths else pd.DataFrame()

    # normalize minimal schema; merge
    meta_cols = ["subject","condition","subset","method","m_used"]
    for c in meta_cols:
        if c not in lb_df.columns:   lb_df[c] = np.nan
        if c not in base_df.columns: base_df[c] = np.nan

    all_df = pd.concat([lb_df, base_df], ignore_index=True, sort=False)
    all_df["subject"] = all_df["subject"].astype(str)
    raw_csv = os.path.join(outdir, "all_methods_summary_raw.csv")
    all_df.to_csv(raw_csv, index=False)

    rng = np.random.default_rng(0)

    # curves bootstrap for detected R2_K*
    r2_cols = [c for c in all_df.columns if c.startswith("R2_K")]
    def parse_k(col):
        try: return int(col.split("R2_K")[1])
        except: return None
    k_map = {c: parse_k(c) for c in r2_cols}
    r2_cols = [c for c,k in k_map.items() if k is not None]

    curve_rows = []
    if r2_cols:
        for (subset, condition, method), g in all_df.groupby(["subset","condition","method"]):
            if g.empty: continue
            ks_present = sorted({k_map[c] for c in r2_cols if c in g.columns and g[c].notna().sum()>0})
            for K in ks_present:
                col = f"R2_K{K}"
                if col not in g.columns: continue
                vals = pd.to_numeric(g[col], errors="coerce").values
                if np.isfinite(vals).sum() == 0: continue
                mean, (lo, hi) = bootstrap_mean_ci(vals, B=args.B, alpha=args.alpha, rng=rng)
                curve_rows.append({"subset":subset, "condition":condition, "method":method,
                                   "K":int(K), "R2_mean":mean, "R2_lo":lo, "R2_hi":hi})
    curves_csv = os.path.join(outdir, "group_curves_bootstrap.csv")
    pd.DataFrame(curve_rows).sort_values(["subset","condition","method","K"]).to_csv(curves_csv, index=False)

    # thresholds: K70/K75/K90 (attainment + attainer means + censored)
    thr_specs = [
        ("70", "K70", "K70_norm", "K70_cens", "K70_norm_cens", "attained_70"),
        ("75", "K75", "K75_norm", "K75_cens", "K75_norm_cens", "attained_75"),
        ("90", "K90", "K90_norm", "K90_cens", None,             "attained_90"),  # no K90n_cens
    ]

    stat_rows = []
    for (subset, condition, method), g in all_df.groupby(["subset","condition","method"]):
        if g.empty: continue
        row = {"subset":subset, "condition":condition, "method":method, "N":int(g.shape[0])}

        for label, k_col, kn_col, kc_col, knc_col, att_col in thr_specs:
            # attainment (prop)
            if att_col in g.columns:
                flags = pd.to_numeric(g[att_col], errors="coerce").fillna(0).values
                p, (plo, phi) = bootstrap_prop_ci(flags, B=args.B, alpha=args.alpha, rng=rng)
            else:
                p, plo, phi = (np.nan, np.nan, np.nan)
            row[f"attain_{label}"]    = p
            row[f"attain_{label}_lo"] = plo
            row[f"attain_{label}_hi"] = phi

            # means among attainers (abs + normalized)
            if (att_col in g.columns) and (g[att_col].fillna(0).astype(int) == 1).any():
                g_att = g[g[att_col].fillna(0).astype(int) == 1]
                if k_col in g_att.columns:
                    k_mean,(k_lo,k_hi) = bootstrap_mean_ci(pd.to_numeric(g_att[k_col], errors="coerce").values,
                                                           B=args.B, alpha=args.alpha, rng=rng)
                else:
                    k_mean,k_lo,k_hi = (np.nan,np.nan,np.nan)
                if kn_col in g_att.columns:
                    kn_mean,(kn_lo,kn_hi) = bootstrap_mean_ci(pd.to_numeric(g_att[kn_col], errors="coerce").values,
                                                              B=args.B, alpha=args.alpha, rng=rng)
                else:
                    kn_mean,kn_lo,kn_hi = (np.nan,np.nan,np.nan)
            else:
                k_mean,k_lo,k_hi = (np.nan,np.nan,np.nan)
                kn_mean,kn_lo,kn_hi = (np.nan,np.nan,np.nan)

            row[f"K{label}_att_mean"]  = k_mean
            row[f"K{label}_att_lo"]    = k_lo
            row[f"K{label}_att_hi"]    = k_hi
            row[f"K{label}n_att_mean"] = kn_mean
            row[f"K{label}n_att_lo"]   = kn_lo
            row[f"K{label}n_att_hi"]   = kn_hi

            # censored means (everyone)
            if kc_col in g.columns:
                kc_mean,(kc_lo,kc_hi) = bootstrap_mean_ci(pd.to_numeric(g[kc_col], errors="coerce").values,
                                                          B=args.B, alpha=args.alpha, rng=rng)
            else:
                kc_mean,kc_lo,kc_hi = (np.nan,np.nan,np.nan)
            row[f"K{label}_cens_mean"] = kc_mean
            row[f"K{label}_cens_lo"]   = kc_lo
            row[f"K{label}_cens_hi"]   = kc_hi

            if (knc_col is not None) and (knc_col in g.columns):
                knc_mean,(knc_lo,knc_hi) = bootstrap_mean_ci(pd.to_numeric(g[knc_col], errors="coerce").values,
                                                             B=args.B, alpha=args.alpha, rng=rng)
                row[f"K{label}n_cens_mean"] = knc_mean
                row[f"K{label}n_cens_lo"]   = knc_lo
                row[f"K{label}n_cens_hi"]   = knc_hi

        stat_rows.append(row)

    stats_csv = os.path.join(outdir, "thresholds_bootstrap.csv")
    pd.DataFrame(stat_rows).sort_values(["subset","condition","method"]).to_csv(stats_csv, index=False)

    print("[Merged] Wrote:")
    print(" ", raw_csv)
    print(" ", curves_csv)
    print(" ", stats_csv)

if __name__ == "__main__":
    main()
