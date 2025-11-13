#!/usr/bin/env python3
# run_lb_fits.py   - LB OLS in native order; thresholds from full-K; degree-block R2; group bootstrap CIs

import os, json, argparse
import numpy as np
import pandas as pd

def zscore_rows(A, eps=1e-12):
    mu = A.mean(axis=1, keepdims=True); sd = A.std(axis=1, keepdims=True)
    return (A - mu) / (sd + eps)

def load_dictionary(path):
    d = np.load(os.path.expanduser(path), allow_pickle=True)
    D = d["D"]; channels = list(d["channels"])
    K_sym = int(d["K"][0]) if "K" in d else D.shape[1]
    return D, channels, K_sym

def load_manifest(path):
    return pd.read_csv(os.path.expanduser(path))

def r2_rows(Y, Yhat):
    num = np.sum((Y - Yhat)**2, axis=1); den = np.sum(Y**2, axis=1) + 1e-12
    return 1.0 - num/den

def qr_no_pivot(B):
    if B.size == 0: return np.empty((B.shape[0], 0))
    Q, R = np.linalg.qr(B, mode="reduced")
    if R.size == 0: return np.empty((B.shape[0], 0))
    d = np.abs(np.diag(R))
    keep = d > (1e-12 * (d.max() if d.size else 1.0))
    return Q[:, keep] if keep.size else np.empty((B.shape[0], 0))

def degree_block_Ks(Kcap):
    # first col indices for degrees: 1^2, 2^2, 3^2, ...
    Lmax = int(np.floor(np.sqrt(max(int(Kcap), 0)))) - 1
    if Lmax < 0: return []
    return [(L+1)**2 for L in range(0, Lmax+1)]

def first_K_at_threshold_full(r2_full, thr):
    arr = np.array(r2_full); idx = np.where(arr >= thr)[0]
    return (int(idx[0]+1) if idx.size > 0 else np.nan)

def bootstrap_mean_ci(vals, B=2000, alpha=0.05, seed=0):
    # simple nonparametric mean CI
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    n = vals.size
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = float(np.mean(vals[idx]))
    mean = float(np.mean(vals))
    lo, hi = np.percentile(boot, [100*alpha/2.0, 100*(1.0 - alpha/2.0)])
    return mean, float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser(description="LB OLS (native order; degree-block checkpoints; group bootstrap CIs)")
    ap.add_argument("--dict-npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", default=os.path.expanduser("~/lemon_work"))
    ap.add_argument("--use-bands", action="store_true")
    ap.add_argument("--conditions", default="EO,EC")
    ap.add_argument("--subset-file", default=None)
    # bootstrap opts
    ap.add_argument("--boot-B", type=int, default=2000)
    ap.add_argument("--boot-alpha", type=float, default=0.05)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    D, dict_channels, _ = load_dictionary(args.dict_npz)

    subset_keep, subset_label = None, "ALL59"
    if args.subset_file:
        subset_label = os.path.splitext(os.path.basename(args.subset_file))[0]
        with open(os.path.expanduser(args.subset_file)) as f:
            allow = set(ln.strip() for ln in f if ln.strip())
        subset_keep = np.array([ch in allow for ch in dict_channels], bool)

    out_root = os.path.join(os.path.expanduser(args.outdir), "derivatives", "LB_fits_OLS", subset_label)
    os.makedirs(out_root, exist_ok=True)

    conds = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    man = load_manifest(args.manifest)

    rows_wide, rows_long = [], []

    print(f"[LB] N={len(man)} subjects, subset={subset_label}, bands={bool(args.use_bands)}, conds={conds}")

    for _, row in man.iterrows():
        sid = str(row["subject"]).zfill(6) if str(row["subject"]).isdigit() else str(row["subject"])
        subj_npz = os.path.join(os.path.dirname(args.manifest), f"sub-{sid}", f"sub-{sid}_Y_tfr.npz")
        if not os.path.exists(subj_npz):
            print(f"  !! missing {subj_npz}")
            continue
        z = np.load(subj_npz, allow_pickle=True)

        canonical_subj = list(z["canonical_channels"])
        subj_mask_canon = z["subject_mask"].astype(bool)

        # dict-order mask
        subj_idx = {ch:i for i,ch in enumerate(canonical_subj)}
        subj_mask_dict = np.array([subj_mask_canon[subj_idx[ch]] if ch in subj_idx else False
                                   for ch in dict_channels], bool)
        keep = subj_mask_dict.copy()
        if subset_keep is not None:
            keep &= subset_keep

        dict_kept = [ch for j, ch in enumerate(dict_channels) if keep[j]]
        m_kept = len(dict_kept)
        if m_kept < 8:
            print(f"    !! sub-{sid} too few channels ({m_kept}) -> skip")
            continue

        # Y columns in dict_kept order
        subj_used = [canonical_subj[i] for i,f in enumerate(subj_mask_canon) if f]
        idxY = {ch:j for j, ch in enumerate(subj_used)}
        y_cols = [idxY[ch] for ch in dict_kept if ch in idxY]
        if len(y_cols) != m_kept:
            print(f"    !! sub-{sid} Y col mismatch ({len(y_cols)} vs {m_kept}) -> skip")
            continue

        Kcap = m_kept
        Ks_check = degree_block_Ks(Kcap)
        if not Ks_check:
            print(f"    !! sub-{sid} no degree-block Ks -> skip")
            continue

        B_full = D[keep, :]

        def fetch(cond):
            if args.use_bands:
                keyY, keyI = ("Y_eo_band","index_eo_band") if cond=="EO" else ("Y_ec_band","index_ec_band")
            else:
                keyY, keyI = ("Y_eo","index_eo") if cond=="EO" else ("Y_ec","index_ec")
            if keyY not in z:
                return None, None
            Y = z[keyY][:, y_cols]
            Y = zscore_rows(Y)
            bands = None
            if args.use_bands and keyI in z:
                bands = [str(b) for (_, b) in list(z[keyI])]
            return Y, bands

        out_json = {"subject": sid, "m_used": m_kept, "subset": subset_label,
                    "use_bands": bool(args.use_bands), "results": {}}

        for cond in conds:
            Y, _ = fetch(cond)
            if Y is None or Y.size == 0:
                continue

            # full R²(K) with native prefix order
            R2_full = []
            for k in range(1, Kcap+1):
                Qk = qr_no_pivot(B_full[:, :k])
                if Qk.size == 0:
                    R2_full.append(0.0)
                else:
                    Yhat = (Y @ Qk) @ Qk.T
                    R2_full.append(float(np.mean(r2_rows(Y, Yhat))))

            # thresholds from the full curve
            K70 = first_K_at_threshold_full(R2_full, 0.70)
            K75 = first_K_at_threshold_full(R2_full, 0.75)
            K90 = first_K_at_threshold_full(R2_full, 0.90)
            A70 = int(not np.isnan(K70)); A75 = int(not np.isnan(K75)); A90 = int(not np.isnan(K90))
            K70n = (K70 / m_kept) if A70 else np.nan
            K75n = (K75 / m_kept) if A75 else np.nan
            K90n = (K90 / m_kept) if A90 else np.nan
            # censored (last observed)
            K70c = Kcap; K75c = Kcap; K90c = Kcap
            K70nc = Kcap/m_kept; K75nc = Kcap/m_kept; K90nc = Kcap/m_kept

            # R² at degree checkpoints
            R2_atKs = [R2_full[k-1] for k in Ks_check]

            out_json["results"][cond] = {
                "Ks_check": Ks_check, "R2_at_checkpoints": [float(v) for v in R2_atKs],
                "R2_full_last": float(R2_full[-1]) if R2_full else 0.0,
                "K70": K70, "K70_norm": K70n, "K70_cens": K70c, "K70_norm_cens": K70nc, "attained_70": A70,
                "K75": K75, "K75_norm": K75n, "K75_cens": K75c, "K75_norm_cens": K75nc, "attained_75": A75,
                "K90": K90, "K90_norm": K90n, "K90_cens": K90c, "K90_norm_cens": K90nc, "attained_90": A90
            }

            wide = {
                "subject": sid, "condition": cond, "subset": subset_label, "m_used": m_kept,
                "K_last": Kcap, "R2_at_Klast": float(R2_full[-1]) if R2_full else 0.0,
                "K70": K70, "K70_norm": K70n, "K70_cens": K70c, "K70_norm_cens": K70nc, "attained_70": A70,
                "K75": K75, "K75_norm": K75n, "K75_cens": K75c, "K75_norm_cens": K75nc, "attained_75": A75,
                "K90": K90, "K90_norm": K90n, "K90_cens": K90c, "K90_norm_cens": K90nc, "attained_90": A90
            }
            for k_val, r2_val in zip(Ks_check, R2_atKs):
                wide[f"R2_K{k_val}"] = float(r2_val)
            rows_wide.append(wide)

            # long curve
            for k in range(1, Kcap+1):
                rows_long.append({"subject": sid, "condition": cond, "subset": subset_label,
                                  "m_used": m_kept, "K": k, "R2": float(R2_full[k-1])})
            for k_val, r2_val in zip(Ks_check, R2_atKs):
                rows_long.append({"subject": sid, "condition": cond, "subset": subset_label,
                                  "m_used": m_kept, "K": int(k_val), "R2_checkpoint": float(r2_val)})

        # save JSON per subject
        subj_dir = os.path.join(out_root, f"sub-{sid}")
        os.makedirs(subj_dir, exist_ok=True)
        with open(os.path.join(subj_dir, f"sub-{sid}_lb_ols__{subset_label}.json"), "w") as f:
            json.dump(out_json, f, indent=2)

    # per-subject CSVs
    long_csv = os.path.join(out_root, f"lb_ols_curves_long__{subset_label}.csv")
    wide_csv = os.path.join(out_root, f"lb_ols_summary__{subset_label}.csv")
    pd.DataFrame(rows_long).sort_values(["subset","condition","subject","K"]).to_csv(long_csv, index=False)
    pd.DataFrame(rows_wide).sort_values(["subset","condition","subject"]).to_csv(wide_csv, index=False)
    print("Wrote", long_csv)
    print("Wrote", wide_csv)

    # group-level checkpoints with bootstrap CIs
    try:
        df_long = pd.DataFrame(rows_long)
        df_r2 = df_long.loc[df_long["R2"].notna(), ["subset","condition","subject","K","R2"]]
        ext_Ks = [5,10,15,20,25,30,35,40,45,50]
        df_ext = df_r2[df_r2["K"].isin(ext_Ks)].copy()
        if not df_ext.empty:
            rows_grp = []
            for (subset, condition, K), g in df_ext.groupby(["subset","condition","K"]):
                vals = g["R2"].to_numpy(float)
                mean, lo, hi = bootstrap_mean_ci(vals, B=args.boot_B, alpha=args.boot_alpha, seed=args.boot_seed)
                rows_grp.append({
                    "subset": subset, "condition": condition, "K": int(K),
                    "R2_mean": float(mean), "R2_lo": float(lo), "R2_hi": float(hi),
                    "N": int(vals.size), "B": int(args.boot_B), "alpha": float(args.boot_alpha)
                })
            grp = pd.DataFrame(rows_grp).sort_values(["subset","condition","K"])
            ext_csv = os.path.join(out_root, f"lb_group_ext_checkpoints__{subset_label}.csv")
            grp.to_csv(ext_csv, index=False)
            print("Wrote", ext_csv)
        else:
            print("[info] no extended checkpoints present (maybe Kcap<5 for all).")
    except Exception as e:
        print("[warn] group checkpoints failed:", e)

if __name__ == "__main__":
    main()
