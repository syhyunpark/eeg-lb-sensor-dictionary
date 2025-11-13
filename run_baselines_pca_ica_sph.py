#!/usr/bin/env python3
# baselines_ols_span.py —    SPH (degree blocks) + PCA/ICA (full K).,, masked per subject

import os, argparse, warnings
import numpy as np
import pandas as pd
import mne
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from scipy.special import sph_harm_y  # (l, m, theta=polar, phi=azimuth)

# ------------- utils -------------

def zscore_rows(A, eps=1e-12):
    mu = A.mean(axis=1, keepdims=True); sd = A.std(axis=1, keepdims=True)
    return (A - mu) / (sd + eps)

def load_manifest(path):
    return pd.read_csv(os.path.expanduser(path))

def load_dict_channels(dict_npz_path):
    d = np.load(os.path.expanduser(dict_npz_path), allow_pickle=True)
    return list(d["channels"])

def build_canonical_positions(canonical_list):
    info = mne.create_info(ch_names=canonical_list, sfreq=250., ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='ignore')
    pos_map = info.get_montage().get_positions()['ch_pos']
    return np.array([pos_map.get(ch, [np.nan, np.nan, np.nan]) for ch in canonical_list], float)

def r2_rows(Y, Yhat):
    num = np.sum((Y - Yhat)**2, axis=1); den = np.sum(Y**2, axis=1) + 1e-12
    return 1.0 - num/den

def qr_no_pivot(B):
    if B.size == 0: return np.empty((B.shape[0], 0))
    Q, R = np.linalg.qr(B, mode='reduced')
    if R.size == 0: return np.empty((B.shape[0], 0))
    d = np.abs(np.diag(R))
    keep = d > (1e-12 * (d.max() if d.size else 1.0))
    return Q[:, keep] if keep.size else np.empty((B.shape[0], 0))

def r2_curve_full_prefix(Y, build_prefix_matrix, Kcap):
    out = []
    for k in range(1, Kcap+1):
        Bk = build_prefix_matrix(k)
        Qk = qr_no_pivot(Bk)
        if Qk.size == 0: out.append(0.0); continue
        Yhat = (Y @ Qk) @ Qk.T
        out.append(float(np.mean(r2_rows(Y, Yhat))))
    return out

def r2_at_Ks_prefix(Y, build_prefix_matrix, Ks):
    out = []
    for k in Ks:
        Bk = build_prefix_matrix(k)
        Qk = qr_no_pivot(Bk)
        if Qk.size == 0: out.append(0.0); continue
        Yhat = (Y @ Qk) @ Qk.T
        out.append(float(np.mean(r2_rows(Y, Yhat))))
    return out

def degree_block_Ks(Kcap):
    Lmax = int(np.floor(np.sqrt(max(int(Kcap), 0)))) - 1
    if Lmax < 0: return []
    return [(L+1)**2 for L in range(0, Lmax+1)]

def first_K_at_threshold_full(r2_full, thr):
    arr = np.array(r2_full); idx = np.where(arr >= thr)[0]
    return (int(idx[0]) + 1) if idx.size > 0 else np.nan

def first_K_at_threshold_checkpoints(r2_at_Ks, Ks, thr):
    arr = np.array(r2_at_Ks); idx = np.where(arr >= thr)[0]
    return int(Ks[idx[0]]) if idx.size > 0 else np.nan

# ------------- bases -------------

def pca_Q_full(Y, sv_cut=1e-8):
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Yc, full_matrices=False)
    if S.size == 0: return np.empty((Y.shape[1], 0))
    keep = S >= (sv_cut * S.max())
    Q = VT.T[:, keep]
    if Q.size: Q, _ = np.linalg.qr(Q)
    return Q

def ica_mixing(Y, Kbuilder, random_state=0, ica_tol=1e-4, ica_max_iter=3000):
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Yc, full_matrices=False)
    rank = int((S >= (1e-12 * (S[0] if S.size else 1.0))).sum())
    ncomp = max(1, min(Kbuilder, Yc.shape[1], rank))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            ica = FastICA(n_components=ncomp, whiten='unit-variance',
                          tol=ica_tol, max_iter=ica_max_iter,
                          fun='logcosh', random_state=random_state)
            _ = ica.fit_transform(Yc)
            A = ica.mixing_
    except ConvergenceWarning:
        A = None
    return A

def sph_design_degree_major(pos_kept, Kbuilder):
    if pos_kept.size == 0: return np.empty((0, 0))
    P = pos_kept / (np.linalg.norm(pos_kept, axis=1, keepdims=True) + 1e-12)
    theta = np.arccos(np.clip(P[:, 2], -1.0, 1.0))          # polar
    phi   = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2*np.pi)   # azimuth
    cols, l = [], 0
    while len(cols) < Kbuilder:
        for m in range(-l, l+1):
            Ylm = sph_harm_y(l, m, theta, phi)
            if m < 0:  v = np.sqrt(2.0) * np.imag(Ylm)
            elif m==0: v = np.real(Ylm)
            else:      v = np.sqrt(2.0) * np.real(Ylm)
            cols.append(np.asarray(v, float))
            if len(cols) >= Kbuilder: break
        l += 1
        if l > 30 and len(cols) == 0: break
    return np.column_stack(cols) if cols else np.empty((pos_kept.shape[0], 0))

# ------------- main -------------

def main():
    ap = argparse.ArgumentParser(description="Baselines OLS (SPH at degree blocks; PCA/ICA full-K)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dict-npz", required=True)
    ap.add_argument("--canonical-file", required=True)
    ap.add_argument("--outdir", default=os.path.expanduser("~/lemon_work"))
    ap.add_argument("--use-bands", action="store_true")
    ap.add_argument("--conditions", default="EO,EC")
    ap.add_argument("--Kmax", type=int, default=60)
    ap.add_argument("--subset-file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ica-tol", type=float, default=1e-4)
    ap.add_argument("--ica-max-iter", type=int, default=3000)
    args = ap.parse_args()

    dict_channels = load_dict_channels(args.dict_npz)
    subset_keep, subset_label = None, "ALL59"
    if args.subset_file:
        subset_label = os.path.splitext(os.path.basename(args.subset_file))[0]
        with open(os.path.expanduser(args.subset_file)) as f:
            allow = set(ln.strip() for ln in f if ln.strip())
        subset_keep = np.array([ch in allow for ch in dict_channels], bool)

    out_root = os.path.join(os.path.expanduser(args.outdir), "derivatives", "Baselines_OLS", subset_label)
    os.makedirs(out_root, exist_ok=True)

    # SPH positions
    with open(os.path.expanduser(args.canonical_file)) as f:
        canonical_full = [ln.strip() for ln in f if ln.strip()]
    pos_full = build_canonical_positions(canonical_full)
    pos_map_full = {ch: pos_full[i] for i, ch in enumerate(canonical_full)}

    conds = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    man = load_manifest(args.manifest)

    rows_long, rows_wide = [], []

    print(f"[Baselines] N={len(man)} subjects | subset={subset_label} | Kmax={args.Kmax} | bands={bool(args.use_bands)}")

    for _, rr in man.iterrows():
        sid = str(rr["subject"]).zfill(6) if str(rr["subject"]).isdigit() else str(rr["subject"])
        zpath = os.path.join(os.path.dirname(args.manifest), f"sub-{sid}", f"sub-{sid}_Y_tfr.npz")
        if not os.path.exists(zpath):
            print(f"  !! missing {zpath}")
            continue
        z = np.load(zpath, allow_pickle=True)

        canonical_subj  = list(z["canonical_channels"])
        subj_mask_canon = z["subject_mask"].astype(bool)

        # mask dict order
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

        subj_used = [canonical_subj[i] for i, f in enumerate(subj_mask_canon) if f]
        idxY = {ch:j for j, ch in enumerate(subj_used)}
        y_cols = [idxY[ch] for ch in dict_kept if ch in idxY]
        if len(y_cols) != m_kept:
            print(f"    !! sub-{sid} Y column mismatch ({len(y_cols)} vs {m_kept}) -> skip")
            continue
        pos_kept = np.array([pos_map_full[ch] for ch in dict_kept], float)

        Kcap     = m_kept
        Kbuilder = min(args.Kmax, m_kept)
        Ks_check = degree_block_Ks(Kcap)
        if not Ks_check:
            print(f"    !! sub-{sid} no degree-block Ks -> skip")
            continue

        def fetch(cond):
            key = "Y_eo_band" if (args.use_bands and cond=="EO") else \
                  "Y_ec_band" if (args.use_bands and cond=="EC") else \
                  "Y_eo" if (cond=="EO") else "Y_ec"
            if key not in z: return None
            return zscore_rows(z[key][:, y_cols])

        for cond in conds:
            Y = fetch(cond)
            if Y is None or Y.size == 0:
                continue

            # SPH (degree-major), evaluate at blocks only
            B_sph_full = sph_design_degree_major(pos_kept, Kbuilder)
            R2_sph_ck = r2_at_Ks_prefix(Y, lambda k: B_sph_full[:, :k], Ks_check)

            # ICA (fallback to PCA if it fails), full curve + sampled
            A_ica = ica_mixing(Y, Kbuilder, random_state=args.seed,
                               ica_tol=args.ica_tol, ica_max_iter=args.ica_max_iter)
            if A_ica is None or A_ica.size == 0:
                Q_pca_full = pca_Q_full(Y)
                R2_ica_full = r2_curve_full_prefix(Y, lambda k: Q_pca_full[:, :min(k, Q_pca_full.shape[1])], Kcap)
            else:
                R2_ica_full = r2_curve_full_prefix(Y, lambda k: A_ica[:, :min(k, A_ica.shape[1])], Kcap)
            R2_ica_ck = [R2_ica_full[k-1] for k in Ks_check]

            # PCA full curve + sampled
            Q_pca_full = pca_Q_full(Y)
            R2_pca_full = r2_curve_full_prefix(Y, lambda k: Q_pca_full[:, :min(k, Q_pca_full.shape[1])], Kcap)
            R2_pca_ck = [R2_pca_full[k-1] for k in Ks_check]

            # thresholds
            def thr_from_full(r2_full, thr):
                k = first_K_at_threshold_full(r2_full, thr); att = int(not np.isnan(k))
                return (k,
                        (k / m_kept) if att else np.nan,
                        Kcap, (Kcap / m_kept),
                        att)

            def thr_from_check(r2_ck, Ks, thr):
                k = first_K_at_threshold_checkpoints(r2_ck, Ks, thr); att = int(not np.isnan(k))
                k_cens = int(Ks[-1])
                return (k,
                        (k / m_kept) if att else np.nan,
                        k_cens, (k_cens / m_kept),
                        att)

            # SPH from checkpoints
            K70_sph, K70n_sph, K70c_sph, K70nc_sph, A70_sph = thr_from_check(R2_sph_ck, Ks_check, 0.70)
            K75_sph, K75n_sph, K75c_sph, K75nc_sph, A75_sph = thr_from_check(R2_sph_ck, Ks_check, 0.75)
            K90_sph, K90n_sph, K90c_sph, K90nc_sph, A90_sph = thr_from_check(R2_sph_ck, Ks_check, 0.90)

            # PCA / ICA from full
            K70_pca, K70n_pca, K70c_pca, K70nc_pca, A70_pca = thr_from_full(R2_pca_full, 0.70)
            K75_pca, K75n_pca, K75c_pca, K75nc_pca, A75_pca = thr_from_full(R2_pca_full, 0.75)
            K90_pca, K90n_pca, K90c_pca, K90nc_pca, A90_pca = thr_from_full(R2_pca_full, 0.90)

            K70_ica, K70n_ica, K70c_ica, K70nc_ica, A70_ica = thr_from_full(R2_ica_full, 0.70)
            K75_ica, K75n_ica, K75c_ica, K75nc_ica, A75_ica = thr_from_full(R2_ica_full, 0.75)
            K90_ica, K90n_ica, K90c_ica, K90nc_ica, A90_ica = thr_from_full(R2_ica_full, 0.90)

            # long rows
            for k_val, r2_val in zip(Ks_check, R2_sph_ck):
                rows_long.append({"subject": sid, "condition": cond, "method": "SPH",
                                  "subset": subset_label, "m_used": m_kept, "K": int(k_val), "R2": float(r2_val)})
            for k in range(1, Kcap+1):
                rows_long.append({"subject": sid, "condition": cond, "method": "PCA",
                                  "subset": subset_label, "m_used": m_kept, "K": k, "R2": float(R2_pca_full[k-1])})
                rows_long.append({"subject": sid, "condition": cond, "method": "ICA",
                                  "subset": subset_label, "m_used": m_kept, "K": k, "R2": float(R2_ica_full[k-1])})

            # wide rows
            def wide_row(method, r2_ck, r2_full, Ks):
                if method == "SPH":
                    K70,K70n,K70c,K70nc,A70 = (K70_sph,K70n_sph,K70c_sph,K70nc_sph,A70_sph)
                    K75,K75n,K75c,K75nc,A75 = (K75_sph,K75n_sph,K75c_sph,K75nc_sph,A75_sph)
                    K90,K90n,K90c,K90nc,A90 = (K90_sph,K90n_sph,K90c_sph,K90nc_sph,A90_sph)
                    K_last_full = Kcap; R2_last_full = (r2_full[-1] if r2_full else np.nan)
                elif method == "PCA":
                    K70,K70n,K70c,K70nc,A70 = (K70_pca,K70n_pca,K70c_pca,K70nc_pca,A70_pca)
                    K75,K75n,K75c,K75nc,A75 = (K75_pca,K75n_pca,K75c_pca,K75nc_pca,A75_pca)
                    K90,K90n,K90c,K90nc,A90 = (K90_pca,K90n_pca,K90c_pca,K90nc_pca,A90_pca)
                    K_last_full = Kcap; R2_last_full = (r2_full[-1] if r2_full else np.nan)
                else:  # ICA
                    K70,K70n,K70c,K70nc,A70 = (K70_ica,K70n_ica,K70c_ica,K70nc_ica,A70_ica)
                    K75,K75n,K75c,K75nc,A75 = (K75_ica,K75n_ica,K75c_ica,K75nc_ica,A75_ica)
                    K90,K90n,K90c,K90nc,A90 = (K90_ica,K90n_ica,K90c_ica,K90nc_ica,A90_ica)
                    K_last_full = Kcap; R2_last_full = (r2_full[-1] if r2_full else np.nan)

                wide = {
                    "subject": sid, "condition": cond, "method": method,
                    "subset": subset_label, "m_used": m_kept,
                    "K_last_full": int(K_last_full),
                    "R2_at_K_last_full": float(R2_last_full) if not np.isnan(R2_last_full) else np.nan,
                    "K_last_ck": int(Ks[-1]) if len(Ks) else np.nan,
                    "R2_at_K_last_ck": float(r2_ck[-1]) if r2_ck else np.nan,
                    "K70": K70, "K70_norm": K70n, "K70_cens": K70c, "K70_norm_cens": K70nc, "attained_70": A70,
                    "K75": K75, "K75_norm": K75n, "K75_cens": K75c, "K75_norm_cens": K75nc, "attained_75": A75,
                    "K90": K90, "K90_norm": K90n, "K90_cens": K90c, "K90_norm_cens": K90nc, "attained_90": A90,
                }
                for k_val, r2_val in zip(Ks, r2_ck):
                    wide[f"R2_K{k_val}"] = float(r2_val)
                return wide

            rows_wide.append(wide_row("SPH", R2_sph_ck, R2_sph_ck, Ks_check))
            rows_wide.append(wide_row("PCA", R2_pca_ck, R2_pca_full, Ks_check))
            rows_wide.append(wide_row("ICA", R2_ica_ck, R2_ica_full, Ks_check))

        # light progress ping
        if len(rows_wide) and (len(rows_wide) % 150 == 0):
            print(f"[Baselines] progress: {len(rows_wide)} wide rows...")

    # outputs
    out_dir = os.path.join(os.path.expanduser(args.outdir), "derivatives", "Baselines_OLS", subset_label)
    long_csv = os.path.join(out_dir, f"baselines_ols_curves_long__{subset_label}.csv")
    wide_csv = os.path.join(out_dir, f"baselines_ols_summary__{subset_label}.csv")
    pd.DataFrame(rows_long).sort_values(["subset","method","condition","subject","K"]).to_csv(long_csv, index=False)
    pd.DataFrame(rows_wide).sort_values(["subset","method","condition","subject"]).to_csv(wide_csv, index=False)
    print("Wrote", long_csv)
    print("Wrote", wide_csv)

if __name__ == "__main__":
    main()
