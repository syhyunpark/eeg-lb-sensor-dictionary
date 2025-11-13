#!/usr/bin/env python3
# benchmark_methods.py — quick OLS timing per 1k rows..... for LB / PCA / ICA / SPH

import os, gc, time, argparse, warnings
import numpy as np
import pandas as pd
import mne
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning
from scipy.special import sph_harm_y  # (l, m, theta=polar, phi=azimuth)

# ---- tiny utils ----

def zscore_rows(A, eps=1e-12):
    mu = A.mean(1, keepdims=True); sd = A.std(1, keepdims=True)
    return (A - mu) / (sd + eps)

def load_D(path):
    d = np.load(os.path.expanduser(path), allow_pickle=True)
    return d["D"], list(d["channels"])

def build_positions(canonical):
    info = mne.create_info(ch_names=canonical, sfreq=250., ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='ignore')
    mp = info.get_montage().get_positions()['ch_pos']
    return np.array([mp.get(ch, [np.nan, np.nan, np.nan]) for ch in canonical], float)

def qr_no_pivot(B, tol=1e-8):
    # thin QR; drop near-zero cols
    if B.size == 0: return np.empty((B.shape[0], 0))
    Q, R = np.linalg.qr(B, mode='reduced')
    if R.size == 0: return np.empty((B.shape[0], 0))
    d = np.abs(np.diag(R))
    if d.size == 0: return np.empty((B.shape[0], 0))
    keep = d > (tol * d.max())
    return Q[:, keep] if keep.any() else np.empty((B.shape[0], 0))

def pca_Q_full(Y, sv_cut=1e-8):
    # PCA loadings (channel space) -> orthonormal Q
    Yc = Y - Y.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(Yc, full_matrices=False)
    if S.size == 0: return np.empty((Y.shape[1], 0))
    keep = S >= (sv_cut * S.max())
    Q = VT.T[:, keep]
    if Q.size: Q, _ = np.linalg.qr(Q)
    return Q

def ica_mixing_Q(Y, Kmax, seed=0, tol=1e-4, max_iter=3000):
    # FastICA mixing -> QR
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Yc, full_matrices=False)
    rank = int((S >= (1e-12 * (S[0] if S.size else 1.0))).sum())
    ncomp = max(1, min(Kmax, Y.shape[1], rank))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            ica = FastICA(n_components=ncomp, whiten='unit-variance',
                          tol=tol, max_iter=max_iter, fun='logcosh',
                          random_state=seed)
            _ = ica.fit_transform(Yc)
            A = ica.mixing_
    except ConvergenceWarning:
        A = None
    if A is None or A.size == 0:
        return np.empty((Y.shape[1], 0))
    return qr_no_pivot(A)

def sph_design_degree_major(pos_kept, Kbuilder):
    # real SH (degree-major) using sph_harm_y
    if pos_kept.size == 0: return np.empty((0, 0))
    P = pos_kept / (np.linalg.norm(pos_kept, axis=1, keepdims=True) + 1e-12)
    theta = np.arccos(np.clip(P[:, 2], -1.0, 1.0))
    phi   = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2*np.pi)
    cols, l = [], 0
    while len(cols) < Kbuilder:
        for m in range(-l, l+1):
            Ylm = sph_harm_y(l, m, theta, phi)
            if m < 0:  v = np.sqrt(2.0)*np.imag(Ylm)
            elif m==0: v = np.real(Ylm)
            else:      v = np.sqrt(2.0)*np.real(Ylm)
            cols.append(np.asarray(v, float))
            if len(cols) >= Kbuilder: break
        l += 1
        if l > 64 and len(cols) == 0: break
    return np.column_stack(cols) if cols else np.empty((pos_kept.shape[0], 0))

def bench_time(fn, repeats=5):
    # avg secs/call
    gc.collect()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = time.perf_counter() - t0
    return dt / max(1, repeats)

#   ---- main -------

def main():
    ap = argparse.ArgumentParser(description="Microbenchmark OLS LB vs PCA/ICA/SPH (prep+projection)")
    ap.add_argument("--dict-npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--canonical-file", required=True)
    ap.add_argument("--subset-file", default=None)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--use-bands", action="store_true")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--limit", type=int, default=202)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--outdir", default=os.path.expanduser("~/lemon_work"))
    args = ap.parse_args()

    D, dict_channels = load_D(args.dict_npz)
    with open(os.path.expanduser(args.canonical_file)) as f:
        canonical = [ln.strip() for ln in f if ln.strip()]
    pos_full = build_positions(canonical)
    pos_map  = {ch: pos_full[canonical.index(ch)] for ch in canonical}

    subset_keep, subset_lbl = None, "Full"
    if args.subset_file:
        subset_lbl = os.path.splitext(os.path.basename(args.subset_file))[0]
        with open(os.path.expanduser(args.subset_file)) as f:
            allow = set(ln.strip() for ln in f if ln.strip()])
        subset_keep = np.array([ch in allow for ch in dict_channels], bool)

    man = pd.read_csv(os.path.expanduser(args.manifest))
    rows_out = []
    rng = np.random.default_rng(0)

    subjects = list(man["subject"])
    if args.limit and args.limit > 0:
        subjects = subjects[:args.limit]

    print(f"[bench] N={len(subjects)} K={args.K} rows_cap={args.rows} subset={subset_lbl} bands={bool(args.use_bands)}")

    for sid_raw in subjects:
        sid = str(sid_raw).zfill(6) if str(sid_raw).isdigit() else str(sid_raw)
        zpath = os.path.join(os.path.dirname(args.manifest), f"sub-{sid}", f"sub-{sid}_Y_tfr.npz")
        if not os.path.exists(zpath):
            continue
        z = np.load(zpath, allow_pickle=True)

        canonical_subj  = list(z["canonical_channels"])
        subj_mask_canon = z["subject_mask"].astype(bool)

        # DICT -> subject canonical
        try:
            idx_in_canon = [canonical_subj.index(ch) for ch in dict_channels]
        except ValueError:
            continue
        subj_mask_dict = np.array(subj_mask_canon)[idx_in_canon]
        keep = subj_mask_dict.copy()
        if subset_keep is not None:
            keep &= subset_keep

        dict_kept = [ch for j, ch in enumerate(dict_channels) if keep[j]]
        m = len(dict_kept)
        if m < 8:
            continue

        subj_used = [canonical_subj[i] for i, f in enumerate(subj_mask_canon) if f]
        idxY = {ch: j for j, ch in enumerate(subj_used)}
        y_cols = [idxY[ch] for ch in dict_kept if ch in idxY]
        if len(y_cols) != m:
            continue

        pos_kept = np.array([pos_map[ch] for ch in dict_kept if ch in idxY], float)
        if np.isnan(pos_kept).any():
            good = ~np.isnan(pos_kept).any(axis=1)
            pos_kept = pos_kept[good]
            y_cols   = [c for c, g in zip(y_cols, good) if g]
            idxs_dict = [j for j, ch in enumerate(dict_channels) if keep[j]]
            keep2 = np.zeros_like(keep, bool)
            for idx_local, g in enumerate(good):
                if g: keep2[idxs_dict[idx_local]] = True
            keep = keep2
            dict_kept = [dict_channels[j] for j in range(len(dict_channels)) if keep[j]]
            m = len(dict_kept)
            if m < 8:
                continue

        def getY(cond):
            key = ("Y_eo_band" if args.use_bands else "Y_eo") if cond == "EO" else \
                  ("Y_ec_band" if args.use_bands else "Y_ec")
            if key not in z: return None
            Y = z[key][:, y_cols]
            if Y.shape[0] > args.rows:
                Y = Y[rng.choice(Y.shape[0], args.rows, replace=False)]
            return zscore_rows(Y)

        for cond in ["EO", "EC"]:
            Y = getY(cond)
            if Y is None: continue

            # each fn builds Q (prep) and projects
            def time_lb():
                Q = qr_no_pivot(D[keep, :])
                if Q.size == 0: return
                Qk = Q[:, :min(args.K, Q.shape[1])]
                _ = (Y @ Qk) @ Qk.T

            def time_sph():
                B = sph_design_degree_major(pos_kept, max(args.K, 1))
                Q = qr_no_pivot(B)
                if Q.size == 0: return
                Qk = Q[:, :min(args.K, Q.shape[1])]
                _ = (Y @ Qk) @ Qk.T

            def time_pca():
                Q = pca_Q_full(Y)
                if Q.size == 0: return
                Qk = Q[:, :min(args.K, Q.shape[1])]
                _ = (Y @ Qk) @ Qk.T

            def time_ica():
                Q = ica_mixing_Q(Y, min(args.K, m), seed=0, tol=1e-4, max_iter=3000)
                if Q.size == 0: return
                Qk = Q[:, :min(args.K, Q.shape[1])]
                _ = (Y @ Qk) @ Qk.T

            for label, fn in [("LB", time_lb), ("SPH", time_sph), ("PCA", time_pca), ("ICA", time_ica)]:
                sec = bench_time(fn, repeats=args.repeat)
                rows_out.append({
                    "subject": sid, "condition": cond, "method": label,
                    "subset": subset_lbl, "m_used": m, "rows": Y.shape[0], "K_used": args.K,
                    "repeat": args.repeat,
                    "sec_per_1000rows": sec * (1000.0 / max(1, Y.shape[0])),
                })

        if len(rows_out) and (len(rows_out) % 200 == 0):
            print(f"[bench] progress: {len(rows_out)} rows logged...")

    out_dir = os.path.join(os.path.expanduser(args.outdir), "derivatives", "Benchmark_OLS")
    os.makedirs(out_dir, exist_ok=True)
    det_csv = os.path.join(out_dir, "benchmark_ols.csv")
    pd.DataFrame(rows_out).to_csv(det_csv, index=False)
    print("Wrote", det_csv)

    if rows_out:
        df = pd.DataFrame(rows_out)
        summary = (
            df.groupby(["method","condition"])
              .agg(sec_per_1000rows_median=("sec_per_1000rows","median"))
              .reset_index()
              .sort_values(["method","condition"])
        )
        sum_csv = os.path.join(out_dir, "benchmark_ols_summary.csv")
        summary.to_csv(sum_csv, index=False)
        print("Wrote", sum_csv)
        print("\nSummary (median sec/1000 rows):")
        try:
            print(summary.to_string(index=False))
        except Exception:
            print(summary)

if __name__ == "__main__":
    main()
