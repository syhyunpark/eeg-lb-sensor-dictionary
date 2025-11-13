#!/usr/bin/env python3
# run_shared_coeffs_icc.py —   shared-basis coeffs ICC(3,1) EO vs EC  .. (and bootstrap CIs)

import os, json, argparse
import numpy as np
import pandas as pd
import mne

# SciPy sph_harm guard (new API uses sph_harm_y)
try:
    from scipy.special import sph_harm_y as _sph_harm
    _USE_Y = True
except ImportError:
    from scipy.special import sph_harm as _sph_harm
    _USE_Y = False

# --------- small utils ---------

def zscore_rows(A, eps: float = 1e-12):
    mu = A.mean(axis=1, keepdims=True); sd = A.std(axis=1, keepdims=True)
    return (A - mu) / (sd + eps)

def load_dict_LB(path):
    d = np.load(os.path.expanduser(path), allow_pickle=True)
    D = d["D"]
    dict_channels = list(d["channels"])
    K_sym = int(d["K"][0]) if "K" in d else D.shape[1]
    meta = {
        "spacing": str(d.get("spacing", ["?"])[0]),
        "subject": str(d.get("subject", ["fsaverage"])[0]),
        "combine": str(d.get("combine", ["sym"])[0]),
        "K": int(K_sym),
    }
    return D, dict_channels, meta

def read_canonical_and_positions(canonical_file):
    with open(os.path.expanduser(canonical_file)) as f:
        canonical_full = [ln.strip() for ln in f if ln.strip()]
    info = mne.create_info(ch_names=canonical_full, sfreq=250., ch_types='eeg')
    info.set_montage(mne.channels.make_standard_montage('standard_1005'), on_missing='ignore')
    pos_map = info.get_montage().get_positions()['ch_pos']
    pos_full = np.array([pos_map.get(ch, [np.nan, np.nan, np.nan]) for ch in canonical_full], float)
    pos_map_full = {ch: pos_full[i] for i, ch in enumerate(canonical_full)}
    return canonical_full, pos_map_full

def sph_dict_from_positions(dict_channels, pos_map_full, Kmax):
    # real SH design, DICT row order, degree-major
    M = len(dict_channels)
    P = np.zeros((M, 3), float)
    for i, ch in enumerate(dict_channels):
        P[i, :] = pos_map_full.get(ch, np.array([np.nan, np.nan, np.nan], float))
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))                 # polar
    phi = np.mod(np.arctan2(y, x), 2*np.pi)              # azimuth
    L = int(np.ceil(np.sqrt(Kmax) - 1))
    cols = []
    for l in range(0, L+1):
        for m in range(-l, l+1):
            Ylm = _sph_harm(l, m, theta, phi) if _USE_Y else _sph_harm(m, l, phi, theta)
            if m < 0: v = np.sqrt(2.0) * np.imag(Ylm)
            elif m == 0: v = np.real(Ylm)
            else: v = np.sqrt(2.0) * np.real(Ylm)
            cols.append(np.asarray(v, float))
            if len(cols) >= Kmax:
                return np.column_stack(cols)
    return np.column_stack(cols) if cols else np.empty((M, 0))

def solver_cols_in_original_coords(X, mode="ridge", sv_cut=1e-12, lam=None, standardize=True):
    # return W so C = Y @ W^T is in ORIGINAL column coords of X (no per-subject rotations)
    if X.size == 0:
        return np.empty((X.shape[1], X.shape[0])), 0

    if standardize:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-12
        Xn = (X - mu) / sd
    else:
        Xn = X
        sd = 1.0

    if mode == "svd":
        U, S, VT = np.linalg.svd(Xn, full_matrices=False)
        if S.size == 0:
            return np.empty((X.shape[1], X.shape[0])), 0
        keep = S >= (sv_cut * S.max())
        r_eff = int(np.sum(keep))
        Sinv = np.zeros_like(S); Sinv[keep] = 1.0 / S[keep]
        W = VT.T @ np.diag(Sinv) @ U.T
    elif mode == "ridge":
        K = Xn.shape[1]
        if lam is None:
            lam = 1e-4 * (np.trace(Xn.T @ Xn) / max(K, 1))
        A = Xn.T @ Xn + lam * np.eye(K)
        W = np.linalg.solve(A, Xn.T)
        r_eff = int(np.linalg.matrix_rank(Xn))
    else:
        raise ValueError("mode must be 'ridge' or 'svd'")

    if standardize:
        W = W / sd.T
    return W, r_eff

def icc_3_1(x, y):
    # Shrout & Fleiss ICC(3,1) (consistency) for two raters (EO,EC)
    data = np.vstack([x, y]).T
    n, k = data.shape
    if n < 2:
        return np.nan
    m_subj = data.mean(axis=1, keepdims=True)
    m_rater = data.mean(axis=0, keepdims=True)
    grand = data.mean()
    SSR = k * np.sum((m_subj.squeeze() - grand)**2)
    SSE = np.sum((data - m_subj - m_rater + grand)**2)
    MSR = SSR / (n - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    return float((MSR - MSE) / (MSR + (k - 1) * MSE + 1e-12))

def degree_of_mode_index(k1: int) -> int:
    # 1-based index -> spherical degree l (l^2 < k <= (l+1)^2)
    return int(np.floor(np.sqrt(max(int(k1), 1) - 1)))

# --------- main ---------

def main():
    ap = argparse.ArgumentParser(description="Shared-basis coeffs + ICC(3,1) (EO vs EC)")
    ap.add_argument("--method", choices=["LB", "SPH"], default="LB")
    ap.add_argument("--dict-npz", help="LB needs this; SPH optional (dict order)")
    ap.add_argument("--canonical-file", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--outdir", default=os.path.expanduser("~/lemon_work"))
    ap.add_argument("--subset-file", default=None)
    ap.add_argument("--use-bands", action="store_true")
    ap.add_argument("--Ksph", type=int, default=60)
    # solver
    ap.add_argument("--solver", choices=["ridge", "svd"], default="ridge")
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--sv-cut", type=float, default=1e-12)
    ap.add_argument("--no-standardize", action="store_true")
    # bootstrap
    ap.add_argument("--icc-bootstrap", type=int, default=1000)
    ap.add_argument("--degree-only", action="store_true",
                    help="SPH only: write ICC rows only at degree endpoints K=(l+1)^2")
    args = ap.parse_args()

    standardize = (not args.no_standardize)
    subset_label = "ALL59" if not args.subset_file else os.path.splitext(os.path.basename(args.subset_file))[0]
    method_tag = "LB_icc_OLS" if args.method == "LB" else "SPH_icc_OLS"
    out_root = os.path.join(os.path.expanduser(args.outdir), "derivatives", method_tag, subset_label)
    os.makedirs(out_root, exist_ok=True)

    # shared basis in DICT row order
    if args.method == "LB":
        if not args.dict_npz:
            raise ValueError("--dict-npz is required for method=LB")
        B_full, dict_channels, meta = load_dict_LB(args.dict_npz)
    else:
        if args.dict_npz:
            dict_channels = load_dict_LB(args.dict_npz)[1]
        else:
            with open(os.path.expanduser(args.canonical_file)) as f:
                dict_channels = [ln.strip() for ln in f if ln.strip()]
        _, pos_map_full = read_canonical_and_positions(args.canonical_file)
        B_full = sph_dict_from_positions(dict_channels, pos_map_full, args.Ksph)
        meta = {"spacing": "NA", "subject": "fsaverage", "combine": "sph", "K": int(B_full.shape[1])}

    subset_keep = None
    if args.subset_file:
        with open(os.path.expanduser(args.subset_file)) as f:
            allow = set(ln.strip() for ln in f if ln.strip())
        subset_keep = np.array([ch in allow for ch in dict_channels], bool)

    print(f"[Shared-ICC] method={args.method} subset={subset_label} K_cols={B_full.shape[1]}")
    man = pd.read_csv(os.path.expanduser(args.manifest))
    print(f"[Shared-ICC] N={len(man)} subjects | bands={bool(args.use_bands)}")

    subj_band_means = {}  # (sid, band) -> {"EO": vecK, "EC": vecK}
    coeff_rows = []       # optional long coeff dump

    for _, rr in man.iterrows():
        sid = str(rr["subject"]).zfill(6) if str(rr["subject"]).isdigit() else str(rr["subject"])
        zpath = os.path.join(os.path.dirname(args.manifest), f"sub-{sid}", f"sub-{sid}_Y_tfr.npz")
        if not os.path.exists(zpath):
            print(f"  !! missing {zpath}")
            continue
        z = np.load(zpath, allow_pickle=True)

        canonical_subj = list(z["canonical_channels"])
        subj_mask_canon = z["subject_mask"].astype(bool)
        subj_idx = {ch: i for i, ch in enumerate(canonical_subj)}
        keep = np.array([subj_mask_canon[subj_idx.get(ch, -1)] if ch in subj_idx else False
                         for ch in dict_channels], bool)
        if subset_keep is not None:
            keep &= subset_keep

        dict_kept = [ch for j, ch in enumerate(dict_channels) if keep[j]]
        m_kept = len(dict_kept)
        if m_kept < 8:
            print(f"    !! sub-{sid} too few channels ({m_kept}) -> skip")
            continue

        subj_used = [canonical_subj[i] for i, f in enumerate(subj_mask_canon) if f]
        idx_in_Y = {ch: j for j, ch in enumerate(subj_used)}
        y_cols = [idx_in_Y[ch] for ch in dict_kept if ch in idx_in_Y]
        if len(y_cols) != m_kept:
            print(f"    !! sub-{sid} Y col mismatch ({len(y_cols)} vs {m_kept}) -> skip")
            continue

        X = B_full[keep, :]
        W, r_eff = solver_cols_in_original_coords(
            X, mode=args.solver, sv_cut=args.sv_cut, lam=args.lam, standardize=standardize
        )

        def fetch(cond):
            if args.use_bands:
                keyY = "Y_eo_band" if cond == "EO" else "Y_ec_band"
                keyI = "index_eo_band" if cond == "EO" else "index_ec_band"
            else:
                keyY = "Y_eo" if cond == "EO" else "Y_ec"
                keyI = None
            if keyY not in z:
                return None, None
            Y = z[keyY][:, y_cols]
            Y = zscore_rows(Y)
            bands = None
            if args.use_bands and keyI in z:
                bands = [str(b) for (_, b) in list(z[keyI])]
            return Y, bands

        YEO, bEO = fetch("EO")
        YEC, bEC = fetch("EC")
        if YEO is None or YEC is None:
            print(f"    -- sub-{sid}: missing EO/EC -> skip")
            continue

        # coeffs in shared coords
        C_eo = YEO @ W.T
        C_ec = YEC @ W.T

        def band_mean(C, bands):
            df = pd.DataFrame(C); df["band"] = bands
            return df.groupby("band").mean().sort_index()

        if args.use_bands and bEO and bEC:
            B_eo = band_mean(C_eo, bEO)
            B_ec = band_mean(C_ec, bEC)
            common_bands = sorted(set(B_eo.index).intersection(set(B_ec.index)))
            for b in common_bands:
                vec_eo = B_eo.loc[b].values.astype(float)
                vec_ec = B_ec.loc[b].values.astype(float)
                subj_band_means[(sid, b)] = {"EO": vec_eo, "EC": vec_ec}
                # long coeffs (optional)
                for k, val in enumerate(vec_eo, start=1):
                    coeff_rows.append({"subject": sid, "band": b, "mode": k, "coeff": float(val), "condition": "EO"})
                for k, val in enumerate(vec_ec, start=1):
                    coeff_rows.append({"subject": sid, "band": b, "mode": k, "coeff": float(val), "condition": "EC"})

        print(f"[sub-{sid}] m_used={m_kept} r_eff={r_eff} K_cols={B_full.shape[1]}")

        # minimal per-subject meta
        sub_meta = {
            "subject": sid, "subset": subset_label,
            "method": args.method, "K_cols": int(B_full.shape[1]),
            "effective_rank": int(r_eff), "m_used": int(m_kept),
            "bands": sorted(list(set((bEO or []) + (bEC or []))))
        }
        subj_dir = os.path.join(out_root, f"sub-{sid}")
        os.makedirs(subj_dir, exist_ok=True)
        with open(os.path.join(subj_dir, f"sub-{sid}_shared_coeffs.json"), "w") as f:
            json.dump(sub_meta, f, indent=2)

    # --------- ICC per band × mode (bootstrap) ----------
    rows_icc = []
    if subj_band_means:
        dfm = pd.DataFrame([{"subject": s, "band": b, "EO": v["EO"], "EC": v["EC"]}
                            for (s, b), v in subj_band_means.items()])
        for band, g in dfm.groupby("band"):
            EO = np.vstack(g["EO"].to_numpy()); EC = np.vstack(g["EC"].to_numpy())
            n, Kc = EO.shape
            for k in range(Kc):
                if args.method == "SPH" and args.degree_only:
                    l = degree_of_mode_index(k+1)
                    if (k+1) != (l+1)**2:   # not an endpoint -> skip if degree-only
                        continue
                x = EO[:, k]; y = EC[:, k]
                icc = icc_3_1(x, y)
                boot = []
                for _ in range(args.icc_bootstrap):
                    idx = np.random.randint(0, n, n)
                    boot.append(icc_3_1(x[idx], y[idx]))
                lo, hi = np.percentile(boot, [2.5, 97.5])
                l = degree_of_mode_index(k+1)
                is_endpoint = int((k+1) == (l+1)**2)
                rows_icc.append({
                    "method": args.method, "subset": subset_label, "band": band,
                    "mode": int(k+1), "degree": int(l), "is_degree_endpoint": is_endpoint,
                    "ICC3_1": float(icc), "CI_low": float(lo), "CI_high": float(hi)
                })

    # write
    icc_csv   = os.path.join(out_root, f"icc_by_mode__{args.method}__{subset_label}.csv")
    coeff_csv = os.path.join(out_root, f"coefficients_long__{args.method}__{subset_label}.csv")
    pd.DataFrame(rows_icc).to_csv(icc_csv, index=False)
    pd.DataFrame(coeff_rows).to_csv(coeff_csv, index=False)
    print(f"Wrote:\n  {icc_csv}\n  {coeff_csv}\nDone.")

if __name__ == "__main__":
    main()
