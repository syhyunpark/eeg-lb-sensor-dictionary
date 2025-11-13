# lemon_batch_tfr.py
# Batch Morlet TFR for LEMON (EO/EC). Writes Y_tfr npz per subject + manifest.csv

import os, re, argparse
import requests
import numpy as np
import pandas as pd
import mne

BASE_URL = ("https://ftp.gwdg.de/pub/misc/"
            "MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/"
            "EEG_Preprocessed_BIDS_ID/EEG_Preprocessed")

DEFAULT_BANDS = {
    "delta": (2.0, 4.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta":  (13.0, 30.0),
}

def http_download(url, dst_path, chunk=2**20):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return dst_path
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dst_path, "wb") as f:
            for c in r.iter_content(chunk_size=chunk):
                if c:
                    f.write(c)
    return dst_path

def discover_subjects():
    # quick scrape of listing; keep sids that have both EO and EC .set
    r = requests.get(BASE_URL, timeout=60)
    r.raise_for_status()
    html = r.text
    matches = re.findall(r'href="(sub-(\d{6})_(EO|EC)\.set)"', html)
    eo, ec = set(), set()
    for _full, sid, cond in matches:
        (eo if cond == "EO" else ec).add(sid)
    return sorted(eo & ec)

def load_condition(set_path, montage_name="standard_1005"):
    # read .set, keep EEG only, set montage, avg ref
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    picks = mne.pick_types(raw.info, eeg=True, eog=False, ecg=False, stim=False, misc=False)
    raw.pick(picks)
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing='ignore', verbose=False)
    raw.set_eeg_reference('average', verbose=False)
    return raw

def boundary_intervals_samples(raw):
    # collect [start,end) in samples for annotations that look like "boundary"
    sf = raw.info["sfreq"]
    out = []
    for ann in raw.annotations:
        desc = ann['description']
        if isinstance(desc, str) and ('boundary' in desc.lower()):
            onset = float(ann['onset'])
            dur = float(ann['duration']) if ann['duration'] is not None else 0.0
            s0 = int(np.floor(onset * sf))
            s1 = int(np.ceil((onset + max(dur, 0.0)) * sf))
            if s1 <= s0:
                s1 = s0 + 1
            s0 = int(np.clip(s0, 0, raw.n_times))
            s1 = int(np.clip(s1, 0, raw.n_times))
            if s1 > s0:
                out.append((s0, s1))
    return out

def epochify_rest_boundary_aware(raw, win_len_s=10.0):
    # simple non-overlap windows; drop any that touch a boundary
    sf = raw.info["sfreq"]
    n_samp = int(np.floor(win_len_s * sf))
    total = raw.n_times
    n_ep_all = total // n_samp
    data = raw.get_data()[:, :n_ep_all * n_samp]

    ep_ranges = [(i*n_samp, (i+1)*n_samp) for i in range(n_ep_all)]
    boundaries = boundary_intervals_samples(raw)

    def overlaps(a0, a1, b0, b1):
        return (a0 < b1) and (b0 < a1)

    keep = []
    for i, (s0, s1) in enumerate(ep_ranges):
        bad = any(overlaps(s0, s1, b0, b1) for (b0, b1) in boundaries)
        if not bad:
            keep.append(i)

    if not keep:
        # dead simple fallback, keep everything
        n_ep = n_ep_all
        ep = data.reshape(raw.info['nchan'], n_ep, n_samp, order='F').transpose(1, 0, 2)
        return ep, sf, list(raw.ch_names)

    data = data.reshape(raw.info['nchan'], n_ep_all, n_samp, order='F')[:, keep, :]
    ep = np.transpose(data, (1, 0, 2))
    return ep, sf, list(raw.ch_names)

def morlet_topographies(epochs, sfreq, fmin=2.0, fmax=30.0, n_freqs=15,
                        cycles_rule="constQ", decim=2):
    # Morlet power per epoch/freq, then average over time
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_freqs)
    if cycles_rule == "const":
        n_cycles = 7.0
    else:
        n_cycles = np.clip(freqs / 2.0, 3.0, 10.0)

    power = mne.time_frequency.tfr_array_morlet(
        epochs, sfreq=sfreq, freqs=freqs, n_cycles=n_cycles,
        output='power', decim=decim, n_jobs=1, verbose=False
    )  # (n_ep, n_ch, n_freq, n_time)
    topo = power.mean(axis=-1)  # (n_ep, n_ch, n_freq)
    n_ep, n_ch, n_f = topo.shape
    Y = topo.transpose(0, 2, 1).reshape(n_ep * n_f, n_ch)
    index = [(int(e), float(freqs[f])) for e in range(n_ep) for f in range(n_f)]
    return freqs, Y, index

def aggregate_bands(freqs, Y, index, bands=DEFAULT_BANDS):
    # average rows within freq bands; keep (epoch, band) order
    df_idx = pd.DataFrame(index, columns=["epoch", "freq"])
    epochs = df_idx["epoch"].to_numpy()
    uniq = np.unique(epochs)
    band_names = list(bands.keys())
    rows, idx_out = [], []
    F = len(freqs)
    for ep in uniq:
        s = int(ep * F); t = int((ep + 1) * F)
        Y_ep = Y[s:t, :]
        for bname, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs <= hi)
            if np.any(mask):
                topo = np.nanmean(Y_ep[mask, :], axis=0)
            else:
                topo = np.full((Y_ep.shape[1],), np.nan)
            rows.append(topo)
            idx_out.append((int(ep), bname))
    Y_band = np.vstack(rows) if rows else np.empty((0, Y.shape[1]))
    return np.array(band_names, dtype=object), Y_band, idx_out

def load_canonical_list(path):
    # read list (one per line), keep order, drop dups
    with open(path, "r") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    seen, out = set(), []
    for ch in names:
        if ch not in seen:
            out.append(ch)
            seen.add(ch)
    return out

def make_subject_mask(canonical_list, subject_ch_names):
    # boolean mask over canonical; and indices into subject list
    subj_set = set(subject_ch_names)
    mask = np.array([ch in subj_set for ch in canonical_list], dtype=bool)
    idx = [subject_ch_names.index(ch) for ch in np.array(canonical_list)[mask]]
    return mask, idx

def process_subject(sid, out_root, canonical_list, win_len_s=10.0, fmin=2.0, fmax=30.0,
                    n_freqs=15, cycles_rule="constQ", decim=2, do_band_agg=True,
                    min_kept_epochs=1):
    # pull files if needed
    subj_dir_raw = os.path.join(out_root, "preprocessed_cache", f"sub-{sid}")
    os.makedirs(subj_dir_raw, exist_ok=True)
    files = [f"sub-{sid}_EO.set", f"sub-{sid}_EO.fdt",
             f"sub-{sid}_EC.set", f"sub-{sid}_EC.fdt"]
    local = {}
    for fname in files:
        url = f"{BASE_URL}/{fname}"
        dst = os.path.join(subj_dir_raw, fname)
        http_download(url, dst)
        local[fname] = dst

    # load both
    raw_eo = load_condition(local[f"sub-{sid}_EO.set"])
    raw_ec = load_condition(local[f"sub-{sid}_EC.set"])

    # same channels & order
    common = [ch for ch in raw_eo.ch_names if ch in raw_ec.ch_names]
    raw_eo.pick(common); raw_ec.pick(common)

    # epochs (boundary-aware)
    ep_eo, sf, ch_eo = epochify_rest_boundary_aware(raw_eo, win_len_s=win_len_s)
    ep_ec, _,  ch_ec = epochify_rest_boundary_aware(raw_ec, win_len_s=win_len_s)

    if ep_eo.shape[0] < min_kept_epochs and ep_ec.shape[0] < min_kept_epochs:
        raise RuntimeError("No epochs kept after boundary exclusion.")

    # per-subject mask over canonical
    mask, idx = make_subject_mask(canonical_list, ch_eo)
    ep_eo = ep_eo[:, idx, :]
    ep_ec = ep_ec[:, idx, :]

    # TFR -> time-avg topographies
    freqs, Y_eo, idx_eo = morlet_topographies(ep_eo, sf, fmin, fmax, n_freqs, cycles_rule, decim)
    _,     Y_ec, idx_ec = morlet_topographies(ep_ec, sf, fmin, fmax, n_freqs, cycles_rule, decim)

    # optional bands
    if do_band_agg:
        band_names, Y_eo_band, idx_eo_band = aggregate_bands(freqs, Y_eo, idx_eo, DEFAULT_BANDS)
        _,          Y_ec_band, idx_ec_band = aggregate_bands(freqs, Y_ec, idx_ec, DEFAULT_BANDS)
    else:
        band_names = np.array([], dtype=object)
        Y_eo_band = np.empty((0, ep_eo.shape[1])); idx_eo_band = []
        Y_ec_band = np.empty((0, ep_ec.shape[1])); idx_ec_band = []

    # save (channels saved as canonical[mask])
    subj_deriv = os.path.join(out_root, "derivatives", "Y_tfr", f"sub-{sid}")
    os.makedirs(subj_deriv, exist_ok=True)
    out_npz = os.path.join(subj_deriv, f"sub-{sid}_Y_tfr.npz")
    np.savez_compressed(
        out_npz,
        canonical_channels=np.array(canonical_list, dtype=object),
        subject_mask=mask,
        channels=np.array(np.array(canonical_list)[mask], dtype=object),
        freqs=np.array(freqs, dtype=float),
        win_len_s=np.array([win_len_s], dtype=float),
        Y_ec=Y_ec, index_ec=np.array(idx_ec, dtype=object),
        Y_eo=Y_eo, index_eo=np.array(idx_eo, dtype=object),
        bands=np.array(band_names, dtype=object),
        Y_ec_band=Y_ec_band, index_ec_band=np.array(idx_ec_band, dtype=object),
        Y_eo_band=Y_eo_band, index_eo_band=np.array(idx_eo_band, dtype=object),
        cycles_rule=np.array([cycles_rule], dtype=object),
        decim=np.array([decim], dtype=int)
    )

    return {
        "subject": sid,
        "m_used": int(mask.sum()),
        "m_canonical": len(canonical_list),
        "eo_epochs": len({e for (e, _) in idx_eo}),
        "ec_epochs": len({e for (e, _) in idx_ec}),
        "fmin": fmin, "fmax": fmax, "n_freqs": n_freqs,
        "cycles_rule": cycles_rule, "decim": decim,
        "win_len_s": win_len_s,
        "do_band_agg": do_band_agg,
        "npz_path": out_npz
    }

def load_subjects_from_file(path):
    with open(path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return [s for s in ids if re.fullmatch(r"\d{6}", s)]

def main():
    ap = argparse.ArgumentParser(description="LEMON batch Morlet TFR -> Y(t,f) (boundary-aware, masked)")
    ap.add_argument("--outdir", default=os.path.expanduser("~/data/LEMON_preproc"))
    ap.add_argument("--subjects-file", default=None)
    ap.add_argument("--canonical-file", default=None,
                    help="Text file with canonical EEG channels (one per line)")
    ap.add_argument("--win-len", type=float, default=10.0)
    ap.add_argument("--fmin", type=float, default=2.0)
    ap.add_argument("--fmax", type=float, default=30.0)
    ap.add_argument("--n-freqs", type=int, default=15)
    ap.add_argument("--cycles-rule", choices=["const", "constQ"], default="constQ")
    ap.add_argument("--decim", type=int, default=2)
    ap.add_argument("--no-band-agg", action="store_true")
    args = ap.parse_args()

    out_root = args.outdir
    os.makedirs(out_root, exist_ok=True)

    if args.subjects_file:
        subjects = load_subjects_from_file(args.subjects_file)
    else:
        print("Discovering subjects (HTTP scrape)...")
        subjects = discover_subjects()
    if not subjects:
        raise SystemExit("No subjects found. If listing blocks scrape, pass --subjects-file.")

    # canonical list
    if args.canonical_file:
        canonical_list = load_canonical_list(args.canonical_file)
        print(f"Loaded canonical ({len(canonical_list)} ch) from {args.canonical_file}")
    else:
        print("No --canonical-file. Will take channels from first subject (post-EO/EC intersect).")
        canonical_list = None

    print(f"Found {len(subjects)} subjects with both EO and EC.")
    manifest_rows = []
    for i, sid in enumerate(subjects, 1):
        try:
            print(f"[{i}/{len(subjects)}] sub-{sid}")
            if canonical_list is None:
                # peek channels from first subject after EO/EC intersection
                p_eo = os.path.join(out_root, "preprocessed_cache", f"sub-{sid}", f"sub-{sid}_EO.set")
                p_ec = os.path.join(out_root, "preprocessed_cache", f"sub-{sid}", f"sub-{sid}_EC.set")
                if os.path.exists(p_eo):
                    tmp_raw_eo = load_condition(p_eo)
                else:
                    tmp_raw_eo = load_condition(http_download(f"{BASE_URL}/sub-{sid}_EO.set", p_eo))
                if os.path.exists(p_ec):
                    tmp_raw_ec = load_condition(p_ec)
                else:
                    tmp_raw_ec = load_condition(http_download(f"{BASE_URL}/sub-{sid}_EC.set", p_ec))
                common = [ch for ch in tmp_raw_eo.ch_names if ch in tmp_raw_ec.ch_names]
                canonical_list = list(common)
                print(f"Derived canonical of {len(canonical_list)} channels from sub-{sid}.")
            row = process_subject(
                sid, out_root, canonical_list,
                win_len_s=args.win_len,
                fmin=args.fmin, fmax=args.fmax, n_freqs=args.n_freqs,
                cycles_rule=args.cycles_rule, decim=args.decim,
                do_band_agg=(not args.no_band_agg)
            )
            manifest_rows.append(row)
        except Exception as e:
            print(f"  !! sub-{sid} failed: {e}")

    # write manifest
    manifest_dir = os.path.join(out_root, "derivatives", "Y_tfr")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_csv = os.path.join(manifest_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).sort_values("subject").to_csv(manifest_csv, index=False)
    print(f"\nWrote manifest: {manifest_csv}\nDone.")

if __name__ == "__main__":
    main()
