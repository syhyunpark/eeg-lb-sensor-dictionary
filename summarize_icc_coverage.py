#!/usr/bin/env python3
# summarize_icc_coverage.py — ICC coverage (LB vs SPH) ... cap to first K modes, bootstrap CIs

import os, re, argparse
import numpy as np
import pandas as pd

def load_icc_table(root, method, subset):
    # one method/subset table
    p = os.path.join(os.path.expanduser(root),
                     "derivatives",
                     "LB_icc_OLS" if method.upper() == "LB" else "SPH_icc_OLS",
                     subset,
                     f"icc_by_mode__{method}__{subset}.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    need = {"band", "mode", "ICC3_1"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"{p} missing columns: {sorted(miss)}")
    df["method"] = method.upper()
    df["subset"] = subset
    df["mode"] = pd.to_numeric(df["mode"], errors="coerce")
    df["ICC3_1"] = pd.to_numeric(df["ICC3_1"], errors="coerce")
    return df

def infer_m_from_subset(subset: str) -> int:
    # ALL59 / canonical_32 / canonical_19 / fallback parse last number
    s = subset.strip()
    lut = {"ALL59": 59, "all59": 59, "canonical_32": 32, "canonical_19": 19}
    if s in lut:
        return lut[s]
    m = re.findall(r"(\d+)", s)
    if m:
        return int(m[-1])
    raise ValueError(f"Could not infer M from subset='{subset}'")

def boot_ci_mean(vals, B=2000, alpha=0.05, seed=0):
    # bootstrap mean CI
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    n = vals.size
    if n == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.empty(B, float)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = float(np.mean(vals[idx]))
    mean = float(np.mean(vals))
    lo, hi = np.percentile(boot, [100*alpha/2.0, 100*(1.0 - alpha/2.0)])
    return mean, float(lo), float(hi)

def main():
    ap = argparse.ArgumentParser(description="ICC coverage with cap to first K modes (LB vs SPH) + bootstrap CIs")
    ap.add_argument("--root", required=True, help="e.g., ~/lemon_work")
    ap.add_argument("--subset", default="ALL59", help="ALL59 / canonical_32 / canonical_19")
    ap.add_argument("--thresholds", default="0.3,0.5", help="comma-separated ICC thresholds")
    ap.add_argument("--topk", type=int, default=20, help="use first K modes")
    ap.add_argument("--boot-B", type=int, default=2000)
    ap.add_argument("--boot-alpha", type=float, default=0.05)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    thr_list = [float(x) for x in args.thresholds.split(",") if x.strip()]
    M = infer_m_from_subset(args.subset)
    K_cap = max(1, min(int(args.topk), M))

    frames = []
    for method in ["LB", "SPH"]:
        try:
            frames.append(load_icc_table(args.root, method, args.subset))
        except (FileNotFoundError, ValueError) as e:
            print(f"[warn] missing/invalid ICC file for {method}/{args.subset}: {e}")
    if not frames:
        raise SystemExit("no ICC tables found; nothing to do")

    df = pd.concat(frames, ignore_index=True)
    df = df[pd.to_numeric(df["mode"], errors="coerce").notna()].copy()
    df = df[df["mode"] <= K_cap].copy()

    print(f"[coverage] subset={args.subset} (M={M}) topK={K_cap} "
          f"methods={sorted(df['method'].unique().tolist())} "
          f"bands={sorted(df['band'].dropna().unique().tolist())} Nrows={len(df)}")

    rows = []
    for (subset, method, band), g in df.groupby(["subset", "method", "band"], dropna=True):
        icc = g["ICC3_1"].to_numpy(float)
        icc = icc[np.isfinite(icc)]
        K_eff = int(icc.size)

        if K_eff == 0:
            for t in thr_list:
                rows.append({
                    "subset": subset, "method": method, "band": band,
                    "K_eff": 0, "thr": t,
                    "count_ge_thr": np.nan, "pct_ge_thr": np.nan,
                    "pct_lo": np.nan, "pct_hi": np.nan,
                    "B": args.boot_B, "alpha": args.boot_alpha
                })
            rows.append({
                "subset": subset, "method": method, "band": band,
                "K_eff": 0, "thr": "AUC",
                "auc_mean": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
                "B": args.boot_B, "alpha": args.boot_alpha
            })
            continue

        # AUC-ish mean ICC over first K_eff modes
        auc_mean, auc_lo, auc_hi = boot_ci_mean(icc, B=args.boot_B, alpha=args.boot_alpha, seed=args.boot_seed)
        rows.append({
            "subset": subset, "method": method, "band": band,
            "K_eff": K_eff, "thr": "AUC",
            "auc_mean": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
            "B": args.boot_B, "alpha": args.boot_alpha
        })

        # coverage at thresholds (resample modes; denom fixed=K_eff)
        rng = np.random.default_rng(args.boot_seed)
        for t in thr_list:
            flags = (icc >= t).astype(float)
            cnt = int(flags.sum())
            pct = 100.0 * float(flags.mean())
            boot = np.empty(args.boot_B, float)
            for b in range(args.boot_B):
                idx = rng.integers(0, K_eff, size=K_eff)
                boot[b] = 100.0 * float(flags[idx].mean())
            lo, hi = np.percentile(boot, [100*args.boot_alpha/2.0, 100*(1.0 - args.boot_alpha/2.0)])

            rows.append({
                "subset": subset, "method": method, "band": band,
                "K_eff": K_eff, "thr": t,
                "count_ge_thr": cnt, "pct_ge_thr": pct,
                "pct_lo": float(lo), "pct_hi": float(hi),
                "B": args.boot_B, "alpha": args.boot_alpha
            })

    out = pd.DataFrame(rows).sort_values(["subset", "band", "method", "thr"])
    out_path = os.path.join(os.path.expanduser(args.root),
                            "derivatives", "Merged_Report",
                            f"icc_coverage__{args.subset}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print("[coverage] wrote", out_path)

if __name__ == "__main__":
    main()
