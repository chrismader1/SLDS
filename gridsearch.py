# --------------------------------------------------------------------------------------
# Import modules
# --------------------------------------------------------------------------------------

import numpy as np
import numpy.random as npr
import pandas as pd
import re
import json
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
from collections import Counter
from itertools import groupby
import matplotlib.pyplot as plt
import itertools
from fractions import Fraction
from scipy.special import logsumexp
import types
import sys
sys.path.insert(0, "/Users/chrismader/Python/SLDS")

# switch of widgets before importing ssm
import os
os.environ["TQDM_NOTEBOOK"] = "0"   # no notebook widget
import tqdm
import tqdm.auto as tqa
from tqdm.std import tqdm as tqdm_std
def _trange_no_widget(n, *a, **k):
    k.setdefault("disable", True)          # True = no bars; False = console bars
    return tqdm_std(range(n), *a, **k)     # real tqdm object (has set_description)
tqdm.tqdm = tqdm_std
tqdm.trange = _trange_no_widget
tqa.tqdm = tqdm_std
tqa.trange = _trange_no_widget
import ssm

from rSLDS import *


# --------------------------------------------------------------------------------------
# Data import
# --------------------------------------------------------------------------------------

FF_PRESETS = {
    "ff3":     ["mkt", "smb", "hml"],
    "ff3mom":  ["mkt", "smb", "hml", "mom"],
    "ff5":     ["mkt", "smb", "hml", "rmw", "cma"],
    "ff5mom":  ["mkt", "smb", "hml", "rmw", "cma", "mom"],
}

def import_data(filename):

    def _sheet(sheet):
        df = pd.read_excel(filename, sheet_name=sheet, skiprows=4, index_col=0)
        df = df.iloc[3:, :]
        df.index = pd.to_datetime(df.index)
        df = df.apply(pd.to_numeric, errors='coerce').ffill()
        return df

    # Core panels
    df_eq_px  = _sheet("SPX_PX")
    df_idx_px = _sheet("IDX_PX")
    df_eq_eps = _sheet("SPX_EPS")
    df_idx_eps= _sheet("IDX_EPS")

    # Extract VIX and remove from IDX_PX
    vix_col = next((c for c in df_idx_px.columns if str(c).strip().upper() == "VIX"), None)
    if vix_col is not None:
        ser_vix = df_idx_px.pop(vix_col).astype(float)
        ser_vix.name = "VIX"
    else:
        ser_vix = pd.Series(index=df_idx_px.index, dtype=float, name="VIX")

    # Combine panels
    px_all  = pd.concat([df_eq_px, df_idx_px], axis=1)
    eps_all = pd.concat([df_eq_eps, df_idx_eps], axis=1)
    eps_pos = eps_all.where(eps_all > 0)
    pe_all  = (px_all / eps_pos).where(lambda df: df > 0)

    return px_all, eps_all, pe_all, ser_vix


def import_factors():
    """
    Load Fama–French daily files you provided, using their exact column names:
    - F-F_Research_Data_5_Factors_2x3_daily.csv  -> Mkt-RF, SMB, HML, RMW, CMA, RF
    - F-F_Research_Data_Factors_daily.csv        -> Mkt-RF, SMB, HML, RF
    - F-F_Momentum_Factor_daily.csv              -> Mom
    Returns a DataFrame indexed by datetime with those columns (when present).
    """

    p5   = "/Users/chrismader/Python/SLDS/Data/F-F_Research_Data_5_Factors_2x3_daily.csv"
    p3   = "/Users/chrismader/Python/SLDS/Data/F-F_Research_Data_Factors_daily.csv"
    pmom = "/Users/chrismader/Python/SLDS/Data/F-F_Momentum_Factor_daily.csv"

    def _read_ff(path):
        # find the header line (the first line that starts with a comma)
        with open(path, "r", encoding="latin1", errors="ignore") as f:
            lines = f.readlines()
        header_idx = next(i for i, l in enumerate(lines) if l.startswith(","))

        df = pd.read_csv(path, skiprows=header_idx)
        # first column is the date (unnamed in the CSVs)
        df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
        # keep only YYYYMMDD rows
        df = df[df["DATE"].astype(str).str.strip().str.match(r"^\d{8}$", na=False)]
        df["DATE"] = pd.to_datetime(df["DATE"].astype(str).str.strip(), format="%Y%m%d")
        df.set_index("DATE", inplace=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        return df

    df5   = _read_ff(p5)     # Mkt-RF, SMB, HML, RMW, CMA, RF (where available)
    df3   = _read_ff(p3)     # Mkt-RF, SMB, HML, RF
    dfmom = _read_ff(pmom)   # Mom
    ff = df5.join(df3, how="outer", rsuffix="_3").join(dfmom, how="outer")
    keep_order = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF", "Mom"]
    rename_map = {
        "Mkt-RF": "MKT",
        "SMB":    "SMB",
        "HML":    "HML",
        "RMW":    "RMW",
        "CMA":    "CMA",
        "RF":     "RF",
        "Mom":    "MOM",}
    keep = [c for c in keep_order if c in ff.columns]
    ff = ff[keep].rename(columns=rename_map).sort_index()
    return ff


# --------------------------------------------------------------------------------------
# Gridsearch (unrestricted or restricted via r_grid entries)
# --------------------------------------------------------------------------------------

def gridsearch_actual(y, px, r_grid, CONFIG, seed=None):
    
    """
    Fully backward-compatible:
    - Keeps h_z as an explicit input (via CONFIG).
    - Keeps CUSUM overlay & ex-ante predictions.
    - Adds optional restricted dispatch (if an r in r_grid has 'restrictions').
    - Accepts optional batch_grid; defaults to the original windows if None.
    """

    dt = CONFIG["dt"]
    n_iters = CONFIG["n_iters"]
    batch_grid = CONFIG["batch_grid"]
    h_z = CONFIG["h_z"]
    verbose = CONFIG["verbose"]
    display = CONFIG["display"]

    leaderboard = []
    details = []
    success = 0
    combo_list = list(itertools.product(r_grid, batch_grid))
    combo_total = len(combo_list)
    print(f"combo_total= {combo_total}")

    for combo_idx, (r, b) in enumerate(combo_list, 1):
        
        # print(f"\n\n{'='*72}\n({combo_idx}/{combo_total}) Params: {r}; {b}\n{'='*72}")

        label = r.get("model_name", "unrestricted")
        print(f"\n\n{'='*72}\n({combo_idx}/{combo_total}) {label} | Params: {r}; {b}\n{'='*72}")

        zhat_all, xhat_all, elbo_all = [], [], []
        zhat_cusum_all = []         # stitched CUSUM
        zhat_pred_all = []          # ex-ante predictions stitched
        pred_idx = []    # their absolute indices
        retained_idx = []

        T = len(y)
        t0 = 0
        train_window = b["train_window"]
        overlap = b["overlap_window"]

        while t0 < T:
            end_t = min(t0 + train_window, T)
            y_tr = y[t0:end_t]
            y_tr = y_tr.reshape(-1, 1) if y_tr.ndim == 1 else y_tr

            # Fit model on batch (unrestricted vs restricted)
            params = dict(
                n_regimes=r["n_regimes"],
                dim_latent=r["dim_latent"],
                single_subspace=r.get("single_subspace", True),)
            if r.get("restrictions") is not None:
                R = r["restrictions"]
                C = R["C"]    # shape (N_obs, D)
                d = R["d"]    # shape (N_obs,)
                bpat = R.get("b_pattern")      # e.g. ["mu_form"]*D
                xhat, zhat, elbo, q, model = fit_rSLDS_restricted(
                    y_tr, params, C, d, n_iter_em=n_iters, seed=seed, b_pattern=bpat, enforce_diag_A=True,
                    C_mask=R.get("C_mask", None), d_mask=R.get("d_mask", None))
            
            else:
                xhat, zhat, elbo, q, model = fit_rSLDS(y_tr, params, n_iter_em=n_iters, seed=seed)

            elbo_all.append(elbo)

            # boundary posterior for discrete-only seeding
            mask_tr = np.ones_like(y_tr, dtype=bool) 
            gamma_tr, *_ = model.expected_states(xhat, y_tr, mask=mask_tr)
            gamma_T = gamma_tr[-1]  # (K,)


            # NO CUSUM FOR ACTUAL DATA
            zhat_cus = zhat.copy()
            '''
            # CUSUM overlay on the primary series ONLY (y column)
            y_cus = y_tr[:, 0] if y_tr.ndim == 2 else y_tr      # (T,)
            assert y_cus.ndim == 1, f"y_cus must be 1D, got {y_cus.shape}"
            zhat_cus = cusum_overlay(px.iloc[t0:end_t], y_cus, xhat, model, h_z)
            assert zhat_cus is not None, "cusum_overlay returned None"
            assert len(zhat_cus) == len(zhat), f"CUSUM length {len(zhat_cus)} != {len(zhat)}"
            '''
            
            
            # ---- Stitch (trim overlap for all streams except first batch)
            if t0 == 0:
                zhat_all.append(zhat)
                xhat_all.append(xhat)
                zhat_cusum_all.append(zhat_cus)
                retained_idx.extend(range(t0, end_t))
            else:
                overlap_eff = min(overlap, end_t - t0)
                zhat_all.append(zhat[overlap_eff:])
                xhat_all.append(xhat[overlap_eff:])
                zhat_cusum_all.append(zhat_cus[overlap_eff:])
                retained_idx.extend(range(t0 + overlap_eff, end_t))

            # ---- Ex-ante (discrete-only seed) on next overlap window
            if (end_t < T) and (overlap > 0):
                t2_end = min(end_t + overlap, T)
                if t2_end > end_t:
                    y_te = y[end_t:t2_end]
                    y_te = y_te.reshape(-1, 1) if y_te.ndim == 1 else y_te
            
                    # save + seed discrete init only
                    pi0_orig = getattr(model.init_state_distn, "pi", None)
                    model.init_state_distn.pi = gamma_T / max(gamma_T.sum(), 1e-12)
            
                    # ssm posterior (no refit)
                    T2   = len(y_te)
                    Fs   = getattr(model.emissions, "Fs", [])
                    D_in = Fs[0].shape[1] if len(Fs) else 0
                    inputs2 = np.zeros((T2, D_in))
                    mask2 = np.ones_like(y_te, dtype=bool)              # (T2, N)
                    q_te = model._make_variational_posterior(
                        variational_posterior="structured_meanfield",
                        datas=[y_te], inputs=[inputs2], masks=[mask2], tags=[None], method="smf")
                    xhat_te = q_te.mean_continuous_states[0]
                    zhat_te = model.most_likely_states(xhat_te, y_te)
            
                    # restore
                    if pi0_orig is not None:
                        model.init_state_distn.pi = pi0_orig
            
                    zhat_pred_all.append(zhat_te)
                    pred_idx.extend(range(end_t, t2_end))

            success += 1
            if end_t == T:
                break
            t0 += train_window - overlap

        if not zhat_all:
            continue

        # ---- Stitch + mask to data length
        zhat_all = np.concatenate(zhat_all)
        zhat_cusum_all = np.concatenate(zhat_cusum_all)
        xhat_all = np.concatenate(xhat_all)
        retained_idx = np.array(retained_idx)

        valid_mask = retained_idx < len(y)
        retained_idx = retained_idx[valid_mask]
        zhat_all = zhat_all[valid_mask]
        xhat_all = xhat_all[valid_mask]
        zhat_cusum_all = zhat_cusum_all[valid_mask]

        y_valid = y[retained_idx]
        y_valid = y_valid.reshape(-1, 1) if y_valid.ndim == 1 else y_valid
        px_valid = px.iloc[retained_idx]

        # ---- CPLL (smoothed) and rough upper bound
        mask_valid = np.ones_like(y_valid, dtype=bool)      # (T,N)
        gamma_valid, *_ = model.expected_states(xhat_all, y_valid, mask=mask_valid)
        cpll = compute_smoothed_cpll(model, xhat_all, y_valid, gamma_valid)
        if y_valid.shape[1] == 1:
            v = float(np.var(y_valid[:, 0], ddof=1))
            v = max(v, 1e-12)
            entropy = 0.5 * np.log(2*np.pi*np.e*v)
        else:
            Sigma = np.cov(y_valid.T, ddof=1)
            sign, logdet = np.linalg.slogdet(Sigma)
            logdet = logdet if sign > 0 else np.log(1e-12)
            N = y_valid.shape[1]
            entropy = 0.5 * (N*np.log(2*np.pi*np.e) + logdet)        
        max_cpll = -y_valid.shape[0] * entropy

        # ---- Plot overlay when lengths match (kept; not passed to evaluator)
        overlays = {"CUSUM": zhat_cusum_all} if len(zhat_cusum_all) == len(zhat_all) else None
        
        # ---- Evaluation (now pass cpll & max_cpll)
        summary = evaluate_rSLDS_actual(
            y_valid, px_valid, zhat_cusum_all, xhat_all, elbo_all, model, 
            cpll, max_cpll, dt, display=display)
        
        # ---- CUSUM performance on the same stitched window
        c_rel_cus, c_str_cus, c_ben_cus, *_ = compute_score(px_valid, zhat_cusum_all, dt)
        summary["cagr_rel_cusum"] = c_rel_cus
        summary["cagr_strat_cusum"] = c_str_cus
        summary["cagr_bench_cusum"] = c_ben_cus

        # ---- Ex-ante (live) performance on predicted overlap windows
        if len(zhat_pred_all):
            zhat_pred_all = np.concatenate(zhat_pred_all)
            pred_idx = np.array(pred_idx)
            if len(zhat_pred_all) == len(pred_idx):
                px_pred = px.iloc[pred_idx]
                c_rel_pred, c_str_pred, c_bench_pred, *_ = compute_score(px_pred, zhat_pred_all, dt)
                summary["cagr_rel_ex_ante"] = c_rel_pred
                summary["cagr_strat_ex_ante"] = c_str_pred
                summary["cagr_bench_ex_ante"] = c_bench_pred

        # --- Niceness features
        ncpll = float(cpll / max(max_cpll, 1e-12))
        agree_cusum = float(np.mean(zhat_all[:len(zhat_cusum_all)] == zhat_cusum_all)) if len(zhat_cusum_all)==len(zhat_all) else np.nan
        Lbar = summary.get("avg_inferred_regime_length", np.nan)
        mode_usage = summary.get("mode_usage", None)
        if isinstance(mode_usage, dict):
            p_dom = float(max(mode_usage.values()))
        elif hasattr(mode_usage, "__iter__"):
            p_dom = float(np.max(mode_usage))
        else:
            p_dom = np.nan
        dELBO = summary.get("elbo_delta (last run)", np.nan)
        c_rel = summary.get("cagr_rel", np.nan)
        c_rel_ex = summary.get("cagr_rel_ex_ante", np.nan)
        
        # --- Rule
        nice = (
            (Lbar >= 20) and
            (agree_cusum >= 0.70 if not np.isnan(agree_cusum) else True) and
            (ncpll >= 0.60) and
            (dELBO >= 0) and
            (c_rel >= 0) and
            (np.isnan(c_rel_ex) or c_rel_ex >= 0)
        )
        difficult = (
            (Lbar <= 5) or
            (agree_cusum <= 0.55 if not np.isnan(agree_cusum) else False) or
            (ncpll <= 0.40) or
            (p_dom >= 0.90) or
            ((c_rel < 0) and (not np.isnan(c_rel_ex) and c_rel_ex < 0))
        )
        tag = "nice" if nice else ("difficult" if difficult else "borderline")
        
        summary.update({
            "ncpll": ncpll,
            "agree_cusum": agree_cusum,
            "p_dom": p_dom,
            "niceness_tag": tag
        })

        leaderboard.append(dict(score=summary["cagr_rel"], params=(r, b), summary=summary))

        details.append(dict(
        zhat=zhat_all.copy(),
        zhat_cusum=zhat_cusum_all.copy(),
        retained_idx=np.array(retained_idx, int).copy(),
        px_index=px_valid.index.copy(),
        params=(r, b),
        summary=summary))
    
    print("\nfits succeeded:", success)
    if leaderboard:
        sorted_leaderboard = sorted(leaderboard, key=lambda d: d["score"], reverse=True)
        print("\nLEADERBOARD:")
        rows = []
        for i, entry in enumerate(sorted_leaderboard, 1):
            r, b = entry["params"]
            s = entry["summary"]
            r_clean = {k: v for k, v in r.items() if k != "restrictions"}
            row = {
                "rank": i,
                "score": entry["score"],
                **r_clean, **b,
                "avg_inferred_regime_length": s.get("avg_inferred_regime_length"),
                "elbo_start (last run)": s.get("elbo_start (last run)"),
                "elbo_end (last run)": s.get("elbo_end (last run)"),
                "elbo_delta (last run)": s.get("elbo_delta (last run)"),
                "mode_usage": s.get("mode_usage"),
                "cagr_rel": s.get("cagr_rel"),
                "cagr_strat": s.get("cagr_strat"),
                "cagr_bench": s.get("cagr_bench"),
            }
            
            if "cagr_rel_cusum" in s:
                row.update({
                    "cagr_rel_cusum": s["cagr_rel_cusum"],
                    "cagr_strat_cusum": s["cagr_strat_cusum"],
                    "cagr_bench_cusum": s["cagr_bench_cusum"],
                })
                
            if "cagr_rel_ex_ante" in s:
                row.update({
                    "cagr_rel_ex_ante": s["cagr_rel_ex_ante"],
                    "cagr_strat_ex_ante": s["cagr_strat_ex_ante"],
                    "cagr_bench_ex_ante": s["cagr_bench_ex_ante"],
                })
    
            row.update({
                "ncpll": s.get("ncpll"),
                "agree_cusum": s.get("agree_cusum"),
                "p_dom": s.get("p_dom"),
                "niceness_tag": s.get("niceness_tag"),
            })
        
            rows.append(row)
        
        df = pd.DataFrame(rows)

        print_cols_all = ['rank','score','n_regimes','dim_latent','single_subspace','model_name']
        print_cols = [c for c in print_cols_all if c in df.columns]
        print(df[print_cols].to_string(index=False))

        
        return df, details

    
# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _intersect_indexes(series_list):
    idx = series_list[0].index
    for s in series_list[1:]:
        idx = idx.intersection(s.index)
    return idx

def _build_restrictions(C_type, Y_obs, base_channels, D):
    
    """
    Build emission constraints (C,d) in the row order of `base_channels`.

    Returns a dict with:
      - "C":      (N, D) array of fixed values (zeros where free to learn)
      - "d":      (N,)   array of fixed values (zeros where free to learn)
      - "C_mask": (N, D) mask with 1 = learn / keep-updated, 0 = clamp to C
      - "d_mask": (N,)   mask with 1 = learn / keep-updated, 0 = clamp to d
      - "b_pattern": list[str] of length D (e.g., ["mu_form"]*D)

    Conventions by model:
      • fund1:        y = v + g                                 (D=2)              fully fixed
      • fund1_vix:    y = v + g;  h = h_latent                  (D=3)              fully fixed
      • fund2:        y = v + g;  observe g                     (D=2)              fully fixed
      • fund2_vix:    y = v + g;  observe g;  h = h_latent      (D=3)              fully fixed
      • fund3:        y = v + g;  observe v,g                   (D=2)              fully fixed
      • fund3_vix:    y = v + g;  observe v,g;  h = h_latent    (D=3)              fully fixed
      • factor1:      y loads on all D latents (learned)        (D>=1)             masks only
      • factor1_vix:  y loads on first D-1 (learned); h on last (D>=2)             mixed fixed/learned
      • factor2:      y = alpha + beta' f; proxies observe own latents (D=m)       fixed (beta via OLS, alpha in d[0])
    """

    N = len(base_channels)
    base_channels = list(base_channels)

    def idx(name):
        try:
            return base_channels.index(name)
        except ValueError:
            return None

    # ---------------------- FUNDAMENTAL MODELS (fully fixed) ----------------------
    
    if C_type == "fund1":
        # channels: ["y"]; latents: [v,g] -> D must be 2
        assert base_channels == ["y"], "fund1 requires channels=['y']"
        assert D == 2, "fund1 requires D=2"
        C = np.array([[1.0, 1.0]], dtype=float)
        d = np.zeros((1,), dtype=float)
        return {"C": C, "d": d, "b_pattern": ["mu_form"] * D}

    if C_type == "fund1_vix":
        # channels: ["y","h"]; latents: [v,g,h] -> D must be 3
        assert set(base_channels) == {"y", "h"} and N == 2, "fund1_vix requires channels=['y','h']"
        assert D == 3, "fund1_vix requires D=3"
        iy, ih = idx("y"), idx("h")
        C = np.zeros((N, D))
        C[iy, :2] = [1.0, 1.0]  # y = v + g
        C[ih, 2]  = 1.0         # h = h_latent
        d = np.zeros((N,), float)
        return {"C": C, "d": d, "b_pattern": ["mu_form"] * D}

    if C_type == "fund2":
        # channels: ["y","g"]; latents: [v,g] -> D must be 2
        assert set(base_channels) == {"y", "g"} and N == 2, "fund2 requires channels=['y','g']"
        assert D == 2, "fund2 requires D=2"
        iy, ig = idx("y"), idx("g")
        C = np.zeros((N, D))
        C[iy, :2] = [1.0, 1.0]  # y = v + g
        C[ig, 1]  = 1.0         # observe g
        d = np.zeros((N,), float)
        return {"C": C, "d": d, "b_pattern": ["mu_form"] * D}

    if C_type == "fund2_vix":
        # channels: ["y","g","h"]; latents: [v,g,h] -> D must be 3
        assert set(base_channels) == {"y", "g", "h"} and N == 3, "fund2_vix requires channels=['y','g','h']"
        assert D == 3, "fund2_vix requires D=3"
        iy, ig, ih = idx("y"), idx("g"), idx("h")
        C = np.zeros((N, D))
        C[iy, :2] = [1.0, 1.0]  # y = v + g
        C[ig, 1]  = 1.0         # observe g
        C[ih, 2]  = 1.0         # h = h_latent
        d = np.zeros((N,), float)
        return {"C": C, "d": d, "b_pattern": ["mu_form"] * D}

    if C_type == "fund3":
        # channels: ["y","v","g"]; latents: [v,g] -> D must be 2
        assert set(base_channels) == {"y", "v", "g"} and N == 3, "fund3 requires channels=['y','v','g']"
        assert D == 2, "fund3 requires D=2"
        iy, iv, ig = idx("y"), idx("v"), idx("g")
        C = np.zeros((N, D))
        C[iy, :2] = [1.0, 1.0]  # y = v + g
        C[iv, 0]  = 1.0         # observe v
        C[ig, 1]  = 1.0         # observe g
        d = np.zeros((N,), float)
        return {"C": C, "d": d, "b_pattern": ["mu_form"] * D}

    if C_type == "fund3_vix":
        # channels: ["y","v","g","h"]; latents: [v,g,h] -> D must be 3
        assert set(base_channels) == {"y", "v", "g", "h"} and N == 4, "fund3_vix requires channels=['y','v','g','h']"
        assert D == 3, "fund3_vix requires D=3"
        iy, iv, ig, ih = idx("y"), idx("v"), idx("g"), idx("h")
        C = np.zeros((N, D))
        C[iy, :2] = [1.0, 1.0]  # y = v + g
        C[iv, 0]  = 1.0         # observe v
        C[ig, 1]  = 1.0         # observe g
        C[ih, 2]  = 1.0         # h = h_latent
        d = np.zeros((N,), float)
        return {"C": C, "d": d, "b_pattern": ["mu_form"] * D}

    # ---------------------- FACTOR MODELS ----------------------
    
    if C_type == "factor1":
        # channels: ["y"]; y loads on all D latents (all learned); d=0
        assert base_channels == ["y"], "factor1 requires channels=['y']"
        assert D >= 1, "factor1 requires D>=1"
        C_fix = np.zeros((1, D), dtype=float)
        d_fix = np.zeros((1,), dtype=float)
        C_mask = np.ones((1, D), dtype=float)   # learn all y loadings
        d_mask = np.zeros((1,), dtype=float)    # keep d=0
        return {
            "C": C_fix, "d": d_fix,
            "C_mask": C_mask, "d_mask": d_mask,
            "b_pattern": ["mu_form"] * D}

    if C_type == "factor1_vix":
        # channels: ["y","h"]; y learns first D-1; h fixed to last latent; d=0
        assert set(base_channels) == {"y", "h"} and N == 2, "factor1_vix requires channels=['y','h']"
        assert D >= 2, "factor1_vix requires D>=2"
        iy, ih = idx("y"), idx("h")
        C_fix = np.zeros((N, D), dtype=float)
        d_fix = np.zeros((N,), dtype=float)

        # h observes ONLY the last latent
        C_fix[ih, D-1] = 1.0

        # Masks
        C_mask = np.zeros((N, D), dtype=float)
        C_mask[iy, :D-1] = 1.0      # learn y's first D-1 loadings
        # y's loading on last latent is clamped to 0 via C_fix (mask=0)
        # h row fully fixed (mask=0)
        d_mask = np.zeros((N,), dtype=float)  # keep d=0

        return {
            "C": C_fix, "d": d_fix,
            "C_mask": C_mask, "d_mask": d_mask,
            "b_pattern": ["mu_form"] * D}

    if C_type == "factor2":
        # channels: ["y", f1, ..., fm]; D must be m
        assert base_channels[0] == "y", "factor2 requires first channel to be 'y'"
        m = len(base_channels) - 1
        assert D == m, f"factor2 requires D=m (#factors). Got D={D}, m={m}"
        if m == 0:
            # degenerate case (no factors)
            C = np.zeros((1, 0), float)
            d = np.zeros((1,), float)
            return {"C": C, "d": d, "b_pattern": ["mu_form"]}

        # OLS: y_t = alpha + beta' f_t + eps_t  (alpha stored in d[0])
        X = Y_obs[:, 1:1 + m]                 # (T, m) proxy factors
        y = Y_obs[:, 0]                       # (T,)
        beta = np.linalg.lstsq(X, y, rcond=None)[0].reshape(m)  # (m,)
        alpha = float((y - X @ beta).mean())

        # Build C: first row = beta'; next m rows = I_m (each proxy observes its latent)
        C = np.zeros((1 + m, m), float)
        C[0, :] = beta
        for j in range(m):
            C[1 + j, j] = 1.0

        # Build d: alpha for y row; zeros for proxies
        d = np.zeros((1 + m,), float)
        d[0] = alpha

        return {"C": C, "d": d, "b_pattern": ["mu_form"] * max(m, 1)}

    # ---------------------- Fallback ----------------------
    C = np.zeros((len(base_channels), max(D, 1)), float)
    d = np.zeros((len(base_channels),), float)
    
    return {"C": C, "d": d, "b_pattern": ["mu_form"] * C.shape[1]}

def _params_key(r, b):
    """Stable string key for (r, b) without dumping big matrices."""
    r0 = dict(r)
    r0.pop("restrictions", None)  # keep CSV light
    return json.dumps({
        "n_regimes": r0.get("n_regimes"),
        "dim_latent": r0.get("dim_latent"),
        "single_subspace": r0.get("single_subspace", True),
        "model_name": r0.get("model_name", "unrestricted"),
        "train_window": b.get("train_window"),
        "overlap_window": b.get("overlap_window"),
    }, sort_keys=True)

def _append_segments(path, security, config_label, details):
    """
    Append stitched label rows to CSV without re-reading the whole file.
    Keeps your schema: ["security","config","date","t","z"].
    """
    if not details:
        return
    frames = []
    for d in details:
        idx = pd.DatetimeIndex(d["px_index"])
        frames.append(pd.DataFrame({
            "security": security,
            "config":   config_label,
            "date":     idx,
            "t":        np.arange(len(idx), dtype=int),
            "z":        np.asarray(d["zhat_cusum"], dtype=int),
        }))
    out = pd.concat(frames, ignore_index=True)
    # append-only; write header if file missing/empty
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    out.to_csv(path, mode="a", header=write_header, index=False)


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------

def pipeline_actual(securities, CONFIG, filename):
    """
    - Imports data via import_data (VIX from IDX_PX).
    - Unrestricted path -> original 4 cases (|Ω|≈0..3), passing h_z through.
    - Restricted fundamental models ({fund1, fund1_vix, fund2, fund2_vix}).
    - Restricted factor models (factor1, factor1_vix, factor2, factor2_vix).
      * factor1: learned C with latent size m in CONFIG['factor1_m_grid'].
      * factor2: OLS for row0 (y on proxies), identities for proxy rows; logs R^2 into r dict.
    """
    n_iters = CONFIG["n_iters"]
    h_z = CONFIG["h_z"]
    verbose = CONFIG["verbose"]
    display = CONFIG["display"]
    run_unrestricted = CONFIG["run_unrestricted"]
    run_restricted = CONFIG["run_restricted"]

    # store results
    Z_labels, Segs, Idx = {}, {}, {}
    def _stash_labels(security, details):
        if not details:
            return
        first = details[0]
        z_seq = np.asarray(first["zhat_cusum"], int)
        idx_dt = pd.DatetimeIndex(first["px_index"])
        Z_labels[security] = pd.Series(z_seq, index=idx_dt, name="z")
        cps = np.flatnonzero(z_seq[1:] != z_seq[:-1]) + 1
        Segs[security] = [0] + cps.tolist() + [len(z_seq)]
        Idx[security] = idx_dt
    
    # output CSV
    gridsearch_csv = "gridsearch_results.csv"
    columns = [
        "security", "config", "rank", "score", "dt",
        "n_regimes", "dim_latent", "single_subspace",
        "train_window", "overlap_window", "avg_inferred_regime_length",
        "elbo_start (last run)", "elbo_end (last run)", "elbo_delta (last run)",
        "mode_usage", "cagr_rel", "cagr_strat", "cagr_bench",
        "ncpll", "agree_cusum", "p_dom", "niceness_tag"]
    
    pd.DataFrame(columns=columns).to_csv(gridsearch_csv, index=False)

    # segments store (for stitched labels)
    segments_csv = "gridsearch_segments.csv"
    pd.DataFrame(columns=["security","config","date","t","z"]).to_csv(segments_csv, index=False)

    # import data
    px_all, eps_all, pe_all, ser_vix = import_data(filename)
    ff_df = import_factors()

    # helper to append results
    def _append(security, cfg, df_res):
        if df_res is None or len(df_res) == 0:
            return
    
        out = df_res.copy()
        if out.dropna(axis=1, how="all").empty:
            return
    
        # add identifying columns
        out.insert(0, "security", security)
        out.insert(1, "config",   cfg)
        dt_val = CONFIG["dt"]
        dt_str = f"1/{int(round(1.0/dt_val))}" if dt_val > 0 else str(dt_val)
        out.insert(4, "dt", dt_str)
    
        # read current CSV (has the schema you wrote initially)
        cur = pd.read_csv(gridsearch_csv)
    
        # ensure identical column order/schema; fill any missing with NaN
        out = out.reindex(columns=cur.columns, fill_value=np.nan)
    
        # avoid concatenating with an empty frame (causes the FutureWarning)
        if cur.empty:
            out.to_csv(gridsearch_csv, index=False)
        else:
            pd.concat([cur, out], ignore_index=True).to_csv(gridsearch_csv, index=False)

    # canonical series helper (per security)
    for security in securities:

        print(f"\n\n{'='*72}\n{security}\n{'='*72}")

        # remove nans
        ser_px = px_all[security].dropna()
        ser_eps = eps_all[security].dropna()
        ser_eps = ser_eps.where(ser_eps > 0).dropna()                         # EPS > 0
        ser_pe  = (ser_px / ser_eps).where(lambda s: s > 0).dropna()          # PE > 0
        ser_vix = ser_vix.dropna()
                   
        # diffs on the common index
        common_idx = ser_px.index.intersection(ser_eps.index).intersection(ser_pe.index).intersection(ser_vix.index)
        d_px  = np.log(ser_px.loc[common_idx]).diff().dropna()
        d_eps = np.log(ser_eps.loc[common_idx]).diff().dropna()
        d_pe  = np.log(ser_pe.loc[common_idx]).diff().dropna()
        d_vix = np.log(ser_vix.loc[common_idx]).diff().dropna()

        # intersection after differencing
        ci = d_px.index.intersection(d_eps.index).intersection(d_pe.index).intersection(d_vix.index)
        
        # canonical series
        y = d_px.loc[ci].values.reshape(-1, 1)   # log returns
        g = d_eps.loc[ci].values.reshape(-1, 1)  # diff log eps (growth)
        v = d_pe.loc[ci].values.reshape(-1, 1)   # log PE change
        h = d_vix.loc[ci].values.reshape(-1, 1)  # log VIX change
        px = ser_px.loc[ci]                       # benchmark price, aligned to diffs

        log_px  = np.log(ser_px)          # assume prices > 0; add .where(>0) if needed
        log_eps = np.log(ser_eps)         # already EPS > 0
        log_pe  = np.log(ser_pe)          # already PE > 0
        log_vix = np.log(ser_vix)         # VIX is > 0; add mask if needed

        series_by_key = {
            "y": log_px.diff().dropna(),  # log returns 
            "g": log_eps.diff().dropna(), # diff log eps (eps log growth)
            "v": log_pe.diff().dropna(),  # log PE change
            "h": log_vix.diff().dropna(), # log VIX change
            "mkt": ff_df["MKT"],
            "smb": ff_df["SMB"],
            "hml": ff_df["HML"],
            "rmw": ff_df["RMW"],
            "cma": ff_df["CMA"],
            "mom": ff_df["MOM"],}

        K_grid = CONFIG["K_grid"]
        
        # Unrestricted models
        if run_unrestricted:
            for case in CONFIG.get("unrestricted_models", []):
                chans = case["channels"]
                D_list = case["dim_latent"]  # list
                needed = [series_by_key[k] for k in chans]
                common_idx = _intersect_indexes(needed)
                if common_idx.empty:
                    print(f"[WARN] {security} {case['label']}: no overlap across requested channels; skipping.")
                    continue
        
                y_cols = [series_by_key[k].loc[common_idx].values.reshape(-1, 1) for k in chans]
                Y_obs = np.concatenate(y_cols, axis=1)
                px_case = ser_px.loc[common_idx]
        
                r_grid = [{"n_regimes": K, "dim_latent": D, "single_subspace": True}
                          for D in D_list for K in CONFIG["K_grid"]]
        
                cfg = case["label"]
                df, details = gridsearch_actual(y=Y_obs, px=px_case, r_grid=r_grid, CONFIG=CONFIG)
                _append(security, cfg, df)
                _stash_labels(security, details)
                _append_segments(segments_csv, security, cfg, details)

        # Restricted models
        if run_restricted:
         
            # --- Fundamental models ---
            for model_def in CONFIG.get("restricted_models", []):
                if model_def["C_type"] not in {"fund1","fund1_vix","fund2","fund2_vix","fund3","fund3_vix"}:
                    continue

                base_channels = list(model_def["channels"])  # exactly as provided
                # build common index & Y_obs
                needed = [series_by_key[k].dropna() for k in base_channels]
                common_idx = needed[0].index
                for s in needed[1:]:
                    common_idx = common_idx.intersection(s.index)
                if common_idx.empty:
                    print(f"[WARN] {security} {model_def['label']}: no overlap; skipping.")
                    continue
        
                Y_obs = np.concatenate([series_by_key[k].loc[common_idx].values.reshape(-1,1)
                                        for k in base_channels], axis=1)
                px_case = ser_px.loc[common_idx]
                D = int(model_def["dim_latent"][0])  # you set it in CONFIG

                restrictions = _build_restrictions(model_def["C_type"], Y_obs, base_channels, D)
                # fundamental models are fully fixed -> no masks needed

                r_grid = [{"n_regimes": K, "dim_latent": D, "single_subspace": True,
                           "restrictions": restrictions, "model_name": model_def["label"]}
                          for K in CONFIG["K_grid"]]
        
                df, details = gridsearch_actual(Y_obs, px_case, r_grid, CONFIG)
                _append(security, model_def["label"], df)
                _stash_labels(security, details)
                _append_segments(segments_csv, security, model_def["label"], details)
        
            # --- Factor models ---
            for model_def in CONFIG.get("restricted_models", []):
                ctype = model_def["C_type"]
                if ctype not in {"factor1","factor1_vix","factor2"}:
                    continue

                base_channels = list(model_def["channels"])
                missing = [k for k in base_channels if k not in series_by_key]
                if missing:
                    print(f"[WARN] {security} {model_def['label']}: missing series {missing}; skipping.")
                    continue

                needed = [series_by_key[k].dropna() for k in base_channels]
                common_idx = needed[0].index
                for s in needed[1:]:
                    common_idx = common_idx.intersection(s.index)
                if common_idx.empty:
                    print(f"[WARN] {security} {model_def['label']}: no overlap; skipping.")
                    continue

                Y_obs = np.concatenate([series_by_key[k].loc[common_idx].values.reshape(-1,1)
                                        for k in base_channels], axis=1)
                px_case = ser_px.loc[common_idx]

                # D: user-specified for factor1 / factor1_vix, implied for factor2
                if ctype in {"factor1", "factor1_vix"}:
                    D = int(model_def["dim_latent"][0])
                elif ctype == "factor2":
                    D = len(base_channels) - 1

                restrictions = _build_restrictions(ctype, Y_obs, base_channels, D)

                r_grid = [{"n_regimes": K, "dim_latent": D, "single_subspace": True,
                           "restrictions": restrictions, "model_name": model_def["label"]}
                          for K in CONFIG["K_grid"]]

                df, details = gridsearch_actual(Y_obs, px_case, r_grid, CONFIG)
                _append(security, model_def["label"], df)
                _stash_labels(security, details)
                _append_segments(segments_csv, security, model_def["label"], details)

    print("Gridsearch completed.\n")

    return {"Z_labels": Z_labels, "segments": Segs, "index": Idx,
            "meta": {"dt": CONFIG["dt"], "K_grid": CONFIG["K_grid"], "batch_grid": CONFIG["batch_grid"]}}

