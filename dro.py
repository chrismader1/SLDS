import numpy as np, hashlib
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy import stats
from scipy.optimize import linear_sum_assignment
from gridsearch import *

# ---------------------------------------------------------------
# Wasserstein helper functions
# ---------------------------------------------------------------

def _rng_from_params(params):
    import numpy as _np
    seed = None if params is None else params.get("seed", None)
    return _np.random.default_rng(seed)

def _sqrtm_psd(A, eps=1e-12):
    """Symmetric PSD principal square root via eigen-decomposition."""
    vals, vecs = np.linalg.eigh(0.5*(A + A.T))
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals + eps)) @ vecs.T

def w2_empirical_uniform_exact(X, Y):
    """
    Exact W2 between two uniform empirical measures with the same number of points.
    Returns W2 (not squared). Uses Hungarian assignment on squared Euclidean costs.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2, "X and Y must have the same dimension"
    assert n == m,  "Uniform empirical W2 requires equal sample sizes"
    # cost matrix C_{ij} = ||x_i - y_j||^2
    C = ((X[:, None, :] - Y[None, :, :])**2).sum(axis=2)
    r, c = linear_sum_assignment(C)
    return float(np.sqrt(C[r, c].mean()))

def wasserstein2_gaussian(mu1, Sigma1, mu2, Sigma2, eps=1e-12):
    """
    Gelbrich formula: W2^2(N(mu1,S1), N(mu2,S2)) =
      ||mu1-mu2||^2 + tr(S1 + S2 - 2 (S2^{1/2} S1 S2^{1/2})^{1/2})
    Returns W2 (not squared).
    """
    dmu2 = float(np.dot(mu1 - mu2, mu1 - mu2))
    S2h = _sqrtm_psd(Sigma2, eps=eps)
    mid = S2h @ Sigma1 @ S2h
    midh = _sqrtm_psd(mid, eps=eps)
    trpart = float(np.trace(Sigma1 + Sigma2 - 2.0 * midh))
    w2_sq = max(dmu2 + trpart, 0.0)
    return float(np.sqrt(w2_sq))

def sliced_w2_empirical(X, Y, n_proj=64, rng=None):
    """
    Approximate multivariate W2 by averaging 1D W2 across random directions.
    X, Y: (n,d) samples with same n preferred (bootstrap uses same n).
    Returns W2 (not squared).
    """
    rng = _rng_from_params({}) if rng is None else rng
    X = np.asarray(X); Y = np.asarray(Y)
    n, d = X.shape
    assert Y.shape[1] == d
    m = Y.shape[0]
    w2_sq = 0.0
    for _ in range(int(n_proj)):
        u = rng.normal(size=d); u /= max(np.linalg.norm(u), 1e-12)
        x = np.sort(X @ u)
        y = np.sort(Y @ u)
        if m != n:
            # match quantiles if sizes differ
            q = np.linspace(0, 1, min(n, m), endpoint=True)
            xq = np.quantile(x, q); yq = np.quantile(y, q)
            diff = xq - yq
        else:
            diff = x - y
        w2_sq += float(np.mean(diff*diff))
    w2_sq /= float(n_proj)
    return float(np.sqrt(max(w2_sq, 0.0)))

def _mvnrnd_psd(mu, Sigma, n, rng, eps=1e-9):
    """Draw n samples ~ N(mu, Sigma) with PSD projection; avoids SVD path."""
    mu = np.asarray(mu, float); d = mu.size
    S  = 0.5*(Sigma + Sigma.T)
    vals, vecs = np.linalg.eigh(S)
    L = (vecs * np.sqrt(np.clip(vals, eps, None))) @ vecs.T
    Z = rng.normal(size=(n, d))
    return mu + Z @ L.T

# ---------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------

def compute_delta(kappa, mu_est, Sigma=None, R=None, params=None):
    """
    Selectable Wasserstein radius δ. Defaults to δ = κ‖μ‖₂ (backward compatible).
    Methods (set via params['delta_method']):
      - 'kappa_l2'       : δ = κ‖μ‖₂
      - 'kappa_rate'     : δ = κ · σ̄ · sqrt(d/n),  σ̄² = tr(Σ)/d
      - 'fixed'          : δ = params['delta']
      - 'bound_ek'       : Esfahani–Kuhn bound δ_n(α) (no κ)
      - 'bootstrap_np'       : Nonparametric bootstrap quantile of W2( P̂0, P̂0^(b) )
      - 'bootstrap_gaussian': Parametric Gaussian bootstrap using Gelbrich W2
    """
    if not isinstance(params, dict) or "delta_method" not in params:
        raise ValueError("A 'delta_method' must be explicitly specified in the params dictionary.")
    method = params["delta_method"]
    
    kappa  = float(kappa)

    # ----- κ-based rules
    if method == "kappa_rate":
        d     = int(np.size(mu_est))
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n_eff = int((params or {}).get("n_ref", n_obs))
        sbar  = float(np.sqrt(np.trace(Sigma) / max(d, 1))) if Sigma is not None else 0.0
        return kappa * sbar * np.sqrt(d / max(n_eff, 1))

    if method == "fixed":
        return float((params or {}).get("delta", 0.0))
    
    if method == "kappa_l2":
        return kappa * float(np.linalg.norm(mu_est, 2))

    # ----- non-κ rules (ignore kappa)
    if method == "bound_ek":
        alpha = float((params or {}).get("alpha", 0.05))
        c1    = float((params or {}).get("c1", 3.0))
        c2    = float((params or {}).get("c2", 1.0))
        a     = float((params or {}).get("a", 2.0))
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n     = int((params or {}).get("n_ref", n_obs))
        d     = int(np.size(mu_est))
        num   = np.log(c1 / max(alpha, 1e-12))
        den   = c2 * max(n, 1)
        thresh= num / max(c2, 1e-12)
        base  = max(num / max(den, 1e-12), 1e-12)
        exp   = 1.0 / max(max(d, 2), 1) if n >= thresh else 1.0 / max(a, 1e-12)
        return float(base ** exp)

    if method == "bootstrap_np":
        assert R is not None, "bootstrap_np needs raw sample matrix R."
        alpha  = float((params or {}).get("alpha", 0.05))
        B      = int((params or {}).get("B", 200))
        rng    = _rng_from_params(params or {})
        n_src  = int(R.shape[0])
        # Bootstrap sample size must equal source data size for the exact W2 function.
        n_boot = n_src
        
        dists = []
        for _ in range(B):
            idx = rng.integers(0, n_src, size=n_boot)
            Rb = R[idx]
            dists.append(w2_empirical_uniform_exact(R, Rb))
        return float(np.quantile(np.asarray(dists), 1.0 - alpha))

    if method == "bootstrap_gaussian":
        assert R is not None, "bootstrap_gaussian needs raw sample matrix R."
        alpha  = float((params or {}).get("alpha", 0.05))
        B      = int((params or {}).get("B", 200))
        rng    = _rng_from_params(params or {})
        n_src, d = R.shape
        if n_src < 2:
            return 0.0
        # Set bootstrap sample size equal to regime size, in line with non-parametric bootstrap
        n_boot = n_src

        mu0   = np.mean(R, axis=0)
        S0    = np.cov(R.T, ddof=1)
        eps   = float((params or {}).get("epsilon_sigma", 1e-9))
        S0p = 0.5*(S0 + S0.T)
        vals, vecs = np.linalg.eigh(S0p)
        S0p = (vecs * np.clip(vals, eps, None)) @ vecs.T
        deltas = []
        for _ in range(B):
            Xb  = _mvnrnd_psd(mu0, S0p, n_boot, rng, eps=eps)
            mub = np.mean(Xb, axis=0)
            Sb  = np.cov(Xb.T, ddof=1)
            deltas.append(wasserstein2_gaussian(mu0, S0p, mub, Sb, eps=eps))
        return float(np.quantile(np.asarray(deltas), 1.0 - alpha))
    
    raise ValueError(f"Unknown delta_method='{method}'")
    

def psd_cholesky(Sigma, eps):
    """Symmetrize, regularize to PSD, then return Cholesky factor L s.t. L.T @ L ≈ Sigma_psd."""
    Sigma_sym = 0.5 * (Sigma + Sigma.T)                 # symmetrize
    Sigma_reg = Sigma_sym + eps * np.eye(Sigma_sym.shape[0])  # regularize
    try:
        L = np.linalg.cholesky(Sigma_reg)
        return L
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Sigma_sym)
        vals = np.clip(vals, eps, None)                 # floor small/negative eigenvalues
        Sigma_psd = vecs @ np.diag(vals) @ vecs.T + eps * np.eye(Sigma_sym.shape[0])
        L = np.linalg.cholesky(Sigma_psd)
        return L

def solve_optimizer(mu, Sigma, delta, config, verbose=False):
    n = len(mu)
    rb = config["risk_budget"]
    eps = config["epsilon_sigma"]

    L = psd_cholesky(Sigma, eps)

    w = cp.Variable(n)
    constraints = [cp.norm(L @ w, 2) <= rb, cp.sum(w) == 1]
    objective = cp.Maximize(mu @ w - delta * cp.norm(w, 2))
    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.MOSEK, verbose=verbose)
        if w.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return np.asarray(w.value).reshape(-1)
    except cp.SolverError:
        print(f"[ERROR] MOSEK solver failed: status={prob.status}")
    return None


# ---------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------

def fit_mvo(data, params, G):
    delta = 0.0
    w = solve_optimizer(data["mu_ann_full"], data["Sigma_ann_full"], delta,
                        {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
    return {"type": "static", "w": w, "kappa": 0.0, "delta": delta}


def fit_dro(data, params, G):
    delta = compute_delta(params.get("kappa", 1.0),
                          data["mu_ann_full"], data["Sigma_ann_full"], data["train"], params)
    w = solve_optimizer(data["mu_ann_full"], data["Sigma_ann_full"], delta,
                        {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
    return {"type": "static", "w": w, "kappa": params.get("kappa", np.nan), "delta": float(delta)}


def fit_regime_dro(data, params, G):
    n_days = data["n_days"]
    AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))

    # segmentation hook: to provide regime boundaries instead of using midpoints
    segs = params.get("segs")  # explicit segments override
    if segs is None:
        segs_fn = params.get("segs_fn", None)  # callable: (data, params, G) -> [0,...,n_days]
        if segs_fn is not None:
            segs = segs_fn(data, params, G)
        else:
            # old midpoint default
            taus  = data.get("taus_true", [0, n_days])
            delay = int(params.get("delay", 0))
            mids  = [int((taus[k-1] + taus[k]) / 2) for k in range(1, len(taus) - 1)]
            dets  = [min(m + delay, n_days - 1) for m in mids]
            for i in range(1, len(dets)):
                if dets[i] <= dets[i - 1]:
                    dets[i] = min(dets[i - 1] + 1, n_days - 1)
            segs = [0] + dets + [n_days]
    w_list, deltas = [], []        
    
    for a, b in zip(segs[:-1], segs[1:]):
        R_seg = data["train"][a:b]
        if (b - a) < 2:
            mu_est, Sigma_est = data["mu_ann_full"], data["Sigma_ann_full"]
            R_source = data["train"]                 # degenerate short segment
        else:
            log_seg   = np.log1p(R_seg)
            mu_est    = np.expm1(log_seg.mean(axis=0) * AF)
            Sigma_est = np.cov(log_seg.T, ddof=1) * AF
            R_source  = R_seg                         # << keep regime-k distribution
    
        # pass full-sample N via n_ref but bootstrap from R_source
        ### params_k = dict(params); params_k["n_ref"] = n_days
        params_k = dict(params); params_k["n_ref"] = (b - a)   # use segment length

        delta_k = compute_delta(params_k.get("kappa", 1.0), mu_est, Sigma_est, R_source, params_k)

        w_k = solve_optimizer(mu_est, Sigma_est, delta_k,
                              {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
        deltas.append(float(delta_k)); w_list.append(w_k)
        
    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": params.get("kappa", np.nan),
            "delta_list": deltas,
            "delta": np.nan}


def fit_dro_reverse(data, params, G):
    """
    Reverse-optimised scalar δ (provided by caller).
    params: {"delta": <float>}
    """
    delta = float(params["delta"])
    w = solve_optimizer(data["mu_ann_full"], data["Sigma_ann_full"], delta,
                        {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
    return {"type": "static", "w": w, "delta": delta, "kappa": np.nan}


def fit_regime_dro_reverse(data, params, G):
    """
    Reverse-optimised per-segment deltas.
    params: {"segs": [0, ... , n_days]  OR  "segs_fn": callable,  "delta_list": [δ1,...,δK]}
    """
    n_days = data["n_days"]
    AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))
    segs = params.get("segs")
    if segs is None:
        segs_fn = params.get("segs_fn")
        assert segs_fn is not None, "fit_regime_dro_rev: provide 'segs' or 'segs_fn'."
        segs = segs_fn(data, params, G)

    delta_list = list(map(float, params["delta_list"]))
    assert len(delta_list) == len(segs) - 1, "delta_list length must equal number of segments."

    w_list = []
    for j, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
        R = data["train"][a:b]
        if (b - a) < 2:
            mu_est = data["mu_ann_full"]; Sigma_est = data["Sigma_ann_full"]
        else:
            log_seg = np.log1p(R)
            mu_est  = np.expm1(log_seg.mean(axis=0) * AF)
            Sigma_est = np.cov(log_seg.T, ddof=1) * AF
        w_k = solve_optimizer(mu_est, Sigma_est, delta_list[j],
                              {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
        w_list.append(w_k)

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "delta_list": delta_list, "kappa": np.nan}


def fit_regime_dro_rev_constSigma(data, params, G):
    segs = params["segs"]
    Sigma_fix = data["Sigma_ann_full"]          # constant across segments
    w_list = []
    for j, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
        R_seg = data["train"][a:b]
        log_seg = np.log1p(R_seg)
        mu_est  = np.expm1(log_seg.mean(axis=0) * data["n_days"])
        w = solve_optimizer(mu_est, Sigma_fix, float(params["delta_list"][j]),
                            {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
        w_list.append(w)
    return {"type":"piecewise","w_list":w_list,"segs":segs,"delta_list":params["delta_list"]}


# ---------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------

def evaluate_portfolio(fit, data, G):
    train, test = data["train"], data["test"]; 
    n_days = data["n_days"]
    AF = int(data.get("ann_factor", 252))
    if fit["type"] == "static":
        stats_oos = portfolio_stats(fit["w"], test, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], 
                                                     "risk_budget": G["risk_budget"], "annualization_factor": AF})
        ge = float(np.sum(np.abs(fit["w"])))
        port_tr = train @ fit["w"]
        _, risk_tr, _ = stats_from_series(port_tr, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})
        stats_oos["gross_exp"] = ge
        stats_oos["gap"] = float(stats_oos["sigma_ann"] - risk_tr)
        stats_oos["kappa"] = float(fit.get("kappa", np.nan))
        stats_oos["delta"] = float(fit.get("delta", np.nan))
        return stats_oos
    else:  # piecewise
        cfg = {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "risk_budget": G["risk_budget"], "annualization_factor": AF}
        stats_oos = portfolio_stats_multipiece(fit["w_list"], fit["segs"], test, cfg)
        seg_lengths = np.diff(np.array(fit["segs"]))
        ge = float(np.sum(seg_lengths * np.array([np.sum(np.abs(wk)) for wk in fit["w_list"]])) / n_days)

        port_tr = np.zeros(n_days)
        for (a,b), wk in zip(zip(fit["segs"][:-1], fit["segs"][1:]), fit["w_list"]):
            port_tr[a:b] = train[a:b] @ wk
        _, risk_tr, _ = stats_from_series(port_tr, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})

        stats_oos["gross_exp"] = ge
        stats_oos["gap"] = float(stats_oos["sigma_ann"] - risk_tr)
        stats_oos["kappa"] = float(fit.get("kappa", np.nan))

        # REPORT PER-SEGMENT DELTAS; DO NOT AVERAGE
        dlist = list(map(float, fit.get("delta_list", [])))
        stats_oos["delta"] = np.nan                 # leave single 'delta' empty for Regime-DRO
        for j, dj in enumerate(dlist, start=1):     # expose delta_k1, delta_k2, ...
            stats_oos[f"delta_k{j}"] = dj

        seg_excess = []
        for (a, b), wk in zip(zip(fit["segs"][:-1], fit["segs"][1:]), fit["w_list"]):
            seg_length = b - a
            if seg_length <= 1:
                seg_excess.append(0.0)
                continue
            # Isolate the segment's return series
            seg_series = test[a:b] @ wk
            # Create a config for this segment, providing the correct annualization factor
            seg_config = dict(G)
            seg_config["annualization_factor"] = AF
            seg_config["n_days"] = n_days
            # Calculate stats using the correct factor
            _, sigma_seg_ann, _ = stats_from_series(seg_series, seg_config)
            seg_excess.append(max(sigma_seg_ann - G["risk_budget"], 0.0))
        stats_oos["seg_excess"] = seg_excess
    
        return stats_oos

def evaluate_regime_independently(fit, data, G):
    """
    Performs an independent evaluation for each segment of a piecewise portfolio.
    """
    n_days = data["n_days"]
    test = data["test"]
    
    # This dictionary will hold all the independent, per-segment stats
    stats_oos = {}
    
    # Add the delta list to the output for reference
    dlist = list(map(float, fit.get("delta_list", [])))
    for j, dj in enumerate(dlist, start=1):
        stats_oos[f"delta_k{j}"] = dj

    # Calculate and store performance for each segment independently
    for k, (a, b) in enumerate(zip(fit["segs"][:-1], fit["segs"][1:])):
        wk = fit["w_list"][k]
        seg_length = b - a
        
        # Define default values for empty/trivial segments
        mu_seg, sigma_seg, sharpe_seg, vol_breach_seg = np.nan, np.nan, np.nan, np.nan
        gross_exp_seg = np.sum(np.abs(wk))

        if seg_length > 1:
            # 1. Isolate the segment's out-of-sample data
            seg_series_oos = test[a:b] @ wk
            
            # 2. Create a config for this segment's independent evaluation
            #    (relies on the corrected stats_from_series from dro.py)
            seg_config = dict(G)
            seg_config["n_days"] = n_days
            seg_config["annualization_factor"] = int(data.get("ann_factor", 252))
            
            # 3. Calculate statistics for this segment ONLY
            mu_seg, sigma_seg, sharpe_seg = stats_from_series(seg_series_oos, seg_config)
            vol_breach_seg = max(sigma_seg - G["risk_budget"], 0.0)

        # 4. Store results with segment-specific keys
        stats_oos[f"mu_ann_k{k+1}"] = mu_seg
        stats_oos[f"sigma_ann_k{k+1}"] = sigma_seg
        stats_oos[f"sharpe_ann_k{k+1}"] = sharpe_seg
        stats_oos[f"vol_breach_k{k+1}"] = vol_breach_seg
        stats_oos[f"gross_exp_k{k+1}"] = gross_exp_seg
    
    return stats_oos
    
def stats_from_series(port_daily, config):
    n_days = config["n_days"]
    rf_annual = config["risk_free_rate"]
    AF = int(config.get("annualization_factor", n_days))  # fallback to old behavior if not set
    rf_daily = (1 + rf_annual) ** (1 / AF) - 1
    sigma_daily = np.std(port_daily, ddof=1)
    sigma_annual = sigma_daily * np.sqrt(AF)
    mu_annual_geom = np.exp(AF * np.mean(np.log1p(port_daily))) - 1
    sharpe_annual = (np.mean(port_daily) - rf_daily) / sigma_daily * np.sqrt(AF) if sigma_daily > 0 else np.nan
    return mu_annual_geom, sigma_annual, sharpe_annual

def portfolio_stats(weights, returns, config):
    """Static weights over full horizon."""
    weights = np.asarray(weights).reshape(-1)
    port_daily = returns @ weights
    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    return {
        "mu_ann": mu_annual_geom,
        "sigma_ann": sigma_annual,
        "sharpe_ann": sharpe_annual,
        "vol_breach": vol_breach}

def portfolio_stats_multipiece(w_list, taus, returns, config):
    """
    w_list: list of weights per piece, length = len(taus)-1
    taus:   [0=τ0, τ1, ..., τK=n_days]
    """
    n_days = config["n_days"]
    assert taus[0] == 0 and taus[-1] == n_days and len(w_list) == len(taus) - 1
    port_daily = np.empty(n_days, dtype=float)
    for k in range(len(w_list)):
        a, b = taus[k], taus[k + 1]
        port_daily[a:b] = returns[a:b] @ np.asarray(w_list[k]).reshape(-1)
    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    return {"mu_ann": mu_annual_geom, "sigma_ann": sigma_annual, "sharpe_ann": sharpe_annual, "vol_breach": vol_breach}


# ---------------------------------------------------------------
# Statistical tests (for hypothesis testing)
# ---------------------------------------------------------------

def format_ci(mean, std):
    return f"{mean:.4f} ({(mean - std):.4f}, {(mean + std):.4f})"
    
def paired_onesided_less(x, y):
    # H0: mean(x - y) >= 0  vs  H1: mean(x - y) < 0
    d = x - y
    t, p = stats.ttest_1samp(d, popmean=0.0, alternative="less")
    return t, p

def paired_t_twosided(x, y):
    # H0: mean(x - y) == 0  vs  H1: ≠ 0
    d = x - y
    t, p = stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    return t, p

def noninferiority_paired(x, y, delta):
    # H0: mean(x - y) <= -delta  vs  H1: > -delta
    d = (x - y) + delta
    t, p = stats.ttest_1samp(d, popmean=0.0, alternative="greater")
    return t, p

def superiority_paired(x, y):
    # H0: mean(x - y) <= 0  vs  H1: > 0
    d = x - y
    t, p = stats.ttest_1samp(d, popmean=0.0, alternative="greater")
    return t, p

def paired_two_sided_test_with_ci(x, y, alpha=0.05):
    """
    Paired two-sided t-test for mean(x - y) == 0 with 95% CI for the mean difference.
    Returns dict(mean_diff, t, p, ci_low, ci_high, n).
    """
    d = x - y
    n = len(d)
    mean_diff = np.mean(d)
    sd = np.std(d, ddof=1)
    se = sd / np.sqrt(n)
    t, p = stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_low, ci_high = mean_diff - tcrit * se, mean_diff + tcrit * se
    return dict(mean_diff=mean_diff, t=t, p=p, ci_low=ci_low, ci_high=ci_high, n=n)


# ---------------------------------------------------------------
# Hypothesis Testing
# ---------------------------------------------------------------

def hypothesis_tests(results_dict, tests, alpha=0.05):
    """
    Verbose hypothesis test reporting (no hardcoding of model names or columns).
    tests: list of {"kind": "breach_less" | "equality_sharpe" | "superiority_sharpe",
                    "A": "<model name>", "B": "<model name>"}
    """
    # ---- column resolver (generic) ----
    COLS = {
        "mu": ["mu_ann", "mu_annual_geom", "Expected Return (CAGR)", "CAGR"],
        "sh": ["sharpe_ann", "Sharpe annual", "Sharpe Ratio"],
        "br": ["vol_breach", "Volatility Breach"],
    }
    def pick_col_any(results_dict, candidates):
        for df in results_dict.values():
            for c in candidates:
                if c in df.columns:
                    return c
        raise KeyError(f"None of {candidates} found in any results DataFrame columns.")

    col_mu = pick_col_any(results_dict, COLS["mu"])
    col_sh = pick_col_any(results_dict, COLS["sh"])
    col_br = pick_col_any(results_dict, COLS["br"])

    # ---- header ----
    print("\n" + "=" * 72)
    print(f"HYPOTHESIS TESTS  (alpha = {alpha:.2f}, confidence = {int((1 - alpha) * 100)}%)")
    print("=" * 72)

    # helpers
    def align_pair(A, B):
        if A not in results_dict or B not in results_dict:
            raise KeyError(f"Missing model in results: needed '{A}' and '{B}'.")
        dfA, dfB = results_dict[A], results_dict[B]
        m = min(len(dfA), len(dfB))
        if m == 0:
            raise ValueError(f"No overlapping trials for pair ({A}, {B}).")
        return (dfA.iloc[:m].reset_index(drop=True),
                dfB.iloc[:m].reset_index(drop=True))

    printed_section1 = False
    printed_section2 = False
    idx1 = 0  # 1A, 1B, ...
    idx2 = 0  # 2A, 2B, ...
    def ab_label(k): return chr(64 + k)  # 1->A, 2->B,...

    for t in tests:
        kind = t["kind"]; A, B = t["A"], t["B"]

        if kind == "breach_less":
            if not printed_section1:
                print("\n[1] Risk-budget breaches (vol_breach)")
                printed_section1 = True
            idx1 += 1
            label = f"1{ab_label(idx1)})"
            dfA, dfB = align_pair(A, B)
            x = dfA[col_br].to_numpy()
            y = dfB[col_br].to_numpy()

            print(f"\n{label} {A} vs {B} — vol_breach (paired t-test, one-sided)")
            print(f"   H0: mean({A}_vol_breach - {B}_vol_breach) = 0")
            print(f"   H1: mean({A}_vol_breach - {B}_vol_breach) < 0")
            T, P = paired_onesided_less(x, y)
            mean_diff = (x - y).mean()
            print(f"   Test: Paired t-test on differences ({A} - {B})")
            print(f"   alpha={alpha:.2f}, t={T:.3f}, p(one-sided)={P:.4g}, mean diff={mean_diff:.6f}")
            if P < alpha:
                print(f"   Conclusion: REJECT H0 at {int((1 - alpha) * 100)}% confidence → {A} breaches LESS than {B}.")
            else:
                print("   Conclusion: FAIL TO REJECT H0 — No significant reduction in breaches.")

        elif kind == "equality_sharpe":
            if not printed_section2:
                print("\n[2] Performance")
                printed_section2 = True
            idx2 += 1
            label = f"2{ab_label(idx2)})"
            dfA, dfB = align_pair(A, B)
            x = dfA[col_sh].to_numpy()
            y = dfB[col_sh].to_numpy()

            print(f"\n{label} Equality: {A} vs {B} — Sharpe (paired t-test, two-sided)")
            print(f"   H0: mean({A}_sharpe - {B}_sharpe) = 0")
            print(f"   H1: mean({A}_sharpe - {B}_sharpe) ≠ 0")
            res = paired_two_sided_test_with_ci(x, y, alpha=alpha)
            print("   Test: Paired two-sided t-test on "
                  f"({A} - {B})")
            print(
                f"   alpha={alpha:.2f}, t={res['t']:.3f}, p(two-sided)={res['p']:.4g}, "
                f"mean diff={res['mean_diff']:.6f}, 95% CI=({res['ci_low']:.6f}, {res['ci_high']:.6f}), n={res['n']}"
            )
            if res["p"] < alpha:
                direction = f"{A} > {B}" if res["mean_diff"] > 0 else f"{A} < {B}"
                print(f"   Conclusion: REJECT H0 at {int((1 - alpha) * 100)}% confidence → Sharpe differs ({direction}).")
            else:
                print("   Conclusion: FAIL TO REJECT H0 — No statistically significant Sharpe difference.")

        elif kind == "superiority_sharpe":
            if not printed_section2:
                print("\n[2] Performance")
                printed_section2 = True
            idx2 += 1
            label = f"2{ab_label(idx2)})"
            dfA, dfB = align_pair(A, B)
            x = dfA[col_sh].to_numpy()
            y = dfB[col_sh].to_numpy()

            print(f"\n{label} Superiority: {A} vs {B} — Sharpe (paired t-test)")
            print(f"   H0: mean({A}_sharpe - {B}_sharpe) ≤ 0")
            print(f"   H1: mean({A}_sharpe - {B}_sharpe) > 0")
            T, P = superiority_paired(x, y)
            mean_diff = (x - y).mean()
            print(f"   alpha={alpha:.2f}, t={T:.3f}, p(one-sided)={P:.4g}, mean diff={mean_diff:.6f}")
            if P < alpha:
                print(f"   Conclusion: REJECT H0 at {int((1 - alpha) * 100)}% confidence → {A} Sharpe is SUPERIOR to {B}.")
            else:
                print("   Conclusion: FAIL TO REJECT H0 — No significant Sharpe improvement detected.")


# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------

def _fmt_series(x: pd.Series) -> str:
    x = pd.Series(x).dropna()
    if len(x) < 2:
        return f"{float(x.iloc[0]) if len(x) else float('nan'):.4f}"
    m = x.mean(); s = x.std(ddof=1)
    return f"{m:.4f} ({(m - s):.4f}, {(m + s):.4f})"

def oos_summary(results: dict, model_order=None) -> pd.DataFrame:
    """
    Build OOS table (mean ± std bounds).
    Rows: mu_ann, sigma_ann, sharpe_ann, vol_breach, p_viol, gross_exp, kappa, gap, delta
    Cols: in the order provided by `model_order` (or insertion order of `results`).
    """
    base_rows = ["mu_ann","sigma_ann","sharpe_ann","vol_breach","p_viol",
                 "gross_exp","kappa","gap","delta"]
    if model_order is None:
        model_order = list(results.keys())

    # >>> collect any per-segment delta columns present in any model
    delta_k_cols = []
    for df in results.values():
        for c in df.columns:
            if str(c).startswith("delta_k"):
                delta_k_cols.append(c)
    # sort by segment index: delta_k1, delta_k2, ...
    def _seg_ix(name):
        try: return int(str(name).replace("delta_k",""))
        except: return 10**9
    delta_k_cols = sorted(set(delta_k_cols), key=_seg_ix)
    rows = base_rows + delta_k_cols
    # <<<

    table = {}
    for m in model_order:
        if m not in results: continue
        df = results[m]
        s = {}
        for col in base_rows:
            if col in df.columns and len(df[col].dropna())>0:
                s[col] = _fmt_series(df[col])
        if "vol_breach" in df.columns:
            z = (df["vol_breach"] > 0).astype(float)
            s["p_viol"] = _fmt_series(z)
        # >>> add the per-segment delta summaries if present
        for c in delta_k_cols:
            if c in df.columns and len(df[c].dropna())>0:
                s[c] = _fmt_series(df[c])
        # <<<
        table[m] = pd.Series(s)
    return pd.DataFrame(table).reindex(rows)


def print_oos_table(results_dict, model_order):
    n_by_model = {m: len(results_dict[m]) for m in model_order if m in results_dict}
    single = all(n==1 for n in n_by_model.values()) and len(n_by_model)>0
    print("\n" + "=" * 72)
    if single:
        print("OOS Portfolio Performance (single trial)")
    else:
        print("OOS Portfolio Performance Summary (mean ± std bounds)")
    print("=" * 72)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(oos_summary(results_dict, model_order=model_order))
        
def print_single_portfolio_block(label, w, returns_train, returns_eval, rho, Sigma_ann, config, rtol=1e-6, atol=1e-9):
    n_days, n_assets = returns_train.shape
    AF = int(config.get("annualization_factor", config["n_days"]))
    mu_train_ann_assets    = AF * returns_train.mean(axis=0)
    sigma_train_ann_assets = np.sqrt(AF) * returns_train.std(axis=0, ddof=1)

    # exact constraint metric (matches solver): ||L w||_2 with L^T L ≈ Σ_ann
    L = psd_cholesky(Sigma_ann, config["epsilon_sigma"])
    risk_train_ann = float(np.linalg.norm(L @ w))
    tol = max(atol, rtol * max(rho, risk_train_ann))
    ok_train = bool(risk_train_ann <= rho + tol)

    # returns (annualized)
    ret_train_ann = float(mu_train_ann_assets @ w)

    # OOS realized vol like multi-trial breach (from series)
    port_eval = returns_eval @ w
    _, risk_eval_ann, _ = stats_from_series(port_eval, dict(config, annualization_factor=AF))
    mu_eval_ann_assets = AF * returns_eval.mean(axis=0)
    ret_eval_ann = float(mu_eval_ann_assets @ w)

    gross_exposure = float(np.sum(np.abs(w)))
    top_idx = np.argsort(w)[-3:][::-1]
    nz = np.where(w != 0)[0]
    bot_idx = nz[np.argsort(w[nz])[:3]] if nz.size else np.array([], dtype=int)

    print("\n" + "=" * 50)
    print(label)
    print("=" * 50 + "\n")
    print("TRAIN SAMPLE STATISTICS:")
    print(f"n_assets: {n_assets}")
    print(f"n_days:   {n_days}")
    print(f"Max mean return (annualized): {mu_train_ann_assets.max():.4f}")
    print(f"Min mean return (annualized): {mu_train_ann_assets.min():.4f}")
    print(f"Max volatility (annualized) : {sigma_train_ann_assets.max():.4f}")
    print(f"Min volatility (annualized) : {sigma_train_ann_assets.min():.4f}")

    print("\nTRAIN PORTFOLIO (on Σ_train, μ_train):")
    print(f"Annualized expected return:   {ret_train_ann:.4f}")
    print(f"Annualized risk (vol)     :   {risk_train_ann:.4f}")
    print(f"Risk bound ρ              :   {rho:.4f}")
    print(f"Train constraint satisfied:   {ok_train}")
    print(f"Gross exposure (‖w‖₁)     :   {gross_exposure:.4f}")

    print("\nTop 3 assets with largest weights:")
    for i in top_idx:
        print(f"Asset {i:2d}: weight = {w[i]:+.4f}, μ = {mu_train_ann_assets[i]:+.4f}, σ = {sigma_train_ann_assets[i]:.4f}")

    print("\nTop 3 assets with smallest nonzero weights:")
    for i in bot_idx:
        print(f"Asset {i:2d}: weight = {w[i]:+.4f}, μ = {mu_train_ann_assets[i]:+.4f}, σ = {sigma_train_ann_assets[i]:.4f}")

    print("\nEVAL / OOS PORTFOLIO (on Σ_eval, μ_eval):")
    print(f"OOS annualized return     : {ret_eval_ann:.4f}")
    print(f"OOS annualized risk (vol) : {risk_eval_ann:.4f}")
    print(f"Risk bound ρ              : {rho:.4f}")

def print_regime_block(label, returns_train, returns_eval, w_list, segs, rho,
                       taus_display, seg_deltas, config=None):
    """
    Pretty-printer for piecewise portfolios.
    Uses 'annualization_factor' (AF) if provided in config, else falls back to n_days.
    """
    n_days, n_assets = returns_train.shape
    AF = int((config or {}).get("annualization_factor",
                                (config or {}).get("n_days", n_days)))

    # concatenated series for realized stats (like multi-trial)
    port_train = np.zeros(n_days); port_eval = np.zeros(n_days)
    for k, w in enumerate(w_list):
        a, b = segs[k], segs[k+1]
        port_train[a:b] = returns_train[a:b] @ w
        port_eval[a:b]  = returns_eval[a:b]  @ w

    # Use the same stats helper used everywhere else (respects AF)
    cfg = {"n_days": n_days,
           "risk_free_rate": (config or {}).get("risk_free_rate", 0.0),
           "annualization_factor": AF}
    ret_train_ann, risk_train_ann, _ = stats_from_series(port_train, cfg)
    ret_eval_ann,  risk_eval_ann,  _ = stats_from_series(port_eval,  cfg)

    # Asset-level sample stats (arith. daily → annualized with AF)
    mu_train_ann_assets    = AF * returns_train.mean(axis=0)
    sigma_train_ann_assets = np.sqrt(AF) * returns_train.std(axis=0, ddof=1)

    print("\n" + "=" * 50)
    print(label)
    print("=" * 50 + "\n")
    print("TRAIN SAMPLE STATISTICS:")
    print(f"n_assets: {n_assets}")
    print(f"n_obs:    {n_days}")
    print(f"Max mean return (annualized): {mu_train_ann_assets.max():.4f}")
    print(f"Min mean return (annualized): {mu_train_ann_assets.min():.4f}")
    print(f"Max volatility (annualized) : {sigma_train_ann_assets.max():.4f}")
    print(f"Min volatility (annualized) : {sigma_train_ann_assets.min():.4f}")

    print("\nSEQUENTIAL PIECES (train timeline):")
    for k in range(len(segs) - 1):
        a, b = segs[k], segs[k+1]
        detect_note = "" if k == 0 else f" (detected at day {a})"
        print(f"Piece {k+1}: days [{a}, {b}){detect_note}")
    print("\nRegime switch points τ (true): " + ", ".join(map(str, taus_display)))
    for k, delta_k in enumerate(seg_deltas, 1):
        print(f"Piece {k}: δ_k = {delta_k:.4f}")

    print("\nTRAIN PORTFOLIO (concatenated):")
    print(f"Annualized expected return:   {ret_train_ann:.4f}")
    print(f"Annualized risk (vol)     :   {risk_train_ann:.4f}")
    print(f"Risk bound ρ              :   {rho:.4f}")

    print("\nEVAL / OOS PORTFOLIO (concatenated):")
    print(f"OOS annualized return     : {ret_eval_ann:.4f}")
    print(f"OOS annualized risk (vol) : {risk_eval_ann:.4f}")
    print(f"Risk bound ρ              : {rho:.4f}")

    print("\nPER-PIECE WEIGHT SUMMARIES:")
    for k, w in enumerate(w_list):
        a, b = segs[k], segs[k+1]
        # Asset-level segment stats, annualized with AF
        if (b - a) > 0:
            mu_seg_ann = AF * returns_train[a:b].mean(axis=0)
        else:
            mu_seg_ann = mu_train_ann_assets
        if (b - a) > 1:
            sig_seg_ann = np.sqrt(AF) * returns_train[a:b].std(axis=0, ddof=1)
        else:
            sig_seg_ann = sigma_train_ann_assets

        top_idx = np.argsort(w)[-3:][::-1]
        nz = np.where(w != 0)[0]
        bot_idx = nz[np.argsort(w[nz])[:3]] if nz.size else np.array([], dtype=int)

        print(f"\nPiece {k+1}  days [{a},{b}):")
        print("Top 3 assets with largest weights:")
        for i in top_idx:
            print(f"Asset {i:2d}: weight = {w[i]:+.4f}, μ = {mu_seg_ann[i]:+.4f}, σ = {sig_seg_ann[i]:.4f}")
        print("Top 3 assets with smallest nonzero weights:")
        for i in bot_idx:
            print(f"Asset {i:2d}: weight = {w[i]:+.4f}, μ = {mu_seg_ann[i]:+.4f}, σ = {sig_seg_ann[i]:.4f}")

# -------------------------
# Pipeline helpers
# -------------------------

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

def _num_series(s):
    return pd.to_numeric(s, errors="coerce").astype("float64")

def map_labels_to_calendar(z_ser: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    """
    Map labels only on exact matching dates in `cal`.
    No forward/back fill. Missing elsewhere.
    """
    z = pd.Series(z_ser).sort_index()
    z.index = pd.to_datetime(z.index)
    cal = pd.DatetimeIndex(cal)
    out = pd.Series(np.nan, index=cal)
    inter = cal.intersection(z.index)
    if len(inter):
        out.loc[inter] = z.reindex(inter).to_numpy()
    return out.to_numpy(dtype=float)

def build_calendar_union_px(tickers, px_all, start_dt, end_dt):
    """UNION of PX return dates over [start_dt,end_dt]."""
    cal = None
    for t in tickers:
        px_ser = _num_series(px_all[t].loc[start_dt:end_dt]).dropna()
        idx = np.log(px_ser).diff().dropna().index  # dates where return exists
        cal = idx if cal is None else cal.union(idx)
    if cal is None or len(cal) == 0:
        raise RuntimeError("Empty calendar after applying start/end.")
    return pd.DatetimeIndex(cal)

def make_data_from_returns_panel(R: pd.DataFrame, ann_factor=252):
    R = pd.DataFrame(R).astype(np.float64)
    X = np.ascontiguousarray(R.to_numpy(np.float64, copy=False))
    T, N = X.shape
    logR  = np.log1p(X)
    mu_d  = logR.mean(axis=0)
    Sig_d = np.cov(logR.T, ddof=1)
    return {
        "train": X, "test": X, "n_days": T, "ann_factor": ann_factor,
        "mu_ann_full": np.expm1(mu_d * ann_factor), "Sigma_ann_full": Sig_d * ann_factor,
        "px_cols": R.columns.tolist(), "index": R.index}

def pooled_moments_by_regime(R_df, Z_labels, t_idx, ann=252, min_pair=60, mode="pairwise"):
    """
    R_df: DataFrame (T x N) of returns; may contain NaNs (unequal histories).
    Z_labels: dict[name] -> array length T with labels; NaN allowed outside span.
    Uses the label state at t_idx for each asset, then pools over all rows where that state holds,
    intersected with availability (non-NaN returns). Pairwise covariance uses pairwise overlaps.
    """
    names = list(R_df.columns); N = len(names)
    L = np.log1p(R_df.values)  # (T, N), may have NaNs
    T = L.shape[0]
    t_idx = int(min(max(0, t_idx), T-1))

    # current regime state per asset (must exist and not NaN at t_idx)
    s = []
    valid_asset = []
    for n in names:
        z = np.asarray(Z_labels[n], float)  # may contain NaN
        if z.shape[0] != T:
            raise ValueError("Z_labels arrays must have length T after mapping to calendar.")
        if np.isfinite(z[t_idx]):
            s.append(int(z[t_idx]))
            valid_asset.append(True)
        else:
            s.append(np.nan)
            valid_asset.append(False)

    # filter to assets that have a valid label at t_idx
    keep = [i for i, ok in enumerate(valid_asset) if ok]
    if not keep:
        # no assets valid at this time -> return zeros of right shape
        mu_ann = np.zeros(N)
        Sig_ann = np.zeros((N, N))
        return mu_ann, Sig_ann

    # compute masks and moments
    mu_d = np.zeros(N); var_d = np.zeros(N)
    S_sets = [None]*N

    for i, n in enumerate(names):
        if not valid_asset[i]:
            mu_d[i] = 0.0; var_d[i] = 0.0; S_sets[i] = np.array([], dtype=int); continue
        zi = np.asarray(Z_labels[n], float)
        # indices where label equals current state and return is available
        S_i = np.where((zi == s[i]) & np.isfinite(L[:, i]))[0]
        S_sets[i] = S_i
        li = L[S_i, i]
        mu_d[i]  = float(np.nanmean(li)) if li.size else 0.0
        var_d[i] = float(np.nanvar(li, ddof=1)) if li.size > 1 else 0.0

    Sig_d = np.zeros((N, N))
    if mode == "diag":
        np.fill_diagonal(Sig_d, var_d)
    else:
        min_pairs = np.inf
        for i in range(N):
            for j in range(i, N):
                if not (valid_asset[i] and valid_asset[j]):
                    cij = 0.0 if i != j else var_d[i]
                else:
                    S_ij = np.intersect1d(S_sets[i], S_sets[j], assume_unique=False)
                    # drop rows where either return is NaN (already filtered in S_i by column-wise)
                    if S_ij.size >= 2:
                        cij = np.cov(L[S_ij, i], L[S_ij, j], ddof=1)[0, 1]
                    else:
                        cij = 0.0 if i != j else var_d[i]
                    min_pairs = min(min_pairs, S_ij.size if S_ij.size else np.inf)
                Sig_d[i, j] = Sig_d[j, i] = float(cij)
        # shrink if pairwise sample is thin
        if not np.isinf(min_pairs):
            lam = float(min(1.0, (min_pair / max(int(min_pairs), 1))))
            Sig_d = (1 - lam) * Sig_d + lam * np.diag(np.diag(Sig_d))

    mu_ann  = np.expm1(ann * mu_d)
    Sig_ann = ann * Sig_d
    return mu_ann, Sig_ann

def _select_best_config(results_df, security, prefer_configs=None):
    df = results_df[results_df["security"] == security].copy()
    if prefer_configs:
        pref = df["config"].isin(prefer_configs)
        df = df[pref] if pref.any() else df
    if "rank" in df.columns:
        df = df.sort_values(["rank", "score"], ascending=[True, False])
    else:
        df = df.sort_values("score", ascending=False)
    if df.empty:
        return None
    return str(df.iloc[0]["config"])

def _labels_from_segments_df(segments_df, security, config):
    df = segments_df[(segments_df["security"] == security) &
                     (segments_df["config"] == config)].copy()
    if df.empty:
        return None
    df = df.sort_values("date")
    return pd.Series(
        df["z"].astype(int).to_numpy(),
        index=pd.DatetimeIndex(df["date"]), name="z",)
    
# -------------------------
# Pipeline
# -------------------------

def dro_pipeline(tickers, RSLDS_CONFIG, DRO_CONFIG, DELTA_DEFAULTS):
    
    import numpy as np, pandas as pd, hashlib, os

    # ---------- helpers (debug only) ----------
    def _sha(arr: np.ndarray) -> str:
        a = np.ascontiguousarray(np.asarray(arr, float))
        return hashlib.sha256(a.tobytes()).hexdigest()[:16]

    def _dbg_dataset(tag: str, data: dict):
        tr = data["train"]; te = data["test"]
        print(f"[DEBUG:{tag}] train.shape={tr.shape}  test.shape={te.shape}")
        # column names may be kept in px_cols
        cols_tr = data.get("px_cols", None)
        if cols_tr is None and isinstance(tr, pd.DataFrame):
            cols_tr = list(tr.columns)
        cols_te = cols_tr
        print(f"[DEBUG:{tag}] columns(train)==columns(test)? {True}")
        print(f"[DEBUG:{tag}] columns={cols_tr}")
        if isinstance(tr, pd.DataFrame):
            na_tr = tr.isna().sum().to_dict()
            na_te = te.isna().sum().to_dict() if isinstance(te, pd.DataFrame) else {}
            print(f"[DEBUG:{tag}] NaNs(train) per col={na_tr}")
            print(f"[DEBUG:{tag}] NaNs(test)  per col={na_te}")
            print(f"[DEBUG:{tag}] train index: {tr.index.min()} → {tr.index.max()} (len={len(tr.index)})")
            print(f"[DEBUG:{tag}] test  index:  {te.index.min()} → {te.index.max()} (len={len(te.index)})")
            print(f"[DEBUG:{tag}] dtypes(train)={[str(t) for t in tr.dtypes]}")
            print(f"[DEBUG:{tag}] dtypes(test) ={[str(t) for t in te.dtypes]}")
        # numeric fingerprints
        tr_np = tr.to_numpy(dtype=float) if isinstance(tr, pd.DataFrame) else np.asarray(tr, float)
        te_np = te.to_numpy(dtype=float) if isinstance(te, pd.DataFrame) else np.asarray(te, float)
        print(f"[DEBUG:{tag}] sha(train)={_sha(tr_np)}  sha(test)={_sha(te_np)}")

    def _resolve_rSLDS_outputs(RSLDS_CONFIG):
        res_csv  = RSLDS_CONFIG["results_csv"]
        seg_parq = RSLDS_CONFIG["segments_parquet"]
        return res_csv, seg_parq

    # =========================================================
    
    # 0) Panels
    px_all, eps_all, pe_all, ser_vix = import_data(RSLDS_CONFIG["data_excel"])

    # 1) RETURNS calendar (UNION; PX-only; honours start/end)
    GIDX = build_calendar_union_px(
        tickers, px_all, DRO_CONFIG["start_dt"], DRO_CONFIG["end_dt"]
    )

    # 2) Returns panel — IDENTICAL construction for both modes
    R_cols = []
    for t in tickers:
        px = _num_series(px_all[t].loc[DRO_CONFIG["start_dt"]:DRO_CONFIG["end_dt"]]).dropna()
        r  = px.pct_change()                 # native calendar returns
        R_cols.append(r.reindex(GIDX))       # align to union; keep NaN where absent
    R_use    = pd.concat(R_cols, axis=1)
    R_use.columns = list(tickers)
    R_df_all = R_use.copy()

    # Part A on intersection only (OK to drop NaN here because Part A is "static common")
    R_use_clean = R_use.dropna(how="any")
    dataA = make_data_from_returns_panel(R_use_clean)

    # Debug digest for the returns dataset (same in both modes)
    print(f"[DEBUG:MODE] run_gridsearch={bool(DRO_CONFIG.get('run_gridsearch', False))}")
    print(f"[DEBUG:R] R_use.shape={R_use.shape}  R_use_clean.shape={R_use_clean.shape}")
    print(f"[DEBUG:R] cols={list(R_use.columns)}")
    print(f"[DEBUG:R] index span: {R_use.index.min()} → {R_use.index.max()}  (len={len(R_use.index)})")
    _dbg_dataset("DATA_A", {"train": R_use_clean, "test": R_use_clean, "px_cols": list(R_use_clean.columns)})

    # 3) Part A: static DRO (on common intersection)
    N = dataA["train"].shape[1]
    print(f"[DEBUG:FIT_A] N={N} (assets in DATA_A)")
    paramsA = dict(DELTA_DEFAULTS[DRO_CONFIG["delta_name"]])
    fitA = fit_dro(dataA, paramsA, DRO_CONFIG["GLOBAL"])
    print(f"[DEBUG:FIT_A] len(w)={len(fitA['w'])}")
    assert len(fitA["w"]) == N, f"len(w)={len(fitA['w'])} != N={N} from DATA_A"
    summA = evaluate_portfolio({"type": "static", "w": fitA["w"]}, dataA, DRO_CONFIG["GLOBAL"])

    # 4) rSLDS labels (CSV affects model selection; Parquet supplies segments)
    Z_labels = {}
    if DRO_CONFIG.get("run_gridsearch", False):
        # Run fresh gridsearch (no file IO for labels)
        best = pipeline_actual(tickers, RSLDS_CONFIG)
        for sec in tickers:
            if sec in best["Z_labels"]:
                Z_labels[sec] = map_labels_to_calendar(best["Z_labels"][sec], R_df_all.index)

    else:
        # Reuse artifacts: CSV (results) + Parquet (segments), both from RSLDS_CONFIG
        import os
        res_csv, seg_parq = _resolve_rSLDS_outputs(RSLDS_CONFIG)
    
        if not os.path.exists(res_csv):
            raise FileNotFoundError(f"Results CSV not found: {res_csv}")
        if not os.path.exists(seg_parq):
            raise FileNotFoundError(f"Segments Parquet not found: {seg_parq}")
    
        df_res = pd.read_csv(res_csv)
        df_seg = pd.read_parquet(seg_parq)
    
        # --- STRICT ARTIFACT VALIDATION (Parquet only; no legacy CSV fallback) ---
        required_cols = {"security", "config", "date", "z"}
        missing_cols = required_cols - set(df_seg.columns)
        if missing_cols:
            raise ValueError(f"Segments parquet missing required columns: {missing_cols}")
    
        if df_seg["date"].dtype != "datetime64[ns]":
            df_seg["date"] = pd.to_datetime(df_seg["date"], errors="coerce")
    
        requested = set(tickers)
        res_securities = set(df_res["security"].astype(str).unique())
        missing_in_results = requested - res_securities
        if missing_in_results:
            raise ValueError(
                "Requested tickers missing in gridsearch_results.csv: "
                + ", ".join(sorted(missing_in_results))
            )
    
        # ensure segment dates overlap eval calendar
        gidx_norm = pd.DatetimeIndex(GIDX.normalize())
        bad_no_config = []
        bad_no_segments = []
        bad_no_overlap = []
    
        for sec in tickers:
            cfg_best = _select_best_config(df_res, sec, DRO_CONFIG.get("prefer_configs"))
            if cfg_best is None:
                bad_no_config.append(sec)
                continue
    
            mask = (
                (df_seg["security"].astype(str) == sec)
                & (df_seg["config"].astype(str) == str(cfg_best))
            )
            if not mask.any():
                bad_no_segments.append(f"{sec}[{cfg_best}]")
                continue
    
            seg_dates = pd.DatetimeIndex(df_seg.loc[mask, "date"].dropna().unique())
            if len(seg_dates) == 0 or gidx_norm.intersection(seg_dates.normalize()).empty:
                bad_no_overlap.append(sec)
    
        errs = []
        if bad_no_config:
            errs.append("No winning config in results CSV for: " + ", ".join(sorted(bad_no_config)))
        if bad_no_segments:
            errs.append("No segment rows in parquet for: " + ", ".join(sorted(bad_no_segments)))
        if bad_no_overlap:
            errs.append(
                "No date overlap between parquet segments and returns calendar for: "
                + ", ".join(sorted(bad_no_overlap))
            )
        if errs:
            raise ValueError("Artifact validation failed:\n - " + "\n - ".join(errs))
        # --- END VALIDATION ---
    
        # Build Z_labels (helper works with a DataFrame loaded from parquet)
        for sec in tickers:
            cfg_best = _select_best_config(df_res, sec, DRO_CONFIG.get("prefer_configs"))
            z_ser = _labels_from_segments_csv(df_seg, sec, cfg_best)  # <-- fixed helper name
            Z_labels[sec] = map_labels_to_calendar(z_ser, R_df_all.index)
    
    missing = [sec for sec in tickers if sec not in Z_labels]
    if missing:
        print(f"[WARN] Missing regimes for: {missing}. Dropped from pooled moments.")
    avail = [sec for sec in tickers if sec in Z_labels]
    if not avail:
        raise RuntimeError("No assets produced rSLDS labels. Cannot proceed with Part B.")

    def _sha1(x):  # short fingerprint for equality checks across modes
        a = np.asarray(x, float)
        return hashlib.sha1(a.tobytes()).hexdigest()[:12]
    
    T = len(R_df_all.index)
    print(f"[DEBUG:Z] T={T} (union calendar length)  tickers={list(avail)}")
    
    for sec in avail:
        z = np.asarray(Z_labels[sec], float)
        assert z.shape[0] == T, f"[{sec}] label length {len(z)} != T={T}"
        n_nan = int(np.isnan(z).sum())
        uniq  = sorted(set([int(u) for u in np.unique(z[np.isfinite(z)])])) if np.isfinite(z).any() else []
        print(f"[DEBUG:Z] {sec}: len={len(z)}  NaN={n_nan}  unique_states={uniq}  sha={_sha1(z)}")
    
    # stable ordering used downstream
    print(f"[DEBUG:NAMES] order={list(avail)}")

    # 5) Per-asset segments + entry indices
    per_asset_segs_on_cal = {}
    for sec in avail:
        z_arr = np.asarray(Z_labels[sec], float)  # length T, NaNs allowed
        # changepoints on the union calendar (ignore NaN→state and state→NaN edges)
        finite = np.isfinite(z_arr)
        cps = []
        for t in range(1, T):
            if finite[t] and finite[t-1] and (z_arr[t] != z_arr[t-1]):
                cps.append(t)
        segs = [0] + cps + [T]
        segs = [segs[0]] + [q for p, q in zip(segs[:-1], segs[1:]) if q > p]
        per_asset_segs_on_cal[sec] = segs
    
    # global union: switches ∪ endpoints
    global_segs = sorted({0, len(R_df_all.index)} | set().union(*[set(s) for s in per_asset_segs_on_cal.values()]))

    # Print segments
    for sec in per_asset_segs_on_cal:
        print(f"[{sec}] raw segments: {per_asset_segs_on_cal[sec]}")
    print("\n[UNION] segments:", global_segs)

    # 6) Regime-DRO solves (time-varying universe; no coarsen)
    names_all = list(avail)
    taus = list(global_segs)
    w_list = []

    for k in range(len(taus) - 1):
        t_mid = min(max(taus[k], 0), T - 1)

        # active assets at t_mid: entered and finite label
        A_k = []
        for n in names_all:
            z = np.asarray(Z_labels[n], float)
            r = R_df_all[n].to_numpy(dtype=float, copy=False)
            if np.isfinite(z[t_mid]) and np.isfinite(r[t_mid]):
                A_k.append(n)
        
        print(f"[t={t_mid} | {R_df_all.index[t_mid].date()}] active: {A_k}")
        min_assets = int(DRO_CONFIG.get("min_assets", 1))
        if len(A_k) < min_assets:
            print(f"[WARN][t={t_mid} | {R_df_all.index[t_mid].date()}] only {len(A_k)} active assets (<{min_assets}).")
        
        if len(A_k) == 0:
            w_list.append(np.zeros(len(names_all))); continue

        R_df_k = R_df_all[A_k]  # keep NaNs; moments handle masks
        mu_ann, Sig_ann = pooled_moments_by_regime(
            R_df_k, {n: Z_labels[n] for n in A_k}, t_mid,
            ann=252, min_pair=int(DRO_CONFIG.get("min_seg_len_obs", 20)), mode="pairwise")

        data_k = {
            "train": R_df_k.fillna(0.0).to_numpy(dtype=float),  # content unused when using moments override
            "test":  R_df_k.fillna(0.0).to_numpy(dtype=float),
            "n_days": R_df_k.shape[0],
            "ann_factor": 252,
            "mu_ann_full": np.asarray(mu_ann, float),
            "Sigma_ann_full": np.asarray(Sig_ann, float),
            "px_cols": list(A_k),
            "index": R_df_k.index
        }
        paramsR = dict(DELTA_DEFAULTS[DRO_CONFIG["delta_name"]])
        paramsR["use_moments_override"] = True

        w_sub = np.asarray(fit_dro(data_k, paramsR, DRO_CONFIG["GLOBAL"])["w"]).reshape(-1)

        # expand to full vector
        w_full = np.zeros(len(names_all))
        pos = {n: i for i, n in enumerate(names_all)}
        for j, n in enumerate(A_k):
            w_full[pos[n]] = w_sub[j]
        w_list.append(w_full)

    fitB = {
        "type": "piecewise",
        "w_list": [np.asarray(w, float) for w in w_list],
        "segs": np.asarray(taus, dtype=np.int64),
        "names": names_all
    }

    # evaluation uses the union calendar
    data_eval = {
        "train": R_df_all.fillna(0.0).to_numpy(dtype=float),
        "test":  R_df_all.fillna(0.0).to_numpy(dtype=float),
        "n_days": T,
        "ann_factor": 252,
        "mu_ann_full": np.zeros(len(names_all)),
        "Sigma_ann_full": np.eye(len(names_all)),
        "px_cols": names_all,
        "index": R_df_all.index
    }
    # extra dataset digest for Part B eval
    print(f"[DEBUG:DATA_B] eval.shape={data_eval['train'].shape}  cols={names_all}")
    print(f"[DEBUG:DATA_B] sha(eval)={_sha(data_eval['train'])}")

    X = R_df_all.fillna(0.0).to_numpy(dtype=float)
    print(f"[DEBUG:EVAL] eval.shape={X.shape}  cols={list(names_all)}")
    print(f"[DEBUG:EVAL] eval.sha={hashlib.sha1(X.tobytes()).hexdigest()[:12]}")

    summB = evaluate_portfolio(fitB, data_eval, DRO_CONFIG["GLOBAL"])

    print("\n[Part A] Static DRO summary:\n", pd.Series(summA).round(4))
    print("\n[Part B] Regime-DRO summary:\n", pd.Series(summB).round(4))
    
    return {
        "PartA": {"fit": fitA, "data": dataA, "summary": summA},
        "PartB": {"fit": fitB, "summary": summB,
                  "per_asset_segs": per_asset_segs_on_cal, "global_segs": taus,
                  "Z_labels": Z_labels},
        "tickers": avail}
