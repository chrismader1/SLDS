# ---------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------

# GPU if available
try:
    import cupy as xp
    from cupyx.scipy.optimize import linear_sum_assignment  # GPU Hungarian
    from cupyx.scipy import stats as xp_stats               # for stats
    GPU = True
except Exception:
    import numpy as xp
    from scipy.optimize import linear_sum_assignment
    from scipy import stats as xp_stats
    GPU = False
print(f"GPU={GPU}")

# Others
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import os, re, ast
import pickle, gzip
from scipy import stats as sp_stats

import warnings
# kill the ECOS deprecation blurb from CVXPY’s solving_chain
warnings.filterwarnings(
    "ignore",
    message=".*ECOS will no longer be installed by default.*",
    category=FutureWarning,
    module="cvxpy.reductions.solvers.solving_chain",
)
# kill the "Solution may be inaccurate" user warning from CVXPY
warnings.filterwarnings(
    "ignore",
    message=".*Solution may be inaccurate.*",
    category=UserWarning,
    module="cvxpy.problems.problem",
)

# -------------------------
# IO
# -------------------------

def save_out(out: dict, path: str):
    """
    Save `out` dict to `path`. Use .pkl or .pkl.gz.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_out(path: str) -> dict:
    """
    Load dict saved by save_out.
    """
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    else:
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------
# Wasserstein helperS
# ---------------------------------------------------------------

def _rng_from_params(params):
    import numpy as _np
    seed = None if params is None else params.get("seed", None)
    return _np.random.default_rng(seed)

def _sqrtm_psd(A, eps=1e-12):
    """Symmetric PSD principal square root via eigen-decomposition."""
    vals, vecs = xp.linalg.eigh(0.5*(A + A.T))
    vals = xp.clip(vals, 0.0, None)
    return (vecs * xp.sqrt(vals + eps)) @ vecs.T

def w2_empirical_uniform_exact(X, Y):
    """
    Exact W2 between two uniform empirical measures with the same number of points.
    Returns W2 (not squared). Uses Hungarian assignment on squared Euclidean costs.
    """
    X = xp.asarray(X, float); Y = xp.asarray(Y, float)
    n, d = X.shape
    m, d2 = Y.shape
    assert d == d2, "X and Y must have the same dimension"
    assert n == m,  "Uniform empirical W2 requires equal sample sizes"
    # cost matrix C_{ij} = ||x_i - y_j||^2
    # C = ((X[:, None, :] - Y[None, :, :])**2).sum(axis=2)
    # r, c = linear_sum_assignment(C)
    # NOTE: exact Hungarian is cubic; keep for small n only
    if n > 4096:
        # fall back to sliced-W2 for safety on large n
        return float(sliced_w2_empirical(X, Y, n_proj=128, rng=None))
    C = ((X[:, None, :] - Y[None, :, :])**2).sum(axis=2)
    r, c = linear_sum_assignment(C)
    return float(xp.sqrt(C[r, c].mean()))

def wasserstein2_gaussian(mu1, Sigma1, mu2, Sigma2, eps=1e-12):
    """
    Gelbrich formula: W2^2(N(mu1,S1), N(mu2,S2)) =
      ||mu1-mu2||^2 + tr(S1 + S2 - 2 (S2^{1/2} S1 S2^{1/2})^{1/2})
    Returns W2 (not squared).
    """
    dmu2 = float(xp.dot(mu1 - mu2, mu1 - mu2))
    S2h = _sqrtm_psd(Sigma2, eps=eps)
    mid = S2h @ Sigma1 @ S2h
    midh = _sqrtm_psd(mid, eps=eps)
    trpart = float(xp.trace(Sigma1 + Sigma2 - 2.0 * midh))
    w2_sq = max(dmu2 + trpart, 0.0)
    return float(xp.sqrt(w2_sq))

def sliced_w2_empirical(X, Y, n_proj=256, rng=None):
    """
    1D sliced W2 between empirical measures using random projections.
    Unequal sample sizes use mid-quantiles q_i = (i-0.5)/k
    Avoids sorting when taking quantiles path (m != n)
    """
    X = xp.asarray(X, dtype=xp.float32)
    Y = xp.asarray(Y, dtype=xp.float32)
    n, d = X.shape
    m = Y.shape[0]

    # Generate directions on device when possible
    if (rng is None) and hasattr(xp.random, "standard_normal"):
        try:
            U = xp.random.standard_normal((n_proj, d), dtype=X.dtype)  # CuPy fast path
        except TypeError:
            U = xp.random.standard_normal((n_proj, d)).astype(X.dtype, copy=False)  # NumPy fallback
    else:
        rng = _rng_from_params({}) if rng is None else rng
        U = xp.asarray(rng.normal(size=(n_proj, d)), dtype=X.dtype)

    U = U / xp.maximum(xp.linalg.norm(U, axis=1, keepdims=True), 1e-12)

    XU = X @ U.T
    YU = Y @ U.T

    if m == n:
        # Equal sizes → sort and match order statistics
        XU = xp.sort(XU, axis=0)
        YU = xp.sort(YU, axis=0)
        diff = XU - YU
        w2_sq = xp.mean(diff * diff)
        return float(xp.sqrt(xp.maximum(w2_sq, 0.0)))
    else:
        # Unequal sizes → mid-quantiles without prior sort
        k = int(min(n, m))
        if k <= 1:
            XU = xp.mean(XU, axis=0, keepdims=True)
            YU = xp.mean(YU, axis=0, keepdims=True)
        else:
            q = (xp.arange(1, k + 1, dtype=XU.dtype) - 0.5) / k  # mid-quantiles
            XU = xp.quantile(XU, q, axis=0)
            YU = xp.quantile(YU, q, axis=0)
        diff = XU - YU
        w2_sq = xp.mean(diff * diff)
        return float(xp.sqrt(xp.maximum(w2_sq, 0.0)))

def _mvnrnd_psd(mu, Sigma, n, rng, eps=1e-9):
    """Draw n samples ~ N(mu, Sigma) with PSD projection; avoids SVD path."""
    mu = xp.asarray(mu, float); d = mu.size
    S  = 0.5*(Sigma + Sigma.T)
    vals, vecs = xp.linalg.eigh(S)
    L = (vecs * xp.sqrt(xp.clip(vals, eps, None))) @ vecs.T
    Z = rng.normal(size=(n, d))
    return mu + Z @ L.T

def _cov_batched(Xb: "xp.ndarray[B,n,d]"):
    """
    Batched sample mean & covariance on device.
    Returns (mub[B,d], Sb[B,d,d]) with ddof=1 and symmetrization.
    """
    B, n, d = Xb.shape
    mub = Xb.mean(axis=1)                       # (B, d)
    C   = Xb - mub[:, None, :]                  # (B, n, d)
    # einsum over batches: sum_k C_{b,k,:} C_{b,k,:}^T
    Sb  = xp.einsum('bij,bik->bjk', C, C) / max(n - 1, 1)
    # enforce symmetry (numerical)
    Sb  = 0.5 * (Sb + xp.transpose(Sb, (0, 2, 1)))
    return mub, Sb

def bootstrap_gaussian_delta(R, alpha=0.05, B=512, eps=1e-9, rng=None):
    """
    Batched Gaussian bootstrap for Wasserstein δ.
    All math stays on device (CuPy if available).
    """
    rng = _rng_from_params({}) if rng is None else rng

    X = xp.asarray(R, float)
    n, d = X.shape
    if n < 2:
        return 0.0

    # reference moments
    mu0 = xp.mean(X, axis=0)                    # (d,)
    Xc  = X - mu0
    S0  = (Xc.T @ Xc) / (n - 1)                 # (d,d)
    L   = _sqrtm_psd(S0, eps)                   # S0^{1/2}; L @ L = S0

    # batched draws on device
    Z   = xp.asarray(rng.normal(size=(B, n, d)))    # CPU->GPU once
    Xb  = mu0 + Z @ L.T                              # (B,n,d)

    # batched moments on device
    mub, Sb = _cov_batched(Xb)                       # (B,d), (B,d,d)

    # Gelbrich W2 across batches (loop over B, typically cheap)
    deltas = xp.empty(B, dtype=float)
    for b in range(B):
        deltas[b] = wasserstein2_gaussian(mu0, S0, mub[b], Sb[b], eps)

    # upper (1 - alpha) quantile
    return float(xp.quantile(deltas, 1.0 - alpha))

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
        raise ValueError("delta_method must be provided (no default).")
    method = params["delta_method"]

    # Enforce bootstrap sample count (no silent defaults)
    if method in ("bootstrap_np", "bootstrap_gaussian") and "B" not in params:
        raise ValueError("Bootstrap delta requires 'B' in params (match legacy value).")

    kappa  = float(kappa)

    # ----- κ-based rules
    if method == "kappa_rate":
        d     = int(xp.size(mu_est))
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n_eff = int((params or {}).get("n_ref", n_obs))
        sbar  = float(xp.sqrt(xp.trace(Sigma) / max(d, 1))) if Sigma is not None else 0.0
        return kappa * sbar * xp.sqrt(d / max(n_eff, 1))

    if method == "fixed":
        return float((params or {}).get("delta", 0.0))
    
    if method == "kappa_l2":
        return kappa * float(xp.linalg.norm(mu_est, 2))

    # ----- non-κ rules (ignore kappa)
    if method == "bound_ek":
        alpha = float((params or {}).get("alpha", 0.05))
        c1    = float((params or {}).get("c1", 3.0))     # inside the log
        C     = float((params or {}).get("c2", 1.0))     # outside the log (renamed use)
        a     = float((params or {}).get("a", 2.0))      # small-sample fallback exponent
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n     = int((params or {}).get("n_ref", n_obs))
        d     = int(xp.size(mu_est))
        num   = xp.log(c1 / max(alpha, 1e-12))           # log(c1/alpha)
        base  = (C * num) / max(n, 1)                    # C·log(c1/α) / n
        # threshold where asymptotic exponent becomes reliable
        n0    = float((params or {}).get("n0", 100.0))
        expo  = (1.0 / max(d, 2)) if (n >= n0) else (1.0 / max(a, 1e-12))
        return float(max(base, 1e-12) ** expo)

    if method == "bootstrap_np":
        # Fast replacement: sliced W2 with optional subsampling (no Hungarian; ~O(B·n_proj·n log n))
        alpha   = float((params or {}).get("alpha", 0.05))
        B       = int((params or {}).get("B", 256))
        n_proj  = int((params or {}).get("n_proj", 128))
        m_cap   = int((params or {}).get("m_cap", 4096))
        rng_np  = _rng_from_params(params or {})
        R_xp    = xp.asarray(R, dtype=xp.float32)
        n_src   = int(R_xp.shape[0])
        m       = int(min(n_src, m_cap))
        
        # ensure device RNG when on GPU (repro if seed provided)
        seed = (params or {}).get("seed", None)
        if GPU and (seed is not None):
            xp.random.seed(int(seed))
        
        dists = []
        for _ in range(B):
            if GPU:
                idx = xp.random.randint(0, n_src, size=m)          # device indices
            else:
                idx = rng_np.integers(0, n_src, size=m)            # NumPy on CPU
            idx = xp.asarray(idx, dtype=xp.int64)                  # ensure xp index dtype
            Rb  = R_xp[idx]
            dists.append(sliced_w2_empirical(R_xp[:m], Rb, n_proj=n_proj, rng=None))
        
        return float(xp.quantile(xp.asarray(dists), 1.0 - alpha))

    if method == "bootstrap_gaussian":
        assert R is not None, "bootstrap_gaussian needs raw sample matrix R."
        alpha = float((params or {}).get("alpha", 0.05))
        B     = int((params or {}).get("B", 512))
        eps   = float((params or {}).get("epsilon_sigma", 1e-9))
        rng   = _rng_from_params(params or {})
        return bootstrap_gaussian_delta(R, alpha=alpha, B=B, eps=eps, rng=rng)
    
    raise ValueError(f"Unknown delta_method='{method}'")

def psd_cholesky(Sigma, eps):
    """Symmetrize, regularize to PSD, then return lower Cholesky L with Σ ≈ L @ L.T."""
    Sigma_sym = 0.5 * (Sigma + Sigma.T)                 # symmetrize
    Sigma_reg = Sigma_sym + eps * xp.eye(Sigma_sym.shape[0])  # regularize
    try:
        L = xp.linalg.cholesky(Sigma_reg)
        return L
    except xp.linalg.LinAlgError:
        vals, vecs = xp.linalg.eigh(Sigma_sym)
        vals = xp.clip(vals, eps, None)                 # floor small/negative eigenvalues
        Sigma_psd = vecs @ xp.diag(vals) @ vecs.T + eps * xp.eye(Sigma_sym.shape[0])
        L = xp.linalg.cholesky(Sigma_psd)
        return L

def psd_factor_LtL(Sigma, eps):
    """
    Return L such that Sigma ≈ L.T @ L (so the constraint is ||L w|| ≤ rho).
    We form L by transposing a lower Cholesky.
    """
    import numpy as _np
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    Sigma_reg = Sigma_sym + eps * xp.eye(Sigma_sym.shape[0])
    try:
        C = xp.linalg.cholesky(Sigma_reg)
    except xp.linalg.LinAlgError:
        vals, vecs = xp.linalg.eigh(Sigma_sym)
        vals = xp.clip(vals, eps, None)
        Sigma_psd = vecs @ xp.diag(vals) @ vecs.T + eps * xp.eye(Sigma_sym.shape[0])
        C = xp.linalg.cholesky(Sigma_psd)
    return _np.asarray(C.T)   # ensure NumPy for CVXPY

def _sigma_unconditional(
    R_df: pd.DataFrame,
    t_idx: int,
    ann: int = 252,
    min_obs: int = 21,
    max_lookback: int = 1260,
    shrink_lambda: float = 0.0,
):
    """
    Unconditional (no regime conditioning) sample covariance from a rolling window
    [t0, t_idx], using log-returns and pairwise NaN-safe estimator, then annualized
    and shrunk toward scaled identity:
        Σ_shrunk = (1-λ) Σ + λ * (tr(Σ)/N) I,  λ in [0,1].
    Returns (Sigma_ann[N,N], ok: bool, counts: dict[col]->int).
    """
    import numpy as _np
    R_df = pd.DataFrame(R_df)
    T = len(R_df.index)
    t_idx = int(min(max(0, t_idx), T - 1))
    t0 = int(max(0, t_idx - int(max_lookback) + 1))

    X = R_df.to_numpy(_np.float64, copy=False)      # (T,N) possibly with NaNs
    L = xp.log1p(xp.asarray(X))                     # log-returns
    M = ~xp.isnan(L)                                # availability mask (T,N)

    # restrict window
    win = xp.zeros(T, dtype=bool); win[t0:t_idx+1] = True
    W = M & win[:, None]                            # rows used per asset

    N = L.shape[1]
    counts = {}
    for j, name in enumerate(R_df.columns):
        counts[str(name)] = int(W[:, j].sum())

    # require at least min_obs per asset
    if any(c < int(min_obs) for c in counts.values()):
        return xp.zeros((N, N)), False, counts

    # pairwise NaN-safe covariance on window
    n_i = W.sum(axis=0).astype(L.dtype)             # (N,)
    sums = (W * L).sum(axis=0)                      # (N,)
    means = xp.where(n_i > 0, sums / n_i, 0.0)
    Xc = xp.where(W, L - means[None, :], 0.0)

    N_ij = (W.astype(L.dtype)).T @ W.astype(L.dtype)
    S_ij = Xc.T @ Xc

    with xp.errstate(invalid="ignore", divide="ignore"):
        C_ij = xp.where(N_ij >= 2.0, S_ij / (N_ij - 1.0), 0.0)

    # set diagonal from unbiased per-asset sample variances
    for j in range(N):
        nj = int(n_i[j])
        if nj >= 2:
            xj = Xc[:, j][W[:, j]]
            C_ij[j, j] = float((xj @ xj) / (nj - 1))
        else:
            C_ij[j, j] = 0.0

    Sigma_ann = ann * C_ij

    # shrinkage toward scaled identity
    lam = float(max(0.0, min(1.0, shrink_lambda)))
    if lam > 0.0:
        s2_bar = float(xp.trace(Sigma_ann) / max(N, 1))
        Sigma_ann = (1.0 - lam) * Sigma_ann + lam * s2_bar * xp.eye(N)

    return Sigma_ann, True, counts

def solve_optimizer(mu, Sigma, delta, config, verbose=False):
    import numpy as _np
    n = len(mu)
    rho = float(config["risk_budget"])
    eps = float(config["epsilon_sigma"])

    Sigma = xp.asarray(Sigma, dtype=xp.float64)
    if not xp.isfinite(Sigma).all():
        Sigma = xp.nan_to_num(Sigma, nan=0.0, posinf=0.0, neginf=0.0)

    # Build L such that ||L w||_2 <= rho  (Sigma ≈ L.T @ L)
    L = _np.asarray(psd_factor_LtL(Sigma, eps))

    w = cp.Variable(n)
    t = cp.Variable(nonneg=True)

    mu_np = _np.asarray(mu, dtype=float)
    objective = cp.Minimize(float(delta) * t - mu_np @ w)

    long_only   = bool(config.get("long_only", False))
    no_leverage = bool(config.get("no_leverage", False))

    # Base SOC constraints (outer problem epigraph)
    constr = [
        cp.norm(L @ w, 2) <= rho,  # risk budget
        cp.norm(w, 2)    <= t,     # epigraph of ||w||_2
        t >= 0,
    ]

    # Enforce w >= 0 if requested
    if long_only:
        constr += [w >= 0]

    # Enforce sum(w) <= 1 (cash allowed, no leverage) if requested
    if no_leverage:
        constr += [cp.sum(w) <= 1]

    prob = cp.Problem(objective, constr)
    try:
        if verbose:
            print(f"[solve_optimizer] delta = {float(delta):.6g}, rho = {rho:.6g}")
        prob.solve(solver=cp.MOSEK, verbose=verbose)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=verbose)

    if (w.value is None) or (prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)):
        raise RuntimeError(f"ECOS/MOSEK failed: status={prob.status}")

    return xp.asarray(_np.asarray(w.value).reshape(-1))


# ---------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------

def fit_mvo_rebalanced(R_df: pd.DataFrame, G, ann: int, marks: list[int],
                       min_lb: int, max_lb: int, lam_shr: float, verbose: bool = False):

    """
    Piecewise MVO over rebalance marks using rolling windows with min/max lookback.
    """
    idx = R_df.index
    w_list, segs = [], marks
    N = R_df.shape[1]

    for a, b in zip(marks[:-1], marks[1:]):
        # rebalance at 'a' using window [ws, a)
        if a == 0:
            w_list.append(xp.zeros(N))
            continue
        ws = _window_start(a, min_lb, max_lb)
        R_win = R_df.iloc[ws:a].dropna(how="any")
        if len(R_win) < max(2, min_lb):
            w_list.append(w_list[-1] if w_list else xp.zeros(N))
            continue

        mu_ann, Sig_ann = _moments_from_window(R_win, ann=ann, shrink_lambda=lam_shr)
        w = solve_optimizer(mu_ann, Sig_ann, delta=0.0, config=G, verbose=False)
        if verbose:
            dt = idx[a]
            print(f"[MVO] t={a} {getattr(dt, 'date', lambda: dt)()}  delta=0.0000")
            _print_mu_by_name(R_win.columns.tolist(), mu_ann)
        w_list.append(w)

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": xp.nan, "delta_list": []}

def fit_dro_rebalanced(R_df: pd.DataFrame, params, G, ann: int, marks: list[int],
                       min_lb: int, max_lb: int, lam_shr: float, verbose: bool = False):

    """
    Piecewise static DRO over rebalance marks using rolling windows with min/max lookback.
    """
    idx = R_df.index
    w_list, segs, delta_list = [], marks, []
    N = R_df.shape[1]

    for a, b in zip(marks[:-1], marks[1:]):
        if a == 0:
            w_list.append(xp.zeros(N)); delta_list.append(xp.nan)
            continue
        ws = _window_start(a, min_lb, max_lb)
        R_win = R_df.iloc[ws:a].dropna(how="any")
        if len(R_win) < max(2, min_lb):
            # carry forward
            w_list.append(w_list[-1] if w_list else xp.zeros(N))
            delta_list.append(delta_list[-1] if delta_list else xp.nan)
            continue

        # window moments
        mu_ann, Sig_ann = _moments_from_window(R_win, ann=ann, shrink_lambda=lam_shr)
        # delta from window (pass raw sample as numpy array)
        delta = compute_delta(params.get("kappa", 1.0),
                              mu_ann, Sig_ann,
                              R=R_win.to_numpy(dtype=float),
                              params=params)
        w = solve_optimizer(mu_ann, Sig_ann, delta, config=G, verbose=bool(params.get("verbose", False)))
        if verbose:
            dt = idx[a]
            print(f"[DRO] t={a} {getattr(dt, 'date', lambda: dt)()}  delta={float(delta):.4f}")
            _print_mu_by_name(R_win.columns.tolist(), mu_ann)
        w_list.append(w); delta_list.append(float(delta))

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": params.get("kappa", xp.nan), "delta_list": delta_list}


def fit_dro(data, params, G):
    delta = compute_delta(params.get("kappa", 1.0),
                          data["mu_ann_full"], data["Sigma_ann_full"], data["train"], params)
    if bool(params.get("verbose", False)): 
        print(f"[DRO] delta = {float(delta):.6g}")
    w = solve_optimizer(data["mu_ann_full"], data["Sigma_ann_full"], delta,
                        G, verbose=bool(params.get("verbose", False)))
    return {"type": "static", "w": w, "kappa": params.get("kappa", xp.nan), "delta": float(delta)}

def fit_regime_dro(data, params, G):
    n_days = data["n_days"]
    AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))

    # Report segmentation before optimizing
    segs_preview, k_preview = _count_segments_from_params_or_data(data, params)
    _section(f"Regime-DRO — planned number of segments: {k_preview}")
    print(f"Segments (indices): {segs_preview}")

    # final segments actually used (resolve override / fn / midpoint logic)
    segs = params.get("segs")
    if segs is None:
        segs_fn = params.get("segs_fn", None)
        if segs_fn is not None:
            segs = segs_fn(data, params, G)
        else:
            taus  = data.get("taus_true", [0, n_days])
            delay = int(params.get("delay", 0))
            mids  = [int((taus[k-1] + taus[k]) / 2) for k in range(1, len(taus) - 1)]
            dets  = [min(m + delay, n_days - 1) for m in mids]
            for i in range(1, len(dets)):
                if dets[i] <= dets[i - 1]:
                    dets[i] = min(dets[i - 1] + 1, n_days - 1)
            segs = [0] + dets + [n_days]

    _section("Regime-DRO — segments to be used (final)")
    print(f"Final segments (indices): {segs}")

    # Start solving
    w_list, deltas = [], []        
    
    for a, b in zip(segs[:-1], segs[1:]):
        R_seg = data["train"][a:b]
        # regime-specific μ from segment; Σ = unconditional (full-sample here) with shrinkage
        if (b - a) < 2:
            mu_est = data["mu_ann_full"]
        else:
            log_seg = xp.log1p(R_seg)
            mu_est  = xp.expm1(log_seg.mean(axis=0) * AF)

        # unconditional Σ from rolling window ending at b-1 over the current asset set (full panel here)
        min_obs = int(params.get("min_lookback_days", 21))
        max_lb  = int(params.get("max_lookback_days", 1260))
        lam_shr = float(params.get("sigma_shrinkage_lambda", 0.0))
        
        # Build a DataFrame for the helper (columns optional but helpful for counts)
        import numpy as _np
        R_df_full = pd.DataFrame(_np.asarray(data["train"], dtype=float),
                                 columns=list(data.get("px_cols", range(data["train"].shape[1]))))
        
        t_for_sigma = max(0, min(int(b) - 1, int(data["n_days"]) - 1))
        Sigma_est, ok_sig, _ = _sigma_unconditional(
            R_df_full, t_idx=t_for_sigma, ann=AF,
            min_obs=min_obs, max_lookback=max_lb, shrink_lambda=lam_shr,
        )
        if not ok_sig:
            Sigma_est = xp.asarray(data["Sigma_ann_full"], float)
        else:
            Sigma_est = xp.asarray(Sigma_est, float)

        R_source  = R_seg
    
        # pass full-sample N via n_ref but bootstrap from R_source
        params_k = dict(params); params_k["n_ref"] = (b - a)   # use segment length

        delta_k = compute_delta(params_k.get("kappa", 1.0), mu_est, Sigma_est, R_source, params_k)
        if bool(params.get("verbose", False)):
            print(f"[RegDRO] segment [{a},{b})  delta_k = {float(delta_k):.6g}")
        w_k = solve_optimizer(mu_est, Sigma_est, delta_k, G, verbose=bool(params.get("verbose", False)))
        deltas.append(float(delta_k)); w_list.append(w_k)
        
    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": params.get("kappa", xp.nan),
            "delta_list": deltas,
            "delta": xp.nan}

def fit_dro_reverse(data, params, G):
    """
    Reverse-optimised scalar δ (provided by caller).
    params: {"delta": <float>}
    """
    delta = float(params["delta"])
    if bool(params.get("verbose", False)):
        print(f"[DRO-REV] delta = {delta:.6g}")
    w = solve_optimizer(
        data["mu_ann_full"], data["Sigma_ann_full"], delta,
        G, verbose=bool(params.get("verbose", False)))
    return {"type": "static", "w": w, "delta": delta, "kappa": xp.nan}

def fit_regime_dro_reverse(data, params, G):
    """
    Reverse-optimised per-segment deltas.
    Uses regime-specific μ (from each segment) and UNCONDITIONAL Σ built from a
    rolling lookback window (min/max) with shrinkage toward identity.
    params must include:
        - "delta_list": [δ1,...,δK]
        - optionally "segs" or "segs_fn"
        - "min_lookback_days", "max_lookback_days"
        - "sigma_shrinkage_lambda" in [0,1]
        - "verbose": bool
    """
    n_days = int(data["n_days"])
    AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))
    segs = params.get("segs")
    if segs is None:
        segs_fn = params.get("segs_fn")
        assert segs_fn is not None, "fit_regime_dro_rev: provide 'segs' or 'segs_fn'."
        segs = segs_fn(data, params, G)

    delta_list = list(map(float, params["delta_list"]))
    assert len(delta_list) == len(segs) - 1, "delta_list length must equal number of segments."

    min_obs   = int(params.get("min_lookback_days", 21))
    max_lb    = int(params.get("max_lookback_days", 1260))
    lam_shr   = float(params.get("sigma_shrinkage_lambda", 0.0))
    lam_shr   = max(0.0, min(1.0, lam_shr))
    be_verbose = bool(params.get("verbose", False))

    X_full = xp.asarray(data["train"], float)  # unconditional returns panel (daily)

    w_list = []
    for j, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
        # ---- regime-specific μ from this segment ----
        R_seg = X_full[a:b]
        if (b - a) < 2:
            mu_est = xp.asarray(data["mu_ann_full"], float)
        else:
            mu_est = xp.expm1(xp.log1p(R_seg).mean(axis=0) * AF)

        # ---- unconditional, windowed Σ with shrinkage ----
        # window ends at b (exclusive), look back up to max_lb, but enforce min_obs
        t_end = int(b)
        t_start = max(0, t_end - max_lb)
        W = X_full[t_start:t_end]              # window [t_start, b)
        # if window too short, try expanding to start of sample; else fall back
        if W.shape[0] < min_obs:
            W = X_full[0:t_end]
        if W.shape[0] >= 2:
            Lw = xp.log1p(W)
            Sig_d = xp.cov(Lw.T, ddof=1)      # daily
            Sig_ann = Sig_d * AF
            if lam_shr > 0.0:
                N = Sig_ann.shape[0]
                s2_bar = float(xp.trace(Sig_ann) / max(N, 1))
                Sig_ann = (1.0 - lam_shr) * Sig_ann + lam_shr * s2_bar * xp.eye(N)
            Sigma_est = Sig_ann
            sigma_src = f"unconditional_window[{t_start}:{t_end}), shrinkage={lam_shr:.3f}"
        else:
            Sigma_est = xp.asarray(data["Sigma_ann_full"], float)
            sigma_src = "fallback_data_Sigma_ann_full"

        # ---- verbose diagnostics ----
        if be_verbose:
            print(f"[RegDRO-REV] k={j+1}  seg=[{a},{b})  delta={delta_list[j]:.6g}")
            print(f"[RegDRO-REV] k={j+1}  Sigma source: {sigma_src}")

        # ---- solve ----
        w_k = solve_optimizer(mu_est, Sigma_est, delta_list[j], G, verbose=be_verbose)
        w_list.append(w_k)

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "delta_list": delta_list, "kappa": xp.nan}

def fit_regime_dro_rev_constSigma(data, params, G):
    segs = params["segs"]
    Sigma_fix = data["Sigma_ann_full"]          # constant across segments
    w_list = []
    for j, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
        R_seg = data["train"][a:b]
        log_seg = xp.log1p(R_seg)
        AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))
        mu_est = xp.expm1(log_seg.mean(axis=0) * AF)
        if bool(params.get("verbose", False)):
            print(f"[RegDRO-REV-ConstΣ] segment {j+1} [{a},{b})  delta_k = {float(params['delta_list'][j]):.6g}")
        w = solve_optimizer(mu_est, Sigma_fix, float(params["delta_list"][j]),
                            G, verbose=bool(params.get("verbose", False)))
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
        ge = float(xp.sum(xp.abs(fit["w"])))
        port_tr = train @ fit["w"]
        _, risk_tr, _ = stats_from_series(port_tr, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})
        stats_oos["gross_exp"] = ge
        stats_oos["gap"] = float(stats_oos["sigma_ann"] - risk_tr)
        stats_oos["kappa"] = float(fit.get("kappa", xp.nan))
        stats_oos["delta"] = float(fit.get("delta", xp.nan))
        rebal = [0, n_days]
        stats_oos["avg_holding_per"] = _avg_holding_period_from_marks(rebal)
        return stats_oos
    
    else:  # piecewise
        cfg = {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "risk_budget": G["risk_budget"], "annualization_factor": AF}
        stats_oos = portfolio_stats_multipiece(fit["w_list"], fit["segs"], test, cfg)
        seg_lengths = xp.diff(xp.array(fit["segs"]))
        ge = float(xp.sum(seg_lengths * xp.array([xp.sum(xp.abs(wk)) for wk in fit["w_list"]])) / n_days)

        port_tr = xp.zeros(n_days)
        for (a,b), wk in zip(zip(fit["segs"][:-1], fit["segs"][1:]), fit["w_list"]):
            port_tr[a:b] = train[a:b] @ wk
        _, risk_tr, _ = stats_from_series(port_tr, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})

        stats_oos["gross_exp"] = ge
        stats_oos["gap"] = float(stats_oos["sigma_ann"] - risk_tr)
        stats_oos["kappa"] = float(fit.get("kappa", xp.nan))

        # Aggregate per-segment deltas
        dlist = xp.asarray(fit.get("delta_list", []), dtype=float)
        if dlist.size:
            stats_oos["delta_mean"] = float(xp.nanmean(dlist))
            stats_oos["delta_min"]  = float(xp.nanmin(dlist))
            stats_oos["delta_max"]  = float(xp.nanmax(dlist))
        else:
            stats_oos["delta_mean"] = xp.nan
            stats_oos["delta_min"]  = xp.nan
            stats_oos["delta_max"]  = xp.nan
        stats_oos["delta"] = xp.nan  # keep legacy key empty for Regime-DRO
                
        rebal = list(fit.get("segs", []))
        if not rebal:
            # fallback if segs missing: treat as static
            rebal = [0, n_days]
        stats_oos["avg_holding_per"] = _avg_holding_period_from_marks(rebal)
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
        mu_seg, sigma_seg, sharpe_seg, vol_breach_seg = xp.nan, xp.nan, xp.nan, xp.nan
        gross_exp_seg = xp.sum(xp.abs(wk))

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
    sigma_daily = xp.std(port_daily, ddof=1)
    sigma_annual = sigma_daily * xp.sqrt(AF)
    mu_annual_geom = xp.exp(AF * xp.mean(xp.log1p(port_daily))) - 1
    sharpe_annual = (xp.mean(port_daily) - rf_daily) / sigma_daily * xp.sqrt(AF) if sigma_daily > 0 else xp.nan
    return float(mu_annual_geom), float(sigma_annual), float(sharpe_annual)

def _max_drawdown_from_series(port_daily):
    """
    Max drawdown of a daily-return series.
    Returns the minimum (most negative) drawdown, e.g. -0.27 for -27%.
    """
    x = xp.asarray(port_daily, float)
    if x.size == 0:
        return float("nan")
    equity = xp.cumprod(1.0 + x)
    peak = xp.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(xp.min(dd))

def portfolio_stats(weights, returns, config):
    """Static weights over full horizon."""
    weights = xp.asarray(weights).reshape(-1)
    port_daily = returns @ weights
    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    max_dd = _max_drawdown_from_series(port_daily)
    return {
        "mu_ann": mu_annual_geom,
        "sigma_ann": sigma_annual,
        "sharpe_ann": sharpe_annual,
        "vol_breach": vol_breach,
        "max_drawdown": max_dd,}

def portfolio_stats_multipiece(w_list, taus, returns, config):
    """
    w_list: list of weights per piece, length = len(taus)-1
    taus:   [0=τ0, τ1, ..., τK=n_days]
    """
    n_days = config["n_days"]
    assert taus[0] == 0 and taus[-1] == n_days and len(w_list) == len(taus) - 1
    port_daily = xp.empty(n_days, dtype=float)
    for k in range(len(w_list)):
        a, b = taus[k], taus[k + 1]
        port_daily[a:b] = returns[a:b] @ xp.asarray(w_list[k]).reshape(-1)
    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    max_dd = _max_drawdown_from_series(port_daily)
    return {
        "mu_ann": mu_annual_geom,
        "sigma_ann": sigma_annual,
        "sharpe_ann": sharpe_annual,
        "vol_breach": vol_breach,
        "max_drawdown": max_dd,}

# ---------------------------------------------------------------
# Statistical tests (for hypothesis testing)
# ---------------------------------------------------------------

def format_ci(mean, std):
    return f"{mean:.4f} ({(mean - std):.4f}, {(mean + std):.4f})"
    
def paired_onesided_less(x, y):
    # H0: mean(x - y) >= 0  vs  H1: mean(x - y) < 0
    d = x - y
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="less")
    return t, p

def paired_t_twosided(x, y):
    # H0: mean(x - y) == 0  vs  H1: ≠ 0
    d = x - y
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    return t, p

def noninferiority_paired(x, y, delta):
    # H0: mean(x - y) <= -delta  vs  H1: > -delta
    d = (x - y) + delta
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="greater")
    return t, p

def superiority_paired(x, y):
    # H0: mean(x - y) <= 0  vs  H1: > 0
    d = x - y
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="greater")
    return t, p

def paired_two_sided_test_with_ci(x, y, alpha=0.05):
    """
    Paired two-sided t-test for mean(x - y) == 0 with 95% CI for the mean difference.
    Returns dict(mean_diff, t, p, ci_low, ci_high, n).
    """
    d = x - y
    n = len(d)
    mean_diff = xp.mean(d)
    sd = xp.std(d, ddof=1)
    se = sd / xp.sqrt(n)
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    tcrit = sp_stats.t.ppf(1 - alpha / 2, df=n - 1)
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
                 "gross_exp","kappa","gap","delta_mean","delta_min","delta_max","max_drawdown"]
    if model_order is None:
        model_order = list(results.keys())

    rows = base_rows
    table = {}

    for m in model_order:
        if m not in results:
            continue
        df = results[m]
        s = {}

        # standard metrics
        for col in base_rows:
            if col in df.columns and len(df[col].dropna()) > 0:
                s[col] = _fmt_series(df[col])

        # probability of breach column (derived)
        if "vol_breach" in df.columns:
            z = (df["vol_breach"] > 0).astype(float)
            s["p_viol"] = _fmt_series(z)

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
    sigma_train_ann_assets = xp.sqrt(AF) * returns_train.std(axis=0, ddof=1)

    # exact constraint metric (matches solver): ||L w||_2 with L^T L ≈ Σ_ann
    L = psd_factor_LtL(Sigma_ann, config["epsilon_sigma"])
    risk_train_ann = float(xp.linalg.norm(L @ w))
    tol = max(atol, rtol * max(rho, risk_train_ann))
    ok_train = bool(risk_train_ann <= rho + tol)

    # returns (annualized)
    ret_train_ann = float(mu_train_ann_assets @ w)

    # OOS realized vol like multi-trial breach (from series)
    port_eval = returns_eval @ w
    _, risk_eval_ann, _ = stats_from_series(port_eval, dict(config, annualization_factor=AF))
    mu_eval_ann_assets = AF * returns_eval.mean(axis=0)
    ret_eval_ann = float(mu_eval_ann_assets @ w)

    gross_exposure = float(xp.sum(xp.abs(w)))
    top_idx = xp.argsort(w)[-3:][::-1]
    nz = xp.where(w != 0)[0]
    bot_idx = nz[xp.argsort(w[nz])[:3]] if nz.size else xp.array([], dtype=int)

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
    port_train = xp.zeros(n_days); port_eval = xp.zeros(n_days)
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
    sigma_train_ann_assets = xp.sqrt(AF) * returns_train.std(axis=0, ddof=1)

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

    if len(seg_deltas):
        import numpy as _np
        _d = _np.array(seg_deltas, dtype=float)
        d_mean = _np.nanmean(_d) if _d.size else _np.nan
        d_min  = _np.nanmin(_d)  if _d.size else _np.nan
        d_max  = _np.nanmax(_d)  if _d.size else _np.nan
        print(f"\nPer-piece δ summary — mean: {d_mean:.4f}, min: {d_min:.4f}, max: {d_max:.4f}")
    
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
            sig_seg_ann = xp.sqrt(AF) * returns_train[a:b].std(axis=0, ddof=1)
        else:
            sig_seg_ann = sigma_train_ann_assets

        top_idx = xp.argsort(w)[-3:][::-1]
        nz = xp.where(w != 0)[0]
        bot_idx = nz[xp.argsort(w[nz])[:3]] if nz.size else xp.array([], dtype=int)

        print(f"\nPiece {k+1}  days [{a},{b}):")
        print("Top 3 assets with largest weights:")
        for i in top_idx:
            print(f"Asset {i:2d}: weight = {w[i]:+.4f}, μ = {mu_seg_ann[i]:+.4f}, σ = {sig_seg_ann[i]:.4f}")
        print("Top 3 assets with smallest nonzero weights:")
        for i in bot_idx:
            print(f"Asset {i:2d}: weight = {w[i]:+.4f}, μ = {mu_seg_ann[i]:+.4f}, σ = {sig_seg_ann[i]:.4f}")

def _avg_holding_period_from_marks(rebal_marks):
    """
    Avg holding period = max(rebal)/(len(rebal)-1)
    where `rebal` is a list/array of rebalance indices (e.g., [0, ..., T]).
    """
    if rebal_marks is None:
        return float("nan")
    r = [int(x) for x in rebal_marks]
    if len(r) <= 1:
        return float("nan")
    return float(max(r) / (len(r) - 1))

def _fmt4(a):
    return xp.array2string(
        xp.asarray(a, float),
        separator=' ',
        formatter={'float_kind': lambda x: f"{x:.4f}"})

def _print_mu_by_name(names, mu_vec, prefix="   "):
    names = list(names)
    mu_vec = xp.asarray(mu_vec, float).ravel()
    s = ", ".join(f"{names[i]}:{float(mu_vec[i]):+.4f}" for i in range(len(names)))
    print(prefix + "mu_ann: [" + s + "]")

def _count_segments_from_params_or_data(data, params):
    """
    Returns (segs, n_segments) using the exact logic that fit_regime_dro uses,
    but WITHOUT solving anything. This lets us announce segment count beforehand.
    """
    n_days = data["n_days"]
    segs = params.get("segs")
    if segs is None:
        segs_fn = params.get("segs_fn", None)
        if segs_fn is not None:
            segs = segs_fn(data, params, {})
        else:
            # old midpoint default (same as fit_regime_dro)
            taus  = data.get("taus_true", [0, n_days])
            delay = int(params.get("delay", 0))
            mids  = [int((taus[k-1] + taus[k]) / 2) for k in range(1, len(taus) - 1)]
            dets  = [min(m + delay, n_days - 1) for m in mids]
            for i in range(1, len(dets)):
                if dets[i] <= dets[i - 1]:
                    dets[i] = min(dets[i - 1] + 1, n_days - 1)
            segs = [0] + dets + [n_days]
    return segs, (len(segs) - 1)

def report_mvo(fit, data, G, label="MVO"):
    _section(f"{label}: static portfolio")
    print_single_portfolio_block(
        label=f"[{label}] Weights & stats",
        w=fit["w"],
        returns_train=data["train"],
        returns_eval=data["test"],
        rho=G["risk_budget"],
        Sigma_ann=data["Sigma_ann_full"],
        config=dict(G, n_days=data["n_days"], annualization_factor=int(data.get("ann_factor", 252)))
    )

def report_dro(fit, data, G, label="DRO"):
    _section(f"{label}: static DRO portfolio")
    print(f"delta: {fit.get('delta', float('nan')):.6f} | kappa: {fit.get('kappa', float('nan')):.6f}")
    print_single_portfolio_block(
        label=f"[{label}] Weights & stats",
        w=fit["w"],
        returns_train=data["train"],
        returns_eval=data["test"],
        rho=G["risk_budget"],
        Sigma_ann=data["Sigma_ann_full"],
        config=dict(G, n_days=data["n_days"], annualization_factor=int(data.get("ann_factor", 252)))
    )

def report_regdro(fit, data, G, taus_true=None, label="RegDRO"):
    _section(f"{label}: piecewise portfolio")
    segs = fit.get("segs", [])
    dlist = [fit.get(f"delta_k{k+1}", None) for k in range(len(segs)-1)]
    if not any(np.isfinite(dlist)):
        dlist = list(map(float, fit.get("delta_list", [])))
    if taus_true is None:
        taus_true = data.get("taus_true", [0, data["n_days"]])

    print_regime_block(
        label=f"[{label}] Weights & stats by piece",
        returns_train=data["train"],
        returns_eval=data["test"],
        w_list=fit["w_list"],
        segs=segs,
        rho=G["risk_budget"],
        taus_display=taus_true,
        seg_deltas=[float(x) if x is not None else float("nan") for x in dlist],
        config=dict(G, n_days=data["n_days"], annualization_factor=int(data.get("ann_factor", 252)))
    )

def report_all(models_results: dict, model_order=None, title="OOS Summary"):
    """
    models_results: {"MVO": {"fit":..., "data":..., "summary":...}, "DRO": {...}, "RegDRO": {...}}
    """
    # 1) sectioned blocks
    if "MVO" in models_results:
        report_mvo(models_results["MVO"]["fit"], models_results["MVO"]["data"], models_results["G"], label="MVO")
    if "DRO" in models_results:
        report_dro(models_results["DRO"]["fit"], models_results["DRO"]["data"], models_results["G"], label="DRO")
    if "RegDRO" in models_results:
        taus_true = models_results["RegDRO"]["data"].get("taus_true", None) if "data" in models_results["RegDRO"] else None
        report_regdro(models_results["RegDRO"]["fit"], models_results["RegDRO"]["data"], models_results["G"], taus_true=taus_true, label="RegDRO")

    # 2) unified table (clear MVO/DRO/RegDRO columns)
    _section(title)
    results_dict = {}
    if "MVO" in models_results:   results_dict["MVO"]   = pd.DataFrame([models_results["MVO"]["summary"]])
    if "DRO" in models_results:   results_dict["DRO"]   = pd.DataFrame([models_results["DRO"]["summary"]])
    if "RegDRO" in models_results:results_dict["RegDRO"]= pd.DataFrame([models_results["RegDRO"]["summary"]])

    if not model_order:
        model_order = [m for m in ["MVO","DRO","RegDRO"] if m in results_dict]
    print_oos_table(results_dict, model_order=model_order)

def _section(title: str):
    print("\n" + "="*72)
    print(str(title))
    print("="*72 + "\n")

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

def map_labels_to_calendar(z_ser: pd.Series, cal: pd.DatetimeIndex) -> xp.ndarray:
    """
    Map labels only on exact matching dates in `cal`.
    No forward/back fill. Missing elsewhere.
    """
    z = pd.Series(z_ser).sort_index()
    z.index = pd.to_datetime(z.index)
    cal = pd.DatetimeIndex(cal)
    out = pd.Series(xp.nan, index=cal)
    inter = cal.intersection(z.index)
    if len(inter):
        out.loc[inter] = z.reindex(inter).to_numpy()
    return out.to_numpy(dtype=float)

def build_calendar_union_px(securities, px_all, start_dt, end_dt):
    """UNION of PX return dates over [start_dt,end_dt]."""
    import numpy as _np
    cal = None
    for t in securities:
        px_ser = _num_series(px_all[t].loc[start_dt:end_dt]).dropna()
        idx = _np.log(px_ser).diff().dropna().index # dates where return exists
        cal = idx if cal is None else cal.union(idx)
    if cal is None or len(cal) == 0:
        raise RuntimeError("Empty calendar after applying start/end.")
    return pd.DatetimeIndex(cal)

def make_data_from_returns_panel(R: pd.DataFrame, ann_factor=252):
    import numpy as _np
    R = pd.DataFrame(R).astype(_np.float64)
    X_np = R.to_numpy(_np.float64, copy=False)
    X = xp.asarray(X_np)                   # move to GPU once
    T, N = X.shape
    logR = xp.log1p(X)
    mu_d  = logR.mean(axis=0)

    # --- guard: avoid NaN Σ when T < 2 ---
    if T < 2:
        Sig_d = xp.zeros((N, N), dtype=logR.dtype)
    else:
        Sig_d = xp.cov(logR.T, ddof=1)

    return {
        "train": X, "test": X, "n_days": T, "ann_factor": ann_factor,
        "mu_ann_full": xp.expm1(mu_d * ann_factor), "Sigma_ann_full": Sig_d * ann_factor,
        "px_cols": R.columns.tolist(), "index": R.index}

def make_data_from_returns_panel_pairwise(R: pd.DataFrame, ann_factor=252, min_pair=2):
    """
    Pairwise covariance on union calendar with NaNs allowed.
    """
    import numpy as _np
    R = pd.DataFrame(R).astype(_np.float64)
    X = R.to_numpy(_np.float64, copy=False)             # (T,N), may have NaNs
    T, N = X.shape
    L = xp.log1p(xp.asarray(X))                # (T,N) with NaNs
    # per-asset means over available rows
    M = ~xp.isnan(L)
    n_i = M.sum(axis=0).astype(L.dtype)                 # (N,)
    sums = xp.nan_to_num(L).sum(axis=0)
    means = xp.where(n_i > 0, sums / xp.maximum(n_i, 1.0), 0.0)
    Xc = xp.where(M, L - means[None, :], 0.0)

    # pairwise counts and cross-sums
    N_ij = (M.astype(L.dtype)).T @ M.astype(L.dtype)    # (N,N)
    S_ij = Xc.T @ Xc                                    # (N,N)

    with xp.errstate(invalid="ignore", divide="ignore"):
        C = xp.where(N_ij >= max(min_pair, 2), S_ij / (N_ij - 1.0), 0.0)

    # set diagonal with per-asset sample variances if enough obs
    for i in range(N):
        ni = int(n_i[i])
        if ni >= 2:
            xi = Xc[:, i]
            C[i, i] = float((xi @ xi) / (ni - 1))
        else:
            C[i, i] = 0.0

    mu_d = xp.nanmean(L, axis=0)
    Sig_d = C
    return {
        "train": xp.nan_to_num(xp.asarray(X), nan=0.0),   # content not used by solver when moments override
        "test":  xp.nan_to_num(xp.asarray(X), nan=0.0),
        "n_days": T, "ann_factor": ann_factor,
        "mu_ann_full": xp.expm1(mu_d * ann_factor),
        "Sigma_ann_full": Sig_d * ann_factor,
        "px_cols": R.columns.tolist(), "index": R.index}

def pooled_moments_by_regime(
    R_df: pd.DataFrame,
    Z_labels: dict,
    A_names: list,
    t_idx: int,
    ann: int = 252,
    min_obs: int = 21,
    lookback: int = 5*252,
    mode: str = "pairwise",
):
    """
    Windowed, regime-conditioned moments for the ACTIVE asset set A_names.

    For each active asset i in A_names, we:
       - take its current regime s_i = z_i[t_idx]
       - collect ONLY past rows within the window [t0, t_idx] where asset i is in regime s_i
       - require at least `min_obs` rows per asset; else return ok=False.

    Pairwise covariance is computed on the intersection of each asset's *own* regime masks
    (i.e., rows where asset i is in s_i AND asset j is in s_j), restricted to the time window.

    Returns:
        (mu_ann[N], Sig_ann[N,N], ok: bool, counts: dict[name]->int)
    """
    assert set(A_names).issubset(set(R_df.columns)), "A_names must be subset of R_df columns"

    names = list(A_names)
    N = len(names)
    T = len(R_df.index)
    t_idx = int(min(max(0, t_idx), T - 1))
    t0 = int(max(0, t_idx - int(lookback) + 1))

    # window mask: only look BACK up to lookback, including t_idx
    win = xp.zeros(T, dtype=bool)
    win[t0:t_idx+1] = True

    # data (log-returns; NaNs allowed in R_df)
    L = xp.log1p(xp.asarray(R_df[names].values))  # (T, N), may contain NaNs
    M_avail = ~xp.isnan(L)                        # availability mask

    # current regime state s_i per asset at t_idx (must exist and be finite)
    s = []
    valid = []
    for n in names:
        zi = xp.asarray(Z_labels[n], float)
        if zi.shape[0] != T:
            raise ValueError("Z_labels arrays must have length T after mapping to calendar.")
        if xp.isfinite(zi[t_idx]):
            s.append(int(zi[t_idx]))
            valid.append(True)
        else:
            s.append(xp.nan)
            valid.append(False)

    # if any active asset has no current regime at t_idx, fail fast (skip date)
    if not all(valid):
        counts = {n: 0 for n in names}
        return xp.zeros(N), xp.zeros((N, N)), False, counts

    # per-asset regime mask within the time window
    G_mask = xp.zeros((T, N), dtype=bool)
    for j, n in enumerate(names):
        zi = xp.asarray(Z_labels[n], float)
        G_mask[:, j] = (xp.isfinite(zi) & (zi == s[j]) & win)

    # per-asset windowed regime counts and means/vars
    counts = {}
    mu_d = xp.zeros(N, dtype=L.dtype)
    var_d = xp.zeros(N, dtype=L.dtype)
    for j, n in enumerate(names):
        Sj = xp.where(G_mask[:, j] & M_avail[:, j])[0]
        counts[n] = int(Sj.size)
        if counts[n] < int(min_obs):
            # insufficient per-asset regime history in window
            return xp.zeros(N), xp.zeros((N, N)), False, counts
        lj = L[Sj, j]
        mu_d[j]  = float(xp.nanmean(lj))
        var_d[j] = float(xp.nanvar(lj, ddof=1)) if lj.size > 1 else 0.0

    # covariance
    if mode == "diag":
        Sig_d = xp.diag(var_d)
    else:
        # W mask: rows used per asset = in-window & in-current-regime & available
        W = G_mask & M_avail                              # (T, N)
        # counts per asset (already checked >= min_obs)
        n_i = W.sum(axis=0, dtype=L.dtype)               # (N,)
        # sums per asset over active rows
        sums = (W * L).sum(axis=0)                       # (N,)
        means = xp.where(n_i > 0, sums / n_i, 0.0)
        Xc = xp.where(W, L - means[None, :], 0.0)

        # pairwise overlaps and cross-sums
        N_ij = (W.astype(L.dtype)).T @ W.astype(L.dtype)  # (N,N)
        S_ij = Xc.T @ Xc                                   # (N,N)

        with xp.errstate(invalid="ignore", divide="ignore"):
            C_ij = xp.where(N_ij >= 2.0, S_ij / (N_ij - 1.0), 0.0)

        # set diagonal to unbiased sample variances computed above
        for j in range(N):
            C_ij[j, j] = var_d[j]
        Sig_d = C_ij

    mu_ann  = xp.expm1(ann * mu_d)
    Sig_ann = ann * Sig_d
    return mu_ann, Sig_ann, True, counts

def _select_best_config(results_df, security, prefer_configs=None):
    """
    For a given `security`:
      (a) if `prefer_configs` provided → restrict to those config NAMES
          (accept list of strings or list of dicts with 'config');
      (b) else → use ALL rows for that security.
    Then select by: score ↓, n_regimes ↑, dim_latent ↑ (sum if vector).
    Return the chosen `config` string (or None).
    """
    import numpy as np, pandas as pd, ast, re

    if results_df is None or len(results_df) == 0:
        return None
    if "security" not in results_df.columns or "config" not in results_df.columns:
        return None

    df = results_df.copy()
    df["security"] = df["security"].astype(str).str.strip()
    df["config"]   = df["config"].astype(str).str.strip()

    # filter by security
    sec = str(security).strip()
    df = df[df["security"] == sec]
    if df.empty:
        return None

    # (a) restrict to prefer_configs (by NAME) if provided
    if prefer_configs:
        names = []
        for x in prefer_configs:
            if isinstance(x, dict) and "config" in x:
                names.append(str(x["config"]).strip())
            else:
                names.append(str(x).strip())
        mask = df["config"].isin(names)
        df = df[mask]
        if df.empty:
            return None  # nothing matches preferences

    # helpers to extract K and dim
    to_num = lambda x: pd.to_numeric(x, errors="coerce")

    def parse_K(row):
        for kcol in ("n_regimes","K","k","nStates","n_states"):
            if kcol in row and pd.notna(row[kcol]):
                return to_num(row[kcol])
        m = re.search(r"[Kk]\s*=?\s*(\d+)", str(row.get("config","")))
        return float(m.group(1)) if m else np.nan

    def dim_metric(row):
        v = row.get("dim_latent", np.nan)
        if isinstance(v, str):
            try: v = ast.literal_eval(v)
            except Exception: pass
        if isinstance(v, (list, tuple)):
            s = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
            return float(s.sum()) if len(s) else xp.nan
        return float(to_num(v))

    df["score_num"] = to_num(df.get("score", xp.nan))
    df["K_num"]     = [parse_K(r)    for _, r in df.iterrows()]
    df["D_num"]     = [dim_metric(r) for _, r in df.iterrows()]

    df = df.sort_values(["score_num","K_num","D_num"],
                        ascending=[False, True, True],
                        na_position="last")

    if df.empty or pd.isna(df.iloc[0]["score_num"]):
        return None
    return str(df.iloc[0]["config"])

def _labels_from_segments_df(segments_df, security, config):
    df = segments_df[(segments_df["security"] == security) &
                     (segments_df["config"] == config)].copy()
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "z"])
    # keep the last label for a duplicated date (choose policy if you want 'first')
    df = df.drop_duplicates(subset="date", keep="last")
    return pd.Series(
        df["z"].astype(int).to_numpy(),
        index=pd.DatetimeIndex(df["date"]),
        name="z",)

def _all_zero_weights(w, tol=1e-12) -> bool:
    w = xp.asarray(w, float).ravel()
    return bool(xp.all(xp.abs(w) <= tol))

def _eligible_idx(active, counts, min_obs):
    """Return indices into `active` that satisfy the min_obs constraint."""
    return [i for i, name in enumerate(active) if counts.get(name, 0) >= min_obs]

def _slice_data_for_idx(data, idx):
    """
    Slice your data dict to the selected columns (assets).
    Expects keys: 'train' (T x N), 'mu_ann_full' (N,), 'Sigma_ann_full' (N x N).
    """
    out = dict(data)  # shallow copy
    if "train" in out and out["train"] is not None:
        out["train"] = out["train"][:, idx]
    if "mu_ann_full" in out and out["mu_ann_full"] is not None:
        out["mu_ann_full"] = out["mu_ann_full"][idx]
    if "Sigma_ann_full" in out and out["Sigma_ann_full"] is not None:
        out["Sigma_ann_full"] = out["Sigma_ann_full"][np.ix_(idx, idx)]
    return out

def _expand_weights(w_opt, n_total, idx):
    """Map optimized weights on the filtered set back to the full active set."""
    w = np.zeros(n_total, dtype=float)
    w[idx] = w_opt
    return w

def _rebalance_marks_from_index(idx: pd.DatetimeIndex, k_days: int) -> list[int]:
    """Trading-day grid: 0 = first day; then every k_days; include T at end."""
    T = len(idx)
    if (k_days is None) or (k_days <= 0) or (T == 0):
        return [0, T]
    marks = list(range(0, T, int(k_days)))
    if marks[-1] != T:
        marks.append(T)
    return marks

def _window_start(t_end_exclusive: int, min_lb: int, max_lb: int) -> int:
    te = int(t_end_exclusive)
    a = max(0, te - int(max_lb))
    # ensure at least min_lb if history allows
    if te - a < int(min_lb):
        a = max(0, te - int(min_lb))
    return a

def _moments_from_window(R_win: pd.DataFrame, ann: int = 252, shrink_lambda: float = 0.0):
    """
    Returns (mu_ann[N], Sigma_ann[N,N]) using log-return mean & covariance,
    annualized, with shrinkage toward scaled identity.
    """
    import numpy as _np
    X = R_win.to_numpy(_np.float64, copy=False)
    T, N = X.shape
    if T < 2:
        mu_ann = xp.zeros(N)
        Sig_ann = xp.zeros((N, N))
        return mu_ann, Sig_ann

    L = xp.log1p(xp.asarray(X))
    mu_d  = L.mean(axis=0)
    Sig_d = xp.cov(L.T, ddof=1)

    mu_ann  = xp.expm1(mu_d * ann)
    Sig_ann = Sig_d * ann

    lam = float(max(0.0, min(1.0, shrink_lambda)))
    if lam > 0.0:
        s2_bar = float(xp.trace(Sig_ann) / max(N, 1))
        Sig_ann = (1.0 - lam) * Sig_ann + lam * s2_bar * xp.eye(N)

    return mu_ann, Sig_ann

def _make_solver_cfg_from_CONFIG(CONFIG):
    P = CONFIG["PORTFOLIO"]
    return {
        "risk_budget":    float(P["risk_budget"]),
        "risk_free_rate": float(P["risk_free_rate"]),
        "epsilon_sigma":  float(P["epsilon_sigma"]),
        "long_only":      bool(P.get("long_only", False)),
        "no_leverage":    bool(P.get("no_leverage", False)),
    }
    
# -------------------------
# Pipeline
# -------------------------

def dro_pipeline(securities, CONFIG, verbose=True):

    def _resolve_rSLDS_outputs(CONFIG):
        res_csv  = CONFIG["results_csv"]
        seg_parq = CONFIG["segments_parquet"]
        return res_csv, seg_parq

    G = _make_solver_cfg_from_CONFIG(CONFIG)

    # --- artifacts first: derive/validate `securities` from gridsearch results ---
    res_csv, seg_parq = _resolve_rSLDS_outputs(CONFIG)
    if not os.path.exists(res_csv):
        raise FileNotFoundError(f"Results CSV not found: {res_csv}")
    if not os.path.exists(seg_parq):
        raise FileNotFoundError(f"Segments Parquet not found: {seg_parq}")

    # read minimal columns from results
    df_res = pd.read_csv(res_csv, usecols=range(10), engine="python")
    if "security" not in df_res.columns:
        raise ValueError("results_csv missing 'security' column")
    df_res["security"] = df_res["security"].astype(str).str.strip()

    # build or check securities
    if securities is None:
        securities = sorted(df_res["security"].unique())
    else:
        req  = set(map(str, securities))
        have = set(df_res["security"].unique())
        missing = sorted(req - have)
        assert not missing, (
            f"[gridsearch check] Missing {len(missing)} in results CSV: " + ", ".join(missing))

    # also ensure presence in segments parquet
    df_seg_header = pd.read_parquet(seg_parq, columns=["security"])  # light read
    if "security" not in df_seg_header.columns:
        raise ValueError("segments_parquet missing 'security' column")
    have_seg = set(df_seg_header["security"].astype(str).str.strip().unique())
    missing_seg = sorted(set(securities) - have_seg)
    assert not missing_seg, (
        f"[segments check] Missing {len(missing_seg)} in segments parquet: " + ", ".join(missing_seg))

    # --- Panels (now that `securities` is known/validated) ---
    px_all, eps_all, pe_all, ser_vix = import_data(CONFIG["data_excel"])

    # ensure every ticker exists in px_all; warn+drop otherwise
    px_names = set(map(str, px_all.columns))
    keep_px = [t for t in securities if t in px_names]
    dropped_px = [t for t in securities if t not in px_names]
    if dropped_px:
        print("[WARN] Dropping securities not found in PX panel:", ", ".join(sorted(dropped_px)))
    securities = keep_px
    if not securities:
        raise RuntimeError("No securities left after intersecting with PX panel.")

    # RETURNS calendar (UNION; PX-only; honours start/end)
    from pandas.tseries.offsets import BDay
    def _norm_dt(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return pd.to_datetime(x)

    start_req = _norm_dt(CONFIG["DATA"].get("start_dt", None))
    end_req   = _norm_dt(CONFIG["DATA"].get("end_dt",   None))

    # --- Optimisation window (feed the solver extra history) ---
    max_lb = int(CONFIG["REBAL"]["max_lookback_days"])
    if start_req is not None:
        # start earlier by max_lookback_days business days
        start_opt = start_req - BDay(max_lb)
    else:
        start_opt = None  # use earliest available
    end_opt = end_req  # we don't need to extend the tail for optimisation

    # Build UNION calendar over the *optimisation* window
    GIDX = build_calendar_union_px(securities, px_all, start_opt, end_opt)

    # Returns panel over the *optimisation* window (UNION alignment)
    R_cols = []
    for t in securities:
        px = _num_series(px_all[t].loc[start_opt:end_opt]).dropna()
        r  = px.pct_change()
        R_cols.append(r.reindex(GIDX))  # keep NaN where absent (union calendar)
    R_use    = pd.concat(R_cols, axis=1)
    R_use.columns = list(securities)
    R_df_all = R_use.copy()

    # Part A: MVO/DRO on the INTERSECTION calendar; optionally rebalance
    print()
    print(72*"=")
    print("DRO")
    print(72*"=")
    print()
    R_use_clean = R_use.dropna(how="any")
    if len(R_use_clean) < 2:
        print("[INFO] PartA: intersection < 2 rows; skipping PartA.")
        fitA = {"type":"piecewise","w_list":[xp.zeros(R_df_all.shape[1])],"segs":[0,len(R_use_clean)],"kappa": xp.nan,"delta_list":[]}
        summA = {"mu_ann": xp.nan, "sigma_ann": xp.nan, "sharpe_ann": xp.nan,
                 "vol_breach": xp.nan, "max_drawdown": xp.nan, "gross_exp": xp.nan,
                 "kappa": xp.nan, "gap": xp.nan, "delta": xp.nan, "avg_holding_per": xp.nan}
        partA_daily = pd.Series(dtype=float, name="PartA_daily")
    else:
        dataA = make_data_from_returns_panel(R_use_clean)
        AF = int(dataA["ann_factor"])
    
        k_days = int(CONFIG["REBAL"]["rebalance_period_days"])
        if k_days <= 0:
            raise ValueError("rebalance_period_days must be > 0 (non-rebalancing path removed).")
        marks  = _rebalance_marks_from_index(dataA["index"], k_days)
    
        lam_shr = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
        min_lb  = int(CONFIG["REBAL"]["min_lookback_days"])
        max_lb  = int(CONFIG["REBAL"]["max_lookback_days"])
    
        # DRO (Part A) — rebalanced only
        paramsA = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])
        fitA = fit_dro_rebalanced(R_use_clean, paramsA, G, AF, marks, min_lb=min_lb, max_lb=max_lb, 
                                  lam_shr=lam_shr, verbose=bool(verbose))
        summA = evaluate_portfolio(
            fitA,
            {"train": dataA["train"], "test": dataA["test"], "n_days": dataA["n_days"], "ann_factor": AF}, G)

        # daily series from piecewise weights
        T = dataA["n_days"]
        daily = xp.zeros(T, dtype=float)
        for (a,b), wk in zip(zip(fitA["segs"][:-1], fitA["segs"][1:]), fitA["w_list"]):
            daily[a:b] = dataA["train"][a:b] @ xp.asarray(wk).reshape(-1)
        partA_daily = pd.Series(xp.asarray(daily), index=dataA["index"], name="PartA_daily")
    
    # rSLDS labels (CSV affects model selection; Parquet supplies segments)
    Z_labels = {}
    
    # Reuse artifacts: CSV (results) + Parquet (segments)
    res_csv, seg_parq = _resolve_rSLDS_outputs(CONFIG)

    if not os.path.exists(res_csv):
        raise FileNotFoundError(f"Results CSV not found: {res_csv}")
    if not os.path.exists(seg_parq):
        raise FileNotFoundError(f"Segments Parquet not found: {seg_parq}")

    # read the first 10 columns, then select required cols
    df_res = pd.read_csv(res_csv, usecols=range(10), engine="python")  # tolerant parser
    cols   = ["security", "config", "rank", "score"]
    df_res = df_res[[c for c in cols if c in df_res.columns]].copy()

    if securities is not None:
        req  = set(map(str, securities))
        have = set(df_res["security"].astype(str).str.strip().unique())
        missing = sorted(req - have)
        assert not missing, (
            f"[gridsearch check] Missing {len(missing)} in results CSV: " + ", ".join(missing))
    
    df_seg = pd.read_parquet(seg_parq)

    # Optional: normalize dtypes (helps sorting/selection later)
    if "security" in df_res: df_res["security"] = df_res["security"].astype(str).str.strip()
    if "config"   in df_res: df_res["config"]   = df_res["config"].astype(str).str.strip()
    if "rank"     in df_res: df_res["rank"]     = pd.to_numeric(df_res["rank"], errors="coerce")
    if "score"    in df_res: df_res["score"]    = pd.to_numeric(df_res["score"], errors="coerce")

    # schema + date type
    required_cols = {"security", "config", "date", "z"}
    missing_cols = required_cols - set(df_seg.columns)
    if missing_cols:
        raise ValueError(f"Segments parquet missing required columns: {missing_cols}")
    if df_seg["date"].dtype != "datetime64[ns]":
        df_seg["date"] = pd.to_datetime(df_seg["date"], errors="coerce")

    # Non-strict: keep only securities present in BOTH artifacts; warn for the rest
    res_names = set(df_res["security"].astype(str).unique())
    seg_names = set(df_seg["security"].astype(str).unique())
    keep = [s for s in securities if (s in res_names and s in seg_names)]
    dropped = [s for s in securities if s not in keep]
    if dropped:
        print("[WARN] Dropping securities not present in both artifacts:", ", ".join(sorted(dropped)))
    securities = keep
    if not securities:
        raise RuntimeError("No overlapping securities between requested list and artifacts.")

    # Build Z_labels (use the correct helper)
    for sec in securities:
        cfg_best = _select_best_config(df_res, sec, CONFIG.get("prefer_configs"))
        if cfg_best is None:
            print(f"[WARN] No winning config in results for {sec}; skipping.")
            continue
        z_ser = _labels_from_segments_df(df_seg, sec, cfg_best)  # <-- correct helper name
        if z_ser is None:
            print(f"[WARN] No segments for {sec} under config={cfg_best}; skipping.")
            continue
        Z_labels[sec] = map_labels_to_calendar(z_ser, R_df_all.index)

    missing = [sec for sec in securities if sec not in Z_labels]
    if missing:
        print(f"[WARN] Missing regimes for: {missing}. Dropped from pooled moments.")
    avail = [sec for sec in securities if sec in Z_labels]
    if not avail:
        raise RuntimeError("No assets produced rSLDS labels. Cannot proceed with Part B.")
    
    T = len(R_df_all.index)
    
    for sec in avail:
        z = xp.asarray(Z_labels[sec], float)
        assert z.shape[0] == T, f"[{sec}] label length {len(z)} != T={T}"
        n_nan = int(xp.isnan(z).sum())
        uniq  = sorted(set([int(u) for u in xp.unique(z[xp.isfinite(z)])])) if xp.isfinite(z).any() else []
        
    # Per-asset segments + entry indices
    per_asset_segs_on_cal = {}
    for sec in avail:
        z_arr = xp.asarray(Z_labels[sec], float)  # length T, NaNs allowed
        # changepoints on the union calendar (ignore NaN→state and state→NaN edges)
        finite = xp.isfinite(z_arr)
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

    # Regime-DRO solves (time-varying universe)
    print()
    print(72*"=")
    print("Regime-DRO")
    print(72*"=")
    print()
    
    names_all = list(avail)
    taus = list(global_segs)
    w_list = []
    seg_deltas = [] 

    for k in range(len(taus) - 1):
        t_mid = min(max(taus[k], 0), T - 1)
        # strictly pre-t window for moment estimation
        t_use = max(0, t_mid - 1)
    
        # active assets at t_mid: entered and finite label
        A_k = []
        for n in names_all:
            z = xp.asarray(Z_labels[n], float)
            r = R_df_all[n].to_numpy(dtype=float, copy=False)
            if xp.isfinite(z[t_mid]) and xp.isfinite(r[t_mid]):
                A_k.append(n)
        
        print(f"[t={t_mid} | {R_df_all.index[t_mid].date()}] active: {A_k}")
        min_assets = int(CONFIG["PORTFOLIO"]["min_assets"])
        if len(A_k) < min_assets:
            print(f"[WARN][t={t_mid} | {R_df_all.index[t_mid].date()}] only {len(A_k)} active assets (<{min_assets}).")
        
        if len(A_k) == 0:
            # No active assets this segment: append zero weights AND a placeholder delta
            w_list.append(xp.zeros(len(names_all)))
            seg_deltas.append(xp.nan)    # keep alignment with segments
            continue

        # --- Regime-specific, windowed moments (with min_obs filtering) ---
        min_obs    = int(CONFIG["REBAL"]["min_lookback_days"])
        lookback   = int(CONFIG["REBAL"]["max_lookback_days"])
        min_assets = int(CONFIG["PORTFOLIO"]["min_assets"])
        lam_shr    = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])

        R_df_k = R_df_all[A_k]  # keep NaNs; handled in helper

        # 1) Compute regime-conditioned μ and counts per asset over the window
        mu_all, _, _, counts_mu = pooled_moments_by_regime(
            R_df_k,
            {n: Z_labels[n] for n in A_k},
            A_k,
            t_idx=t_use,
            ann=252,
            min_obs=1,              # get counts; we'll enforce min_obs ourselves
            lookback=lookback,
            mode="pairwise",
        )

        # 2) Filter assets by in-regime min_obs for μ
        idx_keep = [i for i, n in enumerate(A_k) if counts_mu.get(n, 0) >= min_obs]
        if verbose and len(idx_keep) < len(A_k):
            dropped = [A_k[i] for i in range(len(A_k)) if i not in idx_keep]
            print(f"[INFO][t={t_mid}] excluded for μ min_obs<{min_obs}: {dropped}")

        if len(idx_keep) < min_assets:
            print(f"[SKIP][t={t_mid} | {R_df_all.index[t_mid].date()}] too few eligible assets "
                  f"after μ filter ({len(idx_keep)}<{min_assets}); carrying previous weights.")
            if len(w_list) > 0:
                w_full = xp.asarray(w_list[-1]).copy()
                delta_k = float(seg_deltas[-1]) if len(seg_deltas) > 0 else xp.nan
            else:
                w_full = xp.zeros(len(names_all)); delta_k = xp.nan
            w_list.append(w_full); seg_deltas.append(delta_k)
            continue

        # Slice to μ-eligible set
        A_keep   = [A_k[i] for i in idx_keep]
        R_df_ke  = R_df_k.iloc[:, idx_keep]
        mu_ke    = xp.asarray(mu_all, float)[idx_keep]

        # 3) Build unconditional Σ on the filtered set and (if needed) tighten again by Σ counts
        Sig_ann, ok_sig, counts_sig = _sigma_unconditional(
            R_df_ke, t_idx=t_use, ann=252,
            min_obs=min_obs, max_lookback=lookback, shrink_lambda=lam_shr,
        )

        if not ok_sig:
            # Second-pass filter using Σ window counts
            idx_keep2 = [i for i, n in enumerate(A_keep) if counts_sig.get(n, 0) >= min_obs]
            if verbose and len(idx_keep2) < len(A_keep):
                dropped2 = [A_keep[i] for i in range(len(A_keep)) if i not in idx_keep2]
                print(f"[INFO][t={t_mid}] excluded for Σ min_obs<{min_obs}: {dropped2}")

            if len(idx_keep2) < min_assets:
                print(f"[SKIP][t={t_mid} | {R_df_all.index[t_mid].date()}] too few eligible assets "
                      f"after Σ filter ({len(idx_keep2)}<{min_assets}); carrying previous weights.")
                if len(w_list) > 0:
                    w_full = xp.asarray(w_list[-1]).copy()
                    delta_k = float(seg_deltas[-1]) if len(seg_deltas) > 0 else xp.nan
                else:
                    w_full = xp.zeros(len(names_all)); delta_k = xp.nan
                w_list.append(w_full); seg_deltas.append(delta_k)
                continue

            # Re-slice and recompute Σ on the twice-filtered set
            A_keep  = [A_keep[i] for i in idx_keep2]
            R_df_ke = R_df_ke.iloc[:, idx_keep2]
            mu_ke   = xp.asarray(mu_ke, float)[idx_keep2]
            Sig_ann, ok_sig, _ = _sigma_unconditional(
                R_df_ke, t_idx=t_use, ann=252,
                min_obs=min_obs, max_lookback=lookback, shrink_lambda=lam_shr,
            )
            if not ok_sig:
                # As a final guard, skip and carry forward (should be rare after filtering).
                print(f"[SKIP][t={t_mid} | {R_df_all.index[t_mid].date()}] Σ still not ok after filtering; carrying previous weights.")
                if len(w_list) > 0:
                    w_full = xp.asarray(w_list[-1]).copy()
                    delta_k = float(seg_deltas[-1]) if len(seg_deltas) > 0 else xp.nan
                else:
                    w_full = xp.zeros(len(names_all)); delta_k = xp.nan
                w_list.append(w_full); seg_deltas.append(delta_k)
                continue

        # 4) Run DRO on the filtered set
        data_k = {
            "train": R_df_ke.fillna(0.0).to_numpy(dtype=float),
            "test":  R_df_ke.fillna(0.0).to_numpy(dtype=float),
            "n_days": R_df_ke.shape[0],
            "ann_factor": 252,
            "mu_ann_full": xp.asarray(mu_ke, float),
            "Sigma_ann_full": xp.asarray(Sig_ann, float),
            "px_cols": list(A_keep),
            "index": R_df_ke.index,
        }
        paramsR = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])
        paramsR["use_moments_override"] = True

        fit_k  = fit_dro(data_k, paramsR, G)
        w_sub  = xp.asarray(fit_k["w"]).reshape(-1)
        delta_k = float(fit_k.get("delta", xp.nan))
        if verbose:
            tstamp = R_df_all.index[t_mid].date()
            # also report which names were used
            print(f"[RegDRO] k={k+1} t={t_mid} {tstamp}  delta_k = {delta_k:.6g}  | eligible={A_keep}")
            _print_mu_by_name(A_keep, mu_ke, prefix="   ")

        # expand to full vector  (use the filtered set A_keep, which matches w_sub)
        w_full = xp.zeros(len(names_all))
        pos = {n: i for i, n in enumerate(names_all)}
        for j, n in enumerate(A_keep):
            w_full[pos[n]] = w_sub[j]
        w_list.append(w_full)
        seg_deltas.append(delta_k)

    fitB = {
        "type": "piecewise",
        "w_list": [xp.asarray(w, float) for w in w_list],
        "segs": xp.asarray(taus, dtype=xp.int64),
        "names": names_all,
        "delta_list": seg_deltas,
    }
    if verbose:
        print("[RegDRO] all deltas:", ", ".join(
            ("nan" if not xp.isfinite(x) else f"{float(x):.6g}") for x in seg_deltas))

    # --- print only if zero weights (per segment / all segments) ---
    _zero_segs = [k for k, wk in enumerate(fitB["w_list"]) if _all_zero_weights(wk)]
    if _zero_segs:
        if len(_zero_segs) == len(fitB["w_list"]):
            print("[WARN] Regime-DRO: all segments have zero weights.")
        else:
            # 1-based indices for readability
            _zlist = ", ".join(str(k+1) for k in _zero_segs)
            print(f"[WARN] Regime-DRO: zero weights in segments {{{_zlist}}}.")

    # evaluation uses the union calendar
    data_eval = {
        "train": R_df_all.fillna(0.0).to_numpy(dtype=float),
        "test":  R_df_all.fillna(0.0).to_numpy(dtype=float),
        "n_days": T,
        "ann_factor": 252,
        "mu_ann_full": np.zeros(len(names_all), dtype=float),
        "Sigma_ann_full": np.eye(len(names_all), dtype=float),
        "px_cols": names_all,
        "index": R_df_all.index}


    # extra dataset digest for Part B eval
    X = R_df_all.fillna(0.0).to_numpy(dtype=float)
    summB = evaluate_portfolio(fitB, data_eval, G)

    # Part B daily portfolio returns (on union calendar)
    X_union = R_df_all.fillna(0.0).to_numpy(dtype=float)   # same as evaluation
    T = X_union.shape[0]
    partB = xp.zeros(T, dtype=float)
    for (a, b), w_k in zip(zip(fitB["segs"][:-1], fitB["segs"][1:]), fitB["w_list"]):
        partB[a:b] = X_union[a:b] @ xp.asarray(w_k).reshape(-1)
    
    partB_daily = pd.Series(partB, index=R_df_all.index, name="PartB_daily")

    # MVO baseline (on same intersection calendar as Part A)
    print()
    print(72*"=")
    print("MVO")
    print(72*"=")
    print()
    
    if len(R_use_clean) < 2:
        fit_mvo0  = {"type":"piecewise","w_list":[xp.zeros(R_df_all.shape[1])],"segs":[0,len(R_use_clean)]}
        summ_mvo0 = {"mu_ann": xp.nan, "sigma_ann": xp.nan, "sharpe_ann": xp.nan,
                     "vol_breach": xp.nan, "max_drawdown": xp.nan}
        mvo_daily = pd.Series(dtype=float, name="MVO_daily")
    else:
        AF = int(dataA["ann_factor"])
        k_days = int(CONFIG["REBAL"]["rebalance_period_days"])
        if k_days <= 0:
            raise ValueError("rebalance_period_days must be > 0 (non-rebalancing path removed).")
        marks  = _rebalance_marks_from_index(dataA["index"], k_days)
        lam_shr = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
        min_lb  = int(CONFIG["REBAL"]["min_lookback_days"])
        max_lb  = int(CONFIG["REBAL"]["max_lookback_days"])
    
        # piecewise MVO using rolling window [ws, a) at each rebalance a
        fit_mvo0 = fit_mvo_rebalanced(
            R_use_clean, G, AF, marks,
            min_lb=min_lb, max_lb=max_lb, lam_shr=lam_shr,
            verbose=bool(verbose)
        )
        summ_mvo0 = evaluate_portfolio(
            fit_mvo0, {"train": dataA["train"], "test": dataA["test"],
                       "n_days": dataA["n_days"], "ann_factor": AF}, G)
    
        # daily series from piecewise weights on the same intersection panel
        T = dataA["n_days"]
        daily = xp.zeros(T, dtype=float)
        for (a, b), wk in zip(zip(fit_mvo0["segs"][:-1], fit_mvo0["segs"][1:]), fit_mvo0["w_list"]):
            daily[a:b] = dataA["train"][a:b] @ xp.asarray(wk).reshape(-1)
        mvo_daily = pd.Series(xp.asarray(daily), index=dataA["index"], name="MVO_daily")
            
    # Keep only dates where all three exist AND are non-NaN
    idx_mvo  = mvo_daily.dropna().index
    idx_dro  = partA_daily.dropna().index
    idx_reg  = partB_daily.dropna().index
    common_idx = idx_mvo.intersection(idx_dro).intersection(idx_reg)

    if len(common_idx) == 0:
        raise RuntimeError("No common-valid window across MVO, DRO, and RegDRO series.")

    # Apply the exact requested reporting window, if provided.
    # Note: optimisation already used (start_req - max_lookback_days), but here we
    #       return precisely [start_req, end_req] when possible.
    requested_idx = common_idx
    if start_req is not None:
        requested_idx = requested_idx[requested_idx >= start_req]
    if end_req is not None:
        requested_idx = requested_idx[requested_idx <= end_req]

    if len(requested_idx) == 0:
        # Nothing to return in the requested range given availability
        raise RuntimeError("Requested [start_dt, end_dt] produced an empty common window.")

    # Overwrite the three series with the aligned, *requested* versions
    mvo_daily   = mvo_daily.reindex(requested_idx).astype(float)
    partA_daily = partA_daily.reindex(requested_idx).astype(float)
    partB_daily = partB_daily.reindex(requested_idx).astype(float)
    
    AF = 252
    n_aligned = len(requested_idx)

    def _summ_from_series(series, G, AF, n_days):
        x = series.to_numpy(dtype=float)
        mu, sig, sh = stats_from_series(x, {
            "n_days": n_days,
            "risk_free_rate": G["risk_free_rate"],
            "annualization_factor": AF
        })
        return {
            "mu_ann": mu,
            "sigma_ann": sig,
            "sharpe_ann": sh,
            "vol_breach": max(sig - G["risk_budget"], 0.0),
            "max_drawdown": _max_drawdown_from_series(x)
        }

    summ_mvo0 = _summ_from_series(mvo_daily,   G, AF, n_aligned)
    summA     = _summ_from_series(partA_daily, G, AF, n_aligned)
    summB     = _summ_from_series(partB_daily, G, AF, n_aligned)

    # Print results
    
    # 1) Build single-row DataFrames for each model
    df_mvo = pd.DataFrame([summ_mvo0]).copy()
    df_dro = pd.DataFrame([summA]).copy()

    '''
    df_reg = pd.DataFrame([summB]).copy()
    # Attach per-segment deltas for RegDRO as separate columns (delta_k1, delta_k2, …)
    if "delta_list" in fitB and fitB["delta_list"]:
        for j, dj in enumerate(fitB["delta_list"], start=1):
            df_reg[f"delta_k{j}"] = float(dj) if dj is not None else float("nan")
    '''
    df_reg = pd.DataFrame([summB]).copy()
    if "delta_list" in fitB and fitB["delta_list"]:
        _d = pd.Series([float(x) if x is not None else float("nan") for x in fitB["delta_list"]])
        df_reg["delta_mean"] = float(_d.mean(skipna=True))
        df_reg["delta_min"]  = float(_d.min(skipna=True))
        df_reg["delta_max"]  = float(_d.max(skipna=True))
    
    # 2) Print a single, clean table with MVO, DRO, RegDRO columns
    results_dict = {"MVO": df_mvo, "DRO": df_dro, "RegDRO": df_reg}
    print_oos_table(results_dict, model_order=["MVO", "DRO", "RegDRO"])
    
    # 3) Also show the detailed RegDRO block (weights by segment, piece boundaries, per-piece δk)
    _section("RegDRO — detailed piecewise report")
    report_regdro(
        fit=fitB,
        data=data_eval,       # union calendar data you already built above
        G=G,
        taus_true=None,       # or pass data_eval.get("taus_true") if you keep true τ around
        label="RegDRO")

    print()
    print("\n[MVO]    MVO baseline (aligned) summary:\n", pd.Series(summ_mvo0).round(4))
    print("\n[DRO]    Static DRO (aligned) summary:\n", pd.Series(summA).round(4))
    print("\n[RegDRO] Regime-DRO (aligned) summary:\n", pd.Series(summB).round(4))
    
    # Return results
    out = {
        "MVO":   {"fit": fit_mvo0, "summary": summ_mvo0},
        "PartA": {"fit": fitA, "data": dataA, "summary": summA},
        "PartB": {"fit": fitB, "summary": summB,
                  "per_asset_segs": per_asset_segs_on_cal,
                  "global_segs": taus, "Z_labels": Z_labels},
        "returns_union": R_df_all,
        "series": {
            "MVO_daily": mvo_daily,
            "PartA_daily": partA_daily,
            "PartB_daily": partB_daily
        },
        "securities": avail
    }
    # add stable names used elsewhere
    out["DRO"]    = {"fit": fitA, "data": dataA, "summary": summA}
    out["RegDRO"] = {"fit": fitB, "data": {
                        "train": R_df_all.fillna(0.0).to_numpy(float),
                        "test":  R_df_all.fillna(0.0).to_numpy(float),
                        "n_days": len(R_df_all.index),
                        "ann_factor": 252
                    }, "summary": summB}
    

    # Save results (if path configured)
    if "dro_pickle" in CONFIG and CONFIG["dro_pickle"]:
        save_out(out, CONFIG["dro_pickle"])

    return out

