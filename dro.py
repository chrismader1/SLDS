
# ---------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------

# GPU if available
try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False
print(f"GPU={GPU}")

# Others
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import os, re, ast
import io, gzip, pickle
from scipy import stats as sp_stats
from datetime import datetime

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
# IO helpers
# -------------------------

class _NPCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_out(path: str) -> dict:
    """
    Load dict saved by save_out (handles .pkl or .pkl.gz, regardless of extension).
    Also fixes NumPy namespace changes when unpickling older files.
    """
    with open(path, "rb") as raw:
        head = raw.read(2)
        raw.seek(0)

        # If gzipped (magic: 0x1f 0x8b), wrap with GzipFile; else use raw file.
        fp = gzip.GzipFile(fileobj=raw) if head == b"\x1f\x8b" else raw

        try:
            return pickle.load(fp)
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e):
                # Rewind the inner stream and retry with compat unpickler
                try:
                    fp.seek(0)
                except Exception:
                    # If fp is a GzipFile, reopen to reset
                    raw.seek(0)
                    fp = gzip.GzipFile(fileobj=raw) if head == b"\x1f\x8b" else raw
                return _NPCompatUnpickler(fp).load()
            raise

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

# ---------------------------------------------------------------
# Wasserstein helpers
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
        
def _mbb_indices(T: int, m: int, L: int, rng=None) -> np.ndarray:
    """
    Moving-block bootstrap: draw start positions U(0..T-1), take L-length
    circular blocks until we have m indices. Returns shape (m,).
    """
    rng = np.random.default_rng(None) if rng is None else rng
    L = int(max(1, L)); T = int(T); m = int(m)
    idx = np.empty(m, dtype=np.int64)
    filled = 0
    while filled < m:
        s = int(rng.integers(0, T))
        block = (s + np.arange(L)) % T
        k = min(L, m - filled)
        idx[filled:filled+k] = block[:k]
        filled += k
    return idx

def bootstrap_np_block_delta(R, n_proj=128, B=256, block_len=10, alpha=0.05, seed=None):
    """
    10-day moving-block bootstrap of the empirical daily panel.
    Per replicate: draw TWO independent block-resamples of length n,
    compute sliced-W2 between them -> daily distance. Return (1-alpha) quantile.
    """
    R_xp = xp.asarray(R, dtype=xp.float32)
    T = int(R_xp.shape[0])
    n = T  # resample to full length; keeps your current behavior
    rng = np.random.default_rng(seed)
    dists = xp.empty(int(B), dtype=float)
    for b in range(int(B)):
        i1 = _mbb_indices(T, n, block_len, rng=rng)
        i2 = _mbb_indices(T, n, block_len, rng=rng)
        X1 = R_xp[xp.asarray(i1, dtype=xp.int64)]
        X2 = R_xp[xp.asarray(i2, dtype=xp.int64)]
        dists[b] = sliced_w2_empirical(X1, X2, n_proj=n_proj, rng=None)
    return float(xp.quantile(dists, 1.0 - alpha))

def bootstrap_gaussian_block_delta(R, alpha=0.05, B=512, block_len=10, eps=1e-9, seed=None):
    """
    10-day moving-block bootstrap, but distance is Gelbrich W2 between the
    Gaussian fitted to the original sample (mu0,S0) and the Gaussian fitted to
    each block-resampled sample (mu_b,S_b). Returns daily (not annualized) delta.
    """
    X = xp.asarray(R, float)
    n, d = X.shape
    if n < 2:
        return 0.0

    # reference (DAILY) moments
    mu0 = xp.mean(X, axis=0)
    Xc  = X - mu0
    S0  = (Xc.T @ Xc) / (n - 1)

    rng = np.random.default_rng(seed)
    deltas = xp.empty(int(B), dtype=float)
    for b in range(int(B)):
        idx = _mbb_indices(n, n, block_len, rng=rng)
        Xb  = X[xp.asarray(idx, dtype=xp.int64)]
        mub = xp.mean(Xb, axis=0)
        Xbc = Xb - mub
        Sb  = (Xbc.T @ Xbc) / max(n - 1, 1)
        deltas[b] = wasserstein2_gaussian(mu0, S0, mub, Sb, eps)
    return float(xp.quantile(deltas, 1.0 - alpha))

# ---------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------

def to_numpy(a, dtype=float):
    """Return a NumPy array on host, even if `a` is a CuPy array."""
    import numpy as _np
    asnp = getattr(xp, "asnumpy", None)
    if callable(asnp):
        try:
            return asnp(a).astype(dtype, copy=False)
        except Exception:
            pass
    return _np.asarray(a, dtype=dtype)
    
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

    AF = int((params or {}).get("annualization_factor", 252))
    
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
        alpha  = float((params or {}).get("alpha", 0.05))
        B      = int((params or {}).get("B", 256))
        n_proj = int((params or {}).get("n_proj", 128))
        seed   = (params or {}).get("seed", None)
        L      = int((params or {}).get("block_len", 10))  # fixed 10 by default
        delta_daily = bootstrap_np_block_delta(R, n_proj=n_proj, B=B, block_len=L, alpha=alpha, seed=seed)
        return AF * float(delta_daily)
    
    if method == "bootstrap_gaussian":
        assert R is not None, "bootstrap_gaussian needs raw sample matrix R."
        alpha = float((params or {}).get("alpha", 0.05))
        B     = int((params or {}).get("B", 512))
        eps   = float((params or {}).get("epsilon_sigma", 1e-9))
        seed  = (params or {}).get("seed", None)
        L     = int((params or {}).get("block_len", 10))  # fixed 10 by default
        delta_daily = bootstrap_gaussian_block_delta(R, alpha=alpha, B=B, block_len=L, eps=eps, seed=seed)
        return AF * float(delta_daily)
    
    raise ValueError(f"Unknown delta_method='{method}'")

def psd_factor_LtL(Sigma, eps):
    """
    Return L such that Sigma ≈ L.T @ L (so the constraint is ||L w|| ≤ rho).
    We form L by transposing a lower Cholesky.
    """
    import numpy as _np
    Sigma_sym = 0.5 * (Sigma + Sigma.T)
    try:
        # single-shot minimal regularization
        C = xp.linalg.cholesky(Sigma_sym + eps * xp.eye(Sigma_sym.shape[0]))
    except xp.linalg.LinAlgError:
        # project once, no extra +eps*I
        vals, vecs = xp.linalg.eigh(Sigma_sym)
        vals = xp.clip(vals, eps, None)
        Sigma_psd = vecs @ xp.diag(vals) @ vecs.T
        C = xp.linalg.cholesky(Sigma_psd)
    # Explicit device→host copy when using CuPy; no-op for NumPy
    asnumpy = getattr(xp, "asnumpy", None)
    if callable(asnumpy):
        return asnumpy(C.T).astype(_np.float64, copy=False)
    else:
        return _np.asarray(C.T, dtype=_np.float64)

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

    no_shorting = bool(config.get("no_shorting", False))
    no_leverage = bool(config.get("no_leverage", False))

    # Base SOC constraints (outer problem epigraph)
    constr = [
        cp.norm(L @ w, 2) <= rho,  # risk budget
        cp.norm(w, 2)    <= t,     # epigraph of ||w||_2
        t >= 0,
    ]

    # Enforce w >= 0 if requested
    if no_shorting:
        constr += [w >= 0]

    # Enforce sum(w) <= 1 (cash allowed, no leverage) if requested
    if no_leverage:
        constr += [cp.sum(w) <= 1]

    # Max position size
    mpos = config.get("max_pos_size", None)
    if mpos is not None:
        mpos = float(mpos)
        if np.isfinite(mpos) and mpos >= 0.0:
            if no_shorting:
                # 0 ≤ w_i ≤ mpos
                constr += [w <= mpos]
            else:
                # −mpos ≤ w_i ≤ mpos
                constr += [cp.abs(w) <= mpos]
    
    # Cash cap: cash_t = max(0, 1 - sum(w)) ≤ max_cash  ⇔  sum(w) ≥ 1 - max_cash
    mc = config.get("max_cash", None)
    if mc is not None and _np.isfinite(float(mc)):
        mc = float(mc)
        mc = max(0.0, min(1.0, mc))
        constr += [cp.sum(w) >= 1.0 - mc]
        
    prob = cp.Problem(objective, constr)
    try:
        if verbose:
            print(f"[solve_optimizer] delta = {float(delta):.6g}, rho = {rho:.6g}")
        prob.solve(solver=cp.MOSEK, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)

    if (w.value is None) or (prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)):
        raise RuntimeError(f"ECOS/MOSEK failed: status={prob.status}")

    # Always return NumPy on host to avoid CuPy→NumPy implicit conversion errors downstream
    return _np.asarray(w.value, dtype=float).reshape(-1)

# ---------------------------------------------------------------
# Fitting - Actual Data
# ---------------------------------------------------------------

def solve_dro(mu, Sigma, params, G, R=None, *, verbose=None):
    """
    DRO with δ computed from `params`.
    Works for ALL delta_method values. If the method needs samples (bootstrap_*),
    pass R (window matrix of returns); otherwise R is ignored.
    """
    if verbose is None:
        verbose = bool(params.get("verbose", False))
    delta = compute_delta(params.get("kappa", 1.0), mu, Sigma, R=R, params=params)
    w = solve_optimizer(mu, Sigma, delta, config=G, verbose=verbose)
    return w, float(delta)

    
def fit_mvo_rebalanced(R_df: pd.DataFrame, G, ann: int, marks: list[int],
                       min_lb: int, max_lb: int, lam_shr: float, verbose: bool = False):

    """
    Piecewise MVO over rebalance marks using rolling windows with min/max lookback.
    """
    idx = R_df.index
    w_list, segs = [], marks
    N = R_df.shape[1]

    if verbose:
        _section("MVO")
    
    for a, b in zip(marks[:-1], marks[1:]):
        # rebalance at 'a' using window [ws, a)
        if a == 0:
            w_list.append(_feasible_placeholder(N, G))
            continue
        ws = _window_start(a, min_lb, max_lb)
        R_win = R_df.iloc[ws:a].dropna(how="any")
        if len(R_win) < max(2, min_lb):
            w_list.append(w_list[-1] if w_list else _feasible_placeholder(N, G))
            continue
   
        mask_all = np.ones(R_win.shape[0], dtype=bool)
        mu_ann = compute_mean_from_window(R_win, mask_all, min_obs=min_lb, ann=ann)
        Sig_ann = compute_cov_from_window(R_win, ann=ann, shrink_lambda=lam_shr, min_obs=min_lb)
        
        w = solve_optimizer(mu_ann, Sig_ann, delta=0.0, config=G, verbose=bool(verbose))
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

    if verbose:
        _section("DRO") 
    
    for a, b in zip(marks[:-1], marks[1:]):
        if a == 0:
            w_list.append(_feasible_placeholder(N, G)); delta_list.append(xp.nan)
            continue
        ws = _window_start(a, min_lb, max_lb)
        R_win = R_df.iloc[ws:a].dropna(how="any")
        if len(R_win) < max(2, min_lb):
            # carry forward
            w_list.append(w_list[-1] if w_list else _feasible_placeholder(N, G))
            delta_list.append(delta_list[-1] if delta_list else xp.nan)
            continue

        # window moments
        mask_all = np.ones(R_win.shape[0], dtype=bool)
        mu_ann = compute_mean_from_window(R_win, mask_all, min_obs=min_lb, ann=ann)
        Sig_ann = compute_cov_from_window(R_win, ann=ann, shrink_lambda=lam_shr, min_obs=min_lb)

        # delta from window (pass raw sample as numpy array)
        delta = compute_delta(params.get("kappa", 1.0),
                              mu_ann, Sig_ann,
                              R=R_win.to_numpy(dtype=float),
                              params=params)
        w = solve_optimizer(mu_ann, Sig_ann, delta, config=G, verbose=bool(verbose))
        if verbose:
            dt = idx[a]
            print(f"[DRO] t={a} {getattr(dt, 'date', lambda: dt)()}  delta={float(delta):.4f}")
            _print_mu_by_name(R_win.columns.tolist(), mu_ann)

        w_list.append(w); delta_list.append(float(delta))

    return {"type": "piecewise", "w_list": w_list, "segs": segs,
            "kappa": params.get("kappa", xp.nan), "delta_list": delta_list}

# ---------------------------------------------------------------
# Fitting - Synthetic Data
# ---------------------------------------------------------------

def print_single_portfolio_block(label, w, returns_train, returns_eval, rho, Sigma_ann, config, rtol=1e-6, atol=1e-9):
    n_days, n_assets = returns_train.shape
    # AF = int(config.get("annualization_factor", config["n_days"]))
    AF = int(config.get("annualization_factor", 252))
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

def print_regime_block(label, returns_train, returns_eval, w_list, segs, rho,
                       taus_display, seg_deltas, config=None):
    """
    Pretty-printer for piecewise portfolios.
    Uses 'annualization_factor' (AF) if provided in config, else falls back to n_days.
    """
    n_days, n_assets = returns_train.shape
    # AF = int((config or {}).get("annualization_factor", (config or {}).get("n_days", n_days)))
    AF = int((config or {}).get("annualization_factor", 252))

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

def fit_mvo(data, params, G):
    """
    Mean–variance optimizer: same as fit_dro with fixed δ = 0.
    """
    delta = 0.0
    if bool(params.get("verbose", False)):
        print(f"[MVO] delta = {delta:.6g}")
        _print_mu_by_name(list(data.get("px_cols", range(len(data["mu_ann_full"])))), data["mu_ann_full"])
    w = solve_optimizer(
        data["mu_ann_full"], data["Sigma_ann_full"],
        delta, G, verbose=bool(params.get("verbose", False)),)
    return {"type": "static", "w": w, "kappa": xp.nan, "delta": float(delta)}

def fit_dro(data, params, G):
    delta = compute_delta(params.get("kappa", 1.0),
                          data["mu_ann_full"], data["Sigma_ann_full"], data["train"], params)
    if bool(params.get("verbose", False)):
        print(f"[DRO] delta = {float(delta):.6g}")
        _print_mu_by_name(list(data.get("px_cols", range(len(data["mu_ann_full"])))), data["mu_ann_full"])
    w = solve_optimizer(data["mu_ann_full"], data["Sigma_ann_full"], delta,
                        G, verbose=bool(params.get("verbose", False)))
    return {"type": "static", "w": w, "kappa": params.get("kappa", xp.nan), "delta": float(delta)}

def fit_regime_dro(data, params, G):
    n_days = data["n_days"]
    AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))

    # Report segmentation before optimizing
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

    # Start solving
    
    if bool(params.get("verbose", False)):
        _section("RegDRO")
    
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
            t_fit = max(0, min(int(b) - 1, int(n_days) - 1))
            D_pos = t_fit
            dt = data.get("index", None)
            dt_str = ""
            if dt is not None and 0 <= t_fit < len(dt):
                d = dt[t_fit]
                dt_str = f"{getattr(d, 'date', lambda: d)()}"
            print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta={float(delta_k):.4f}")
            _print_mu_by_name(keep if 'keep' in locals() else list(data.get('px_cols', [])), mu_est)
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
    w = solve_optimizer(
        data["mu_ann_full"], data["Sigma_ann_full"], delta,
        G, verbose=bool(params.get("verbose", False)))
    return {"type": "static", "w": w, "delta": delta, "kappa": xp.nan}

def fit_regime_dro_rev_constSigma(data, params, G):
    segs = params["segs"]
    Sigma_fix = data["Sigma_ann_full"]          # constant across segments
    w_list = []
    for j, (a, b) in enumerate(zip(segs[:-1], segs[1:])):
        R_seg = data["train"][a:b]
        log_seg = xp.log1p(R_seg)
        AF = int(params.get("annualization_factor", data.get("ann_factor", 252)))
        mu_est = xp.expm1(log_seg.mean(axis=0) * AF)
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
        stats_oos = portfolio_stats(
            fit["w"], test, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"],
                             "risk_budget": G["risk_budget"], "annualization_factor": AF})
        ge = float(xp.sum(xp.abs(fit["w"])))
        port_tr = train @ fit["w"]
        _, sigma_train_ann, _ = stats_from_series(
            port_tr,
            {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})
        # SOC (training) risk: ||L w||_2 where L^T L ≈ Σ_ann (if available)
        train_soc = float("nan")
        if isinstance(data, dict) and ("Sigma_ann_full" in data) and (data["Sigma_ann_full"] is not None):
            L = psd_factor_LtL(data["Sigma_ann_full"], G["epsilon_sigma"])
            train_soc = float(xp.linalg.norm(L @ fit["w"]))
        
        # enrich & rename “gap”
        stats_oos["gross_exp"] = ge
        stats_oos["sigma_train_ann"] = float(sigma_train_ann)
        stats_oos["sigma_oos_ann"] = float(stats_oos["sigma_ann"])
        stats_oos["train_soc_risk"] = train_soc
        stats_oos["train_constraint_slack"] = float(G["risk_budget"] - train_soc) if xp.isfinite(train_soc) else xp.nan
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
        for (a, b), wk in zip(zip(fit["segs"][:-1], fit["segs"][1:]), fit["w_list"]):
            port_tr[a:b] = train[a:b] @ wk

        _, sigma_train_ann, _ = stats_from_series(
            port_tr, {"n_days": n_days, "risk_free_rate": G["risk_free_rate"], "annualization_factor": AF})
        stats_oos["gross_exp"] = ge
        stats_oos["sigma_train_ann"] = float(sigma_train_ann)
        stats_oos["sigma_oos_ann"]  = float(stats_oos["sigma_ann"])
        # SOC per-piece not available here → leave NaN placeholders
        stats_oos["train_soc_risk"] = xp.nan
        stats_oos["train_constraint_slack"] = xp.nan
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
    AF = int(config.get("annualization_factor", 252))
    rf_daily = (1 + rf_annual) ** (1 / AF) - 1
    x = xp.asarray(port_daily, dtype=float).reshape(-1)
    mask = xp.isfinite(x)
    x = x[mask]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    sigma_daily = xp.std(x, ddof=1)
    sigma_annual = sigma_daily * xp.sqrt(AF)
    mu_annual_geom = xp.exp(AF * xp.mean(xp.log1p(x))) - 1
    sharpe_annual = (xp.mean(x) - rf_daily) / sigma_daily * xp.sqrt(AF) if sigma_daily > 0 else xp.nan
    return float(mu_annual_geom), float(sigma_annual), float(sharpe_annual)

def _max_drawdown_from_series(port_daily):
    """
    Max drawdown (most negative) from daily simple returns.
    Uses device ops for prod; does cummax on host when GPU=True (CuPy lacks accumulate).
    """
    import numpy as _np
    x = xp.asarray(port_daily, float).reshape(-1)
    if x.size == 0:
        return float("nan")

    # equity curve on current backend
    equity = xp.cumprod(1.0 + x)

    if GPU and hasattr(xp, "asnumpy"):  # CuPy path: cummax on host
        eq_h   = xp.asnumpy(equity)
        peak_h = _np.maximum.accumulate(eq_h)
        dd_h   = eq_h / peak_h - 1.0
        return float(_np.min(dd_h))
    else:                               # NumPy path: pure xp (xp==np)
        peak = xp.maximum.accumulate(equity)
        dd   = equity / peak - 1.0
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
        "max_dd": max_dd,}

def portfolio_stats_multipiece(w_list, taus, returns, config):
    """
    w_list: list of weights per piece, length = len(taus)-1
    taus:   [0=τ0, τ1, ..., τK=n_days]
    """

    import numpy as _np
    taus = [int(x) for x in list(taus)]
    R = _np.asarray(returns, dtype=float)
    assert taus[0] == 0 and taus[-1] == int(config["n_days"]) and len(w_list) == len(taus) - 1, \
        "segments/weights mismatch or calendar length mismatch"
    
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
        "max_dd": max_dd,}

# ---------------------------------------------------------------
# Hypothesis Testing
# ---------------------------------------------------------------
    
def _paired_diff(x, y):
    """Return finite paired differences d = x - y and effective n."""
    d = xp.asarray(x, float) - xp.asarray(y, float)
    mask = xp.isfinite(d)
    d = d[mask]
    return d, int(d.size)

def paired_onesided_less(x, y):
    # H0: mean(x - y) >= 0  vs  H1: mean(x - y) < 0
    d, n = _paired_diff(x, y)
    if n < 2:
        return float("nan"), float("nan")
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="less")
    return t, p
    
def superiority_paired(x, y):
    # H0: mean(x - y) <= 0  vs  H1: > 0
    d, n = _paired_diff(x, y)
    if n < 2:
        return float("nan"), float("nan")
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="greater")
    return t, p

def paired_two_sided_test_with_ci(x, y, alpha=0.05):
    """
    Paired two-sided t-test for mean(x - y) == 0 with (1-α) CI.
    Returns dict(mean_diff, t, p, ci_low, ci_high, n).
    """
    d, n = _paired_diff(x, y)
    mean_diff = float(xp.mean(d)) if n else float("nan")
    if n < 2:
        return dict(mean_diff=mean_diff, t=float("nan"), p=float("nan"),
                    ci_low=float("nan"), ci_high=float("nan"), n=n)
    sd = float(xp.std(d, ddof=1))
    se = sd / float(xp.sqrt(n))
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    tcrit = float(sp_stats.t.ppf(1 - alpha / 2, df=n - 1))
    ci_low  = mean_diff - tcrit * se
    ci_high = mean_diff + tcrit * se
    return dict(mean_diff=mean_diff, t=t, p=p, ci_low=ci_low, ci_high=ci_high, n=n)

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
# Block bootstrap for OOS CIs
# ---------------------------------------------------------------

def _stationary_bootstrap_indices(T: int, B: int, avg_block: int, rng=None):
    """Build all bootstrap index vectors ONCE (CPU)."""
    rng = np.random.default_rng(None) if rng is None else rng
    p = 1.0 / max(int(avg_block), 1)
    idx_mat = np.empty((int(B), T), dtype=np.int64)
    for b in range(int(B)):
        idx = idx_mat[b]
        idx[0] = rng.integers(0, T)
        for t in range(1, T):
            if rng.random() < p:
                idx[t] = rng.integers(0, T)
            else:
                idx[t] = (idx[t-1] + 1) % T
    return idx_mat  # shape (B, T) on CPU

def _metrics_from_indices_xp(x_dev, b_dev_or_none, G: dict, AF: int, idx_dev):
    """Compute all metrics on DEVICE using `xp` (no pandas)."""
    x = xp.take(x_dev, idx_dev)
    has_bench = (b_dev_or_none is not None)
    if has_bench:
        b = xp.take(b_dev_or_none, idx_dev)
        mask = xp.isfinite(x) & xp.isfinite(b)
    else:
        b = None
        mask = xp.isfinite(x)

    # stats_from_series and _max_drawdown_from_series already use xp
    mu_ann, sigma_ann, sharpe_ann = stats_from_series(x, {
        "n_days": int(x.size),
        "risk_free_rate": G["risk_free_rate"],
        "annualization_factor": AF
    })
    vol_breach = max(sigma_ann - G["risk_budget"], 0.0)
    max_dd = _max_drawdown_from_series(x)

    alpha = te = ir = float("nan")
    hr = float("nan")
    if has_bench:
        if int(mask.sum()) > 0:
            ex = x[mask] - b[mask]
            alpha = AF * float(ex.mean())
            te    = (AF ** 0.5) * float(ex.std(ddof=1))
            ir    = (alpha / te) if np.isfinite(te) and te != 0 else float("nan")
            hr    = float(xp.mean((x[mask] - b[mask]) >= 0.0))

    return {
        "mu_ann": mu_ann,
        "sigma_ann": sigma_ann,
        "sharpe_ann": sharpe_ann,
        "vol_breach": vol_breach,
        "max_dd": max_dd,
        "alpha_ann": alpha,
        "te_ann": te,
        "ir_ann": ir,
        "hit_rate": hr,
    }

def block_bootstrap_oos(port_daily, bench_daily_or_none, G, AF,
                        B=1000, avg_block=10, alpha=0.05, seed=None):
    """
    GPU-aware block bootstrap:
      • uses `xp` everywhere;
      • keeps data on device;
      • builds all indices once on CPU, then copies per replicate to device and xp.take’s.
    """
    # host arrays (minimal coercion; no pandas in the loop)
    x_h = np.asarray(port_daily, dtype=float).reshape(-1)
    b_h = None if bench_daily_or_none is None else np.asarray(bench_daily_or_none, dtype=float).reshape(-1)
    T = int(x_h.size)

    # move to device
    x_d = xp.asarray(x_h, dtype=float)
    b_d = None if b_h is None else xp.asarray(b_h, dtype=float)

    # indices once on CPU
    idx_mat_h = _stationary_bootstrap_indices(T, B, avg_block, rng=np.random.default_rng(seed))

    keys = ["mu_ann","sigma_ann","sharpe_ann","vol_breach","max_dd",
            "alpha_ann","te_ann","ir_ann","hit_rate"]
    coll = {k: [] for k in keys}

    # loop: copy one index vector to device, compute metrics entirely on device
    for b in range(int(B)):
        idx_d = xp.asarray(idx_mat_h[b], dtype=xp.int64)
        m = _metrics_from_indices_xp(x_d, b_d, G, AF, idx_d)
        for k in keys:
            coll[k].append(m[k])

    # summarize (host)
    lo_q, hi_q = alpha/2.0, 1.0 - alpha/2.0
    out = {}
    for k, vals in coll.items():
        arr = np.asarray(vals, dtype=float)
        if not np.isfinite(arr).any():
            out[k] = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        else:
            out[k] = {
                "mean": float(np.nanmean(arr)),
                "ci_low": float(np.nanquantile(arr, lo_q)),
                "ci_high": float(np.nanquantile(arr, hi_q)),
            }
    return out

# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------

def oos_summary(results: dict, model_order=None) -> pd.DataFrame:
    base_rows = [
        "mu_ann",
        "sigma_ann",
        "sharpe_ann",
        "vol_breach",
        "max_dd",
        "gross_exp",
        "delta_mean","delta_min","delta_max",  
        # "alpha_ann","te_ann","ir_ann","hit_rate",
    ]

    ALLOW_CI = {"mu_ann","sigma_ann","sharpe_ann","vol_breach"}
    NO_CI_MODELS = {"SPX"}

    if model_order is None:
        model_order = list(results.keys())

    out = pd.DataFrame(index=base_rows, columns=model_order, dtype=object)

    def _fmt_value_only(v):
        try:
            vf = float(v)
            return "" if not np.isfinite(vf) else f"{vf:.3f}"
        except Exception:
            return ""

    for m in model_order:
        if m not in results or len(results[m]) == 0:
            continue
        row0 = results[m].iloc[0]
        for col in base_rows:
            if col not in results[m].columns:
                continue
            v  = row0.get(col, np.nan)
            lo = row0.get(f"{col}_ci_low",  np.nan)
            hi = row0.get(f"{col}_ci_high", np.nan)

            use_ci = (
                (m not in NO_CI_MODELS) and
                (col in ALLOW_CI) and
                pd.notna(lo) and pd.notna(hi) and
                np.isfinite(float(lo)) and np.isfinite(float(hi))
            )

            out.at[col, m] = (
                f"{float(v):.3f} ({float(lo):.3f}, {float(hi):.3f})"
                if use_ci else _fmt_value_only(v)
            )

    def _all_blank(series):
        return all((isinstance(x, str) and x == "") or (x is None) for x in series.values)
    out = out.loc[~out.apply(_all_blank, axis=1)]
    return out

def print_oos_table(results_dict, model_order):
    model_order = [m for m in model_order if m in results_dict and len(results_dict[m]) > 0]
    if not model_order:
        print("\nNo models to display."); return
    print("\n" + "=" * 108)
    print("OOS Portfolio Performance")
    print("=" * 108)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(oos_summary(results_dict, model_order=model_order))

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

def _print_mu_by_name(names, mu_vec, prefix="   "):
    names = list(names)
    mu_vec = xp.asarray(mu_vec, float).ravel()
    s = ", ".join(f"{names[i]}:{float(mu_vec[i]):+.4f}" for i in range(len(names)))
    print(prefix + "mu_ann: [" + s + "]")

def _section(title: str):
    print("\n" + "="*72)
    print(str(title))
    print("="*72 + "\n")

# -------------------------
# Pipeline helpers
# -------------------------

def import_data(filename):

    def _sheet(sheet, skiprows=4, drop_head_rows=3):
        df = pd.read_excel(filename, sheet_name=sheet, skiprows=skiprows, index_col=0)
        if drop_head_rows:
            df = df.iloc[drop_head_rows:, :]
        # sanity: index must be datetime-like
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
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
    Returns a NumPy float64 array (never CuPy).
    """
    z = pd.Series(z_ser).sort_index()
    z.index = pd.to_datetime(z.index)
    cal = pd.DatetimeIndex(cal)
    out = pd.Series(np.nan, index=cal)
    inter = cal.intersection(z.index)
    if len(inter):
        out.loc[inter] = z.reindex(inter).to_numpy()
    return out.to_numpy(dtype="float64")

def snap_start_prev(cal: pd.DatetimeIndex, start_dt):
    """
    If start_dt is not on the union calendar, return the closest date
    in `cal` that is <= start_dt. If start_dt is None, return cal[0].
    If start_dt is before cal[0], return cal[0].
    """
    if start_dt is None:
        return cal[0]
    s = pd.to_datetime(start_dt)
    i = cal.searchsorted(s, side="right") - 1
    return cal[0] if i < 0 else cal[i]

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

def _window_start(t_end_exclusive: int, min_lb: int, max_lb: int) -> int:
    te = int(t_end_exclusive)
    a = max(0, te - int(max_lb))
    # ensure at least min_lb if history allows
    if te - a < int(min_lb):
        a = max(0, te - int(min_lb))
    return a

def compute_mean_from_window(
    R_win: np.ndarray | pd.DataFrame,
    mask: np.ndarray,
    *,
    min_obs: int = 252,
    ann: int = 252,
) -> np.ndarray:
    """
    Mean-only estimator for SIMPLE returns (from pct_change()).
    For MVO/DRO pass an all-True mask; for RegDRO pass the in-regime mask.

    Parameters
    ----------
    R_win : (T,d) array-like
        Simple returns in the lookback window.
    mask : (T,) bool
        True where the observation is included (regime filter).
    min_obs : int
        Minimum required usable observations per asset inside the mask.
    ann : int
        Annualization factor (e.g., 252).

    Returns
    -------
    mu_ann : (d,) float64
        Annualized arithmetic mean of simple returns.
    """
    X = R_win.to_numpy(np.float64, copy=False) if isinstance(R_win, pd.DataFrame) else np.asarray(R_win, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("R_win must be 2D (T,d).")
    m = np.asarray(mask)
    if m.dtype != bool or m.ndim != 1 or m.shape[0] != X.shape[0]:
        raise ValueError(f"mask must be (T,) bool matching R_win.shape[0]; got mask {m.shape}, R {X.shape}")
    if not np.any(m):
        raise ValueError("in-regime mask is empty.")

    Xm = X[m, :]                                   # (N,d)
    finite = np.isfinite(Xm)
    counts = finite.sum(axis=0)
    if np.any(counts < min_obs):
        raise ValueError(f"Insufficient in-regime observations: min {int(counts.min())} < required {min_obs}")

    Xm = np.where(finite, Xm, np.nan)
    mu_periodic = np.nanmean(Xm, axis=0)           # arithmetic mean of simple returns
    mu_ann = mu_periodic * float(ann)              # annualize once

    if not np.all(np.isfinite(mu_ann)):
        raise ValueError("Non-finite annualized mean encountered.")
    return mu_ann

def compute_cov_from_window(
    R_win: np.ndarray | pd.DataFrame,
    *,
    ann: int = 252,
    shrink_lambda: float = 0.0,
    min_obs: int = 2,
) -> np.ndarray:
    """
    Unconditional covariance for SIMPLE returns on the full lookback window.
    Used by MVO/DRO/RegDRO (same Σ for all).

    Shrinkage towards scaled identity: (1-λ)Σ + λ * s2_bar * I.
    """
    X = R_win.to_numpy(np.float64, copy=False) if isinstance(R_win, pd.DataFrame) else np.asarray(R_win, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("R_win must be 2D (T,d).")
    # Keep only rows where all assets are finite to ensure a consistent time index for Σ
    row_ok = np.isfinite(X).all(axis=1)
    Xc = X[row_ok, :]
    if Xc.shape[0] < min_obs:
        raise ValueError(f"Not enough observations for covariance: {Xc.shape[0]} < {min_obs}")

    Sig = np.cov(Xc.T, ddof=1)                     # simple returns cov
    Sig_ann = Sig * float(ann)

    lam = float(np.clip(shrink_lambda, 0.0, 1.0))
    if lam > 0.0:
        N = Sig_ann.shape[0]
        s2_bar = float(np.trace(Sig_ann) / max(N, 1))
        Sig_ann = (1.0 - lam) * Sig_ann + lam * s2_bar * np.eye(N, dtype=np.float64)

    if not np.all(np.isfinite(Sig_ann)):
        raise ValueError("Non-finite covariance encountered.")
    return Sig_ann

def _make_solver_cfg_from_CONFIG(CONFIG):
    P = CONFIG["PORTFOLIO"]

    max_cash = P.get("max_cash", None)
    max_cash = None if max_cash is None else float(max_cash)

    max_pos_size = P.get("max_pos_size", None)
    max_pos_size = None if max_pos_size is None else float(max_pos_size)
       
    return {
        "risk_budget":    float(P["risk_budget"]),
        "risk_free_rate": float(P["risk_free_rate"]),
        "epsilon_sigma":  float(P["epsilon_sigma"]),
        "no_shorting":    bool(P.get("no_shorting", False)),
        "no_leverage":    bool(P.get("no_leverage", False)),
        "max_cash":       max_cash,
        "max_pos_size":   max_pos_size,
    }

def _select_regime_set_S_star_from_train(px_train: pd.Series, zhat_train: np.ndarray, dt: float) -> set:
    """
    Pick S* (subset of regimes) by maximizing in-sample CAGR_rel using the
    0/1 mapping w_k ∈ {0,1}. Returns set of regimes with w_k=1.
    """
    regimes = np.unique(zhat_train)
    # daily returns (fillna 0 so cumprod works cleanly)
    bench_ret = px_train.pct_change().fillna(0.0)
    bench_idx = (1.0 + bench_ret).cumprod()
    n_years = len(bench_ret) * dt
    cagr_bench = bench_idx.iloc[-1] ** (1.0 / max(n_years, 1e-12)) - 1.0

    best_cagr_rel, best_S = -np.inf, set()
    # try all non-trivial 0/1 assignments (at least one 0 and one 1)
    import itertools
    for w in itertools.product([0, 1], repeat=len(regimes)):
        if 0 not in w or 1 not in w:
            continue
        S = {rk for rk, val in zip(regimes, w) if val == 1}
        weights = np.array([1 if z in S else 0 for z in zhat_train], dtype=float)
        strat_ret = bench_ret.values * weights
        strat_idx = np.cumprod(1.0 + strat_ret)
        cagr_strat = strat_idx[-1] ** (1.0 / max(n_years, 1e-12)) - 1.0
        cagr_rel = (1.0 + cagr_strat) / (1.0 + cagr_bench) - 1.0
        if cagr_rel > best_cagr_rel:
            best_cagr_rel, best_S = cagr_rel, S
    return best_S

def _signals_from_zhat(zhat, S_star):
    z = np.asarray(zhat)
    sig = np.zeros(z.shape, dtype=int)
    for i, val in enumerate(z):
        try:
            k = int(val)
        except (ValueError, TypeError):
            k = None
        sig[i] = 1 if (k is not None and k in S_star) else 0
    return sig

def _gross_exp_on_window(fit, T_req, win=None):
    """Average gross exposure over a reporting window of length T_req (optionally [a,b) slice)."""
    import numpy as _np
    if fit["type"] == "static":
        return float(_np.sum(_np.abs(to_numpy(fit["w"], float))))
    segs = [int(x) for x in fit["segs"]]
    a0, b0 = (0, 10**12) if win is None else (int(win[0]), int(win[1]))
    num = 0.0
    for (a, b), w in zip(zip(segs[:-1], segs[1:]), fit["w_list"]):
        L = max(0, min(b, b0) - max(a, a0))
        if L > 0:
            num += L * float(_np.sum(_np.abs(to_numpy(w, float))))
    return num / max(T_req, 1)

def make_index_rebal(
    intersection_index: pd.DatetimeIndex,
    start_dt: str | None,
    end_dt: str | None,
    rebalance_period_days: int,
) -> tuple[pd.DatetimeIndex, list[int]]:
    """
    Fixed-period rebalancing dates on the INTERSECTION calendar (MVO/DRO).
    Returns:
      index_rebal: DatetimeIndex of rebal dates within [start_dt, end_dt]
      marks:       integer positions into `intersection_index` incl. 0 and T
    """
    idx_req = pd.DatetimeIndex(
        pd.Series(True, index=intersection_index).loc[start_dt:end_dt].index
    )
    if len(idx_req) == 0:
        return idx_req, [0, 0]
    k = int(max(1, rebalance_period_days))
    take = np.arange(0, len(idx_req), k, dtype=int)
    idx_rebal = idx_req.take(take)
    if (len(idx_rebal) == 0) or (idx_rebal[-1] != idx_req[-1]):
        idx_rebal = idx_rebal.append(idx_req[-1:])  # always include last date
    pos = intersection_index.get_indexer(idx_rebal)
    pos = [p for p in pos if p >= 0]
    marks = sorted(set([0] + pos + [len(intersection_index)]))
    return pd.DatetimeIndex(idx_rebal), marks

def make_index_union(
    union_index: pd.DatetimeIndex,
    Z_labels: dict[str, np.ndarray],
    start_dt: str | None,
    end_dt: str | None,
) -> tuple[pd.DatetimeIndex, list[int]]:
    """
    Union of regime-change dates on the UNION calendar (RegDRO).
    Returns:
      index_union: DatetimeIndex of regime-change dates in [start_dt, end_dt]
      taus:        integer breakpoints on union_index (includes 0 and T)
    """
    T = len(union_index)
    if T == 0:
        return pd.DatetimeIndex([]), [0]

    # regime change anywhere
    chg_any = np.zeros(T, dtype=bool)
    for _, z in (Z_labels or {}).items():
        z = np.asarray(z, float)
        finite = np.isfinite(z)
        c = np.zeros(T, dtype=bool)
        if T >= 2:
            c[1:] = finite[1:] & finite[:-1] & (z[1:] != z[:-1])
        chg_any |= c

    idx_req = pd.DatetimeIndex(
        pd.Series(True, index=union_index).loc[start_dt:end_dt].index
    )
    in_req = pd.Index(union_index).isin(idx_req)
    index_union = pd.DatetimeIndex(union_index[chg_any & in_req])

    if len(index_union) == 0:
        taus = [0, T]
    else:
        pos = union_index.get_indexer(index_union)
        pos = [int(p) for p in pos if p >= 0]
        taus = sorted(set([0] + pos + [T]))

    return index_union, taus
    
def _expand_daily_weights(weights_on_dates: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill piecewise weights to every day in full_index."""
    w = weights_on_dates.sort_index()
    if len(w.index) == 0:
        return pd.DataFrame(0.0, index=full_index, columns=weights_on_dates.columns)
    if w.index[0] > full_index[0]:
        zero_row = pd.DataFrame([np.zeros(w.shape[1])], index=[full_index[0]], columns=w.columns)
        w = pd.concat([zero_row, w], axis=0)
    return w.reindex(full_index).ffill().fillna(0.0)

def _expand_daily_signals(signals_on_dates: pd.DataFrame, full_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forward-fill 0/1 signals (rows are sparse change dates) to every day in full_index.
    Returns an int DataFrame (0/1).
    """
    s = signals_on_dates.sort_index()
    if len(s.index) == 0:
        return pd.DataFrame(0, index=full_index, columns=signals_on_dates.columns, dtype=int)
    if s.index[0] > full_index[0]:
        zero_row = pd.DataFrame([np.zeros(s.shape[1])], index=[full_index[0]], columns=s.columns)
        s = pd.concat([zero_row, s], axis=0)
    out = s.reindex(full_index).ffill().fillna(0.0)
    return out.astype(int)

def compute_rslds_hit_rate(
    signals_dense: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    returns_df: pd.DataFrame,
    securities: list[str],
) -> float:
    """
    rSLDS hit rate from a *dense* daily 0/1 signals table.
    Returns a single number: hr_mean = P(return >= 0 | signal == 1) across all secs/days.
    """
    S = signals_dense.copy()
    for m in securities:
        if m not in S.columns:
            S[m] = 0
    S = (S[securities].reindex(full_index).fillna(0) > 0).astype(int)
    R = returns_df.loc[full_index, securities].astype(float)
    X = R.to_numpy(dtype=float, copy=False)
    Z = S.to_numpy(dtype=int, copy=False)
    finite = np.isfinite(X)
    trials = (Z == 1) & finite
    hits = trials & (X >= 0.0)
    n = int(trials.sum())
    return float(hits.sum() / n) if n > 0 else float("nan")
    
def pnl_with_delay_and_cost(
    W_on_dates: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    R_df: pd.DataFrame,
    delay: int,
    tc: float,
    name: str,
):
    """
    Expand weights to daily, apply execution delay and transaction costs, return PnL series.
    Transaction cost model: tc * sum_i |w_eff_t - w_eff_{t-1}| with day-1 turnover = ||w_eff_0||_1.
    """
    W_daily = _expand_daily_weights(W_on_dates, full_index)
    W_eff   = W_daily.shift(int(delay)).fillna(0.0)
    pnl_g   = (W_eff * R_df).sum(axis=1)
    to      = W_eff.diff().abs().sum(axis=1)
    if len(to):
        to.iloc[0] = W_eff.iloc[0].abs().sum()
    return (pnl_g - float(tc) * to).rename(name), W_daily, W_eff

def _feasible_placeholder(N, G):
    import numpy as _np
    N = int(N)
    # Default: all cash if we can't construct a simple feasible vector
    if N <= 0:
        return _np.zeros(0, dtype=float)

    no_short = bool(G.get("no_shorting", False))
    no_lev   = bool(G.get("no_leverage", False))
    max_cash = G.get("max_cash", None)
    cap      = G.get("max_pos_size", None)

    if no_short and no_lev and (max_cash is not None):
        s = max(0.0, 1.0 - float(max_cash))  # target invested weight sum
        if s <= 0.0:
            return _np.zeros(N, dtype=float)

        if cap is None or not _np.isfinite(cap) or cap < 0.0:
            # no cap provided → equal-weight
            return _np.full(N, s / max(N, 1), dtype=float)

        cap = float(cap)
        # Try equal-weight under cap
        per = min(cap, s / max(N, 1))
        w = _np.full(N, per, dtype=float)
        invested = float(w.sum())

        # If per < s/N but cap binds, we may leave extra in cash (fine under no leverage).
        # If we can still add without breaching cap, distribute remainder greedily.
        rem = max(0.0, s - invested)
        if rem > 0 and cap > 0:
            for i in range(N):
                if rem <= 0:
                    break
                add = min(cap - w[i], rem)
                if add > 0:
                    w[i] += add
                    rem  -= add
        return w

    # Fallback: zeros (all cash)
    return _np.zeros(N, dtype=float)

def _period_ends(idx: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Return last available date per period on `idx` (e.g., month-end on trading calendar)."""
    return pd.DatetimeIndex(pd.Series(idx).groupby(idx.to_period(freq)).max())

# -------------------------
# Pipeline
# -------------------------

def dro_pipeline(securities, CONFIG, verbose=True):
    """
    Strict version:
      • ONLY make_index_rebal (MVO/DRO) and make_index_union (RegDRO)
      • Returns panel: px_all.loc[start:end].pct_change().fillna(0.0)
      • Portfolio series via matrix mult on that panel
      • _select_best_config/_expand_weights preserved elsewhere in file
      • δ aggregates included for DRO and RegDRO
      • SPX included in OOS summary table
    """
    G = _make_solver_cfg_from_CONFIG(CONFIG)

    # ----- artifacts -----
    res_csv  = CONFIG["results_csv"]
    seg_parq = CONFIG["segments_parquet"]
    if not os.path.exists(res_csv):  raise FileNotFoundError(res_csv)
    if not os.path.exists(seg_parq): raise FileNotFoundError(seg_parq)

    df_res  = pd.read_csv(res_csv, usecols=range(10), engine="python")
    df_res["security"] = df_res["security"].astype(str).str.strip()

    # securities list from results (or validate provided)
    if securities is None:
        securities = sorted(df_res["security"].unique())
    else:
        req = set(map(str, securities)); have = set(df_res["security"].unique())
        missing = sorted(req - have)
        assert not missing, f"[gridsearch check] Missing in results CSV: {', '.join(missing)}"

    # parquet presence
    seg_hdr = pd.read_parquet(seg_parq, columns=["security"])
    have_seg = set(seg_hdr["security"].astype(str).str.strip().unique())
    missing_seg = sorted(set(securities) - have_seg)
    assert not missing_seg, f"[segments check] Missing in segments parquet: {', '.join(missing_seg)}"

    # ----- data -----
    px_all, _, _, _ = import_data(CONFIG["data_excel"])

    # filter requested to those present
    px_cols = [t for t in securities if t in map(str, px_all.columns)]
    dropped = sorted(set(securities) - set(px_cols))
    if dropped:
        print("[WARN] Dropping securities not found in PX panel:", ", ".join(dropped))
    if not px_cols:
        raise RuntimeError("No securities left after intersecting with PX panel.")
    df_raw = px_all[px_cols].astype(float)
    
    # returns panel
    s = CONFIG["DATA"].get("start_dt")
    e = CONFIG["DATA"].get("end_dt")
    
    # --- extend history by max_lookback_days BEFORE start_dt for fitting ---
    idx  = df_raw.index
    s_dt = pd.to_datetime(s) if s is not None else idx[0]
    e_dt = pd.to_datetime(e) if e is not None else idx[-1]
    
    i_start = idx.get_indexer([s_dt], method="nearest")[0]
    i_end   = idx.get_indexer([e_dt], method="nearest")[0]
    i_hist  = max(0, i_start - int(CONFIG["REBAL"]["max_lookback_days"]))
    
    # History for FITTING = [s_dt - max_lb, e_dt]
    df_raw_slice_hist = df_raw.iloc[i_hist : i_end + 1]
    if df_raw_slice_hist.shape[0] < 2:
        raise RuntimeError("Not enough rows after slicing df_raw with pre-start history.")
    
    # Returns for FITTING (extended history)
    df_returns_full = df_raw_slice_hist.pct_change().fillna(0.0)
    full_index_fit  = df_returns_full.index
    
    # OOS evaluation window = strictly [start_dt, end_dt] on the FIT index
    oos_index = full_index_fit[(full_index_fit >= s_dt) & (full_index_fit <= e_dt)]
    if len(oos_index) == 0:
        raise RuntimeError("Empty OOS index after applying [start_dt, end_dt].")
    
    # ---- define the canonical OOS panel & index used downstream ----
    df_returns = df_returns_full              # keep full history available
    full_index = oos_index                    # OOS index used for PnL/eval

    # returns strictly on [start_dt, end_dt] for PnL
    R_oos = df_returns_full.loc[full_index]   # already fillna(0.0) upstream
    
    # SPX benchmark series (for table + relative stats)
    if "SPX" in px_all.columns:
        spx_daily = pd.to_numeric(px_all["SPX"], errors="coerce").loc[s:e].pct_change().fillna(0.0)
    else:
        spx_daily = pd.Series(index=full_index, dtype=float, name="SPX_daily")

    # ===== MVO & DRO (rebalanced on intersection == full_index) =====
    k_days = int(CONFIG["REBAL"]["rebalance_period_days"])
    if k_days <= 0: raise ValueError("rebalance_period_days must be > 0.")
    # marks on the FIT index (history included)
    index_rebal, marks = make_index_rebal(full_index_fit, s, e, k_days)

    lam    = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
    min_lb = int(CONFIG["REBAL"]["min_lookback_days"])
    max_lb = int(CONFIG["REBAL"]["max_lookback_days"])
    AF     = int(CONFIG.get("annualization_factor", 252))

    R_use = df_returns_full.copy()  # extended history for fitting

    # MVO (piecewise)
    fit_mvo = fit_mvo_rebalanced(R_use, G, AF, marks,
                                 min_lb=min_lb, max_lb=max_lb, lam_shr=lam,
                                 verbose=bool(verbose))

    # --- MVO: build weights-on-dates from piecewise fit, then apply unified PnL helper ---
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))

    # keep only rebal dates >= start_dt so PnL respects the cash cap from day 1 of OOS
    rebal_dates_fit = full_index_fit[marks[:-1]]
    oos_start = oos_index[0]

    mvo_rows = []
    for dt, w in zip(rebal_dates_fit, fit_mvo["w_list"]):
        if dt >= oos_start:
            mvo_rows.append(pd.Series(to_numpy(w, float), index=R_use.columns, name=dt))
    W_rebal_mvo = pd.DataFrame(mvo_rows).sort_index()
        
    mvo_daily, W_daily_mvo, W_eff_mvo = pnl_with_delay_and_cost(
        W_on_dates=W_rebal_mvo,
        full_index=full_index, R_df=R_oos,
        delay=k_delay, tc=tc, name="MVO_daily",)

    # DRO (piecewise)
    params_dro = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])
    fit_dro_pw = fit_dro_rebalanced(R_use, params_dro, G, AF, marks,
                                    min_lb=min_lb, max_lb=max_lb, lam_shr=lam,
                                    verbose=bool(verbose))
    
    # --- DRO: build weights-on-dates from piecewise fit, then apply unified PnL helper ---
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))
    
    dro_rows = []
    for dt, w in zip(rebal_dates_fit, fit_dro_pw["w_list"]):
        if dt >= oos_start:
            dro_rows.append(pd.Series(to_numpy(w, float), index=R_use.columns, name=dt))
    W_rebal_dro = pd.DataFrame(dro_rows).sort_index()
        
    dro_daily, W_daily_dro, W_eff_dro = pnl_with_delay_and_cost(
        W_on_dates=W_rebal_dro,
        full_index=full_index, R_df=R_oos,
        delay=k_delay, tc=tc, name="DRO_daily",)

    # ===== Regime-DRO (piecewise on UNION) =====
    # labels via winning configs
    df_seg = pd.read_parquet(seg_parq)
    df_seg["security"] = df_seg["security"].astype(str).str.strip()
    if df_seg["date"].dtype != "datetime64[ns]":
        df_seg["date"] = pd.to_datetime(df_seg["date"], errors="coerce")

    Z_labels     = {}
    Z_labels_fit = {}  # labels on the FIT calendar (includes pre-start history)
    for sec in px_cols:
        cfg_best = _select_best_config(df_res, sec, CONFIG.get("prefer_configs"))
        if cfg_best is None:
            print(f"[WARN] No winning config in results for {sec}; skipping.")
            continue
        z_ser = _labels_from_segments_df(df_seg, sec, cfg_best)
        if z_ser is None:
            print(f"[WARN] No segments for {sec} under config={cfg_best}; skipping.")
            continue
        # keep OOS-mapped labels for union breaks and display
        Z_labels[sec]     = map_labels_to_calendar(z_ser, full_index)
        # labels mapped to FIT calendar for conditioning inside lookback windows
        Z_labels_fit[sec] = map_labels_to_calendar(z_ser, full_index_fit)

    avail = [t for t in px_cols if t in Z_labels]
    if not avail:
        raise RuntimeError("No assets produced rSLDS labels → cannot run RegDRO.")

    _, taus = make_index_union(full_index,
                               {k: np.asarray(v, float) for k, v in Z_labels.items()},
                               s, e)
    taus = [int(x) for x in taus]

    params_reg = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])
    lookback = int(CONFIG["REBAL"]["max_lookback_days"])
    min_obs  = int(CONFIG["REBAL"]["min_lookback_days"])

    names_all = list(avail)
    pos = {n:i for i,n in enumerate(names_all)}

    # ===== Signals table (FIT range) & Hit Rate =====
    # Build DAILY 0/1 signals once on the FIT calendar [start − lookback : end],
    dt = 1.0 / 252.0
    cal_full = pd.DatetimeIndex(px_all.index)
    first_oos_dt = full_index[0]
    
    # (a) Winning-config labels per asset on full calendar (to learn S* from pre-OOS data)
    z_raw_by_sec = {}
    for sec in names_all:
        cfg_best = _select_best_config(df_res, sec, CONFIG.get("prefer_configs"))
        z_raw_by_sec[sec] = _labels_from_segments_df(df_seg, sec, cfg_best)
    
    # (b) Learn S* per asset using only pre-OOS train
    S_star_by_sec = {}
    for tic in names_all:
        px_ser_full = _num_series(px_all[tic].reindex(cal_full))
        z_ser_full  = z_raw_by_sec[tic].reindex(cal_full).astype("float64")
        train_mask = (cal_full < first_oos_dt)
        train_mask &= z_ser_full.notna().to_numpy()
        train_mask &= px_ser_full.notna().to_numpy()
        if train_mask.sum() >= 2:
            px_train   = px_ser_full.loc[cal_full[train_mask]]
            zhat_train = z_ser_full.loc[cal_full[train_mask]].astype(int).to_numpy()
            S_star_by_sec[tic] = _select_regime_set_S_star_from_train(
                px_train=px_train, zhat_train=zhat_train, dt=dt
            )
        else:
            S_star_by_sec[tic] = set()
    
    # (c) DAILY signals on FIT index [start − max_lookback_days : end] using Z_labels_fit[*]
    signals_fit = pd.DataFrame(
        {
            tic: pd.Series(
                _signals_from_zhat(
                    (lambda z: (
                        lambda zf: (zf[0].__setitem__(zf[1], z[zf[1]].astype(np.int64, copy=False)) or zf[0])
                    )( (np.full(z.shape, -1, dtype=np.int64), np.isfinite(z)) )
                    )(np.asarray(Z_labels_fit[tic], dtype=float)),
                    S_star_by_sec[tic],
                ),
                index=full_index_fit,
                name=tic,
            ).astype(int)
            for tic in names_all
        },
        index=full_index_fit,
    ).astype(int)
    
    # (d) Save once to CSV (no config entry)
    signals_csv_path = "signals_table.csv"
    signals_fit.to_csv(signals_csv_path, index_label="date")
    
    # (e) OOS subset & sparse on-dates (for display/diagnostics)
    signals_fit_oos = signals_fit.loc[full_index, names_all]
    signals_on_dates = signals_fit_oos.iloc[[0]].copy()
    chg = signals_fit_oos.diff().fillna(0).ne(0)
    if chg.values.any():
        chg_dates = signals_fit_oos.index[chg.any(axis=1)]
        signals_on_dates = pd.concat([signals_on_dates, signals_fit_oos.loc[chg_dates]], axis=0).sort_index()
    
    # (f) Hit rate using the precomputed *dense* signals table and the updated function (hr only)
    hr_mean = compute_rslds_hit_rate(
        signals_dense=signals_fit_oos,
        full_index=full_index,
        returns_df=df_returns,
        securities=names_all,)

    if bool(verbose):
        _section("RegDRO")
    
    w_list = []; delta_list = []
    _cap_skips = 0
    _cap_total = len(taus) - 1
    for a, b in zip(taus[:-1], taus[1:]):
        
        t_mid = min(max(a, 0), len(full_index) - 1)
        D     = full_index[t_mid]                    # decision date on OOS calendar
        D_pos = full_index_fit.get_loc(D)            # corresponding position on FIT calendar
        
        # Active at decision date (use FIT index by position to avoid any .loc pitfalls)
        sig_today = signals_fit.iloc[D_pos].reindex(names_all).fillna(0).astype(int)
        ret_today = df_returns_full.iloc[D_pos].reindex(names_all).to_numpy(float)
        A_k = [n for n in names_all if (sig_today[n] == 1) and np.isfinite(ret_today[names_all.index(n)])]

        if not A_k:
            if bool(verbose):
                dt_str = getattr(full_index[t_mid], 'date', lambda: full_index[t_mid])()
                print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta=nan  (skipped: no active assets)")
            w_list.append(np.asarray(_feasible_placeholder(len(names_all), G)))
            delta_list.append(np.nan)
            continue
    
        # regime-conditioned window on FIT history: [D-lookback, D-1]
        D = full_index[t_mid]
        D_pos = full_index_fit.get_loc(D)
        a_win = max(0, D_pos - lookback)   # lookback = CONFIG["REBAL"]["max_lookback_days"]
        b_win = D_pos                      # end exclusive -> up to D-1
        win_idx = full_index_fit.take(np.arange(a_win, b_win, dtype=int))

        # --- Window matrix on FIT window (simple returns) for the **active** set ---
        X_win_df = df_returns_full.loc[win_idx, A_k]

        # --- Per-asset in-regime masks from the **dense signals** table ---
        keep, masks, counts = [], [], []
        for n in A_k:
            m = (signals_fit.loc[win_idx, n].to_numpy(dtype=int) == 1)  # (T_win,) bool
            xj = X_win_df[n].to_numpy(float)
            cnt = int(np.isfinite(xj[m]).sum())
            if cnt >= min_obs:
                keep.append(n); masks.append(m); counts.append(cnt)

        # If nothing passes min_obs, skip before any caps/solve
        if len(keep) == 0:
            if bool(verbose):
                dt_str = getattr(full_index[t_mid], 'date', lambda: full_index[t_mid])()
                print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta=nan  (skipped: insufficient in-regime obs)")
            if w_list:
                w_list.append(np.asarray(w_list[-1], float))
            else:
                w_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            delta_list.append(np.nan)
            continue

        # --- CAP FEASIBILITY (use number of optimization vars = len(keep)) ---
        c_max = float(CONFIG["PORTFOLIO"]["max_cash"])
        u     = float(CONFIG["PORTFOLIO"]["max_pos_size"])

        cap_applies = bool(G.get("no_shorting", False)) and np.isfinite(u) and (u > 0.0)
        N_req  = int(np.ceil((1.0 - c_max) / max(u, 1e-12))) if cap_applies else 0

        if cap_applies and (len(keep) < N_req):
            if bool(verbose):
                print(f"[MIN SAMPLE CHECK] date={pd.to_datetime(D).date()}  N_keep={len(keep)}  N_req={N_req}  -> SKIP")
                dt_str = getattr(full_index[t_mid], 'date', lambda: full_index[t_mid])()
                print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta=nan  (skipped: <min sample)")
            _cap_skips += 1
            if w_list:
                w_list.append(np.asarray(w_list[-1], float))
            else:
                w_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            delta_list.append(np.nan)
            continue
            
        # --- Mean (per asset) and covariance (unconditional on kept assets) ---
        mu_vals = []
        for j, n in enumerate(keep):
            mu_j = compute_mean_from_window(X_win_df[[n]], masks[j], min_obs=min_obs, ann=AF)[0]
            mu_vals.append(float(mu_j))
        mu = np.asarray(mu_vals, dtype=float)
        
        lam = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
        Sig = compute_cov_from_window(X_win_df[keep], ann=AF, shrink_lambda=lam, min_obs=min_obs)
        
        # --- Validate inputs and solve DRO (no try/except; pre-check and print/skip) ---
        X_win = X_win_df[keep].to_numpy(float)
        
        # 1) Finite checks on μ and Σ
        Sig_np = np.atleast_2d(np.asarray(Sig, dtype=float))
        if (not np.isfinite(mu).all()) or (not np.isfinite(Sig_np).all()):
            if bool(verbose):
                dt_str = getattr(full_index[t_mid], 'date', lambda: full_index[t_mid])()
                print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta=nan  (skipped: non-finite mu/Sigma)")
            if w_list:
                w_list.append(np.asarray(w_list[-1], float))
            else:
                w_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            delta_list.append(np.nan)
            continue
        
        # 2) PSD check on Σ (symmetrize, then eigenvals)
        Sig_sym = 0.5 * (Sig_np + Sig_np.T)
        eigvals = np.linalg.eigvalsh(Sig_sym)
        if (eigvals.size == 0) or (not np.isfinite(eigvals).all()) or (eigvals.min() < -1e-10):
            if bool(verbose):
                dt_str = getattr(full_index[t_mid], 'date', lambda: full_index[t_mid])()
                evmin = float(np.nanmin(eigvals)) if eigvals.size else float("nan")
                print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta=nan  (skipped: Sigma not PSD; min eig={evmin:.3e})")
            if w_list:
                w_list.append(np.asarray(w_list[-1], float))
            else:
                w_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            delta_list.append(np.nan)
            continue
        
        # 3) Safe to solve
        w_sub, delta_k = solve_dro(mu, Sig, params_reg, G, R=X_win, verbose=bool(verbose))

        if not np.isfinite(delta_k):
            if bool(verbose):
                dt_str = getattr(full_index[t_mid], 'date', lambda: full_index[t_mid])()
                print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta=nan  (skipped: non-finite delta)")
            if w_list:
                w_list.append(np.asarray(w_list[-1], float))
            else:
                w_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            delta_list.append(np.nan)
            continue
            
        if bool(verbose):
            dt_str = getattr(full_index[t_mid], 'date', lambda: full_index[t_mid])()
            print(f"[RegDRO] t={D_pos} {dt_str}  seg=[{a},{b})  delta={float(delta_k):.4f}")
            excluded = [n for n in A_k if n not in keep]
            print(f"   Excluded(<min_obs={min_obs}): {', '.join(excluded) if excluded else '(none)'}")
    
            _print_mu_by_name(keep, mu)

        # map sub-weights back to full universe
        w_full = np.zeros(len(names_all))
        for j, n in enumerate(keep):
            w_full[pos[n]] = w_sub[j]
        w_list.append(w_full)
        delta_list.append(float(delta_k))

    # --- CAP CHECK summary (percent skipped due to caps) ---
    _cap_total = int(_cap_total)
    if _cap_total <= 0:
        print("\n[MIN SAMPLE CHECK SUMMARY] no regime segments -> nothing to check.")
    else:
        _cap_pct = 100.0 * float(_cap_skips) / float(_cap_total)
        print(f"\n[MIN SAMPLE CHECK SUMMARY] skipped {_cap_skips}/{_cap_total} segments ({_cap_pct:.1f}%) due to <min sample.")

    fit_reg = {
        "type": "piecewise",
        "w_list": [np.asarray(w, float) for w in w_list],
        "segs":   np.asarray(taus, dtype=int),
        "names":  names_all,
        "delta_list": [float(d) if np.isfinite(d) else np.nan for d in delta_list],
    }

    # --- RegDRO: build weights-on-dates from union breaks, then apply unified PnL helper ---
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))
    
    R_reg  = df_returns.loc[full_index, names_all]
    taus   = [int(x) for x in fit_reg["segs"]]
    
    reg_rows = []
    for dt, w in zip(full_index[taus[:-1]], fit_reg["w_list"]):
        reg_rows.append(pd.Series(to_numpy(w, float), index=names_all, name=dt))
        
    W_on_dates_reg = pd.DataFrame(reg_rows).sort_index()
    
    regdro_daily, W_daily_reg, W_eff_reg = pnl_with_delay_and_cost(
        W_on_dates=W_on_dates_reg,
        full_index=full_index, R_df=R_reg,
        delay=k_delay, tc=tc, name="RegDRO_daily",)

    # report weights
    me_idx = _period_ends(full_index, "M")
    H_mvo = W_eff_mvo.reindex(me_idx).ffill().rename_axis("date")
    H_dro = W_eff_dro.reindex(me_idx).ffill().rename_axis("date")
    H_reg = W_eff_reg.reindex(me_idx).ffill().rename_axis("date")

    # ===== OOS summaries (strict s:e window) =====
    def _summ_from_series(series, G, AF, n_days):
        x = series.to_numpy(float)
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
            "max_dd": _max_drawdown_from_series(x)
        }

    AF = int(CONFIG.get("annualization_factor", 252))
    n_aligned = len(full_index)
    spx_daily = spx_daily.reindex(full_index).fillna(0.0)

    # Point estimates first
    rows_mvo = dict(_summ_from_series(mvo_daily,    G, AF, n_aligned))
    rows_dro = dict(_summ_from_series(dro_daily,    G, AF, n_aligned))
    rows_reg = dict(_summ_from_series(regdro_daily, G, AF, n_aligned))
    rows_spx = dict(_summ_from_series(spx_daily,    G, AF, n_aligned))

    # Gross exposure (point)
    a_oos = full_index_fit.get_loc(full_index[0])
    b_oos = full_index_fit.get_loc(full_index[-1]) + 1
    win_oos = (a_oos, b_oos)
    T_req = len(full_index)
    rows_mvo["gross_exp"] = _gross_exp_on_window(fit_mvo,    T_req, win=win_oos)
    rows_dro["gross_exp"] = _gross_exp_on_window(fit_dro_pw, T_req, win=win_oos)
    rows_reg["gross_exp"] = _gross_exp_on_window(fit_reg,    T_req)   # already aligned

    # Deltas: blank for MVO & SPX; aggregated for DRO & RegDRO
    for k in ("delta_mean","delta_min","delta_max"):
        rows_mvo[k] = float("nan")
        rows_spx[k] = float("nan")
    if len(fit_dro_pw.get("delta_list", [])):
        d = pd.Series([float(x) for x in fit_dro_pw["delta_list"] if np.isfinite(x)])
        rows_dro["delta_mean"] = float(d.mean()) if len(d) else np.nan
        rows_dro["delta_min"]  = float(d.min())  if len(d) else np.nan
        rows_dro["delta_max"]  = float(d.max())  if len(d) else np.nan
    if len(fit_reg.get("delta_list", [])):
        d = pd.Series([float(x) for x in fit_reg["delta_list"] if np.isfinite(x)])
        rows_reg["delta_mean"] = float(d.mean()) if len(d) else np.nan
        rows_reg["delta_min"]  = float(d.min())  if len(d) else np.nan
        rows_reg["delta_max"]  = float(d.max())  if len(d) else np.nan

    # Bench-relative point estimates (only for strategy columns)
    def _bench_stats(port, bench, AF=252):
        ex = (port - bench).dropna()
        if ex.empty: return float("nan"), float("nan"), float("nan")
        alpha = AF * ex.mean()
        te    = (AF ** 0.5) * ex.std(ddof=1)
        ir    = alpha / te if (np.isfinite(te) and te != 0) else float("nan")
        return float(alpha), float(te), float(ir)

    for rows, ser in ((rows_mvo, mvo_daily), (rows_dro, dro_daily), (rows_reg, regdro_daily)):
        a, te, ir = _bench_stats(ser, spx_daily, AF)
        rows["alpha_ann"] = a
        rows["te_ann"]    = te
        rows["ir_ann"]        = ir

    # SPX presentation rules (no bench-relative, no deltas, vol_breach blank, ge=1)
    rows_spx["alpha_ann"]  = float("nan")
    rows_spx["te_ann"]     = float("nan")
    rows_spx["ir_ann"]         = float("nan")
    rows_spx["hit_rate"] = float("nan")
    rows_spx["vol_breach"]        = float("nan")
    rows_spx["gross_exp"]         = 1.0

    # Hit-rate vs bench (point): P(port >= bench) on overlapping days
    def _hit_rate(port, bench):
        m = pd.Series(np.isfinite(port) & np.isfinite(bench), index=port.index)
        if not m.any(): return float("nan")
        return float(((port[m] - bench[m]) >= 0.0).mean())

    rows_mvo["hit_rate"] = _hit_rate(mvo_daily, spx_daily)
    rows_dro["hit_rate"] = _hit_rate(dro_daily, spx_daily)
    rows_reg["hit_rate"] = _hit_rate(regdro_daily, spx_daily)
    # SPX left blank by rule

    # --- Bootstraps (strategies only) ---
    B_boot   = int(CONFIG.get("BOOTSTRAP", {}).get("B", 1000))
    L_block  = int(CONFIG.get("BOOTSTRAP", {}).get("avg_block", 10))
    alpha_ci = float(CONFIG.get("BOOTSTRAP", {}).get("alpha", 0.05))
    seed_bs  = CONFIG.get("BOOTSTRAP", {}).get("seed", None)

    bb_mvo = block_bootstrap_oos(mvo_daily,    spx_daily, G, AF, B=B_boot, avg_block=L_block, alpha=alpha_ci, seed=seed_bs)
    bb_dro = block_bootstrap_oos(dro_daily,    spx_daily, G, AF, B=B_boot, avg_block=L_block, alpha=alpha_ci, seed=seed_bs)
    bb_reg = block_bootstrap_oos(regdro_daily, spx_daily, G, AF, B=B_boot, avg_block=L_block, alpha=alpha_ci, seed=seed_bs)
    # NOTE: no SPX bootstrap — we never attach *_ci_* to SPX

    def _apply_bb(rows: dict, bb: dict):
        for k, trip in bb.items():
            if k in rows:
                rows[f"{k}_ci_low"]  = trip["ci_low"]
                rows[f"{k}_ci_high"] = trip["ci_high"]
        return rows

    rows_mvo = _apply_bb(rows_mvo, bb_mvo)
    rows_dro = _apply_bb(rows_dro, bb_dro)
    rows_reg = _apply_bb(rows_reg, bb_reg)

    # Assemble DataFrames for the table
    df_mvo = pd.DataFrame([rows_mvo])
    df_dro = pd.DataFrame([rows_dro])
    df_reg = pd.DataFrame([rows_reg])
    df_spx = pd.DataFrame([rows_spx])

    results_dict = {"MVO": df_mvo, "DRO": df_dro, "RegDRO": df_reg, "SPX": df_spx}
    print_oos_table(results_dict, model_order=["MVO", "DRO", "RegDRO", "SPX"])
    
    # outputs
    out = {
        "MVO":     {"fit": fit_mvo,     "summary": rows_mvo},
        "DRO":     {"fit": fit_dro_pw,  "summary": rows_dro},
        "RegDRO":  {"fit": fit_reg,     "summary": rows_reg,
                    "global_segs": [int(x) for x in fit_reg["segs"]],
                    "Z_labels": {k: np.asarray(v) for k, v in Z_labels.items()}},
        "SPX":     {"series": spx_daily, "summary": rows_spx},
        "returns": df_returns,
        "series": {
            "MVO_daily": mvo_daily,
            "DRO_daily": dro_daily,
            "RegDRO_daily": regdro_daily,
            "SPX_daily": spx_daily,
        },
        "securities": names_all,
        "G": G,
        "signals": {
        "signals_on_dates": signals_on_dates,
        "signals_oos": signals_fit_oos,
        "rSLDS_hit_rate": float(hr_mean),
        },
        "holdings": {"MVO": H_mvo, "DRO": H_dro, "RegDRO": H_reg},
    }

    if "dro_pickle" in CONFIG and CONFIG["dro_pickle"]:
        save_out(out, CONFIG["dro_pickle"])

    return out
