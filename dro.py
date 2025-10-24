 
# ---------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------

# GPU if available
try:
    import cupy as xp
    from cupyx.scipy.optimize import linear_sum_assignment  # GPU Hungarian
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
import io, gzip, pickle
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
                idx1 = xp.random.randint(0, n_src, size=m)
                idx2 = xp.random.randint(0, n_src, size=m)
            else:
                idx1 = rng_np.integers(0, n_src, size=m)
                idx2 = rng_np.integers(0, n_src, size=m)
            idx1 = xp.asarray(idx1, dtype=xp.int64)
            idx2 = xp.asarray(idx2, dtype=xp.int64)
            dists.append(sliced_w2_empirical(R_xp[idx1], R_xp[idx2], n_proj=n_proj, rng=None))
 
        delta_daily = float(xp.quantile(xp.asarray(dists), 1.0 - alpha))
        return AF * delta_daily

    if method == "bootstrap_gaussian":
        assert R is not None, "bootstrap_gaussian needs raw sample matrix R."
        alpha = float((params or {}).get("alpha", 0.05))
        B     = int((params or {}).get("B", 512))
        eps   = float((params or {}).get("epsilon_sigma", 1e-9))
        rng   = _rng_from_params(params or {})
        delta_daily = bootstrap_gaussian_delta(R, alpha=alpha, B=B, eps=eps, rng=rng)
        return AF * float(delta_daily)

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
    try:
        # single-shot minimal regularization
        C = xp.linalg.cholesky(Sigma_sym + eps * xp.eye(Sigma_sym.shape[0]))
    except xp.linalg.LinAlgError:
        # project once, no extra +eps*I
        vals, vecs = xp.linalg.eigh(Sigma_sym)
        vals = xp.clip(vals, eps, None)
        Sigma_psd = vecs @ xp.diag(vals) @ vecs.T
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
        prob.solve(solver=cp.MOSEK, verbose=verbose)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=verbose)

    if (w.value is None) or (prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)):
        raise RuntimeError(f"ECOS/MOSEK failed: status={prob.status}")

    return xp.asarray(_np.asarray(w.value).reshape(-1))


# ---------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------

def solve_mvo(mu, Sigma, G, *, verbose=False):
    """MVO = DRO with δ=0."""
    return solve_optimizer(mu, Sigma, delta=0.0, config=G, verbose=verbose)

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
    # segs_preview, k_preview = _count_segments_from_params_or_data(data, params)
    # _section(f"Regime-DRO — planned number of segments: {k_preview}")
    # print(f"Segments (indices): {segs_preview}")

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

    # _section("Regime-DRO — segments to be used (final)")
    # print(f"Final segments (indices): {segs}")
    if bool(params.get("verbose", False)) and params.get("log_segments", False):
        segs_preview, k_preview = _count_segments_from_params_or_data(data, params)
        _section(f"Regime-DRO — planned number of segments: {k_preview}")
        print(f"Segments (indices): {segs_preview}")
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
        stats_oos["gap_oos_vs_train_realized"] = float(stats_oos["sigma_ann"] - sigma_train_ann)
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
        stats_oos["gap_oos_vs_train_realized"] = float(stats_oos["sigma_ann"] - sigma_train_ann)
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
        "max_drawdown": max_dd,}

# ---------------------------------------------------------------
# Statistical tests (for hypothesis testing)
# ---------------------------------------------------------------

def format_ci(mean, std):
    return f"{mean:.4f} ({(mean - std):.4f}, {(mean + std):.4f})"
    
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

def paired_t_twosided(x, y):
    # H0: mean(x - y) == 0  vs  H1: ≠ 0
    d, n = _paired_diff(x, y)
    if n < 2:
        return float("nan"), float("nan")
    t, p = sp_stats.ttest_1samp(d, popmean=0.0, alternative="two-sided")
    return t, p

def noninferiority_paired(x, y, delta):
    # H0: mean(x - y) <= -delta  vs  H1: > -delta
    d, n = _paired_diff(x, y)
    if n < 2:
        return float("nan"), float("nan")
    t, p = sp_stats.ttest_1samp(d + float(delta), popmean=0.0, alternative="greater")
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
    
    base_rows = [
        "mu_ann","sigma_ann","sharpe_ann","vol_breach",
        # "p_viol",
        "gross_exp",
        "gap_oos_vs_train_realized",
        "delta_mean","delta_min","delta_max",
        "max_drawdown",
        "alpha_ann_vs_spx","te_ann_vs_spx","ir_vs_spx",
        "hit_rate_vs_bench","hit_rate_vs_bench_se",
        "hit_rate_vs_bench_ci_low","hit_rate_vs_bench_ci_high",
]

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
   
    T = pd.DataFrame(table).reindex(rows)
    T = T.drop(index=[i for i in T.index if T.loc[i].isna().all()])
    return T

def print_oos_table(results_dict, model_order):
    model_order = [m for m in model_order if m in results_dict and len(results_dict[m]) > 0]
    if not model_order:
        print("\nNo models to display."); return
    
    n_by_model = {m: len(results_dict[m]) for m in model_order}
    single = all(n == 1 for n in n_by_model.values()) and len(n_by_model) > 0

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
    _raw_dlist = [fit.get(f"delta_k{k+1}", None) for k in range(len(segs)-1)]
    def _to_float_or_nan(x):
        try:
            return float(np.asarray(x).squeeze())
        except Exception:
            return np.nan
    _dlist = np.array([_to_float_or_nan(x) for x in _raw_dlist], dtype=float)
    if not np.isfinite(_dlist).any():
        _dlist = np.array([_to_float_or_nan(x) for x in fit.get("delta_list", [])], dtype=float)
    dlist = _dlist.tolist()

    # ---- robust taus_true handling ----
    if taus_true is None:
        taus_true = data.get("taus_true", None)
    # if still None or falsy, default to [0, n_days]
    if not taus_true:
        n_days = int(data.get("n_days", len(data.get("train", []))))
        taus_true = [0, n_days]

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

def make_index_opt_from_parquet(segments_df: pd.DataFrame,
                                start_date: str | None,
                                end_date: str | None) -> pd.DatetimeIndex:
    """
    segments_df columns required: ['date'] (plus anything else, e.g., security, z).
    Builds the available datetime index from the parquet, then applies slicing.
    """
    if "date" not in segments_df.columns:
        raise ValueError("segments_df must contain a 'date' column.")
    dates = pd.to_datetime(segments_df["date"], errors="coerce").dropna().sort_values().unique()
    cal_all = pd.DatetimeIndex(dates)
    # slice to [start_date:end_date] exactly as agreed
    index_opt = pd.DatetimeIndex(pd.Series(True, index=cal_all).loc[start_date:end_date].index)
    return index_opt

def make_index_union_from_parquet(segments_df: pd.DataFrame,
                                  index_opt: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    segments_df must have ['security','date','z'].
    Returns dates within index_opt where at least one security changes regime
    relative to the previous date in index_opt.
    """
    if len(index_opt) == 0:
        return pd.DatetimeIndex([])

    req = {"security", "date", "z"}
    missing = req - set(segments_df.columns)
    if missing:
        raise ValueError(f"segments_df missing required columns: {missing}")

    df = segments_df.loc[:, ["security", "date", "z"]].copy()
    df["security"] = df["security"].astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["security", "date"])
    # keep last label per (security, date)
    df = df.drop_duplicates(subset=["security", "date"], keep="last")

    # wide matrix of labels, reindexed to index_opt and forward-filled *inside* window
    Z_wide = df.pivot(index="date", columns="security", values="z").sort_index()
    Z_opt = Z_wide.reindex(index_opt).ffill()

    if Z_opt.shape[0] <= 1:
        return pd.DatetimeIndex([])

    Z = Z_opt.to_numpy(dtype=float)   # << enforce float
    finite_t   = np.isfinite(Z[1:, :])
    finite_tm1 = np.isfinite(Z[:-1, :])
    changed = finite_t & finite_tm1 & (Z[1:, :] != Z[:-1, :])
    any_changed = changed.any(axis=1)                # len = len(index_opt)-1
    return pd.DatetimeIndex(index_opt[1:][any_changed])

def make_index_rebal_from_opt(index_opt: pd.DatetimeIndex,
                              rebalance_period_days: int) -> tuple[pd.DatetimeIndex, list[int]]:
    """
    Returns (index_rebal, marks); marks are integer positions in index_opt incl. 0 and T.
    """
    if len(index_opt) == 0:
        return pd.DatetimeIndex([]), [0, 0]

    k = int(max(1, rebalance_period_days))
    take = np.arange(0, len(index_opt), k, dtype=int)
    index_rebal = index_opt.take(take)
    # always include last date
    if (len(index_rebal) == 0) or (index_rebal[-1] != index_opt[-1]):
        index_rebal = pd.DatetimeIndex(np.concatenate([index_rebal.values, index_opt[-1:].values]))

    pos = index_opt.get_indexer(index_rebal)
    pos = [int(p) for p in pos if p >= 0]
    marks = sorted(set([0] + pos + [len(index_opt)]))
    return index_rebal, marks

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
        "train": xp.asarray(X),     # keep NaNs for series; handle NaNs only inside moment estimators
        "test":  xp.asarray(X),
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

def _port_series_from_fit(fit, R):
    """Daily portfolio series r_t from weights (static or piecewise) on matrix R (T×N)."""
    import numpy as _np
    R = _np.asarray(R, float)
    T = R.shape[0]
    if fit["type"] == "static":
        return R @ _np.asarray(fit["w"]).reshape(-1)
    s = _np.zeros(T, float)
    segs = [int(x) for x in fit["segs"]]
    for (a, b), w in zip(zip(segs[:-1], segs[1:]), fit["w_list"]):
        a = max(0, min(a, T)); b = max(0, min(b, T))
        if b > a:
            s[a:b] = R[a:b] @ _np.asarray(w).reshape(-1)
    return s

def _gross_exp_on_window(fit, T_req, win=None):
    """Average gross exposure over a reporting window of length T_req (optionally [a,b) slice)."""
    import numpy as _np
    if fit["type"] == "static":
        return float(_np.sum(_np.abs(fit["w"])))
    segs = [int(x) for x in fit["segs"]]
    a0, b0 = (0, 10**12) if win is None else (int(win[0]), int(win[1]))
    num = 0.0
    for (a, b), w in zip(zip(segs[:-1], segs[1:]), fit["w_list"]):
        L = max(0, min(b, b0) - max(a, a0))
        if L > 0:
            num += L * float(_np.sum(_np.abs(w)))
    return num / max(T_req, 1)

def _delta_aggregates(fit):
    import numpy as _np
    if fit["type"] == "static":
        d = float(fit.get("delta", _np.nan))
        return d, d, d
    dlist = _np.asarray(fit.get("delta_list", []), float)
    dlist = dlist[_np.isfinite(dlist)]
    if dlist.size == 0:
        return _np.nan, _np.nan, _np.nan
    return float(dlist.mean()), float(dlist.min()), float(dlist.max())

def _build_indices_from_calendar(
    cal_all: pd.DatetimeIndex,
    start_dt: str | None,
    end_dt: str | None,
    rebalance_period_days: int,
    Z_labels: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Returns:
      index_opt   : cal_all sliced to [start_dt:end_dt]
      index_union : dates in index_opt where ANY asset changes regime
      index_rebal : every k-th date in index_opt (include last)
    """
    # 1) slice → index_opt
    index_opt = pd.DatetimeIndex(pd.Series(True, index=cal_all).loc[start_dt:end_dt].index)

    # 2) union of regime-change dates → index_union (if labels provided)
    if not Z_labels:
        index_union = pd.DatetimeIndex([])
    else:
        T = len(cal_all)
        chg_any = np.zeros(T, dtype=bool)
        for _, z in Z_labels.items():
            z = np.asarray(z, float)
            finite = np.isfinite(z)
            c = np.zeros(T, dtype=bool)
            if T >= 2:
                c[1:] = finite[1:] & finite[:-1] & (z[1:] != z[:-1])
            chg_any |= c
        in_opt = pd.Index(cal_all).isin(index_opt)
        index_union = pd.DatetimeIndex(cal_all[chg_any & in_opt])

    # 3) fixed-period rebalancing → index_rebal
    k = int(max(1, rebalance_period_days))
    if len(index_opt) == 0:
        index_rebal = index_opt
    else:
        take = np.arange(0, len(index_opt), k, dtype=int)
        idx = index_opt.take(take)
        if (len(idx) == 0) or (idx[-1] != index_opt[-1]):
            idx = pd.DatetimeIndex(np.concatenate([idx.values, index_opt[-1:].values]))
        index_rebal = pd.DatetimeIndex(idx)

    return index_opt, index_union, index_rebal

def union_returns_calendar(securities, px_all) -> pd.DatetimeIndex:
    """
    Union calendar of *return* dates across all requested securities
    (built directly from the raw price panel).
    """
    cal = None
    for t in securities:
        r_idx = (pd.to_numeric(px_all[t], errors="coerce")
                   .astype("float64")
                   .dropna()
                   .pct_change()
                   .dropna()
                   .index)
        cal = r_idx if cal is None else cal.union(r_idx)
    if cal is None or len(cal) == 0:
        raise RuntimeError("Empty union calendar of returns.")
    return pd.DatetimeIndex(cal)
    
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



# ----------------------
# Helpers
# ----------------------

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
    signals_on_dates: pd.DataFrame,
    full_index: pd.DatetimeIndex,
    returns_df: pd.DataFrame,
    securities: list[str],
) -> tuple[float, float, int, pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:

    """
    Forward-fill signals to full_index, then compute rSLDS hit rate across ALL
    securities & days:
        Indicator I_{t,i} = 1{ signal_{t,i} == 1 and return_{t,i} >= 0 }.
    Returns:
        (hr_mean, hr_std, n_trials, signals_ffill, hits_mask_df, trials_mask_df, ci_low, ci_high)
    
    """
    # forward-fill signals (0/1) on full index
    signals_ffill = _expand_daily_signals(signals_on_dates, full_index)  # T×N int
    # align to the requested securities and index
    S = signals_ffill.reindex(full_index)[securities].to_numpy(dtype=int, copy=False)
    R = returns_df.loc[full_index, securities].astype(float).to_numpy(copy=False)

    finite = np.isfinite(R)
    trials = (S == 1) & finite               # denominator mask
    hits   = trials & (R >= 0.0)             # numerator mask

    n = int(trials.sum())
    if n == 0:
        hr_mean, hr_std = float("nan"), float("nan")
        ci_low, ci_high = float("nan"), float("nan")
    else:
        p = float(hits.sum() / n)
        # sample std of Bernoulli across trials
        hr_std = float(np.sqrt(p * (1.0 - p) * (n / (n - 1.0)))) if n > 1 else 0.0
        hr_mean = p
        # Wilson CI
        k = int(hits.sum())
        ci_low, ci_high = _wilson_ci(k, n, alpha=0.05)

    hits_df   = pd.DataFrame(hits,   index=full_index, columns=securities).astype(bool)
    trials_df = pd.DataFrame(trials, index=full_index, columns=securities).astype(bool)
    return hr_mean, hr_std, n, signals_ffill.astype(int), hits_df, trials_df, ci_low, ci_high

        
def _wilson_ci(k: int, n: int, alpha: float = 0.05):
    """
    95% Wilson score interval for a binomial proportion by default (alpha=0.05).
    Returns (ci_low, ci_high). If n==0 returns (nan, nan).
    """
    if n <= 0:
        return float("nan"), float("nan")
    from math import sqrt
    p = k / n
    try:
        z = sp_stats.norm.ppf(1 - alpha / 2.0)
    except Exception:
        z = 1.959963984540054  # ~N(0,1) 97.5% quantile
    denom = 1.0 + (z**2) / n
    center = p + (z**2) / (2.0 * n)
    margin = z * sqrt((p * (1 - p) + (z**2) / (4.0 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return float(max(0.0, low)), float(min(1.0, high))

def hit_rate_vs_bench_stats(model_series, bench_series, index_ref, alpha=0.05):
    a = pd.Series(model_series).reindex(index_ref).astype(float)
    b = pd.Series(bench_series).reindex(index_ref).astype(float)
    mask = a.notna() & b.notna()
    if not mask.any():
        return float("nan"), float("nan"), 0, float("nan"), float("nan")

    z = (a[mask] - b[mask] >= 0.0).astype(float).to_numpy()
    n = int(z.size)
    p_hat = float(z.mean())
    se = float(np.sqrt(p_hat * (1.0 - p_hat) / n)) if n > 0 else float("nan")
    k = int(z.sum())
    ci_low, ci_high = _wilson_ci(k, n, alpha=alpha)
    return p_hat, se, n, ci_low, ci_high

def _shrunk_cov(X: np.ndarray, lam: float) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0, keepdims=True)
    T = max(1, X.shape[0] - 1)
    S = (Xc.T @ Xc) / T if X.size else np.zeros((X.shape[1], X.shape[1]))
    tr = np.trace(S) / S.shape[0] if S.shape[0] else 0.0
    return (1.0 - lam) * S + lam * tr * np.eye(S.shape[0])

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

# ----------------------
# Optimizers
# ----------------------
def run_mvo(df_px, df_ret, index_rebal, CONFIG, G, solve_mvo, verbose=True):
    tickers = list(df_px.columns)
    rebal_dates = df_px.index.intersection(index_rebal)
    if len(rebal_dates) == 0:
        raise RuntimeError("index_rebal has no overlap with price calendar.")
    k_min = int(CONFIG["REBAL"]["min_lookback_days"])
    k_max = int(CONFIG["REBAL"]["max_lookback_days"])
    lam   = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])

    rows = []
    for dt in rebal_dates:
        loc = df_px.index.get_loc(dt)
        start_loc = max(0, loc - k_max)
        hist_idx = df_px.index[start_loc:loc]
        X = df_ret.loc[hist_idx, tickers].to_numpy(float)
        if X.shape[0] < k_min:
            w = np.zeros(len(tickers))
        else:
            mu  = X.mean(axis=0) * 252.0
            Sig = _shrunk_cov(X, lam)
            w   = solve_mvo(mu, Sig, G)
        rows.append(pd.Series(w, index=tickers, name=dt))

    W_rebal = pd.DataFrame(rows).sort_index()

    # --- execution delay & trading costs ---
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))
    
    pnl_mvo, W_daily, _W_eff = pnl_with_delay_and_cost(
        W_on_dates=W_rebal, full_index=df_px.index,
        R_df=df_ret, delay=k_delay, tc=tc, name="MVO_daily",)

    return W_daily, pnl_mvo

def run_dro(df_px, df_ret, index_rebal, CONFIG, G, solve_dro, verbose=True):
    tickers = list(df_px.columns)
    rebal_dates = df_px.index.intersection(index_rebal)
    if len(rebal_dates) == 0:
        raise RuntimeError("index_rebal has no overlap with price calendar.")
    k_min = int(CONFIG["REBAL"]["min_lookback_days"])
    k_max = int(CONFIG["REBAL"]["max_lookback_days"])
    lam   = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
    params = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])

    rows = []
    for dt in rebal_dates:
        loc = df_px.index.get_loc(dt)
        start_loc = max(0, loc - k_max)
        hist_idx = df_px.index[start_loc:loc]
        X = df_ret.loc[hist_idx, tickers].to_numpy(float)
        if X.shape[0] < k_min:
            w = np.zeros(len(tickers))
        else:
            mu  = X.mean(axis=0) * 252.0
            Sig = _shrunk_cov(X, lam)
            w, _delta = solve_dro(mu, Sig, params, G, R=X)
        rows.append(pd.Series(w, index=tickers, name=dt))

    W_rebal = pd.DataFrame(rows).sort_index()
    
    # --- execution delay & trading costs ---
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))
    
    pnl_dro, W_daily, _W_eff = pnl_with_delay_and_cost(
        W_on_dates=W_rebal, full_index=df_px.index,
        R_df=df_ret, delay=k_delay, tc=tc, name="DRO_daily",)
    
    return W_daily, pnl_dro

def run_regdro(df_px, df_ret, index_union, Z_labels, CONFIG, G, solve_dro, verbose=True):
    tickers = list(df_px.columns)
    seg_dates = df_px.index.intersection(index_union)
    if len(seg_dates) == 0:
        raise RuntimeError("index_union has no overlap with price calendar.")

    min_obs  = int(CONFIG["REBAL"]["min_lookback_days"])
    lookback = int(CONFIG["REBAL"]["max_lookback_days"])
    lam      = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
    paramsR  = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])

    # align labels to price calendar (cleaner)
    Z = {k: pd.Series(v, index=df_px.index).to_numpy(float) for k, v in Z_labels.items()}
    
    rows = []
    for dt in seg_dates:
        loc = df_px.index.get_loc(dt)
        start_loc = max(0, loc - lookback)
        win_idx = df_px.index[start_loc:loc]

        A_keep, cols = [], []
        for n in tickers:
            z_mid = Z[n][loc] if np.isfinite(Z[n][loc]) else np.nan
            if not np.isfinite(z_mid):
                continue
            mask = (Z[n] == z_mid)
            r_in_regime = df_ret.loc[win_idx, n][mask[start_loc:loc]]
            if r_in_regime.shape[0] >= min_obs:
                A_keep.append(n)
                cols.append(r_in_regime.rename(n))

        if not A_keep:
            if rows:
                rows.append(rows[-1].rename(dt))
            else:
                rows.append(pd.Series(_feasible_placeholder(len(tickers), G), index=tickers, name=dt))
            continue

        Rk  = pd.concat(cols, axis=1).reindex(columns=A_keep).fillna(0.0).to_numpy(float)
        mu  = Rk.mean(axis=0) * 252.0
        Sig = _shrunk_cov(Rk, lam)
        w_sub, _delta = solve_dro(mu, Sig, paramsR, G, R=Rk)

        w_full = np.zeros(len(tickers)); pos = {t:i for i,t in enumerate(tickers)}
        for j, n in enumerate(A_keep): w_full[pos[n]] = w_sub[j]
        rows.append(pd.Series(w_full, index=tickers, name=dt))

    W_union = pd.DataFrame(rows).sort_index()
    
    # --- execution delay & trading costs ---
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))

    pnl_reg, W_daily, _W_eff = pnl_with_delay_and_cost(
        W_on_dates=W_union, full_index=df_px.index, R_df=df_ret,
        delay=k_delay, tc=tc, name="RegDRO_daily",)
    
    return W_daily, pnl_reg

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
            mvo_rows.append(pd.Series(np.asarray(w, float), index=R_use.columns, name=dt))
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
            dro_rows.append(pd.Series(np.asarray(w, float), index=R_use.columns, name=dt))
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

    Z_labels = {}
    for sec in px_cols:
        cfg_best = _select_best_config(df_res, sec, CONFIG.get("prefer_configs"))
        if cfg_best is None:
            print(f"[WARN] No winning config in results for {sec}; skipping.")
            continue
        z_ser = _labels_from_segments_df(df_seg, sec, cfg_best)
        if z_ser is None:
            print(f"[WARN] No segments for {sec} under config={cfg_best}; skipping.")
            continue
        Z_labels[sec] = map_labels_to_calendar(z_ser, full_index)

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
    
    # ===== Signals & Hit Rate (OOS)
    dt = 1.0 / 252.0
    cal_full = pd.DatetimeIndex(px_all.index)
    first_oos_dt = full_index[0]

    # Pull raw z-series per asset on full calendar (no ffill/bfill), keyed by their winning config
    z_raw_by_sec = {}
    for sec in names_all:
        cfg_best = _select_best_config(df_res, sec, CONFIG.get("prefer_configs"))
        z_raw_by_sec[sec] = _labels_from_segments_df(df_seg, sec, cfg_best)

    R_oos = df_returns.loc[full_index, names_all]

    signals_oos_cols = {}
    hit_rate_by_asset = {}
    total_hits = 0
    total_trials = 0

    for tic in names_all:
        # full-calendar price & labels
        px_ser_full = _num_series(px_all[tic].reindex(cal_full))
        z_ser_full  = z_raw_by_sec[tic].reindex(cal_full).astype("float64")

        # TRAIN = strictly before OOS window start
        train_mask = (cal_full < first_oos_dt)
        train_mask &= z_ser_full.notna().to_numpy()
        train_mask &= px_ser_full.notna().to_numpy()

        if train_mask.sum() >= 2:
            px_train   = px_ser_full.loc[cal_full[train_mask]]
            zhat_train = z_ser_full.loc[cal_full[train_mask]].astype(int).to_numpy()
            S_star = _select_regime_set_S_star_from_train(px_train=px_train, zhat_train=zhat_train, dt=dt)
        else:
            S_star = set()

        # OOS signals on requested window
        z_oos = z_ser_full.reindex(full_index).to_numpy()
        z_oos_int = np.full(z_oos.shape, -1, dtype=int)
        finite = np.isfinite(z_oos)
        z_oos_int[finite] = z_oos[finite].astype(int, copy=False)
        sig = _signals_from_zhat(z_oos_int, S_star)  # 1 if in S*, else 0
        sig_ser = pd.Series(sig, index=full_index, name=tic).astype(int)
        signals_oos_cols[tic] = sig_ser

        # per-asset hit rate: P(return >= 0 | signal == 1)
        mask = (sig_ser == 1) & R_oos[tic].notna()
        trials = int(mask.sum())
        hits   = int(((R_oos[tic] >= 0) & mask).sum())
        hit_rate_by_asset[tic] = (hits / trials) if trials > 0 else 0.0
        total_hits   += hits
        total_trials += trials

    signals_oos = pd.DataFrame(signals_oos_cols, index=full_index).astype(int)

    # -- build sparse "on dates" series (first day + change points), then forward-fill
    signals_dense = pd.DataFrame(signals_oos_cols, index=full_index).astype(int)
    signals_on_dates = signals_dense.loc[[full_index[0]]].copy()
    changes = signals_dense.diff().fillna(0).ne(0)  # boolean change points per asset
    if changes.values.any():
        change_dates = full_index[changes.any(axis=1)]
        signals_on_dates = pd.concat([signals_on_dates, signals_dense.loc[change_dates]], axis=0).sort_index()
    
    # rSLDS hit rate (overall) — mean & std across ALL secs/days (signal==1 trials)
    hr_mean, hr_std, hr_n, signals_ffill, rSLDS_mask_df, trials_df, hr_ci_low, hr_ci_high = compute_rslds_hit_rate(
        signals_on_dates=signals_on_dates,
        full_index=full_index,
        returns_df=df_returns,
        securities=names_all,
    )
    
    # per-asset diagnostics
    sig_counts = signals_ffill.sum(axis=0).replace(0, np.nan)
    hits_counts = rSLDS_mask_df.sum(axis=0)
    rSLDS_hit_rate_by_asset = (hits_counts / sig_counts).astype(float).fillna(0.0)
    
    # macro (equal-weight across assets with >=1 trial)
    valid_assets = sig_counts.notna()
    macro_hr = float(rSLDS_hit_rate_by_asset[valid_assets].mean()) if valid_assets.any() else float("nan")
    
    # print BEFORE OOS summary
    _section("rSLDS hit rate (overall)")
    print(f"rSLDS hit rate — mean: {hr_mean:.4f}, std: {hr_std:.4f}, n={hr_n:d}, CI95=({hr_ci_low:.4f}, {hr_ci_high:.4f})")
    print(f"rSLDS macro-avg hit rate (per-asset): {macro_hr:.4f}")

    w_list = []; delta_list = []
    _cap_skips = 0
    _cap_total = len(taus) - 1
    for a, b in zip(taus[:-1], taus[1:]):
        t_mid = min(max(a, 0), len(full_index) - 1)

        # active at t_mid
        A_k = [n for n in names_all
               if np.isfinite(Z_labels[n][t_mid]) and np.isfinite(df_returns[n].iloc[t_mid])]
        if not A_k:
            w_list.append(np.asarray(_feasible_placeholder(len(names_all), G)))
            delta_list.append(np.nan)
            continue
    
        # regime-conditioned window [t0, t_mid]
        t0 = max(0, t_mid - lookback + 1)
        win_idx = full_index[t0:t_mid+1]

        cols = []; keep = []
        for n in A_k:
            z_now = int(Z_labels[n][t_mid]) if np.isfinite(Z_labels[n][t_mid]) else None
            if z_now is None: continue
            mask = (pd.Series(Z_labels[n], index=full_index).loc[win_idx] == z_now)
            r_in_reg = df_returns.loc[win_idx, n].loc[mask]
            if r_in_reg.shape[0] >= min_obs:
                keep.append(n)
                cols.append(r_in_reg.rename(n))

        if not keep:
            if w_list:
                w_list.append(w_list[-1])
            else:
                w_list.append(np.asarray(_feasible_placeholder(len(names_all), G)))
            delta_list.append(np.nan)
            continue

        # --- CAP CHECK (skip & forward-fill instead of crash) ---
        c_max = float(CONFIG["PORTFOLIO"]["max_cash"])
        u     = float(CONFIG["PORTFOLIO"]["max_pos_size"])
        N_act = len(keep)
        N_req = int(np.ceil((1.0 - c_max) / max(u, 1e-12)))
        dt    = full_index[t_mid]
        if (G.get("no_shorting", False)) and (N_act < N_req):
            print(f"[CAP CHECK] date={pd.to_datetime(dt).date()}  N_act={N_act}  N_req={N_req}  -> SKIP")
            _cap_skips += 1
            # forward-fill weights & delta
            if w_list:
                w_list.append(np.asarray(w_list[-1], float))
            else:
                w_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            delta_list.append(np.nan)
            continue

        Rk  = pd.concat(cols, axis=1).reindex(columns=keep).fillna(0.0).to_numpy(float)
        mu  = Rk.mean(axis=0) * AF
        Xc  = Rk - Rk.mean(0)
        Sig = (Xc.T @ Xc) / max(1, Rk.shape[0]-1)
        lam = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
        Sig = (1-lam)*Sig + lam*np.trace(Sig)/Sig.shape[0]*np.eye(Sig.shape[0])

        w_sub, delta_k = solve_dro(mu, Sig, params_reg, G, R=Rk, verbose=False)
        w_full = np.zeros(len(names_all))
        for j, n in enumerate(keep): w_full[pos[n]] = w_sub[j]
        w_list.append(w_full); delta_list.append(float(delta_k))

        if verbose:
            print(f"[RegDRO] seg [{a},{b})  t_mid={t_mid}  eligible={keep}  delta_k={float(delta_k):.6g}")

    # --- CAP CHECK summary (percent skipped due to caps) ---
    if _cap_total > 0:
        _cap_pct = 100.0 * _cap_skips / _cap_total
        print(f"[CAP CHECK SUMMARY] skipped {_cap_skips}/{_cap_total} segments ({_cap_pct:.1f}%) due to caps.")

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
        reg_rows.append(pd.Series(np.asarray(w, float), index=names_all, name=dt))
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
            "max_drawdown": _max_drawdown_from_series(x)
        }

    AF = int(CONFIG.get("annualization_factor", 252))
    n_aligned = len(full_index)
    
    spx_daily = spx_daily.reindex(full_index).fillna(0.0)
    
    summ_mvo  = _summ_from_series(mvo_daily, G, AF, n_aligned)
    summ_dro  = _summ_from_series(dro_daily, G, AF, n_aligned)
    summ_reg  = _summ_from_series(regdro_daily, G, AF, n_aligned)
    summ_spx  = _summ_from_series(spx_daily, G, AF, n_aligned)
    
    def _bench_stats(port, bench, AF=252):
        ex = (port - bench).dropna()
        if ex.empty: return float("nan"), float("nan"), float("nan")
        alpha = AF * ex.mean()
        te    = (AF ** 0.5) * ex.std(ddof=1)
        ir    = alpha / te if np.isfinite(te) and te != 0 else float("nan")
        return float(alpha), float(te), float(ir)
    
    # add bench-relative stats
    rows_mvo = dict(summ_mvo)
    rows_dro = dict(summ_dro)
    rows_reg = dict(summ_reg)
    rows_spx = dict(summ_spx)
    
    # --- FIX: average gross exposure over the OOS slice only (not the pre-start lookback) ---
    # Map the OOS [start,end] back to the fit index coordinates to define a half-open window [a_oos, b_oos)
    a_oos = full_index_fit.get_loc(full_index[0])
    b_oos = full_index_fit.get_loc(full_index[-1]) + 1
    win_oos = (a_oos, b_oos)
    
    T_req = len(full_index)
    rows_mvo["gross_exp"] = _gross_exp_on_window(fit_mvo,    T_req, win=win_oos)
    rows_dro["gross_exp"] = _gross_exp_on_window(fit_dro_pw, T_req, win=win_oos)
    rows_reg["gross_exp"] = _gross_exp_on_window(fit_reg,    T_req)   # RegDRO already aligned to OOS

    # Hit rate (portfolio vs SPX): mean & std, CI
    mvo_hr_mean, mvo_hr_se, _, mvo_ci_lo, mvo_ci_hi = hit_rate_vs_bench_stats(mvo_daily,  spx_daily, full_index)
    dro_hr_mean, dro_hr_se, _, dro_ci_lo, dro_ci_hi = hit_rate_vs_bench_stats(dro_daily,  spx_daily, full_index)
    reg_hr_mean, reg_hr_se, _, reg_ci_lo, reg_ci_hi = hit_rate_vs_bench_stats(regdro_daily, spx_daily, full_index)

    rows_mvo["hit_rate_vs_bench"]         = mvo_hr_mean
    rows_mvo["hit_rate_vs_bench_se"]      = mvo_hr_se
    rows_mvo["hit_rate_vs_bench_ci_low"]  = mvo_ci_lo
    rows_mvo["hit_rate_vs_bench_ci_high"] = mvo_ci_hi
    
    rows_dro["hit_rate_vs_bench"]         = dro_hr_mean
    rows_dro["hit_rate_vs_bench_se"]      = dro_hr_se
    rows_dro["hit_rate_vs_bench_ci_low"]  = dro_ci_lo
    rows_dro["hit_rate_vs_bench_ci_high"] = dro_ci_hi
    
    rows_reg["hit_rate_vs_bench"]         = reg_hr_mean
    rows_reg["hit_rate_vs_bench_se"]      = reg_hr_se
    rows_reg["hit_rate_vs_bench_ci_low"]  = reg_ci_lo
    rows_reg["hit_rate_vs_bench_ci_high"] = reg_ci_hi
    
    # Optional: SPX as its own model column in the OOS table
    # We already use SPX for alpha/TE/IR; this makes it visible as a column
    spx_on_win = spx_daily.reindex(full_index).fillna(0.0)
    rows_spx = _summ_from_series(spx_on_win, G, AF, len(full_index))
    rows_spx["gross_exp"] = float("nan")       # not applicable
    rows_spx["gap_oos_vs_train_realized"] = float("nan")
    rows_spx["alpha_ann_vs_spx"] = 0.0         # vs self
    rows_spx["te_ann_vs_spx"]    = 0.0
    rows_spx["ir_vs_spx"]        = float("nan")
    rows_spx["hit_rate"]         = float("nan")  # only defined for regime signals
    
    for rows, ser in ((rows_mvo, mvo_daily), (rows_dro, dro_daily), (rows_reg, regdro_daily)):
        a, te, ir = _bench_stats(ser, spx_daily, AF)
        rows["alpha_ann_vs_spx"] = a
        rows["te_ann_vs_spx"]    = te
        rows["ir_vs_spx"]        = ir
    # SPX vs SPX (leave as NaNs)
    rows_spx["alpha_ann_vs_spx"] = float("nan")
    rows_spx["te_ann_vs_spx"]    = float("nan")
    rows_spx["ir_vs_spx"]        = float("nan")

    # δ aggregates for both DRO (piecewise) and RegDRO
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

    # assemble DataFrames for oos_summary/printing
    df_mvo = pd.DataFrame([rows_mvo])
    df_dro = pd.DataFrame([rows_dro])
    df_reg = pd.DataFrame([rows_reg])
    df_spx = pd.DataFrame([rows_spx])

    # print OOS (include SPX)
    results_dict = {"MVO": df_mvo, "DRO": df_dro, "RegDRO": df_reg, "SPX": df_spx}
    print_oos_table(results_dict, model_order=["MVO", "DRO", "RegDRO", "SPX"])

    # detailed RegDRO block
    _section("RegDRO — detailed piecewise report")
    report_regdro(
        fit=fit_reg,
        data={"train": df_returns[names_all].to_numpy(float),
              "test":  df_returns[names_all].to_numpy(float),
              "n_days": len(full_index),
              "ann_factor": AF,
              "taus_true": list(taus),
              "px_cols": names_all},
        G=G,
        taus_true=None,
        label="RegDRO"
    )

    # outputs (no PartA/PartB)
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
        "signals_oos": signals_ffill,
        "rSLDS_mask": rSLDS_mask_df,
        "rSLDS_trials": trials_df,  # denominator mask (requested in item 6)
        "rSLDS_hit_rate_by_asset": rSLDS_hit_rate_by_asset.to_dict(),
        "rSLDS_hit_rate_overall": float(hr_mean),
        "rSLDS_hit_rate_std": float(hr_std),
        "rSLDS_hit_rate_ci_low": float(hr_ci_low),
        "rSLDS_hit_rate_ci_high": float(hr_ci_high),
        "rSLDS_hit_rate_n": int(hr_n),
        "rSLDS_hit_rate_macro": macro_hr,

        "holdings": {"MVO": H_mvo, "DRO": H_dro, "RegDRO": H_reg},
    },}

    if "dro_pickle" in CONFIG and CONFIG["dro_pickle"]:
        save_out(out, CONFIG["dro_pickle"])

    return out
