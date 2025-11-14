# dro.py

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
import numpy as np
import pandas as pd
import cvxpy as cp
import os
import gzip, pickle
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

# -------------------------
# Array adapters (GPU/CPU boundaries)
# -------------------------

def _is_cupy_array(a):
    try:
        import cupy as _cp
        return isinstance(a, _cp.ndarray) or hasattr(a, "__cuda_array_interface__")
    except Exception:
        return hasattr(a, "__cuda_array_interface__")

def asnumpy_strict(a, dtype=None, order=None):
    """
    Return a *NumPy* ndarray (never CuPy) from possibly-CuPy input.
    Optionally cast dtype and memory order.
    """
    import numpy as _np
    if _is_cupy_array(a):
        import cupy as _cp
        out = _cp.asnumpy(a)
    else:
        out = _np.asarray(a)
    if dtype is not None:
        out = out.astype(dtype, copy=False)
    if order in ("C", "F"):
        out = _np.array(out, dtype=out.dtype, order=order, copy=False)
    return out

def asxp(a, dtype=None):
    """
    Convert to xp array (CuPy if GPU, else NumPy). If already xp, return as-is.
    """
    if _is_cupy_array(a):
        return a.astype(dtype, copy=False) if dtype is not None else a
    return xp.asarray(a, dtype=dtype)

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

def sliced_w2_empirical(X, Y, n_proj=256, rng=None, U=None):
    """
    1D sliced W2 between empirical measures using random projections.
    If U is provided (shape = [n_proj, d], row-unit vectors), it is used and
    n_proj/rng are ignored. This lets us reuse the same directions across
    bootstrap, within, and between computations to avoid duplication/drift.
    """
    X = xp.asarray(X, dtype=xp.float32)
    Y = xp.asarray(Y, dtype=xp.float32)
    n, d = X.shape
    m = Y.shape[0]

    if U is None:
        # Generate directions on device when possible
        if (rng is None) and hasattr(xp.random, "standard_normal"):
            try:
                U = xp.random.standard_normal((int(n_proj), d), dtype=X.dtype)
            except TypeError:
                U = xp.random.standard_normal((int(n_proj), d)).astype(X.dtype, copy=False)
        else:
            rng = _rng_from_params({}) if rng is None else rng
            U = xp.asarray(rng.normal(size=(int(n_proj), d)), dtype=X.dtype)
        U = U / xp.maximum(xp.linalg.norm(U, axis=1, keepdims=True), 1e-12)
    else:
        U = xp.asarray(U, dtype=X.dtype)
        # assume caller already normalized U

    XU = X @ U.T
    YU = Y @ U.T

    if m == n:
        XU = xp.sort(XU, axis=0)
        YU = xp.sort(YU, axis=0)
        diff = XU - YU
        w2_sq = xp.mean(diff * diff)
        return float(xp.sqrt(xp.maximum(w2_sq, 0.0)))
    else:
        k = int(min(n, m))
        if k <= 1:
            XU = xp.mean(XU, axis=0, keepdims=True)
            YU = xp.mean(YU, axis=0, keepdims=True)
        else:
            q = (xp.arange(1, k + 1, dtype=XU.dtype) - 0.5) / k
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

def bootstrap_np_block_delta(
    R, n_proj=512, B=100, block_len=55, alpha=0.05, seed=None, n_sample=512,
    standardize=False, U=None):
    """
    Moving-block bootstrap of the empirical daily panel.
    If `standardize` is True, apply column-wise z-scoring once to the pool
    to prevent scale dominance in sliced-W2. Per replicate: draw TWO independent
    block-resamples of fixed length n, compute sliced-W2 → daily distance.
    Return the (1-alpha) quantile (daily).
    """
    # pool
    R_xp = xp.asarray(R, dtype=xp.float32)
    T = int(R_xp.shape[0])
    n = int(n_sample)

    # scale factor to bring standardized distances back to original units
    scale = 1.0

    if standardize:
        # NaN-safe per-asset z-scoring of the pool (leaves all-NaN cols as zeros)
        finite = xp.isfinite(R_xp)
        n_j = xp.maximum(finite.sum(axis=0, dtype=xp.float32), 1.0)
        sum_j = xp.where(finite, R_xp, 0.0).sum(axis=0)
        mu_j  = sum_j / n_j
        Xc    = xp.where(finite, R_xp - mu_j[None, :], 0.0)
        ss    = (Xc * Xc).sum(axis=0)
        std_j = xp.sqrt(xp.maximum(ss / xp.maximum(n_j - 1.0, 1.0), 0.0))
        std_j = xp.where(std_j < 1e-12, 1.0, std_j)

        # RMS daily std across assets ≈ sqrt(tr(Sigma)/d)
        scale = float(xp.sqrt(xp.mean(std_j * std_j)))

        pool = xp.where(finite, Xc / std_j[None, :], 0.0)
    else:
        pool = R_xp

    rng = np.random.default_rng(seed)
    dists = xp.empty(int(B), dtype=float)
    for b in range(int(B)):
        i1 = _mbb_indices(T, n, int(block_len), rng=rng)
        i2 = _mbb_indices(T, n, int(block_len), rng=rng)
        X1 = pool[xp.asarray(i1, dtype=xp.int64)]
        X2 = pool[xp.asarray(i2, dtype=xp.int64)]
        dists[b] = sliced_w2_empirical(X1, X2, n_proj=int(n_proj), rng=None, U=U)

    # If standardized, dists are in “z-score” units; rescale by typical daily σ
    return float(scale * xp.quantile(dists, 1.0 - float(alpha)))


def bootstrap_gaussian_block_delta(
    R, alpha=0.05, B=100, block_len=55, eps=1e-9, seed=None, n_sample=512,
    standardize=False):
    """
    Moving-block bootstrap; distance is Gelbrich W2 between the Gaussian fitted to
    the reference pool (mu0,S0) and the Gaussian fitted to each block-resample.
    Uses a fixed resample length `n_sample` via circular MBB (can wrap when n_sample > T).
    If `standardize` is True, first z-score columns ONCE on the pool (NaN-safe), then
    compute moments on the standardized data for both reference and resamples.
    Returns the (1-alpha) quantile of *daily* W2 radii.
    """
    X = xp.asarray(R, dtype=float)
    T, d = int(X.shape[0]), int(X.shape[1])
    if T < 2:
        return 0.0

    # Scale factor to move standardized distances back to original units
    scale = 1.0

    # Optional: NaN-safe per-asset z-scoring on the pool (once)
    if bool(standardize):
        finite = xp.isfinite(X)
        n_j = xp.maximum(finite.sum(axis=0, dtype=xp.float32), 1.0)
        sum_j = xp.where(finite, X, 0.0).sum(axis=0)
        mu_j  = sum_j / n_j
        Xc    = xp.where(finite, X - mu_j[None, :], 0.0)
        ss    = (Xc * Xc).sum(axis=0)
        std_j = xp.sqrt(xp.maximum(ss / xp.maximum(n_j - 1.0, 1.0), 0.0))
        std_j = xp.where(std_j < 1e-12, 1.0, std_j)

        # RMS daily std across assets
        scale = float(xp.sqrt(xp.mean(std_j * std_j)))

        X = xp.where(finite, Xc / std_j[None, :], 0.0)

    # Reference moments on the (possibly standardized) pool
    mu0 = xp.mean(X, axis=0)
    Xc  = X - mu0
    S0  = (Xc.T @ Xc) / max(T - 1, 1)

    # Fixed resample length n via circular MBB (no cap at T)
    n = int(max(2, n_sample))
    rng = np.random.default_rng(seed)

    deltas = xp.empty(int(B), dtype=float)
    for b in range(int(B)):
        idx = _mbb_indices(T, n, int(block_len), rng=rng)
        Xb  = X[xp.asarray(idx, dtype=xp.int64)]
        mub = xp.mean(Xb, axis=0)
        Xbc = Xb - mub
        Sb  = (Xbc.T @ Xbc) / max(n - 1, 1)
        deltas[b] = wasserstein2_gaussian(mu0, S0, mub, Sb, float(eps))

    # If standardized, Gaussian W2 is in z-units; rescale by typical daily σ
    return float(scale * xp.quantile(deltas, 1.0 - float(alpha)))


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
        if Sigma is None:
            raise ValueError("kappa_rate requires Sigma.")
        d     = int(xp.size(mu_est))
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n_eff = int((params or {}).get("n_ref", n_obs))
        sbar  = float(xp.sqrt(xp.trace(Sigma) / max(d, 1)))
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
        alpha       = float((params or {}).get("alpha", 0.05))
        B           = int((params or {}).get("B", 256))
        n_proj      = int((params or {}).get("n_proj", 128))
        seed        = (params or {}).get("seed", None)
        L           = int((params or {}).get("block_len", 10))
        n_sample    = int((params or {}).get("n_sample", 252))
        standardize = bool((params or {}).get("standardize", True))
        delta_daily = bootstrap_np_block_delta(
            R, n_proj=n_proj, B=B, block_len=L, alpha=alpha, seed=seed,
            n_sample=n_sample, standardize=standardize, U=params.get("U", None))
        return float(np.sqrt(AF)) * float(delta_daily)
    
    if method == "bootstrap_gaussian":
        assert R is not None, "bootstrap_gaussian needs raw sample matrix R."
        alpha       = float((params or {}).get("alpha", 0.05))
        B           = int((params or {}).get("B", 512))
        eps         = float((params or {}).get("epsilon_sigma", 1e-9))
        seed        = (params or {}).get("seed", None)
        L           = int((params or {}).get("block_len", 10))
        n_sample    = int((params or {}).get("n_sample", 252))
        standardize = bool((params or {}).get("standardize", True))
        delta_daily = bootstrap_gaussian_block_delta(
            R, alpha=alpha, B=B, block_len=L, eps=eps, seed=seed,
            n_sample=n_sample, standardize=standardize,)    
        return float(np.sqrt(AF)) * float(delta_daily)

    raise ValueError(f"Unknown delta_method='{method}'")

def _to_xp(A):
    """
    Ensure A is an xp (CuPy/NumPy via 'xp') array for downstream xp math.
    If A is already an xp array, it is returned as-is.
    """
    mod = getattr(A, "__module__", "")
    # If A is a NumPy ndarray (or anything not xp), convert to xp
    if mod.startswith("numpy"):
        return xp.asarray(A)
    return A

def psd_factor_LtL(Sigma, eps):
    """
    Return L (NumPy float64, contiguous) such that Sigma ≈ L.T @ L.
    Absolutely no implicit CuPy→NumPy conversions.
    """
    import numpy as _np

    # --- FORCE HOST ARRAY (defensive against any GPU-backed object) ---
    try:
        import cupy as _cp
        if isinstance(Sigma, _cp.ndarray) or hasattr(Sigma, "__cuda_array_interface__"):
            S = _cp.asnumpy(Sigma)                           # explicit device→host
        elif hasattr(Sigma, "get") and callable(Sigma.get):  # generic CuPy-like
            S = Sigma.get()
        else:
            S = _np.asarray(Sigma)
    except Exception:
        # If CuPy isn't available (or any import fails), just coerce to NumPy
        S = _np.asarray(Sigma)

    # Now ensure dtype/layout without re-entering CuPy
    S = _np.array(S, dtype=_np.float64, order="C", copy=False)

    # --- Symmetrize & regularize; strictly NumPy below this line ---
    S_sym = 0.5 * (S + S.T)
    try:
        C = _np.linalg.cholesky(S_sym + float(eps) * _np.eye(S_sym.shape[0], dtype=S_sym.dtype))
    except _np.linalg.LinAlgError:
        vals, vecs = _np.linalg.eigh(S_sym)
        vals = _np.clip(vals, float(eps), None)
        S_psd = (vecs * vals) @ vecs.T
        C = _np.linalg.cholesky(S_psd)

    # L so that L^T L ≈ Σ
    return _np.ascontiguousarray(C.T, dtype=_np.float64)

def solve_optimizer(mu, Sigma, delta, config, verbose=False):
    
    import numpy as _np
    
    n   = int(len(mu))
    rho = float(config["risk_budget"])
    eps = float(config["epsilon_sigma"])

    # ---- Strict NumPy boundary for CVXPY ----
    Sigma_np = asnumpy_strict(Sigma, dtype=_np.float64, order="C")
    _np.nan_to_num(Sigma_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Absolute guard (even if asnumpy_strict ever regresses)
    try:
        import cupy as _cp
        if isinstance(Sigma_np, _cp.ndarray) or hasattr(Sigma_np, "__cuda_array_interface__"):
            Sigma_np = _cp.asnumpy(Sigma_np)
    except Exception:
        pass
    
    Sigma_np = _np.array(Sigma_np, dtype=_np.float64, order="C", copy=False)
    
    # Sanity while debugging (safe to leave on)
    assert not hasattr(Sigma_np, "__cuda_array_interface__"), "Sigma_np is still CUDA-backed!"
    assert isinstance(Sigma_np, _np.ndarray), f"Sigma_np type={type(Sigma_np)}"
    
    L = psd_factor_LtL(Sigma_np, eps)  # guaranteed pure NumPy
    mu_np = asnumpy_strict(mu, dtype=_np.float64)     # NumPy float64

    # ---- CVXPY variables / objective ----
    w = cp.Variable(n)
    t = cp.Variable(nonneg=True)
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

    return xp.asarray(_np.asarray(w.value).reshape(-1))

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
    """
    Safe matmul version: keep math on a single backend per op.
    - Portfolio series computed on xp (GPU if available).
    - Any NumPy-only ops (e.g., psd_factor_LtL) are cast back to xp for norms.
    """
    # Coerce inputs
    Rtr = asxp(returns_train, dtype=float)   # (T,d) xp
    Rev = asxp(returns_eval,  dtype=float)   # (T,d) xp
    wxp = asxp(w, dtype=float).reshape(-1)   # (d,) xp

    n_days, n_assets = Rtr.shape
    AF = int(config.get("annualization_factor", 252))

    # Asset-level stats on xp
    mu_train_ann_assets    = AF * xp.mean(Rtr, axis=0)
    sigma_train_ann_assets = xp.sqrt(AF) * xp.std(Rtr, axis=0, ddof=1)

    # Exact SOC risk with NumPy Cholesky, then xp-norm
    L_np = psd_factor_LtL(Sigma_ann, config["epsilon_sigma"])  # NumPy
    L_xp = asxp(L_np, dtype=float)                             # xp
    risk_train_ann = float(xp.linalg.norm(L_xp @ wxp))
    tol = max(atol, rtol * max(rho, risk_train_ann))
    ok_train = bool(risk_train_ann <= rho + tol)

    # Train/Eval returns (annualized means) on xp
    ret_train_ann = float(mu_train_ann_assets @ wxp)
    port_eval = Rev @ wxp                                   # (T,)
    _, risk_eval_ann, _ = stats_from_series(port_eval, dict(config, annualization_factor=AF))
    mu_eval_ann_assets = AF * xp.mean(Rev, axis=0)
    ret_eval_ann = float(mu_eval_ann_assets @ wxp)

    # Some quick exposure summaries (unchanged semantics)
    gross_exposure = float(xp.sum(xp.abs(wxp)))
    top_idx = xp.argsort(wxp)[-3:][::-1]
    nz = xp.where(wxp != 0)[0]
    bot_idx = nz[xp.argsort(wxp[nz])[:3]] if nz.size else xp.array([], dtype=int)

    return {
        "ret_train_ann": ret_train_ann,
        "risk_train_ann": risk_train_ann,
        "ok_train": ok_train,
        "ret_eval_ann": ret_eval_ann,
        "risk_eval_ann": risk_eval_ann,
        "gross_exposure": gross_exposure,
        "top_idx": asnumpy_strict(top_idx).tolist(),
        "bot_idx": asnumpy_strict(bot_idx).tolist(),
    }

def print_regime_block(label, returns_train, returns_eval, w_list, segs, rho,
                       taus_display, seg_deltas, config=None):
    """
    Safe accumulation of piecewise portfolio series on xp backend.
    """
    # Coerce to xp once
    Rtr = asxp(returns_train, dtype=float)   # (T,d)
    Rev = asxp(returns_eval,  dtype=float)   # (T,d)
    n_days, n_assets = Rtr.shape
    AF = int((config or {}).get("annualization_factor", 252))

    # Concatenate per-segment portfolio series on xp
    port_train = xp.zeros(n_days, dtype=float)
    port_eval  = xp.zeros(n_days, dtype=float)
    for k, w in enumerate(w_list):
        a, b = segs[k], segs[k+1]
        wxp = asxp(w, dtype=float).reshape(-1)
        port_train[a:b] = (Rtr[a:b] @ wxp)
        port_eval[a:b]  = (Rev[a:b] @ wxp)

    # Annualized stats (helper coerces to xp internally)
    cfg = {
        "n_days": n_days,
        "risk_free_rate": float((config or {}).get("risk_free_rate", 0.0)),
        "annualization_factor": AF,
    }
    ret_train_ann, risk_train_ann, _ = stats_from_series(port_train, cfg)
    ret_eval_ann,  risk_eval_ann,  _ = stats_from_series(port_eval,  cfg)

    # Asset-level sample stats (arith. daily → annualized)
    mu_train_ann_assets    = AF * xp.mean(Rtr, axis=0)
    sigma_train_ann_assets = xp.sqrt(AF) * xp.std(Rtr, axis=0, ddof=1)

    return {
        "ret_train_ann": float(ret_train_ann),
        "risk_train_ann": float(risk_train_ann),
        "ret_eval_ann": float(ret_eval_ann),
        "risk_eval_ann": float(risk_eval_ann),
        "mu_train_ann_assets": asnumpy_strict(mu_train_ann_assets).tolist(),
        "sigma_train_ann_assets": asnumpy_strict(sigma_train_ann_assets).tolist(),
        "taus_display": list(taus_display),
        "seg_deltas": list(seg_deltas),
    }

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

        # unconditional Σ from rolling window [ws, t_for_sigma] on SIMPLE returns, shrunk
        min_obs = int(params.get("min_lookback_days", 21))
        max_lb  = int(params.get("max_lookback_days", 1260))
        lam_shr = float(params.get("sigma_shrinkage_lambda", 0.0))
        
        import numpy as _np
        R_df_full = pd.DataFrame(_np.asarray(data["train"], dtype=float),
                                 columns=list(data.get("px_cols", range(data["train"].shape[1]))))
        
        t_for_sigma = max(0, min(int(b) - 1, int(data["n_days"]) - 1))
        ws = _window_start(t_for_sigma + 1, min_obs, max_lb)  # end-exclusive; +1 to include t_for_sigma
        R_win_df = R_df_full.iloc[ws : t_for_sigma + 1]
        
        try:
            Sigma_est = xp.asarray(
                compute_cov_from_window(R_win_df, ann=AF, shrink_lambda=lam_shr, min_obs=min_obs),
                dtype=float
            )
        except Exception:
            Sigma_est = xp.asarray(data["Sigma_ann_full"], float)

        
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
            names = list(data.get("px_cols", range(len(mu_est))))
            _print_mu_by_name(names, mu_est)
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
            L = psd_factor_LtL(data["Sigma_ann_full"], G["epsilon_sigma"])  # NumPy for CVXPY
            L_xp = _to_xp(L)                                                # cast to xp for xp math
            train_soc = float(xp.linalg.norm(L_xp @ fit["w"]))
     
        # enrich & rename “gap”
        stats_oos["gross_exp"] = ge
        stats_oos["sigma_train_ann"] = float(sigma_train_ann)
        stats_oos["sigma_oos_ann"] = float(stats_oos["sigma_ann"])
        stats_oos["train_soc_risk"] = train_soc
        stats_oos["train_constraint_slack"] = float(G["risk_budget"] - train_soc) if xp.isfinite(train_soc) else xp.nan                
        stats_oos["kappa"] = float(fit.get("kappa", xp.nan))
        # new delta schema for static fits: only 'delta' populated
        stats_oos["delta"] = float(fit.get("delta", xp.nan))
        stats_oos["delta_uncond"] = xp.nan
        stats_oos["delta_gap"] = xp.nan
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

        # New delta schema for piecewise fits:
        # 'delta' = average of optimisation deltas; no unconditional counterfactual here.
        dlist = xp.asarray(fit.get("delta_list", []), dtype=float)
        if dlist.size:
            mask = xp.isfinite(dlist)
            stats_oos["delta"] = float(xp.mean(dlist[mask])) if int(mask.sum()) else xp.nan
        else:
            stats_oos["delta"] = xp.nan
        stats_oos["delta_uncond"] = xp.nan
        stats_oos["delta_gap"] = xp.nan
                
        rebal = list(fit.get("segs", []))
        if not rebal:
            # fallback if segs missing: treat as static
            rebal = [0, n_days]
        stats_oos["avg_holding_per"] = _avg_holding_period_from_marks(rebal)
        return stats_oos

def evaluate_regime_independently(fit, data, G):
    """
    Performs an independent evaluation for each segment of a piecewise portfolio.
    Uses NumPy for segment matmul to avoid CuPy↔pandas boundary issues,
    then stats_from_series handles xp coercion internally.
    """
    n_days = int(data["n_days"])
    test = data["test"]

    # Force test panel to NumPy for safe matmul
    R_test = np.asarray(test, dtype=float)

    stats_oos = {}
    dlist = list(map(float, fit.get("delta_list", [])))
    for j, dj in enumerate(dlist, start=1):
        stats_oos[f"delta_k{j}"] = dj

    for k, (a, b) in enumerate(zip(fit["segs"][:-1], fit["segs"][1:])):
        wk = fit["w_list"][k]
        seg_length = int(b - a)

        mu_seg = sigma_seg = sharpe_seg = vol_breach_seg = np.nan
        gross_exp_seg = float(np.sum(np.abs(asnumpy_strict(wk, dtype=float))))

        if seg_length > 1:
            # Segment series in NumPy
            wk_np = asnumpy_strict(wk, dtype=float).reshape(-1)
            seg_series_oos = R_test[a:b] @ wk_np

            seg_config = dict(G)
            seg_config["n_days"] = n_days
            seg_config["annualization_factor"] = int(data.get("ann_factor", 252))

            mu_seg, sigma_seg, sharpe_seg = stats_from_series(seg_series_oos, seg_config)
            vol_breach_seg = max(sigma_seg - G["risk_budget"], 0.0)

        stats_oos[f"mu_ann_k{k+1}"] = mu_seg
        stats_oos[f"sigma_ann_k{k+1}"] = sigma_seg
        stats_oos[f"sharpe_ann_k{k+1}"] = sharpe_seg
        stats_oos[f"vol_breach_k{k+1}"] = vol_breach_seg
        stats_oos[f"gross_exp_k{k+1}"] = gross_exp_seg

    return stats_oos
    
def stats_from_series(port_daily, config):
    AF = int(config.get("annualization_factor", 252))
    rf_annual = float(config.get("risk_free_rate", 0.0))
    # Force to xp on entry to avoid NumPy/CuPy function mismatch
    x = asxp(port_daily, dtype=float).ravel()
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rf_daily = (1.0 + rf_annual) ** (1.0 / AF) - 1.0
    sigma_daily = xp.std(x, ddof=1)
    sigma_annual = sigma_daily * xp.sqrt(AF)
    mu_annual_geom = xp.exp(AF * xp.mean(xp.log1p(x))) - 1.0
    sharpe_annual = (xp.mean(x) - rf_daily) / sigma_daily * xp.sqrt(AF) if sigma_daily > 0 else xp.nan
    return float(mu_annual_geom), float(sigma_annual), float(sharpe_annual)

def _max_drawdown_from_series(port_daily):
    """
    Max drawdown of a daily-return series.
    Returns the minimum (most negative) drawdown, e.g. -0.27 for -27%.
    Uses NumPy fallback for prefix-maximum on GPU (CuPy lacks maximum.accumulate).
    """
    x = xp.asarray(port_daily, float)
    if x.size == 0:
        return float("nan")

    # If we're on GPU, do the cumprod + running-max on NumPy, then compute dd there.
    try:
        import cupy as _cp
        is_gpu = isinstance(x, _cp.ndarray) or hasattr(x, "__cuda_array_interface__")
    except Exception:
        is_gpu = False

    if is_gpu:
        x_np = asnumpy_strict(x, dtype=float)
        equity_np = np.cumprod(1.0 + x_np)
        peak_np = np.maximum.accumulate(equity_np)
        dd_np = equity_np / peak_np - 1.0
        return float(np.min(dd_np))
    else:
        equity = np.cumprod(1.0 + np.asarray(x, dtype=float))
        peak = np.maximum.accumulate(equity)
        dd = equity / peak - 1.0
        return float(np.min(dd))

def portfolio_stats(weights, returns, config):
    """
    Static weights over full horizon.
    Multiplication done in NumPy to avoid CuPy<->pandas issues.
    """
    R = np.asarray(returns, dtype=float)                  # (T,d) NumPy
    w = asnumpy_strict(weights, dtype=float).reshape(-1)  # (d,) NumPy
    port_daily = R @ w                                    # (T,) NumPy
    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    max_dd = _max_drawdown_from_series(port_daily)
    return {
        "mu_ann": mu_annual_geom,
        "sigma_ann": sigma_annual,
        "sharpe_ann": sharpe_annual,
        "vol_breach": vol_breach,
        "max_dd": max_dd,
    }
    
def portfolio_stats_multipiece(w_list, taus, returns, config):
    import numpy as _np
    taus = [int(x) for x in list(taus)]
    R = _np.asarray(returns, dtype=float)   # NumPy
    n_days = int(config["n_days"])
    assert taus[0] == 0 and taus[-1] == n_days and len(w_list) == len(taus) - 1

    # Build in NumPy to avoid device mismatch, convert later for stats
    port_daily_np = _np.empty(n_days, dtype=float)
    for k in range(len(w_list)):
        a, b = taus[k], taus[k + 1]
        w_np = asnumpy_strict(w_list[k], dtype=float).reshape(-1)
        port_daily_np[a:b] = R[a:b] @ w_np

    # stats_from_series will coerce to xp internally
    mu_annual_geom, sigma_annual, sharpe_annual = stats_from_series(port_daily_np, config)
    vol_breach = max(sigma_annual - config["risk_budget"], 0.0)
    max_dd = _max_drawdown_from_series(port_daily_np)
    return {
        "mu_ann": mu_annual_geom,
        "sigma_ann": sigma_annual,
        "sharpe_ann": sharpe_annual,
        "vol_breach": vol_breach,
        "max_dd": max_dd,
    }

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
# Jackknife universe resampling
# ---------------------------------------------------------------

def _aggregate_ci(samples: dict[str, list[float]], alpha: float) -> dict[str, dict]:
    out = {}
    lo_q, hi_q = alpha/2.0, 1.0 - alpha/2.0
    for k, vals in samples.items():
        arr = np.asarray(vals, float)
        good = arr[np.isfinite(arr)]
        if good.size == 0:
            out[k] = {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        else:
            out[k] = {
                "mean": float(np.mean(good)),
                "ci_low": float(np.quantile(good, lo_q)),
                "ci_high": float(np.quantile(good, hi_q)),
            }
    return out

def _aggregate_ci_xp(samples: dict[str, list[float]], alpha: float) -> dict[str, dict]:
    out = {}
    lo_q, hi_q = alpha/2.0, 1.0 - alpha/2.0
    for k, vals in samples.items():
        arr = xp.asarray(vals, dtype=float)
        mask = xp.isfinite(arr)
        if int(mask.sum()) == 0:
            out[k] = {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
        else:
            good = arr[mask]
            out[k] = {
                "mean": float(xp.mean(good)),
                "ci_low": float(xp.quantile(good, lo_q)),
                "ci_high": float(xp.quantile(good, hi_q)),
            }
    return out

def _aggregate_mean_std(samples: dict[str, list[float]]) -> dict[str, dict]:
    out = {}
    for k, vals in samples.items():
        arr = np.asarray(vals, float)
        good = arr[np.isfinite(arr)]
        if good.size == 0:
            out[k] = {"mean": np.nan, "std": np.nan}
        else:
            out[k] = {
                "mean": float(np.mean(good)),
                "std":  float(np.std(good, ddof=1)) if good.size > 1 else np.nan,
            }
    return out
    
def jackknife_universe_oos(
    *,
    securities: list[str],
    CONFIG: dict,
    d: int,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Leave-d-out jackknife over the security universe.

    CONFIG["JACKKNIFE"]: {"d": <block size>, "seed": <int>}
    - n = len(securities)
    - require n % d == 0
    - permute securities once
    - for each block b, drop that block, run dro_pipeline on remaining names
    - aggregate metrics across jackknife samples (quantiles + mean/std)

    Note: uses current dro_pipeline model names:
          "MVO_fixed", "DRO_fixed", "RegDRO".
    """
    artifacts = _load_artifacts_cached(CONFIG)

    names = [str(x) for x in securities]
    n = int(len(names))
    d = int(d)

    if d <= 0:
        raise ValueError("JACKKNIFE['d'] must be positive.")
    if n == 0:
        raise ValueError("No securities provided for jackknife.")
    if n % d != 0:
        raise ValueError(f"Jackknife requires n % d == 0; got n={n}, d={d}.")

    n_blocks = n // d
    rng = np.random.default_rng(seed)

    # Single permutation used to define disjoint leave-d-out blocks
    perm_idx = rng.permutation(n)

    # Mark context so dro_pipeline suppresses repeated tables
    CONFIG["__bootstrap_run"] = {"active": True, "B": int(n_blocks), "i": 0}

    METRICS = [
        "mu_ann","sigma_ann","sharpe_ann","vol_breach","max_dd",
        "alpha_ann","te_ann","ir_ann","hit_rate","gross_exp",
        "delta","delta_uncond","delta_gap",
    ]

    # Use current dro_pipeline keys
    STRATS = ("MVO_fixed", "DRO_fixed", "RegDRO")
    buckets = {s: {k: [] for k in METRICS} for s in STRATS}

    for b in range(n_blocks):
        start = b * d
        end = start + d

        left_out_idx = perm_idx[start:end]
        left_out = {names[int(i)] for i in left_out_idx}
        jackknife_sec = [s for s in names if s not in left_out]

        print()
        print("=" * 72)
        print(f"[jackknife] block {b+1}/{n_blocks}, left_out={len(left_out)}, kept={len(jackknife_sec)}")
        print("=" * 72)
        print()

        CONFIG["__bootstrap_run"]["i"] = int(b)

        res = dro_pipeline(
            jackknife_sec,
            CONFIG,
            verbose=False,
            run_jackknife=False,   # prevent recursion
            artifacts=artifacts,
        )

        for strat in STRATS:
            if strat not in res:
                continue
            row = res[strat]["summary"]
            for k in METRICS:
                buckets[strat][k].append(float(row.get(k, np.nan)))

    # cleanup context
    CONFIG.pop("__bootstrap_run", None)

    # percentile-style intervals from jackknife replicates
    ci = {s: _aggregate_ci_xp(buckets[s], alpha) for s in buckets}
    # mean/std across jackknife samples
    ms = {s: _aggregate_mean_std(buckets[s]) for s in buckets}

    return {"ci": ci, "mean_std": ms}


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
        "delta",
        "delta_uncond",
        "delta_gap",
    ]

    ALLOW_CI = {
        "mu_ann",
        "sigma_ann",
        "sharpe_ann",
        "vol_breach",
        "delta",
        "delta_uncond",
        "delta_gap",
    }
    
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

# ---- cache artifacts once ----
def _load_artifacts_cached(CONFIG):
    res_csv  = CONFIG["results_csv"]
    seg_parq = CONFIG["segments_parquet"]
    if not os.path.exists(res_csv):  raise FileNotFoundError(res_csv)
    if not os.path.exists(seg_parq): raise FileNotFoundError(seg_parq)

    df_res = pd.read_csv(res_csv, usecols=range(10), engine="python")
    df_res["security"] = df_res["security"].astype(str).str.strip()
    df_seg = pd.read_parquet(seg_parq)
    df_seg["security"] = df_seg["security"].astype(str).str.strip()
    if df_seg["date"].dtype != "datetime64[ns]":
        df_seg["date"] = pd.to_datetime(df_seg["date"], errors="coerce")

    px_all, eps_all, pe_all, ser_vix = import_data(CONFIG["data_excel"])
    return dict(df_res=df_res, df_seg=df_seg, px_all=px_all, eps_all=eps_all, pe_all=pe_all, ser_vix=ser_vix)

def _num_series(s):
    return pd.to_numeric(s, errors="coerce").astype("float64")

def map_labels_to_calendar(z_ser: pd.Series, cal: pd.DatetimeIndex) -> np.ndarray:
    """
    Map (and forward-fill) regime labels to a daily trading calendar.
    Returns float64 array; NaN only before the first seen label.
    """
    z = pd.Series(z_ser).copy()
    z.index = pd.to_datetime(z.index, errors="coerce")
    z = z[~z.index.isna()].sort_index()

    cal = pd.DatetimeIndex(cal)

    # Reindex to full calendar and forward-fill *within* the calendar
    # so all in-range dates (incl. the end date) carry the latest known label.
    z_cal = z.reindex(cal).ffill()

    # Keep NaN before the first known label; no forward-fill from “nothing”.
    return z_cal.to_numpy(dtype="float64")

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

def _select_best_config(results_df, security):
    """
    For a given `security`, select BEST tuple among rows already limited to
    PERMISSIBLE tuples. Sort: score ↓, n_regimes ↑, dim_latent ↑.
    Returns (config, n_regimes, dim_latent) or None.
    """
    import numpy as np, pandas as pd

    if results_df is None or len(results_df) == 0:
        return None
    need = {"security","config","n_regimes","dim_latent"}
    if not need.issubset(results_df.columns):
        return None

    df = results_df.copy()
    df["security"]   = df["security"].astype(str).str.strip()
    df["config"]     = df["config"].astype(str).str.strip()
    df["n_regimes"]  = pd.to_numeric(df["n_regimes"],  errors="coerce").astype("Int64")
    df["dim_latent"] = pd.to_numeric(df["dim_latent"], errors="coerce").astype("Int64")
    df["score_num"]  = pd.to_numeric(df.get("score", np.nan), errors="coerce")

    df = df[df["security"] == str(security).strip()]
    if df.empty:
        return None

    df = df.sort_values(
        ["score_num","n_regimes","dim_latent"],
        ascending=[False, True, True], na_position="last"
    )
    r0 = df.iloc[0]
    if pd.isna(r0.get("score_num")):
        return None
    return (str(r0["config"]), int(r0["n_regimes"]), int(r0["dim_latent"]))

def _labels_from_segments_df(segments_df, security, config, n_regimes, dim_latent):
    df = segments_df.copy()
    df["security"] = df["security"].astype(str).str.strip()
    df["config"]   = df["config"].astype(str).str.strip()
    # strict tuple filter
    df = df[
        (df["security"] == str(security).strip()) &
        (df["config"]   == str(config).strip()) &
        (pd.to_numeric(df["n_regimes"],  errors="coerce").astype("Int64") == int(n_regimes)) &
        (pd.to_numeric(df["dim_latent"], errors="coerce").astype("Int64") == int(dim_latent))
    ]
    if df.empty:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["date", "z"]).drop_duplicates(subset="date", keep="last")
    return pd.Series(df["z"].astype(int).to_numpy(),
                     index=pd.DatetimeIndex(df["date"]), name="z")

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
    Always returns a 2D (N,N) array, including N=1.
    """
    # ---- to 2D numeric array ----
    if isinstance(R_win, pd.DataFrame):
        X = R_win.to_numpy(np.float64, copy=False)
    else:
        X = np.asarray(R_win, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"R_win must be 2D (T,d); got shape {X.shape}")

    # rows with all assets finite
    row_ok = np.isfinite(X).all(axis=1)
    Xc = X[row_ok, :]
    if Xc.shape[0] < min_obs:
        raise ValueError(
            f"Not enough observations for covariance: {Xc.shape[0]} < {min_obs}"
        )

    # ---- sample covariance ----
    Sig = np.cov(Xc.T, ddof=1)          # 0-D for d=1, (d,d) for d>1
    Sig = np.asarray(Sig, dtype=np.float64)

    # Force 2D
    if Sig.ndim == 0:
        # single asset: wrap variance
        Sig = Sig.reshape(1, 1)
    elif Sig.ndim != 2:
        raise ValueError(f"Unexpected covariance ndim={Sig.ndim}")

    # ---- annualize ----
    Sig_ann = Sig * float(ann)

    # ---- shrinkage ----
    lam = float(np.clip(shrink_lambda, 0.0, 1.0))
    if lam > 0.0:
        N = Sig_ann.shape[0]
        s2_bar = float(np.trace(Sig_ann) / max(N, 1))
        Sig_ann = (1.0 - lam) * Sig_ann + lam * s2_bar * np.eye(N, dtype=np.float64)

    # ---- sanity ----
    if not np.all(np.isfinite(Sig_ann)):
        raise ValueError("Non-finite covariance encountered.")
    if Sig_ann.shape[0] != Sig_ann.shape[1]:
        raise ValueError(f"Covariance must be square; got {Sig_ann.shape}")

    return Sig_ann

def regdro_decision_context(
    *,
    D_pos: int,
    full_index_fit: pd.DatetimeIndex,
    df_returns_full: pd.DataFrame,
    names_all: list[str],
    Z_labels_fit: dict[str, np.ndarray],
    AF: int,
    min_obs: int,
    max_lb: int,
    lam_shr: float,
    G: dict,
):
    """
    Build the optimisation context at a *single* decision date (position on FIT index).
    Returns:
        ok: bool
        keep: list[str]
        pos_map: dict[str,int]
        X_win_df: pd.DataFrame
        mu_cond: np.ndarray
        mu_uncond: np.ndarray
        Sig: np.ndarray
        X_win: np.ndarray                    # numpy window matrix for bootstrap δ
        mask_cond_all: np.ndarray[bool]      # row mask used to compute δ on conditional data
    """
    # Window on FIT calendar: [a_win, D_pos)
    a_win = max(0, int(D_pos) - int(max_lb))
    b_win = int(D_pos)
    if b_win - a_win < max(2, int(min_obs)):
        return False, [], {}, pd.DataFrame(), np.array([]), np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=bool)

    win_idx = full_index_fit.take(np.arange(a_win, b_win, dtype=int))

    keep, masks = [], []
    for n in names_all:
        z_ser = np.asarray(Z_labels_fit[n], dtype=float)
        z_now = z_ser[D_pos] if (0 <= D_pos < len(z_ser)) else np.nan
        if not np.isfinite(z_now):
            continue
        # same-regime mask inside the lookback window
        m = (z_ser[a_win:b_win] == z_now)
        if not np.any(m):
            continue
        x = df_returns_full.loc[win_idx, n].to_numpy(float)
        m = m & np.isfinite(x)
        if int(m.sum()) >= int(min_obs):
            keep.append(n)
            masks.append(m)
            
    if not keep:
        return False, [], {}, pd.DataFrame(), np.array([]), np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=bool)

    # Optional cap-feasibility
    cap_applies = bool(G.get("no_shorting", False)) and np.isfinite(G.get("max_pos_size", np.nan)) and (G.get("max_pos_size", 0.0) > 0.0)

    if cap_applies:
        c_max = float(G.get("max_cash", 0.0))
        u     = float(G.get("max_pos_size", 1.0))
        N_req = int(np.ceil((1.0 - c_max) / max(u, 1e-12)))
        if len(keep) < N_req:
            return False, [], {}, pd.DataFrame(), np.array([]), np.array([]), np.zeros((0, 0)), np.zeros((0, 0)), np.zeros(0, dtype=bool)

    # Assemble window DF and moments
    X_win_df = df_returns_full.loc[win_idx, keep]
    # conditional μ (per-asset, masked)
    mu_cond = np.asarray(
        [compute_mean_from_window(X_win_df[[n]], masks[j], min_obs=min_obs, ann=AF)[0]
         for j, n in enumerate(keep)],
        dtype=float
    )
    # unconditional Σ and μ on the same window and asset set
    Sig = compute_cov_from_window(X_win_df[keep], ann=AF, shrink_lambda=lam_shr, min_obs=min_obs)

    mask_all = np.ones(X_win_df.shape[0], dtype=bool)
    mu_uncond = compute_mean_from_window(X_win_df, mask_all, min_obs=min_obs, ann=AF)

    # intersection mask: rows that are “in-regime & finite” for all kept assets
    mask_cond_all = np.logical_and.reduce(masks) if len(masks) else np.zeros(X_win_df.shape[0], dtype=bool)

    X_win = X_win_df.to_numpy(float)
    pos_map = {n: i for i, n in enumerate(names_all)}
    return True, keep, pos_map, X_win_df, mu_cond, mu_uncond, Sig, X_win, mask_cond_all

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

def _gross_exp_on_window(fit, T_req, win=None):
    """Average gross exposure over a reporting window of length T_req (optionally [a,b) slice)."""
    import numpy as _np
    def _ge_vec(w):
        # Force any xp (CuPy/NumPy) vector to a pure NumPy array first
        w_np = asnumpy_strict(w, dtype=float).ravel()
        return float(_np.sum(_np.abs(w_np)))

    if fit["type"] == "static":
        return _ge_vec(fit["w"])

    segs = [int(x) for x in fit["segs"]]
    a0, b0 = (0, 10**12) if win is None else (int(win[0]), int(win[1]))
    num = 0.0
    for (a, b), w in zip(zip(segs[:-1], segs[1:]), fit["w_list"]):
        L = max(0, min(b, b0) - max(a, a0))
        if L > 0:
            num += L * _ge_vec(w)
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

def _available_cfg_tuples_from_parquet(seg_parq: str) -> set[tuple[str, int, int]]:
    """
    Read available (config, n_regimes, dim_latent) tuples from segments parquet.
    """
    seg = pd.read_parquet(seg_parq, columns=["config","n_regimes","dim_latent"])
    seg["config"]     = seg["config"].astype(str).str.strip()
    seg["n_regimes"]  = pd.to_numeric(seg["n_regimes"],  errors="raise").astype(int)
    seg["dim_latent"] = pd.to_numeric(seg["dim_latent"], errors="raise").astype(int)
    tups = set(seg[["config","n_regimes","dim_latent"]]
               .drop_duplicates()
               .itertuples(index=False, name=None))
    return tups

def _permissible_tuples_from_CONFIG(CONFIG) -> list[tuple[str,int,int]]:
    """
    Build all permissible (config, n_regimes, dim_latent) tuples from CONFIG["RSLDS"].
    Each entry must be a dict with those three keys.
    """
    lst = CONFIG.get("RSLDS", [])
    if not isinstance(lst, (list, tuple)) or len(lst) == 0:
        raise ValueError("CONFIG['RSLDS'] must be a non-empty list of dicts.")
    out = []
    for x in lst:
        if not isinstance(x, dict) or not all(k in x for k in ("config","n_regimes","dim_latent")):
            raise ValueError("Each RSLDS entry must be a dict with keys: config, n_regimes, dim_latent.")
        out.append((str(x["config"]).strip(), int(x["n_regimes"]), int(x["dim_latent"])))
    return out

def _tuples_in_results_csv(df_res, tuples: list[tuple[str,int,int]]) -> set[tuple[str,int,int]]:
    """
    Return the subset of tuples present in results_csv.
    """
    import pandas as pd, numpy as np
    df = df_res.copy()
    df["config"]     = df["config"].astype(str).str.strip()
    df["n_regimes"]  = pd.to_numeric(df.get("n_regimes", np.nan),  errors="coerce").astype("Int64")
    df["dim_latent"] = pd.to_numeric(df.get("dim_latent", np.nan), errors="coerce").astype("Int64")
    seen = set()
    for _, r in df.iterrows():
        c = str(r["config"])
        K = r["n_regimes"]; D = r["dim_latent"]
        if pd.notna(K) and pd.notna(D):
            seen.add((c, int(K), int(D)))
    return {t for t in tuples if t in seen}

# -------------------------
# Pipeline
# -------------------------

def dro_pipeline(securities, CONFIG, verbose=True, run_jackknife=False, artifacts=None,
                 models=["MVO_fixed", "DRO_fixed", "RegDRO"]):

    """
    Strict version:
      • ONLY make_index_rebal (MVO/DRO) and make_index_union (RegDRO)
      • Returns panel: px_all.loc[start:end].pct_change().fillna(0.0)
      • Portfolio series via matrix mult on that panel
      • _select_best_config/_expand_weights preserved elsewhere in file
      • δ aggregates included for DRO and RegDRO
      • SPX included in OOS summary table

    Parameters
    ----------
    securities : list[str] or None
        Universe of securities.
    CONFIG : dict
        Global configuration.
    models : list[str] or None
        Which optimisation models to expose in outputs and tables.
        Valid values:
            "MVO_fixed", "DRO_fixed", "MVO_event", "DRO_event", "RegDRO".
        If None, all are used.
    """

    # --- model selection -------------------------------------------------
    ALL_MODELS = ("MVO_fixed", "DRO_fixed", "MVO_event", "DRO_event", "RegDRO")
    if models is None:
        models = list(ALL_MODELS)
    else:
        models = [str(m) for m in models]
        invalid = [m for m in models if m not in ALL_MODELS]
        if invalid:
            raise ValueError(f"Unknown models: {invalid}. Valid models: {ALL_MODELS}")
    models_set = set(models)
    # ---------------------------------------------------------------------

    # detect bootstrap context (set by bootstrap_universe_oos)
    _boot = CONFIG.get("__bootstrap_run", {})
    IN_BOOT = bool(_boot.get("active", False))
    BOOT_I  = int(_boot.get("i", -1))
    BOOT_B  = int(_boot.get("B", -1))

    G = _make_solver_cfg_from_CONFIG(CONFIG)
    if artifacts is None:
        artifacts = _load_artifacts_cached(CONFIG)
    df_res = artifacts["df_res"]
    df_seg = artifacts["df_seg"]
    px_all = artifacts["px_all"]
    
    # --- normalize securities ---
    if securities is None:
        have_px  = set(map(str, px_all.columns))
        have_res = set(df_res["security"].astype(str).str.strip().unique())
        have_seg = set(df_seg["security"].astype(str).str.strip().unique())
        securities = sorted(have_px & have_res & have_seg)
    else:
        securities = [str(x).strip() for x in securities]
    
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
        spx_daily = pd.to_numeric(px_all["SPX"], errors="coerce").pct_change().fillna(0.0)
        spx_daily = spx_daily.reindex(full_index).fillna(0.0)
    else:
        spx_daily = pd.Series(index=full_index, dtype=float, name="SPX_daily")

    # ===== UNIFIED DECISION-DAY ENGINE (RegDRO + MVO/DRO event-based + MVO/DRO fixed-with-RegDRO-universe) =====    
    lam    = float(CONFIG["PORTFOLIO"]["sigma_shrinkage_lambda"])
    min_lb = int(CONFIG["REBAL"]["min_lookback_days"])
    max_lb = int(CONFIG["REBAL"]["max_lookback_days"])
    AF     = int(CONFIG.get("annualization_factor", 252))
    k_days = int(CONFIG["REBAL"]["rebalance_period_days"])
    if k_days <= 0: raise ValueError("rebalance_period_days must be > 0.")
    
    # 1) Rebalancing calendars
    index_rebal, marks = make_index_rebal(full_index_fit, s, e, k_days)   # fixed schedule on FIT index
    
    # 2) rSLDS labels (strict tuple selection) on both OOS and FIT calendars
    Z_labels     = {}
    Z_labels_fit = {}
    
    perm_tuples = _permissible_tuples_from_CONFIG(CONFIG)      # list of (config,K,D)
    present     = _tuples_in_results_csv(df_res, perm_tuples)
    missing     = sorted(set(perm_tuples) - present)
    if missing:
        raise RuntimeError(f"results_csv is missing REQUIRED rSLDS tuples: {missing}")
    
    df_res_perm = df_res.copy()
    df_res_perm["config"]     = df_res_perm["config"].astype(str).str.strip()
    df_res_perm["n_regimes"]  = pd.to_numeric(df_res_perm["n_regimes"],  errors="coerce").astype("Int64")
    df_res_perm["dim_latent"] = pd.to_numeric(df_res_perm["dim_latent"], errors="coerce").astype("Int64")
    mask_perm = pd.Series(False, index=df_res_perm.index)
    perm_set = set(perm_tuples)
    mask_perm |= df_res_perm.apply(
        lambda r: (str(r["config"]), int(r["n_regimes"]) if pd.notna(r["n_regimes"]) else -1,
                   int(r["dim_latent"]) if pd.notna(r["dim_latent"]) else -1) in perm_set,
        axis=1
    )
    df_res_perm = df_res_perm[mask_perm]
    
    px_cols_req = [t for t in px_cols if t in map(str, px_all.columns)]
    for sec in px_cols_req:
        best = _select_best_config(df_res_perm, sec)
        if best is None:
            continue
        c_name, K, D = best
        z_ser = _labels_from_segments_df(df_seg, sec, c_name, K, D)
        if z_ser is None:
            continue
        Z_labels[sec]     = map_labels_to_calendar(z_ser, full_index)
        Z_labels_fit[sec] = map_labels_to_calendar(z_ser, full_index_fit)
    
    names_all = [t for t in px_cols_req if t in Z_labels]
    if not names_all:
        raise RuntimeError("No assets produced rSLDS labels → cannot run RegDRO / aligned MVO/DRO.")
    
    # Event-based dates: union of regime changes on OOS calendar
    _, taus = make_index_union(full_index, {k: np.asarray(v, float) for k, v in Z_labels.items()}, s, e)
    taus = [int(x) for x in taus]
    
    params_reg = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])   # for RegDRO
    params_dro = dict(CONFIG["DELTA_DEFAULTS"][CONFIG["PORTFOLIO"]["delta_name"]])   # for DRO (unconditional)
    
    # 3) Solve on event dates (RegDRO; plus MVO/DRO event-based using the SAME context)
    w_reg_list, del_reg_list, del_uncond_list, del_gap_list = [], [], [], []
    w_evt_mvo_list, w_evt_dro_list = [], []

    # --- cache for bootstrap-based δ per decision date (ignores mu when method permits) ---
    delta_cache = {}
    def _delta_uncond_cached(mu_u, Sig, X_win, params):
        method = (params or {}).get("delta_method", "")
        if method in ("bootstrap_np", "bootstrap_gaussian"):
            # cache key ties to method, window size, Σ scale, and μ scale to avoid
            # reusing δ across materially different windows that happen to share shape/counts
            key = (method, Sig.shape,
                int(np.isfinite(X_win).sum()),
                float(np.trace(Sig)),
                float(np.linalg.norm(mu_u)))
            
            if key not in delta_cache:
                delta_cache[key] = compute_delta(params.get("kappa", 1.0), mu_u, Sig, R=X_win, params=params)
            return float(delta_cache[key])
        # fallback for κ-based or fixed methods
        return float(compute_delta(params.get("kappa", 1.0), mu_u, Sig, R=X_win, params=params))

    for a, b in zip(taus[:-1], taus[1:]):
        # decision date is the segment start 'a' on OOS calendar
        t_mid = min(max(a, 0), len(full_index) - 1)
        D     = full_index[t_mid]
        D_pos = full_index_fit.get_loc(D)  # position on FIT calendar

        ok, keep, pos_map, X_win_df, mu_cond, mu_uncond, Sig, X_win, mask_cond_all = regdro_decision_context(
            D_pos=D_pos,
            full_index_fit=full_index_fit,
            df_returns_full=df_returns_full,
            names_all=names_all,
            Z_labels_fit=Z_labels_fit,
            AF=AF,
            min_obs=min_lb,
            max_lb=max_lb,
            lam_shr=lam,
            G=G,)
        
        if not ok:
            w_reg_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            del_reg_list.append(np.nan); del_uncond_list.append(np.nan); del_gap_list.append(np.nan)
            w_evt_mvo_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            w_evt_dro_list.append(np.asarray(_feasible_placeholder(len(names_all), G), float))
            continue
        
        # RegDRO: conditional μ, unconditional Σ.
        # Prefer δ from the in-regime panel, but if the intersection is empty,
        # fall back to the full window for δ to avoid empty bootstrap samples.
        use_cond = bool(mask_cond_all.size) and int(mask_cond_all.sum()) > 0
        X_cond = X_win[mask_cond_all, :] if use_cond else X_win
        w_reg_sub, delta_k = solve_dro(mu_cond, Sig, params_reg, G, R=X_cond, verbose=bool(verbose))

        # Embed to full name list (O(1) via pos_map)
        w_reg_full = np.zeros(len(names_all))
        w_reg_sub_np = asnumpy_strict(w_reg_sub, float).ravel()
        idxs = [pos_map[n] for n in keep]
        w_reg_full[idxs] = w_reg_sub_np
        w_reg_list.append(w_reg_full); del_reg_list.append(float(delta_k))
          
        # unconditional δ for reporting (same window, same keep)
        try:
            delta_uncond_k = _delta_uncond_cached(mu_uncond, Sig, X_win, params_reg)
        except Exception:
            delta_uncond_k = np.nan
            
        del_uncond_list.append(float(delta_uncond_k) if np.isfinite(delta_uncond_k) else np.nan)
        del_gap_list.append((float(delta_uncond_k) - float(delta_k)) if (np.isfinite(delta_uncond_k) and np.isfinite(delta_k)) else np.nan)
    
        # Event-based MVO (unconditional μ, same Σ)
        w_evt_mvo_sub = solve_optimizer(mu_uncond, Sig, delta=0.0, config=G, verbose=False)
        w_evt_mvo = np.zeros(len(names_all))
        w_evt_mvo_sub_np = asnumpy_strict(w_evt_mvo_sub, float).ravel()
        idxs = [pos_map[n] for n in keep]
        w_evt_mvo[idxs] = w_evt_mvo_sub_np
        w_evt_mvo_list.append(w_evt_mvo)

        # Event-based DRO — reuse cached δ when bootstrap method
        delta_evt = _delta_uncond_cached(mu_uncond, Sig, X_win, params_dro)
        w_evt_dro_sub = solve_optimizer(mu_uncond, Sig, delta_evt, config=G, verbose=False)
        
        w_evt_dro = np.zeros(len(names_all))
        w_evt_dro_sub_np = asnumpy_strict(w_evt_dro_sub, float).ravel()
        idxs = [pos_map[n] for n in keep]
        w_evt_dro[idxs] = w_evt_dro_sub_np
        w_evt_dro_list.append(w_evt_dro)

    fit_reg = {
        "type": "piecewise",
        "w_list": [np.asarray(w, float) for w in w_reg_list],
        "segs":   np.asarray(taus, dtype=int),
        "names":  names_all,
        "delta_list":        [float(d) if np.isfinite(d) else np.nan for d in del_reg_list],
        "delta_uncond_list": [float(d) if np.isfinite(d) else np.nan for d in del_uncond_list],
        "delta_gap_list":    [float(d) if np.isfinite(d) else np.nan for d in del_gap_list],
    }
    
    fit_evt_mvo = {"type": "piecewise", "w_list": [np.asarray(w, float) for w in w_evt_mvo_list], "segs": np.asarray(taus, dtype=int)}
    fit_evt_dro = {"type": "piecewise", "w_list": [np.asarray(w, float) for w in w_evt_dro_list], "segs": np.asarray(taus, dtype=int)}
    
    # 4) Fixed-schedule MVO/DRO — BUT use the *same* RegDRO-eligible universe per fixed date
    w_fix_mvo_list, w_fix_dro_list = [], []
    
    for a, b in zip(marks[:-1], marks[1:]):
        if a == 0:
            w_fix_mvo_list.append(_feasible_placeholder(len(names_all), G))
            w_fix_dro_list.append(_feasible_placeholder(len(names_all), G))
            continue
    
        D_pos = int(a)
        ok, keep, pos_map, X_win_df, mu_cond, mu_uncond, Sig, X_win, _ = regdro_decision_context(
            D_pos=D_pos,
            full_index_fit=full_index_fit,
            df_returns_full=df_returns_full,
            names_all=names_all,
            Z_labels_fit=Z_labels_fit,
            AF=AF,
            min_obs=min_lb,
            max_lb=max_lb,
            lam_shr=lam,
            G=G,)

        if not ok:
            w_fix_mvo_list.append(_feasible_placeholder(len(names_all), G))
            w_fix_dro_list.append(_feasible_placeholder(len(names_all), G))
            continue
    
        # MVO fixed (δ=0)
        w_mvo_sub = solve_optimizer(mu_uncond, Sig, delta=0.0, config=G, verbose=False)
        w_mvo = np.zeros(len(names_all))
        w_mvo_sub_np = asnumpy_strict(w_mvo_sub, float).ravel()
        idxs = [pos_map[n] for n in keep]
        w_mvo[idxs] = w_mvo_sub_np
        w_fix_mvo_list.append(w_mvo)
        
        # DRO fixed (unconditional δ on same window) — reuse cached δ when bootstrap method
        delta_fix = _delta_uncond_cached(mu_uncond, Sig, X_win, params_dro)
        w_dro_sub = solve_optimizer(mu_uncond, Sig, delta_fix, config=G, verbose=False)
        w_dro = np.zeros(len(names_all))
        w_dro_sub_np = asnumpy_strict(w_dro_sub, float).ravel()
        idxs = [pos_map[n] for n in keep]
        w_dro[idxs] = w_dro_sub_np
        w_fix_dro_list.append(w_dro)

    fit_fix_mvo = {"type": "piecewise", "w_list": [np.asarray(w, float) for w in w_fix_mvo_list], "segs": np.asarray(marks, dtype=int)}
    fit_fix_dro = {"type": "piecewise", "w_list": [np.asarray(w, float) for w in w_fix_dro_list], "segs": np.asarray(marks, dtype=int)}
    
    # 5) Build weights-on-dates and PnL for the four strategies + RegDRO
    k_delay = int(CONFIG["EXECUTION"].get("execution_delay", 0))
    tc      = float(CONFIG["EXECUTION"].get("trading_cost", 0.0))
    
    # Helper to convert a piecewise fit to dated rows (keep only dates >= OOS start)
    def _dated_rows_from_fit(fit, calendar, col_names):
        rows = []
        rebal_dates = calendar[np.asarray(fit["segs"], int)[:-1]]
        oos_start = full_index[0]
        for dt, w in zip(rebal_dates, fit["w_list"]):
            if dt >= oos_start:
                rows.append(pd.Series(asnumpy_strict(w, float).ravel(), index=col_names, name=dt))
        return pd.DataFrame(rows).sort_index()
    
    R_oos_names = names_all
    R_oos_panel = df_returns.loc[full_index, R_oos_names]
    
    W_on_dates_reg   = _dated_rows_from_fit(fit_reg,     full_index,     R_oos_names)
    W_on_dates_evt_m = _dated_rows_from_fit(fit_evt_mvo, full_index,     R_oos_names)
    W_on_dates_evt_d = _dated_rows_from_fit(fit_evt_dro, full_index,     R_oos_names)
    W_on_dates_fix_m = _dated_rows_from_fit(fit_fix_mvo, full_index_fit, R_oos_names)
    W_on_dates_fix_d = _dated_rows_from_fit(fit_fix_dro, full_index_fit, R_oos_names)
    
    regdro_daily, W_daily_reg, W_eff_reg = pnl_with_delay_and_cost(W_on_dates_reg,   full_index, R_oos_panel, k_delay, tc, "RegDRO_daily")
    mvo_evt_daily, W_daily_me, W_eff_me  = pnl_with_delay_and_cost(W_on_dates_evt_m, full_index, R_oos_panel, k_delay, tc, "MVO_event_daily")
    dro_evt_daily, W_daily_de, W_eff_de  = pnl_with_delay_and_cost(W_on_dates_evt_d, full_index, R_oos_panel, k_delay, tc, "DRO_event_daily")
    mvo_fix_daily, W_daily_mf, W_eff_mf  = pnl_with_delay_and_cost(W_on_dates_fix_m, full_index, R_oos_panel, k_delay, tc, "MVO_fixed_daily")
    dro_fix_daily, W_daily_df, W_eff_df  = pnl_with_delay_and_cost(W_on_dates_fix_d, full_index, R_oos_panel, k_delay, tc, "DRO_fixed_daily")

    # Monthly holdings snapshots
    me_idx = _period_ends(full_index, "M")
    H_reg = W_eff_reg.reindex(me_idx).ffill().rename_axis("date")
    H_mvo_fix = W_eff_mf.reindex(me_idx).ffill().rename_axis("date")
    H_dro_fix = W_eff_df.reindex(me_idx).ffill().rename_axis("date")
    H_mvo_evt = W_eff_me.reindex(me_idx).ffill().rename_axis("date")
    H_dro_evt = W_eff_de.reindex(me_idx).ffill().rename_axis("date")
    
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
    
    rows_mvo_fix = dict(_summ_from_series(mvo_fix_daily, G, AF, n_aligned))
    rows_dro_fix = dict(_summ_from_series(dro_fix_daily, G, AF, n_aligned))
    rows_mvo_evt = dict(_summ_from_series(mvo_evt_daily, G, AF, n_aligned))
    rows_dro_evt = dict(_summ_from_series(dro_evt_daily, G, AF, n_aligned))
    rows_reg     = dict(_summ_from_series(regdro_daily,  G, AF, n_aligned))
    rows_spx     = dict(_summ_from_series(spx_daily,     G, AF, n_aligned))
    
    # Gross exposure (point)
    a_oos = full_index_fit.get_loc(full_index[0])
    b_oos = full_index_fit.get_loc(full_index[-1]) + 1
    win_oos = (a_oos, b_oos)
    T_req = len(full_index)
    rows_mvo_fix["gross_exp"] = _gross_exp_on_window(fit_fix_mvo, T_req, win=win_oos)
    rows_dro_fix["gross_exp"] = _gross_exp_on_window(fit_fix_dro, T_req, win=win_oos)
    rows_mvo_evt["gross_exp"] = _gross_exp_on_window(fit_evt_mvo, T_req)
    rows_dro_evt["gross_exp"] = _gross_exp_on_window(fit_evt_dro, T_req)
    rows_reg["gross_exp"]     = _gross_exp_on_window(fit_reg,     T_req)
    
    # Deltas
    for rows in (rows_mvo_fix, rows_mvo_evt, rows_dro_evt, rows_spx):
        rows["delta"] = float("nan"); rows["delta_uncond"] = float("nan"); rows["delta_gap"] = float("nan")
    
    # DRO fixed/event: δ is unconditional by construction (average over dates)
    def _avg_list_delta(fit, params):
        # recompute average δ using same windows used for that fit
        segs = [int(x) for x in fit["segs"]]
        if len(segs) <= 1: return np.nan
        dlst = []
        for a, b in zip(segs[:-1], segs[1:]):
            D_pos = int(a)
            ok, keep, _, X_win_df, mu_cond, mu_uncond, Sig, X_win, _ = regdro_decision_context(
                D_pos=D_pos, full_index_fit=full_index_fit, df_returns_full=df_returns_full,
                names_all=names_all, Z_labels_fit=Z_labels_fit, AF=AF,
                min_obs=min_lb, max_lb=max_lb, lam_shr=lam, G=G)

            if not ok: continue
            try:
                dlst.append(float(_delta_uncond_cached(mu_uncond, Sig, X_win, params)))
            except Exception:
                pass
        return float(np.nanmean(dlst)) if len(dlst) else np.nan
    
    delta_dro_fix = _avg_list_delta(fit_fix_dro, params_dro)
    delta_dro_evt = _avg_list_delta(fit_evt_dro, params_dro)
    
    rows_dro_fix["delta"] = delta_dro_fix
    rows_dro_fix["delta_uncond"] = delta_dro_fix
    rows_dro_fix["delta_gap"] = 0.0
    
    rows_dro_evt["delta"] = delta_dro_evt
    rows_dro_evt["delta_uncond"] = delta_dro_evt
    rows_dro_evt["delta_gap"] = 0.0
    
    # RegDRO deltas (conditional vs unconditional)
    d_cond   = np.asarray(fit_reg.get("delta_list", []), dtype=float)
    d_uncond = np.asarray(fit_reg.get("delta_uncond_list", []), dtype=float)
    d_gap    = np.asarray(fit_reg.get("delta_gap_list", []), dtype=float)
    rows_reg["delta"]        = float(np.nanmean(d_cond))   if d_cond.size   else np.nan
    rows_reg["delta_uncond"] = float(np.nanmean(d_uncond)) if d_uncond.size else np.nan
    rows_reg["delta_gap"]    = float(np.nanmean(d_gap))    if d_gap.size    else (rows_reg["delta_uncond"] - rows_reg["delta"] if (np.isfinite(rows_reg["delta_uncond"]) and np.isfinite(rows_reg["delta"])) else np.nan)
    
    # Bench-relative point estimates
    def _bench_stats(port, bench, AF=252):
        ex = (port - bench).dropna()
        if ex.empty: return float("nan"), float("nan"), float("nan")
        alpha = AF * ex.mean(); te = (AF ** 0.5) * ex.std(ddof=1); ir = alpha / te if (np.isfinite(te) and te != 0) else float("nan")
        return float(alpha), float(te), float(ir)
    
    for rows, ser in ((rows_mvo_fix, mvo_fix_daily), (rows_dro_fix, dro_fix_daily), (rows_mvo_evt, mvo_evt_daily), (rows_dro_evt, dro_evt_daily), (rows_reg, regdro_daily)):
        a, te, ir = _bench_stats(ser, spx_daily, AF)
        rows["alpha_ann"] = a; rows["te_ann"] = te; rows["ir_ann"] = ir
    
    # SPX presentation
    rows_spx["alpha_ann"] = rows_spx["te_ann"] = rows_spx["ir_ann"] = rows_spx["hit_rate"] = float("nan")
    rows_spx["vol_breach"] = float("nan"); rows_spx["gross_exp"] = 1.0
    # Hit-rate vs SPX
    def _hit_rate(port, bench):
        m = pd.Series(np.isfinite(port) & np.isfinite(bench), index=port.index)
        if not m.any(): return float("nan")
        return float(((port[m] - bench[m]) >= 0.0).mean())
    
    rows_mvo_fix["hit_rate"] = _hit_rate(mvo_fix_daily, spx_daily)
    rows_dro_fix["hit_rate"] = _hit_rate(dro_fix_daily, spx_daily)
    rows_mvo_evt["hit_rate"] = _hit_rate(mvo_evt_daily, spx_daily)
    rows_dro_evt["hit_rate"] = _hit_rate(dro_evt_daily, spx_daily)
    rows_reg["hit_rate"]     = _hit_rate(regdro_daily,  spx_daily)

    # Assemble DataFrames for the table (no jackknife applied here to keep patch concise)
    df_mvo_fix = pd.DataFrame([rows_mvo_fix])
    df_dro_fix = pd.DataFrame([rows_dro_fix])
    df_mvo_evt = pd.DataFrame([rows_mvo_evt])
    df_dro_evt = pd.DataFrame([rows_dro_evt])
    df_reg     = pd.DataFrame([rows_reg])
    df_spx     = pd.DataFrame([rows_spx])
    
    # --- build results_dict only for requested models (+ SPX benchmark) ---
    results_dict = {}
    if "MVO_fixed" in models_set:
        results_dict["MVO_fixed"] = df_mvo_fix
    if "DRO_fixed" in models_set:
        results_dict["DRO_fixed"] = df_dro_fix
    if "MVO_event" in models_set:
        results_dict["MVO_event"] = df_mvo_evt
    if "DRO_event" in models_set:
        results_dict["DRO_event"] = df_dro_evt
    if "RegDRO" in models_set:
        results_dict["RegDRO"] = df_reg

    # SPX always in results for comparison
    results_dict["SPX"] = df_spx

    # Optionally run universe jackknife and inject CIs
    jack_ci = None
    if run_jackknife and not IN_BOOT:
        jk_cfg = CONFIG.get("JACKKNIFE", {})
        d_block = int(jk_cfg.get("d", 0))
        if d_block <= 0:
            raise ValueError("CONFIG['JACKKNIFE']['d'] must be positive when run_jackknife=True.")
        alpha_jk = float(jk_cfg.get("alpha", 0.05))
        seed_jk = jk_cfg.get("seed", None)

        # jackknife over the actual investable universe (names_all)
        jk = jackknife_universe_oos(
            securities=names_all,
            CONFIG=CONFIG,
            d=d_block,
            alpha=alpha_jk,
            seed=seed_jk,)
        
        jack_ci = jk.get("ci", {})

        # Add *_ci_low / *_ci_high columns to result DataFrames
        for strat_name, ci_metrics in jack_ci.items():
            if strat_name not in results_dict:
                continue
            df = results_dict[strat_name]
            for metric, agg in ci_metrics.items():
                lo = agg.get("ci_low", np.nan)
                hi = agg.get("ci_high", np.nan)
                df[f"{metric}_ci_low"] = [lo]
                df[f"{metric}_ci_high"] = [hi]

    # print once (respect bootstrap flag)
    show_table = True
    _boot = CONFIG.get("__bootstrap_run", {})
    if bool(_boot.get("active", False)):
        show_table = (int(_boot.get("i", -1)) == int(_boot.get("B", -1)) - 1)

    if show_table:
        # order: requested models in canonical order, then SPX
        base_order = [m for m in ["MVO_fixed", "DRO_fixed", "MVO_event", "DRO_event", "RegDRO"]
                      if m in results_dict]
        if "SPX" in results_dict:
            base_order.append("SPX")
        print_oos_table(results_dict, model_order=base_order)

    # outputs
    out = {}

    # Per-model blocks only if requested
    if "MVO_fixed" in models_set:
        out["MVO_fixed"] = {"fit": fit_fix_mvo, "summary": rows_mvo_fix}
    if "DRO_fixed" in models_set:
        out["DRO_fixed"] = {"fit": fit_fix_dro, "summary": rows_dro_fix}
    if "MVO_event" in models_set:
        out["MVO_event"] = {"fit": fit_evt_mvo, "summary": rows_mvo_evt}
    if "DRO_event" in models_set:
        out["DRO_event"] = {"fit": fit_evt_dro, "summary": rows_dro_evt}
    if "RegDRO" in models_set:
        out["RegDRO"] = {
            "fit": fit_reg,
            "summary": rows_reg,
            "global_segs": [int(x) for x in fit_reg["segs"]],
            "Z_labels": {k: np.asarray(v) for k, v in Z_labels.items()},
        }

    # SPX and shared artifacts always returned
    out["SPX"] = {"series": spx_daily, "summary": rows_spx}
    out["returns"] = df_returns

    # Time series: only requested models + SPX
    series_dict = {}
    if "MVO_fixed" in models_set:
        series_dict["MVO_fixed_daily"] = mvo_fix_daily
    if "DRO_fixed" in models_set:
        series_dict["DRO_fixed_daily"] = dro_fix_daily
    if "MVO_event" in models_set:
        series_dict["MVO_event_daily"] = mvo_evt_daily
    if "DRO_event" in models_set:
        series_dict["DRO_event_daily"] = dro_evt_daily
    if "RegDRO" in models_set:
        series_dict["RegDRO_daily"] = regdro_daily
    series_dict["SPX_daily"] = spx_daily
    out["series"] = series_dict

    out["securities"] = names_all
    out["G"] = G

    # Holdings: only requested models
    holdings_dict = {}
    if "MVO_fixed" in models_set:
        holdings_dict["MVO_fixed"] = H_mvo_fix
    if "DRO_fixed" in models_set:
        holdings_dict["DRO_fixed"] = H_dro_fix
    if "MVO_event" in models_set:
        holdings_dict["MVO_event"] = H_mvo_evt
    if "DRO_event" in models_set:
        holdings_dict["DRO-event"] = H_dro_evt   # keep original key name
    if "RegDRO" in models_set:
        holdings_dict["RegDRO"] = H_reg
    out["holdings"] = holdings_dict

    # Attach jackknife CIs to the summary dicts as well
    if jack_ci is not None:
        for strat_name, ci_metrics in jack_ci.items():
            if strat_name not in out:
                continue
            summ = out[strat_name]["summary"]
            for metric, agg in ci_metrics.items():
                summ[f"{metric}_ci_low"] = float(agg.get("ci_low", np.nan))
                summ[f"{metric}_ci_high"] = float(agg.get("ci_high", np.nan))

    if "dro_pickle" in CONFIG and CONFIG["dro_pickle"]:
        save_out(out, CONFIG["dro_pickle"])

    return out

