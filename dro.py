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
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
import os, re, ast
import pickle, gzip
from scipy import stats as sp_stats

# checks
print("cvxpy:", cvx.__version__, "| clarabel available?", "CLARABEL" in cvx.installed_solvers())
print()

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
# Wasserstein helper functions
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
    return float(xp.sqrt(C[r, c].mean()).item())

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
    Approximate multivariate W2 by averaging 1D W2 across random directions.
    X, Y: (n,d) samples with same n preferred (bootstrap uses same n).
    Returns W2 (not squared).
    """
    # keep inputs on device, prefer float32 to cut bandwidth
    X = xp.asarray(X, dtype=xp.float32)
    Y = xp.asarray(Y, dtype=xp.float32)
    n, d = X.shape; m = Y.shape[0]

    # generate directions directly on device when rng is None
    if (rng is None) and hasattr(xp.random, "standard_normal"):
        U = xp.random.standard_normal((n_proj, d), dtype=X.dtype)
    else:
        rng = _rng_from_params({}) if rng is None else rng
        U = xp.asarray(rng.normal(size=(n_proj, d)), dtype=X.dtype)

    U = U / xp.maximum(xp.linalg.norm(U, axis=1, keepdims=True), 1e-12)

    XU = X @ U.T         # (n, P)
    YU = Y @ U.T         # (m, P)
    XU = xp.sort(XU, axis=0)
    YU = xp.sort(YU, axis=0)

    if m != n:
        k = int(min(n, m))
        if k <= 1:
            # fall back to mean difference on a single quantile (avoids empty/NaN)
            XU = xp.mean(XU, axis=0, keepdims=True)
            YU = xp.mean(YU, axis=0, keepdims=True)
        else:
            q = xp.linspace(0, 1, k)
            XU = xp.quantile(XU, q, axis=0)
            YU = xp.quantile(YU, q, axis=0)

    diff = XU - YU
    w2_sq = xp.mean(diff * diff)
    return float(xp.sqrt(xp.maximum(w2_sq, 0.0)).item())

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

    if not isinstance(params, dict):
        params = {}
    method = params.get("delta_method", "bootstrap_gaussian")  # fast, GPU-friendly default
    
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
        c1    = float((params or {}).get("c1", 3.0))
        c2    = float((params or {}).get("c2", 1.0))
        a     = float((params or {}).get("a", 2.0))
        n_obs = int(R.shape[0]) if (R is not None and hasattr(R, "shape")) else 1
        n     = int((params or {}).get("n_ref", n_obs))
        d     = int(xp.size(mu_est))
        num   = xp.log(c1 / max(alpha, 1e-12))
        den   = c2 * max(n, 1)
        thresh= num / max(c2, 1e-12)
        base  = max(num / max(den, 1e-12), 1e-12)
        exp   = 1.0 / max(max(d, 2), 1) if n >= thresh else 1.0 / max(a, 1e-12)
        return float(base ** exp)

    if method == "bootstrap_np":
        # Fast replacement: sliced W2 with optional subsampling (no Hungarian; ~O(B·n_proj·n log n))
        assert R is not None, "bootstrap_np needs raw sample matrix R."
        alpha   = float((params or {}).get("alpha", 0.05))
        B       = int((params or {}).get("B", 256))               # lower default
        n_proj  = int((params or {}).get("n_proj", 128))          # slices per bootstrap
        m_cap   = int((params or {}).get("m_cap", 4096))          # subsample cap
        rng_np  = _rng_from_params(params or {})
        R_xp    = xp.asarray(R, dtype=xp.float32)
        n_src   = int(R_xp.shape[0])
        # subsample to m ≤ m_cap for each bootstrap (keeping equal sizes)
        m       = int(min(n_src, m_cap))
        dists   = []
        for _ in range(B):
            idx = rng_np.integers(0, n_src, size=m)
            Rb  = R_xp[idx]
            # sliced-W2 (works best when sizes match; we enforce m==m)
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
    """Symmetrize, regularize to PSD, then return Cholesky factor L s.t. L.T @ L ≈ Sigma_psd."""
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

def solve_optimizer_kkt(mu, Sigma, delta, config, max_it=50, tol=1e-7):
    """
    Solve:  maximize   mu^T w - delta * ||w||_2
            subject to ||L w||_2 <= r_b,  1^T w = 1
    Using KKT with worst-case u* = delta * w / ||w|| and scalar λ for the ellipsoidal constraint.
    Iteration: fix u, solve linear KKT for (w, ν) at given λ, then bisection on λ to hit ||L w||=r_b.
    """
    rb   = float(config["risk_budget"])
    eps  = float(config["epsilon_sigma"])
    mu   = xp.asarray(mu, dtype=xp.float64).reshape(-1)
    Sigma= xp.asarray(Sigma, dtype=xp.float64)
    n    = mu.size
    I    = xp.eye(n, dtype=mu.dtype)
    L    = psd_cholesky(Sigma, eps)  # (n,n)
    LtL  = L.T @ L
    
    # ---- Feasibility precheck: min ||L w|| s.t. 1^T w = 1
    one = xp.ones(n, dtype=mu.dtype)
    # Use pseudoinverse for robustness
    try:
        z = xp.linalg.solve(LtL, one)
    except xp.linalg.LinAlgError:
        z = xp.linalg.pinv(LtL) @ one
    den = float(one @ z)
    if abs(den) < 1e-16:
        raise RuntimeError("Feasibility check failed: singular LtL with 1^T w = 1.")
    w_min = z / den
    rb_min = float(xp.linalg.norm(L @ w_min))
    if rb < rb_min - 1e-12:
        raise RuntimeError(f"Infeasible risk budget: rho={rb:.6g} < rho_min={rb_min:.6g}")

    # helper: given u and λ, solve (2λ LtL + t I) w + ν 1 = mu - u, with 1^T w = 1
    # use a tiny Tikhonov (t) to keep system well-conditioned when λ≈0
    def solve_given_u_lambda(u, lam, t=None):
        # ridge scaled to matrix magnitude
        if t is None:
            t = 1e-10 * (float(xp.trace(LtL)) / max(n,1) + 1.0)
        A = 2.0*lam*LtL + t*I
        # Solve KKT:
        # [A   1][w] = [mu-u]
        # [1^T 0][ν]   [  1  ]
        # Use Schur complement on ν
        # s = 1^T A^{-1} 1 ; ν = (1 - 1^T A^{-1} (mu-u)) / s
        # w = A^{-1} (mu-u - ν 1)

        try:
            chol = xp.linalg.cholesky(A)  # U (upper), A = U^T U
            def Ainv(b):
                y = xp.linalg.solve(chol.T, b)  # solve U^T y = b
                return xp.linalg.solve(chol, y) # solve U x  = y
 
        except xp.linalg.LinAlgError:
            # Fallback to symmetric solve
            def Ainv(b):
                return xp.linalg.solve(A, b)

        one = xp.ones(n, dtype=mu.dtype)
        Au1 = Ainv(one)
        rhs = mu - u
        Aurb= Ainv(rhs)
        s   = float(one @ Au1)
        num = 1.0 - float(one @ Aurb)
        nu  = num / max(s, 1e-18)
        w   = Ainv(rhs - nu*one)
        return w, nu

    # outer fixed-point on u = delta * w / ||w||
    u = xp.zeros_like(mu)
    for it in range(max_it):
        
        # inner bisection on λ to meet ||L w|| = rb (monotone ↓ in λ)
        lam_lo, lam_hi = 0.0, 1.0

        # ensure upper bracket yields bw <= rb (feasible boundary)
        for _ in range(60):
            w_hi, _ = solve_given_u_lambda(u, lam_hi)
            bw_hi = float(xp.linalg.norm(L @ w_hi))
            if (bw_hi <= rb) or (lam_hi > 1e12):
                break
            lam_hi *= 2.0

        # if even huge λ doesn’t reduce below rb, the check above would have caught infeasibility
        # bisection on [lam_lo, lam_hi]
        w_best = w_hi
        for _ in range(80):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            w_mid, _ = solve_given_u_lambda(u, lam_mid)
            bw_mid = float(xp.linalg.norm(L @ w_mid))
            w_best = w_mid
            if abs(bw_mid - rb) <= 1e-8 or (lam_hi - lam_lo) <= max(1e-14, 1e-10 * lam_hi):
                break
            if bw_mid > rb:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid

        w = w_best
        
        wn = float(xp.linalg.norm(w))
        if delta == 0.0 or wn < 1e-15:
            break
        u_new = (delta / max(wn, 1e-15)) * w
        if float(xp.linalg.norm(u_new - u)) <= tol * (1.0 + float(xp.linalg.norm(u_new))):
            u = u_new
            break
        u = u_new
    return xp.asarray(w).reshape(-1)

def solve_optimizer(mu, Sigma, delta, config, verbose=False):
    import numpy as _np
    n = len(mu); rb = config["risk_budget"]; eps = config["epsilon_sigma"]

    # do algebra on GPU/CPU with xp, but convert to NumPy for CVXPY
    L_xp = psd_cholesky(Sigma, eps)
    L = _np.asarray(L_xp)                   # <- host copy
    mu = _np.asarray(mu)
    # Sigma isn't used directly by CVXPY; no need to convert unless you do

    # ---- Fast KKT solver (no CVXPY) ----
    try:
        w_fast = solve_optimizer_kkt(mu, Sigma, float(delta), {"risk_budget": rb, "epsilon_sigma": eps})
        return xp.asarray(w_fast).reshape(-1)
    except Exception as _e:
        if verbose:
            print("KKT solver failed, falling back to CVXPY:", _e)

    # Convex form: minimize  δ‖w‖₂ − μᵀw  s.t. ‖Lw‖₂ ≤ ρ, 1ᵀw = 1
    w = cp.Variable(n)
    rb_safe = float(max(rb, 1e-12))
    Ls = L / rb_safe                    # scale cone: ‖Ls w‖₂ ≤ 1  (improves conditioning)
    constraints = [cp.norm(Ls @ w, 2) <= 1, cp.sum(w) == 1]
    objective   = cp.Minimize(delta * cp.norm(w, 2) - mu @ w)
    prob = cp.Problem(objective, constraints)
    
    # -------- Clarabel (primary) --------
    # Clarabel’s CVXPY wrapper DOES NOT accept 'max_iters' — use 'max_iter' if you set one.
    # Start with no extra settings (most robust). Then a second pass with mild settings if needed.
    try:
        prob.solve(solver=cp.CLARABEL, verbose=verbose)
        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            # one more Clarabel attempt with mild settings (valid kw names!)
            prob.solve(
                solver=cp.CLARABEL,
                verbose=True if verbose else False,
                max_iter=10000,            # <-- correct key (no 's')
                tol_gap_abs=1e-8,
                tol_gap_rel=1e-8,
                tol_feas=1e-8,)
            
    except Exception as e:
        if verbose:
            print("Clarabel failed:", e)
    
    # If Clarabel didn’t deliver, use ECOS as a *backup* (still no SCS).
    if (w.value is None) or (prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)):
        prob.solve(
            solver=cp.ECOS,
            verbose=verbose,
            max_iters=10000,
            warm_start=True,
            abstol=1e-8, reltol=1e-8, feastol=1e-8,
            abstol_inacc=1e-7, reltol_inacc=1e-7, feastol_inacc=1e-7,)
    
    if (w.value is None) or (prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)):
        raise RuntimeError(f"Solve failed with status={prob.status}")
    
    return xp.asarray(w.value).reshape(-1)

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
    return {"type": "static", "w": w, "kappa": params.get("kappa", xp.nan), "delta": float(delta)}

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
            log_seg   = xp.log1p(R_seg)
            mu_est    = xp.expm1(log_seg.mean(axis=0) * AF)
            Sigma_est = xp.cov(log_seg.T, ddof=1) * AF
            R_source  = R_seg                         # << keep regime-k distribution
    
        # pass full-sample N via n_ref but bootstrap from R_source
        params_k = dict(params); params_k["n_ref"] = (b - a)   # use segment length

        delta_k = compute_delta(params_k.get("kappa", 1.0), mu_est, Sigma_est, R_source, params_k)

        w_k = solve_optimizer(mu_est, Sigma_est, delta_k,
                              {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
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
    w = solve_optimizer(data["mu_ann_full"], data["Sigma_ann_full"], delta,
                        {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
    return {"type": "static", "w": w, "delta": delta, "kappa": xp.nan}

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
            log_seg = xp.log1p(R)
            mu_est  = xp.expm1(log_seg.mean(axis=0) * AF)
            Sigma_est = xp.cov(log_seg.T, ddof=1) * AF
        w_k = solve_optimizer(mu_est, Sigma_est, delta_list[j],
                              {"risk_budget": G["risk_budget"], "epsilon_sigma": G["epsilon_sigma"]})
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
        mu_est  = xp.expm1(log_seg.mean(axis=0) * data["n_days"])
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

        # REPORT PER-SEGMENT DELTAS; DO NOT AVERAGE
        dlist = list(map(float, fit.get("delta_list", [])))
        stats_oos["delta"] = xp.nan                 # leave single 'delta' empty for Regime-DRO
        for j, dj in enumerate(dlist, start=1):     # expose delta_k1, delta_k2, ...
            stats_oos[f"delta_k{j}"] = dj
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
                 "gross_exp","kappa","gap","delta","max_drawdown"]
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
        for c in delta_k_cols:
            if c in df.columns and len(df[c].dropna())>0:
                s[c] = _fmt_series(df[c])
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
    L = psd_cholesky(Sigma_ann, config["epsilon_sigma"])
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
    Sig_d = xp.cov(logR.T, ddof=1)
    return {
        "train": X, "test": X, "n_days": T, "ann_factor": ann_factor,
        "mu_ann_full": xp.expm1(mu_d * ann_factor), "Sigma_ann_full": Sig_d * ann_factor,
        "px_cols": R.columns.tolist(), "index": R.index}

def pooled_moments_by_regime(R_df, Z_labels, t_idx, ann=252, min_pair=60, mode="pairwise"):
    """
    R_df: DataFrame (T x N) of returns; may contain NaNs (unequal histories).
    Z_labels: dict[name] -> array length T with labels; NaN allowed outside span.
    Uses the label state at t_idx for each asset, then pools over all rows where that state holds,
    intersected with availability (non-NaN returns). Pairwise covariance uses pairwise overlaps.
    """
    names = list(R_df.columns); N = len(names)
    L = xp.log1p(xp.asarray(R_df.values)) # (T, N), may have NaNs
    T = L.shape[0]
    t_idx = int(min(max(0, t_idx), T-1))

    # current regime state per asset (must exist and not NaN at t_idx)
    s = []
    valid_asset = []
    for n in names:
        z = xp.asarray(Z_labels[n], float)  # may contain NaN
        if z.shape[0] != T:
            raise ValueError("Z_labels arrays must have length T after mapping to calendar.")
        if xp.isfinite(z[t_idx]):
            s.append(int(z[t_idx]))
            valid_asset.append(True)
        else:
            s.append(xp.nan)
            valid_asset.append(False)

    # filter to assets that have a valid label at t_idx
    keep = [i for i, ok in enumerate(valid_asset) if ok]
    if not keep:
        # no assets valid at this time -> return zeros of right shape
        mu_ann = xp.zeros(N)
        Sig_ann = xp.zeros((N, N))
        return mu_ann, Sig_ann

    # compute masks and moments
    mu_d = xp.zeros(N); var_d = xp.zeros(N)
    S_sets = [None]*N

    for i, n in enumerate(names):
        if not valid_asset[i]:
            mu_d[i] = 0.0; var_d[i] = 0.0; S_sets[i] = xp.array([], dtype=int); continue
        zi = xp.asarray(Z_labels[n], float)
        # indices where label equals current state and return is available
        S_i = xp.where((zi == s[i]) & xp.isfinite(L[:, i]))[0]
        S_sets[i] = S_i
        li = L[S_i, i]
        mu_d[i]  = float(xp.nanmean(li)) if li.size else 0.0
        var_d[i] = float(xp.nanvar(li, ddof=1)) if li.size > 1 else 0.0

    Sig_d = xp.zeros((N, N), dtype=L.dtype)
    if mode == "diag":
        xp.fill_diagonal(Sig_d, var_d)
    else:
        # Build availability & regime masks (T x N)
        M_avail = ~xp.isnan(L)                                    # non-NaN
        # per-asset regime indicator for the CURRENT state s[i]
        G_mask  = xp.zeros_like(M_avail, dtype=bool)
        for i, n in enumerate(names):
            if valid_asset[i]:
                # rows where asset i is in state s[i]
                zi = xp.asarray(Z_labels[n])
                G_mask[:, i] = (zi == s[i])
        W = M_avail & G_mask                                      # rows used per asset

        # Centered data with weights:
        # counts per asset, means, then centered with zeros where W=0
        n_i   = xp.maximum(W.sum(axis=0, dtype=L.dtype), 0.0)     # (N,)
        one_t = xp.ones((L.shape[0],), dtype=L.dtype)
        # sums per asset over active rows
        sums  = (W * L).sum(axis=0)
        means = xp.where(n_i > 0, sums / n_i, 0.0)
        Xc    = xp.where(W, L - means[None, :], 0.0)

        # Pairwise counts and cross-sums via GEMM
        N_ij  = (W.astype(L.dtype)).T @ W.astype(L.dtype)         # (N,N)
        S_ij  = Xc.T @ Xc                                         # sum of products over overlaps

        # Unbiased covariance where N_ij>=2
        with xp.errstate(invalid="ignore", divide="ignore"):
            C_ij = xp.where(N_ij >= 2.0, S_ij / (N_ij - 1.0), 0.0)

        # Set diagonal to sample variances computed earlier
        for i in range(N):
            C_ij[i, i] = var_d[i]
        Sig_d = C_ij

        # shrink if pairwise sample is thin (use min observed overlap off-diagonal)
        off = ~xp.eye(N, dtype=bool)
        with xp.errstate(all="ignore"):
            N_ij_off = xp.where(off, N_ij, xp.nan)
            # handle corner case: no off-diagonal overlaps -> skip shrink
            if xp.all(xp.isnan(N_ij_off)):
                min_pairs = xp.inf
            else:
                min_pairs = float(xp.nanmin(N_ij_off))
        if xp.isfinite(min_pairs):
            lam = float(min(1.0, max(0.0, (min_pair / max(int(min_pairs), 1)))))  # in [0,1]
            Sig_d = (1 - lam) * Sig_d + lam * xp.diag(xp.diag(Sig_d))

    mu_ann  = xp.expm1(ann * mu_d)
    Sig_ann = ann * Sig_d
    return mu_ann, Sig_ann

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
        return float(m.group(1)) if m else xp.nan

    def dim_metric(row):
        v = row.get("dim_latent", xp.nan)
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


# -------------------------
# Pipeline
# -------------------------

def dro_pipeline(securities, CONFIG, verbose=True):

    def _resolve_rSLDS_outputs(CONFIG):
        res_csv  = CONFIG["results_csv"]
        seg_parq = CONFIG["segments_parquet"]
        return res_csv, seg_parq

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
    GIDX = build_calendar_union_px(
        securities, px_all, CONFIG["start_dt"], CONFIG["end_dt"])

    # Returns panel — IDENTICAL construction for both modes
    R_cols = []
    for t in securities:
        px = _num_series(px_all[t].loc[CONFIG["start_dt"]:CONFIG["end_dt"]]).dropna()
        r  = px.pct_change()                 # native calendar returns
        R_cols.append(r.reindex(GIDX))       # align to union; keep NaN where absent
    R_use    = pd.concat(R_cols, axis=1)
    R_use.columns = list(securities)
    R_df_all = R_use.copy()

    # Part A on intersection only (OK to drop NaN here because Part A is "static common")
    R_use_clean = R_use.dropna(how="any")
    dataA = make_data_from_returns_panel(R_use_clean)

    # Part A: static DRO (on common intersection)
    N = dataA["train"].shape[1]
    paramsA = dict(CONFIG["delta_defaults"][CONFIG["delta_name"]])
    fitA = fit_dro(dataA, paramsA, CONFIG["GLOBAL"])
    assert len(fitA["w"]) == N, f"len(w)={len(fitA['w'])} != N={N} from DATA_A"
    summA = evaluate_portfolio(fitA, dataA, CONFIG["GLOBAL"]) # pass the full fit so delta, kappa are recorded

    # Part A daily portfolio returns (on intersection calendar)
    partA_daily = pd.Series(
        dataA["train"] @ xp.asarray(fitA["w"]).reshape(-1),
        index=dataA["index"], name="PartA_daily")
    
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

    # Regime-DRO solves (time-varying universe; no coarsen)
    names_all = list(avail)
    taus = list(global_segs)
    w_list = []
    seg_deltas = [] 

    for k in range(len(taus) - 1):
        t_mid = min(max(taus[k], 0), T - 1)
    
        # active assets at t_mid: entered and finite label
        A_k = []
        for n in names_all:
            z = xp.asarray(Z_labels[n], float)
            r = R_df_all[n].to_numpy(dtype=float, copy=False)
            if xp.isfinite(z[t_mid]) and xp.isfinite(r[t_mid]):
                A_k.append(n)
        
        print(f"[t={t_mid} | {R_df_all.index[t_mid].date()}] active: {A_k}")
        min_assets = int(CONFIG.get("min_assets", 1))
        if len(A_k) < min_assets:
            print(f"[WARN][t={t_mid} | {R_df_all.index[t_mid].date()}] only {len(A_k)} active assets (<{min_assets}).")
        
        if len(A_k) == 0:
            # No active assets this segment: append zero weights AND a placeholder delta
            w_list.append(xp.zeros(len(names_all)))
            seg_deltas.append(xp.nan)    # keep alignment with segments
            continue
    
        R_df_k = R_df_all[A_k]  # keep NaNs; moments handle masks
        mu_ann, Sig_ann = pooled_moments_by_regime(
            R_df_k, {n: Z_labels[n] for n in A_k}, t_mid,
            ann=252, min_pair=int(CONFIG.get("min_seg_len_obs", 20)), mode="pairwise")
    
        data_k = {
            "train": R_df_k.fillna(0.0).to_numpy(dtype=float),  # content unused when using moments override
            "test":  R_df_k.fillna(0.0).to_numpy(dtype=float),
            "n_days": R_df_k.shape[0],
            "ann_factor": 252,
            "mu_ann_full": xp.asarray(mu_ann, float),
            "Sigma_ann_full": xp.asarray(Sig_ann, float),
            "px_cols": list(A_k),
            "index": R_df_k.index
        }
        
        paramsR = dict(CONFIG["delta_defaults"][CONFIG["delta_name"]])
        paramsR["use_moments_override"] = True
    
        fit_k = fit_dro(data_k, paramsR, CONFIG["GLOBAL"])
        w_sub = xp.asarray(fit_k["w"]).reshape(-1)
        delta_k = float(fit_k.get("delta", xp.nan))
        
        # expand to full vector
        w_full = xp.zeros(len(names_all))
        pos = {n: i for i, n in enumerate(names_all)}
        for j, n in enumerate(A_k):
            w_full[pos[n]] = w_sub[j]
        w_list.append(w_full)
    
        # collect per-segment delta (aligned with this segment)
        seg_deltas.append(delta_k)

    fitB = {
        "type": "piecewise",
        "w_list": [xp.asarray(w, float) for w in w_list],
        "segs": xp.asarray(taus, dtype=xp.int64),
        "names": names_all,
        "delta_list": seg_deltas,
    }

    # evaluation uses the union calendar
    data_eval = {
        "train": R_df_all.fillna(0.0).to_numpy(dtype=float),
        "test":  R_df_all.fillna(0.0).to_numpy(dtype=float),
        "n_days": T,
        "ann_factor": 252,
        "mu_ann_full": xp.zeros(len(names_all)),
        "Sigma_ann_full": xp.eye(len(names_all)),
        "px_cols": names_all,
        "index": R_df_all.index}
    
    # extra dataset digest for Part B eval
    X = R_df_all.fillna(0.0).to_numpy(dtype=float)
    summB = evaluate_portfolio(fitB, data_eval, CONFIG["GLOBAL"])

    # Part B daily portfolio returns (on union calendar)
    X_union = R_df_all.fillna(0.0).to_numpy(dtype=float)   # same as evaluation
    T = X_union.shape[0]
    partB = xp.zeros(T, dtype=float)
    for (a, b), w_k in zip(zip(fitB["segs"][:-1], fitB["segs"][1:]), fitB["w_list"]):
        partB[a:b] = X_union[a:b] @ xp.asarray(w_k).reshape(-1)
    
    partB_daily = pd.Series(partB, index=R_df_all.index, name="PartB_daily")

    # MVO baseline
    fit_mvo0  = fit_mvo(dataA, {}, CONFIG["GLOBAL"])
    summ_mvo0 = evaluate_portfolio(fit_mvo0, dataA, CONFIG["GLOBAL"])

    # MVO daily portfolio returns (on intersection calendar)
    mvo_daily = pd.Series(
        dataA["train"] @ xp.asarray(fit_mvo0["w"]).reshape(-1),
        index=dataA["index"], name="MVO_daily")
    
    # Print results
    if verbose:
        print("\n[MVO]    MVO baseline summary:\n", pd.Series(summ_mvo0).round(4))
        print("\n[DRO]    Static DRO summary:\n", pd.Series(summA).round(4))
        print("\n[RegDRO] Regime-DRO summary:\n", pd.Series(summB).round(4))

        # portfolios (weights)
        print("\n[DRO]    weights:\n", _fmt4(fitA["w"]), flush=True)
        print("\n[RegDRO] piecewise weights:", flush=True)
        for k, w_k in enumerate(fitB["w_list"], 1):
            print(f"  k={k}: {_fmt4(w_k)}", flush=True)

    # Save results
    out = {
        "MVO":   {"fit": fit_mvo0, "summary": summ_mvo0},
        "PartA": {"fit": fitA, "data": dataA, "summary": summA},
        "PartB": {"fit": fitB, "summary": summB,
                  "per_asset_segs": per_asset_segs_on_cal, 
                  "global_segs": taus, "Z_labels": Z_labels},
        "returns_union": R_df_all,
        "series": {"MVO_daily": mvo_daily, 
                   "PartA_daily": partA_daily, 
                   "PartB_daily": partB_daily},
        "securities": avail
    }

    # Save results (if path configured)
    if "dro_pickle" in CONFIG and CONFIG["dro_pickle"]:
        save_out(out, CONFIG["dro_pickle"])

    return out

