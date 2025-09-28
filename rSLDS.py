# rSLDS Core Functions

# Modules
import autograd.numpy as anp
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
import numpy as np
import pandas as pd
import ssm
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    mean_squared_error, silhouette_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler
from collections import Counter
from itertools import groupby
from fractions import Fraction
from joblib import Parallel, delayed
import itertools, os, sys
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


# ---------------------------------------------------------------
# Fit rSLDS
# ---------------------------------------------------------------

def fit_rSLDS(y, params, n_iter_em=50, seed=None):
    
    """
    params: dict(n_regimes, dim_latent, single_subspace)
    """

    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)
        
    # unpack params
    N = y.shape[1]
    D = params["dim_latent"]
    K = params["n_regimes"]
    single_subspace = params["single_subspace"]

    # observation noise
    def safe_log_inv(var, lo=1e-6, hi=1e6):
        var = np.nan_to_num(var, nan=hi, posinf=hi, neginf=hi)
        var = np.clip(var, lo, hi)
        log_inv = np.log(1.0 / var)
        return log_inv

    var = np.var(y, 0, keepdims=True)
    log_inv = safe_log_inv(var)

    # OPTIMAL CLUSTERS
    if K is None:
        K, cluster_stats = fit_kmeans(y, Ks=[2, 3, 4], display=False)
        print(cluster_stats)

    # INSTANTIATE MODEL
    mdl = ssm.SLDS(N, K, D,
                   transitions="recurrent_only",
                   dynamics="diagonal_gaussian",
                   emissions="gaussian",
                   single_subspace=single_subspace)

    # INITIALIZE MODEL

    if N < D * K:

        # initialize(y) carries out PCA, which fails when trying to extract D components 
        # from N-dim data, but N < D.
        # print(f"N<D*K: initializing manually")
        
        # discrete-state prior π
        # mdl.init_state_dist = np.full(K, 1.0 / K) 
        mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
 
        # continuous-state prior 𝒩(μ_init, diag(σ^2_init))
        mdl.dynamics.mu_init = np.zeros((K, D))
        mdl.dynamics.sigmasq_init = np.ones((K, D))

        # recurrent transition weights R, r
        mdl.transitions.Rs = 0.01 * np.random.randn(K, D)
        mdl.transitions.r = np.zeros(K)

        # linear dynamics A, b, σ^2  (stable ≈ identity)
        mdl.dynamics.As = 0.95 * np.repeat(np.eye(D)[None, :, :], K, axis=0)
        mdl.dynamics.bs = np.zeros((K, D))
        mdl.dynamics.sigmasq = 1e-4 * np.ones((K, D))

        # eps = 1e-6
        # obs_var = np.var(y, 0)
        # padded_var = np.pad(obs_var, (0, D - N), mode='constant', constant_values=obs_var.mean())
        # sigmasq_init = np.maximum(0.1 * padded_var, eps)
        # mdl.dynamics.sigmasq = np.tile(sigmasq_init, (K, 1))

        # emissions C, d, log-variance
        if single_subspace:
            mdl.emissions.Cs = np.eye(N, D)[None, :, :]  # shape (1,N,D)
            mdl.emissions.ds = np.zeros((1, N))
            mdl.emissions.inv_etas = log_inv
        else:
            mdl.emissions.Cs = np.tile(np.eye(N, D), (K, 1, 1))  # shape (K,N,D)
            mdl.emissions.ds = np.zeros((K, N))
            mdl.emissions.inv_etas = np.tile(log_inv, (K, 1))

    else:
        mdl.initialize(y)  #, discrete_state_init_method="kmeans")  # default

    # FIT MODEL
    elbo, q = mdl.fit(y,
                      method="laplace_em",
                      variational_posterior="structured_meanfield",
                      num_iters=n_iter_em,
                      alpha=0.0,  # default: 0.0. Laplace-EM param: new params=(1−α)⋅M-step params+α⋅old 
                      initialize=False)

    xhat = q.mean_continuous_states[0]
    zhat = mdl.most_likely_states(xhat, y)
    
    return xhat, zhat, elbo, q, mdl


def fit_rSLDS_restricted(y, params, C=None, d=None, n_iter_em=10, seed=None, 
    b_pattern=None, enforce_diag_A=True, 
    lam_dyn=None,            # dynamics repulsion (around mean); None -> from params or 0
    lam_trn=None,            # transitions repulsion
    delta=None,              # hard min separation radius
    q_min=1e-6,              # floor on dynamics variance
    stickiness=0.0,          # small bias to stay in same regime in E-step scoring
    C_mask=None, d_mask=None):
    
    """
    True rSLDS via ssm (Laplace EM + structured mean field):
    - If C,d provided: emissions are fixed & shared (single_subspace=True).
    - If C,d None: emissions are learned as usual (unrestricted).
    - Robust emissions inversion with SVD+ridge (handles N×D and rank-deficient C).
    - Split-EM outer loop (alpha=0.0):
        * E-only pass (freeze all m_steps) -> 1 iter
        * M-enabled pass (enable dynamics/transitions/init; emissions fixed iff C,d provided) -> 1 iter
        * Repeat n_iter_em times.
    - After each M pass:
    - Optional dynamics repulsion + hard min separation (avoid collapse)
    - Optional enforce b pattern with µ bookkeeping (µ = b/(1-ρ))
    - Final E-only pass
    - Returns (xhat, zhat, elbo_trace, q_last, mdl)
    """
    
    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)

    # Shapes & params
    y = np.asarray(y, dtype=float)
    T, N = y.shape
    K = int(params["n_regimes"])
    D = int(params["dim_latent"])
    assert bool(params.get("single_subspace", True)), "Require single_subspace=True."

    # knobs
    if lam_dyn is None:
        lam_dyn = float(params.get("repulsion_strength_dynamics", 0.0))
    if lam_trn is None:
        lam_trn = float(params.get("repulsion_strength_transitions", 0.0))
    if delta is None:
        delta = float(params.get("min_separation", 0.0))
    if b_pattern is None:
        b_pattern = ["mu_form"] * D
    assert len(b_pattern) == D and all(m in {"free", "zero", "mu_form"} for m in b_pattern)

    # ----- model
    mdl = ssm.SLDS(
        N, K, D,
        transitions="recurrent_only",       # stick-breaking recurrent logistic gating
        dynamics="diagonal_gaussian",
        emissions="gaussian",
        single_subspace=True,)

    # ----- emissions (fixed if C,d provided; else learned)
    fixed_emissions = (C is not None) and (d is not None)
    if fixed_emissions:
        C = np.asarray(C, dtype=float)
        d = np.asarray(d, dtype=float)
        assert C.shape == (N, D),  f"C must be (N,D)=({N},{D}), got {C.shape}"
        assert d.shape == (N,),    f"d must be (N,), got {d.shape}"

        # simple data-driven noise init, then lock
        obs_var = np.var(y, axis=0)
        obs_var = np.clip(np.nan_to_num(obs_var, nan=1.0, posinf=1e6, neginf=1e6), 1e-8, 1e6)
        inv_etas_row = np.log(1.0 / obs_var)[None, :]     # (1, N)
        inv_etas     = np.tile(inv_etas_row, (K, 1))      # (K, N)
        mdl.emissions.Cs = C[None, :, :]
        mdl.emissions.ds = d[None, :]
        mdl.emissions.inv_etas = inv_etas          
        mdl.emissions.m_step = (lambda *_, **__: None)  # lock emissions

    # robust inversion for N<D and low-rank C, whether fixed or learned
    def _invert_ridge(self, data, input=None, mask=None, tag=None, ridge=1e-6):
        Y = np.atleast_2d(np.asarray(data, dtype=float)) # (T,N)
        # pull correct C,d (single_subspace)
        Cc = self.Cs[0] if self.Cs.ndim == 3 else self.Cs   # (N, D)
        dc = self.ds[0] if self.ds.ndim == 2 else self.ds   # (N,)
        Yc = Y - dc
        # SVD ridge pseudoinverse: Pinv ≈ (C^T C + λI)^{-1} C^T, done via SVD
        U, s, Vt = np.linalg.svd(Cc, full_matrices=False)   # C = U diag(S) Vt
        s_f = s / (s**2 + ridge)
        Pinv = (Vt.T * s_f) @ U.T                           # (D,N) = C^+
        X = (Pinv @ (Yc.T)).T                               # (T,D)
        return X

    # attach
    ### mdl.emissions._invert = types.MethodType(_invert_ridge, mdl.emissions)
    mdl.emissions._invert = _invert_ridge.__get__(mdl.emissions, mdl.emissions.__class__)

    # ----- initial dynamics & gates (stable + symmetry breaking)
    mdl.dynamics.As = np.repeat(0.95 * np.eye(D)[None, :, :], K, axis=0)  # (K,D,D)
    mdl.dynamics.bs = np.zeros((K, D))
    mdl.dynamics.sigmasq = np.ones((K, D))
    mdl.dynamics.mu_init = np.zeros((K, D))
    mdl.dynamics.sigmasq_init = np.ones((K, D))
    if hasattr(mdl.transitions, "Rs"):
        mdl.transitions.Rs = 0.01 * npr.randn(K, D)  # break symmetry for stick-breaking gates
    if hasattr(mdl.transitions, "r"):
        mdl.transitions.r = np.zeros(K)
    if hasattr(mdl.init_state_distn, "log_pi0"):
        mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
    
    # store µ for reporting (µ = b/(1-ρ))
    mdl.dynamics_mu_param = np.zeros((K, D))

    # ----- helpers: bind m_step, freeze/enable, repulsion, min-sep, constraints
    def _bind_mstep(comp):
        return (getattr(comp.__class__, "_m_step", None) 
                or getattr(comp.__class__, "m_step")).__get__(comp)

    dyn_mstep_base = _bind_mstep(mdl.dynamics)
    trn_mstep_base = _bind_mstep(mdl.transitions)
    pio_mstep_base = _bind_mstep(mdl.init_state_distn)
    emi_mstep_base = _bind_mstep(mdl.emissions)

    def _freeze_all():
        mdl.dynamics.m_step = (lambda *_, **__: None)
        mdl.transitions.m_step = (lambda *_, **__: None)
        mdl.init_state_distn.m_step = (lambda *_, **__: None)
        mdl.emissions.m_step = (lambda *_, **__: None)

    def _enforce_min_sep_matrix(V, delta_):
        if delta_ <= 0 or K <= 1:
            return V
        for i in range(K):
            for j in range(i+1, K):
                diff = V[i] - V[j]
                nrm = float(np.linalg.norm(diff))
                if nrm < delta_:
                    if nrm < 1e-12:
                        dvec = npr.randn(V.shape[1]); dvec /= (np.linalg.norm(dvec) + 1e-12)
                        V[i] += 0.5 * delta_ * dvec
                        V[j] -= 0.5 * delta_ * dvec
                    else:
                        u = diff / nrm
                        push = 0.5 * (delta_ - nrm) * u
                        V[i] += push
                        V[j] -= push
        return V

    def dyn_mstep_with_repulsion(*args, **kwargs):
        dyn_mstep_base(*args, **kwargs)     # ssm closed-form update
        # repulsion around mean + min-sep on [vec(A_k); b_k]
        A = mdl.dynamics.As; B = mdl.dynamics.bs
        if K > 1 and lam_dyn > 0.0:
            A_bar = A.mean(axis=0, keepdims=True); B_bar = B.mean(axis=0, keepdims=True)
            A = A + lam_dyn * (A - A_bar)
            B = B + lam_dyn * (B - B_bar)
            V = np.concatenate([A.reshape(K, -1), B], axis=1)
            V = _enforce_min_sep_matrix(V, delta)
            A = V[:, :D*D].reshape(K, D, D); B = V[:, D*D:]
        mdl.dynamics.As = A; mdl.dynamics.bs = B
        # floor process variances
        if hasattr(mdl.dynamics, "sigmasq"):
            mdl.dynamics.sigmasq = np.maximum(mdl.dynamics.sigmasq, q_min)

    def trn_mstep_with_repulsion(*args, **kwargs):
        if K == 1:
            return
        trn_mstep_base(*args, **kwargs)     # ssm optimization over stick-breaking gates
        if lam_trn > 0.0:
            if hasattr(mdl.transitions, "Rs"):
                R = mdl.transitions.Rs
                Rm = R.mean(axis=0, keepdims=True)
                mdl.transitions.Rs = R + lam_trn * (R - Rm)
            if hasattr(mdl.transitions, "r"):
                r = mdl.transitions.r
                rm = r.mean(axis=0, keepdims=True)
                mdl.transitions.r = r + lam_trn * (r - rm)
            # optional min-sep on [R_k, r_k]
            if delta > 0.0 and hasattr(mdl.transitions, "Rs") and hasattr(mdl.transitions, "r"):
                VT = np.concatenate([mdl.transitions.Rs, mdl.transitions.r[:, None]], axis=1)
                VT = _enforce_min_sep_matrix(VT, delta)
                mdl.transitions.Rs = VT[:, :D]
                mdl.transitions.r = VT[:, -1]
                
    def _enable_M_pass():
        mdl.dynamics.m_step = dyn_mstep_with_repulsion
        mdl.transitions.m_step = trn_mstep_with_repulsion
        mdl.init_state_distn.m_step = pio_mstep_base
        # emissions: only if unrestricted
        mdl.emissions.m_step = (lambda *_, **__: None) if fixed_emissions else emi_mstep_base

    def _enforce_identifiability_and_mu():
        # Diagonal A and clip |rho| < 1
        A = mdl.dynamics.As
        for k in range(K):
            diag = np.clip(np.diag(A[k]), -0.999, 0.999)
            A[k] = np.diag(diag)
        mdl.dynamics.As = A
    
        # Only record μ = b/(1-ρ); do not change b
        B = mdl.dynamics.bs
        for d_idx, mode in enumerate(b_pattern):
            if mode == "zero":
                B[:, d_idx] = 0.0
                mdl.dynamics_mu_param[:, d_idx] = 0.0
            elif mode == "mu_form":
                rho = A[:, d_idx, d_idx]
                denom = np.clip(1.0 - rho, 1e-8, None)
                mdl.dynamics_mu_param[:, d_idx] = B[:, d_idx] / denom
        mdl.dynamics.bs = B
    
        # Re-lock emissions
        if fixed_emissions:
            mdl.emissions.Cs = C[None, :, :]
            mdl.emissions.ds = d[None, :]

    def _project_emissions_(emissions, C_fix, d_fix, C_mask=None, d_mask=None):
        Cs = emissions.Cs  # shape: (K,N,D) or (N,D)
        ds = emissions.ds  # shape: (K,N)   or (N,)
    
        # default: fully fixed if mask not provided
        if C_mask is None:
            C_mask = np.zeros_like(Cs, dtype=float)
        else:
            C_mask = np.asarray(C_mask, dtype=float)
    
        if d_mask is None:
            d_mask = np.zeros_like(ds, dtype=float)
        else:
            d_mask = np.asarray(d_mask, dtype=float)
    
        C_fix = np.asarray(C_fix, dtype=float)
        d_fix = np.asarray(d_fix, dtype=float)
    
        # -------- broadcast C parts --------
        if Cs.ndim == 3 and C_fix.ndim == 2:
            C_fix = np.broadcast_to(C_fix, Cs.shape)
        elif Cs.ndim == 2 and C_fix.ndim == 3:
            C_fix = C_fix[0]
    
        if Cs.ndim == 3 and C_mask.ndim == 2:
            C_mask = np.broadcast_to(C_mask, Cs.shape)
        elif Cs.ndim == 2 and C_mask.ndim == 3:
            C_mask = C_mask[0]
    
        # -------- broadcast d parts --------
        if ds.ndim == 2 and d_fix.ndim == 1:
            d_fix = np.broadcast_to(d_fix, ds.shape)
        elif ds.ndim == 1 and d_fix.ndim == 2:
            d_fix = d_fix[0]
    
        if ds.ndim == 2 and d_mask.ndim == 1:
            d_mask = np.broadcast_to(d_mask, ds.shape)
        elif ds.ndim == 1 and d_mask.ndim == 2:
            d_mask = d_mask[0]
    
        # -------- apply projection --------
        emissions.Cs = C_mask * Cs + (1.0 - C_mask) * C_fix
        emissions.ds = d_mask * ds + (1.0 - d_mask) * d_fix

    
    # ----- outer split-EM (alpha=0.0)
    elbo_trace = []
    q_last = None

    for _ in range(int(n_iter_em)):
        
        # E-only
        _freeze_all()
        elbo_E, q = mdl.fit(
            y,
            method="laplace_em",
            variational_posterior="structured_meanfield",
            num_iters=1,
            alpha=0.0,
            initialize=False)
        elbo_trace.extend(list(elbo_E))

        # M-enabled
        _enable_M_pass()
        elbo_M, q = mdl.fit(
            y,
            method="laplace_em",
            variational_posterior="structured_meanfield",
            num_iters=1,
            alpha=0.0,
            initialize=False)
        elbo_trace.extend(list(elbo_M))
        q_last = q

        # identifiability constraints (after M)
        _enforce_identifiability_and_mu()
        _project_emissions_(mdl.emissions, C_fix=C, d_fix=d, C_mask=C_mask, d_mask=d_mask)

    # Final E-only (sharpen posterior)
    _freeze_all()
    elbo_F, q_last = mdl.fit(
        y,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=1,
        alpha=0.0,
        initialize=False)
    elbo_trace.extend(list(elbo_F))

    # outputs
    xhat = q_last.mean_continuous_states[0]
    zhat = mdl.most_likely_states(xhat, y)
    return xhat, zhat, np.asarray(elbo_trace, dtype=float), q_last, mdl



# ---------------------------------------------------------------
# Load rSLDS (parameters pre-determined)
# ---------------------------------------------------------------

def load_rSLDS_synthetic_price(params, verbose=False):

    dt = params['dt']
    mu_diff = params['mu_diff']
    sigma = params['sigma']
    sigma_diff = params['sigma_diff']
    
    # Model config
    N=1
    K=2
    D=1
    single_subspace=True  ### CHECK
    
    # Instantiate
    mdl = ssm.SLDS(N, K, D,
                   transitions="recurrent_only",
                   dynamics="diagonal_gaussian",
                   emissions="gaussian",
                   single_subspace=single_subspace)
    
    # known values 
    sigma_hi = sigma + sigma_diff/2          # high-vol, negative drift
    sigma_lo = sigma - sigma_diff/2          # low-vol,  positive drift
    var_hi   = sigma_hi**2 * dt
    var_lo   = sigma_lo**2 * dt
    drift_lo = +0.5 * mu_diff * dt                      # regime 1 (index 1)
    drift_hi = -0.5 * mu_diff * dt                      # regime 0 (index 0)
    
    # ---- Emissions ----
    log_inv = np.log(1.0 / np.full(N, var_hi))  # shape (N,)
    if single_subspace:
        mdl.emissions.Cs       = np.eye(N, D)[None, :, :]
        mdl.emissions.ds       = np.zeros((1, N))
        mdl.emissions.inv_etas = log_inv                     # shape (N,)
    else:
        mdl.emissions.Cs       = np.tile(np.eye(N, D), (K,1,1))
        mdl.emissions.ds       = np.zeros((K, N))
        mdl.emissions.inv_etas = np.tile(log_inv, (K,1))     # shape (K,N)
    
    # ---- Dynamics ----
    mdl.dynamics.As = np.zeros((K, D, D))                # x_{t+1}=b+η
    mdl.dynamics.bs = np.zeros((K, D))
    mdl.dynamics.bs[0, 0] = drift_hi            # regime 0    
    mdl.dynamics.bs[1, 0] = drift_lo            # regime 1    
    sigmasq             = np.zeros((K, D))
    sigmasq[0, 0] = var_hi                      # regime 0    
    sigmasq[1, 0] = var_lo                      # regime 1    
    mdl.dynamics.sigmasq  = sigmasq
    mdl.dynamics.mu_init      = np.zeros((K, D))
    mdl.dynamics.sigmasq_init = np.full((K, D), 1e-4)       # fine tune?
    
    # ---- Transitions ----

    w = compute_required_w(sigma=sigma, dt=dt, confidence=0.95)   
    
    mdl.transitions.Rs = np.array([
        [-w], 
        [+w]])
    
    mdl.transitions.r  = np.zeros(K)
    
    # --- initial regime prior ---
    
    # print model
    if verbose:
        print('\nPre-determined model:\n')
        print_rSLDS_matrices(mdl)

    return mdl


def load_rSLDS_synthetic_fundamental(params, verbose=False):
    
    dt=params['dt']
    mu_diff_EPS=params['mu_diff_EPS']
    sigma_EPS=params['sigma_EPS']
    mu_PE=params['mu_PE']
    sigma_PE=params['sigma_PE']
    sigma_diff=params['sigma_diff_PE']
    theta_PE=params['theta_PE']

    # Model config
    N=2
    K=2
    D=2
    single_subspace=True
    
    # Instantiate
    mdl = ssm.SLDS(N, K, D,
                   transitions="recurrent_only",
                   dynamics="diagonal_gaussian",
                   emissions="gaussian",
                   single_subspace=single_subspace)
    
    # known values 
    var_PE_r0 = (sigma_PE + sigma_diff/2)**2 * dt
    var_PE_r1 = (sigma_PE - sigma_diff/2)**2 * dt
    var_EPS   =  sigma_EPS**2 * dt
    
    drift_EPS_r0 = -0.5 * mu_diff_EPS * dt
    drift_EPS_r1 = +0.5 * mu_diff_EPS * dt
    
    # Populate model parameters with known values
    
    # ---- Emissions ----
    log_inv = np.log(1 / np.array([var_PE_r0, var_EPS]))    # shape (N,)
    
    if single_subspace:
        mdl.emissions.Cs       = np.eye(N, D)[None, :, :]
        mdl.emissions.ds       = np.zeros((1, N))
        mdl.emissions.inv_etas = log_inv                     # shape (N,)
    else:
        mdl.emissions.Cs       = np.tile(np.eye(N, D), (K,1,1))
        mdl.emissions.ds       = np.zeros((K, N))
        mdl.emissions.inv_etas = np.tile(log_inv, (K,1))     # shape (K,N)
    
    # ---- Dynamics ----
    mdl.dynamics.As = np.tile(np.eye(D), (K,1,1))
    mdl.dynamics.bs = np.array([[0.0, drift_EPS_r0],   # 
                                [0.0, drift_EPS_r1]])  # 
    mdl.dynamics.sigmasq = np.array([[var_PE_r0, var_EPS],    # 
                                     [var_PE_r1, var_EPS]])   # 
    mdl.dynamics.mu_init      = np.zeros((K, D))
    mdl.dynamics.sigmasq_init = np.full((K, D), 1e-4)
    
    # ---- Transitions ----
    
    w = compute_required_w(sigma=sigma_EPS, dt=dt, confidence=0.95)   # 
    
    mdl.transitions.Rs = np.array([
        [0.0, -w],   # regime 0 favoured when Δlog-EPS < 0
        [0.0, +w]])  # regime 1 favoured when Δlog-EPS > 0
    
    mdl.transitions.r  = np.full(K, 0)
    
    # --- initial regime prior ---
    mdl.init_state_distn.log_pi0 = np.log(np.ones(K)/K)
    
    # print model
    if verbose: 
        print('\nPre-determined model:\n')
        print_rSLDS_matrices(mdl)

    return mdl


# ---------------------------------------------------------------
# rSLDS Inference
# ---------------------------------------------------------------

def inference_rSLDS(px, mdl, y_test, q_train=None, z_true=None, y_train=None, dt=1/252, display=False):
    
    Fs = getattr(mdl.emissions, "Fs", [])
    D_in = Fs[0].shape[1] if len(Fs) else 0
    T2 = y_test.shape[0]
    inputs2 = np.zeros((T2, D_in))
    mask2 = np.ones_like(y_test, dtype=bool)

    q_test = mdl._make_variational_posterior(
        variational_posterior="structured_meanfield",
        datas=[y_test], inputs=[inputs2], masks=[mask2], tags=[None], method="smf")

    xhat_test = q_test.mean_continuous_states[0]                 # (T2, D_latent)
    zhat_test = mdl.most_likely_states(xhat_test, y_test)        # (T2,)
    
    # posterior marginals:
    gamma_test, *_  = mdl.expected_states(xhat_test, y_test, mask=mask2) # shape (T_test, K), rows sum to 1

    # CPLL
    cpll = compute_smoothed_cpll(mdl, xhat_test, y_test, gamma_test)
    entropy = 0.5 * np.log(2 * np.pi * np.e * np.var(y_test))
    max_cpll = -len(y_test) * entropy

    return {"xhat": xhat_test, "zhat": zhat_test, "gamma": gamma_test,
            "cpll": cpll, "max_cpll": max_cpll, "mdl": mdl}


# ---------------------------------------------------------------
# CUSUM Overlay
# ---------------------------------------------------------------

def cusum_overlay(prices, y, xhat, mdl, h_z, verbose=False):
    
    """
    Adaptive-scale CUSUM on latent-dynamics + transition log-likelihoods.

    Definitions:
        log P(xₜ | xₜ₋₁, regime): measures how well the latent dynamics explain the observed state.
        log P(regime | xₜ₋₁): measures how likely the regime is given the past state.
        The sum is the joint log-probability; a complete measure of model fit at time t

    sₜ is the log-ratio of how well the alternative regime explains the current state vs the current regime.
    
    At each time t:
        sₜ = log-likelihood ratio between alternative and current regime:
            sₜ = [log P(xₜ | xₜ₋₁, alt regime) + log P(alt regime | xₜ₋₁)]
                − [log P(xₜ | xₜ₋₁, curr regime) + log P(curr regime | xₜ₋₁)]
    
        zₜ = (sₜ − μₜ) / σₜ        # z-score of sₜ
        Sₜ = max(0, Sₜ₋₁ + zₜ)     # cumulative sum of z-scores

    If Sₜ > h_z, a regime change is triggered.

    Reasonable ranges for h_z. High sensitivity: 2-3. Low sensitivity: 5-7
    
    """
    from scipy.special import logsumexp
    import numpy as np
    import matplotlib.pyplot as plt

    if xhat.shape[0] <= 10: 
        # print(f'skipping CUSUM: time series too short {xhat.shape[0]}')
        return None  # Skip CUSUM if time series too short
        
    # Ensure shapes
    y = np.atleast_2d(y)
    xhat = np.atleast_2d(xhat)
    if y.shape[0] < y.shape[1]: y = y.T
    if xhat.shape[0] < xhat.shape[1]: xhat = xhat.T
    T, D = xhat.shape

    # Model parameters
    A = mdl.dynamics.As
    b = mdl.dynamics.ƒbs
    sig2 = mdl.dynamics.sigmasq
    Rw  = mdl.transitions.Rs
    r   = mdl.transitions.r

    # Storage
    z_hat = np.zeros(T, dtype=int)
    z_hat[0] = 1  # start in regime 1
    S_arr, z_arr = [], []
    p_new_series = []

    # Online stats
    mu = 0.0
    m2 = 0.0
    S = 0.0

    for t in range(1, T):
        k_curr = z_hat[t - 1]
        k_alt = 1 - k_curr
        xp = xhat[t - 1]
        xn = xhat[t]

        eps_c = xn - (A[k_curr] @ xp + b[k_curr])
        eps_a = xn - (A[k_alt]  @ xp + b[k_alt])

        ll_c = -0.5 * np.sum(eps_c**2 / sig2[k_curr] + np.log(2*np.pi*sig2[k_curr]))
        ll_a = -0.5 * np.sum(eps_a**2 / sig2[k_alt ] + np.log(2*np.pi*sig2[k_alt ]))

        logits = Rw @ xp + r
        logits -= logsumexp(logits)

        s = (ll_a + logits[k_alt]) - (ll_c + logits[k_curr])

        # Posterior for plotting
        p_new = 1.0 / (1.0 + np.exp(ll_c - ll_a))
        p_new_series.append(p_new)
        
        # Cumulative sum of s-values (no z-score)
        S = max(0.0, S + s)
        
        S_arr.append(S)
        
        if S > h_z:
            z_hat[t] = k_alt
            S = 0.0  # Reset cumulative sum after regime change
        else:
            z_hat[t] = k_curr

    # Δn estimate
    S_arr = np.asarray(S_arr)
    pos_inc = np.maximum(np.diff(np.concatenate(([0.0], S_arr))), 0.0)
    avg_inc = np.convolve(pos_inc, np.ones(10) / 10, mode="same")
    numer = np.maximum(h_z - S_arr, 0.0)
    denom = avg_inc
    Delta_n = np.full_like(S_arr, np.inf, dtype=float)
    np.divide(numer, denom, out=Delta_n, where=(denom > 0))
    pad = len(prices) - len(Delta_n) - 1
    if pad > 0:
        Delta_n = np.concatenate([Delta_n, np.full(pad, np.nan)])

    # -------- plots --------
    if verbose:
        t_idx = prices.index[1:]
        fig, ax = plt.subplots(6, 1, figsize=(9, 6), sharex=True)

        ax[0].plot(t_idx, p_new_series, label="P(new regime)")
        ax[0].set_ylabel("Posterior"); ax[0].legend(); ax[0].grid(True)

        ax[1].plot(t_idx, S_arr, label="CUSUM $S_t$")
        ax[1].axhline(h_z, color="red", linestyle="--", label="Threshold $h_z$")
        ax[1].legend(); ax[1].grid(True)

        ax[2].step(prices.index, z_hat, where='post', label=r"$\hat{z}_t$")
        ax[2].legend(); ax[2].grid(True)

        ax[3].plot(prices.index, prices.values, label="Price")
        ax[3].legend(); ax[3].grid(True)

        ax[4].plot(t_idx, Delta_n, label="Δn estimate")
        ax[4].set_yscale("log"); ax[4].legend(); ax[4].grid(True)

        ax[5].plot(prices.index, y[:, 0] if y.ndim == 2 else y, label="Observed $y_t$")
        ax[5].legend(); ax[5].grid(True)

        plt.tight_layout()

    return z_hat


def cusum_overlay_zscore(prices, y, xhat, mdl, h_z, verbose=False):
    
    """
    Adaptive-scale CUSUM on latent-dynamics + transition log-likelihoods.

    Definitions:
        log P(xₜ | xₜ₋₁, regime): measures how well the latent dynamics explain the observed state.
        log P(regime | xₜ₋₁): measures how likely the regime is given the past state.
        The sum is the joint log-probability; a complete measure of model fit at time t

    sₜ is the log-ratio of how well the alternative regime explains the current state vs the current regime.
    
    At each time t:
        sₜ = log-likelihood ratio between alternative and current regime:
            sₜ = [log P(xₜ | xₜ₋₁, alt regime) + log P(alt regime | xₜ₋₁)]
                − [log P(xₜ | xₜ₋₁, curr regime) + log P(curr regime | xₜ₋₁)]
    
        zₜ = (sₜ − μₜ) / σₜ        # z-score of sₜ
        Sₜ = max(0, Sₜ₋₁ + zₜ)     # cumulative sum of z-scores

    If Sₜ > h_z, a regime change is triggered.

    Reasonable ranges for h_z. High sensitivity: 2-3. Low sensitivity: 5-7
    
    """
    from scipy.special import logsumexp
    import numpy as np
    import matplotlib.pyplot as plt

    if xhat.shape[0] <= 10: 
        # print(f'skipping CUSUM: time series too short {xhat.shape[0]}')
        return None  # Skip CUSUM if time series too short
        
    # Ensure shapes
    y = np.atleast_2d(y)
    xhat = np.atleast_2d(xhat)
    if y.shape[0] < y.shape[1]: y = y.T
    if xhat.shape[0] < xhat.shape[1]: xhat = xhat.T
    T, D = xhat.shape

    # Model parameters
    A = mdl.dynamics.As
    b = mdl.dynamics.bs
    sig2 = mdl.dynamics.sigmasq
    Rw  = mdl.transitions.Rs
    r   = mdl.transitions.r

    # Storage
    z_hat = np.zeros(T, dtype=int)
    z_hat[0] = 1  # start in regime 1
    S_arr, z_arr = [], []
    p_new_series = []

    # Online stats
    mu = 0.0
    m2 = 0.0
    S = 0.0

    for t in range(1, T):
        k_curr = z_hat[t - 1]
        k_alt = 1 - k_curr
        xp = xhat[t - 1]
        xn = xhat[t]

        eps_c = xn - (A[k_curr] @ xp + b[k_curr])
        eps_a = xn - (A[k_alt]  @ xp + b[k_alt])

        ll_c = -0.5 * np.sum(eps_c**2 / sig2[k_curr] + np.log(2*np.pi*sig2[k_curr]))
        ll_a = -0.5 * np.sum(eps_a**2 / sig2[k_alt ] + np.log(2*np.pi*sig2[k_alt ]))

        logits = Rw @ xp + r
        logits -= logsumexp(logits)

        s = (ll_a + logits[k_alt]) - (ll_c + logits[k_curr])

        # Posterior for plotting
        p_new = 1.0 / (1.0 + np.exp(ll_c - ll_a))
        p_new_series.append(p_new)

        # Online z-score
        δ = s - mu
        mu += δ / t
        m2 += δ * (s - mu)
        var = m2 / max(t - 1, 1)
        std = np.sqrt(var + 1e-12)

        z_score = (s - mu) / std
        S = max(0.0, S + z_score)

        S_arr.append(S)
        z_arr.append(z_score)

        if S > h_z:
            z_hat[t] = k_alt
            S = 0.0
            mu = 0.0
            m2 = 0.0
        else:
            z_hat[t] = k_curr

    # Δn estimate
    S_arr = np.asarray(S_arr)
    pos_inc = np.maximum(np.diff(np.concatenate(([0.0], S_arr))), 0.0)
    avg_inc = np.convolve(pos_inc, np.ones(10) / 10, mode="same")
    numer = np.maximum(h_z - S_arr, 0.0)
    denom = avg_inc
    Delta_n = np.full_like(S_arr, np.inf, dtype=float)
    np.divide(numer, denom, out=Delta_n, where=(denom > 0))
    pad = len(prices) - len(Delta_n) - 1
    if pad > 0:
        Delta_n = np.concatenate([Delta_n, np.full(pad, np.nan)])

    # -------- plots --------
    if verbose:
        t_idx = prices.index[1:]
        fig, ax = plt.subplots(6, 1, figsize=(9, 6), sharex=True)

        ax[0].plot(t_idx, p_new_series, label="P(new regime)")
        ax[0].set_ylabel("Posterior"); ax[0].legend(); ax[0].grid(True)

        ax[1].plot(t_idx, S_arr, label="CUSUM $S_t$")
        ax[1].axhline(h_z, color="red", linestyle="--", label="Threshold $h_z$")
        ax[1].legend(); ax[1].grid(True)

        ax[2].step(prices.index, z_hat, where='post', label=r"$\hat{z}_t$")
        ax[2].legend(); ax[2].grid(True)

        ax[3].plot(prices.index, prices.values, label="Price")
        ax[3].legend(); ax[3].grid(True)

        ax[4].plot(t_idx, Delta_n, label="Δn estimate")
        ax[4].set_yscale("log"); ax[4].legend(); ax[4].grid(True)

        ax[5].plot(prices.index, y[:, 0] if y.ndim == 2 else y, label="Observed $y_t$")
        ax[5].legend(); ax[5].grid(True)

        plt.tight_layout()

    return z_hat
    

# ---------------------------------------------------------------
# Evaluate rSLDS
# ---------------------------------------------------------------

def compute_smoothed_cpll(mdl, x_smooth, y, gamma_smooth):

    """
    Conditional predictive log-likelihood 
    True CPLL uses E[x_{t-1} | y_{1:t-1}] for one-step (online) forecasts.
    Smoothed CPLL uses E[x_t | y_{1:T}] and hence overestimates predictive power.
    Max CPLL: approx. practical upper bound on CPLL (Gaussian entropy limit)
    See: https://en.wikipedia.org/wiki/Differential_entropy
    """
    
    T, N = y.shape
    K = gamma_smooth.shape[1]
    logL = 0.0
    shared = mdl.emissions.Cs.shape[0] == 1

    eps = 1e-6  # Regularization for numerical stability
    var_floor = 1e-6  # Prevent zero variance

    for t in range(1, T):
        mu_bar = np.zeros(N)
        Sigma = np.zeros((N, N))

        for k in range(K):
            Ck = mdl.emissions.Cs[0] if shared else mdl.emissions.Cs[k]  # (N, D)
            dk = mdl.emissions.ds[0] if shared else mdl.emissions.ds[k]  # (N,)
            inv_eta_k = mdl.emissions.inv_etas[0] if shared else mdl.emissions.inv_etas[k]  # (N,)

            # Variance with clipping
            var_k = np.clip(1.0 / np.exp(inv_eta_k), var_floor, None)  # (N,)
            Rk = np.diagflat(var_k)  # (N, N)

            mu_k = Ck @ x_smooth[t - 1] + dk  # (N,)
            mu_bar += gamma_smooth[t - 1, k] * mu_k

        for k in range(K):
            Ck = mdl.emissions.Cs[0] if shared else mdl.emissions.Cs[k]
            dk = mdl.emissions.ds[0] if shared else mdl.emissions.ds[k]
            inv_eta_k = mdl.emissions.inv_etas[0] if shared else mdl.emissions.inv_etas[k]
            var_k = np.clip(1.0 / np.exp(inv_eta_k), var_floor, None)
            Rk = np.diagflat(var_k)

            mu_k = Ck @ x_smooth[t - 1] + dk
            delta_mu = mu_k - mu_bar
            Sigma += gamma_smooth[t - 1, k] * (Rk + np.outer(delta_mu, delta_mu))

        # Regularize Sigma to ensure it's invertible
        Sigma_reg = Sigma + eps * np.eye(N)

        # Safety check (optional)
        if not np.all(np.isfinite(Sigma_reg)):
            raise ValueError(f"Non-finite Sigma at t={t}")
        if np.linalg.cond(Sigma_reg) > 1e12:
            print(f"Warning: ill-conditioned Sigma at t={t}, cond = {np.linalg.cond(Sigma_reg):.2e}")

        mu_bar = mu_bar.flatten()
        resid = (y[t] - mu_bar).flatten()
        _, logdet = np.linalg.slogdet(Sigma_reg)
        logL += -0.5 * (N * np.log(2 * np.pi) + logdet + resid.T @ np.linalg.solve(Sigma_reg, resid))

    return logL


def evaluate_rSLDS_synthetic(prices, y, xhat, zhat, z_true, elbo, mdl, cpll, max_cpll, dt=1/252, label_invariant=True, display=False):
 
    T = len(zhat)
    time = np.arange(T)

    if label_invariant: 
        # Loss: label-invariant misclassification rate
        loss1 = np.mean(zhat != z_true)
        loss2 = np.mean((1 - zhat) != z_true)
        loss = min(loss1, loss2)
        
        # label-invariant accuracy score
        acc1 = np.mean(zhat == z_true)
        acc2 = np.mean((1 - zhat) == z_true)  
        flipped = False
        if acc2 > acc1:
            zhat_adj = 1 - zhat
            accuracy = acc2
            flipped = True
        else:
            zhat_adj = zhat
            accuracy = acc1

    else: 
        loss = np.mean(zhat != z_true)
        acc = np.mean(zhat == z_true)
        zhat_adj = zhat
        accuracy = acc

    ari = adjusted_rand_score(z_true, zhat_adj)

    conf_mat = confusion_matrix(z_true, zhat_adj, labels=[0, 1])
    prec = precision_score(z_true, zhat_adj, average='macro', zero_division=0)
    rec = recall_score(z_true, zhat_adj, average='macro', zero_division=0)
    denom = prec + rec
    f1_score = 2 * prec * rec / denom if denom > 1e-8 else 0.0

    # Changepoint detection error and lag
    true_cp = np.where(np.diff(z_true) != 0)[0]
    pred_cp = np.where(np.diff(zhat_adj) != 0)[0]

    matched = []
    unmatched_true = []
    unmatched_pred = list(pred_cp)

    for cp in true_cp:
        found = False
        for pcp in pred_cp:
            if abs(pcp - cp) <= 5:
                matched.append(abs(pcp - cp))
                if pcp in unmatched_pred:
                    unmatched_pred.remove(pcp)
                found = True
                break
        if not found:
            unmatched_true.append(cp)

    cp_error = np.mean(matched) if matched else None

    smoothing = np.mean([len(list(g)) for k, g in groupby(zhat_adj)])

    lag_list = []
    for t in range(1, T):
        if z_true[t] != z_true[t - 1]:
            true_regime = z_true[t]
            for lag in range(0, T - t):
                if zhat_adj[t + lag] == true_regime:
                    lag_list.append(lag)
                    break
    detection_lag_mean = np.mean(lag_list) if lag_list else np.nan

    # Stability margin
    # Stability margin measures how far each regime's dynamics matrix is from instability (spectral radius ≥ 1).
    # The spectral radius of a matrix is the largest absolute value among its eigenvalues.
    stability_margins, stability_decision = compute_stability_margin(mdl)
    
    # ELBO diagnostics
    # ELBO measures how well the estimated posterior q(z, x) fits the true posterior.
    # The closer q(z,x) is to the true posterior p(z,x|y), the smaller the KL divergence.
    # Bigger ELBO = smaller KL divergence = better q(z,x).
    if elbo is not None:
        elbo_start = elbo[0] # first ELBO of last batch
        elbo_end = elbo[-1] # last ELBO of last batch
        elbo_delta = elbo_end - elbo_start
    else:
        elbo_start = elbo_end = elbo_delta = np.nan

    # Mode usage
    mode_usage = dict(Counter(zhat_adj))

    # Ensure all arrays are the same length (last prediction can overshoot)
    min_len = min(len(y), len(zhat), len(prices))
    y = y[:min_len]
    zhat = zhat[:min_len]
    zhat_adj = zhat_adj[:min_len]
    z_true = z_true[:min_len]
    xhat = xhat[:min_len]
    prices = prices.iloc[:min_len] if hasattr(prices, "iloc") else prices[:min_len]
    time = np.arange(min_len)

    # Posterior marginals (gamma) for plotting
    mask = np.ones_like(y, dtype=bool)
    gamma, *_ = mdl.expected_states(xhat, y, mask=mask)  # shape (min_len, K)
    if label_invariant and flipped:
        gamma = gamma[:, ::-1]

    # Performance indices for the performance panel (moved from inference)
    _, _, _, strategy_index, bench_index, _ = compute_score(prices, zhat_adj, dt=dt)

    # Price vs reconstructed latent-implied price
    observed_price = np.asarray(prices).flatten()
    latent_price = observed_price[0] * np.exp(np.cumsum(xhat[:, 0]))

    # Summary metrics
    summary = {
        "loss": loss,
        "accuracy": accuracy,
        "ari": ari,
        "precision": prec,
        "recall": rec,
        "f1_score": f1_score,
        "changepoint_error": cp_error,
        "avg_inferred_regime_length": smoothing,
        "detection_lag_mean": detection_lag_mean,
        "detection_lag_all": lag_list,
        "elbo_start": elbo_start,
        "elbo_end": elbo_end,
        "elbo_delta": elbo_delta,
        "mode_usage": mode_usage,
        "confusion_matrix": conf_mat,
        "n_matched_changepoints": len(matched),
        "unmatched_true_changepoints": unmatched_true,
        "unmatched_pred_changepoints": unmatched_pred, 
        "stability_margins": stability_margins, 
        "stability_decision": stability_decision,
        "cpll": cpll,
        "max_cpll": max_cpll,
    }
    
    if display:

        print_rSLDS_matrices(mdl)

        # Plotting
        n_panels = 5 if elbo is None else 6   
        
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 1.5 * n_panels), sharex=False)

        # Panel 0: Observed data
        axes[0].plot(time, prices, label="Price")
        for cp in true_cp:
            axes[0].axvline(cp, color="gray", linestyle="--", alpha=0.7)
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        
        # Panel 1: Observed data
        axes[1].plot(time, y, label="Observed $y_t$")
        for cp in true_cp:
            axes[1].axvline(cp, color="gray", linestyle="--", alpha=0.7)
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Observed $y_t$")
        axes[1].legend()

        # Panel 2: Regime labels
        axes[2].step(time, z_true, where="mid", label="True", color="black", linewidth=1.5)
        axes[2].step(time, zhat_adj, where="mid", label="Inferred", color="red", alpha=0.7)  
        axes[2].set_xlabel("Time step")
        axes[2].set_ylabel("Regime")
        axes[2].legend()
        
        # Panel 3: Posterior
        for k in range(gamma.shape[1]):
            axes[3].plot(time, gamma[:, k], label=f"Regime {k}")
        axes[3].set_xlabel("Time step")
        axes[3].set_ylabel("Posterior")
        axes[3].legend()

        # Panel 4: Performance
        axes[4].plot(time, bench_index, label="Benchmark index")
        axes[4].plot(time, strategy_index, label="Strategy index")
        bmin, bmax = np.min(bench_index), np.max(bench_index)
        for regime in np.unique(zhat_adj):
            mask_reg = (zhat_adj == regime)
            axes[4].fill_between(time, bmin, bmax, where=mask_reg, alpha=0.2, label=f"Regime {regime}")
        axes[4].set_ylabel("Performance Index")
        axes[4].legend()
        
        # Panel 5: ELBO trace
        if elbo is not None:
            axes[5].plot(np.arange(len(elbo)), elbo, label="ELBO")
            axes[5].set_xlabel("EM iteration (all batches)")
            axes[5].set_ylabel("ELBO")
            axes[5].legend()

        plt.tight_layout()
        plt.show()

        # Summary table
        print("\n========== EVALUATION SUMMARY ==========")
        print("\nRegime Classification")
        print("---------------------")
        print(f"Misclassification rate:           {loss:.3f}")
        print(f"Accuracy:                         {accuracy:.3f}")
        print(f"Adjusted Rand Index (ARI):        {ari:.3f}")
        print(f"Precision (macro):                {prec:.3f}")
        print(f"Recall (macro):                   {rec:.3f}")
        print(f"F1 score (macro):                 {f1_score:.3f}")
        print("Confusion Matrix :")
        print(conf_mat)
    
        print("\nRegime Change Detection")
        print("------------------------")
        print(f"Changepoint error (mean matched): {cp_error}")
        print(f"Matched changepoints:             {len(matched)}")
        print(f"Unmatched true changepoints:      {unmatched_true}")
        print(f"Avg regime length:                {smoothing:.1f}")
        print(f"Detection lag mean:               {detection_lag_mean:.1f}")
        print(f"Detection lag samples:            {lag_list}")

        print("\nMode Usage")
        print("----------")
        print(mode_usage)

        if elbo is not None:
            print("\nELBO Diagnostics (last run)")
            print("----------------")
            print(f"ELBO start:                       {elbo_start:.2f}")
            print(f"ELBO end:                         {elbo_end:.2f}")
            print(f"ELBO delta:                       {elbo_delta:.2f}")
        
        print("\nSpectral Radius Test")
        print("----------")
        print(f"Stability margins:                {np.round(stability_margins, 2)}")
        print(f"Stability decision:               {stability_decision}")

        print("\nCPLL ")
        print("----------")
        print(f"CPLL (test):                      {cpll:.1f}")
        print(f"Approx. upper bound on CPLL:      {max_cpll:.1f}")
        
    return summary


def evaluate_rSLDS_actual(y, px, zhat, xhat, elbo, mdl, cpll, max_cpll, dt, display=True):

    # Mode usage + smoothing
    mode_usage = dict(Counter(zhat))
    smoothing = np.mean([len(list(g)) for _, g in groupby(zhat)])

    # ELBO (last run)
    if elbo is not None and len(elbo):
        elbo_start = elbo[-1][0]
        elbo_end   = elbo[-1][-1]
        elbo_delta = elbo_end - elbo_start
    else:
        elbo_start = elbo_end = elbo_delta = np.nan

    # Performance
    cagr_rel, cagr_strat, cagr_bench, strategy_index, bench_index, hyp_result = compute_score(px, zhat, dt)

    # Stability (exactly like synthetic)
    stability_margins, stability_decision = compute_stability_margin(mdl)

    summary = {
        "avg_inferred_regime_length": smoothing,
        "elbo_start (last run)": elbo_start,
        "elbo_end (last run)": elbo_end,
        "elbo_delta (last run)": elbo_delta,
        "mode_usage": mode_usage,
        "cagr_rel": cagr_rel,
        "cagr_strat": cagr_strat,
        "cagr_bench": cagr_bench,
        "stability_margins": stability_margins,
        "stability_decision": stability_decision,
        "cpll": cpll,
        "max_cpll": max_cpll,
    }

    if display:

        print_rSLDS_matrices(mdl)
        
        n_panels = 5 if elbo is not None else 4
        fig, axes = plt.subplots(n_panels, 1, figsize=(8, 1.5 * n_panels), sharex=False)

        # Panel 0: Performance
        axes[0].plot(px.index, bench_index, label="Benchmark index")
        axes[0].plot(px.index, strategy_index, label="Strategy index")
        for regime in np.unique(zhat):
            m = (zhat == regime)
            axes[0].fill_between(px.index, bench_index.min(), bench_index.max(), where=m, alpha=0.2, label=f"Regime {regime}")
        axes[0].set_ylabel("Performance Index"); axes[0].legend()

        # Panel 1: Observed price
        axes[1].plot(px.index, px, label="Observed Price"); axes[1].set_ylabel("Observed Price"); axes[1].legend()

        # Panel 2: Observed y_t
        axes[2].plot(px.index, y, label=r"Observed $y_t$"); axes[2].set_ylabel(r"Observed $y_t$"); axes[2].legend()

        # Panel 3: Inferred regimes
        axes[3].step(px.index, zhat, where="mid", label="Inferred", color="red", alpha=0.7)
        axes[3].set_ylabel("Regime"); axes[3].legend()

        # Panel 4: ELBO
        if elbo is not None:
            elbo_flat = np.concatenate(elbo)
            axes[4].plot(np.arange(len(elbo_flat)), elbo_flat, label="ELBO")
            axes[4].set_xlabel("EM iteration (all batches)"); axes[4].set_ylabel("ELBO"); axes[4].legend()

        plt.tight_layout(); plt.show()

        # Summary print (mirror synthetic style)
        print("\n========== EVALUATION SUMMARY ==========")
        if elbo is not None:
            print("\nELBO Diagnostics (last run)")
            print("----------------")
            print(f"ELBO start            : {elbo_start:.2f}")
            print(f"ELBO end              : {elbo_end:.2f}")
            print(f"ELBO delta            : {elbo_delta:.2f}")

        print("\nMode Usage")
        print("----------"); print(mode_usage)

        print("\nRegime Properties")
        print("-----------------"); print(f"Avg inferred regime length : {smoothing:.1f}")

        print("\nPerformance")
        print("-----------------")
        print(f"CAGR Relative     : {cagr_rel:.4f}")
        print(f"CAGR Strategy     : {cagr_strat:.4f}")
        print(f"CAGR Benchmark    : {cagr_bench:.4f}")
        print("\nHypothesis Test"); print(hyp_result)

        print("\nSpectral Radius Test")
        print("--------------------")
        print(f"Stability margins : {np.round(stability_margins, 2)}")
        print(f"Stability decision: {stability_decision}")

        print("\nCPLL")
        print("----")
        print(f"CPLL (stitched)   : {cpll:.1f}")
        print(f"Upper bound       : {max_cpll:.1f}")

    return summary

# ---------------------------------------------------------------
# Utils
# ---------------------------------------------------------------

def compute_stability_margin(mdl):
    """
    Compute per-regime stability margins and apply decision rule.
    
    Check that all regimes are dynamically stable by ensuring 
    spectral radius of each A_k is < 1.

    Decision: 'accept' if all margins > 0, else 'reject'
    """
    A_matrices = mdl.dynamics.As  # shape (K, D_latent, D_latent)
    margins = np.array([
        1.0 - max(abs(np.linalg.eigvals(A_k)))
        for A_k in A_matrices
    ])
    decision = 'accept' if np.all(margins > 0) else 'reject'
    return margins, decision


def compute_score(price, zhat, dt):

    # Label-invariant score

    n_obs   = len(price)          # not N
    n_years = n_obs * dt          # horizon in years

    def compute_cagr_rel(weights, bench_return, cagr_bench, n_years):
        strategy_return = bench_return * weights
        strategy_index = (1 + strategy_return).cumprod()
        cagr_strat = strategy_index.iloc[-1] ** (1 / n_years) - 1
        cagr_rel = (1 + cagr_strat) / (1 + cagr_bench) - 1
        return cagr_rel, cagr_strat, strategy_index, strategy_return

    regimes = np.unique(zhat)
    combos = [c for c in itertools.product([0, 1], repeat=len(regimes)) if 0 in c and 1 in c]

    bench_return = price.pct_change().fillna(0)
    bench_index = (1 + bench_return).cumprod()
    cagr_bench = bench_index.iloc[-1] ** (1 / n_years) - 1
        
    best_cagr_rel = -np.inf
    best_cagr_strat = -np.inf
    best_strategy_index = bench_index
    best_strategy_return = bench_return
 
    for c in combos:
        weight_lookup = dict(zip(regimes, c))
        w = np.array([weight_lookup[z] for z in zhat])
        cagr_rel, cagr_strat, strategy_index, strategy_return = \
        compute_cagr_rel(w, bench_return, cagr_bench, n_years)
        if cagr_rel > best_cagr_rel:
            best_cagr_rel = cagr_rel
            best_cagr_strat = cagr_strat
            best_strategy_index = strategy_index
            best_strategy_return = strategy_return

    # Null hypothesis test: excess return = 0 vs H1: > 0
    diff = best_strategy_return - bench_return
    mean_diff = np.mean(diff)
    n = len(diff)

    if n <= 1:
        # Not enough data → conservative
        t_stat = np.nan
        p_value = 1.0
    else:
        std_diff = np.std(diff, ddof=1) / np.sqrt(n)
        tiny = 1e-12
        if (not np.isfinite(std_diff)) or (std_diff < tiny):
            # Degenerate variance → fallback
            t_stat = np.inf if mean_diff > 0 else -np.inf
            p_value = 0.0 if mean_diff > 0 else 1.0
        else:
            t_stat = mean_diff / std_diff
            p_value = 1 - stats.t.cdf(t_stat, df=n - 1)  # one-sided

    alpha = 0.05
    hyp_result = (
        f"H0: CAGR Rel = 0 vs H1: CAGR Rel > 0. "
        f"P-val = {p_value:.4f}. "
        f"H0 {'rejected' if p_value < alpha else 'accepted'} at {1 - alpha:.2f}.")

    return best_cagr_rel, best_cagr_strat, cagr_bench, best_strategy_index, bench_index, hyp_result


# ---------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------

def print_rSLDS_matrices(model, restricted=False, dt=None, latent_names=None):

    print("\n========== FITTED MODEL ==========\n")

    def pmat(arr, label, semantic_key):
        # Shape strings chosen to MATCH the OLD function exactly:
        semantic_shape_str = {
            "Cs": "(K, N, D)",
            "ds": "(K, N)",        # IMPORTANT: match OLD (was (N, K) in the newer draft)
            "inv_etas": "(K, N)",
            "As": "(K, D, D)",
            "bs": "(K, D)",
            "sigmasq": "(K, D)",
            "Rs": "(K, D)",
            "r": "(K,)",
        }[semantic_key]
        # IMPORTANT: two spaces after label to match OLD
        print(f"{label}  {semantic_shape_str} = {arr.shape}")
        K = arr.shape[0]
        if arr.ndim == 3:
            rows = arr.shape[1]
            cols = arr.shape[2]
            for r in range(rows):
                line = ""
                for k in range(K):
                    row_str = " ".join(f"{arr[k, r, c]: .4f}" for c in range(cols))
                    # IMPORTANT: four spaces between blocks to match OLD
                    line += f"[{row_str}]    "
                print(line)
        elif arr.ndim == 2:
            for k in range(K):
                row_str = " ".join(f"{arr[k, c]: .4f}" for c in range(arr.shape[1]))
                # IMPORTANT: four spaces and no trailing newline until after loop, to match OLD
                print(f"[{row_str}]", end="    ")
            print()
        elif arr.ndim == 1:
            row_str = " ".join(f"{arr[k]: .4f}" for k in range(K))
            print(f"[{row_str}]")
        print()

    # Matrix dumps (identical to OLD)
    pmat(model.emissions.Cs, "Cs", "Cs")
    pmat(model.emissions.ds, "ds", "ds")
    pmat(model.emissions.inv_etas, "inv_etas", "inv_etas")
    pmat(model.dynamics.As, "As", "As")
    pmat(model.dynamics.bs, "bs", "bs")
    pmat(model.dynamics.sigmasq, "sigmasq", "sigmasq")
    pmat(model.transitions.Rs, "Rs", "Rs")
    pmat(model.transitions.r, "r", "r")

    # Regime-type classification (identical to OLD)
    print("Dynamics: xₜ = a⋅xₜ₋₁ + b + ε")
    print()
    print("Inferred Regime Types:")
    K = model.dynamics.As.shape[0]
    D = model.dynamics.As.shape[1]

    for k in range(K):
        a_vals = np.array([model.dynamics.As[k, d, d] for d in range(D)])
        b_vals = np.array([model.dynamics.bs[k, d] for d in range(D)])

        a_zero = np.allclose(a_vals, 0, atol=1e-2)
        a_one = np.allclose(a_vals, 1.0, atol=1e-2)
        a_ar1 = np.all(np.abs(a_vals) < 1.0)
        b_zero = np.allclose(b_vals, 0.0, atol=1e-4)

        if a_zero and b_zero:
            eqn = "xₜ = ε        (white_noise)"
        elif a_zero and not b_zero:
            eqn = "xₜ = b + ε    (iid_drift)"
        elif a_ar1 and b_zero:
            eqn = "xₜ = a xₜ₋₁ + ε    (ar1)"
        elif a_ar1 and not b_zero:
            eqn = "xₜ = a xₜ₋₁ + b + ε    (ar1_drift)"
        elif a_one and b_zero:
            eqn = "xₜ = xₜ₋₁ + ε    (rw)"
        elif a_one and not b_zero:
            eqn = "xₜ = xₜ₋₁ + b + ε    (rw_drift)"
        else:
            eqn = "Unclassified"

        print(f"Regime {k}:  {eqn}")
    print("\n")

    # If restricted==False, we exactly matched OLD; stop here.
    if not restricted:
        return

    # Otherwise, add interpretable AR(1) parameters for restricted models.
    As = model.dynamics.As  # (K, D, D), assumed diagonal
    bs = model.dynamics.bs  # (K, D)
    sig2 = model.dynamics.sigmasq  # (K, D)
    K, D, _ = As.shape
    eps = 1e-8

    print()
    print("-------------------------------------------------")
    print("Interpretable AR(1) parameters (restricted=True) ")
    print("-------------------------------------------------")
    if latent_names is None:
        latent_names = [f"latent{d}" for d in range(D)]

    for k in range(K):
        print(f"Regime {k}:")
        for d in range(D):
            rho = As[k, d, d]
            b = bs[k, d]
            mu = (b / (1.0 - rho)) if abs(1.0 - rho) > eps else float("nan")
            s2 = sig2[k, d]
            if rho > 0 and abs(rho) < 1:
                hl_steps = float(np.log(2.0) / abs(np.log(rho)))
            else:
                hl_steps = float("inf")
            if (dt is not None and np.isfinite(hl_steps)):
                hl_years = (hl_steps * dt)
            else:
                hl_years = float("nan")
            var_stat = (s2 / (1.0 - rho**2)) if abs(rho) < 1 else float("nan")
            name = latent_names[d] if d < len(latent_names) else f"latent{d}"
            print(
                f"  {name:>10s}: rho={rho: .4f}, mu={mu: .6f}, sigma2={s2: .6e}, "
                f"half_life_steps={hl_steps: .2f}, half_life_years={hl_years: .4f}, "
                f"var_stat={var_stat: .6e}"
            )
        print()


def get_rSLDS_params(model, include_values=False):

    # Example: params_dict = get_model_params(mdl, include_values=False)

    seen = set()
    out = {}

    def visit(obj, prefix=""):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, (int, float, str, bool, type(None))):
            if include_values:
                print(f"{prefix}: scalar\n{obj}\n")
            else:
                print(f"{prefix}: scalar")
            out[prefix] = obj

        elif isinstance(obj, np.ndarray):
            if include_values:
                print(f"{prefix}: {obj.shape}\n{obj}\n")
            else:
                print(f"{prefix}: {obj.shape}")
            out[prefix] = obj

        elif isinstance(obj, (list, tuple)):
            print(f"{prefix}: list[{len(obj)}]")
            for i, item in enumerate(obj):
                visit(item, f"{prefix}[{i}]")

        elif isinstance(obj, dict):
            print(f"{prefix}: dict[{len(obj)}]")
            for k, v in obj.items():
                visit(v, f"{prefix}.{k}")

        else:
            for attr in dir(obj):
                if attr.startswith("_") or attr == "params":
                    continue
                try:
                    val = getattr(obj, attr)
                    visit(val, f"{prefix}.{attr}" if prefix else attr)
                except Exception:
                    continue

    def collect_params(component, label):
        if hasattr(component, "parameters"):
            for k, v in component.parameters.items():
                key = f"{label}.{k}"
                if include_values:
                    print(f"{key}: {v.shape}\n{v}\n")
                else:
                    print(f"{key}: {v.shape}")
                out[key] = v

    # Traverse full model
    visit(model)

    # Add labeled parameter blocks
    collect_params(model.transitions, "transitions")
    collect_params(model.dynamics, "dynamics")
    collect_params(model.emissions, "emissions")
    collect_params(model.init_state_distn, "init_state_distn")

    return out


def get_rSLDS_args():
    
    import inspect
    
    sig = inspect.signature(ssm.SLDS.__init__)
    print("SLDS.__init__ parameters:\n")
    for name, param in sig.parameters.items():
        print(f"{name} : {param}")
    
    def print_constructor_params(obj, name, depth=0):
        indent = "  " * depth
        cls = obj.__class__
        print(f"\n{indent}{name} ({cls.__name__}) constructor:")
        sig = inspect.signature(cls.__init__)
        for pname, param in sig.parameters.items():
            if pname != "self":
                print(f"{indent}  {pname}: {param}")
    
            # Check if param is a class itself (e.g. a submodule), recurse
            try:
                attr = getattr(obj, pname)
                if hasattr(attr, '__class__') and not isinstance(attr, (int, float, str, list, tuple, dict, type(None))):
                    print_constructor_params(attr, f"{name}.{pname}", depth + 1)
            except Exception:
                continue  # skip inaccessible attributes
    
    
    # Instantiate a model with valid components
    mdl = ssm.SLDS(N=2, K=2, D=2,
                   transitions="recurrent_only",
                   dynamics="diagonal_gaussian",
                   emissions="gaussian",
                   single_subspace=True)
    
    # Recurse into submodules
    print_constructor_params(mdl, "SLDS")
    print_constructor_params(mdl.transitions, "Transitions")
    print_constructor_params(mdl.dynamics, "Dynamics")
    print_constructor_params(mdl.emissions, "Emissions")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def compute_required_w(sigma, dt, confidence=0.95):

    # Note: compute_required_w is analytical, compute_R_r_from_latents is data-driven.
    
    """
    Compute w such that the softmax transition assigns the correct regime 
    with given probability for a 1-σ move (i.e., signal magnitude = σ√dt).

    Parameters:
    - sigma: std of the signal (e.g., Δlog(price) or Δlog(EPS))
    - dt: time increment
    - confidence: desired classification confidence (e.g., 0.95)

    Returns:why 
    - required value of w
    """
    logit_margin = np.log(confidence / (1.0 - confidence))
    typical = sigma * np.sqrt(dt)
    w = logit_margin / (2.0 * typical)
    return w


def compute_R_r_from_latents(x, z, K=2):
    """
    Fit softmax transition parameters (R, r) from latent data and regime labels.
    """

    from sklearn.linear_model import LogisticRegression

    # Guard against trivial case: all z same
    if len(np.unique(z)) < 2:
        # Force balanced fake labels for logistic fit
        z[:len(z)//2] = 0
        z[len(z)//2:] = 1

    clf = LogisticRegression(fit_intercept=True)
    clf.fit(x, z)
    w = clf.coef_[0]
    b = clf.intercept_[0]

    Rs = np.stack([w, -w])       # shape (K=2, D)
    r = np.array([b, -b])        # shape (K=2,)

    return Rs, r

def compute_R_r_from_soft_probs(x, Ez, K):
    """
    x  : (T-1, D) latent states x_{t-1}
    Ez : (T-1, K) posterior P(z_t = k), we use Ez[:,1] as soft label
    K  : number of regimes (assume 2)
    """
    assert K == 2, "Only supports binary case"
    x = np.asarray(x)
    y_soft = Ez[:, 1]  # target: P(z_t = 1)

    T, D = x.shape
    X_aug = np.hstack([x, np.ones((T, 1))])  # shape (T, D+1)

    def loss(w_aug):
        logits = X_aug @ w_aug       # shape (T,)
        probs = 1 / (1 + np.exp(-logits))
        eps = 1e-8
        ce = - y_soft * np.log(probs + eps) - (1 - y_soft) * np.log(1 - probs + eps)
        return np.mean(ce)

    w0 = np.zeros(D + 1)
    res = minimize(loss, w0, method='L-BFGS-B')
    w_aug = res.x

    w, b = w_aug[:-1], w_aug[-1]
    Rs = np.stack([w, -w])
    r = np.array([b, -b])
    return Rs, r

    
def fit_kmeans(y, Ks=[2, 3, 4], display=False):
    """
    Optimal clusters: fit KMeans on (drift, diffusion) features from y, return best K and cluster stats.
    """
    dy = np.diff(y, axis=0).ravel()
    features = np.column_stack([dy, dy ** 2])
    features = StandardScaler().fit_transform(features)

    fit_stats = []
    cluster_stats = {}
    cluster_labels = {}

    # Compute cluster fits and stats
    for K in Ks:
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(features)
        labels = km.labels_
        sil = silhouette_score(features, labels)
        fit_stats.append({"K": K, "silhouette": sil})
        cluster_labels[K] = (labels, km.cluster_centers_)

        stats = []
        for k in range(K):
            dy_k = dy[labels == k]
            abs_dy_k = np.abs(dy_k)
            stats.append({
                "cluster": k,
                "mean_drift": np.mean(dy_k),
                "std_drift": np.std(dy_k),
                "mean_diffusion": np.mean(abs_dy_k),
                "std_diffusion": np.std(abs_dy_k)
            })
        cluster_stats[K] = pd.DataFrame(stats)

    # Select best K
    best_K = max(fit_stats, key=lambda x: x['silhouette'])['K']

    if display:
        fig, axes = plt.subplots(len(Ks), 1, figsize=(6, 3 * len(Ks)), sharex=True, sharey=True)
        if len(Ks) == 1:
            axes = [axes]
        for ax, K in zip(axes, Ks):
            labels, centers = cluster_labels[K]
            ax.scatter(features[:, 0], features[:, 1], c=labels, cmap='tab10', s=6, alpha=0.8)
            ax.scatter(centers[:, 0], centers[:, 1], marker='x', c='black', s=80, lw=2)
            sil = next(f['silhouette'] for f in fit_stats if f['K'] == K)
            ax.set_title(f'KMeans K={K} silhouette={sil:.3f}')
            ax.set_xlabel('std Δy (drift)')
            ax.set_ylabel('std (Δy)^2 (diffusion)')
        plt.tight_layout()
        plt.show()

        # Print fit stats
        df_fit = pd.DataFrame(fit_stats).set_index('K')
        print("\nModel fit stats (silhouette):")
        print(df_fit.to_string())

        # Print cluster stats
        for K in Ks:
            print(f"\nCluster stats for K={K}:")
            print(cluster_stats[K].to_string(index=False))

        print(f"\nBest model: K={best_K}")

    return best_K, cluster_stats


def infer_params_from_model(model, mu_true, sigma_true, sigma_diff_true, dt=1/252):
    
    """
    Return a DataFrame with estimated and true drift/volatility (continuous time)
    for each regime of a 1-D two-regime rSLDS.

    Parameters
    ----------
    model : rSLDS object (fit with K=2, D=N=1)
    mu_true : float                      # absolute drift used in generator
    sigma_true : float                   # centre volatility in generator
    sigma_diff_true : float              # sigma_up − sigma_down in generator
    dt : float                           # data time step (default 1/252)
    """
    # estimated continuous-time drift & vol -------------------------------
    A  = model.dynamics.As[:, 0, 0]
    b  = model.dynamics.bs[:, 0]
    s2 = model.dynamics.sigmasq[:, 0]

    mu_c_est  = b / (1 - A) / dt
    sig_c_est = np.sqrt(s2 / (1 - A**2)) / np.sqrt(dt)

    # true values as used in generate_synthetic_data ----------------------
    drift_actual = np.array([-mu_true,  mu_true])              # regime 0/1
    vol_actual   = np.array([sigma_true + sigma_diff_true/2,   # regime 0
                             sigma_true - sigma_diff_true/2])  # regime 1

    # pack and return -----------------------------------------------------
    df = pd.DataFrame({
        "drift_est"   : mu_c_est,
        "vol_est"     : sig_c_est,
        "drift_actual": drift_actual,
        "vol_actual"  : vol_actual,
    })
    df = df.set_index(pd.Index([0, 1], name="regime"))
    print(df)

