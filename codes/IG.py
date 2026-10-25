# ------------------------------------------------------------------------------
# PARAMETERS OF THE NUMERICAL STUDY (as specified)
# ------------------------------------------------------------------------------
N_VALUES = [10, 50, 100, 500, 1000, 10000]        # sample size n per population
K_VALUES = [5, 10, 50, 100, 250, 500]                  # number of populations k
RATIOS = [1.0, 1.5, 2.0, 5.0, 7.5, 10.0]     # rho = mu_(k) / mu_(k-1)
MC_SIMULATIONS = 10_000                                 # Monte Carlo replications M
C_PARAM = 3.0                                           # extended Jeffreys index c (> 2/3)
LAMBDA_PARAM = 5.0                                     # known common IG shape lambda
LARGEST_MEAN = 5.0                                      # mu_(k), fixed

# ------------------------------------------------------------------------------
# CHOICES THE REPORT DOES NOT FIX (stated here so they are visible)
# ------------------------------------------------------------------------------
# The remaining k-2 true means are distinct, equally spaced in
# [LOWER_MEAN_FRACTION * mu_(k-1), mu_(k-1)); never all equal, also at rho = 1.
LOWER_MEAN_FRACTION = 0.25
# EB hyper-parameters (a, b) are searched in this closed box (report: a>0, b>0).
EB_A_RANGE = (1e-3, 1e5)
EB_B_RANGE = (1e-3, 1e5)
EB_MAX_FAILURE_FRACTION = 0.01      # abort a configuration if more EB fits fail than this

# ------------------------------------------------------------------------------
# HARDWARE / REPRODUCIBILITY / OUTPUT
# ------------------------------------------------------------------------------
NUM_WORKERS = 4                     # Intel i5 quad-core
RANDOM_SEED = 1
OUTPUT_DIR = "outputs"

# ------------------------------------------------------------------------------
# NUMERICAL CONTROLS (accuracy checks raise IntegrationError; nothing falls back)
# ------------------------------------------------------------------------------
WINDOW_TAIL_UPPER = 1e-12           # mass of the max-sample-mean law beyond the window (right)
WINDOW_TAIL_LOWER = 1e-18           # mass of the max-sample-mean law beyond the window (left)
EXACT_NODE_SEQUENCE = (128, 256, 512, 1024, 2048)   # Gauss-Legendre nodes, doubled until converged
EXACT_REL_TOL, EXACT_ABS_TOL, MASS_TOL = 1e-8, 1e-11, 1e-7
BC_NODES = 256                      # nodes for the plug-in bias integral (Eq. 3.5)
BC_CHECK_ROWS = 4                   # replicates per batch re-integrated with 2*BC_NODES
BC_REL_TOL = 1e-5
GB_NODES, GB_WINDOW_WIDTH = 64, 14.0                # J_nu integrals (Eq. 3.16)
EB_NODES_AT_9_SIGMA = 32            # nodes for M_nu integrals (Eq. 3.25) at window +-9 sigma
EB_WINDOW_WIDTHS = (9.0, 18.0, 36.0)
EB_BISECT_ITERS = 30
EB_MAX_DOUBLINGS = 4                # node-doubling check for M_nu: up to 4 doublings (16x nodes) before raising
EB_U_LIMIT = 60.0                   # |log theta| limit of the M_nu window
EB_MAXITER, EB_GRAD_TOL = 500, 1e-4
EB_FTOL, EB_GTOL = 1e-12, 1e-6      # L-BFGS-B stopping tolerances (tighter than EB_GRAD_TOL so the post-check is consistent)
EDGE_LOG_TOL = 34.5                 # integrand at window edge must be < e^-34.5 of its maximum
MC_Z_LIMIT = 6.0                    # MC vs exact check for MLE and GB (|z| above this raises)
ELEMENTS_PER_BATCH = 2_000_000      # array size per vectorised batch (memory control)
MC_BATCH_MAX = 500
BC_ABS_TOL = 1e-5

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import math
import multiprocessing as mp
import queue
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, gammaln, log_ndtr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
EST_NAMES = {
    "MLE": "Naive MLE",
    "BC": "Bias-Corrected (BC)",
    "GB": "Generalized Bayes (GB)",
    "EB": "Empirical Bayes (EB)",
}


class IntegrationError(RuntimeError):
    """A numerical integral failed one of its own accuracy checks."""


@lru_cache(maxsize=None)
def gl_rule(q):
    """Gauss-Legendre nodes and weights on [-1, 1]."""
    return np.polynomial.legendre.leggauss(q)


def fp_strict():
    """Floating-point errors raise (underflow to 0 is harmless and allowed)."""
    return np.errstate(divide="raise", invalid="raise", over="raise", under="ignore")


# ==============================================================================
# 1. TRUE MEANS
# ==============================================================================
def construct_true_means(k, rho):
    """mu_(k) = LARGEST_MEAN, mu_(k-1) = LARGEST_MEAN / rho, other k-2 means distinct and below."""
    mu_second = LARGEST_MEAN / rho
    others = np.linspace(LOWER_MEAN_FRACTION * mu_second, mu_second, k - 2, endpoint=False)
    return np.concatenate([others, [mu_second, LARGEST_MEAN]])


# ==============================================================================
# 2. INVERSE GAUSSIAN: log-density, log-CDF (Eqs. 2.1-2.2), exact sampler
# ==============================================================================
def log_f_ig(x, mu, shape):
    return 0.5 * (np.log(shape) - np.log(2.0 * np.pi) - 3.0 * np.log(x)) - shape * (x - mu) ** 2 / (
        2.0 * mu ** 2 * x
    )


def log_F_ig(x, mu, shape):
    """log of Phi(sqrt(s/x)(x/mu-1)) + exp(2s/mu) Phi(-sqrt(s/x)(x/mu+1)), s = shape."""
    r = np.sqrt(shape / x)
    xm = x / mu
    return np.logaddexp(log_ndtr(r * (xm - 1.0)), 2.0 * shape / mu + log_ndtr(-r * (xm + 1.0)))


def sample_ig(rng, mu, shape, size):
    """Michael-Schucany-Haas exact sampler for IG(mu, shape)."""
    v = rng.standard_normal(size) ** 2
    y = mu + mu ** 2 * v / (2.0 * shape) - mu / (2.0 * shape) * np.sqrt(4.0 * mu * shape * v + mu ** 2 * v ** 2)
    u = rng.random(size)
    return np.where(u <= mu / (mu + y), y, mu ** 2 / y)


def max_window(means, n_lam):
    """
    Window [t_lo, t_hi] (per row of `means`, shape (B, k)) for the law of max_j X_j,
    X_j ~ IG(means_j, n_lam).  IG(mu, s) is stochastically increasing in mu and
    P(X > t) <= exp(-s (t-mu)^2 / (2 mu^2 t)) (Chernoff, t > mu), same bound for P(X < t) (t < mu).
    With mu = largest mean: P(max > t_hi) <= WINDOW_TAIL_UPPER, P(max < t_lo) <= WINDOW_TAIL_LOWER.
    """
    k = means.shape[1]
    mu = means.max(axis=1)

    def root(L, sign):          # solves n_lam (t - mu)^2 = 2 mu^2 L t
        return mu + (mu ** 2 * L + sign * mu * np.sqrt(2.0 * n_lam * mu * L + (mu * L) ** 2)) / n_lam

    t_hi = root(math.log(k) - math.log(WINDOW_TAIL_UPPER), +1.0)
    t_lo = root(-math.log(WINDOW_TAIL_LOWER), -1.0)
    return t_lo, t_hi


def selection_kernel(t, means, n_lam):
    """
    kernel_i(t) = f_IG(t; mu_i, n_lam) * prod_{j != i} F_IG(t; mu_j, n_lam)   (Lemma 2.2).
    t: (B, Q), means: (B, k)  ->  (B, Q, k).
    """
    t3, m3 = t[:, :, None], means[:, None, :]
    log_F = log_F_ig(t3, m3, n_lam)
    return np.exp(log_f_ig(t3, m3, n_lam) + log_F.sum(axis=2, keepdims=True) - log_F)


# ==============================================================================
# 3. GENERALIZED BAYES  g_c(x) = J_{3c/2-1}(x) / J_{3c/2}(x)   (Eq. 3.16)
# ==============================================================================
def log_J(nu, x, n_lam, q):
    """log J_nu(x), J_nu(x) = int_0^inf theta^nu exp(-(n_lam x/2)(theta-1/x)^2) dtheta, nu > 0."""
    a = n_lam * x
    mode = 0.5 / x + np.sqrt(0.25 / x ** 2 + nu / a)
    sigma = 1.0 / np.sqrt(a + nu / mode ** 2)
    lo = np.maximum(0.0, mode - GB_WINDOW_WIDTH * sigma)
    hi = mode + GB_WINDOW_WIDTH * sigma

    def h(th):
        aa, xx = (a[:, None], x[:, None]) if th.ndim == 2 else (a, x)
        return nu * np.log(th) - 0.5 * aa * (th - 1.0 / xx) ** 2

    h_mode = h(mode)
    safe_lo = np.where(lo > 0.0, lo, mode)          # at theta = 0 the integrand is 0 (nu > 0)
    edge_ok = (h(hi) - h_mode < -EDGE_LOG_TOL) & ((lo <= 0.0) | (h(safe_lo) - h_mode < -EDGE_LOG_TOL))
    if not edge_ok.all():
        raise IntegrationError("J_nu window edge check failed")
    nodes, weights = gl_rule(q)
    half = 0.5 * (hi - lo)
    th = 0.5 * (hi + lo)[:, None] + half[:, None] * nodes[None, :]
    integral = (half[:, None] * weights[None, :] * np.exp(h(th) - h_mode[:, None])).sum(axis=1)
    return h_mode + np.log(integral)


def g_c(x, n_lam, c):
    """Generalized Bayes rule under RSELF; requires c > 2/3 (Remark 3.3).  Checked with 2x nodes."""
    if c <= 2.0 / 3.0:
        raise ValueError("Generalized Bayes requires c > 2/3.")
    x = np.asarray(x, dtype=float)
    nu1, nu2 = 1.5 * c - 1.0, 1.5 * c
    coarse = np.exp(log_J(nu1, x, n_lam, GB_NODES) - log_J(nu2, x, n_lam, GB_NODES))
    fine = np.exp(log_J(nu1, x, n_lam, 2 * GB_NODES) - log_J(nu2, x, n_lam, 2 * GB_NODES))
    if not np.allclose(coarse, fine, rtol=1e-10, atol=0.0):
        raise IntegrationError("g_c failed its node-doubling check")
    return fine


# ==============================================================================
# 4. EXACT BIAS / RELATIVE BIAS / RISK OF MLE AND GB  (Lemma 2.2, Eqs. 3.2-3.4, 3.17-3.19)
# ==============================================================================
def _exact_pass(mu, t_lo, t_hi, n_lam, c, q):
    nodes, weights = gl_rule(q)
    half = 0.5 * (t_hi - t_lo)
    t = 0.5 * (t_hi + t_lo) + half * nodes
    wk = (half * weights)[:, None] * selection_kernel(t[None, :], mu[None, :], n_lam)[0]    # (q, k)
    out = {"mass": float(wk.sum())}
    for name, g in (("MLE", t), ("GB", g_c(t, n_lam, c))):
        rel = g[:, None] / mu[None, :] - 1.0
        out[f"{name}_Bias"] = float((wk * (g[:, None] - mu[None, :])).sum())
        out[f"{name}_Rel_Bias"] = float((wk * rel).sum())
        out[f"{name}_Risk"] = float((wk * rel ** 2).sum())
    return out


def exact_mle_and_gb(mu, n_lam, c):
    """Sum over populations of the 1-D integrals; nodes are doubled until two rules agree."""
    t_lo, t_hi = max_window(mu[None, :], n_lam)
    previous = None
    for q in EXACT_NODE_SEQUENCE:
        current = _exact_pass(mu, float(t_lo[0]), float(t_hi[0]), n_lam, c, q)
        if previous is not None and all(
            abs(current[key] - previous[key]) <= EXACT_ABS_TOL + EXACT_REL_TOL * abs(current[key])
            for key in current
        ):
            if abs(current["mass"] - 1.0) > MASS_TOL:
                raise IntegrationError(f"density of the maximum integrates to {current['mass']}, not 1")
            return current
        previous = current
    raise IntegrationError("exact integrals did not converge")


# ==============================================================================
# 5. BIAS-CORRECTED ESTIMATOR  mu_BC = X_S - B_hat_MLE(Xbar)   (Eqs. 3.5, 3.8)
# ==============================================================================
def _plugin_bias_pass(xbar, t_lo, t_hi, n_lam, q):
    nodes, weights = gl_rule(q)
    half = 0.5 * (t_hi - t_lo)
    t = 0.5 * (t_hi + t_lo)[:, None] + half[:, None] * nodes[None, :]                   # (B, q)
    w = half[:, None] * weights[None, :]
    kern = selection_kernel(t, xbar, n_lam)                                              # (B, q, k)
    integrand = ((t[:, :, None] - xbar[:, None, :]) * kern).sum(axis=2)
    return (w * integrand).sum(axis=1)


def plugin_bias_mle(xbar, n_lam):
    """B_hat_MLE(Xbar) = sum_i int (t - Xbar_i) f(t; Xbar_i) prod_{j!=i} F(t; Xbar_j) dt, per row of xbar."""
    t_lo, t_hi = max_window(xbar, n_lam)
    b_hat = _plugin_bias_pass(xbar, t_lo, t_hi, n_lam, BC_NODES)
    r = slice(0, BC_CHECK_ROWS)
    fine = _plugin_bias_pass(xbar[r], t_lo[r], t_hi[r], n_lam, 2 * BC_NODES)
    if not np.allclose(b_hat[r], fine, rtol=BC_REL_TOL, atol=BC_ABS_TOL):
        raise IntegrationError("B_hat_MLE failed its node-doubling check")
    if np.any(b_hat < -BC_ABS_TOL):
        raise IntegrationError("B_hat_MLE < 0 beyond quadrature tolerance contradicts Proposition 3.1")
    return b_hat


# ==============================================================================
# 6. EMPIRICAL BAYES  (Eqs. 3.25-3.30)
#    M_nu(x; b) = int_0^inf theta^nu exp(-b/theta + n_lam theta - n_lam x theta^2 / 2) dtheta
#    With theta = e^u the integrand is exp(h(u)),
#       h(u) = (nu+1) u - b e^-u + n_lam e^u - (n_lam x / 2) e^{2u}.
#    Stationary points solve the cubic p(s) = -A s^3 + n_lam s^2 + (nu+1) s + b = 0, s = e^u, A = n_lam x.
#    All local maxima (at most two) are located exactly and the window covers them.
# ==============================================================================
def _bisect_log(fn, lo, hi):
    """Root of fn in [lo, hi] (fn(lo) > 0 > fn(hi)), bisection in log-space."""
    for _ in range(EB_BISECT_ITERS):
        mid = np.sqrt(lo * hi)
        up = fn(mid) > 0.0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return np.sqrt(lo * hi)


def _eb_modes(A, x, c, b, n_lam):
    """Log-locations, curvatures (-h'') and validity of the (up to two) local maxima of h(u)."""

    def p(s):
        return ((-A * s + n_lam) * s + c) * s + b

    # Cauchy bounds: every positive root lies in [s_lo, s_hi]; p > 0 below, p < 0 above.
    s_lo = 0.5 / (1.0 + np.maximum(np.maximum(abs(c), n_lam), A) / b)
    s_hi = 2.0 * (1.0 + np.maximum(np.maximum(1.0 / x, abs(c) / A), b / A))
    if c < 0.0:
        disc = 4.0 * n_lam ** 2 + 12.0 * A * c
        two = disc > 0.0                                     # p' has two positive roots s1 < s2
        root = np.sqrt(np.where(two, disc, 0.0))
        s1 = (2.0 * n_lam - root) / (6.0 * A)
        s2 = (2.0 * n_lam + root) / (6.0 * A)
    else:
        two = np.zeros(x.shape, dtype=bool)
        s1 = s2 = s_hi
    left_hi = np.where(two, s1, s_hi)
    right_lo = np.where(two, s2, s_hi)
    left_ok = p(left_hi) < 0.0                               # root of p where p goes + to -  (a maximum)
    right_ok = two & (p(right_lo) > 0.0)                     # second maximum (bimodal case)
    s = np.stack([_bisect_log(p, s_lo, left_hi), _bisect_log(p, right_lo, s_hi)], axis=1)
    valid = np.stack([left_ok, right_ok], axis=1)
    curv = b / s - n_lam * s + 2.0 * A[:, None] * s ** 2
    if not valid.any(axis=1).all():
        raise IntegrationError("no local maximum found for M_nu integrand")
    if np.any(valid & (curv <= 0.0)):
        raise IntegrationError("degenerate stationary point in M_nu integrand")
    return np.log(s), curv, valid


def _eb_pass(x, c, b, n_lam, width, q):
    """log M_nu, E[log theta], E[1/theta] under the density prop. to the M_nu integrand."""
    A = n_lam * x
    u_mode, curv, valid = _eb_modes(A, x, c, b, n_lam)
    sigma = 1.0 / np.sqrt(np.where(valid, curv, 1.0))
    lo = np.maximum(np.where(valid, u_mode - width * sigma, np.inf).min(axis=1), -EB_U_LIMIT)
    hi = np.minimum(np.where(valid, u_mode + width * sigma, -np.inf).max(axis=1), EB_U_LIMIT)

    def h(u):
        Au = A[:, None] if u.ndim == 2 else A
        return c * u - b * np.exp(-u) + n_lam * np.exp(u) - 0.5 * Au * np.exp(2.0 * u)

    h_ref = np.where(valid, h(u_mode), -np.inf).max(axis=1)
    nodes, weights = gl_rule(q)
    half = 0.5 * (hi - lo)
    u = 0.5 * (hi + lo)[:, None] + half[:, None] * nodes[None, :]
    w = half[:, None] * weights[None, :] * np.exp(h(u) - h_ref[:, None])
    Z = w.sum(axis=1)
    edges_ok = bool(np.all((h(lo) - h_ref < -EDGE_LOG_TOL) & (h(hi) - h_ref < -EDGE_LOG_TOL)))
    return h_ref + np.log(Z), (w * u).sum(axis=1) / Z, (w * np.exp(-u)).sum(axis=1) / Z, edges_ok


def eb_log_M(x, nu, b, n_lam, check=False):
    """
    log M_nu(x; b) and the two posterior moments; window widened until its edges are negligible.
    With check=True the node count is doubled until two successive rules agree (as in exact_mle_and_gb);
    the finer of the two agreeing rules is returned.  If they never agree, IntegrationError is raised.
    """
    c = nu + 1.0
    for width in EB_WINDOW_WIDTHS:
        q = int(round(EB_NODES_AT_9_SIGMA * width / 9.0))
        logM, Eu, Einv, ok = _eb_pass(x, c, b, n_lam, width, q)
        if ok:
            break
    else:
        raise IntegrationError("M_nu window edge check failed")
    if not check:
        return logM, Eu, Einv
    for _ in range(EB_MAX_DOUBLINGS):
        q *= 2
        logM2, Eu2, Einv2, ok2 = _eb_pass(x, c, b, n_lam, width, q)
        if (
            ok2
            and np.allclose(logM, logM2, rtol=0.0, atol=1e-8)
            and np.allclose(Eu, Eu2, rtol=1e-7, atol=1e-8)
            and np.allclose(Einv, Einv2, rtol=1e-7, atol=1e-8)
        ):
            return logM2, Eu2, Einv2
        logM, Eu, Einv = logM2, Eu2, Einv2
    raise IntegrationError("M_nu failed its node-doubling check")


def _eb_objective(theta, x, n_lam, check=False):
    """-(1/k) l(a, b) of Eq. 3.28 and its gradient w.r.t. (log a, log b)."""
    a, b = math.exp(theta[0]), math.exp(theta[1])
    logM, Eu, Einv = eb_log_M(x, -a - 1.0, b, n_lam, check)
    f = -(a * theta[1] - gammaln(a) + logM.mean())
    grad = np.array([-a * (theta[1] - digamma(a) - Eu.mean()), -(a - b * Einv.mean())])
    return f, grad


def _method_of_moments_start(m1, s2, n_lam):
    """
    Starting value of a from Section 3.4.2:  E = a/b = m1,  Var = a(a+1)(a+2)/(n_lam b^3) + a/b^2 = s2,
    i.e. (c3 - s2) a^2 + (3 c3 + m1^2) a + 2 c3 = 0 with c3 = m1^3 / n_lam.
    """
    c3 = m1 ** 3 / n_lam
    if s2 > c3:
        B = 3.0 * c3 + m1 ** 2
        return (B + math.sqrt(B * B + 8.0 * c3 * (s2 - c3))) / (2.0 * (s2 - c3)), True
    return math.sqrt(EB_A_RANGE[0] * EB_A_RANGE[1]), False      # no solution: counted in the CSV


def fit_eb_hyperparameters(xbar, n_lam):
    """(a_hat, b_hat) = argmax l(a, b)  (Eq. 3.29), with diagnostics that are counted, not hidden."""
    a0, mom_ok = _method_of_moments_start(float(xbar.mean()), float(xbar.var(ddof=1)), n_lam)
    lows = np.log([EB_A_RANGE[0], EB_B_RANGE[0]])
    highs = np.log([EB_A_RANGE[1], EB_B_RANGE[1]])
    x0 = np.clip(np.log([a0, a0 / float(xbar.mean())]), lows, highs)
    res = minimize(
        _eb_objective, x0, args=(xbar, n_lam), jac=True, method="L-BFGS-B",
        bounds=list(zip(lows, highs)), options={"maxiter": EB_MAXITER, "ftol": EB_FTOL, "gtol": EB_GTOL},
    )
    _, grad = _eb_objective(res.x, xbar, n_lam, check=True)       # gradient from a checked integral
    on_bound = (np.abs(res.x - lows) < 1e-6) | (np.abs(res.x - highs) < 1e-6)
    stationary = bool(np.all(on_bound | (np.abs(grad) < EB_GRAD_TOL)))
    flags = {
        "nonconverged": not (bool(res.success) and stationary),
        "boundary": bool(on_bound.any()),
        "mom_undefined": not mom_ok,
    }
    return math.exp(res.x[0]), math.exp(res.x[1]), flags


def eb_estimator(x_s, a_hat, b_hat, n_lam):
    """g_{a,b}(X_S) = M_{-a}(X_S; b) / M_{-a+1}(X_S; b)   (Eq. 3.30)."""
    x = np.array([x_s], dtype=float)
    log_num = eb_log_M(x, -a_hat, b_hat, n_lam, check=True)[0]
    log_den = eb_log_M(x, -a_hat + 1.0, b_hat, n_lam, check=True)[0]
    return float(np.exp(log_num[0] - log_den[0]))


# ==============================================================================
# 7. MONTE CARLO (Eq. 2.15) FOR BC AND EB, WITH AN MC CHECK OF THE EXACT MLE / GB INTEGRALS
# ==============================================================================
_PROGRESS_QUEUE = None


def _init_worker(progress_queue):
    global _PROGRESS_QUEUE
    _PROGRESS_QUEUE = progress_queue


def report_progress(units):
    if _PROGRESS_QUEUE is not None:
        _PROGRESS_QUEUE.put(units)


def run_monte_carlo(mu, n_lam, rng):
    k, M = mu.size, MC_SIMULATIONS
    batch = int(max(1, min(MC_BATCH_MAX, ELEMENTS_PER_BATCH // (BC_NODES * k))))
    est = {name: np.full(M, np.nan) for name in EST_NAMES}
    mu_s = np.empty(M)
    eb_counts = {"failed": 0, "nonconverged": 0, "boundary": 0, "mom_undefined": 0}
    done = 0
    while done < M:
        m = min(batch, M - done)
        sl = slice(done, done + m)
        xbar = sample_ig(rng, mu[None, :], n_lam, (m, k))              # Xbar_i ~ IG(mu_i, n lambda)
        s_idx = xbar.argmax(axis=1)                                     # Eq. 2.3
        x_s = xbar[np.arange(m), s_idx]
        mu_s[sl] = mu[s_idx]
        est["MLE"][sl] = x_s
        est["GB"][sl] = g_c(x_s, n_lam, C_PARAM)
        est["BC"][sl] = x_s - plugin_bias_mle(xbar, n_lam)
        for j in range(m):
            try:
                a_hat, b_hat, flags = fit_eb_hyperparameters(xbar[j], n_lam)
                est["EB"][done + j] = eb_estimator(x_s[j], a_hat, b_hat, n_lam)
            except (IntegrationError, FloatingPointError):
                eb_counts["failed"] += 1
                continue
            for key in ("nonconverged", "boundary", "mom_undefined"):
                eb_counts[key] += int(flags[key])
        done += m
        report_progress(m * k)
    if eb_counts["failed"] > EB_MAX_FAILURE_FRACTION * M:
        raise RuntimeError(f"EB failed for {eb_counts['failed']} of {M} replicates")
    return est, mu_s, eb_counts


def summarize(est, mu_s):
    """Bias, |bias|, relative bias, risk (Eqs. 2.8-2.11) and their MC standard errors (finite estimates only)."""
    ok = np.isfinite(est)
    d = est[ok] - mu_s[ok]
    r = est[ok] / mu_s[ok] - 1.0
    m = int(ok.sum())
    se = lambda v: float(v.std(ddof=1) / math.sqrt(m))
    return {
        "Bias": float(d.mean()), "Abs_Bias": abs(float(d.mean())),
        "Rel_Bias": float(r.mean()), "Risk": float((r ** 2).mean()),
        "Bias_MC_SE": se(d), "Rel_Bias_MC_SE": se(r), "Risk_MC_SE": se(r ** 2),
        "n_used": m,
    }


def check_mc_against_exact(exact, est, mu_s):
    """MC estimates of MLE and GB must agree with the exact integrals (|z| <= MC_Z_LIMIT)."""
    worst = {}
    for name in ("MLE", "GB"):
        d = est[name] - mu_s
        r = est[name] / mu_s - 1.0
        for metric, v in (("Bias", d), ("Rel_Bias", r), ("Risk", r ** 2)):
            z = abs(float(v.mean()) - exact[f"{name}_{metric}"]) / (v.std(ddof=1) / math.sqrt(v.size))
            worst[name] = max(worst.get(name, 0.0), float(z))
    if max(worst.values()) > MC_Z_LIMIT:
        raise RuntimeError(f"Monte Carlo disagrees with the exact integrals: max |z| = {worst}")
    return worst


# ==============================================================================
# 8. ONE CONFIGURATION (n, k, rho), CHECKPOINTS
# ==============================================================================
def parameter_signature():
    keys = dict(
        C_PARAM=C_PARAM, LAMBDA_PARAM=LAMBDA_PARAM, LARGEST_MEAN=LARGEST_MEAN,
        MC_SIMULATIONS=MC_SIMULATIONS, RANDOM_SEED=RANDOM_SEED, LOWER_MEAN_FRACTION=LOWER_MEAN_FRACTION,
        EB_A_RANGE=EB_A_RANGE, EB_B_RANGE=EB_B_RANGE,
        EB_MAXITER=EB_MAXITER, EB_GRAD_TOL=EB_GRAD_TOL, EB_FTOL=EB_FTOL, EB_GTOL=EB_GTOL,
        BC_NODES=BC_NODES, GB_NODES=GB_NODES,
        EB_NODES_AT_9_SIGMA=EB_NODES_AT_9_SIGMA, WINDOW=(WINDOW_TAIL_UPPER, WINDOW_TAIL_LOWER),
    )
    return hashlib.sha256(json.dumps(keys, sort_keys=True).encode()).hexdigest()


def checkpoint_path(task):
    n, k, rho = task
    return os.path.join(CHECKPOINT_DIR, f"n{n}_k{k}_rho{rho:g}.json")


def load_checkpoint(task):
    path = checkpoint_path(task)
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        content = json.load(handle)
    if content["signature"] != parameter_signature():
        # Ignore stale checkpoints when parameters change; the configuration will be recomputed.
        return None
    return content["records"]


def build_records(n, k, rho, exact, mc, eb_counts, z_check):
    base = {"n": n, "k": k, "ratio": rho, "mu_max": LARGEST_MEAN, "mu_second": LARGEST_MEAN / rho,
            "c": C_PARAM, "lambda": LAMBDA_PARAM}
    rows = []
    for key in ("MLE", "GB"):                                       # exact (Lemma 2.2)
        bias = exact[f"{key}_Bias"]
        rows.append({**base, "Estimator": EST_NAMES[key], "Bias": bias, "Abs_Bias": abs(bias),
                     "Rel_Bias": exact[f"{key}_Rel_Bias"], "Risk": exact[f"{key}_Risk"],
                     "MC_check_max_abs_z": z_check[key]})
    for key in ("BC", "EB"):                                        # Monte Carlo (Eq. 2.15)
        row = {**base, "Estimator": EST_NAMES[key], **mc[key]}
        if key == "EB":
            row.update({"EB_failed_fits": eb_counts["failed"], "EB_nonconverged": eb_counts["nonconverged"],
                        "EB_boundary_hits": eb_counts["boundary"], "EB_mom_undefined": eb_counts["mom_undefined"]})
        rows.append(row)
    return rows


def evaluate_configuration(task):
    n, k, rho = task
    n_lam = n * LAMBDA_PARAM                                         # Xbar_i ~ IG(mu_i, n lambda)
    mu = construct_true_means(k, rho)
    with fp_strict():
        exact = exact_mle_and_gb(mu, n_lam, C_PARAM)
        rng = np.random.default_rng([RANDOM_SEED, n, k, RATIOS.index(rho)])
        est, mu_s, eb_counts = run_monte_carlo(mu, n_lam, rng)
        z_check = check_mc_against_exact(exact, est, mu_s)
    mc = {key: summarize(est[key], mu_s) for key in ("BC", "EB")}
    records = build_records(n, k, rho, exact, mc, eb_counts, z_check)
    path = checkpoint_path(task)
    with open(path + ".tmp", "w") as handle:
        json.dump({"signature": parameter_signature(), "records": records}, handle)
    os.replace(path + ".tmp", path)
    return task


def _worker(task):
    try:
        return evaluate_configuration(task)
    except Exception as exc:
        raise RuntimeError(f"configuration {task} failed: {exc}\n{traceback.format_exc()}") from None


# ==============================================================================
# 9. PLOTS (PDF only; journal style; Okabe-Ito colour-blind-safe palette)
# ==============================================================================
STYLES = {
    "Naive MLE": {"color": "#D55E00", "marker": "o", "ls": "-"},
    "Bias-Corrected (BC)": {"color": "#0072B2", "marker": "s", "ls": "--"},
    "Generalized Bayes (GB)": {"color": "#009E73", "marker": "^", "ls": "-."},
    "Empirical Bayes (EB)": {"color": "#CC79A7", "marker": "D", "ls": ":"},
}
METRICS = [
    ("Risk", r"Risk $R(\mu,\delta)$", "i_Risk_vs_Ratio.pdf", "v_Surface_Risk_by_k_and_n.pdf"),
    ("Bias", r"Bias $B(\mu,\delta)$", "ii_Bias_vs_Ratio.pdf", "v_Surface_Bias_by_k_and_n.pdf"),
    ("Abs_Bias", r"Absolute bias $|B(\mu,\delta)|$", "iii_Absolute_Bias_vs_Ratio.pdf", "v_Surface_AbsBias_by_k_and_n.pdf"),
    ("Rel_Bias", r"Relative bias $\beta(\mu,\delta)$", "iv_Relative_Bias_vs_Ratio.pdf", "v_Surface_RelBias_by_k_and_n.pdf"),
]


def configure_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 10,
        "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "figure.dpi": 150,
    })


def make_plots(df):
    """(i)-(iv): ratio on x, metric on y, one PDF page per (n, k).  (v): 3-D surfaces over (k, n), one page per ratio."""
    configure_style()
    total = len(METRICS) * (len(N_VALUES) * len(K_VALUES) + len(RATIOS))
    bar = tqdm(total=total, desc="Plots (PDF)", unit="page")
    for col, label, ratio_file, surface_file in METRICS:
        with PdfPages(os.path.join(OUTPUT_DIR, ratio_file)) as pdf:
            for n in N_VALUES:
                for k in K_VALUES:
                    fig, ax = plt.subplots(figsize=(6.4, 4.4))
                    sub = df[(df["n"] == n) & (df["k"] == k)]
                    for name, st in STYLES.items():
                        s = sub[sub["Estimator"] == name].sort_values("ratio")
                        ax.plot(s["ratio"], s[col], label=name, color=st["color"], marker=st["marker"],
                                linestyle=st["ls"], linewidth=1.6, markersize=5)
                    if col in ("Bias", "Rel_Bias"):
                        ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
                    ax.set_xticks(RATIOS)
                    ax.set_xlabel(r"Ratio $\rho=\mu_{(k)}/\mu_{(k-1)}$")
                    ax.set_ylabel(label)
                    ax.set_title(rf"$n={n},\ k={k},\ \lambda={LAMBDA_PARAM:g},\ c={C_PARAM:g},\ \mu_{{(k)}}={LARGEST_MEAN:g}$")
                    ax.legend(frameon=False)
                    fig.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                    bar.update(1)
        log_k, log_n = np.log10(K_VALUES), np.log10(N_VALUES)
        K_mesh, N_mesh = np.meshgrid(log_k, log_n)
        with PdfPages(os.path.join(OUTPUT_DIR, surface_file)) as pdf:
            for rho in RATIOS:
                fig = plt.figure(figsize=(11.0, 8.6))
                for idx, name in enumerate(STYLES, start=1):
                    ax = fig.add_subplot(2, 2, idx, projection="3d")
                    s = df[(df["ratio"] == rho) & (df["Estimator"] == name)]
                    Z = s.pivot(index="n", columns="k", values=col).reindex(index=N_VALUES, columns=K_VALUES).to_numpy(dtype=float)
                    surf = ax.plot_surface(K_mesh, N_mesh, Z, cmap="viridis", edgecolor="black", linewidth=0.2)
                    ax.set_title(name, fontsize=10)
                    ax.set_xlabel(r"$k$", labelpad=6)
                    ax.set_ylabel(r"$n$", labelpad=6)
                    ax.set_zlabel(label, labelpad=6)
                    ax.set_xticks(log_k)
                    ax.set_xticklabels([str(v) for v in K_VALUES], fontsize=7)
                    ax.set_yticks(log_n)
                    ax.set_yticklabels([str(v) for v in N_VALUES], fontsize=7)
                    ax.view_init(elev=28, azim=-135)
                    fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.1)
                fig.suptitle(rf"{label} over $(k,n)$ at $\rho={rho:g}$, $\lambda={LAMBDA_PARAM:g}$, $c={C_PARAM:g}$, $\mu_{{(k)}}={LARGEST_MEAN:g}$")
                fig.tight_layout(rect=[0, 0, 1, 0.95])
                pdf.savefig(fig)
                plt.close(fig)
                bar.update(1)
    bar.close()


# ==============================================================================
# 10. MAIN
# ==============================================================================
def _drain(progress_queue, bar):
    while True:
        try:
            bar.update(progress_queue.get_nowait())
        except queue.Empty:
            return


def main():
    if C_PARAM <= 2.0 / 3.0:
        raise ValueError("c must exceed 2/3 (Remark 3.3).")
    if min(RATIOS) < 1.0 or min(K_VALUES) < 2:
        raise ValueError("Need ratios >= 1 and k >= 2.")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    start = time.time()
    tasks = [(n, k, rho) for n in N_VALUES for k in K_VALUES for rho in RATIOS]
    pending = [t for t in tasks if load_checkpoint(t) is None]
    pending.sort(key=lambda t: (-t[1], -t[0]))                      # heaviest first: better load balance
    print(f"{len(tasks)} configurations, {len(tasks) - len(pending)} already finished, {len(pending)} to run "
          f"on {NUM_WORKERS} workers (M = {MC_SIMULATIONS} replications each).")

    failures = []
    if pending:
        total_units = sum(MC_SIMULATIONS * k for (_, k, _) in pending)
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            with ProcessPoolExecutor(max_workers=NUM_WORKERS, initializer=_init_worker, initargs=(progress_queue,)) as pool:
                with tqdm(total=total_units, desc="Simulating (ETA)", unit="rep*k", smoothing=0.05,
                          bar_format="{l_bar}{bar}| {percentage:5.1f}% [{elapsed}<{remaining}] {postfix}") as bar:
                    waiting = {pool.submit(_worker, t) for t in pending}
                    finished_count = 0
                    while waiting:
                        finished, waiting = wait(waiting, timeout=1.0, return_when=FIRST_COMPLETED)
                        _drain(progress_queue, bar)
                        for fut in finished:
                            try:
                                fut.result()
                                finished_count += 1
                            except Exception as exc:
                                failures.append(str(exc))
                        bar.set_postfix_str(f"configs {finished_count}/{len(pending)}")
                    _drain(progress_queue, bar)
    if failures:
        with open(os.path.join(OUTPUT_DIR, "failures.txt"), "w") as handle:
            handle.write("\n\n".join(failures))
        raise RuntimeError(f"{len(failures)} configuration(s) failed (see {OUTPUT_DIR}/failures.txt). "
                           "Finished configurations are saved; fix the cause and re-run to resume.")

    records = []
    for t in tqdm(tasks, desc="Collecting results", unit="cfg"):
        records.extend(load_checkpoint(t))
    df = pd.DataFrame(records).sort_values(["n", "k", "ratio", "Estimator"]).reset_index(drop=True)
    df.to_csv(os.path.join(OUTPUT_DIR, "results_all_estimators.csv"), index=False, float_format="%.10g")
    df.pivot_table(index=["n", "k", "ratio"], columns="Estimator",
                   values=["Risk", "Bias", "Abs_Bias", "Rel_Bias"]).to_csv(
        os.path.join(OUTPUT_DIR, "results_pivot.csv"), float_format="%.10g")

    eb = df[df["Estimator"] == EST_NAMES["EB"]]
    print("EB diagnostics (counted over all replications): "
          f"failed fits = {int(eb['EB_failed_fits'].sum())}, non-converged = {int(eb['EB_nonconverged'].sum())}, "
          f"hyper-parameters at a bound = {int(eb['EB_boundary_hits'].sum())}, "
          f"MoM start undefined = {int(eb['EB_mom_undefined'].sum())}")
    make_plots(df)
    print(f"Done in {(time.time() - start) / 3600.0:.2f} h. CSV tables and PDF plots are in '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
