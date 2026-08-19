"""
pymaslow.dirichlet
==================

Dirichlet-distribution models for compositional need profiles.

At any time of day, an individual's *need profile* is a composition: a
5-vector of relative frequencies over the Maslow hierarchy levels that sums
to one, i.e. a point on the 4-simplex. Such data are naturally modeled by a
Dirichlet distribution

.. math::

    p(\\mathbf{y}) \\sim \\mathcal{D}(\\alpha_1, \\dots, \\alpha_K)
    = \\frac{\\Gamma(\\sum_k \\alpha_k)}{\\prod_k \\Gamma(\\alpha_k)}
      \\prod_k y_k^{\\alpha_k - 1},

whose concentration parameters :math:`\\alpha` are estimated from observed
frequency vectors by maximum likelihood.

The estimation routines are a Python port of Thomas P. Minka's fastfit
MATLAB code, described in his note *"Estimating a Dirichlet distribution"*
(see http://research.microsoft.com/en-us/um/people/minka/), adapted from
``codes/dirichlet/dirichlet.py`` of the companion research repository.
"""

from __future__ import annotations

import sys

import numpy as np
from numpy import asanyarray, exp, log, ones, vstack
from numpy.linalg import norm
from scipy import stats
from scipy.special import gammaln, polygamma, psi

__all__ = [
    "NotConvergingError",
    "fit_temporal_profile",
    "loglikelihood",
    "meanprecision",
    "mle",
    "pdf",
    "test",
]

_MAXINT = sys.maxsize

_EULER = -1 * psi(1)  # Euler-Mascheroni constant


class NotConvergingError(Exception):
    """Raised when a successive-approximation method fails to converge."""


def test(D1, D2, method="meanprecision", maxiter=None):
    """Likelihood-ratio test for difference between two proportion datasets.

    Parameters
    ----------
    D1 : (N1, K) array
    D2 : (N2, K) array
        Observed proportions; rows must sum to 1. ``K`` is the number of
        categories (``K = 5`` for the Maslow hierarchy).
    method : {'meanprecision', 'fixedpoint'}
        MLE algorithm used to fit the Dirichlet models.
    maxiter : int or None
        Maximum number of iterations (default: unbounded).

    Returns
    -------
    D : float
        Test statistic, ``-2 * log`` of the likelihood ratio.
    p : float
        Chi-squared p-value of the test.
    a0, a1, a2 : (K,) arrays
        MLE parameters fitted to ``D1`` and ``D2`` pooled, to ``D1``, and
        to ``D2``, respectively.
    """
    K1 = D1.shape[1]
    K2 = D2.shape[1]
    if K1 != K2:
        raise ValueError("D1 and D2 must have the same number of columns")

    D0 = vstack((D1, D2))
    a0 = mle(D0, method=method, maxiter=maxiter)
    a1 = mle(D1, method=method, maxiter=maxiter)
    a2 = mle(D2, method=method, maxiter=maxiter)

    D = 2 * (loglikelihood(D1, a1) + loglikelihood(D2, a2) - loglikelihood(D0, a0))
    return (D, stats.chi2.sf(D, K1), a0, a1, a2)


def pdf(alphas):
    """Return the Dirichlet probability density function for parameters ``alphas``.

    Parameters
    ----------
    alphas : (K,) array
        Concentration parameters of the distribution.

    Returns
    -------
    callable
        Function mapping an ``(N, K)`` array of simplex points to an ``(N,)``
        array of densities.
    """
    alphap = alphas - 1
    c = np.exp(gammaln(alphas.sum()) - gammaln(alphas).sum())

    def dirichlet(xs):
        return c * (xs**alphap).prod(axis=1)

    return dirichlet


def meanprecision(a):
    """Mean and precision (concentration) of a Dirichlet distribution.

    Parameters
    ----------
    a : (K,) array
        Concentration parameters.

    Returns
    -------
    mean : (K,) array
        Expected simplex value, ``a / a.sum()``.
    precision : float
        Total concentration, ``a.sum()``.
    """
    s = a.sum()
    m = a / s
    return (m, s)


def loglikelihood(D, a):
    """Log-likelihood ``log p(D | a)`` of data under a Dirichlet model.

    Parameters
    ----------
    D : (N, K) array
        Observed proportions (rows sum to 1).
    a : (K,) array
        Concentration parameters.

    Returns
    -------
    float
    """
    N = D.shape[0]
    logp = log(D).mean(axis=0)
    return N * (gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * logp).sum())


def mle(D, tol=1e-7, method="meanprecision", maxiter=None):
    """Maximum-likelihood Dirichlet parameters for observed proportions.

    Iteratively computes the concentration parameters ``a`` maximizing
    ``log p(D | a)``.

    Parameters
    ----------
    D : (N, K) array
        Observed proportions (rows sum to 1).
    tol : float
        Convergence tolerance on the change in log-likelihood.
    method : {'meanprecision', 'fixedpoint'}
        Estimation algorithm. ``'meanprecision'`` (default) alternates
        updates of the mean and the precision and is faster;
        ``'fixedpoint'`` uses Minka's fixed-point iteration.
    maxiter : int or None
        Maximum number of iterations (default: unbounded).

    Returns
    -------
    (K,) array
        Maximum-likelihood concentration parameters.
    """
    if method == "meanprecision":
        return _meanprecision(D, tol=tol, maxiter=maxiter)
    return _fixedpoint(D, tol=tol, maxiter=maxiter)


def fit_temporal_profile(freq_vectors, tol=1e-7, method="meanprecision", maxiter=None):
    """Fit a Dirichlet model to observed need-profile frequency vectors.

    Convenience wrapper around :func:`mle` for temporal Maslow profiles:
    each row of ``freq_vectors`` is the relative frequency of the five
    hierarchy levels within one time bin.

    Parameters
    ----------
    freq_vectors : (N, 5) array
        Relative frequencies of the hierarchy levels per time bin; rows
        must be strictly positive and sum to 1.
    tol, method, maxiter
        Passed through to :func:`mle`.

    Returns
    -------
    alphas : (5,) array
        MLE concentration parameters.
    mean : (5,) array
        Mean need profile (expected simplex value).
    precision : float
        Total concentration.
    """
    freq_vectors = asanyarray(freq_vectors, dtype=float)
    alphas = mle(freq_vectors, tol=tol, method=method, maxiter=maxiter)
    mean, precision = meanprecision(alphas)
    return alphas, mean, precision


# =============================================================================
# Internal estimation routines (Minka fastfit port)
# =============================================================================


def _fixedpoint(D, tol=1e-7, maxiter=None):
    """Simple fixed-point iteration for the Dirichlet MLE."""
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)

    a1 = a0
    if maxiter is None:
        maxiter = _MAXINT
    for _i in range(maxiter):
        a1 = _ipsi(psi(a0.sum()) + logp)
        # Much faster convergence than with `norm(a1 - a0) < tol`
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    raise NotConvergingError(
        f"Failed to converge after {maxiter} iterations, values are {a1}."
    )


def _meanprecision(D, tol=1e-7, maxiter=None):
    """Mean/precision alternating method for the Dirichlet MLE."""
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)
    s0 = a0.sum()
    if s0 < 0:
        a0 = a0 / s0
        s0 = 1
    elif s0 == 0:
        a0 = ones(a0.shape) / len(a0)
        s0 = 1

    a1 = a0
    if maxiter is None:
        maxiter = _MAXINT
    for _i in range(maxiter):
        a1 = _fit_s(D, a0, logp, tol=tol)
        a1 = _fit_m(D, a1, logp, tol=tol)
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    raise NotConvergingError(
        f"Failed to converge after {maxiter} iterations, values are {a1}."
    )


def _fit_s(D, a0, logp, tol=1e-7, maxiter=1000):
    """Update the precision with the mean held fixed (Newton's method)."""
    s1 = a0.sum()
    m = a0 / s1
    mlogp = (m * logp).sum()
    for _i in range(maxiter):
        s0 = s1
        g = psi(s1) - (m * psi(s1 * m)).sum() + mlogp
        h = _trigamma(s1) - ((m**2) * _trigamma(s1 * m)).sum()

        if g + s1 * h < 0:
            s1 = 1 / (1 / s0 + g / h / (s0**2))
        if s1 <= 0:
            s1 = s0 * exp(-g / (s0 * h + g))  # Newton on log s
        if s1 <= 0:
            s1 = 1 / (1 / s0 + g / ((s0**2) * h + 2 * s0 * g))  # Newton on 1/s
        if s1 <= 0:
            s1 = s0 - g / h  # Newton
        if s1 <= 0:
            raise NotConvergingError(f"Unable to update s from {s0}")

        a = s1 * m
        if abs(s1 - s0) < tol:
            return a

    raise NotConvergingError(
        f"Failed to converge after {maxiter} iterations, s is {s1}"
    )


def _fit_m(D, a0, logp, tol=1e-7, maxiter=1000):
    """Update the mean with the precision held fixed."""
    s = a0.sum()
    for _i in range(maxiter):
        m = a0 / s
        a1 = _ipsi(logp + (m * (psi(a0) - logp)).sum())
        a1 = a1 / a1.sum() * s

        if norm(a1 - a0) < tol:
            return a1
        a0 = a1

    raise NotConvergingError(f"Failed to converge after {maxiter} iterations, s is {s}")


def _init_a(D):
    """Method-of-moments initial guess for the Dirichlet parameters."""
    E = D.mean(axis=0)
    E2 = (D**2).mean(axis=0)
    return ((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E


def _ipsi(y, tol=1.48e-9, maxiter=10):
    """Inverse digamma (psi) via Newton's method, mapping R -> (0, inf)."""
    y = asanyarray(y, dtype="float")
    x0 = np.piecewise(
        y,
        [y >= -2.22, y < -2.22],
        [(lambda x: exp(x) + 0.5), (lambda x: -1 / (x + _EULER))],
    )
    x1 = x0
    for _i in range(maxiter):
        x1 = x0 - (psi(x0) - y) / _trigamma(x0)
        if norm(x1 - x0) < tol:
            return x1
        x0 = x1
    raise NotConvergingError(
        f"Failed to converge after {maxiter} iterations, value is {x1}"
    )


def _trigamma(x):
    return polygamma(1, x)
