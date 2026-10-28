"""Auxiliary reference for scipy.stats.chi at df = 2e8 (a = df/2 = 1e8).

Reuses refs/chi_ref.py, whose lower incomplete gamma switches to the fully
converged Kummer series for a >= 1e7 (mp.gammainc does not converge there).
Grid: x = sqrt(df + k*sqrt(2 df)), i.e. y = x**2/2 = a + k*sqrt(a).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "refs"))
import numpy as np
import chi_ref as R

DIST = "chi"
DPS = 100
PARAMS = [(2e8,)]
METHODS = ["pdf", "logpdf", "cdf", "sf"]
REPLACE_GRID = True
REFS = R.REFS


def x_points(df):
    return np.array([np.sqrt(df + k * np.sqrt(2 * df)) for k in
                     (-30, -20, -15, -10, -7, -5, -4.6, -4.4, -3, 0, 3, 5, 10)])


def q_points(df):
    return np.array([1e-8, 1e-20])
