"""Supplementary run for beta ppf/isf: same references as refs/beta_ref.py,
extra shape sets where Boost's ibeta_inv fails and a dense q grid."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "refs"))
from beta_ref import REFS, DPS, x_points  # noqa: F401,E402

DIST = "beta"
PARAMS = [(0.5, 2.0), (0.5, 3.0), (2.0, 10.0), (5.0, 2.0), (2.0, 1e6), (20.0, 50.0)]


def q_points(a, b):
    return np.concatenate([10.0 ** np.arange(-300, -2, 1.0),
                           1 - 10.0 ** np.arange(-15, -2, 1.0)])
