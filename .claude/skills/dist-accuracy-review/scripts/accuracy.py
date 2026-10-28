"""Measure the accuracy of a scipy.stats continuous distribution against mpmath.

The distribution's public methods (loc=0, scale=1) are evaluated on grids that
reach far into both tails and compared with high-precision mpmath references.

Usage
-----
    python accuracy.py REF_FILE                      # full run, summary + flagged points
    python accuracy.py REF_FILE --verbose            # plus worst case per parameter set
    python accuracy.py REF_FILE --methods sf,logsf   # only some methods
    python accuracy.py REF_FILE --params "2,0.5;3,1" # override parameter sets
    python accuracy.py REF_FILE --at logpdf 1e-10 2 0.5   # one point (reproducer)
    python accuracy.py REF_FILE --candidate cand.py:obj   # also measure an alternative
    python accuracy.py REF_FILE --plot err.png --json points.json

Reference module (REF_FILE)
---------------------------
A Python file that defines

    DIST    = "gamma"                      # name in scipy.stats
    PARAMS  = [(0.5,), (3.0,), (1e3,)]     # shape parameter tuples
    REFS    = {"pdf": f, "cdf": g, ...}    # method -> f(arg, *shapes) -> mpmath number

and optionally

    METHODS = [...]                 # default: implemented methods that have a reference
    def x_points(*shapes): ...      # extra x values (added to the default grid)
    def q_points(*shapes): ...      # extra q values for ppf/isf
    REPLACE_GRID = False            # True: x_points/q_points replace the defaults
    DPS = 400                       # working precision (digits)

Reference functions receive mpmath numbers (exact copies of the float inputs)
and must return mpmath numbers. ``from accuracy import invert`` gives a robust
root finder to build ppf/isf references from a cdf/sf reference.

Error measure
-------------
    ulp = |computed - reference| / (|reference| * eps),  eps = 2.2e-16

Only points whose reference is a normal double are measured. For points with
more than 8 ulp the condition numbers are computed numerically,

    cond (arg)    = |x * dF/dx / F|   (x or q, the method's argument)
    cond (shapes) = sum over non-integer shapes p of |p * dF/dp / F|

A 1/2-ulp change of the argument moves the result by about cond/2 ulp, so a
point is "within conditioning" if ulp <= 8 + cond (arg). Shapes are treated as
exact user choices (c = d = 1 means exactly 1); cond (shapes) is shown for
context only.
Severity of points beyond that:

    loss      no correct digits: 0, inf, nan, or relative error >= 1
    severe    > 1e6 ulp
    moderate  > 1e3 ulp
    minor     beyond conditioning, <= 1e3 ulp
"""
import argparse
import contextlib
import hashlib
import importlib.util
import math
import os
import pickle
import signal
import sys
import time
import warnings

import numpy as np
import mpmath as mp
import scipy
from scipy import stats

EPS = np.finfo(float).eps
TINY = np.finfo(float).tiny
HUGE = np.finfo(float).max
X_METHODS = ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf"]
Q_METHODS = ["ppf", "isf"]
ALL_METHODS = X_METHODS + Q_METHODS
SEVERITIES = ["loss", "severe", "moderate", "minor"]


# ------------------------------------------------------------------ helpers

class RefTimeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    """Raise RefTimeout after `seconds` (Unix only; no limit elsewhere)."""
    if not seconds or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(signum, frame):
        raise RefTimeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def invert(F, target, lo=None, hi=None, increasing=True, positive=None, digits=60):
    """Solve F(x) = target for monotone F at the current mpmath precision.

    For 0 < target < 1 the equation is solved for log(F) = log(target) when
    target <= 1/2 and for log(1 - F) = log(1 - target) otherwise, so tail
    targets such as 1e-300 or 1 - 1e-300 converge quickly and to full relative
    accuracy of the small quantity (|log ratio| <= 10**-digits).

    lo, hi: optional bracket; it is widened automatically if needed.
    positive: True for supports in (0, inf) (solved in log x, so tiny or huge
              roots get full relative accuracy); default: True unless lo <= 0.
    """
    target = mp.mpf(target)
    sgn = 1 if increasing else -1
    if positive is None:
        positive = lo is None or lo > 0
    to_x = mp.exp if positive else (lambda u: u)
    big = mp.mpf(10) ** 8

    if 0 < target < 1:
        upper = target > mp.mpf(0.5)
        Lt = mp.log(1 - target) if upper else mp.log(target)
        dirn = -sgn if upper else sgn

        def H(u):                                 # increasing in u
            v = F(to_x(u))
            v = 1 - v if upper else v
            if v <= 0:
                return -dirn * big                # saturated: far on one side
            return dirn * (mp.log(v) - Lt)
        tol_ok = mp.mpf(10) ** (-digits)
    else:
        def H(u):
            return sgn * (F(to_x(u)) - target)
        tol_ok = mp.mpf(10) ** (-digits) * max(abs(target), 1)

    if positive:
        a = mp.log(mp.mpf(lo)) if lo is not None else mp.mpf(0)
        b = mp.log(mp.mpf(hi)) if hi is not None else mp.mpf(0)
    else:
        a = mp.mpf(lo) if lo is not None else mp.mpf(-1)
        b = mp.mpf(hi) if hi is not None else mp.mpf(1)
    step = mp.mpf(1) if positive else max(abs(b - a), mp.mpf(1))
    for _ in range(80):                           # widen geometrically
        if H(a) <= 0:
            break
        a -= step
        step *= 2
    step = mp.mpf(1) if positive else max(abs(b - a), mp.mpf(1))
    for _ in range(80):
        if H(b) >= 0:
            break
        b += step
        step *= 2
    for _ in range(200):                          # coarse bisection
        if b - a < mp.mpf("1e-3") * (1 + abs(a) + abs(b)):
            break
        m = (a + b) / 2
        if H(m) < 0:
            a = m
        else:
            b = m
    u = None
    try:
        u = mp.findroot(H, (a, b), solver="anderson", maxsteps=200,
                        tol=mp.mpf(10) ** (-2 * digits - 20), verify=False)
        if not (a <= u <= b) or abs(H(u)) > tol_ok:
            u = None
    except (ZeroDivisionError, ValueError):
        u = None
    if u is None:                                 # fallback: bisection to tolerance
        for _ in range(int(3.4 * (digits + 30)) + 400):
            m = (a + b) / 2
            hm = H(m)
            if abs(hm) <= tol_ok:
                break
            if hm < 0:
                a = m
            else:
                b = m
        u = m
    return to_x(u)


def load_module(path, name="refmod"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def implemented_methods(dist):
    cls = type(dist)
    out = []
    for m in ALL_METHODS:
        for k in cls.__mro__:
            if "_" + m in k.__dict__:
                if k is not stats.rv_continuous and k.__name__ != "rv_generic":
                    out.append(m)
                break
    return out


def progress(msg):
    print(f"  ... {msg}", file=sys.stderr, flush=True)


def is_normal(v):
    return np.isfinite(v) and TINY <= abs(v) <= HUGE


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def fmt_ulp(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if math.isinf(v):
        return "inf"
    return f"{v:.3g}" if v < 1e5 else f"{v:.1e}"


def fmt_shapes(shapes):
    return "(" + ", ".join(f"{s:g}" for s in shapes) + ")"


# ------------------------------------------------------------------ grids

def default_x(dist, shapes):
    a, b = (float(v) for v in dist.support(*shapes))
    mags = 10.0 ** np.linspace(-300, 300, 61)
    rel = 10.0 ** np.linspace(-300, 0, 31)
    if np.isinf(a) and np.isinf(b):
        pts = np.concatenate([-mags, [0.0], mags, np.linspace(-10, 10, 21)])
    elif np.isinf(b):
        pts = np.concatenate([a + mags, a + np.linspace(0, 10, 21)[1:]])
    elif np.isinf(a):
        pts = np.concatenate([b - mags, b - np.linspace(0, 10, 21)[1:]])
    else:
        w = b - a
        pts = np.concatenate([a + w * rel, b - w * rel, np.linspace(a, b, 21)])
    pts = pts[np.isfinite(pts) & (pts >= a) & (pts <= b)]
    return np.unique(pts)


def default_q():
    return np.unique(np.concatenate([
        10.0 ** np.linspace(-300, -1, 25),
        np.linspace(0.01, 0.99, 15),
        1 - 10.0 ** np.linspace(-15.5, -1, 15)]))


# ------------------------------------------------------------------ references

class References:
    """Evaluates reference functions with caching, timeouts and error capture."""

    def __init__(self, mod, ref_path, dps, timeout, use_cache=True):
        self.mod, self.dps, self.timeout = mod, dps, timeout
        self.cache, self.dirty = {}, False
        h = hashlib.sha1(open(ref_path, "rb").read()).hexdigest()[:12]
        self.cache_file = None
        if use_cache:
            d = os.path.join(".cache", "dist-accuracy")
            os.makedirs(d, exist_ok=True)
            self.cache_file = os.path.join(d, f"{mod.DIST}_{h}_{dps}.pkl")
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "rb") as f:
                    self.cache = pickle.load(f)
        self.failures = []

    def save(self):
        if self.cache_file and self.dirty:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
            self.dirty = False

    def raw(self, method, arg, shapes, dps=None):
        dps = dps or self.dps
        with mp.workdps(dps):
            f = self.mod.REFS[method]
            return f(mp.mpf(float(arg)), *[mp.mpf(float(s)) for s in shapes])

    def __call__(self, method, arg, shapes):
        key = (method, float(arg), tuple(float(s) for s in shapes))
        if key in self.cache:
            return self.cache[key]
        try:
            with time_limit(self.timeout):
                v = self.raw(method, arg, shapes)
        except RefTimeout:
            v = None
            self.failures.append((method, arg, shapes, "timeout"))
        except Exception as e:                    # reference undefined here
            v = None
            self.failures.append((method, arg, shapes, f"{type(e).__name__}: {e}"))
        self.cache[key] = v
        self.dirty = True
        return v

    def cond(self, method, arg, shapes, integral):
        """Condition numbers (arg, shapes): |p dF/dp / F| for the argument p,
        and the sum of the same over the non-integer shapes."""
        inputs = [float(arg)] + [float(s) for s in shapes]
        skip = [False] + list(integral)
        out = [mp.mpf(0), mp.mpf(0)]
        try:
            with time_limit(self.timeout * 4), mp.workdps(self.dps):
                F0 = self.raw(method, arg, shapes)
                if F0 == 0:
                    return float("nan"), float("nan")
                for i, (p, sk) in enumerate(zip(inputs, skip)):
                    if sk or p == 0:
                        continue

                    def g(t, i=i):
                        v = [mp.mpf(u) for u in inputs]
                        v[i] = v[i] * (1 + t)
                        return self.mod.REFS[method](*v)
                    out[0 if i == 0 else 1] += abs(mp.diff(g, 0) / F0)
            return float(out[0]), float(out[1])
        except Exception:                         # incl. timeout
            return float("nan"), float("nan")


# ------------------------------------------------------------------ evaluation

def evaluate(dist, method, args, shapes):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with np.errstate(all="warn"):
            out = np.asarray(getattr(dist, method)(args, *shapes), dtype=float)
    msgs = sorted({str(x.message) for x in w if issubclass(x.category, RuntimeWarning)})
    return np.broadcast_to(out, np.shape(args)).copy(), msgs


def ulp_error(got, ref):
    if not is_normal(ref):
        return float("nan")
    if not np.isfinite(got) or got == 0 or np.sign(got) != np.sign(ref):
        return float("inf")
    return abs(got - ref) / (abs(ref) * EPS)


def severity(err, cond):
    tol = 8 + (0 if math.isnan(cond) else cond)
    if math.isnan(err) or err <= tol:
        return "ok"
    if math.isinf(err) or err >= 1 / EPS:
        return "loss"
    if err > 1e6:
        return "severe"
    if err > 1e3:
        return "moderate"
    return "minor"


def run(mod, dist, impls, refs, methods, params, integral, only_impl_cond=None):
    """Return a list of point records."""
    recs = []
    replace = getattr(mod, "REPLACE_GRID", False)
    for shapes in params:
        xs = np.asarray(mod.x_points(*shapes), float) if hasattr(mod, "x_points") else np.array([])
        qs = np.asarray(mod.q_points(*shapes), float) if hasattr(mod, "q_points") else np.array([])
        if not replace:
            xs = np.concatenate([default_x(dist, shapes), xs])
            qs = np.concatenate([default_q(), qs])
        lo_s, hi_s = (float(v) for v in dist.support(*shapes))
        xs = np.unique(xs[np.isfinite(xs) & (xs >= lo_s) & (xs <= hi_s)])
        qs = np.unique(qs[(qs > 0) & (qs < 1)])
        # region labels from the reference cdf (x methods) or q itself
        region_x = None
        if "cdf" in mod.REFS:
            region_x = []
            for x in xs:
                F = refs("cdf", x, shapes)
                if F is None:
                    region_x.append("?")
                elif F < mp.mpf("1e-3"):
                    region_x.append("low tail")
                elif 1 - F < mp.mpf("1e-3"):
                    region_x.append("high tail")
                else:
                    region_x.append("bulk")
        for m in methods:
            args = xs if m in X_METHODS else qs
            t0 = time.time()
            ref_mp = [refs(m, a, shapes) for a in args]
            progress(f"{fmt_shapes(shapes)} {m}: {len(args)} points, references {time.time() - t0:.1f} s")
            ref = np.array([to_float(r) if r is not None else np.nan for r in ref_mp])
            outs = {}
            for iname, d in impls.items():
                outs[iname] = evaluate(d, m, args, shapes)
            for k, a in enumerate(args):
                if m in X_METHODS:
                    reg = region_x[k] if region_x else "all"
                else:
                    reg = "low tail" if a < 1e-3 else "high tail" if a > 1 - 1e-3 else "bulk"
                rec = dict(method=m, shapes=tuple(shapes), arg=float(a), ref=float(ref[k]),
                           ref_mp=ref_mp[k], region=reg, impl={})
                for iname, (vals, msgs) in outs.items():
                    rec["impl"][iname] = dict(value=float(vals[k]),
                                              ulp=ulp_error(vals[k], ref[k]), warnings=msgs)
                recs.append(rec)
        refs.save()
    # condition numbers only where some implementation exceeds 8 ulp
    n_cond = sum(1 for r in recs if any(v["ulp"] > 8 for v in r["impl"].values()
                                        if not math.isnan(v["ulp"])))
    progress(f"condition numbers for {n_cond} points above 8 ulp")
    t0 = time.time()
    for rec in recs:
        worst = max((v["ulp"] for v in rec["impl"].values() if not math.isnan(v["ulp"])),
                    default=float("nan"))
        rec["cond"], rec["cond_shapes"] = (
            refs.cond(rec["method"], rec["arg"], rec["shapes"], integral)
            if (not math.isnan(worst) and worst > 8) else (0.0, 0.0))
        for v in rec["impl"].values():
            v["severity"] = severity(v["ulp"], rec["cond"])
    progress(f"condition numbers done in {time.time() - t0:.1f} s")
    return recs


# ------------------------------------------------------------------ self checks

def self_checks(mod, dist, refs, params):
    """Checks of the reference itself. Returns list of (status, message)."""
    out = []
    R = mod.REFS
    for shapes in params[:3]:
        tag = fmt_shapes(shapes)
        progress(f"self-checks {tag}")
        lo_s, hi_s = (float(v) for v in dist.support(*shapes))
        bulk_x = []
        for q in [0.1, 0.5, 0.9]:
            x = None
            if "ppf" in R:
                x = refs("ppf", q, shapes)
                x = to_float(x) if x is not None else None
            if x is None or not np.isfinite(x):
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    x = float(dist.ppf(q, *shapes))
            if np.isfinite(x) and lo_s < x < hi_s and x not in bulk_x:
                bulk_x.append(x)
        grid = default_x(dist, shapes)
        tail_x = [grid[1], grid[-2]] if len(grid) > 3 else []
        # 1. agreement with scipy in the bulk (parametrization check).
        #    A parametrization mistake shows at every bulk point; SciPy returning
        #    0/inf/nan in the bulk is a SciPy problem, not a reference problem.
        for m in [m for m in X_METHODS if m in R]:
            agree, disagree, broken = [], [], []
            for x in bulk_x:
                r = refs(m, x, shapes)
                if r is None or not is_normal(to_float(r)):
                    continue
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    s_ = float(getattr(dist, m)(x, *shapes))
                rf = to_float(r)
                if not np.isfinite(s_) or s_ == 0:
                    broken.append((x, s_, rf))
                elif abs(s_ / rf - 1) > 1e-6:
                    disagree.append((x, s_, rf))
                else:
                    agree.append(x)
            if disagree and not agree:
                out.append(("FAIL", f"{m}{tag}: reference disagrees with scipy at every bulk point "
                                    f"(check the parametrization): x, scipy, ref = {disagree[0]}"))
            elif disagree:
                out.append(("WARN", f"{m}{tag}: reference disagrees with scipy at some bulk points "
                                    f"(scipy inaccuracy or reference error; check with --at): "
                                    f"x, scipy, ref = {disagree[0]}"))
            if broken:
                out.append(("NOTE", f"{m}{tag}: scipy returns {broken[0][1]} in the bulk where the "
                                    f"reference is {broken[0][2]!r} (x = {broken[0][0]!r}); likely a "
                                    "scipy problem - verify with --at, then report it"))
        # 2. stability in precision at bulk and tail points
        for m in R:
            pts = bulk_x + tail_x if m in X_METHODS else [1e-200, 0.3, 1 - 1e-12]
            for x in pts:
                try:
                    with time_limit(refs.timeout * 2):
                        lo = refs.raw(m, x, shapes)
                        hi = refs.raw(m, x, shapes, dps=refs.dps + 100)
                except Exception:
                    continue
                if not (mp.isfinite(hi) and TINY <= abs(hi) <= HUGE):
                    continue                      # not a normal double: never measured
                if lo == 0 or abs(lo / hi - 1) > mp.mpf(10) ** -30:
                    out.append(("FAIL", f"{m}{tag} at {x:g}: changes with precision "
                                        f"({mp.nstr(lo, 20)} vs {mp.nstr(hi, 20)}); raise DPS "
                                        "or use a better-conditioned reference formula"))
        # 3. internal consistency of the references (at 50 digits, fast)
        with mp.workdps(50):
            P = [mp.mpf(s) for s in shapes]
            for x in bulk_x:
                X = mp.mpf(x)
                try:
                    if "cdf" in R and "sf" in R and abs(R["cdf"](X, *P) + R["sf"](X, *P) - 1) > 1e-40:
                        out.append(("FAIL", f"cdf + sf != 1 at {x:g}{tag}"))
                    for lg, base in [("logpdf", "pdf"), ("logcdf", "cdf"), ("logsf", "sf")]:
                        if lg in R and base in R:
                            if abs(R[lg](X, *P) - mp.log(R[base](X, *P))) > 1e-40 * (1 + abs(R[lg](X, *P))):
                                out.append(("FAIL", f"{lg} != log({base}) at {x:g}{tag}"))
                except Exception as e:
                    out.append(("WARN", f"consistency check failed to run at {x:g}{tag}: {e!r}"))
            if "pdf" in R and "cdf" in R and len(bulk_x) == 3:
                try:
                    with mp.workdps(30):
                        X = [mp.mpf(v) for v in bulk_x]
                        for x1, x2 in [(X[0], X[1]), (X[1], X[2])]:
                            integral = mp.quad(lambda t: R["pdf"](t, *P), [x1, x2])
                            diff = R["cdf"](x2, *P) - R["cdf"](x1, *P)
                            if abs(integral / diff - 1) > mp.mpf("1e-12"):
                                out.append(("FAIL", f"integral of pdf != cdf difference on "
                                                    f"[{mp.nstr(x1, 6)}, {mp.nstr(x2, 6)}]{tag}"))
                except Exception as e:
                    out.append(("WARN", f"pdf/cdf integral check failed to run{tag}: {e!r}"))
            for m, base in [("ppf", "cdf"), ("isf", "sf")]:
                if m in R and base in R:
                    for q in [1e-10, 0.3, 1 - 1e-10]:
                        try:
                            with time_limit(refs.timeout * 2):
                                x = refs(m, q, shapes)
                                if x is None or not mp.isfinite(x):
                                    continue
                                back = R[base](mp.mpf(x), *P)
                            if abs(back / mp.mpf(q) - 1) > 1e-25:
                                out.append(("FAIL", f"{base}({m}({q:g})) != {q:g}{tag}"))
                        except Exception:
                            pass
    if not any(st in ("FAIL", "WARN") for st, _ in out):
        out.insert(0, ("PASS", "references agree with scipy in the bulk (where scipy is finite), "
                               "are stable in precision and internally consistent"))
    return out


# ------------------------------------------------------------------ reporting

def summary(recs, impls, methods):
    names = list(impls)
    head = f"{'method':8}{'region':11}{'n':>5}" + "".join(
        f" | {n[:14]:>14}: max ulp  median   loss severe moder. minor" for n in names)
    print(head)
    print("-" * len(head))
    for m in methods:
        rs = [r for r in recs if r["method"] == m]
        regions = [g for g in ["low tail", "bulk", "high tail", "all", "?"] if any(r["region"] == g for r in rs)]
        for k, g in enumerate(regions):
            sub = [r for r in rs if r["region"] == g and not math.isnan(r["ref"]) and is_normal(r["ref"])]
            if not sub:
                continue
            cells = []
            for n in names:
                e = np.array([r["impl"][n]["ulp"] for r in sub])
                e = e[~np.isnan(e)]
                if e.size == 0:
                    cells.append(f" | {'-':>23} {'-':>7} {'':>23}")
                    continue
                fin = e[np.isfinite(e)]
                med = np.median(fin) if fin.size else float("inf")
                counts = [sum(r["impl"][n]["severity"] == s for r in sub) for s in SEVERITIES]
                cells.append(f" | {fmt_ulp(e.max()):>23} {fmt_ulp(med):>7}"
                             + "".join(f"{c:>7}" for c in counts))
            print(f"{m if k == 0 else '':8}{g:11}{len(sub):>5}" + "".join(cells))
        warn = {}
        for n in names:
            sets = {r["shapes"] for r in rs if r["impl"][n]["warnings"]}
            msgs = sorted({w for r in rs for w in r["impl"][n]["warnings"]})
            if sets:
                warn[n] = f"{len(sets)} parameter set(s): {'; '.join(msgs)}"
        for n, w in warn.items():
            print(f"{'':8}RuntimeWarning ({n}) in {w}")
    print()


def flagged(recs, impls, limit):
    names = list(impls)
    order = {s: i for i, s in enumerate(SEVERITIES + ["ok"])}
    print("Points beyond conditioning (markdown, worst first per method):\n")
    any_ = False
    for m in sorted({r["method"] for r in recs}, key=ALL_METHODS.index):
        rows = [r for r in recs if r["method"] == m
                and any(r["impl"][n]["severity"] != "ok" for n in names)]
        if not rows:
            continue
        any_ = True
        rows.sort(key=lambda r: (min(order[r["impl"][n]["severity"]] for n in names),
                                 -max(r["impl"][n]["ulp"] for n in names)))
        print(f"**{m}** ({len(rows)} points)\n")
        print("| shapes | arg | reference | " + " | ".join(f"{n} value | {n} ulp | {n} severity" for n in names)
              + " | cond (arg) | cond (shapes) |")
        print("|---|---|---|" + "---|---|---|" * len(names) + "---|---|")
        for r in rows[:limit]:
            cells = []
            for n in names:
                v = r["impl"][n]
                cells.append(f"{v['value']!r} | {fmt_ulp(v['ulp'])} | {v['severity']}")
            print(f"| {fmt_shapes(r['shapes'])} | {r['arg']!r} | {float(r['ref'])!r} | "
                  + " | ".join(cells) + f" | {fmt_ulp(r['cond'])} | {fmt_ulp(r['cond_shapes'])} |")
        print()
    if not any_:
        print("none\n")


def per_params(recs, impls, methods, params):
    print("Worst ulp per parameter set ('inf' = 0/inf/nan where the reference is finite):\n")
    names = list(impls)
    for m in methods:
        print(f"{m}:")
        print(f"  {'shapes':24}" + "".join(f"{n[:16]:>18}" for n in names) + f"{'cond there':>12}")
        for s in params:
            rs = [r for r in recs if r["method"] == m and r["shapes"] == tuple(s)]
            row, cond = [], "-"
            for n in names:
                e = [r for r in rs if not math.isnan(r["impl"][n]["ulp"])]
                if e:
                    w = max(e, key=lambda r: r["impl"][n]["ulp"])
                    row.append(fmt_ulp(w["impl"][n]["ulp"]))
                    cond = fmt_ulp(w["cond"])
                else:
                    row.append("-")
            print(f"  {fmt_shapes(s):24}" + "".join(f"{v:>18}" for v in row) + f"{cond:>12}")
        print()


def plot(recs, impls, methods, params, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#eb6834", "#2a78d6", "#1baf7a"]
    markers = ["x", "o", "s"]
    params = params[:4]
    fig, axes = plt.subplots(len(methods), len(params), squeeze=False, sharey=True,
                             figsize=(4.2 * len(params), 2.0 * len(methods)), facecolor="#fcfcfb")
    for i, m in enumerate(methods):
        for j, s in enumerate(params):
            ax = axes[i, j]
            ax.set_facecolor("#fcfcfb")
            rs = [r for r in recs if r["method"] == m and r["shapes"] == tuple(s)]
            if m in Q_METHODS:
                u = [math.log10(r["arg"]) - math.log10(1 - r["arg"]) for r in rs]
                xlabel = "log10(q / (1 - q))"
            else:
                u = [math.log10(abs(r["arg"])) if r["arg"] != 0 else -330 for r in rs]
                xlabel = "log10 |x|  (negative x not distinguished)"
            ax.axhspan(1e17, 1e18, color="#e9e8e3", lw=0)
            for k, n in enumerate(impls):
                e = np.array([r["impl"][n]["ulp"] for r in rs])
                ok = np.isfinite(e)
                ax.plot(np.array(u)[ok], np.maximum(e[ok], 0.1), ls="none", marker=markers[k % 3],
                        ms=3.5, mfc="none" if k else colors[k], color=colors[k % 3], label=n)
                bad = np.isinf(e)
                ax.plot(np.array(u)[bad], np.full(bad.sum(), 3e17), ls="none", marker=markers[k % 3],
                        ms=3.5, color=colors[k % 3])
            ax.set_yscale("log")
            ax.set_ylim(0.1, 1e18)
            ax.grid(axis="y", color="#e4e3dd", lw=0.5)
            ax.tick_params(labelsize=7, colors="#6b6a64")
            if j == 0:
                ax.set_ylabel(f"{m}\nerror [ulp]", fontsize=8)
            if i == 0:
                ax.set_title(fmt_shapes(s), fontsize=9)
            ax.set_xlabel(xlabel, fontsize=7, color="#6b6a64")
            ax.text(1.01, 3e17, "fail", transform=ax.get_yaxis_transform(), va="center", fontsize=6)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(impls), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=120)
    print(f"plot written to {path}")


# ------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("ref", help="reference module (.py)")
    p.add_argument("--methods", help="comma-separated methods (default: implemented ones with a reference)")
    p.add_argument("--params", help='override PARAMS, e.g. "2,0.5;3,1"')
    p.add_argument("--dps", type=int, help="mpmath digits (default: DPS in module, else 400)")
    p.add_argument("--timeout", type=float, default=20, help="seconds per reference evaluation")
    p.add_argument("--at", nargs="+", metavar="X", help="METHOD ARG SHAPES...: evaluate one point")
    p.add_argument("--candidate", help="FILE.py:OBJECT, an rv_continuous instance to measure too")
    p.add_argument("--limit", type=int, default=12, help="flagged rows per method (default 12)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--plot", metavar="PNG")
    p.add_argument("--json", metavar="FILE", help="write all points as JSON")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--no-checks", action="store_true", help="skip reference self-checks")
    a = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # for `from accuracy import invert`
    mod = load_module(a.ref)
    dist = getattr(stats, mod.DIST)
    dps = a.dps or getattr(mod, "DPS", 400)
    mp.mp.dps = dps
    refs = References(mod, a.ref, dps, a.timeout, use_cache=not a.no_cache)
    impls = {f"scipy {scipy.__version__}": dist}
    if a.candidate:
        f, obj = a.candidate.rsplit(":", 1)
        impls["candidate"] = getattr(load_module(f, "candmod"), obj)
    integral = [s.integrality for s in dist._shape_info()]

    if a.at:
        m, arg, shapes = a.at[0], float(a.at[1]), [float(s) for s in a.at[2:]]
        r = refs(m, arg, shapes)
        rf = to_float(r) if r is not None else float("nan")
        cond, cond_s = refs.cond(m, arg, shapes, integral)
        print(f"{mod.DIST}.{m}({arg!r}, {', '.join(map(repr, shapes))})")
        print(f"  reference      {mp.nstr(r, 25) if r is not None else 'unavailable'}  (float {rf!r})")
        print(f"  cond (arg)     {fmt_ulp(cond)}  (within conditioning: <= {fmt_ulp(8 + (0 if math.isnan(cond) else cond))} ulp)")
        print(f"  cond (shapes)  {fmt_ulp(cond_s)}  (context only: error from a 1-ulp change of the shapes)")
        for n, d in impls.items():
            vals, msgs = evaluate(d, m, np.array([arg]), shapes)
            e = ulp_error(vals[0], rf)
            print(f"  {n:14} {vals[0]!r:26} {fmt_ulp(e):>8} ulp  {severity(e, cond)}"
                  + (f"  [{'; '.join(msgs)}]" if msgs else ""))
        refs.save()
        return

    params = ([tuple(float(v) for v in s.split(",")) for s in a.params.split(";")]
              if a.params else [tuple(map(float, s)) for s in mod.PARAMS])
    impl = implemented_methods(dist)
    if a.methods:
        methods = a.methods.split(",")
    else:
        methods = getattr(mod, "METHODS", [m for m in ALL_METHODS if m in impl and m in mod.REFS])
    methods = [m for m in ALL_METHODS if m in methods]
    missing_ref = [m for m in methods if m not in mod.REFS]
    if missing_ref:
        raise SystemExit(f"no reference for: {missing_ref}")
    not_impl = [m for m in methods if m not in impl]

    print(f"{mod.DIST}: SciPy {scipy.__version__}, mpmath {mp.__version__}, {dps} digits")
    print(f"implemented: {impl}; inherited: {[m for m in ALL_METHODS if m not in impl]}")
    print(f"measured: {methods}" + (f"  (NOTE: {not_impl} are generic defaults)" if not_impl else ""))
    print(f"parameter sets: {[fmt_shapes(s) for s in params]}\n")

    t0 = time.time()
    if not a.no_checks:
        print("Reference self-checks:")
        for status, msg in self_checks(mod, dist, refs, params):
            print(f"  {status}: {msg}")
        print()
    recs = run(mod, dist, impls, refs, methods, params, integral)
    refs.save()
    summary(recs, impls, methods)
    flagged(recs, impls, a.limit)
    if a.verbose:
        per_params(recs, impls, methods, params)
    if refs.failures:
        kinds = {}
        for m, x, s, why in refs.failures:
            kinds.setdefault((m, why.split(":")[0]), 0)
            kinds[(m, why.split(":")[0])] += 1
        print("Reference unavailable (points skipped): "
              + "; ".join(f"{m}: {n} x {w}" for (m, w), n in kinds.items()) + "\n")
    if a.plot:
        plot(recs, impls, methods, params, a.plot)
    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "ref_mp"} for r in recs], f,
                      default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o))
        print(f"points written to {a.json}")
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
