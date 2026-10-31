"""Run accuracy.py with nct methods wrapped so that a Boost OverflowError on
one element does not abort the array call: on exception the call is redone
element by element, failing elements return nan and are logged to
scratch/nct/exceptions.txt.  Runtime-only patch; SciPy on disk is unchanged."""
import sys, runpy, numpy as np
from scipy.stats import _continuous_distns as cd

LOG = r"C:/Users/chbau/Documents/research/dist-accuracy/scratch/nct/exceptions.txt"

def wrap(name):
    orig = getattr(cd.nct_gen, name)
    def f(self, x, df, nc):
        try:
            return orig(self, x, df, nc)
        except Exception:
            x, df, nc = np.broadcast_arrays(np.asarray(x, float), np.asarray(df, float), np.asarray(nc, float))
            out = np.empty(x.shape)
            for i in np.ndindex(x.shape):
                try:
                    out[i] = orig(self, x[i], df[i], nc[i])
                except Exception as e:
                    out[i] = np.nan
                    with open(LOG, "a") as fh:
                        fh.write(f"{name}\t{x[i]!r}\t{df[i]!r}\t{nc[i]!r}\t{type(e).__name__}: {e}\n")
            return out
    setattr(cd.nct_gen, name, f)

for n in ["_pdf", "_cdf", "_sf", "_ppf", "_isf"]:
    wrap(n)
acc = r"C:/Users/chbau/Documents/research/dist-accuracy/.claude/skills/dist-accuracy-review/scripts/accuracy.py"
sys.argv = [acc] + sys.argv[1:]
sys.path.insert(0, r"C:/Users/chbau/Documents/research/dist-accuracy/.claude/skills/dist-accuracy-review/scripts")
runpy.run_path(acc, run_name="__main__")
