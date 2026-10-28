"""Inventory of scipy.stats continuous distributions for the accuracy review.

For each distribution it reports:
  * the SciPy version, source file and line range of the class
  * which of pdf, logpdf, cdf, logcdf, sf, logsf, ppf, isf are implemented
    (and in which class, with line numbers) and which fall back to the
    generic rv_continuous defaults
  * shape parameters with domains (from _shape_info) and example parameters
    (from scipy.stats._distr_params.distcont)
  * support for the example parameters and whether _get_support is overridden
  * module-level helpers and scipy.special functions used in the class
  * related distributions (shared base classes, or calls into another
    distribution such as fisk -> burr)

Usage
-----
    python inventory.py                 # all continuous distributions, text
    python inventory.py fisk gamma      # selected distributions
    python inventory.py --json out.json # also write machine-readable output
    python inventory.py --list          # just the names, one per line
"""
import argparse
import inspect
import json
import re

import numpy as np
import scipy
from scipy import stats
from scipy.stats._distr_params import distcont

METHODS = ["pdf", "logpdf", "cdf", "logcdf", "sf", "logsf", "ppf", "isf"]

# No practical mpmath reference: review the code only (light mode).
LIGHT_REVIEW = {"levy_stable", "studentized_range"}

GENERIC_DEFAULT = {
    "pdf": "numerical derivative of cdf",
    "logpdf": "log(pdf)",
    "cdf": "numerical integration of pdf",
    "logcdf": "log(cdf)",
    "sf": "1 - cdf",
    "logsf": "log(sf)",
    "ppf": "numerical inversion of cdf",
    "isf": "ppf(1 - q)",
}


def all_names():
    return sorted(n for n in dir(stats)
                  if isinstance(getattr(stats, n), stats.rv_continuous))


def defining_class(cls, attr):
    for k in cls.__mro__:
        if attr in k.__dict__:
            return k
    return None


def line_range(obj):
    try:
        lines, start = inspect.getsourcelines(obj)
        return [start, start + len(lines) - 1]
    except (OSError, TypeError):
        return None


def first_math_blocks(doc, n=2):
    if not doc:
        return []
    blocks = re.findall(r"\.\. math::\s*\n\n(.*?)(?:\n\s*\n)", doc, flags=re.S)
    return [" ".join(b.split()) for b in blocks[:n]]


def info(name):
    dist = getattr(stats, name)
    cls = type(dist)
    module = inspect.getmodule(cls)
    src_file = inspect.getsourcefile(cls)
    try:
        src = inspect.getsource(cls)
    except OSError:
        src = ""

    implemented, inherited = {}, {}
    for m in METHODS:
        k = defining_class(cls, "_" + m)
        if k is None or k is stats.rv_continuous or k.__name__ == "rv_generic":
            inherited[m] = GENERIC_DEFAULT[m]
        else:
            implemented[m] = {"class": k.__name__,
                              "lines": line_range(k.__dict__["_" + m])}

    shapes = []
    try:
        for s in dist._shape_info():
            shapes.append({"name": s.name, "domain": [float(v) for v in s.domain],
                           "inclusive": list(s.inclusive),
                           "integrality": bool(s.integrality)})
    except Exception as e:                      # pragma: no cover
        shapes = [{"error": repr(e)}]

    examples = [list(map(float, p)) for n, p in distcont if n == name]
    try:
        support = [float(v) for v in dist.support(*(examples[0] if examples else []))]
    except Exception:
        support = [float(dist.a), float(dist.b)]

    # helpers and special functions referenced in the class source
    helpers = sorted({h for h in re.findall(r"\b(_[A-Za-z][A-Za-z0-9_]*)\s*\(", src)
                      if inspect.isfunction(getattr(module, h, None))})
    special = sorted(set(".".join(t) for t in
                         re.findall(r"\b(sc|scu|special|_ufuncs)\.([A-Za-z_][A-Za-z0-9_]*)", src)))
    calls_into = sorted({d for d in re.findall(r"\b([a-z][a-z0-9_]*)\._[a-z]+\(", src)
                         if d != name and isinstance(getattr(stats, d, None), stats.rv_continuous)}
                        | {d[:-4] for d in re.findall(r"\b([a-z][a-z0-9_]*_gen)\._[a-z]+\(", src)
                           if d[:-4] != name})
    bases = [k.__name__ for k in cls.__mro__[1:]
             if issubclass(k, stats.rv_continuous) and k is not stats.rv_continuous]

    return {
        "name": name,
        "scipy_version": scipy.__version__,
        "class": cls.__name__,
        "source_file": src_file,
        "class_lines": line_range(cls),
        "implemented": implemented,
        "inherited": inherited,
        "shapes": shapes,
        "example_params": examples,
        "support_for_example": support,
        "get_support_overridden": "_get_support" in cls.__dict__,
        "pdf_in_docstring": first_math_blocks(cls.__doc__),
        "helpers": helpers,
        "special_functions": special,
        "base_classes": bases,
        "calls_into": calls_into,
        "light_review": name in LIGHT_REVIEW,
    }


def add_related(infos, everything):
    """Distributions related through base classes or calls in either direction."""
    by_name = {i["name"]: i for i in everything}
    for i in infos:
        rel = set(i["calls_into"])
        for j in everything:
            if j["name"] == i["name"]:
                continue
            if i["name"] in j["calls_into"]:
                rel.add(j["name"])
            if i["class"] in j["base_classes"] or j["class"] in i["base_classes"]:
                rel.add(j["name"])
        i["related"] = sorted(r for r in rel if r in by_name)


def show(i):
    print(f"== {i['name']} ({i['class']}), SciPy {i['scipy_version']}")
    print(f"   source: {i['source_file']} lines {i['class_lines']}")
    if i["light_review"]:
        print("   LIGHT REVIEW: no practical mpmath reference, review code only")
    for m in METHODS:
        if m in i["implemented"]:
            d = i["implemented"][m]
            where = "" if d["class"] == i["class"] else f" (in {d['class']})"
            print(f"   {m:7} implemented{where}, lines {d['lines']}")
        else:
            print(f"   {m:7} inherited: {i['inherited'][m]}")
    for s in i["shapes"]:
        print(f"   shape {s}")
    print(f"   example params: {i['example_params']}, support there: {i['support_for_example']}"
          + (" (_get_support overridden)" if i["get_support_overridden"] else ""))
    for b in i["pdf_in_docstring"]:
        print(f"   docstring math: {b}")
    print(f"   helpers: {i['helpers']}")
    print(f"   special functions: {i['special_functions']}")
    print(f"   related: {i['related']}")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("names", nargs="*", help="distribution names (default: all)")
    p.add_argument("--json", metavar="FILE", help="write JSON to FILE")
    p.add_argument("--list", action="store_true", help="only print names")
    a = p.parse_args()

    names = a.names or all_names()
    unknown = [n for n in names if n not in all_names()]
    if unknown:
        raise SystemExit(f"not continuous scipy.stats distributions: {unknown}")
    if a.list:
        print("\n".join(names))
        return
    everything = [info(n) for n in all_names()]
    infos = [i for i in everything if i["name"] in names]
    add_related(infos, everything)
    for i in infos:
        show(i)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"scipy_version": scipy.__version__, "distributions": infos}, f, indent=1)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
